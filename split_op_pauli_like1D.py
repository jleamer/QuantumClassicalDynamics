import numpy as np
import numexpr as ne
from numexpr import evaluate
from scipy import fftpack # Tools for fourier transform
from scipy import linalg # Linear algebra for dense matrix
from types import MethodType, FunctionType


class SplitOpPauliLike1D(object):
    """
    Time dependent propagation of the 1D Pauli-like equation, which is the Schrodinger equation with the Hamiltonian

        H = \sum_{j=0}^3 [ K_j(t, p) + V_j(t, x) ] \sigma_j,

    where \sigma_j denote the Pauli matricides.
    """
    def __init__(self, *, X_gridDIM, X_amplitude, dt, t=0., abs_boundary="1.",
                 V0="0.", V1="0.", V2="0.", V3="0.", diff_V0="0.", diff_V1="0.", diff_V2="0.", diff_V3="0.",
                 K0="0.", diff_K0="0.", K1="0.", diff_K1="0.", K2="0.", diff_K2="0.", K3="0.", diff_K3="0.",
                 isEhrenfest=True, **kwargs):
        """
        Constructor
        :param X_gridDIM: the coordinate grid size
        :param X_amplitude: maximum value of the coordinates
        :param dt: time increment
        :param t: (optional) initial time

        :param VO: (optional) the coordinate dependent term coupling to sigma_0 (as a string to be evaluated by numexpr)
        :param diff_V0 (optional): the derivative of the VO for Ehrenfest theorem calculations

        :param V1: (optional) the coordinate dependent term coupling to sigma_1 (as a string to be evaluated by numexpr)
        :param diff_V1 (optional): the derivative of the V1 for Ehrenfest theorem calculations

        :param V2: (optional) the coordinate dependent term coupling to sigma_2 (as a string to be evaluated by numexpr)
        :param diff_V2 (optional): the derivative of the V2 for Ehrenfest theorem calculations

        :param V3: (optional) the coordinate dependent term coupling to sigma_3 (as a string to be evaluated by numexpr)
        :param diff_V3 (optional): the derivative of the V3 for Ehrenfest theorem calculations

        :param K0: (optional) the momentum dependent term coupling to sigma_0 (as a string to be evaluated by numexpr)
        :param diff_K0 (optional): the derivative of the K0 for Ehrenfest theorem calculations

        :param K1: (optional) the momentum dependent term coupling to sigma_1 (as a string to be evaluated by numexpr)
        :param diff_K1 (optional): the derivative of the K1 for Ehrenfest theorem calculations

        :param K2: (optional) the momentum dependent term coupling to sigma_2 (as a string to be evaluated by numexpr)
        :param diff_K2 (optional): the derivative of the K2 for Ehrenfest theorem calculations

        :param K3: (optional) the momentum dependent term coupling to sigma_3 (as a string to be evaluated by numexpr)
        :param diff_K3 (optional): the derivative of the K13 for Ehrenfest theorem calculations

        :param abs_boundary: (optional) the absorbing boundary (as a string to be evaluated by numexpr)

        :param isEhrenfest: (optional) a boolean flag to calculate the expectation values

        :param kwargs: other parameters
        """

        # save the values of the parameters
        self.X_amplitude = X_amplitude
        self.X_gridDIM = X_gridDIM
        self.t = t
        self.dt = dt
        self.isEhrenfest = isEhrenfest

        # potential energies
        self.V0 = V0
        self.V1 = V1
        self.V2 = V2
        self.V3 = V3

        # and their derivatives
        self.diff_V0 = diff_V0
        self.diff_V1 = diff_V1
        self.diff_V2 = diff_V2
        self.diff_V3 = diff_V3

        # kinetic energies
        self.K0 = K0
        self.K1 = K1
        self.K2 = K2
        self.K3 = K3

        # and their derivatives
        self.diff_K0 = diff_K0
        self.diff_K1 = diff_K1
        self.diff_K2 = diff_K2
        self.diff_K3 = diff_K3

        self.abs_boundary = abs_boundary

        # system parameters for numexpr
        self.sys_params = dict()

        # save all the other attributes
        for name, value in kwargs.items():
            # if the value supplied is a function, then dynamically assign it as a method;
            if isinstance(value, FunctionType):
                setattr(self, name, MethodType(value, self))
            # otherwise save it as a local variable to be passed to numexpr
            else:
                self.sys_params[name] = value

        ########################################################################################
        #
        #   Initialize the meshes
        #
        ########################################################################################

        # get coordinate step size
        self.dX = 2. * self.X_amplitude / self.X_gridDIM

        # generate coordinate range
        self.k = np.arange(self.X_gridDIM)
        self.X = (self.k - self.X_gridDIM / 2) * self.dX

        # generate momentum range
        self.P = (self.k - self.X_gridDIM / 2) * (np.pi / self.X_amplitude)

        # save the momentum step
        self.dP = self.P[1] - self.P[0]

        # allocate the arrays for the two-component wavefunction
        self.psi1 = np.zeros(self.X.size, dtype=np.complex)
        self.psi2 = np.zeros_like(self.psi1)

        # and its copies
        self.psi1_copy = np.zeros_like(self.psi1)
        self.psi2_copy = np.zeros_like(self.psi2)

        ########################################################################################
        #
        #   Generate codes for components of the P operator
        #       P = exp(1j * a * (c0 + c1 * sigma_1 + c2 * sigma_2 + c3 * sigma_3))
        #
        #
        ########################################################################################

        # Note that to avoid division by zero in the following four equations,  the ratio
        # 1 / {b} was modified to 1 / ({b} + 1e-100)

        P11 = "exp(1j * {{a}} * {{c0}}) * (cos({{a}} * {b}) + 1j * {{c3}} * sin({{a}} * {b}) / ({b} + 1e-100))".format(
            b="sqrt({c1} ** 2 + {c2} ** 2 + {c3} ** 2)"
        )

        P12 = "exp(1j * {{a}} * {{c0}}) * 1j * ({{c1}} - 1j * {{c2}}) * sin({{a}} * {b}) / ({b} + 1e-100)".format(
            b="sqrt({c1} ** 2 + {c2} ** 2 + {c3} ** 2)"
        )

        P21 = "exp(1j * {{a}} * {{c0}}) * 1j * ({{c1}} + 1j * {{c2}}) * sin({{a}} * {b}) / ({b} + 1e-100)".format(
            b="sqrt({c1} ** 2 + {c2} ** 2 + {c3} ** 2)"
        )

        P22 = "exp(1j * {{a}} * {{c0}}) * (cos({{a}} * {b}) - 1j * {{c3}} * sin({{a}} * {b}) / ({b} + 1e-100))".format(
            b="sqrt({c1} ** 2 + {c2} ** 2 + {c3} ** 2)"
        )

        ##################### Generate code for Px = exp( potential energy ) ####################

        Px_params = {
            "a": "(-0.5 * dt)",
            "c0" : self.V0,
            "c1" : self.V1,
            "c2" : self.V2,
            "c3" : self.V3,
        }

        self.code_Px_11 = P11.format(**Px_params)
        self.code_Px_12 = P12.format(**Px_params)
        self.code_Px_21 = P21.format(**Px_params)
        self.code_Px_22 = P22.format(**Px_params)

        # Allocate arrays where the pre-calculated coefficients will be stored
        self.Px_11 = np.empty_like(self.psi1)
        self.Px_12 = np.empty_like(self.psi1)
        self.Px_21 = np.empty_like(self.psi1)
        self.Px_22 = np.empty_like(self.psi1)

        ##################### Generate code for Pp = exp( kinetic energy ) ####################

        Pp_params = {
            "a": "(-dt)",
            "c0" : self.K0,
            "c1" : self.K1,
            "c2" : self.K2,
            "c3" : self.K3,
        }

        self.code_Pb_psi1 = "({}) * psi1_copy + ({}) * psi2_copy".format(P11, P12).format(**Pp_params)
        self.code_Pb_psi2 = "({}) * psi1_copy + ({}) * psi2_copy".format(P21, P22).format(**Pp_params)

        ########################################################################################
        #
        # Codes for calculating the expectation values
        #
        ########################################################################################

        # Codes for calculating the coordinate only dependent expectation value
        coordinate_code = "sum(({}) * (abs(psi1) ** 2 + abs(psi2) ** 2)" \
                          "+ ({}) * 2. * real(conj(psi2) * psi1)" \
                          "- ({}) * 2. * imag(conj(psi2) * psi1)" \
                          "+ ({}) * (abs(psi1) ** 2 - abs(psi2) ** 2))"

        # Codes for calculating the momentum only dependent expectation value
        # Note that the momentum representation of self.psi1 is assumed to be saved in self.psi1_copy
        momentum_code = "sum(({}) * (abs(psi1_copy) ** 2 + abs(psi2_copy) ** 2)" \
                        "+ ({}) * 2. * real(conj(psi2_copy) * psi1_copy)" \
                        "- ({}) * 2. * imag(conj(psi2_copy) * psi1_copy)" \
                        "+ ({}) * (abs(psi1_copy) ** 2 - abs(psi2_copy) ** 2))"

        # coordinate dependent part of the energy operator
        self.energy_potential_code = coordinate_code.format(self.V0, self.V1, self.V2, self.V3)

        # momentum dependent part of the energy operator
        self.energy_kinetic_code = momentum_code.format(self.K0, self.K1, self.K2, self.K3)

        # list to save the energy values
        self.hamiltonian_average = []

        # Lists where the expectation values of X and P
        self.X_average = []
        self.P_average = []

        # Lists where the right hand sides of the Ehrenfest theorems for X and P
        self.X_average_RHS = []
        self.P_average_RHS = []

        self.code_X_average_RHS = momentum_code.format(self.diff_K0, self.diff_K1, self.diff_K2, self.diff_K3)
        self.code_P_average_RHS = coordinate_code.format(self.diff_V0, self.diff_V1, self.diff_V2, self.diff_V3)

        # Call the initialization procedure
        self.post_initialization()

    def post_initialization(self):
        """
        Place holder for the user defined function to be call after at the constructor
        """
        pass

    def propagate(self, time_steps=1):
        """
        Propagate the Pauli-like equation
        :param time_steps: number of time steps
        :return: self
        """
        # pseudonyms
        psi1 = self.psi1
        psi2 = self.psi2

        sys_params = self.sys_params

        for _ in range(time_steps):
            ############################################################################################
            #
            #   Single step propagation
            #
            ############################################################################################

            # make half a step in time
            self.t += 0.5 * self.dt

            # Pre-calculate Px
            evaluate(self.code_Px_11, local_dict=vars(self), global_dict=sys_params, out=self.Px_11)
            evaluate(self.code_Px_12, local_dict=vars(self), global_dict=sys_params, out=self.Px_12)
            evaluate(self.code_Px_21, local_dict=vars(self), global_dict=sys_params, out=self.Px_21)
            evaluate(self.code_Px_22, local_dict=vars(self), global_dict=sys_params, out=self.Px_22)

            # Apply Px
            evaluate(
                "(-1) ** k * (Px_11 * psi1 + Px_12 * psi2)",
                local_dict=vars(self), global_dict=sys_params,
                out=self.psi1_copy
            )
            evaluate(
                "(-1) ** k * (Px_21 * psi1 + Px_22 * psi2)",
                local_dict=vars(self), global_dict=sys_params,
                out=self.psi2_copy
            )

            # go to the momentum representation
            self.psi1_copy = fftpack.fft(self.psi1_copy, overwrite_x=True)
            self.psi2_copy = fftpack.fft(self.psi2_copy, overwrite_x=True)

            # Apply Pp
            evaluate(self.code_Pb_psi1, local_dict=vars(self), global_dict=sys_params, out=self.psi1)
            evaluate(self.code_Pb_psi2, local_dict=vars(self), global_dict=sys_params, out=self.psi2)

            # go to the coordinate representation
            self.psi1 = fftpack.ifft(self.psi1, overwrite_x=True)
            self.psi2 = fftpack.ifft(self.psi2, overwrite_x=True)

            # Apply Px
            evaluate(
                "(-1) ** k * (Px_11 * psi1 + Px_12 * psi2)",
                local_dict=vars(self), global_dict=sys_params,
                out=self.psi1_copy
            )
            evaluate(
                "(-1) ** k * (Px_21 * psi1 + Px_22 * psi2)",
                local_dict=vars(self), global_dict=sys_params,
                out=self.psi2_copy
            )

            # Since the results of the previous step are saved as copies,
            # swap the references between the original and copies
            self.psi1, self.psi1_copy = self.psi1_copy, self.psi1
            self.psi2, self.psi2_copy = self.psi2_copy, self.psi2

            # make half a step in time
            self.t += 0.5 * self.dt

            # normalization
            self.normalize()

            ############################################################################################

            # call user defined post-processing
            self.post_processing()

            # calculate the expectation values, if requested
            self.get_Ehrenfest()

        return self

    def post_processing(self):
        """
        Place holder for the user defined function to be call at the end of each propagation step
        """
        pass

    def normalize(self):
        """
        Normalize the two-component wave function
        :return: self
        """
        normalization = np.sqrt(
            (linalg.norm(self.psi1) ** 2 + linalg.norm(self.psi2) ** 2) * self.dX
        )
        self.psi1 /= normalization
        self.psi2 /= normalization

        return self

    def get_Ehrenfest(self):
        """
        Save the expectation values at time self.t
        :return: None
        """
        if self.isEhrenfest:

            np.copyto(self.psi1_copy, self.psi1)
            np.copyto(self.psi2_copy, self.psi2)

            sys_params = self.sys_params

            # going to the momentum representation
            evaluate("(-1) ** k * psi1_copy", local_dict=vars(self), global_dict=sys_params, out=self.psi1_copy)
            evaluate("(-1) ** k * psi2_copy", local_dict=vars(self), global_dict=sys_params, out=self.psi2_copy)

            self.psi1_copy = fftpack.fft(self.psi1_copy, overwrite_x=True)
            self.psi2_copy = fftpack.fft(self.psi2_copy, overwrite_x=True)

            # going to the momentum representation
            evaluate("(-1) ** k * psi1_copy", local_dict=vars(self), global_dict=sys_params, out=self.psi1_copy)
            evaluate("(-1) ** k * psi2_copy", local_dict=vars(self), global_dict=sys_params, out=self.psi2_copy)

            # normalize the wave function in the momentum representation
            normalization = np.sqrt(
                (linalg.norm(self.psi1_copy) ** 2 + linalg.norm(self.psi2_copy) ** 2) * self.dP
            )
            self.psi1_copy /= normalization
            self.psi2_copy /= normalization

            # Save the energy
            self.hamiltonian_average.append(
                evaluate(self.energy_potential_code, local_dict=vars(self), global_dict=sys_params) * self.dX + \
                evaluate(self.energy_kinetic_code, local_dict=vars(self), global_dict=sys_params) * self.dP
            )

            # Save the mean of X
            self.X_average.append(
                evaluate("sum(X * (abs(psi1) ** 2 + abs(psi2) ** 2))", local_dict=vars(self)) * self.dX
            )

            # and its RHS
            self.X_average_RHS.append(
                evaluate(self.code_X_average_RHS, local_dict=vars(self), global_dict=sys_params) * self.dP
            )

            # Save the mean of P
            self.P_average.append(
                evaluate("sum(P * (abs(psi1_copy) ** 2 + abs(psi2_copy) ** 2))", local_dict=vars(self)) * self.dP
            )

            # and its RHS
            self.P_average_RHS.append(
                -evaluate(self.code_P_average_RHS, local_dict=vars(self), global_dict=sys_params) * self.dX
            )

    @property
    def coordinate_density(self):
        """
        Return a copy of the coordinate density
        :return: numpy.array
        """
        return evaluate("real(abs(psi1) ** 2 + abs(psi2) ** 2)", local_dict=vars(self))

    def set_wavefunction(self, psi1="0. * X", psi2="0. * X"):
        """
        Set the initial wave function
        :param psi1 and psi2: 1D numpy array or string containing the wave function
        :return: self
        """
        if isinstance(psi1, str):
            # psi1 is supplied as a string
            evaluate("({}) + 0j".format(psi1),  local_dict=vars(self), global_dict=self.sys_params, out=self.psi1)

        elif isinstance(psi1, np.ndarray):
            # psi1 is supplied as an array

            # make sure the wavefunction is stored as a complex array
            np.copyto(self.psi1, psi1.astype(np.complex))

        else:
            raise ValueError("psi1 must be either string or numpy.array")

        if isinstance(psi2, str):
            # psi2 is supplied as a string
            evaluate("({}) + 0j".format(psi2),  local_dict=vars(self), global_dict=self.sys_params, out=self.psi2)

        elif isinstance(psi2, np.ndarray):
            # psi2 is supplied as an array

            # make sure the wavefunction is stored as a complex array
            np.copyto(self.psi2, psi2.astype(np.complex))

        else:
            raise ValueError("psi2 must be either string or numpy.array")

        self.normalize()

        return self

##############################################################################
#
#   Run some examples
#
##############################################################################

if __name__ == '__main__':

    # Plotting facility
    import matplotlib.pyplot as plt

    qsys_params = dict(
        X_gridDIM=512,
        X_amplitude=5.,
        dt=0.005,

        omega=2,

        V0="0.5 * (omega * X) ** 2",
        diff_V0="omega ** 2 * X",

        K0="0.5 * P ** 2",
        diff_K0="P",
    )

    qsys_params["V"] = qsys_params["V0"]
    qsys_params["diff_V"] = qsys_params["diff_V0"]

    qsys_params["K"] = qsys_params["K0"]
    qsys_params["diff_K"] = qsys_params["diff_K0"]

    qsys = SplitOpPauliLike1D(**qsys_params).set_wavefunction(psi1="exp(-(X-3.) ** 4)").propagate(4000)

    # Propagate the same system via Schrodinger equation
    from split_op_schrodinger1D import SplitOpSchrodinger1D
    so = SplitOpSchrodinger1D(**qsys_params).set_wavefunction("exp(-(X-3.) ** 4)")
    so.propagate(4000)

    print("\n||Schrodinger - Pauli-like|| = {:.2e}".format(linalg.norm(so.wavefunction - qsys.psi1, np.inf)))
    print("\nSecond compinent of the Pauli like-wavefunction (muist be zero) = {:.2e}".format(linalg.norm(qsys.psi2, np.inf)))

    ##################################################################################################

    plt.subplot(131)
    plt.title("Verify the first Ehrenfest theorem")

    times = qsys.dt * np.arange(len(qsys.X_average))

    plt.plot(
        times,
        np.gradient(qsys.X_average, qsys.dt),
        '-r',
        label='$d\\langle\\hat{x}\\rangle / dt$'
    )
    plt.plot(times, qsys.X_average_RHS, '--b', label='$\\langle\\hat{p}\\rangle$')
    plt.legend()
    plt.ylabel('momentum')
    plt.xlabel('time $t$ (a.u.)')

    plt.subplot(132)
    plt.title("Verify the second Ehrenfest theorem")

    plt.plot(
        times,
        np.gradient(qsys.P_average, qsys.dt),
        '-r',
        label='$d\\langle\\hat{p}\\rangle / dt$'
    )
    plt.plot(times, qsys.P_average_RHS, '--b', label='$\\langle -U\'(\\hat{x})\\rangle$')
    plt.legend()
    plt.ylabel('force')
    plt.xlabel('time $t$ (a.u.)')

    plt.subplot(133)
    plt.title("The expectation value of the hamiltonian")

    # Analyze how well the energy was preserved
    h = np.array(qsys.hamiltonian_average).real
    print(
        "\nHamiltonian is preserved within the accuracy of {:.1e} percent".format(
            100. * (1. - h.min() / h.max())
        )
    )

    plt.plot(times, h)
    plt.ylabel('energy')
    plt.xlabel('time $t$ (a.u.)')

    plt.show()

    ##################################################################################################