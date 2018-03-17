# safe import
try:
    from split_op_schrodinger1D import SplitOpSchrodinger1D, ne, np, fftpack, linalg
except ModuleNotFoundError:
    from .split_op_schrodinger1D import SplitOpSchrodinger1D, ne, np, fftpack, linalg

from numpy.random import uniform

# We will use the inheritance in the object orienting programing (see, e.g.,
# https://en.wikipedia.org/wiki/Inheritance_%28object-oriented_programming%29 and
# https://docs.python.org/2/tutorial/classes.html)
# to add methods to already developed propagator (SplitOpSchrodinger1D)
# that find stationary states via the imaginary-time propagation


class WavefuncMonteCarloPoission(SplitOpSchrodinger1D):
    """
    Wavefunction Monte Carlo with Poission noise, 
    as described in Sec. 4.3.4 of 
        K. Jacobs "Quantum measurement theory and its applications" (Cambridge University, 2014)
    """
    def __init__(self, **kwargs):
        """
        Parameters:
            AdaggerA_X -- list of ne expressions for the coordinate dependent product of the dissipator
            apply_A -- list of the functions applying the operator A onto the wavefunction

            BdaggerB_P -- list of ne expressions for the coordinate dependent product of the dissipator
            apply_B -- list of the functions applying the operator B onto the wavefunction
        """
        # Extract and save the list of dissipators
        AdaggerA_X = kwargs.pop("AdaggerA_X", [])
        self.apply_A = kwargs.pop("apply_A", [])
        assert len(self.apply_A) == len(AdaggerA_X), "Lengths of AdaggerA_X and apply_A must be equal"

        BdaggerB_P = kwargs.pop("BdaggerB_P", [])
        self.apply_B = kwargs.pop("apply_B", [])
        assert len(self.apply_B) == len(BdaggerB_P), "Lengths of BdaggerB_P and apply_B must be equal"

        # Put the operators in the brackets, just to be on safe side for compilation
        AdaggerA_X = ["({})".format(_) for _ in AdaggerA_X]
        BdaggerB_P = ["({})".format(_) for _ in BdaggerB_P]

        # ne codes for calculating lambda_{A_k}(t) = < A_k^\dagger A_k (x) >
        self.code_lambda_A = ["sum({} * abs(wavefunction) ** 2)".format(_) for _ in AdaggerA_X]

        # ne codes for calculating lambda_{B_k}(t) = < B_k^\dagger B_k (p) >
        self.code_lambda_B = ["sum({} * density)".format(_) for _ in BdaggerB_P]

        # Set the arrays for P_A and P_B
        self.P_A = np.ones(len(AdaggerA_X), dtype=np.float)
        self.r_A = uniform(0., 1., len(AdaggerA_X))

        self.P_B = np.ones(len(BdaggerB_P), dtype=np.float)
        self.r_B = uniform(0., 1., len(BdaggerB_P))

        # Modify the potential energy with the contribution \
        # from the coordinate dependent dissipators
        kwargs["V"] = "{} -0.5j * ({})".format(
            kwargs.pop("V"),
            ("+ ".join(AdaggerA_X) if AdaggerA_X else "0.")
        )

        # the save for the kinetic energy
        kwargs["K"] = "{} -0.5j * ({})".format(
            kwargs.pop("K"),
            ("+ ".join(BdaggerB_P) if BdaggerB_P else "0.")
        )

        # Call the parent constructor
        super().__init__(**kwargs)

        if BdaggerB_P:
            # Allocate a copy of the wavefunction for storing the wavefunction in the momentum representation
            # if the momentum dependent dissipator is given
            self.wavefunction_p = np.zeros_like(self.wavefunction)
            # and also for density
            self.density = np.zeros(self.wavefunction_p.shape, dtype=np.float)

        self.AdaggerA_X = AdaggerA_X
        self.BdaggerB_P = BdaggerB_P

    def propagate(self, time_steps=1):
        """
        Perform the wave function Monte Carlo propagation
        :param time_steps:  number of self.dt time increments to make
        :return: self.wavefunction
        """
        # a boolean flag indicating whether a quantum jump has happened
        jump = False

        # Calculate lambda_A(t) and lambda_B(t)
        lambda_A_previous = self.get_lambda_A()
        lambda_B_previous = self.get_lambda_B()

        for _ in range(time_steps):

            # advance the wavefunction by dt using the modified Schrodinger equation
            # where additional dissipator terms where included
            self.single_step_propagation()

            # Calculate lambda_A(t + dt) and lambda_B(t + dt)
            lambda_A_next = self.get_lambda_A()
            lambda_B_next = self.get_lambda_B()

            # update P_A and P_B
            self.P_A *= np.exp(-0.5 * self.dt * (lambda_A_next + lambda_A_previous))
            self.P_B *= np.exp(-0.5 * self.dt * (lambda_B_next + lambda_B_previous))

            ####################################################################################
            #
            #   Apply projector operators
            #
            ####################################################################################
            for k in np.where(self.P_A <= self.r_A)[0]:
                jump = True

                # reset the probabilities
                self.P_A[k] = 1.
                self.r_A[k] = uniform()

                # Apply the dissipator onto the wavefunction
                self.apply_A[k](self)

            for k in np.where(self.P_B <= self.r_B)[0]:
                jump = True

                # reset the probabilities
                self.P_B[k] = 1.
                self.r_B[k] = uniform()

                # Apply the dissipator onto the wavefunction
                self.apply_B[k](self)

            # the wave function jumped, then normalize and re-calculate lambdas
            if jump:
                # normalize
                self.wavefunction /= linalg.norm(self.wavefunction) * np.sqrt(self.dX)

                # Calculate lambda_A(t + dt) and lambda_B(t + dt)
                lambda_A_previous = self.get_lambda_A()
                lambda_B_previous = self.get_lambda_B()

                jump = False
            else:
                # update lambda_A(t) and lambda_B(t)
                lambda_A_previous = lambda_A_next
                lambda_B_previous = lambda_B_next

        return self.wavefunction

    def get_lambda_A(self):
        """
        Calculate lambda_{A_k}(t) = < A_k^\dagger A_k (x) > for all k
        :return: np.array
        """
        # Return 0 if there are no A operators
        if not self.AdaggerA_X:
            return 0.

        result = np.fromiter(
            (ne.evaluate(code, local_dict=vars(self)).real for code in self.code_lambda_A),
            np.float, len(self.code_lambda_A)
        )
        result *= self.dX
        return result

    def get_lambda_B(self):
        """
        Calculate lambda_{A_k}(t) = < A_k^\dagger A_k (x) > for all k
        :return: np.array
        """
        # Return 0 if there are no B operators
        if not self.BdaggerB_P:
            return 0.

        # calculate density in the momentum representation
        ne.evaluate("(-1) ** k * wavefunction", local_dict=vars(self), out=self.wavefunction_p)

        self.wavefunction_p = fftpack.fft(self.wavefunction_p, overwrite_x=True)

        # get the density in the momentum space
        ne.evaluate("real(abs(wavefunction_p)) ** 2", local_dict=vars(self), out=self.density)

        # normalize
        self.density /= self.density.sum()

        result = np.fromiter(
            (ne.evaluate(code, local_dict=vars(self)).real for code in self.code_lambda_B),
            np.float, len(self.code_lambda_B)
        )
        return result

##############################################################################
#
#   Run some examples
#
##############################################################################

if __name__ == '__main__':

    # Plotting facility
    import matplotlib.pyplot as plt

    # Use the documentation string for the developed class
    print(WavefuncMonteCarloPoission.__doc__)

    def apply_A(self):
        """
        Apply X onto the wavefunction
        :param self:
        :return: None
        """
        self.wavefunction *= self.X

    def apply_B(self):
        """
        Apply X onto the wavefunction
        :param self:
        :return: None
        """
        # Go to the momentum representation
        ne.evaluate("(-1) ** k * wavefunction", local_dict=vars(self), out=self.wavefunction)
        self.wavefunction = fftpack.fft(self.wavefunction, overwrite_x=True)

        self.wavefunction *= self.P

        # Go back to the coordinate representation
        self.wavefunction = fftpack.ifft(self.wavefunction, overwrite_x=True)
        ne.evaluate("(-1) ** k * wavefunction", local_dict=vars(self), out=self.wavefunction)

    params = dict(
        X_gridDIM=512,
        X_amplitude=20.,
        dt=0.01,
        t=0.,

        omega=uniform(4, 8),

        V="0.5 * (omega * X) ** 2",
        diff_V="omega ** 2 * X",

        K="0.5 * P ** 2",
        diff_K="P",

        # Specify the dissipators
        gammaA = uniform(0., 0.5),

        AdaggerA_X=("(gammaA * X) ** 2",),
        apply_A=(apply_A,),

        gammaB=uniform(0., 0.2),

        BdaggerB_P=("(gammaB * P) ** 2",),
        apply_B=(apply_B,),
    )

    # initiate the propagator
    q_sys = WavefuncMonteCarloPoission(**params)


    # set the ground state as the initial state
    from mub_qhamiltonian import MUBQHamiltonian

    q_sys.set_wavefunction(
        MUBQHamiltonian(**params).get_eigenstate(0)
    )

    ################################################################################
    #
    #   Propagate
    #
    ################################################################################

    density_evolution = []
    P_A = []
    P_B = []

    r_A = []
    r_B = []

    time = []

    for t in range(3000):
        density_evolution.append(np.abs(q_sys.propagate()) ** 2)

        time.append(q_sys.t)

        P_A.append(q_sys.P_A[0])
        P_B.append(q_sys.P_B[0])

        r_A.append(q_sys.r_A[0])
        r_B.append(q_sys.r_B[0])

    density_evolution = np.array(density_evolution)
    time = np.array(time)

    ################################################################################
    #
    #   Plot
    #
    ################################################################################

    # enable log color plot
    from matplotlib.colors import LogNorm

    # set the common X axis for the plots
    _, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

    ax1.set_title("Wave function evolution (log color scheme)")
    ax1.imshow(
        density_evolution.T,
        # some plotting parameters
        origin='lower',
        extent=[time.min(), time.max(), q_sys.X.min(), q_sys.X.max()],
        norm=LogNorm(1e-6, 1.)
    )
    ax1.set_ylabel('coordinate $x$ (a.u.)')

    ax2.set_title("Probability plot")

    ax2.plot(time, P_A, '-', label='Probablity $P_A(t)$')
    ax2.plot(time, P_B, '-', label='Probablity $P_B(t)$')

    ax2.plot(time, r_A, '-', label='Chosen constant $r_A$')
    ax2.plot(time, r_B, '-', label='Chosen constant $r_B$')

    ax2.legend()
    ax2.set_ylabel('Probablity $P_A(t)$')
    ax2.set_xlabel('time $t$ (a.u.)')

    plt.show()