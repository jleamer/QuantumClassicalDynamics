import numpy as np
import numexpr as ne
from scipy import fftpack # Tools for fourier transform
from scipy import linalg # Linear algebra for dense matrix
from types import MethodType, FunctionType


class SplitOpSchrodinger1D:
    """
    The second-order split-operator propagator of the 1D Schrodinger equation
    in the coordinate representation
    with the time-dependent Hamiltonian H = K(p, t) + V(x, t).
    """
    def __init__(self, **kwargs):
        """
        The following parameters must be specified
            X_gridDIM - specifying the grid size
            X_amplitude - maximum value of the coordinates
            V - potential energy (as a string to be evaluated by numexpr)
            K - momentum dependent part of the hamiltonian (as a string to be evaluated by numexpr)
            diff_V (optional) -- the derivative of the potential energy for the Ehrenfest theorem calculations
            diff_K (optional) -- the derivative of the kinetic energy for the Ehrenfest theorem calculations
            t (optional) - initial value of time
            abs_boundary (optional) -- absorbing boundary (as a string to be evaluated by numexpr)
        """

        # save all attributes
        for name, value in kwargs.items():
            # if the value supplied is a function, then dynamically assign it as a method;
            # otherwise bind it a property
            if isinstance(value, FunctionType):
                setattr(self, name, MethodType(value, self, self.__class__))
            else:
                setattr(self, name, value)

        # Check that all attributes were specified
        try:
            # make sure self.X_gridDIM has a value of power of 2
            assert 2 ** int(np.log2(self.X_gridDIM)) == self.X_gridDIM, \
                "A value of the grid size (X_gridDIM) must be a power of 2"
        except AttributeError:
            raise AttributeError("Grid size (X_gridDIM) was not specified")

        try:
            self.X_amplitude
        except AttributeError:
            raise AttributeError("Coordinate range (X_amplitude) was not specified")

        try:
            self.V
        except AttributeError:
            raise AttributeError("Potential energy (V) was not specified")

        try:
            self.K
        except AttributeError:
            raise AttributeError("Momentum dependence (K) was not specified")

        try:
            self.dt
        except AttributeError:
            raise AttributeError("Time-step (dt) was not specified")

        try:
            self.t
        except AttributeError:
            print("Warning: Initial time (t) was not specified, thus it is set to zero.")
            self.t = 0.

        try:
            self.abs_boundary
        except AttributeError:
            print("Warning: Absorbing boundary (abs_boundary) was not specified, thus it is turned off")
            self.abs_boundary = 1.

        # it is convenient for some numexprsions to declare pi in the local scope
        self.pi = np.pi

        # get coordinate step size
        self.dX = 2. * self.X_amplitude / self.X_gridDIM

        # generate coordinate range
        self.k = np.arange(self.X_gridDIM)
        self.X = (self.k - self.X_gridDIM / 2) * self.dX

        # generate momentum range
        self.P = (self.k - self.X_gridDIM / 2) * (np.pi / self.X_amplitude)

        # allocate the array for wavefunction
        self.wavefunction = np.zeros(self.X.size, dtype=np.complex)

        # allocate an axillary array needed for propagation
        self.expV = np.zeros_like(self.wavefunction)

        # numexpr code to calculate (-)**k * exp(-0.5j * dt * V)
        self.code_expV = "(%s) * (-1) ** k * exp(-0.5j * dt * (%s))" % (self.abs_boundary, self.V)

        # numexpr code to calculate wavefunction * exp(-1j * dt * K)
        self.code_expK = "wavefunction * exp(-1j * dt * (%s))" % self.K

        # Check whether the necessary terms are specified to calculate the first-order Ehrenfest theorems
        try:
            # numexpr codes to calculate the First Ehrenfest theorems
            self.code_P_average_RHS = "sum((%s) * density)" % self.diff_V
            self.code_V_average = "sum((%s) * density)" % self.V
            self.code_X_average = "sum(X * density)"

            self.code_X_average_RHS = "sum((%s) * density)" % self.diff_K
            self.code_K_average = "sum((%s) * density)" % self.K
            self.code_P_average = "sum(P * density)"

            # Lists where the expectation values of X and P
            self.X_average = []
            self.P_average = []

            # Lists where the right hand sides of the Ehrenfest theorems for X and P
            self.X_average_RHS = []
            self.P_average_RHS = []

            # List where the expectation value of the Hamiltonian will be calculated
            self.hamiltonian_average = []

            # Allocate array for storing coordinate or momentum density of the wavefunction
            self.density = np.zeros(self.wavefunction.shape, dtype=np.float)

            # Allocate a copy of the wavefunction for storing the wavefunction in the momentum representation
            self.wavefunction_p = np.zeros_like(self.wavefunction)

            # Flag requesting tha the Ehrenfest theorem calculations
            self.isEhrenfest = True
        except AttributeError:
            # Since self.diff_V and self.diff_K are not specified,
            # the first Ehrenfest theorem will not be calculated
            self.isEhrenfest = False

    def propagate(self, time_steps=1):
        """
        Time propagate the wave function saved in self.wavefunction
        :param time_steps: number of self.dt time increments to make
        :return: self.wavefunction
        """
        for _ in xrange(time_steps):

            # make a half step in time
            self.t += 0.5 * self.dt

            # efficiently calculate expV
            ne.evaluate(self.code_expV, local_dict=self.__dict__, out=self.expV)
            self.wavefunction *= self.expV

            # going to the momentum representation
            self.wavefunction = fftpack.fft(self.wavefunction, overwrite_x=True)
            ne.evaluate(self.code_expK, local_dict=self.__dict__, out=self.wavefunction)

            # going back to the coordinate representation
            self.wavefunction = fftpack.ifft(self.wavefunction, overwrite_x=True)
            self.wavefunction *= self.expV

            # normalize
            # this line is equivalent to
            # self.wavefunction /= np.sqrt(np.sum(np.abs(self.wavefunction)**2)*self.dX)
            self.wavefunction /= linalg.norm(self.wavefunction) * np.sqrt(self.dX)

            # make a half step in time
            self.t += 0.5 * self.dt

            # calculate the Ehrenfest theorems
            self.get_Ehrenfest()

        return self.wavefunction

    def get_Ehrenfest(self):
        """
        Calculate observables entering the Ehrenfest theorems
        """
        if self.isEhrenfest:
            # evaluate the coordinate density
            np.abs(self.wavefunction, out=self.density)
            self.density *= self.density
            # normalize
            self.density /= self.density.sum()

            # save the current value of <X>
            self.X_average.append(
                ne.evaluate(self.code_X_average, local_dict=self.__dict__)
            )
            self.P_average_RHS.append(
                -ne.evaluate(self.code_P_average_RHS, local_dict=self.__dict__)
            )

            # save the potential energy
            self.hamiltonian_average.append(
                ne.evaluate(self.code_V_average, local_dict=self.__dict__)
            )

            # calculate density in the momentum representation
            ne.evaluate("(-1) ** k * wavefunction", local_dict=self.__dict__, out=self.wavefunction_p)
            self.wavefunction_p = fftpack.fft(self.wavefunction_p, overwrite_x=True)
            np.abs(self.wavefunction_p, out=self.density)
            self.density *= self.density
            # normalize
            self.density /= self.density.sum()

            # save the current value of <P>
            self.P_average.append(
                ne.evaluate(self.code_P_average, local_dict=self.__dict__)
            )
            self.X_average_RHS.append(
                ne.evaluate(self.code_X_average_RHS, local_dict=self.__dict__)
            )

            # add the kinetic energy to get the hamiltonian
            self.hamiltonian_average[-1] += \
                ne.evaluate(self.code_K_average, local_dict=self.__dict__)

    def set_wavefunction(self, wavefunc):
        """
        Set the initial wave function
        :param wavefunc: 1D numpy array or string containing the wave function
        :return: self
        """
        if isinstance(wavefunc, str):
            # wavefunction is supplied as a string
            ne.evaluate("%s + 0j" % wavefunc, local_dict=self.__dict__, out=self.wavefunction)

        elif isinstance(wavefunc, np.ndarray):
            # wavefunction is supplied as an array

            # perform the consistency checks
            assert wavefunc.shape == self.wavefunction.shape,\
                "The grid size does not match with the wave function"

            # make sure the wavefunction is stored as a complex array
            np.copyto(self.wavefunction, wavefunc.astype(np.complex))

        else:
            raise ValueError("wavefunc must be either string or numpy.array")

        # normalize
        self.wavefunction /= linalg.norm(self.wavefunction) * np.sqrt(self.dX)

        return self

##############################################################################
#
#   Run some examples
#
##############################################################################

if __name__ == '__main__':

    # Plotting facility
    import matplotlib.pyplot as plt

    # Use the documentation string for the developed class
    print(SplitOpSchrodinger1D.__doc__)

    for omega in [4., 8.]:

        # save parameters as a separate bundle
        harmonic_osc_params = dict(
            X_gridDIM=512,
            X_amplitude=5.,
            dt=0.01,
            t=0.,

            omega=omega,

            V="0.5 * (omega * X) ** 2",
            diff_V="omega ** 2 * X",

            K="0.5 * P ** 2",
            diff_K="P",
        )

        ##################################################################################################

        # create the harmonic oscillator with time-independent hamiltonian
        harmonic_osc = SplitOpSchrodinger1D(**harmonic_osc_params)

        # set the initial condition
        harmonic_osc.set_wavefunction(
            # The same as np.exp(-(harmonic_osc.X + 3.)**2)
            "exp(-(X + 3.) ** 2)"
        )

        # get time duration of 6 periods
        T = 6 * 2. * np.pi / omega

        # get number of steps necessary to reach time T
        time_steps = int(round(T / harmonic_osc.dt))

        # propagate till time T and for each time step save a probability density
        wavefunctions = [harmonic_osc.propagate().copy() for _ in xrange(time_steps)]

        plt.title("Test 1: Time evolution of harmonic oscillator with $\\omega$ = %.2f (a.u.)" % omega)

        # plot the time dependent density
        plt.imshow(
            np.abs(wavefunctions)**2,
            # some plotting parameters
            origin='lower',
            extent=[harmonic_osc.X.min(), harmonic_osc.X.max(), 0., time_steps * harmonic_osc.dt]
        )
        plt.xlabel('coordinate $x$ (a.u.)')
        plt.ylabel('time $t$ (a.u.)')
        plt.show()

        ##################################################################################################

        plt.subplot(131)
        plt.title("Verify the first Ehrenfest theorem")

        times = harmonic_osc.dt * np.arange(len(harmonic_osc.X_average))
        plt.plot(
            times,
            np.gradient(harmonic_osc.X_average, harmonic_osc.dt),
            '-r',
            label='$d\\langle\\hat{x}\\rangle / dt$'
        )
        plt.plot(times, harmonic_osc.X_average_RHS, '--b', label='$\\langle\\hat{p}\\rangle$')
        plt.legend()
        plt.ylabel('momentum')
        plt.xlabel('time $t$ (a.u.)')

        plt.subplot(132)
        plt.title("Verify the second Ehrenfest theorem")

        plt.plot(
            times,
            np.gradient(harmonic_osc.P_average, harmonic_osc.dt),
            '-r',
            label='$d\\langle\\hat{p}\\rangle / dt$'
        )
        plt.plot(times, harmonic_osc.P_average_RHS, '--b', label='$\\langle -U\'(\\hat{x})\\rangle$')
        plt.legend()
        plt.ylabel('force')
        plt.xlabel('time $t$ (a.u.)')

        plt.subplot(133)
        plt.title("The expectation value of the hamiltonian")

        # Analyze how well the energy was preserved
        h = np.array(harmonic_osc.hamiltonian_average)
        print(
            "\nHamiltonian is preserved within the accuracy of %.2e percent" % (100. * (1. - h.min() / h.max()))
        )

        plt.plot(times, h)
        plt.ylabel('energy')
        plt.xlabel('time $t$ (a.u.)')

        plt.show()

        ##################################################################################################

        # re-create the harmonic oscillator
        harmonic_osc = SplitOpSchrodinger1D(**harmonic_osc_params)

        # as the second test, let's check that an eigenstate state is a stationary state
        # to find a good ground state let's use the mutually unbiased bases method already implemented
        from mub_qhamiltonian import MUBQHamiltonian
        eigenstate = MUBQHamiltonian(**harmonic_osc_params).get_eigenstate(3)

        # set the initial condition
        harmonic_osc.set_wavefunction(eigenstate)

        plt.subplot(121)
        plt.title("Test 2: Time evolution of eigenstate\n obtained via MUB")

        # enable log color plot
        from matplotlib.colors import LogNorm

        # propagate and plot
        plt.imshow(
            [np.abs(harmonic_osc.propagate())**2 for _ in xrange(time_steps)],
            # some plotting parameters
            origin='lower',
            extent=[harmonic_osc.X.min(), harmonic_osc.X.max(), 0., time_steps * harmonic_osc.dt],
            norm = LogNorm(1e-16, 1.)
        )
        plt.colorbar()
        plt.xlabel('coordinate $x$ (a.u.)')
        plt.ylabel('time $t$ (a.u.)')

        ##################################################################################################

        # re-create the harmonic oscillator
        harmonic_osc = SplitOpSchrodinger1D(**harmonic_osc_params)

        # let's see what happens if we use an eigenstate generated via the central finite difference method
        from central_diff_qhamiltonian import CentralDiffQHamiltonian
        eigenstate = CentralDiffQHamiltonian(**harmonic_osc_params).get_eigenstate(3)

        # set the initial condition
        harmonic_osc.set_wavefunction(eigenstate)

        plt.subplot(122)
        plt.title("Test 3: Time evolution of eigenstate\n obtained via central finite difference")

        # propagate and plot
        plt.imshow(
            [np.abs(harmonic_osc.propagate())**2 for _ in xrange(time_steps)],
            # some plotting parameters
            origin='lower',
            extent=[harmonic_osc.X.min(), harmonic_osc.X.max(), 0., time_steps * harmonic_osc.dt],
            norm=LogNorm(1e-16, 1.)
        )
        plt.colorbar()
        plt.xlabel('coordinate $x$ (a.u.)')
        plt.ylabel('time $t$ (a.u.)')

        plt.show()