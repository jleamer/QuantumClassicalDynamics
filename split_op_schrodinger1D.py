import numpy as np
from scipy import fftpack # Tools for fourier transform
from scipy import linalg # Linear algebra for dense matrix


class SplitOpSchrodinger1D:
    """
    The second-order split-operator propagator of the 1D Schrodinger equation
    in the coordinate representation
    with the time-dependent Hamiltonian H = K(p, t) + V(x, t).
    (K and V may not depend on time)
    """
    def __init__(self, **kwargs):
        """
        The following parameters must be specified
            X_gridDIM - specifying the grid size
            X_amplitude - maximum value of the coordinates
            V(x) - potential energy (as a function) may depend on time
            diff_V(x) (optional) -- the derivative of the potential energy for the Ehrenfest theorem calculations
            K(p) - momentum dependent part of the hamiltonian (as a function) may depend on time
            diff_K(p) (optional) -- the derivative of the kinetic energy for the Ehrenfest theorem calculations
            t (optional) - initial value of time
        """

        # save all attributes
        for name, value in kwargs.items():
            setattr(self, name, value)

        # Check that all attributes were specified
        try:
            self.X_gridDIM
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

        # get coordinate step size
        self.dX = 2.*self.X_amplitude / self.X_gridDIM

        # generate coordinate range
        self.X_range = np.linspace(-self.X_amplitude, self.X_amplitude - self.dX , self.X_gridDIM)

        # generate momentum range as it corresponds to FFT frequencies
        self.P_range = fftpack.fftfreq(self.X_gridDIM, self.dX/(2*np.pi))

        # get the momentum step size
        self.dP = self.P_range[1] - self.P_range[0]

        try:
            # Pre-calculate the exponent, if the potential is time independent
            self._expV = np.exp(-self.dt*0.5j*self.V(self.X_range))
        except TypeError:
            # If exception is generated, then the potential is time-dependent
            # and caching is not possible
            pass

        try:
            # Pre-calculate the exponent, if the kinetic energy is time independent
            self._expK = np.exp(-self.dt*1j*self.K(self.P_range))
        except TypeError:
            # If exception is generated, then the kinetic energy is time-dependent
            # and caching is not possible
            pass

        # Check whether the necessary terms are specified to calculate the Ehrenfest theorems
        try:
            # Pre-calculate RHS if time independent
            try:
                self._diff_V = self.diff_V(self.X_range)
            except TypeError:
                pass

            # Pre-calculate RHS if time independent
            try:
                self._diff_K = self.diff_K(self.P_range)
            except TypeError:
                pass

            # Pre-calculate the potential and kinetic energies for
            # calculating the expectation value of Hamiltonian
            try:
                self._V = self.V(self.X_range)
            except TypeError:
                pass
            try:
                self._K = self.K(self.P_range)
            except TypeError:
                pass

            # Lists where the expectation values of X and P
            self.X_average = []
            self.P_average = []

            # Lists where the right hand sides of the Ehrenfest theorems for X and P
            self.X_average_RHS = []
            self.P_average_RHS = []

            # List where the expectation value of the Hamiltonian will be calculated
            self.hamiltonian_average = []

            # Flag requesting tha the Ehrenfest theorem calculations
            self.isEhrenfest = True
        except AttributeError:
            # Since self.diff_V and self.diff_K are not specified,
            # the Ehrenfest theorem will not be calculated
            self.isEhrenfest = False


    def propagate(self, time_steps=1):
        """
        Time propagate the wave function saved in self.wavefunction
        :param time_steps: number of self.dt time increments to make
        :return: self.wavefunction
        """
        for _ in xrange(time_steps):

            expV = self.get_expV(self.t)
            self.wavefunction *= expV

            # going to the momentum representation
            self.wavefunction = fftpack.fft(self.wavefunction, overwrite_x=True)
            self.wavefunction *= self.get_expK(self.t)

            # going back to the coordinate representation
            self.wavefunction = fftpack.ifft(self.wavefunction, overwrite_x=True)
            self.wavefunction *= expV

            # normalize
            # this line is equivalent to
            # self.wavefunction /= np.sqrt(np.sum(np.abs(self.wavefunction)**2)*self.dX)
            self.wavefunction /= linalg.norm(self.wavefunction) * np.sqrt(self.dX)

            # calculate the Ehrenfest theorems
            self.get_Ehrenfest(self.t)

            # increment current time
            self.t += self.dt

        return self.wavefunction

    def get_Ehrenfest(self, t):
        """
        Calculate observables entering the Ehrenfest theorems at time (t)
        """
        if self.isEhrenfest:
            # calculate the coordinate density
            density_coord = np.abs(self.wavefunction)**2
            # normalize
            density_coord /= density_coord.sum()

            # save the current value of <X>
            self.X_average.append(
                np.dot(density_coord, self.X_range)
            )
            self.P_average_RHS.append(
                -np.dot(density_coord, self.get_diff_V(t))
            )

            # calculate density in the momentum representation
            density_momentum = np.abs(fftpack.fft(self.wavefunction))**2
            # normalize
            density_momentum /= density_momentum.sum()

            # save the current value of <P>
            self.P_average.append(
                np.dot(density_momentum, self.P_range)
            )
            self.X_average_RHS.append(
                np.dot(density_momentum, self.get_diff_K(t))
            )

            # save the current expectation value of energy
            self.hamiltonian_average.append(
                np.dot(density_coord, self.get_V(t))
                +
                np.dot(density_momentum, self.get_K(t))
            )

    def get_expV(self, t):
        """
        Return the exponent of the potential energy at time (t)
        """
        try:
            # access the pre-calculated value
            return self._expV
        except AttributeError:
            # calculate the exponent
            return np.exp(-self.dt*0.5j*self.V(self.X_range, t))

    def get_expK(self, t):
        """
        Return the exponent of the kinetic energy at time (t)
        """
        try:
            # access the pre-calculated value
            return self._expK
        except AttributeError:
            # calculate the exponent
            return np.exp(-self.dt*1j*self.K(self.P_range, t))

    def get_diff_V(self, t):
        """
        Return the RHS for the Ehrenfest theorem at time (t)
        """
        try:
            # access the pre-calculated value
            return self._diff_V
        except AttributeError:
            return self.diff_V(self.X_range, t)

    def get_diff_K(self, t):
        """
        Return the RHS for the Ehrenfest theorem at time (t)
        """
        try:
            # access the pre-calculated value
            return self._diff_K
        except AttributeError:
            return self.diff_K(self.P_range, t)

    def get_K(self, t):
        """
        Return the kinetic energy at time (t)
        """
        try:
            return self._K
        except AttributeError:
            return self.K(self.P_range, t)

    def get_V(self, t):
        """
        Return the potential energy at time (t)
        """
        try:
            return self._V
        except AttributeError:
            return self.V(self.X_range, t)

    def set_wavefunction(self, wavefunc):
        """
        Set the initial wave function
        :param wavefunc: 1D numoy array contaning the wave function
        :return: self
        """
        # perform the consistency checks
        assert wavefunc.shape == self.X_range.shape,\
            "The grid size does not match with the wave function"

        # make sure the wavefunction is stored as a complex array
        self.wavefunction = wavefunc + 0j

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

            V=lambda x: 0.5*(omega*x)**2,
            diff_V=lambda x: omega**2*x,

            K=lambda p: 0.5*p**2,
            diff_K=lambda p: p
        )

        ##################################################################################################

        # create the harmonic oscillator with time-independent hamiltonian
        harmonic_osc = SplitOpSchrodinger1D(**harmonic_osc_params)

        # set the initial condition
        harmonic_osc.set_wavefunction(
            np.exp(-(harmonic_osc.X_range + 3.)**2)
        )

        # get time duration of 6 periods
        T = 6* 2.*np.pi/omega
        # get number of steps necessary to reach time T
        time_steps = int(round(T/harmonic_osc.dt))

        # propagate till time T and for each time step save a probability density
        wavefunctions = [harmonic_osc.propagate().copy() for _ in xrange(time_steps)]

        plt.title("Test 1: Time evolution of harmonic oscillator with $\\omega$ = %.2f (a.u.)" % omega)

        # plot the time dependent density
        plt.imshow(
            np.abs(wavefunctions)**2,
            # some plotting parameters
            origin='lower',
            extent=[harmonic_osc.X_range.min(), harmonic_osc.X_range.max(), 0., time_steps*harmonic_osc.dt]
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

        plt.plot(times, harmonic_osc.hamiltonian_average)
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
        plt.title("Test 2: Time evolution of eigenstate obtained via MUB")

        # propagate and plot
        plt.imshow(
            [np.abs(harmonic_osc.propagate())**2 for _ in xrange(time_steps)],
            # some plotting parameters
            origin='lower',
            extent=[harmonic_osc.X_range.min(), harmonic_osc.X_range.max(), 0., time_steps*harmonic_osc.dt]
        )
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
        plt.title("Test 3: Time evolution of eigenstate obtained via central finite difference")

        # propagate and plot
        plt.imshow(
            [np.abs(harmonic_osc.propagate())**2 for _ in xrange(time_steps)],
            # some plotting parameters
            origin='lower',
            extent=[harmonic_osc.X_range.min(), harmonic_osc.X_range.max(), 0., time_steps*harmonic_osc.dt]
        )
        plt.xlabel('coordinate $x$ (a.u.)')
        plt.ylabel('time $t$ (a.u.)')

        plt.show()