import numpy as np
from scipy import fftpack # Tools for fourier transform
from scipy import linalg # Linear algebra for dense matrix
import matplotlib.pyplot as plt # Plotting facility


class SplitOpSchrodinger1D:
    """
    Split-operator propagator of the 1D Schrodinger equation
    with the time-dependent Hamiltonian H = K(p, t) + V(x, t).
    (K and V may not depend on time)
    """
    def __init__(self, **kwargs):
        """
        The following parameters must be specified
            X_gridDIM - specifying the grid size
            X_amplitude - maximum value of the coordinates
            V(x) - potential energy (as a function) may depend on time
            K(p) - momentum dependent part of the hamiltonian (as a function) may depend on time
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

        try:
            # Pre-calculate the exponent, if the potential is time independent
            self._expV = np.exp(-self.dt*1j*self.V(self.X_range))
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

    def propagate(self, time_steps=1):
        """
        Time propagate the wave function saved in self.wavefunction
        :param time_steps: number of self.dt time increments to make
        :return: self.wavefunction
        """
        for _ in xrange(time_steps):
            self.wavefunction *= self.get_expV(self.t)

            # going to the momentum representation
            self.wavefunction = fftpack.fft(self.wavefunction, overwrite_x=True)
            self.wavefunction *= self.get_expK(self.t)

            # going back to the coordinate representation
            self.wavefunction = fftpack.ifft(self.wavefunction, overwrite_x=True)

            # normalize (this step is in principle optional)
            self.wavefunction /= np.sqrt(np.sum(np.abs(self.wavefunction)**2)*self.dX)

            # increment current time
            self.t += self.dt

        return self.wavefunction

    def get_expV(self, t):
        """
        Return the exponent of the potential energy at time (t)
        """
        try:
            # aces the pre-calculated value
            return self._expV
        except AttributeError:
            # re-calculate the exponent
            return np.exp(-self.dt*1j*self.V(self.X_range, t))

    def get_expK(self, t):
        """
        Return the exponent of the kinetic energy at time (t)
        """
        try:
            # aces the pre-calculated value
            return self._expK
        except AttributeError:
            # re-calculate the exponent
            return np.exp(-self.dt*1j*self.K(self.X_range, t))

##############################################################################
#
#   Run some examples
#
##############################################################################

if __name__ == '__main__':

    # Use the documentation string for the developed class
    print(SplitOpSchrodinger1D.__doc__)

    for omega in [4., 8.]:

        # save parameters as a separate bundle
        harmonic_osc_params = dict(
            X_gridDIM=512,
            X_amplitude=5.,
            dt=0.001,
            V=lambda x: 0.5*(omega*x)**2,
            K=lambda p: 0.5*p**2
        )

        ##################################################################################################

        # create the harmonic oscillator with time-independent hamiltonian
        harmonic_osc = SplitOpSchrodinger1D(**harmonic_osc_params)

        # set the initial condition (note that the wave function should be of complex type)
        harmonic_osc.wavefunction = np.exp(-(harmonic_osc.X_range - 2)**2) + 0j

        # get time duration of 6 periods
        T = 6* 2.*np.pi/omega
        time_steps = int(T/harmonic_osc.dt)

        plt.title("Test 1: Time evolution of harmonic oscillator with $\\omega$ = %.2f (a.u.)" % omega)

        # propagate for time T and for each time step save a probability density
        density = [np.abs(harmonic_osc.propagate())**2 for _ in xrange(time_steps)]

        # plot
        plt.imshow(
            density,
            # some plotting parameters
            origin='lower',
            extent=[harmonic_osc.X_range.min(), harmonic_osc.X_range.max(), 0., time_steps*harmonic_osc.dt]
        )
        plt.xlabel('coordinate $x$ (a.u.)')
        plt.ylabel('time $t$ (a.u.)')
        plt.show()

        ##################################################################################################

        # re-create the harmonic oscillator
        harmonic_osc = SplitOpSchrodinger1D(**harmonic_osc_params)

        # as the second test, let's check that the ground state is a stationary state
        # to find a good ground state let's use the mutually unbiased bases method already implemented
        from mub_qhamiltonian import MUBQHamiltonian
        eigenstates = linalg.eigh(MUBQHamiltonian(**harmonic_osc_params).Hamiltonian)[1]

        # set the initial condition
        harmonic_osc.wavefunction = eigenstates[:,3] + 0j

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

        # let's see what happens if we use and eigenstate generated via the central finite difference method
        from central_diff_qhamiltonian import CentralDiffQHamiltonian
        eigenstates = linalg.eigh(CentralDiffQHamiltonian(**harmonic_osc_params).Hamiltonian.toarray())[1]

        # set the initial condition
        harmonic_osc.wavefunction = eigenstates[:,3] + 0j

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