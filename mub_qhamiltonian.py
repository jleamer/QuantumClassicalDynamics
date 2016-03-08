import numpy as np
from scipy import fftpack # Tools for fourier transform
from scipy import linalg # Linear algebra for dense matrix
import matplotlib.pyplot as plt # Plotting facility


class MUBQHamiltonian:
    """
    Generate quantum Hamiltonian, H(x,p) = K(p) + V(x),
    for 1D system in the coordinate representation using mutually unbiased bases (MUB).
    """
    def __init__(self, **kwards):
        """
        The following parameters must be specified
            X_gridDIM - specifying the grid size
            X_amplitude - maximum value of the coordinates
            V(x) - potential energy (as a function)
            K(p) - momentum dependent part of the hamiltonian (as a function)
        """

        # save all attributes
        for name, value in kwards.items():
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

        # get coordinate step size
        self.dX = 2.*self.X_amplitude / self.X_gridDIM

        # generate coordinate range
        self.X_range = np.linspace(-self.X_amplitude, self.X_amplitude - self.dX , self.X_gridDIM)

        # generate momentum range as it corresponds to FFT frequencies
        self.P_range = fftpack.fftfreq(self.X_gridDIM, self.dX/(2*np.pi))

        # Construct the momentum dependent part
        self.Hamiltonian = fftpack.fft(np.diag(self.K(self.P_range)), axis=1, overwrite_x=True)
        self.Hamiltonian = fftpack.ifft(self.Hamiltonian, axis=0, overwrite_x=True)

        # Add diagonal potential energy
        self.Hamiltonian += np.diag(self.V(self.X_range))

##############################################################################
#
#   Run some examples
#
##############################################################################

if __name__ == '__main__':

    print("Hamiltonian via mutually unbiased bases (MUB)")

    for omega in [4., 8.]:
        # Find energies of a harmonic oscillator V = 0.5*(omega*x)**2
        harmonic_osc = MUBQHamiltonian(
                            X_gridDIM=512,
                            X_amplitude=5.,
                            V=lambda x: 0.5*(omega*x)**2,
                            K=lambda p: 0.5*p**2
                        )
        # get real sorted energies and underlying wavefunctions
        # using specialized function for Hermitian matrices
        energies, wavefunctions = linalg.eigh(harmonic_osc.Hamiltonian)

        print("\n\nFirst energies for harmonic oscillator with omega = %f" % omega)
        print energies[:20]

        # plot eigenfunctions
        for n in range(4):
            plt.plot(harmonic_osc.X_range, wavefunctions[:,n], label=str(n))

        plt.title("Eigenfunctions for harmonic oscillator with omega = %.2f (a.u.)" % omega)
        plt.xlabel('$x$ (a.u.)')
        plt.ylabel('wave functions ($\\psi_n(x)$)')
        plt.legend()
        plt.show()