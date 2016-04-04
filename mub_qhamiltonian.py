import numpy as np
from scipy import fftpack # Tools for fourier transform
from scipy import linalg # Linear algebra for dense matrix


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

    def get_eigenstate(self, n):
        """
        Return n-th eigenfunction
        :param n: order
        :return: a copy of numpy array containing eigenfunction
        """
        self.diagonalize()
        return self.eigenstates[n].copy()

    def get_energy(self, n):
        """
        Return the energy of the n-th eigenfunction
        :param n: order
        :return: real value
        """
        self.diagonalize()
        return self.energies[n]

    def diagonalize(self):
        """
        Diagonalize the Hamiltonian if necessary
        :return: self
        """
        # check whether the hamiltonian has been diagonalized
        try:
            self.eigenstates
            self.energies
        except AttributeError:
            # eigenstates have not been calculated so
            # get real sorted energies and underlying wavefunctions
            # using specialized function for Hermitian matrices
            self.energies, self.eigenstates = linalg.eigh(self.Hamiltonian)

            # extract real part of the energies
            self.energies = np.real(self.energies)

            # covert to the formal convenient for storage
            self.eigenstates = np.ascontiguousarray(self.eigenstates.T.real)

            # normalize each eigenvector
            for psi in self.eigenstates:
                psi /= linalg.norm(psi) * np.sqrt(self.dX)

            # check that the ground state is not negative
            np.abs(self.eigenstates[0], out=self.eigenstates[0])

        return self

##############################################################################
#
#   Run some examples
#
##############################################################################

if __name__ == '__main__':

    import matplotlib.pyplot as plt # Plotting facility

    print("Hamiltonian via mutually unbiased bases (MUB)")

    for omega in [4., 8.]:
        # Find energies of a harmonic oscillator V = 0.5*(omega*x)**2
        harmonic_osc = MUBQHamiltonian(
                            X_gridDIM=512,
                            X_amplitude=5.,
                            V=lambda x: 0.5*(omega*x)**2,
                            K=lambda p: 0.5*p**2
                        )

        # plot eigenfunctions
        for n in range(4):
            plt.plot(harmonic_osc.X_range, harmonic_osc.get_eigenstate(n), label=str(n))

        print("\n\nFirst energies for harmonic oscillator with omega = %f" % omega)
        print(harmonic_osc.energies[:20])

        plt.title("Eigenfunctions for harmonic oscillator with omega = %.2f (a.u.)" % omega)
        plt.xlabel('$x$ (a.u.)')
        plt.ylabel('wave functions ($\\psi_n(x)$)')
        plt.legend()
        plt.show()