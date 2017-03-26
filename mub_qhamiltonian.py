import numpy as np
import numexpr as ne
from scipy import fftpack # Tools for fourier transform
from scipy import linalg # Linear algebra for dense matrix
from types import MethodType, FunctionType


class MUBQHamiltonian:
    """
    Generate quantum Hamiltonian, H(x,p) = K(p) + V(x),
    for 1D system in the coordinate representation using mutually unbiased bases (MUB).
    """
    def __init__(self, **kwargs):
        """
        The following parameters must be specified
            X_gridDIM - specifying the grid size
            X_amplitude - maximum value of the coordinates
            V - potential energy (as a string to be evaluated by numexpr)
            K - momentum dependent part of the hamiltonian (as a string to be evaluated by numexpr)
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
            # make sure self.X_amplitude has a value of power of 2
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

        # get coordinate step size
        self.dX = 2. * self.X_amplitude / self.X_gridDIM

        # generate coordinate range
        k = np.arange(self.X_gridDIM)
        self.X = (k - self.X_gridDIM / 2) * self.dX
        # The same as
        # self.X = np.linspace(-self.X_amplitude, self.X_amplitude - self.dX , self.X_gridDIM)

        # generate momentum range as it corresponds to FFT frequencies
        self.P = (k - self.X_gridDIM / 2) * (np.pi / self.X_amplitude)

        # 2D array of alternating signs
        minus = (-1) ** (k[:, np.newaxis] + k[np.newaxis, :])

        # Construct the momentum dependent part
        self.Hamiltonian = np.diag(
            ne.evaluate(self.K, local_dict=self.__dict__)
        )
        self.Hamiltonian *= minus
        self.Hamiltonian = fftpack.fft(self.Hamiltonian, axis=1, overwrite_x=True)
        self.Hamiltonian = fftpack.ifft(self.Hamiltonian, axis=0, overwrite_x=True)
        self.Hamiltonian *= minus

        # Add diagonal potential energy
        self.Hamiltonian += np.diag(
            ne.evaluate(self.V, local_dict=self.__dict__)
        )

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

    print(MUBQHamiltonian.__doc__)

    for omega in [4., 8.]:
        # Find energies of a harmonic oscillator V = 0.5*(omega*x)**2
        harmonic_osc = MUBQHamiltonian(
                            X_gridDIM=512,
                            X_amplitude=5.,
                            omega=omega,
                            V="0.5 * (omega * X) ** 2",
                            K="0.5 * P ** 2",
                        )

        # plot eigenfunctions
        for n in range(4):
            plt.plot(harmonic_osc.X, harmonic_osc.get_eigenstate(n), label=str(n))

        print("\n\nFirst energies for harmonic oscillator with omega = %f" % omega)
        print(harmonic_osc.energies[:20])

        plt.title("Eigenfunctions for harmonic oscillator with omega = %.2f (a.u.)" % omega)
        plt.xlabel('$x$ (a.u.)')
        plt.ylabel('wave functions ($\\psi_n(x)$)')
        plt.legend()
        plt.show()