import numpy as np
from scipy.sparse import diags # Construct a sparse matrix from diagonals
#from scipy.sparse import linalg # Linear algebra for sparse matrix
from scipy import linalg # Linear algebra for dense matrix
import matplotlib.pyplot as plt # plotting facility


class CentralDiffQHamiltonian:
    """
    Generate quantum Hamiltonian for 1D system in the coordinate representation
    using the central difference approximation.
    """
    def __init__(self, **kwards):
        """
        The following parameters must be specified
            X_gridDIM - specifying the grid size
            X_amplitude - maximum value of the coordinates
            V(x) - potential energy (as a function)
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

        # generate coordinate range
        self.X_range = np.linspace(-self.X_amplitude, self.X_amplitude, self.X_gridDIM)

        # save the coordinate step size
        self.dX = self.X_range[1] - self.X_range[0]

        # Construct the kinetic energy part as sparse matrix from diagonal
        self.Hamiltonian = diags([1., -2., 1.], [-1, 0, 1], shape=(self.X_gridDIM, self.X_gridDIM))
        self.Hamiltonian *= -0.5/(self.dX**2)

        # Add diagonal potential energy
        self.Hamiltonian = self.Hamiltonian + diags(self.V(self.X_range), 0)

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
            self.energies, self.eigenstates = linalg.eigh(self.Hamiltonian.toarray())

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

    print("Central difference Hamiltonian")

    for omega in [4., 8.]:
        # Find energies of a harmonic oscillator V = 0.5*(omega*x)**2
        harmonic_osc = CentralDiffQHamiltonian(
                            X_gridDIM=512,
                            X_amplitude=5.,
                            V=lambda x: 0.5*(omega*x)**2
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