import numpy as np
import numexpr as ne
from scipy.sparse import diags # Construct a sparse matrix from diagonals
from scipy.sparse import linalg # Linear algebra for sparse matrix
from types import MethodType, FunctionType


class CentralDiffQHamiltonian:
    """
    Generate quantum Hamiltonian for 1D system in the coordinate representation
    using the central difference approximation.
    """
    def __init__(self, **kwargs):
        """
         The following parameters must be specified
             X_gridDIM - the grid size
             X_amplitude - the maximum value of the coordinates
             V - a potential energy (as a string to be evaluated by numexpr)
         """

        # save all attributes
        for name, value in kwargs.items():
            # if the value supplied is a function, then dynamically assign it as a method;
            if isinstance(value, FunctionType):
                setattr(self, name, MethodType(value, self))
            # otherwise bind it as a property
            else:
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

        # get coordinate step size
        self.dX = 2. * self.X_amplitude / self.X_gridDIM

        # generate coordinate range
        k = np.arange(self.X_gridDIM)
        self.X = (k - self.X_gridDIM / 2) * self.dX
        # Note that self.X is generate to be compatible with mub_qhamiltonian.py

        # Construct the kinetic energy part as sparse matrix from diagonal
        self.Hamiltonian = diags([1., -2., 1.], [-1, 0, 1], shape=(self.X_gridDIM, self.X_gridDIM))
        self.Hamiltonian *= -0.5/(self.dX**2)

        # Add diagonal potential energy
        V = ne.evaluate(self.V, local_dict=vars(self))
        self.Hamiltonian += diags(V, 0)

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
            # using specialized function for sparse Hermitian matrices
            self.energies, self.eigenstates = linalg.eigsh(self.Hamiltonian, which='SM', k=20)

            # transpose for convenience
            self.eigenstates = self.eigenstates.T

            # normalize each eigenvector
            for psi in self.eigenstates:
                psi /= np.linalg.norm(psi) * np.sqrt(self.dX)

            # Make sure that the ground state is non negative
            np.abs(self.eigenstates[0], out=self.eigenstates[0])

        return self

##############################################################################
#
#   Run some examples
#
##############################################################################

if __name__ == '__main__':

    # plotting facility
    import matplotlib.pyplot as plt

    print(CentralDiffQHamiltonian.__doc__)

    for omega in [4., 8.]:
        # Find energies of a harmonic oscillator V = 0.5*(omega*x)**2
        harmonic_osc = CentralDiffQHamiltonian(
                            X_gridDIM=512,
                            X_amplitude=5.,
                            omega=omega,
                            V="0.5 * (omega * X) ** 2",
                        )

        # plot eigenfunctions
        for n in range(4):
            plt.plot(harmonic_osc.X, harmonic_osc.get_eigenstate(n), label=str(n))

        print("\n\nFirst energies for harmonic oscillator with omega = {}".format(omega))

        # set precision for printing arrays
        np.set_printoptions(precision=4)
        print(harmonic_osc.energies)

        plt.title("Eigenfunctions for harmonic oscillator with omega = {} (a.u.)".format(omega))
        plt.xlabel('$x$ (a.u.)')
        plt.ylabel('wave functions ($\\psi_n(x)$)')
        plt.legend()
        plt.show()