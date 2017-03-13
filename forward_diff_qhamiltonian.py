import numpy as np
from scipy.sparse import diags # Construct a sparse matrix from diagonals
# from scipy.sparse import linalg # Linear algebra for sparse matrix
from scipy import linalg # Linear algebra for dense matrix
from types import MethodType, FunctionType


class ForwardDiffQHamiltonian:
    """
    Construct the quantum Hamiltonian for an 1D system in the coordinate representation
    using the forward difference approximation.
    """
    def __init__(self, **kwargs):
        """
        The following parameters must be specified
            X_gridDIM - the grid size
            X_amplitude - the maximum value of the coordinates
            V(x) - a potential energy (as a function)
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
        self.Hamiltonian = diags([1., -2., 1.], [0, 1, 2], shape=(self.X_gridDIM, self.X_gridDIM))
        self.Hamiltonian *= -0.5/(self.dX**2)

        # Add diagonal potential energy
        self.Hamiltonian = self.Hamiltonian + diags(self.V(self.X_range), 0)

##############################################################################
#
#   Run some examples
#
##############################################################################

if __name__ == '__main__':

    print(ForwardDiffQHamiltonian.__doc__)

    for omega in [4., 8.]:
        # Find energies of a harmonic oscillator V = 0.5*(omega*x)**2
        harmonic_osc = ForwardDiffQHamiltonian(
                            X_gridDIM=512,
                            X_amplitude=5.,
                            omega=omega,
                            V=lambda self, x: 0.5 * (self.omega * x) ** 2
                        )
        # get energies
        energies = linalg.eigvals(harmonic_osc.Hamiltonian.toarray())

        # sort energies by real part
        energies = energies[np.argsort(energies.real)]

        print("\n\nFirst energies for harmonic oscillator with omega = %f" % omega)
        print energies[:20]