import numpy as np
from scipy import linalg, sparse
from types import MethodType, FunctionType


class DMM:
    """
    Implementation of the Density Matrix Minimization (DMM) Method
    """
    def __init__(self, **kwargs):
        """
        The following parameters must be specified
            H -- the Hamiltonian of the system
            beta (optional) -- the initial value of the inverse temperature
            dbeta -- the inverse temperature increment
            mu (optional) -- the chemical potential, default is zero
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
            self.H
        except AttributeError:
            raise AttributeError("The Hamiltonian (H) was not specified")

        try:
            self.dbeta
        except AttributeError:
            raise AttributeError("The inverse temperature increment (dbeta) was not specified")

        try:
            self.mu
        except AttributeError:
            print("Warning: Chemical potential was not specified, thus it is set to zero.")
            self.mu = 0

        try:
            self.beta
        except AttributeError:
            self.beta = 0.

        # save the identity matrix
        self.identity = sparse.identity(self.H.shape[0], dtype=self.H.dtype)

        # First step: symmetrized Hamiltonian
        self.H += self.H.conj().T
        self.H *= 0.5

        # find eigenvalues of the Hamiltonian for comparision with the exact expression self.get_exact_pop
        self.E = linalg.eigvalsh(
            # Convert a sparse Hamiltonian to a dense matrix
            self.H.toarray() if sparse.issparse(self.H) else self.H
        )

        # Set the initial condition for the matrix
        self.rho = 0.5 * self.identity.toarray()
        
        # Set a copy of the initial rho matrix
        self.rhocopy = self.rho.copy()

    def propagate_beta1(self, nsteps=1, commnorm=False):
        """
        The first order propagation in the inverse temperature
        :param nsteps: number of steps in the inverse temperature to take
        :param commnorm: boolean flag to print the norm of the commutator between the density matrix and hamiltonian
        :return: self
        """
        # Save the scaled Hamiltonian for propagation
        scaledH = self.H - self.mu * self.identity
        scaledH *= -0.5 * self.dbeta
        
        # Reset the copy of rho
        self.rhocopy = self.rho.copy()

        for _ in range(nsteps):
            ###########################################################################
            #
            #   The method is
            #
            #       rho(beta + dbeta) = K rho(beta) K^{\dagger}
            #       rho(beta = 0) = 1/2
            #
            #   where
            #
            #       K = 1 - 0.5 * dbeta * (H - mu) * (1 - rho(beta))
            #
            ###########################################################################

            # Construct the map K
            # Optimized version of
            #   K = self.identity - 0.5 * self.dbeta * self.H.dot(self.identity - self.rho)

            K = scaledH.dot(self.identity - self.rhocopy)
            K += self.identity

            # assert not sparse.issparse(K), "K matrix must not be sparse"
            self.rhocopy = K.dot(self.rhocopy).dot(K.conj().T)

            self.beta += self.dbeta

        if commnorm:
            # Convert a sparce Hamiltonian to a dense matrix
            H = (self.H.toarray() if sparse.issparse(self.H) else self.H)

            print(
                "\nThe norm of the commutator of the obtained density matrix and the Hamiltonian: %.2e\n\n"
                % linalg.norm(self.rho.dot(H) - H.dot(self.rho))
            )

        return self

    def propagate_beta2(self, nsteps=1, commnorm=False):
        """
        The second order propagation in the inverse temperature
        :param nsteps: number of steps in the inverse temperature to take
        :param commnorm: boolean flag to print the norm of the commutator between the density matrix and hamiltonian
        :return: self
        """
        # Save the scaled Hamiltonian for propagation
        scaledH = self.H - self.mu * self.identity
        scaledH *= -0.5 * self.dbeta
        
        # Reset the copy of rho
        self.rhocopy = self.rho.copy()

        for _ in range(nsteps):
            ###########################################################################
            #
            #   The method is
            #
            #       rho(beta + dbeta) = K rho(beta) K^{\dagger} + A rho(beta) A^{\dagger}
            #       rho(beta = 0) = 1/2
            #
            #   where
            #       A = -0.5 * dbeta * (H - mu) * (1 - rho(beta))
            #       K = 1 + A * (1 + A + 0.5 * dbeta * (H - mu))
            #
            ###########################################################################

            # Construct A and K
            A = scaledH.dot(self.identity - self.rhocopy)

            K = A.dot(self.identity + A - scaledH)
            K += self.identity

            self.rhocopy = K.dot(self.rhocopy).dot(K.conj().T) + A.dot(self.rhocopy).dot(A.conj().T)

            self.beta += self.dbeta

        if commnorm:
            # Convert a sparce Hamiltonian to a dense matrix
            H = (self.H.toarray() if sparse.issparse(self.H) else self.H)

            print(
                "\nThe norm of the commutator of the obtained density matrix and the Hamiltonian: %.2e\n\n"
                % linalg.norm(self.rho.dot(H) - H.dot(self.rho))
            )

        return self
    
    def propagate_beta3(self, nsteps=1):
        """
        The second order propagation in the inverse temperature
        :param nsteps: number of steps in the inverse temperature to take
        :param commnorm: boolean flag to print the norm of the commutator between the density matrix and hamiltonian
        :return: self
        """
        # Save the scaled Hamiltonian for propagation
        scaledH = self.H - self.mu * self.identity
        scaledH *= -0.5 * self.dbeta

        for _ in range(nsteps):
            ###########################################################################
            #
            #   The method is
            #
            #       rho(beta + dbeta) = rho(beta) + dbeta * 
            #       rho(beta = 0) = 1/2
            #
            #   where
            #       B = -0.5 * dbeta * (H - mu) * (1 - rho(beta))
            #       A = 1 + B * (1 + B + 0.5 * dbeta * (H - mu))
            #       K = 1 + A * (1 + A + 0.5 * dbeta * (H - mu))
            #
            ###########################################################################
            return 1
        return 1
        
    def propagate_mu1(self, nsteps=1):
        """
        The first order propagate along the chemical potential
        :param nsteps: number of steps to be taken in the chemical potential   
        :return: self
        """
        # Reset copy of rho
        self.rhocopy = self.rho.copy()
        
        for _ in range(nsteps):
            ###########################################################################
            #
            #   The method is
            #
            #       rho(mu + dmu) = K rho(mu) K^{\dagger}
            #
            #   where
            #
            #       K = 1 + 0.5 * dmu * beta * (1 - rho(mu))
            #
            ###########################################################################

            # Construct K
            K = self.identity - self.rhocopy
            K *= 0.5 * self.dmu * self.beta
            K += self.identity

            self.rhocopy = K.dot(self.rhocopy).dot(K.conj().T)

            self.mu += self.dmu

        return self

    def propagate_mu2(self, nsteps=1):
        """
        The second order propagate along the chemical potential
        :param nsteps: number of steps to be taken in the chemical potential
        :return: self
        """
        # Reset the copy of rho
        self.rhocopy = self.rho.copy()
        
        for _ in range(nsteps):
            ###########################################################################
            #
            #   The method is
            #
            #       rho(mu + dmu) = K rho(mu) K^{\dagger} + A rho(mu) A^{\dagger}
            #
            #   where
            #
            #       A = 0.5 * dmu * beta * (1 - rho(mu))
            #
            #       K = 1 + A * (1 + A - 0.5 * dmu * beta)
            #
            ###########################################################################

            # Construct K
            A = self.identity - self.rhocopy
            A *= 0.5 * self.dmu * self.beta

            K = A.dot(self.identity + A - 0.5 * self.dmu * self.beta)
            K += self.identity

            self.rhocopy = K.dot(self.rhocopy).dot(K.conj().T) + A.dot(self.rhocopy).dot(A.conj().T)

            self.mu += self.dmu

        return self

    def get_exact_DF(self):
        """
        :return: The exact Fermi-Dirac density matrix 
        """
        # Convert a sparce Hamiltonian to a dense matrix
        H = (self.H.toarray() if sparse.issparse(self.H) else self.H.copy())
        H -= self.mu * self.identity
        H *= self.beta

        self.rho_exact = linalg.inv(self.identity + linalg.expm(H))

        return self.rho_exact

    def get_exact_pop(self):
        """
        :return: The exact Fermi-Dirac population distributions
        """
        return 1. / (1. + np.exp(self.beta * (self.E - self.mu)))

    def get_average_H(self):
        """
        :return: Tr(H \rho) / Tr(\rho)
        """
        #########################################################
        #
        #   The current implementation is not the most optimal.
        #   A better method to use is
        #
        #       Tr(AB) = A_{ij} B_{ji} = A_{ij} (B^T)_{ij}
        #
        #########################################################
        #return np.sum(self.H * self.rho.T) / self.rho.trace().sum()
        return self.H.dot(self.rho).trace().sum() / self.rho.trace().sum()

    def get_exact_average_H(self):
        """
        :return: Exact expression Tr(H \rho)  
        """
        p = self.get_exact_pop()
        return np.sum(p * self.E) / p.sum()
        
    def deriv(self, K, rhotemp):
        f = (K.dot(rhotemp) + rhotemp.dot(K.conj().T)) - rhotemp*(K.dot(rhotemp) + rhotemp.dot(K.conj().T)).trace().sum()
        return f
    
    def rk1(self, deriv, nsteps=1):
        '''
        '''
        # Save the scaled Hamiltonian for propagation
        scaledH = self.H - self.mu * self.identity
        scaledH *= -0.5 * self.dbeta
        
        # Reset the copy of rho
        self.rhocopy = self.rho.copy()
        
        for i in range(nsteps):
            ######################################################################
            #
            #   The method is 
            #       
            #       rho(beta + dbeta) = rho(beta) + dbeta*(k1)
            #   
            #   where
            #        f(rho) = K * rho + rho * K^{\dagger}
            #        K = -0.5*(H-mu)*(I-rho)
            #    -   k1 = f(rho)
            #        k2 = f(rho + (2/3)*dbeta*k1)
            #
            ######################################################################
            
            K = scaledH.dot(self.identity - self.rhocopy)
            
            rhotemp = self.rhocopy
            k1 = deriv(K, rhotemp)
                        
            self.rhocopy = self.rhocopy + self.dbeta*(k1)
            self.beta += self.dbeta
        
        return self
    
    def rk2(self, deriv, nsteps=1):
        '''
        '''
        # Save the scaled Hamiltonian for propagation
        scaledH = self.H - self.mu * self.identity
        scaledH *= -0.5 * self.dbeta
        
        # Reset the copy of rho
        self.rhocopy = self.rho.copy()
        
        for i in range(nsteps):
            ######################################################################
            #
            #   The method is 
            #       
            #       rho(beta + dbeta) = rho(beta) + (1/4)*dbeta*(k1 + 3k2)
            #   
            #   where
            #        f(rho) = K * rho * K^{\dagger}
            #        K = I-0.5*(I-rho)(H-mu)
            #    -   k1 = f(rho)
            #        k2 = f(rho + (2/3)*dbeta*k1)
            #
            ######################################################################
            
            K = scaledH.dot(self.identity - self.rhocopy)
            K += self.identity
            
            rhotemp = self.rhocopy
            k1 = deriv(K, rhotemp)
            
            rhotemp = self.rhocopy + (2/3.)*self.dbeta*k1
            k2 = deriv(K, rhotemp)
                        
            self.rhocopy = self.rhocopy + 0.25*self.dbeta*(k1 + 3*k2)
            self.beta += self.dbeta
        
        return self
        
    def rk4(self, deriv, nsteps=1):
        """
        Inputs:
            nsteps - the number of steps to take
        Returns:
            self
        """
        # Save the scaled Hamiltonian for propagation
        scaledH = self.H - self.mu * self.identity
        scaledH *= -0.5 * self.dbeta
        
        # Reset the copy of rho
        self.rhocopy = self.rho.copy()
        
        for i in range(nsteps):
            ######################################################################
            #
            #   The method is 
            #       
            #       rho(beta + dbeta) = rho(beta) + (1/6)*dbeta*(k1 + 2k2 + 2k3 + k4)
            #   
            #   where
            #        f(rho) = K * rho * K^{\dagger}
            #        K = I-0.5*(I-rho)(H-mu)
            #    -   k1 = f(rho)
            #        k2 = f(rho + 0.5*dbeta*k1)
            #        k3 = f(rho + 0.5*dbeta*k2)
            #        k4 = f(rho + dbeta*k3)
            #
            ######################################################################
        
            K = scaledH.dot(self.identity - self.rhocopy)
            
            rhotemp = self.rhocopy
            k1 = deriv(K, rhotemp)
            
            rhotemp = self.rhocopy + 0.5*self.dbeta*k1
            k2 = deriv(K, rhotemp)
            
            rhotemp = self.rhocopy + 0.5*self.dbeta*k2
            k3 = deriv(K, rhotemp)
            
            rhotemp = self.rhocopy + self.dbeta*k3
            k4 = deriv(K, rhotemp)
            
            self.rhocopy = self.rhocopy + 1/6.*self.dbeta*(k1 + 2*k2 + 2*k3 + k4)
            self.beta += self.dbeta
       
        return self

##############################################################################
#
#   Run some examples
#
##############################################################################

if __name__ == '__main__':

    import matplotlib.pyplot as plt

    np.random.seed(4936601)

    dmm = DMM(
        dbeta=0.003,
        dmu=0.0005,
        mu=-0.9,
        # randomly initialize Hamiltonian (symmetrization will take place in the constructor)
        #H=sparse.random(70, 70, density=0.1),
        H=np.random.normal(size=(40,40)) + 1j * np.random.normal(size=(40,40)),
        #np.random.rand(40, 40) + 1j * np.random.rand(40, 40)
    )
    
    # First propagate along the inverse temperature
    dmm.propagate_beta1(3000)

    plt.subplot(121)
    plt.title("Population at $\\beta = %.2f$, $\mu = %.2f$" % (dmm.beta, dmm.mu))

    p_dmm = linalg.eigvalsh(dmm.rhocopy)[::-1]

    plt.plot(dmm.E, p_dmm, '*-', label="DMM")
    plt.plot(dmm.E, linalg.eigvalsh(dmm.get_exact_DF())[::-1], '*-', label='exact via expm')
    plt.plot(dmm.E, dmm.get_exact_pop(), '*-', label='population')

    print("Error norm between obtained and exact diagonalization: %.2e" % np.linalg.norm(p_dmm - dmm.get_exact_pop()))

    plt.legend()
    plt.xlabel('Energy')
    plt.ylabel('Population')

    # Second propagate along the chemical potential
    dmm.propagate_mu1(3000)

    plt.subplot(122)
    plt.title("Population at $\\beta = %.2f$, $\mu = %.2f$" % (dmm.beta, dmm.mu))

    p_dmm = linalg.eigvalsh(dmm.rhocopy)[::-1]

    plt.plot(dmm.E, p_dmm, '*-', label="DMM")
    plt.plot(dmm.E, linalg.eigvalsh(dmm.get_exact_DF())[::-1], '*-', label='exact via expm')
    plt.plot(dmm.E, dmm.get_exact_pop(), '*-', label='population')

    print("Error norm between obtained and exact diagonalization: %.2e" % np.linalg.norm(p_dmm - dmm.get_exact_pop()))

    plt.legend()
    plt.xlabel('Energy')
    plt.ylabel('Population')
    
    # It appears that the RK4 method is unstable here. Values for n > 1250
    # cause the values in rho to become infinite
    
    # Third propagate along the inverse temperature using RK4
    dmm.beta = 0
    dmm.mu = -0.9
    dmm.rk4(dmm.deriv, 3000)
    print("Error norm between obtained and exact diagonalization: %.2e" % np.linalg.norm(p_dmm - dmm.get_exact_pop()))
    
    fig2 = plt.figure(2)
    fig2.clf()
    ax2 = fig2.add_subplot(111)
    ax2.set_title("Population at $\\beta = %.2f$, $\mu = %.2f$" % (dmm.beta, dmm.mu))
    
    p_dmm = linalg.eigvalsh(dmm.rhocopy)[::-1]
    
    ax2.plot(dmm.E, p_dmm, '*-', label="DMM")
    ax2.plot(dmm.E, linalg.eigvalsh(dmm.get_exact_DF())[::-1], '*-', label = 'exact via expm')
    ax2.plot(dmm.E, dmm.get_exact_pop(), '*-', label = 'population')
    ax2.set_xlabel("Energy")
    ax2.set_ylabel("Population")
    ax2.legend(numpoints=1)
    plt.show()
    