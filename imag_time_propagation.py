from split_op_schrodinger1D import SplitOpSchrodinger1D, fftpack, np, ne, linalg

# We will use the inheritance in the object orienting programing (see, e.g.,
# https://docs.python.org/2/tutorial/classes.html) to add methods to already developed propagator (SplitOpSchrodinger1D)
# that find stationary states via the imaginary-time propagation


class ImgTimePropagation(SplitOpSchrodinger1D):

    def get_stationary_states(self, nstates, nsteps=10000):
        """
        Obtain stationary states via the imaginary time propagation
        :param nstates: number of states to obtaine.
                If nstates = 1, only the ground state is obtained. If nstates = 2,
                the ground and first exited states are obtained, etc
        :param nsteps: number of the imaginary time steps to take
        :return:self
        """
        # since there is no time dependence (self.t) during the imaginary time propagation
        # pre-calculate imaginary time exponents of the potential and kinetic energy
        img_expV = ne.evaluate("(-1) ** k * exp(-0.5 * dt * (%s))" % self.V, local_dict=self.__dict__)
        img_expK = ne.evaluate("exp(-dt * (%s))" % self.K, local_dict=self.__dict__)

        # initialize the list where the stationary states will be saved
        self.stationary_states = []

        # boolean flag determining the parity of wavefunction
        even = True

        for n in xrange(nstates):

            # initialize the wavefunction depending on the parity
            if even:
                self.set_wavefunction("exp(-X ** 2)")
            else:
                self.set_wavefunction("X * exp(-X ** 2)")
            even = not even

            # get an alias pointer to self.wavefunction
            wavefunction = self.wavefunction

            for _ in xrange(nsteps):
                #################################################################################
                #
                #   Make an imaginary time step
                #
                #################################################################################
                wavefunction *= img_expV

                # going to the momentum representation
                wavefunction = fftpack.fft(wavefunction, overwrite_x=True)
                wavefunction *= img_expK

                # going back to the coordinate representation
                wavefunction = fftpack.ifft(wavefunction, overwrite_x=True)
                wavefunction *= img_expV

                #################################################################################
                #
                #    Project out all previously calculated stationary states
                #
                #################################################################################

                # normalize
                wavefunction /= linalg.norm(wavefunction)

                # calculate the projections
                projs = [np.vdot(psi, wavefunction) for psi in self.stationary_states]

                # project out the stationary states
                for psi, proj in zip(self.stationary_states, projs):
                    ne.evaluate("wavefunction - proj * psi", out=wavefunction)

                # normalize
                wavefunction /= linalg.norm(wavefunction)

            # save obtained approximation to the stationary state
            self.stationary_states.append(wavefunction.copy())

        return self

##############################################################################
#
#   Run some examples
#
##############################################################################

if __name__ == '__main__':

    # load tools for creating animation
    import sys

    if sys.platform == 'darwin':
        # only for MacOS
        import matplotlib

        matplotlib.use('TKAgg')

    import matplotlib.pyplot as plt

    # specify parameters separately
    atom_params = dict(
        X_gridDIM=1024,
        X_amplitude=10.,
        dt=0.05,
        K="0.5 * P ** 2",
        V="-1. / sqrt(X ** 2 + 1.37)",
    )

    # construct the propagator
    atom_sys = ImgTimePropagation(**atom_params)

    # find the ground and first excited states via the imaginary time method
    atom_sys.get_stationary_states(4)

    # get "exact" eigenstates by diagonalizing the MUB hamiltonian
    from mub_qhamiltonian import MUBQHamiltonian
    atom_mub = MUBQHamiltonian(**atom_params)

    plt.subplot(221)
    plt.title("Ground state calculation for argon within the single active electron approximation")

    # set the ground state (obtained via the imaginary time propagation) as the initial condition
    atom_sys.set_wavefunction(atom_sys.stationary_states[0])

    plt.semilogy(atom_sys.X, atom_sys.wavefunction, 'r-', label='state via img-time')
    plt.semilogy(atom_sys.X, atom_sys.propagate(10000), 'b--', label='state after propagation')
    plt.semilogy(atom_sys.X, atom_mub.get_eigenstate(0), 'g-.', label='state via MUB')
    plt.xlabel("$x$ (a.u.)")
    plt.legend(loc='lower center')

    plt.subplot(222)
    plt.title("First exited state calculation of argon")

    # set the first excited state (obtained via the imaginary time propagation) as the initial condition
    atom_sys.set_wavefunction(atom_sys.stationary_states[1])

    plt.semilogy(atom_sys.X, np.abs(atom_sys.wavefunction), 'r-', label='state via img-time')
    plt.semilogy(atom_sys.X, np.abs(atom_sys.propagate(10000)), 'b--', label='state after propagation')
    plt.semilogy(atom_sys.X, np.abs(atom_mub.get_eigenstate(1)), 'g-.', label='state via MUB')
    plt.ylim([1e-6, 1e0])
    plt.xlabel("$x$ (a.u.)")
    plt.legend(loc='lower center')

    plt.subplot(223)
    plt.title("Second exited state calculation of argon")

    # set the second excited state (obtained via the imaginary time propagation) as the initial condition
    atom_sys.set_wavefunction(atom_sys.stationary_states[2])

    plt.semilogy(atom_sys.X, np.abs(atom_sys.wavefunction), 'r-', label='state via img-time')
    plt.semilogy(atom_sys.X, np.abs(atom_sys.propagate(10000)), 'b--', label='state after propagation')
    plt.semilogy(atom_sys.X, np.abs(atom_mub.get_eigenstate(2)), 'g-.', label='state via MUB')
    plt.ylim([1e-6, 1e0])
    plt.xlabel("$x$ (a.u.)")
    plt.legend(loc='lower center')

    plt.subplot(224)
    plt.title("Third exited state calculation of argon")

    # set the third excited state (obtained via the imaginary time propagation) as the initial condition
    atom_sys.set_wavefunction(atom_sys.stationary_states[3])

    plt.semilogy(atom_sys.X, np.abs(atom_sys.wavefunction), 'r-', label='state via img-time')
    plt.semilogy(atom_sys.X, np.abs(atom_sys.propagate(10000)), 'b--', label='state after propagation')
    plt.semilogy(atom_sys.X, np.abs(atom_mub.get_eigenstate(3)), 'g-.', label='state via MUB')
    plt.ylim([1e-6, 1e0])
    plt.xlabel("$x$ (a.u.)")
    plt.legend(loc='lower center')

    plt.show()