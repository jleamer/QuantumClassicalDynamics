from split_op_denisty_matrix import DensityMatrix, np, ne


class BlochPropagation(DensityMatrix):
    """
    Find the thermal state using the Bloch method, i.e., to propagate equation
        d \\rho / d\\beta = -1/2 (\\rho H + H \\rho)
        \\rho(0) = 1
    """
    def get_thermal_state(self, dbeta=None, nsteps=2000):
        """
        Calculate the thermal state via the Bloch propagator
        :param dbeta: inverse temperature increment (default value is dbeta == dt)
        :return: self.rho containing the thermal state
        """

        # set the initial condition to the identity matrix
        self.set_rho("(X == X_prime)")

        # set the step size
        self.dbeta = (dbeta if dbeta else self.dt)

        # string with the total kinetic energy
        KK = "(%s) + (%s)" % (self.K.format(P="P"), self.K.format(P="P_prime"))

        # save max value of the kinetic energy
        self.Kmin = ne.evaluate("min(%s)" % KK, local_dict=vars(self))

        # string with the total potential energy
        VV = "(%s) + (%s)" % (self.V.format(X="X"), self.V.format(X="X_prime"))

        # save max value of the potential energy
        self.Vmin = ne.evaluate("min(%s)" % VV, local_dict=vars(self))

        # calculate the exponent for the bloch propagation
        bloch_expV = ne.evaluate(
            "(-1) ** (k + k_prime) * exp(-0.25 * dbeta * ((%s) - Vmin))" % VV,
            local_dict=vars(self)
        )
        bloch_expK = ne.evaluate("exp(-0.5 * dbeta * ((%s) - Kmin))" % KK, local_dict=vars(self))

        # propagate in the imaginary time
        for step in range(nsteps):
            self.rho *= bloch_expV

            # going to the momentum representation
            self.rho_fft_ax0()
            self.rho_ifft_ax1()

            self.rho *= bloch_expK

            # going back to the coordinate representation
            self.rho_ifft_ax0()
            self.rho_fft_ax1()

            self.rho *= bloch_expV

            # normalize
            self.rho /= self.rho.trace() * self.dX

            if step % 50 == 0:
                print("purity (%1.6f); energy (%1.4e)" % (self.get_purity().real, self.get_energy()))

        return self.rho

##############################################################################
#
#   Run some examples
#
##############################################################################

if __name__ == '__main__':

    import matplotlib.pyplot as plt

    qsys = BlochPropagation(
        t=0.,
        dt=0.005,
        X_gridDIM=256,
        X_amplitude=5.,

        # kinetic energy part of the hamiltonian
        K="0.5 * {P} ** 2",

        # these functions are used for evaluating the Ehrenfest theorems
        diff_K="{P}",

        # potential energy part of the hamiltonian
        omega=np.random.uniform(2., 3.),

        V="0.5 * omega ** 2 * {X} ** 2",

        # these functions are used for evaluating the Ehrenfest theorems
        diff_V="omega ** 2 * {X}",
    )

    # get the ground state
    init_state = qsys.get_thermal_state().copy()

    print("\nexact energy ", 0.5 * qsys.omega)

    # propagate the state and check that it is a stationary state
    qsys.propagate(3000)

    print("\n||initial state - final state|| = ", np.linalg.norm(init_state - qsys.rho))

    #################################################################
    #
    # Plot the Ehrenfest theorems after the animation is over
    #
    #################################################################
    quant_sys = qsys

    # generate time step grid
    dt = quant_sys.dt
    times = np.arange(dt, dt + dt*len(quant_sys.X_average), dt)

    plt.subplot(121)
    plt.title("The first Ehrenfest theorem verification")

    plt.plot(times, np.gradient(quant_sys.X_average, dt), 'r-', label='$d\\langle \\hat{x} \\rangle/dt$')
    plt.plot(times, quant_sys.X_average_RHS, 'b--', label='RHS')

    plt.legend()
    plt.xlabel('time $t$ (a.u.)')

    plt.subplot(122)
    plt.title("The second Ehrenfest theorem verification")

    plt.plot(times, np.gradient(quant_sys.P_average, dt), 'r-', label='$d\\langle \\hat{p} \\rangle/dt$')
    plt.plot(times, quant_sys.P_average_RHS, 'b--', label='RHS')

    plt.legend()
    plt.xlabel('time $t$ (a.u.)')

    plt.show()

    plt.title("The expectation value of the hamiltonian")

    #################################################################
    #
    # Analyze how well the energy was preserved
    #
    #################################################################

    h = np.array(quant_sys.hamiltonian_average)
    print(
        "\nHamiltonian is preserved within the accuracy of %.2e percent" % (100. * (1. - h.min() / h.max()))
    )

    plt.plot(times, h)
    plt.ylabel('energy')
    plt.xlabel('time $t$ (a.u.)')

    plt.show()
