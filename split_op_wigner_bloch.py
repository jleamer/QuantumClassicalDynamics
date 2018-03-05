from split_op_wigner_moyal import SplitOpWignerMoyal, np, ne


class SplitOpWignerBloch(SplitOpWignerMoyal):
    """
    The second-order split-operator propagator for
    finding the Wigner function of the Maxwell-Gibbs canonical state [rho = exp(-beta * H)]
    by split-operator propagation of the Bloch equation in phase space.

    Details about the method can be found at https://arxiv.org/abs/1602.07288

    The Hamiltonian should be of the form H = K(p) + V(x).

    This implementation follows split_op_wigner_moyal.py
    """
    def get_thermal_state(self, beta=None, nsteps=5000):
        """
        Calculate the thermal state via the Bloch propagator
        :param beta: inverse temperature (default beta = self.beta)
        :return: self.wignerfunction containing the thermal state
        """

        # get the inverse temperature increment
        beta = (beta if beta else self.beta)
        self.dbeta = beta / nsteps

        ###################################################################################
        #
        # Pre-calculate exponents used for the split operator propagation
        #
        ###################################################################################

        # Get the sum of the potential energy contributions
        expV = ne.evaluate(
            "-0.25 * dbeta *( ({V_minus}) + ({V_plus}) )".format(
                V_minus=self.V.format(X="(X - 0.5 * Theta)"),
                V_plus=self.V.format(X="(X + 0.5 * Theta)"),
            ),
            local_dict=vars(self)
        )

        # Make sure that the largest value is zero
        expV -= expV.max()

        # such that the following exponent is never larger then one
        np.exp(expV, out=expV)

        # Get the sum of the kinetic energy contributions
        expK = ne.evaluate(
            "-0.5 * dbeta *( ({K_plus}) + ({K_minus}) )".format(
                K_plus=self.K.format(P="(P + 0.5 * Lambda)"),
                K_minus=self.K.format(P="(P - 0.5 * Lambda)")
            ),
            local_dict=vars(self)
        )

        # Make sure that the largest value is zero
        expK -= expK.max()

        # such that the following exponent is never larger then one
        np.exp(expK, out=expK)

        # Initialize the Wigner function as the infinite temperature Gibbs state
        self.set_wignerfunction("1. + 0. * X + 0. * P")

        ###################################################################################
        #
        # Run second order Bloch propagator
        #
        ###################################################################################

        for _ in range(nsteps):
            # p x -> theta x
            self.wignerfunction = self.transform_p2theta(self.wignerfunction)
            self.wignerfunction *= expV

            # theta x  ->  p x
            self.wignerfunction = self.transform_theta2p(self.wignerfunction)

            # p x  ->  p lambda
            self.wignerfunction = self.transform_x2lambda(self.wignerfunction)
            self.wignerfunction *= expK

            # p lambda  ->  p x
            self.wignerfunction = self.transform_lambda2x(self.wignerfunction)

            # p x -> theta x
            self.wignerfunction = self.transform_p2theta(self.wignerfunction)
            self.wignerfunction *= expV

            # theta x  ->  p x
            self.wignerfunction = self.transform_theta2p(self.wignerfunction)

            # normalization
            self.wignerfunction /= self.wignerfunction.sum() * self.dXdP

        return self.wignerfunction

##############################################################################
#
#   Run some examples
#
##############################################################################

if __name__ == '__main__':

    print(SplitOpWignerBloch.__doc__)

    import matplotlib.pyplot as plt

    qsys_params = dict(
        t=0.,
        dt=0.05,

        X_gridDIM=256,
        X_amplitude=10.,

        P_gridDIM=256,
        P_amplitude=10.,

        beta=1. / 0.7,

        # kinetic energy part of the hamiltonian
        K="0.5 * {P} ** 2",

        # potential energy part of the hamiltonian
        V="0.5 * {X} ** 4",
    )

    print("Calculating the Gibbs state...")
    gibbs_state = SplitOpWignerBloch(**qsys_params).get_thermal_state()

    # Propagate this state via the Wigner-Moyal equation

    print("Check that the obtained Gibbs state is stationary under the Wigner-Moyal propagation...")
    propagator = SplitOpWignerMoyal(**qsys_params)
    final_state = propagator.set_wignerfunction(gibbs_state).propagate(3000)

    ##############################################################################
    #
    #   Plot the results
    #
    ##############################################################################

    from wigner_normalize import WignerSymLogNorm

    # save common plotting parameters
    plot_params = dict(
        origin='lower',
        extent=[propagator.X.min(), propagator.X.max(), propagator.P.min(), propagator.P.max()],
        cmap='seismic',
        # make a logarithmic color plot (see, e.g., http://matplotlib.org/users/colormapnorms.html)
        norm=WignerSymLogNorm(linthresh=1e-13, vmin=-0.01, vmax=0.1)
    )
    plt.subplot(121)

    plt.title("The Gibbs state (initial state)")
    plt.imshow(gibbs_state, **plot_params)
    plt.colorbar()
    plt.xlabel('$x$ (a.u.)')
    plt.ylabel('$p$ (a.u.)')

    plt.subplot(122)

    plt.title("The Gibbs state after propagation")
    plt.imshow(final_state, **plot_params)
    plt.colorbar()
    plt.xlabel('$x$ (a.u.)')
    plt.ylabel('$p$ (a.u.)')

    plt.show()