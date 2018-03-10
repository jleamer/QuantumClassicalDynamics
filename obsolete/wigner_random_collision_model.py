import numpy as np

from split_op_wigner_moyal import SplitOpWignerMoyal
from split_op_wigner_bloch import SplitOpWignerBloch


class WignerRandomCollisionModel(SplitOpWignerMoyal):
    """
    The Wigner representation of the random collision model of open system dynamics.

    The Wigner function obeys the following equation

        dW / dt = {{H, W}} - gamma (W_0 - W)

    where {{ , }} is the Moyal bracket, W_0 is a Gibbs-Boltzmann state, and
    gamma characterize the collision rate (i.e., 1/gamma is dephasing time).
    """
    def __init__(self, **kwargs):

        # Initialize the parent class
        SplitOpWignerMoyal.__init__(self, **kwargs)

        # Make sure gamma was specified
        try:
            self.gamma
        except AttributeError:
            raise AttributeError("Collision rate (gamma) was not specified")

        # Calculate the Gibbs state (W_0)
        self.gibbs_state = SplitOpWignerBloch(**kwargs).get_gibbs_state()

        # Pre-calculate the constants needed for the dissipator propagation
        self.w_coeff = np.exp(-self.gamma * self.dt)
        self.w0_coeff = 1 - self.w_coeff

    def single_step_propagation(self):
        """
        Overload the method SplitOpWignerMoyal.single_step_propagation.

        Perform single step propagation. The final Wigner function is not normalized.
        :return: self.wignerfunction
        """
        # First follow the unitary evolution
        SplitOpWignerMoyal.single_step_propagation(self)

        # Dissipation requites to updated rho as
        #   W = W0 * (1 - exp(-gamma*dt)) + exp(-gamma*dt) * W,
        self.wignerfunction *= self.w_coeff
        self.wignerfunction += self.w0_coeff * self.gibbs_state

        return self.wignerfunction

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
    from matplotlib.animation import FuncAnimation

    print(WignerRandomCollisionModel.__doc__)

    class VisualizeDynamicsPhaseSpace:
        """
        Class to visualize the Wigner function function dynamics in phase space.
        """
        def __init__(self, fig):
            """
            Initialize all propagators and frame
            :param fig: matplotlib figure object
            """
            #  Initialize systems
            self.set_quantum_sys()

            #################################################################
            #
            # Initialize plotting facility
            #
            #################################################################

            self.fig = fig

            # import utility to visualize the wigner function
            from wigner_normalize import WignerNormalize #WignerSymLogNorm

            # imshow parameters
            plot_params = dict(
                extent=[self.quant_sys.X.min(), self.quant_sys.X.max(), self.quant_sys.P.min(), self.quant_sys.P.max()],
                origin='lower',
                cmap='seismic',
                interpolation='nearest',
                norm=WignerNormalize(vmin=-0.005, vmax=0.1)
            )

            ax = fig.add_subplot(121)
            ax.set_title('Gibbs state Wigner function ($W_0(x,p)$)\nwith $kT = $ %.2f (a.u.)' % self.quant_sys.kT)

            self.gibbs_state_img = ax.imshow(self.quant_sys.gibbs_state, **plot_params)
            self.fig.colorbar(self.gibbs_state_img)
            ax.set_xlabel('$x$ (a.u.)')
            ax.set_ylabel('$p$ (a.u.)')

            ax = fig.add_subplot(122)
            ax.set_title(
                'Wigner function evolution $W(x,p,t)$\nwith $\\gamma^{-1} = $ %.2f (a.u.)'
                % (1./self.quant_sys.gamma)
            )

            self.img = ax.imshow([[]], **plot_params)
            self.fig.colorbar(self.img)
            ax.set_xlabel('$x$ (a.u.)')
            ax.set_ylabel('$p$ (a.u.)')

        def set_quantum_sys(self):
            """
            Initialize quantum propagator
            :param self:
            :return:
            """
            self.quant_sys = WignerRandomCollisionModel(
                t=0.,
                dt=0.005,

                X_gridDIM=256,
                X_amplitude=10.,

                P_gridDIM=256,
                P_amplitude=16.,

                # kinetic energy part of the hamiltonian
                K=lambda p: 0.5*p**2,

                # potential energy part of the hamiltonian
                V=lambda x: 0.08*x**4,

                # pick a random temperature
                kT = np.random.uniform(0.01, 1.),

                # collision rate
                gamma = np.random.uniform(0.1, 1.),
            )

            # parameter controling the width of the wigner function
            sigma = np.random.uniform(1., 3.)

            # set randomised initial condition
            self.quant_sys.set_wignerfunction(
                np.exp(
                    # randomized position
                    -sigma*(self.quant_sys.X + np.random.uniform(-4., 4.))**2
                    # randomized initial velocity
                    -(1./sigma)*(self.quant_sys.P + np.random.uniform(-4., 4.))**2
                )
            )

        def empty_frame(self):
            """
            Make empty frame and reinitialize quantum system
            :param self:
            :return: image object
            """
            self.set_quantum_sys()
            self.gibbs_state_img.set_array(self.quant_sys.gibbs_state)
            self.img.set_array([[]])
            return self.img,

        def __call__(self, frame_num):
            """
            Draw a new frame
            :param frame_num: current frame number
            :return: image objects
            """

            # the wigner function
            self.img.set_array(self.quant_sys.propagate(20))
            return self.img,

    fig = plt.gcf()
    visualizer = VisualizeDynamicsPhaseSpace(fig)
    animation = FuncAnimation(fig, visualizer, frames=np.arange(100),
                              init_func=visualizer.empty_frame, repeat=True, blit=True)
    plt.show()
