import numpy as np

# numpy.fft has better implementation of real fourier transform
# necessary for real split operator propagator
from numpy import fft


class SplitOpWignerMoyal:
    """
    Split-operator propagator for the Moyal equation for the Wigner function W(x, p, t)
    with the time-dependent Hamiltonian H = K(p, t) + V(x, t).
    (K and V may not depend on time.)

    This implementation stores the Wigner function as a 2D real array.
    """
    def __init__(self, **kwargs):
        """
        The following parameters must be specified
            X_gridDIM - the coordinate grid size
            X_amplitude - maximum value of the coordinates
            P_gridDIM - the momentum grid size
            P_amplitude - maximum value of the momentum
            V(x) - potential energy (as a function) may depend on time
            K(p) - momentum dependent part of the hamiltonian (as a function) may depend on time
            dt - time step
            t (optional) - initial value of time
        """

        # save all attributes
        for name, value in kwargs.items():
            setattr(self, name, value)

        # Check that all attributes were specified
        try:
            self.X_gridDIM
        except AttributeError:
            raise AttributeError("Coordinate grid size (X_gridDIM) was not specified")

        assert self.X_gridDIM % 2 == 0, "Coordinate grid size (X_gridDIM) must be even"

        try:
            self.P_gridDIM
        except AttributeError:
            raise AttributeError("Momentum grid size (P_gridDIM) was not specified")

        assert self.P_gridDIM % 2 == 0, "Momentum grid size (P_gridDIM) must be even"

        try:
            self.X_amplitude
        except AttributeError:
            raise AttributeError("Coordinate grid range (X_amplitude) was not specified")

        try:
            self.P_amplitude
        except AttributeError:
            raise AttributeError("Momentum grid range (P_amplitude) was not specified")

        try:
            self.V
        except AttributeError:
            raise AttributeError("Potential energy (V) was not specified")

        try:
            self.K
        except AttributeError:
            raise AttributeError("Momentum dependence (K) was not specified")

        try:
            self.dt
        except AttributeError:
            raise AttributeError("Time-step (dt) was not specified")

        try:
            self.t
        except AttributeError:
            print("Warning: Initial time (t) was not specified, thus it is set to zero.")
            self.t = 0.

        # get coordinate and momentum step sizes
        self.dX = 2.*self.X_amplitude / self.X_gridDIM
        self.dP = 2.*self.P_amplitude / self.P_gridDIM

        # coordinate grid
        self.X = np.linspace(-self.X_amplitude, self.X_amplitude - self.dX , self.X_gridDIM)
        self.X = self.X[np.newaxis, :]

        # Lambda grid (variable conjugate to the coordinate)
        self.Lambda = fft.fftfreq(self.X_gridDIM, self.dX/(2*np.pi))

        # take only first half, as required by the real fft
        self.Lambda = self.Lambda[:(1 + self.X_gridDIM//2)]
        #
        self.Lambda = self.Lambda[np.newaxis, :]

        # momentum grid
        self.P = np.linspace(-self.P_amplitude, self.P_amplitude - self.dP, self.P_gridDIM)
        self.P = self.P[:, np.newaxis]

        # Theta grid (variable conjugate to the momentum)
        self.Theta = fft.fftfreq(self.P_gridDIM, self.dP/(2*np.pi))

        # take only first half, as required by the real fft
        self.Theta = self.Theta[:(1 + self.P_gridDIM//2)]
        #
        self.Theta = self.Theta[:, np.newaxis]

        try:
            # Pre-calculate the exponent, if the potential is time independent
            self._expV = np.exp(
                -self.dt*1j*(self.V(self.X - 0.5*self.Theta) - self.V(self.X + 0.5*self.Theta))
            )
        except TypeError:
            # If exception is generated, then the potential is time-dependent
            # and caching is not possible
            pass

        try:
            # Pre-calculate the exponent, if the kinetic energy is time independent
            self._expK = np.exp(
                -self.dt*1j*(self.K(self.P + 0.5*self.Lambda) - self.K(self.P - 0.5*self.Lambda))
            )
        except TypeError:
            # If exception is generated, then the kinetic energy is time-dependent
            # and caching is not possible
            pass

    def single_step_propagation(self):
        """
        Perform single step propagation. The final Wigner function is not normalized.
        :return: self.wignerfunction
        """
        # p x -> theta x
        self.wignerfunction = fft.rfft(self.wignerfunction, axis=0)
        self.wignerfunction *= self.get_expV(self.t)

        # theta x  ->  p x
        self.wignerfunction = fft.irfft(self.wignerfunction, axis=0)

        # p x  ->  p lambda
        self.wignerfunction = fft.rfft(self.wignerfunction, axis=1)
        self.wignerfunction *= self.get_expK(self.t)

        # p lambda  ->  p x
        self.wignerfunction = fft.irfft(self.wignerfunction, axis=1)

        return self.wignerfunction

    def propagate(self, time_steps=1):
        """
        Time propagate the Wigner function saved in self.wignerfunction
        :param time_steps: number of self.dt time increments to make
        :return: self.wignerfunction
        """
        # pre-compute the volume element in phase space
        dXdP = self.dX * self.dP

        for _ in xrange(time_steps):

            # advance by one time step
            self.single_step_propagation()

            # normalization
            self.wignerfunction /= self.wignerfunction.sum() * dXdP

            # increment current time
            self.t += self.dt

        return self.wignerfunction

    def get_expV(self, t):
        """
        Return the exponent of the potential energy difference at time (t)
        """
        try:
            # aces the pre-calculated value
            return self._expV
        except AttributeError:
            # Calculate in efficient way
            result = -self.dt*1j*(self.V(self.X - 0.5*self.Theta, t) - self.V(self.X + 0.5*self.Theta, t))
            return np.exp(result, out=result)

    def get_expK(self, t):
        """
        Return the exponent of the kinetic energy difference at time  (t)
        """
        try:
            # aces the pre-calculated value
            return self._expK
        except AttributeError:
            # Calculate result = np.exp(*self.K(self.P1, self.P2, t))
            result = -self.dt*1j*(self.K(self.P + 0.5*self.Lambda, t) - self.K(self.P - 0.5*self.Lambda, t))
            return np.exp(result, out=result)

    def set_wignerfunction(self, new_wigner_func):
        """
        Set the initial Wigner function
        :param new_wigner_func: 2D numoy array contaning the wigner function
        :return: self
        """
        # perform the consistency checks
        assert new_wigner_func.shape == (self.P.size, self.X.size), \
            "The grid sizes does not match with the Wigner function"

        assert new_wigner_func.dtype == np.float, "Supplied Wigner function must be real"

        # make sure the Wigner function is stored as a complex array
        self.wignerfunction = new_wigner_func.copy()

        # normalize
        self.wignerfunction /= self.wignerfunction.sum() * self.dX*self.dP

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
    from matplotlib.animation import FuncAnimation

    # Use the documentation string for the developed class
    print(SplitOpWignerMoyal.__doc__)


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

            ax = fig.add_subplot(111)

            ax.set_title('Wigner function, $W(x,p,t)$')
            extent=[self.quant_sys.X.min(), self.quant_sys.X.max(), self.quant_sys.P.min(), self.quant_sys.P.max()]

            # import utility to visualize the wigner function
            from wigner_normalize import WignerNormalize

            # generate empty plot
            self.img = ax.imshow(
                [[]],
                extent=extent,
                origin='lower',
                cmap='seismic',
                norm=WignerNormalize(vmin=-0.01, vmax=0.1)
            )

            self.fig.colorbar(self.img)

            ax.set_xlabel('$x$ (a.u.)')
            ax.set_ylabel('$p$ (a.u.)')

        def set_quantum_sys(self):
            """
            Initialize quantum propagator
            :param self:
            :return:
            """
            self.quant_sys = SplitOpWignerMoyal(
                t=0,
                dt=0.005,
                X_gridDIM=256,
                X_amplitude=8.,
                P_gridDIM=256,
                P_amplitude=8.,
                # kinetic energy part of the hamiltonian
                K=lambda p: 0.5*p**2,
                # potential energy part of the hamiltonian
                V=lambda x: 0.5*9*x**2,
                # the force, i.e., F = -diff(V, x)
                # Note: this functions are not used for propagation that are used below
                F=lambda x: -9*x
            )

            # parameter controling the width of the wigner function
            sigma = np.random.uniform(1., 3.)

            # set randomised initial condition
            self.quant_sys.set_wignerfunction(
                np.exp(
                    # randomized position
                    -sigma*(self.quant_sys.X + np.random.uniform(-1., 1.))**2
                    # randomized initial velocity
                    -(1./sigma)*(self.quant_sys.P + np.random.uniform(-1., 1.))**2
                )
            )

            # set the list where expectation values will be saved
            self.average_X = []
            self.average_P = []
            self.average_F = []
            self.average_hamiltonian = []

            # pre-calculate some arrays to speed up the calculation of averages

            # the force
            self.F = self.quant_sys.F(self.quant_sys.X)

            # the hamiltonian = kinetic energy + potential energy
            self.H = self.quant_sys.K(self.quant_sys.P) + self.quant_sys.V(self.quant_sys.X)

        def empty_frame(self):
            """
            Make empty frame and reinitialize quantum system
            :param self:
            :return: image object
            """
            self.set_quantum_sys()
            self.img.set_array([[]])
            return self.img,

        def __call__(self, frame_num):
            """
            Draw a new frame
            :param frame_num: current frame number
            :return: image objects
            """
            # Pre-calculate the volume element in phase space
            dXdP = self.quant_sys.dX * self.quant_sys.dP

            for _ in xrange(20):
                # get the Wigner function in the coordinate representation
                W = self.quant_sys.propagate()

                self.average_X.append(np.sum(W * self.quant_sys.X) * dXdP)
                self.average_P.append(np.sum(W * self.quant_sys.P) * dXdP)

                self.average_F.append(np.sum(W * self.F) * dXdP)
                self.average_hamiltonian.append(np.sum(W * self.H) * dXdP)

            # the wigner function
            self.img.set_array(W)
            return self.img,


    fig = plt.gcf()
    visualizer = VisualizeDynamicsPhaseSpace(fig)
    animation = FuncAnimation(fig, visualizer, frames=np.arange(100),
                              init_func=visualizer.empty_frame, repeat=True, blit=True)
    plt.show()

    # Analyze how well the energy was preseved
    h = np.array(visualizer.average_hamiltonian)
    print(
        "\nHamiltonian is preserved within the accuracy of %f percent" % ((1. - h.min()/h.max())*100)
    )

    #################################################################
    #
    # Plot the Ehrenfest theorems after the animation is over
    #
    #################################################################

    # generate time step grid
    dt = visualizer.quant_sys.dt
    times = np.arange(dt, dt + dt*len(visualizer.average_X), dt)

    plt.subplot(121)
    plt.title("The first Ehrenfest theorem verification")

    plt.plot(times, np.gradient(visualizer.average_X, dt), 'r-', label='$d\\langle x \\rangle/dt$')
    plt.plot(times, visualizer.average_P, 'b--', label='$\\langle p \\rangle$')

    plt.legend()
    plt.xlabel('time $t$ (a.u.)')

    plt.subplot(122)
    plt.title("The second Ehrenfest theorem verification")

    plt.plot(times, np.gradient(visualizer.average_P, dt), 'r-', label='$d\\langle p \\rangle/dt$')
    plt.plot(times, visualizer.average_F, 'b--', label='$\\langle -\\partial\ V /\\partial x \\rangle$')

    plt.legend()
    plt.xlabel('time $t$ (a.u.)')

    plt.show()