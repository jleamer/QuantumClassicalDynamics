import numpy as np
import numexpr as ne
from scipy import fftpack # Tools for fourier transform
from scipy import linalg # Linear algebra for dense matrix
from types import MethodType, FunctionType


class SplitOpSchrodinger2D:
    """
    The second-order split-operator propagator of the 2D Schrodinger equation in the coordinate representation
    with the time-dependent Hamiltonian H = K(P1, P2, t) + V(X1, X2, t).
    """
    def __init__(self, **kwargs):
        """
        The following parameters must be specified
            X1_gridDIM, X1_gridDIM - specifying the grid size
            X1_amplitude, X2_amplitude - maximum value of the coordinates

            V(X1, X2) - potential energy as a string to be evaluated by numexpr
            diff_V_X1(X1, X2) (optional)
             and
            diff_V_X2(X1, X2) (optional) -- the potential energy gradients (as a string to be evaluated by numexpr)
                                            for the Ehrenfest theorem calculations

            K(P1, P2) - kinetic energy as a string to be evaluated by numexpr
            diff_K_P1(P1, P2) (optional)
             and
            diff_K_P2(P1, P2) (optional) -- the kinetic energy gradient (as a string to be evaluated by numexpr)
                                            for the Ehrenfest theorem calculations

            dt - time step
            t (optional) - initial value of time
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
            # make sure self.X1_gridDIM and self.X2_gridDIM2 has a value of power of 2
            assert 2 ** int(np.log2(self.X1_gridDIM)) == self.X1_gridDIM and \
                2 ** int(np.log2(self.X2_gridDIM)) == self.X2_gridDIM, \
                "A value of the grid sizes (X1_gridDIM and X2_gridDIM) must be a power of 2"

        except AttributeError:
            raise AttributeError("Grid sizes (X1_gridDIM1 and/or X2_gridDIM) was not specified")

        try:
            self.X1_amplitude
            self.X2_amplitude
        except AttributeError:
            raise AttributeError("Coordinate ranges (X1_amplitude and/or X2_amplitude) was not specified")

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

        # get coordinate step sizes
        self.dX1 = 2. * self.X1_amplitude / self.X1_gridDIM
        self.dX2 = 2. * self.X2_amplitude / self.X2_gridDIM

        # generate coordinate ranges
        self.k1 = np.arange(self.X1_gridDIM)[:, np.newaxis]
        self.k2 = np.arange(self.X2_gridDIM)[np.newaxis, :]
        # see http://docs.scipy.org/doc/numpy/reference/arrays.indexing.html
        # for explanation of np.newaxis and other array indexing operations
        # also https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html
        # for understanding the broadcasting in array operations

        self.X1 = (self.k1 - self.X1_gridDIM / 2) * self.dX1
        self.X2 = (self.k2 - self.X2_gridDIM / 2) * self.dX2

        # generate momentum ranges
        self.P1 = (self.k1 - self.X1_gridDIM / 2) * (np.pi / self.X1_amplitude)
        self.P2 = (self.k2 - self.X2_gridDIM / 2) * (np.pi / self.X2_amplitude)

        # allocate the array for wavefunction
        self.wavefunction = np.zeros((self.X1_gridDIM, self.X2_gridDIM), dtype=np.complex)

        # allocate an axillary array needed for propagation
        self.expV = np.zeros_like(self.wavefunction)

        # numexpr code to calculate (-)**(k1 + k2) * exp(-0.5j * dt * V)
        self.code_expV = "(-1) ** (k1 + k2) * exp(- 0.5j * dt * (%s))" % self.V

        # numexpr code to calculate wavefunction * exp(-1j * dt * K)
        self.code_expK = "wavefunction * exp(-1j * dt * (%s))" % self.K

        # Check whether the necessary terms are specified to calculate the first-order Ehrenfest theorems
        try:
            # numexpr codes to calculate the first-order Ehrenfest theorems
            self.code_P1_average_RHS = "sum((%s) * density)" % self.diff_V_X1
            self.code_P2_average_RHS = "sum((%s) * density)" % self.diff_V_X2
            self.code_V_average = "sum((%s) * density)" % self.V
            self.code_X1_average = "sum(X1 * density)"
            self.code_X2_average = "sum(X2 * density)"

            self.code_X1_average_RHS = "sum((%s) * density)" % self.diff_K_P1
            self.code_X2_average_RHS = "sum((%s) * density)" % self.diff_K_P2
            self.code_K_average = "sum((%s) * density)" % self.K
            self.code_P1_average = "sum(P1 * density)"
            self.code_P2_average = "sum(P2 * density)"

            # Lists where the expectation values of X and P
            self.X1_average = []
            self.P1_average = []
            self.X2_average = []
            self.P2_average = []

            # Lists where the right hand sides of the Ehrenfest theorems for X and P
            self.X1_average_RHS = []
            self.P1_average_RHS = []
            self.X2_average_RHS = []
            self.P2_average_RHS = []

            # List where the expectation value of the Hamiltonian will be calculated
            self.hamiltonian_average = []

            # Allocate array for storing coordinate or momentum density of the wavefunction
            self.density = np.zeros(self.wavefunction.shape, dtype=np.float)

            # Allocate a copy of the wavefunction for storing the wavefunction in the momentum representation
            self.wavefunction_p = np.zeros_like(self.wavefunction)

            # Flag requesting tha the Ehrenfest theorem calculations
            self.isEhrenfest = True
        except AttributeError:
            # Since self.diff_V and self.diff_K are not specified,
            # the Ehrenfest theorem will not be calculated
            self.isEhrenfest = False

    def propagate(self, time_steps=1):
        """
        Time propagate the wave function saved in self.wavefunction
        :param time_steps: number of self.dt time increments to make
        :return: self.wavefunction
        """

        # pre-compute the sqrt of the volume element
        sqrtdX1dX2 = np.sqrt(self.dX1 * self.dX2)

        for _ in xrange(time_steps):
            # make a half step in time
            self.t += 0.5 * self.dt

            # efficiently calculate expV
            ne.evaluate(self.code_expV, local_dict=self.__dict__, out=self.expV)
            self.wavefunction *= self.expV

            # going to the momentum representation
            self.wavefunction = fftpack.fft2(self.wavefunction, overwrite_x=True)
            ne.evaluate(self.code_expK, local_dict=self.__dict__, out=self.wavefunction)

            # going back to the coordinate representation
            self.wavefunction = fftpack.ifft2(self.wavefunction, overwrite_x=True)
            self.wavefunction *= self.expV

            # normalize
            # this line is equivalent to
            # self.wavefunction /= np.sqrt(np.sum(np.abs(self.wavefunction)**2)*self.dX1*self.dX2)
            self.wavefunction /= linalg.norm(self.wavefunction.reshape(-1)) * np.sqrt(self.dX1 * self.dX2)

            # make a half step in time
            self.t += 0.5 * self.dt

            # calculate the Ehrenfest theorems
            self.get_Ehrenfest()

        return self.wavefunction

    def get_Ehrenfest(self):
        """
        Calculate observables entering the Ehrenfest theorems at time (t)
        """
        if self.isEhrenfest:
            # evaluate the coordinate density
            np.abs(self.wavefunction, out=self.density)
            self.density *= self.density
            # normalize
            self.density /= self.density.sum()

            # save the current value of coordinate-dependent observables
            self.X1_average.append(
                ne.evaluate(self.code_X1_average, local_dict=self.__dict__)
            )
            self.X2_average.append(
                ne.evaluate(self.code_X2_average, local_dict=self.__dict__)
            )
            self.P1_average_RHS.append(
                -ne.evaluate(self.code_P1_average_RHS, local_dict=self.__dict__)
            )
            self.P2_average_RHS.append(
                -ne.evaluate(self.code_P2_average_RHS, local_dict=self.__dict__)
            )

            # save the potential energy
            self.hamiltonian_average.append(
                ne.evaluate(self.code_V_average, local_dict=self.__dict__)
            )

            # calculate density in the momentum representation
            ne.evaluate("(-1) ** (k1 + k2) * wavefunction", local_dict=self.__dict__, out=self.wavefunction_p)
            self.wavefunction_p = fftpack.fft2(self.wavefunction_p, overwrite_x=True)
            np.abs(self.wavefunction_p, out=self.density)
            self.density *= self.density
            # normalize
            self.density /= self.density.sum()

            # save the current value of momentum-dependent observables
            self.P1_average.append(
                ne.evaluate(self.code_P1_average, local_dict=self.__dict__)
            )
            self.P2_average.append(
                ne.evaluate(self.code_P2_average, local_dict=self.__dict__)
            )
            self.X1_average_RHS.append(
                ne.evaluate(self.code_X1_average_RHS, local_dict=self.__dict__)
            )
            self.X2_average_RHS.append(
                ne.evaluate(self.code_X2_average_RHS, local_dict=self.__dict__)
            )

            # add the kinetic energy to get the hamiltonian
            self.hamiltonian_average[-1] += \
                ne.evaluate(self.code_K_average, local_dict=self.__dict__)

    def set_wavefunction(self, wavefunc):
        """
        Set the initial wave function
        :param wavefunc: 2D numpy array containing the wave function
        :return: self
        """
        if isinstance(wavefunc, str):
            # wavefunction is supplied as a string
            ne.evaluate("%s + 0j" % wavefunc, local_dict=self.__dict__, out=self.wavefunction)

        elif isinstance(wavefunc, np.ndarray):
            # wavefunction is supplied as an array

            # perform the consistency checks
            assert wavefunc.shape == self.wavefunction.shape,\
                "The grid size does not match with the wave function"

            # make sure the wavefunction is stored as a complex array
            np.copyto(self.wavefunction, wavefunc.astype(np.complex))

        else:
            raise ValueError("wavefunc must be either string or numpy.array")

        # normalize
        self.wavefunction /= linalg.norm(self.wavefunction.reshape(-1)) * np.sqrt(self.dX1 * self.dX2)

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
    from matplotlib.animation import FuncAnimation, writers

    # Use the documentation string for the developed class
    print(SplitOpSchrodinger2D.__doc__)

    class VisualizeDynamics2D:
        """
        Class to visualize the wave function dynamics in 2D.
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

            ax.set_title('Wavefunction density, $| \\Psi(x_1, x_2, t) |^2$')
            extent=[self.quant_sys.X2.min(), self.quant_sys.X2.max(), self.quant_sys.X1.min(), self.quant_sys.X1.max()]
            self.img = ax.imshow([[]], extent=extent, origin='lower')

            self.fig.colorbar(self.img)

            ax.set_xlabel('$x_2$ (a.u.)')
            ax.set_ylabel('$x_1$ (a.u.)')

        def set_quantum_sys(self):
            """
            Initialize quantum propagator
            :param self:
            :return:
            """
            self.quant_sys = SplitOpSchrodinger2D(
                t=0.,
                dt=0.005,
                X1_gridDIM=256,
                X1_amplitude=5.,
                X2_gridDIM=256,
                X2_amplitude=5.,

                # kinetic energy part of the hamiltonian
                K="0.5 * (P1 ** 2 + P2 ** 2)",

                # these functions are used for evaluating the Ehrenfest theorems
                diff_K_P1="P1",
                diff_K_P2="P2",

                # potential energy part of the hamiltonian
                V="0.5 * 3 ** 2 * (X1 ** 2 + X2 ** 2)",

                # these functions are used for evaluating the Ehrenfest theorems
                diff_V_X1="3 ** 2 * X1",
                diff_V_X2="3 ** 2 * X2",
            )

            # set randomised initial condition
            self.quant_sys.set_wavefunction(
                np.exp(
                    # randomized positions
                    -np.random.uniform(0.5, 3.) * (self.quant_sys.X1 + np.random.uniform(-2., 2.)) ** 2
                    -np.random.uniform(0.5, 3.) * (self.quant_sys.X2 + np.random.uniform(-2., 2.)) ** 2
                    # randomized initial velocities
                    -1j * np.random.uniform(-2., 2.) * self.quant_sys.X1
                    -1j * np.random.uniform(-2., 2.) * self.quant_sys.X2
                )
            )

        def empty_frame(self):
            """
            Make empty frame and reinitialize quantum system
            :param self:
            :return: image object
            """
            self.set_quantum_sys()
            self.img.set_array([[0]])
            return self.img,

        def __call__(self, frame_num):
            """
            Draw a new frame
            :param frame_num: current frame number
            :return: image objects
            """
            # propagate and set the density
            self.img.set_array(
                np.abs(self.quant_sys.propagate(20))**2
            )
            return self.img,

    fig = plt.gcf()
    visualizer = VisualizeDynamics2D(fig)
    animation = FuncAnimation(fig, visualizer, frames=np.arange(100),
                              init_func=visualizer.empty_frame, repeat=True, blit=True)

    plt.show()

    # If you want to make a movie, comment "plt.show()" out and uncomment the lines bellow

    # Set up formatting for the movie files
    #writer = writers['mencoder'](fps=10, metadata=dict(artist='Denys Bondar'), bitrate=-1)

    # Save animation into the file
    #animation.save('2D_Schrodinger.mp4', writer=writer)

    # extract the reference to quantum system
    quant_sys = visualizer.quant_sys

    # Analyze how well the energy was preserved
    h = np.array(quant_sys.hamiltonian_average)
    print(
        "\nHamiltonian is preserved within the accuracy of %.2e percent" % (100. * (1. - h.min()/h.max()))
    )

    #################################################################
    #
    # Plot the Ehrenfest theorems after the animation is over
    #
    #################################################################

    # generate time step grid
    dt = quant_sys.dt
    times = np.arange(dt, dt + dt*len(quant_sys.X1_average), dt)

    plt.subplot(121)
    plt.title("The first Ehrenfest theorem verification")

    plt.plot(times, np.gradient(quant_sys.X1_average, dt), 'r-', label='$d\\langle \\hat{x}_1 \\rangle/dt$')
    plt.plot(times, quant_sys.X1_average_RHS, 'b--', label='$\\langle \\hat{p}_1 \\rangle$')

    plt.plot(times, np.gradient(quant_sys.X2_average, dt), 'g-', label='$d\\langle \\hat{x}_2 \\rangle/dt$')
    plt.plot(times, quant_sys.X2_average_RHS,  'k--', label='$\\langle \\hat{p}_2 \\rangle$')

    plt.legend()
    plt.xlabel('time $t$ (a.u.)')

    plt.subplot(122)
    plt.title("The second Ehrenfest theorem verification")

    plt.plot(times, np.gradient(quant_sys.P1_average, dt), 'r-', label='$d\\langle \\hat{p}_1 \\rangle/dt$')
    plt.plot(times, quant_sys.P1_average_RHS, 'b--', label='$\\langle -\\partial\\hat{V}/\\partial\\hat{x}_1 \\rangle$')

    plt.plot(times, np.gradient(quant_sys.P2_average, dt), 'g-', label='$d\\langle \\hat{p}_2 \\rangle/dt$')
    plt.plot(times, quant_sys.P2_average_RHS, 'k--', label='$\\langle -\\partial\\hat{V}/\\partial\\hat{x}_2 \\rangle$')

    plt.legend()
    plt.xlabel('time $t$ (a.u.)')

    plt.show()