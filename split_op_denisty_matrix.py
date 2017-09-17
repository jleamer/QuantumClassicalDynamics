import numpy as np
import numexpr as ne
from scipy import fftpack # Tools for fourier transform
from scipy import linalg # Linear algebra for dense matrix
from types import MethodType, FunctionType


class DensityMatrix:
    """
    The second-order split-operator propagator for the Lindblad master equation

    d \rho / dt = i [\rho, H(t)] + A \rho A^{\dagger} - 1/2 \rho A^{\dagger} A - 1/2 A^{\dagger} A \rho

    in the coordinate representation  with the time-dependent Hamiltonian H = K(p, t) + V(x, t)
    and the coordinate dependent dissipator A = A(x, t).
    """
    def __init__(self, **kwargs):
        """
        The following parameters must be specified
            X_gridDIM - specifying the grid size
            X_amplitude - maximum value of the coordinates
            V - potential energy (as a string to be evaluated by numexpr)
            K - momentum dependent part of the hamiltonian (as a string to be evaluated by numexpr)
            A - a coordinate dependent Lindblad dissipator (as a string to be evaluated by numexpr)
            RHS_P_A (optional) -- the correction to the second Ehrenfest theorem due to A
            diff_V (optional) -- the derivative of the potential energy for the Ehrenfest theorem calculations
            diff_K (optional) -- the derivative of the kinetic energy for the Ehrenfest theorem calculations
            t (optional) - initial value of time
            abs_boundary (optional) -- absorbing boundary (as a string to be evaluated by numexpr)
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
            # make sure self.X_gridDIM has a value of power of 2
            assert 2 ** int(np.log2(self.X_gridDIM)) == self.X_gridDIM, \
                "A value of the grid size (X_gridDIM) must be a power of 2"
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

        try:
            self.K
        except AttributeError:
            raise AttributeError("Momentum dependence (K) was not specified")

        try:
            self.A
        except AttributeError:
            raise AttributeError("Coordinate dependent Lindblad dissipator (A) was not specified")

        try:
            self.dt
        except AttributeError:
            raise AttributeError("Time-step (dt) was not specified")

        try:
            self.t
        except AttributeError:
            print("Warning: Initial time (t) was not specified, thus it is set to zero.")
            self.t = 0.

        try:
            self.abs_boundary
        except AttributeError:
            print("Warning: Absorbing boundary (abs_boundary) was not specified, thus it is turned off")
            self.abs_boundary = "1."

        # get coordinate step size
        self.dX = 2. * self.X_amplitude / self.X_gridDIM

        # generate coordinate range
        k = np.arange(self.X_gridDIM)
        self.k = k[:, np.newaxis]
        self.k_prime = k[np.newaxis, :]

        X = (k - self.X_gridDIM / 2) * self.dX
        self.X = X[:, np.newaxis]
        self.X_prime = X[np.newaxis, :]

        # generate momentum range
        self.dP = np.pi / self.X_amplitude
        P = (k - self.X_gridDIM / 2) * self.dP
        self.P = P[:, np.newaxis]
        self.P_prime = P[np.newaxis, :]

        # allocate the array for density matrix
        self.rho = np.zeros((self.X_gridDIM, self.X_gridDIM), dtype=np.complex)

        # allocate an axillary array needed for propagation
        self.expV = np.zeros_like(self.rho)

        # construct the coordinate dependent phase containing the dissipator as well as coherent propagator
        F = "1.j * (" + self.V.format(X="X_prime") + " - " + self.V.format(X="X") + ") + "\
            + self.A.format(X="X") + " * conj(" + self.A.format(X="X_prime") + ") "\
            "- 0.5 * abs(" + self.A.format(X="X_prime") + ") - 0.5 * abs(" + self.A.format(X="X") + ") "

        # numexpr code to calculate (-)**(k + k_prime) * exp(0.5 * dt * F)
        self.code_expV = "(%s) * (%s) * (-1) ** (k + k_prime) * exp(0.5 * dt * (%s))" % (
            self.abs_boundary.format(X="X"), self.abs_boundary.format(X="X_prime"), F
        )

        # numexpr code to calculate rho * exp(1j * dt * K)
        self.code_expK = "rho * exp(1j * dt * ((%s) - (%s)))" % (
            self.K.format(P="P_prime"), self.K.format(P="P")
        )

        # Check whether the necessary terms are specified to calculate the first-order Ehrenfest theorems
        try:
            # numexpr codes to calculate the First Ehrenfest theorems
            self.code_P_average_RHS = "sum(((%s) - (%s)) * density)" % (
                self.diff_V.format(X="X"), self.RHS_P_A.format(X="X")
            )
            self.code_V_average = "sum((%s) * density)" % self.V.format(X="X")
            self.code_X_average = "sum(X * density)"

            self.code_X_average_RHS = "sum((%s) * density)" % self.diff_K.format(P="P")
            self.code_K_average = "sum((%s) * density)" % self.K.format(P="P")
            self.code_P_average = "sum(P * density)"


            # Lists where the expectation values of X and P
            self.X_average = []
            self.P_average = []

            # Lists where the right hand sides of the Ehrenfest theorems for X and P
            self.X_average_RHS = []
            self.P_average_RHS = []

            # List where the expectation value of the Hamiltonian will be calculated
            self.hamiltonian_average = []

            # Allocate a copy of the wavefunction for storing the wavefunction in the momentum representation
            self.rho_p = np.zeros_like(self.rho)

            # Flag requesting tha the Ehrenfest theorem calculations
            self.isEhrenfest = True
        except AttributeError:
            # Since self.diff_V and self.diff_K are not specified,
            # the first Ehrenfest theorem will not be calculated
            self.isEhrenfest = False

    def propagate(self, time_steps=1):
        """
        Time propagate the density matrix saved in self.rho
        :param time_steps: number of self.dt time increments to make
        :return: self.rho
            """
        for _ in xrange(time_steps):
            # make a half step in time
            self.t += 0.5 * self.dt

            # efficiently calculate expV
            ne.evaluate(self.code_expV, local_dict=vars(self), out=self.expV)
            self.rho *= self.expV

            # going to the momentum representation
            self.rho = fftpack.fft(self.rho, axis=0, overwrite_x=True)
            self.rho = fftpack.ifft(self.rho, axis=1, overwrite_x=True)

            ne.evaluate(self.code_expK, local_dict=vars(self), out=self.rho)

            # going back to the coordinate representation
            self.rho = fftpack.ifft(self.rho, axis=0, overwrite_x=True)
            self.rho = fftpack.fft(self.rho, axis=1, overwrite_x=True)

            self.rho *= self.expV

            # normalize
            self.rho /= self.rho.trace() * self.dX

            # make a half step in time
            self.t += 0.5 * self.dt

            # calculate the Ehrenfest theorems
            self.get_Ehrenfest()

        return self.rho

    def get_Ehrenfest(self):
        """
        Calculate observables entering the Ehrenfest theorems
        """
        if self.isEhrenfest:
            # extract the coordinate density and give it the shape of self.X
            self.density = self.rho.diagonal().reshape(self.X.shape)

            # save the current value of <X>
            self.X_average.append(
                self.dX * ne.evaluate(self.code_X_average, local_dict=vars(self)).real
            )
            self.P_average_RHS.append(
                -self.dX * ne.evaluate(self.code_P_average_RHS, local_dict=vars(self)).real
            )

            # save the potential energy
            self.hamiltonian_average.append(
                self.dX * ne.evaluate(self.code_V_average, local_dict=vars(self)).real
            )

            # calculate density in the momentum representation
            ne.evaluate("(-1) ** (k + k_prime) * rho", local_dict=vars(self), out=self.rho_p)

            self.rho_p = fftpack.fft(self.rho_p, axis=0, overwrite_x=True)
            self.rho_p = fftpack.ifft(self.rho_p, axis=1, overwrite_x=True)

            # normalize
            self.rho_p /= self.rho_p.trace() * self.dP

            self.density = self.rho_p.diagonal().reshape(self.P.shape)

            # save the current value of <P>
            self.P_average.append(
                self.dP * ne.evaluate(self.code_P_average, local_dict=vars(self)).real
            )
            self.X_average_RHS.append(
                self.dP * ne.evaluate(self.code_X_average_RHS, local_dict=vars(self)).real
            )

            # add the kinetic energy to get the hamiltonian
            self.hamiltonian_average[-1] += \
                self.dP * ne.evaluate(self.code_K_average, local_dict=vars(self)).real

    def set_rho(self, rho):
        """
        Set the initial density matrix
        :param rho: 2D numpy array or sting containing the density matrix
        :return: self
        """
        if isinstance(rho, str):
            # density matrix is supplied as a string
            ne.evaluate("%s + 0j" % rho, local_dict=vars(self), out=self.rho)

        elif isinstance(rho, np.ndarray):
            # density matrix is supplied as an array

            # perform the consistency checks
            assert rho.shape == self.rho.shape,\
                "The grid size does not match with the density matrix"

            # make sure the density matrix is stored as a complex array
            np.copyto(self.rho, rho.astype(np.complex))

        else:
            raise ValueError("density matrix must be either string or numpy.array")

        # normalize
        self.rho /= self.rho.trace() * self.dX

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
    print(DensityMatrix.__doc__)

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

            ax.set_title('Density matrix, $| \\rho(x, x\prime, t) |$')
            extent=[self.quant_sys.X.min(), self.quant_sys.X_prime.max(), self.quant_sys.X.min(), self.quant_sys.X_prime.max()]
            self.img = ax.imshow([[]], extent=extent, origin='lower')

            self.fig.colorbar(self.img)

            ax.set_xlabel('$x\prime$ (a.u.)')
            ax.set_ylabel('$x$ (a.u.)')

        def set_quantum_sys(self):
            """
            Initialize quantum propagator
            :param self:
            :return:
            """
            self.quant_sys = DensityMatrix(
                t=0.,
                dt=0.005,
                X_gridDIM=256,
                X_amplitude=5.,

                # dissipator
                alpha=np.random.uniform(0.1, 0.5),
                R=np.random.uniform(-10., 10.),

                A="R * exp(-1j * alpha * 0.25 * {X} ** 4 / R ** 2)",

                RHS_P_A="-alpha * {X} ** 3",

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

            # set randomised initial condition
            self.quant_sys.set_rho("exp(-((X-1.)**2 + (X_prime-1.)**2))")

        def empty_frame(self):
            """
            Make empty frame and reinitialize quantum system
            :param self:
            :return: image object
            """
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
                np.abs(self.quant_sys.propagate(200))**2
            )
            return self.img,

    fig = plt.gcf()
    visualizer = VisualizeDynamics2D(fig)
    animation = FuncAnimation(fig, visualizer, frames=np.arange(100),
                              init_func=visualizer.empty_frame, repeat=True, blit=True)

    plt.show()

    # If you want to make a movie, comment "plt.show()" out and uncomment the lines bellow

    # Set up formatting for the movie files
    #writer = writers['mencoder'](fps=10, metadata=dict(artist='a good student'), bitrate=-1)

    # Save animation into the file
    #animation.save('2D_Schrodinger.mp4', writer=writer)

    # extract the reference to quantum system
    quant_sys = visualizer.quant_sys

    #################################################################
    #
    # Plot the Ehrenfest theorems after the animation is over
    #
    #################################################################

    # generate time step grid
    dt = quant_sys.dt
    times = np.arange(dt, dt + dt*len(quant_sys.X_average), dt)

    plt.subplot(121)
    plt.title("The first Ehrenfest theorem verification")

    plt.plot(times, np.gradient(quant_sys.X_average, dt), 'r-', label='$d\\langle \\hat{x} \\rangle/dt$')
    plt.plot(times, quant_sys.X_average_RHS, 'b--', label='$\\langle \\hat{p} \\rangle$')

    plt.legend()
    plt.xlabel('time $t$ (a.u.)')

    plt.subplot(122)
    plt.title("The second Ehrenfest theorem verification")

    plt.plot(times, np.gradient(quant_sys.P_average, dt), 'r-', label='$d\\langle \\hat{p} \\rangle/dt$')
    plt.plot(times, quant_sys.P_average_RHS, 'b--', label='$\\langle -\\partial\\hat{V}/\\partial\\hat{x} \\rangle$')

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