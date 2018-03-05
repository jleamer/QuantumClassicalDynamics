import numpy as np
import numexpr as ne
from types import MethodType, FunctionType
# in other codes, we have used scipy.fftpack to perform Fourier Transforms.
# In this code, we will use pyfftw, which is more suited for efficient large data
import pyfftw


class SplitOpWignerMoyal(object):
    """
    The second-order split-operator propagator for the Moyal equation for the Wigner function W(x, p, t)
    with the time-dependent Hamiltonian H = K(p, t) + V(x, t).
    (K and V may not depend on time.)

    This implementation using PyFFTW.

    Details about the method can be found at https://arxiv.org/abs/1212.3406

    This implementation stores the Wigner function as a 2D real array.
    """
    def __init__(self, **kwargs):
        """
        The following parameters must be specified
            X_gridDIM - the coordinate grid size
            X_amplitude - maximum value of the coordinates

            P_gridDIM - the momentum grid size
            P_amplitude - maximum value of the momentum

            V(x) - potential energy (as a string to be evaluated by numexpr) may depend on time
            diff_V(x) (optional) -- the derivative of the potential energy for the Ehrenfest theorem calculations

            K(p) - the kinetic energy (as a string to be evaluated by numexpr) may depend on time
            diff_K(p) (optional) -- the derivative of the kinetic energy for the Ehrenfest theorem calculations

            dt - time step
            t (optional) - initial value of time
        """
        # save all attributes
        for name, value in kwargs.items():
            # if the value supplied is a function, then dynamically assign it as a method;
            # otherwise bind it as a property
            if isinstance(value, FunctionType):
                setattr(self, name, MethodType(value, self, self.__class__))
            else:
                setattr(self, name, value)

        # Check that all the necessary attributes were specified
        try:
            # make sure self.X_gridDIM and self.P_gridDIM has a value of power of 2
            assert 2 ** int(np.log2(self.X_gridDIM)) == self.X_gridDIM and \
                2 ** int(np.log2(self.P_gridDIM)) == self.P_gridDIM, \
                "A value of the grid sizes (X_gridDIM and P_gridDIM) must be a power of 2"

        except AttributeError:
            raise AttributeError("Grid sizes (X_gridDIM and/or P_gridDIM) was not specified")

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

        ########################################################################################
        #
        #   Initialize Fourier transform for efficient calculations
        #
        ########################################################################################

        # Turn on the cache for optimum performance
        pyfftw.interfaces.cache.enable()

        # allocate the array for Wigner function
        self.wignerfunction = pyfftw.empty_aligned((self.P_gridDIM, self.X_gridDIM), dtype=np.float)

        # p x -> theta x
        self.transform_p2theta = pyfftw.builders.rfft(
            self.wignerfunction, axis=0,
            overwrite_input=True,  avoid_copy=True, threads=ne.nthreads,
        )

        # theta x  ->  p x
        self.transform_theta2p = pyfftw.builders.irfft(
            self.transform_p2theta(), axis=0,
            overwrite_input=True, avoid_copy=True, threads=ne.nthreads,
        )

        # p x  ->  p lambda
        self.transform_x2lambda = pyfftw.builders.rfft(
            self.wignerfunction, axis=1,
            overwrite_input=True, avoid_copy=True, threads=ne.nthreads,
        )

        # p lambda  ->  p x
        self.transform_lambda2x = pyfftw.builders.irfft(
            self.transform_x2lambda(), axis=1,
            overwrite_input=True, avoid_copy=True, threads=ne.nthreads,
        )

        ########################################################################################
        #
        #   Initialize grids
        #
        ########################################################################################

        # get coordinate and momentum step sizes
        self.dX = 2.*self.X_amplitude / self.X_gridDIM
        self.dP = 2.*self.P_amplitude / self.P_gridDIM

        # pre-compute the volume element in phase space
        self.dXdP = self.dX * self.dP

        # generate coordinate and momentum ranges
        # see http://docs.scipy.org/doc/numpy/reference/arrays.indexing.html
        # for explanation of np.newaxis and other array indexing operations
        # also https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html
        # for understanding the broadcasting in array operations

        self.X = (np.arange(self.X_gridDIM)[np.newaxis, :] - self.X_gridDIM / 2) * self.dX
        self.P = (np.arange(self.P_gridDIM)[:, np.newaxis] - self.P_gridDIM / 2) * self.dP

        #self.kX = np.arange(1 + self.X_gridDIM // 2)[np.newaxis, :]
        #self.kP = np.arange(1 + self.P_gridDIM // 2)[:, np.newaxis]

        # Lambda grid (variable conjugate to the coordinate)
        # (take only first half, as required by the real fft)
        self.Lambda = np.arange(1 + self.X_gridDIM // 2)[np.newaxis, :] * (np.pi / self.X_amplitude)

        # Theta grid (variable conjugate to the momentum)
        # (take only first half, as required by the real fft)
        self.Theta = np.arange(1 + self.P_gridDIM // 2)[:, np.newaxis] * (np.pi / self.P_amplitude)

        # numexpr code to calculate (-)**kP * np.exp(
        #        -self.dt*0.5j*(self.V(self.X - 0.5*self.Theta) - self.V(self.X + 0.5*self.Theta))
        # )
        self.code_expV = "exp(-0.5j * dt * (({V_minus}) - ({V_plus})))".format(
            V_minus=self.V.format(X="(X - 0.5 * Theta)"),
            V_plus=self.V.format(X="(X + 0.5 * Theta)"),
        )

        # allocate the memory for expV
        self.expV = ne.evaluate(self.code_expV, local_dict=vars(self))

        # numexpr code to calculate self.wignerfunction * np.exp(
        #    -self.dt * 1j * (self.K(self.P + 0.5 * self.Lambda) - self.K(self.P - 0.5 * self.Lambda))
        #)
        self.code_expK = "wignerfunction * exp(-dt * 1j *(({K_plus}) - ({K_minus})))".format(
            K_plus=self.K.format(P="(P + 0.5 * Lambda)"),
            K_minus=self.K.format(P="(P - 0.5 * Lambda)")
        )

        # Check whether the necessary terms are specified to calculate the Ehrenfest theorems
        try:
            # numexpr codes to calculate the first-order Ehrenfest theorems
            self.code_P_average_RHS = "sum(({}) * wignerfunction)".format(self.diff_V.format(X="X"))
            self.code_X_average_RHS = "sum(({}) * wignerfunction)".format(self.diff_K.format(P="P"))

            # Lists where the right hand sides of the Ehrenfest theorems for X and P
            self.X_average_RHS = []
            self.P_average_RHS = []

            # Lists where the expectation values of X and P
            self.X_average = []
            self.P_average = []

            # List where the expectation value of the Hamiltonian will be calculated
            self.hamiltonian_average = []

            self.code_V_average = "sum(({}) * wignerfunction)".format(self.V.format(X="X"))
            self.code_K_average = "sum(({}) * wignerfunction)".format(self.K.format(P="P"))

            # Flag requesting tha the Ehrenfest theorem calculations
            self.isEhrenfest = True

        except AttributeError:
            # Since self.diff_V and self.diff_K are not specified,
            # the Ehrenfest theorem will not be calculated
            self.isEhrenfest = False

    def propagate(self, time_steps=1):
        """
        Time propagate the Wigner function saved in self.wignerfunction
        :param time_steps: number of self.dt time increments to make
        :return: self.wignerfunction
        """
        for _ in range(time_steps):
            # advance by one time step
            self.single_step_propagation()

            # normalization
            self.wignerfunction /= self.wignerfunction.sum() * self.dXdP

            # calculate the Ehrenfest theorems
            self.get_Ehrenfest()

        return self.wignerfunction

    def get_purity(self):
        """
        Purity of the current wigner function
        :return:
        """
        return 2. * np.pi * ne.evaluate("sum(wignerfunction ** 2)", local_dict=vars(self)) * self.dXdP

    def single_step_propagation(self):
        """
        Perform single step propagation. The final Wigner function is not normalized.
        :return: self.wignerfunction
        """
        # make a half step in time
        self.t += 0.5 * self.dt

        # efficiently calculate expV
        ne.evaluate(self.code_expV, local_dict=vars(self), out=self.expV)

        # p x -> theta x
        self.wignerfunction = self.transform_p2theta(self.wignerfunction)

        self.wignerfunction *= self.expV

        # theta x  ->  p x
        self.wignerfunction = self.transform_theta2p(self.wignerfunction)

        # p x  ->  p lambda
        self.wignerfunction = self.transform_x2lambda(self.wignerfunction)
        ne.evaluate(self.code_expK, local_dict=vars(self), out=self.wignerfunction)

        # p lambda  ->  p x
        self.wignerfunction = self.transform_lambda2x(self.wignerfunction)

        # p x -> theta x
        self.wignerfunction = self.transform_p2theta(self.wignerfunction)
        self.wignerfunction *= self.expV

        # theta x  ->  p x
        self.wignerfunction = self.transform_theta2p(self.wignerfunction)

        # make a half step in time
        self.t += 0.5 * self.dt

        return self.wignerfunction

    def get_Ehrenfest(self):
        """
        Calculate observables entering the Ehrenfest theorems at time (t)
        """
        if self.isEhrenfest:

            self.X_average.append(
                ne.evaluate("sum(X * wignerfunction)", local_dict=vars(self)) * self.dXdP

            )
            self.P_average.append(
                ne.evaluate("sum(P * wignerfunction)", local_dict=vars(self)) * self.dXdP
            )

            self.P_average_RHS.append(
                -ne.evaluate(self.code_P_average_RHS, local_dict=vars(self)) * self.dXdP
            )
            self.X_average_RHS.append(
                ne.evaluate(self.code_X_average_RHS, local_dict=vars(self)) * self.dXdP
            )

            self.hamiltonian_average.append(
                ne.evaluate(self.code_K_average, local_dict=vars(self)) * self.dXdP
                +
                ne.evaluate(self.code_V_average, local_dict=vars(self)) * self.dXdP
            )

    def set_wignerfunction(self, new_wignerfunction):
        """
        Set the initial Wigner function
        :param new_wignerfunction: a 2D numpy array or sting containing the wigner function
        :return: self
        """
        if isinstance(new_wignerfunction, str):
            # Wigner function is supplied as a string
            ne.evaluate(new_wignerfunction, local_dict=vars(self), out=self.wignerfunction)

        elif isinstance(new_wignerfunction, np.ndarray):

            assert new_wignerfunction.shape == self.wignerfunction.shape, \
                "The grid sizes does not match with the Wigner function"

            # save only real part
            np.copyto(self.wignerfunction, new_wignerfunction.real)

        else:
            raise ValueError("Wigner function must be either string or numpy.array")

        # normalize
        self.wignerfunction /= self.wignerfunction.sum() * self.dXdP

        return self


##############################################################################
#
#   Run some examples
#
##############################################################################

if __name__ == '__main__':

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
            extent = [self.quant_sys.X.min(), self.quant_sys.X.max(), self.quant_sys.P.min(), self.quant_sys.P.max()]

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

                dt=0.05,

                X_gridDIM=256,
                X_amplitude=10.,

                P_gridDIM=256,
                P_amplitude=10.,

                # kinetic energy part of the hamiltonian
                K="0.5 * {P} ** 2",

                #omega=np.random.uniform(1, 3),
                omega=1.,

                # potential energy part of the hamiltonian
                V="0.5 * (omega * {X}) ** 2",

                # these functions are used for evaluating the Ehrenfest theorems
                diff_K="{P}",
                diff_V="omega ** 2 * {X}",
            )

            # set randomised initial condition
            self.quant_sys.set_wignerfunction(
                "exp( -{sigma} * (X - {X0}) ** 2 - (1. / {sigma}) * (P - {P0}) ** 2 )".format(
                    sigma=np.random.uniform(1., 3.),
                    P0=np.random.uniform(-3., 3.),
                    X0=np.random.uniform(-3., 3.),
                )
            )

        def __call__(self, frame_num):
            """
            Draw a new frame
            :param frame_num: current frame number
            :return: image objects
            """
            # propagate the wigner function
            self.img.set_array(self.quant_sys.propagate(20))
            return self.img,


    fig = plt.gcf()
    visualizer = VisualizeDynamicsPhaseSpace(fig)
    animation = FuncAnimation(
        fig, visualizer, frames=np.arange(100), repeat=True, blit=True
    )
    plt.show()

    # extract the reference to quantum system
    quant_sys = visualizer.quant_sys

    # Analyze how well the energy was preseved
    h = np.array(quant_sys.hamiltonian_average)
    print(
        "\nHamiltonian is preserved within the accuracy of %f percent" % ((1. - h.min() / h.max()) * 100)
    )

    #################################################################
    #
    # Plot the Ehrenfest theorems after the animation is over
    #
    #################################################################

    # generate time step grid
    dt = quant_sys.dt
    times = np.arange(dt, dt + dt * len(quant_sys.X_average), dt)

    plt.subplot(131)
    plt.title("The first Ehrenfest theorem verification")

    plt.plot(times, np.gradient(quant_sys.X_average, dt), 'r-', label='$d\\langle x \\rangle/dt$')
    plt.plot(times, quant_sys.X_average_RHS, 'b--', label='$\\langle p \\rangle$')

    plt.legend()
    plt.xlabel('time $t$ (a.u.)')

    plt.subplot(132)
    plt.title("The second Ehrenfest theorem verification")

    plt.plot(times, np.gradient(quant_sys.P_average, dt), 'r-', label='$d\\langle p \\rangle/dt$')
    plt.plot(times, quant_sys.P_average_RHS, 'b--', label='$\\langle -\\partial V/\\partial x \\rangle$')

    plt.legend()
    plt.xlabel('time $t$ (a.u.)')

    plt.subplot(133)
    plt.title('Hamiltonian')
    plt.plot(times, h)
    plt.xlabel('time $t$ (a.u.)')

    plt.show()