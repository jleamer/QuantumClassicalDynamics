import numpy as np
import numexpr as ne
from types import MethodType, FunctionType
import warnings
import pickle
# in other codes, we have used scipy.fftpack to perform Fourier Transforms.
# In this code, we will use pyfftw, which is more suited for efficient large data
import pyfftw


class DensityMatrix(object):
    """
    The second-order split-operator propagator for the Lindblad master equation

    d \\rho / dt = i [\\rho, H(t)]
                + A \\rho A^{\dagger} - 1/2 \\rho A^{\dagger} A - 1/2 A^{\dagger} A \\rho
                + B \\rho B^{\dagger} - 1/2 \\rho B^{\dagger} B - 1/2 B^{\dagger} B \\rho

    in the coordinate representation  with the time-dependent Hamiltonian H = K(p, t) + V(x, t)
    and the coordinate dependent dissipator A = A(x, t) and the momentum dependent dissipator B = B(p, t).
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

            B - a momentum dependent Lindblad dissipator (as a string to be evaluated by numexpr)
            RHS_X_B (optional) -- the correction to the first Ehrenfest theorem due to B

            diff_V (optional) -- the derivative of the potential energy for the Ehrenfest theorem calculations
            diff_K (optional) -- the derivative of the kinetic energy for the Ehrenfest theorem calculations
            t (optional) - initial value of time
            abs_boundary (optional) -- absorbing boundary (as a string to be evaluated by numexpr)
        """

        # save all attributes
        for name, value in kwargs.items():
            # if the value supplied is a function, then dynamically assign it as a method;
            if isinstance(value, FunctionType):
                setattr(self, name, MethodType(value, self))
            # otherwise bind it as a property
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
            self.A = self.RHS_P_A = "0."
            warnings.warn("coordinate dependent Lindblad dissipator (A) was not specified so it is set to zero")

        try:
            self.B
        except AttributeError:
            self.B = self.RHS_X_B = "0."
            warnings.warn("momentum dependent Lindblad dissipator (B) was not specified so it is set to zero")

        try:
            self.dt
        except AttributeError:
            raise AttributeError("Time-step (dt) was not specified")

        try:
            self.t
        except AttributeError:
            warnings.warn("initial time (t) was not specified, thus it is set to zero.")
            self.t = 0.

        try:
            self.abs_boundary
        except AttributeError:
            warnings.warn("absorbing boundary (abs_boundary) was not specified, thus it is turned off")
            self.abs_boundary = "1."

        ########################################################################################
        #
        #   Initialize Fourier transform for efficient calculations
        #
        ########################################################################################

        # Load FFTW wisdom if saved before
        try:
            with open('fftw_wisdom', 'rb') as f:
                pyfftw.import_wisdom(pickle.load(f))

            print("\nFFTW wisdom has been loaded\n")
        except IOError:
            pass

        # allocate the array for density matrix
        self.rho = pyfftw.empty_aligned((self.X_gridDIM, self.X_gridDIM), dtype=np.complex)

        #  FFTW settings to achive good performace. For details see
        # https://hgomersall.github.io/pyFFTW/pyfftw/pyfftw.html#pyfftw.FFTW
        fftw_flags = ('FFTW_MEASURE','FFTW_DESTROY_INPUT')

        # how many threads to use for parallelized calculation of FFT.
        # Use the same number of threads as in numexpr
        fftw_nthreads = ne.nthreads

        # Create plan to pefrom FFT over the zeroth axis. It is equivalent to
        #   fftpack.fft(self.rho, axis=0, overwrite_x=True)
        self.rho_fft_ax0 = pyfftw.FFTW(
            self.rho, self.rho,
            axes=(0,),
            direction='FFTW_FORWARD',
            flags=fftw_flags,
            threads=fftw_nthreads
        )

        self.rho_fft_ax1 = pyfftw.FFTW(
            self.rho, self.rho,
            axes=(1,),
            direction='FFTW_FORWARD',
            flags=fftw_flags,
            threads=fftw_nthreads
        )

        self.rho_ifft_ax0 = pyfftw.FFTW(
            self.rho, self.rho,
            axes=(0,),
            direction='FFTW_BACKWARD',
            flags=fftw_flags,
            threads=fftw_nthreads
        )

        self.rho_ifft_ax1 = pyfftw.FFTW(
            self.rho, self.rho,
            axes=(1,),
            direction='FFTW_BACKWARD',
            flags=fftw_flags,
            threads=fftw_nthreads
        )

        # Save FFTW wisdom
        with open('fftw_wisdom', 'wb') as f:
            pickle.dump(pyfftw.export_wisdom(), f)

        ########################################################################################

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

        # allocate an axillary array needed for propagation
        self.expV = np.zeros_like(self.rho)

        # construct the coordinate dependent phase containing the dissipator as well as coherent propagator
        phase_X = "1j * (({V_X_prime}) - ({V_X})) " \
                  "+ ({A_X}) * conj({A_X_prime}) - 0.5 * abs({A_X}) ** 2 - 0.5 * abs({A_X_prime}) ** 2".format(
                V_X_prime=self.V.format(X="X_prime"),
                V_X=self.V.format(X="X"),
                A_X_prime=self.A.format(X="X_prime"),
                A_X=self.A.format(X="X"),
        )

        # numexpr code to calculate (-)**(k + k_prime) * exp(0.5 * dt * F)
        self.code_expV = "(%s) * (%s) * (-1) ** (k + k_prime) * exp(0.5 * dt * (%s))" % (
            self.abs_boundary.format(X="X"), self.abs_boundary.format(X="X_prime"), phase_X
        )

        # construct the coordinate dependent phase containing the dissipator as well as coherent propagator
        phase_P = "1j * (({K_P_prime}) - ({K_P})) " \
                  "+ ({B_P}) * conj({B_P_prime}) - 0.5 * abs({B_P}) ** 2 - 0.5 * abs({B_P_prime}) ** 2".format(
                K_P_prime=self.K.format(P="P_prime"),
                K_P=self.K.format(P="P"),
                B_P_prime=self.B.format(P="P_prime"),
                B_P=self.B.format(P="P"),
        )

        # numexpr code to calculate rho * exp(1j * dt * K)
        self.code_expK = "rho * exp(dt * (%s))" % phase_P


        # Check whether the necessary terms are specified to calculate the first-order Ehrenfest theorems
        try:
            # Allocate a copy of the wavefunction for storing the density matrix in the momentum representation
            self.rho_p = pyfftw.empty_aligned(self.rho.shape, dtype=self.rho.dtype)

            # Create FFT plans to operate on self.rho_p
            self.rho_p_fft_ax0 = pyfftw.FFTW(
                self.rho_p, self.rho_p,
                axes=(0,),
                direction='FFTW_FORWARD',
                flags=fftw_flags,
                threads=fftw_nthreads
            )

            self.rho_p_ifft_ax1 = pyfftw.FFTW(
                self.rho_p, self.rho_p,
                axes=(1,),
                direction='FFTW_BACKWARD',
                flags=fftw_flags,
                threads=fftw_nthreads
            )

            # numexpr codes to calculate the First Ehrenfest theorems
            self.code_V_average = "sum((%s) * density)" % self.V.format(X="X")
            self.code_K_average = "sum((%s) * density)" % self.K.format(P="P")

            self.code_X_average = "sum(X * density)"
            self.code_P_average = "sum(P * density)"

            self.code_P_average_RHS = "sum((-(%s) + (%s)) * density)" % (
                self.diff_V.format(X="X"), self.RHS_P_A.format(X="X")
            )

            self.code_X_average_RHS = "sum(((%s) + (%s)) * density)" % (
                self.diff_K.format(P="P"), self.RHS_X_B.format(P="P")
            )

            # Lists where the expectation values of X and P
            self.X_average = []
            self.P_average = []

            # Lists where the right hand sides of the Ehrenfest theorems for X and P
            self.X_average_RHS = []
            self.P_average_RHS = []

            # List where the expectation value of the Hamiltonian will be calculated
            self.hamiltonian_average = []

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
        for _ in range(time_steps):
            # make a half step in time
            self.t += 0.5 * self.dt

            # efficiently calculate expV
            ne.evaluate(self.code_expV, local_dict=vars(self), out=self.expV)
            self.rho *= self.expV

            # going to the momentum representation
            self.rho_fft_ax0()
            self.rho_ifft_ax1()

            ne.evaluate(self.code_expK, local_dict=vars(self), out=self.rho)

            # going back to the coordinate representation
            self.rho_ifft_ax0()
            self.rho_fft_ax1()

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
                self.dX * ne.evaluate(self.code_P_average_RHS, local_dict=vars(self)).real
            )

            # save the potential energy
            self.hamiltonian_average.append(
                self.dX * ne.evaluate(self.code_V_average, local_dict=vars(self)).real
            )

            # calculate density in the momentum representation
            ne.evaluate("(-1) ** (k + k_prime) * rho", local_dict=vars(self), out=self.rho_p)
            self.rho_p_fft_ax0()
            self.rho_p_ifft_ax1()

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

    def get_purity(self):
        """
        :return: The purity of the density matrix (self.rho)
        """
        return ne.evaluate("sum(abs(rho) ** 2)", local_dict=vars(self)) * self.dX ** 2

    def get_energy(self):
        """
        :return: the expectation value of the Hamiltonian with respect to the state (self.rho)
        """
        # extract the coordinate density and give it the shape of self.X
        self.density = self.rho.diagonal().reshape(self.X.shape)

        H = self.dX * ne.evaluate(self.code_V_average, local_dict=vars(self)).real

        # calculate density in the momentum representation
        ne.evaluate("(-1) ** (k + k_prime) * rho", local_dict=vars(self), out=self.rho_p)
        self.rho_p_fft_ax0()
        self.rho_p_ifft_ax1()

        # normalize
        self.rho_p /= self.rho_p.trace() * self.dP

        self.density = self.rho_p.diagonal().reshape(self.P.shape)

        # add the kinetic energy to get the hamiltonian
        H += self.dP * ne.evaluate(self.code_K_average, local_dict=vars(self)).real

        return H

    def __setattr__(self, key, value):
        """
        Make sure that attributes are not overwritten except some specific variables
        """
        assert key in {"t", "rho", "rho_p", "density"} or key not in vars(self), \
            "Attribute (%s) already belongs to the instance of this class. " \
            "You may want to use another name for the attribute." % key

        super(DensityMatrix, self).__setattr__(key, value)

##############################################################################
#
#   Run some examples
#
##############################################################################

if __name__ == '__main__':

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

                # dissipators
                alpha=np.random.uniform(0.1, 0.3),
                beta=np.random.uniform(-0.02, 0.02),

                R=np.random.uniform(-10., 10.),
                S=np.random.uniform(-1., 1.),

                # coordinate dependant dissipator and its correction to the Ehrenfest theorem
                A="R * exp(-1j * alpha * 0.25 * {X} ** 4 / R ** 2)",
                RHS_P_A="-alpha * {X} ** 3",

                # momentum dependant dissipator and its correction to the Ehrenfest theorem
                B="S * {P} * exp(-1j * beta * 0.5 * {P} ** 2 / S ** 2)",
                RHS_X_B="beta * {P} ** 3",

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
            self.img.set_array([[]])
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