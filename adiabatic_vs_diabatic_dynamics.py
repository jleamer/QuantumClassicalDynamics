"""
Demonstration of the difference between adiabatic (slow) and diabatic (fast) evolution.
In particular, we will illustrate the adiabatic theorem that states:
A physical system remains in its instantaneous eigenstate if a given perturbation is acting slowly enough.
"""
import functools
import numpy as np

import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation # tools for creating animation

from split_op_schrodinger1D import SplitOpSchrodinger1D


class DynamicVisualized:
    """
    We bundle creation of animation into this class
    """
    def __init__(self, fig):
        """
        Initialize all propagators and frame
        :param fig: matplotlib figure object
        """
        #################################################################
        #
        #   Initialize systems
        #
        #################################################################

        # common quantum system parameters
        self.qsys_params = dict(
            t=-7.,
            X_gridDIM=1024,
            X_amplitude=10.,
            K=lambda p: 0.5*p**2,
        )
        self.dt = 0.005

        # initialize adiabatic system (i.e, with slow time dependence)
        self.adiabatic_sys = SplitOpSchrodinger1D(
                V=lambda x, t: 0.01 * (1. - 0.95/(1. + np.exp(-0.2*t))) * x**4,
                dt=self.dt, **self.qsys_params
        )

        # initialize diabatic system (i.e, with fast time dependence)
        self.diabatic_sys = SplitOpSchrodinger1D(
                V=lambda x, t: 0.01 * (1. - 0.95/(1. + np.exp(-5.*t))) * x**4,
                dt=self.dt, **self.qsys_params
        )

        #################################################################
        #
        #   Initialize plotting facility
        #
        #################################################################

        self.fig = fig

        # plotting axis limits
        xmin = self.diabatic_sys.X_range.min()
        xmax = self.diabatic_sys.X_range.max()
        ymin = 1e-10
        ymax = 1e2

        # prepare for plotting diabatic dynamics
        adiabatic_ax = self.fig.add_subplot(121)
        adiabatic_ax.set_title("Adiabatic evolution")
        self.adiabatic_line, = adiabatic_ax.semilogy([], [], 'r-', label='exact wave function')
        self.ad_instant_eigns_line, = adiabatic_ax.semilogy([], [], 'b--', label='instantaneous eigenstate')
        adiabatic_ax.set_xlim(xmin, xmax)
        adiabatic_ax.set_ylim(ymin, ymax)
        adiabatic_ax.legend()
        adiabatic_ax.set_xlabel("$x$ (a.u.)")
        adiabatic_ax.set_ylabel("probability density")

        # prepare for plotting diabatic dynamics
        diabatic_ax = self.fig.add_subplot(122)
        diabatic_ax.set_title("Diabatic evolution")
        self.diabatic_line, = diabatic_ax.semilogy([], [], 'r-', label='exact wave function')
        self.d_instant_eigns_line, = diabatic_ax.semilogy([], [], 'b--', label='instantaneous eigenstate')
        diabatic_ax.set_xlim(xmin, xmax)
        diabatic_ax.set_ylim(ymin, ymax)
        diabatic_ax.legend()
        diabatic_ax.set_xlabel("$x$ (a.u.)")
        #diabatic_ax.set_ylabel("probability density")

    def get_instant_ground_state(self, qsys):
        """
        Return the instantaneous ground state
        :param qsys: object representing quantum system propagation
        :return:
        """
        # get the instantaneous potential energy by freezing time
        V = functools.partial(qsys.V, t=qsys.t)

        # perform the imaginary time propagation
        ground_state = SplitOpSchrodinger1D(V=V, dt=-1j*2*self.dt, **self.qsys_params) \
                        .set_wavefunction(np.exp(-qsys.X_range**2)) \
                        .propagate(4000)
        # from mub_qhamiltonian import MUBQHamiltonian
        # ground_state = MUBQHamiltonian(V=V, **self.qsys_params).get_eigenstate(0)
        return ground_state

    def empty_frame(self):
        """
        Reset make empty frame
        :return:
        """
        lines = (self.ad_instant_eigns_line, self.adiabatic_line, self.d_instant_eigns_line, self.diabatic_line)
        for L in lines:
            L.set_data([], [])
        return lines

    def __call__(self, frame_num):
        """
        Draw a new frame
        :param frame_num: current frame number
        :return: line objects
        """

        # find instantaneous ground states
        ad_ground_state = self.get_instant_ground_state(self.adiabatic_sys)
        self.ad_instant_eigns_line.set_data(self.adiabatic_sys.X_range, np.abs(ad_ground_state)**2)

        d_ground_state = self.get_instant_ground_state(self.diabatic_sys)
        self.d_instant_eigns_line.set_data(self.adiabatic_sys.X_range, np.abs(d_ground_state)**2)

        if frame_num == 0:
            # this is the first frame then, set the initial condition
            self.adiabatic_sys.set_wavefunction(ad_ground_state)
            self.diabatic_sys.set_wavefunction(d_ground_state)
        else:
            # propagate
            self.adiabatic_sys.propagate(100)
            self.diabatic_sys.propagate(100)

        # update plots
        self.adiabatic_line.set_data(self.adiabatic_sys.X_range, np.abs(self.adiabatic_sys.wavefunction)**2)
        self.diabatic_line.set_data(self.diabatic_sys.X_range, np.abs(self.diabatic_sys.wavefunction)**2)

        return self.ad_instant_eigns_line, self.adiabatic_line, self.d_instant_eigns_line, self.diabatic_line,

fig = plt.gcf()
visualizer = DynamicVisualized(fig)
animation = FuncAnimation(fig, visualizer, frames=np.arange(100),
                          init_func=visualizer.empty_frame, repeat=True, blit=True)
plt.show()

# Save animation into the file
# animation.save('adiabatic_theorem.mp4', metadata={'artist':'good PhD student'})