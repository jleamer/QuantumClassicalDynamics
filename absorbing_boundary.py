"""
This script demonstrates the utility of the absorbing boundary,
which is an imaginary addition to the potential, to simulate the evolution of
bounded potentials (e.g., such as a soft-core Coulomb potential, a Morse potential that has a bounded arm).
"""
import numpy as np
from split_op_schrodinger1D import SplitOpSchrodinger1D # Previously developed propagator
import matplotlib.pyplot as plt # Plotting facility
from matplotlib.colors import LogNorm # Color normalizer to perform logarithmic color plots
from scipy.signal import blackman # to be used as an absorbing boundary


class SplitOpSchrodinger1DAbsBoundary(SplitOpSchrodinger1D):
    """
    Split operator propagator with absorbing boundary condition added.

    As far as programming is concerned, this class is implemented using inheritance, see, e.g.
    https://en.wikipedia.org/wiki/Inheritance_%28object-oriented_programming%29
    """
    def __init__(self, **kwargs):
        # Call the parent's constructor
        SplitOpSchrodinger1D.__init__(self, **kwargs)
        # Note that the absorbing boundary can be any function
        # as long as the results of simulations do not depend on its choice, i.e.,
        # numerical results are converged
        #
        # Cache the absorbing boundary
        self._abs_boundary = np.abs(blackman(self.X_gridDIM))**abs(self.dt*0.05)

    def propagate(self, time_steps=1):
        # Overload the methid from SplitOpSchrodinger1D

        for _ in xrange(time_steps):
            # Apply the absorbing boundary
            # the following line is equivalent to
            #   V(x) -> V(x) +1j*np.log(blackman(x))
            self.wavefunction *= self._abs_boundary

            # having applied the absorbing boundary,
            # propagate by one step within the split operator propagator
            SplitOpSchrodinger1D.propagate(self)

        return self.wavefunction

#########################################################################


def test(Propagator, params):
    """
    Run tests of the propagators
    :param Propagator: class that propagates
    :param params: propagation parameters
    :return: list of densities obtained as result of the propagation
    """
    #########################################################################
    # Find the ground state using the imaginary time propagator
    img_time_params = params.copy()
    img_time_params['dt'] *= -1j

    # extract potential
    V = img_time_params['V']

    try:
        V(0., 0.)
        # potential is time dependent so let's fix time to zero
        import functools
        img_time_params['V'] = functools.partial(V, t=0.)
    except TypeError:
        # no need to modify it because the potential is time independent
        pass

    # Make a copy of parameters and do the Wick rotation
    sys = Propagator(**img_time_params)

    # make a guess for the ground state
    # it is a good idea to make it nodeless
    sys.set_wavefunction(
        np.exp(-sys.X_range**2)
    )
    ground_state = sys.propagate(4000)

    # return real propagator with the ground state as the initial condition
    return Propagator(**params).set_wavefunction(ground_state)

#########################################################################

# specify parameters separately
# note the time step was not included
params = dict(
    t=0.,
    dt=0.005,
    X_gridDIM=1024,
    X_amplitude=30.,
    K=lambda p: 0.5*p**2,
    diff_K=lambda p: p,
    # the soft core Coulomb potential
    V=lambda x: -1./np.sqrt(x**2 + 1.37),
    # the derivative of the potential to calculate Ehrenfest
    diff_V=lambda x: x*np.power(x**2 + 1.37, -1.5),
)

# This is how to make logarithmic color plot
norm=LogNorm(vmin=1e-10, vmax=0.1)

#########################################################################
#
# Test propagation under time independent potential
#
#########################################################################
"""
plt.subplot(121)
plt.title("No absorbing boundary time independent potential, $|\\Psi(x, t)|^2$")

sys = test(SplitOpSchrodinger1D, params)

plt.imshow(
    [np.abs(sys.propagate(2))**2 for _ in xrange(15000)],
    origin='lower',
    norm=norm,
    aspect=0.4, # image aspect ration
    extent=[sys.X_range.min(), sys.X_range.max(), 0., sys.t]
)
plt.xlabel('coordinate $x$ (a.u.)')
plt.ylabel('time $t$ (a.u.)')
plt.colorbar()

plt.subplot(122)
plt.title("Absorbing boundary time independent potential, $|\\Psi(x, t)|^2$")

sys = test(SplitOpSchrodinger1DAbsBoundary, params)

plt.imshow(
    [np.abs(sys.propagate(2))**2 for _ in xrange(15000)],
    origin='lower',
    norm=norm,
    aspect=0.4,
    extent=[sys.X_range.min(), sys.X_range.max(), 0., sys.t]
)
plt.xlabel('coordinate $x$ (a.u.)')
plt.ylabel('time $t$ (a.u.)')
plt.colorbar()

plt.show()
"""
#########################################################################
#
# Test propagation under time dependent potential
#
#########################################################################

# add some laser field to the potential
V = params['V']
params['V'] = lambda x, t: V(x) + 0.015*x*np.sin(0.05*t)

diff_V = params['diff_V']
params['diff_V'] = lambda x, t: diff_V(x) + 0.015*np.sin(0.05*t)

plt.subplot(121)
plt.title("No absorbing boundary time DEPENDENT potential, $|\\Psi(x, t)|^2$")

sys1 = test(SplitOpSchrodinger1D, params)

plt.imshow(
    [np.abs(sys1.propagate(20))**2 for _ in xrange(2000)],
    origin='lower',
    norm=norm,
    aspect=0.4, # image aspect ration
    extent=[sys1.X_range.min(), sys1.X_range.max(), 0., sys1.t]
)
plt.xlabel('coordinate $x$ (a.u.)')
plt.ylabel('time $t$ (a.u.)')
plt.colorbar()

plt.subplot(122)
plt.title("Absorbing boundary time DEPENDENT potential, $|\\Psi(x, t)|^2$")

sys2 = test(SplitOpSchrodinger1DAbsBoundary, params)

plt.imshow(
    [np.abs(sys2.propagate(20))**2 for _ in xrange(2000)],
    origin='lower',
    norm=norm,
    aspect=0.4,
    extent=[sys2.X_range.min(), sys2.X_range.max(), 0., sys2.t]
)
plt.xlabel('coordinate $x$ (a.u.)')
plt.ylabel('time $t$ (a.u.)')
plt.colorbar()

plt.show()

#########################################################################

times1 = sys1.dt * np.arange(len(sys1.X_average))

plt.subplot(221)
plt.title("No absorbing boundary time DEPENDENT potential")
plt.plot(times1, np.gradient(sys1.X_average, sys1.dt), '-r', label='$d\\langle\\hat{x}\\rangle / dt$')
plt.plot(times1, sys1.X_average_RHS, '--b', label='$\\langle\\hat{p}\\rangle$')
plt.legend()

plt.subplot(222)
plt.title("No absorbing boundary time DEPENDENT potential")
plt.plot(times1, np.gradient(sys1.P_average, sys1.dt), '-r', label='$d\\langle\\hat{p}\\rangle / dt$')
plt.plot(times1, sys1.P_average_RHS, '--b', label='$\\langle -U\'(\\hat{x})\\rangle$')
plt.legend()

times2 = sys2.dt * np.arange(len(sys2.X_average))

plt.subplot(223)
plt.title("Absorbing boundary time DEPENDENT potential")
plt.plot(times2, np.gradient(sys2.X_average, sys2.dt), '-r', label='$d\\langle\\hat{x}\\rangle / dt$')
plt.plot(times2, sys2.X_average_RHS, '--b', label='$\\langle\\hat{p}\\rangle$')
plt.legend()

plt.subplot(224)
plt.title("Absorbing boundary time DEPENDENT potential")
plt.plot(times2, np.gradient(sys2.P_average, sys2.dt), '-r', label='$d\\langle\\hat{p}\\rangle / dt$')
plt.plot(times2, sys2.P_average_RHS, '--b', label='$\\langle -U\'(\\hat{x})\\rangle$')
plt.legend()

plt.show()