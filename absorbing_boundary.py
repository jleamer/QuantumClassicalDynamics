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
from scipy import fftpack


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

        # define the parameter to control smoothness of the absorbing boundary (see below),
        # if the user has not done so
        try:
            self.alpha
        except AttributeError:
            self.alpha = 1

        # Cache the absorbing boundary
        self._abs_boundary = np.abs(blackman(self.X_gridDIM))**abs(self.dt*self.alpha)

    def get_expV(self, t):
        # Overload the methid from SplitOpSchrodinger1D

        # Apply the absorbing boundary
        # the following line is equivalent to
        #   V(x) -> V(x) + 1j*self.alpha*np.log(blackman(x))
        return SplitOpSchrodinger1D.get_expV(self, t) * self._abs_boundary

#########################################################################


def test(Propagator, params):
    """
    Run tests for the specified propagators and plot the probability density
    of the time dependent propagation
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
        np.exp(-np.abs(sys.X_range))
    )

    # turn off the Ehrenfets theorem calculations for ground state
    sys.isEhrenfest = False

    # get the ground state using the imagenary time propagation
    ground_state = sys.propagate(6000)

    # set the real time propagator for the Schrodinger equation
    sys = Propagator(**params).set_wavefunction(ground_state)

    # display the propagator
    plt.imshow(
        [np.abs(sys.propagate(105))**2 for _ in xrange(750)],
        origin='lower',
        norm=norm,
        aspect=0.4, # image aspect ratio
        extent=[sys.X_range.min(), sys.X_range.max(), 0., sys.t]
    )
    plt.xlabel('coordinate $x$ (a.u.)')
    plt.ylabel('time $t$ (a.u.)')
    plt.colorbar()

    # return propagator
    return sys

#########################################################################

# specify parameters separately
# note the time step was not included
params = dict(
    t=0.,
    dt=0.01,
    X_gridDIM=1024,
    X_amplitude=100.,

    # parameter controling smoothness of the absorbing boundary, if the latter is used
    alpha = 0.02,

    K=lambda p: 0.5*p**2,
    diff_K=lambda p: p,

    # the soft core Coulomb potential
    V=lambda x: -1./np.sqrt(x**2 + 1.37),

    # the derivative of the potential to calculate Ehrenfest
    diff_V=lambda x: x*np.power(x**2 + 1.37, -1.5),
)

# This is how to make logarithmic color plot
norm=LogNorm(vmin=1e-12, vmax=0.1)

#########################################################################
#
# Test propagation under time independent potential
#
#########################################################################

plt.subplot(121)
plt.title("No absorbing boundary time independent potential, $|\\Psi(x, t)|^2$")
test(SplitOpSchrodinger1D, params)

plt.subplot(122)
plt.title("Absorbing boundary time independent potential, $|\\Psi(x, t)|^2$")
test(SplitOpSchrodinger1DAbsBoundary, params)


plt.show()

#########################################################################
#
# Test propagation under time dependent potential
#
#########################################################################

# frequency of laser pulse
omega_laser = 0.06

# period of laser pulse
T = 2*np.pi/omega_laser

# Define laser field:
# Always add an envelop to the laser field to avoid all sorts of artifacts
# we use a sin**2 envelope, which resembles the Blackman filter
E = lambda t: 0.04*np.sin(omega_laser*t) * np.sin(np.pi*t/(8*T))**2

# add some laser field to the potential
V = params['V']
params['V'] = lambda x, t: V(x) + x*E(t)

# update the derivative of the potential correspondingly
diff_V = params['diff_V']
params['diff_V'] = lambda x, t: diff_V(x) + E(t)

plt.subplot(121)
plt.title("No absorbing boundary time DEPENDENT potential, $|\\Psi(x, t)|^2$")
sys_no_abs_boundary = test(SplitOpSchrodinger1D, params)

plt.subplot(122)

plt.title("Absorbing boundary time dependent potential, $|\\Psi(x, t)|^2$")
sys_with_abs_boundary = test(SplitOpSchrodinger1DAbsBoundary, params)

plt.show()

#########################################################################

def test_Ehrenfest1(sys):
    """
    Test the first Ehenfest theorem for the specified quantum system
    """
    times = sys.dt * np.arange(len(sys.X_average))

    plt.plot(times, np.gradient(sys.X_average, sys.dt), '-r', label='$d\\langle\\hat{x}\\rangle / dt$')
    plt.plot(times, sys.X_average_RHS, '--b', label='$\\langle\\hat{p}\\rangle$')
    plt.legend()
    plt.ylabel('momentum')
    plt.xlabel('time $t$ (a.u.)')

def test_Ehrenfest2(sys):
    """
    Test the second Ehenfest theorem for the specified quantum system
    """
    times = sys.dt * np.arange(len(sys.X_average))

    plt.plot(times, np.gradient(sys.P_average, sys.dt), '-r', label='$d\\langle\\hat{p}\\rangle / dt$')
    plt.plot(times, sys.P_average_RHS, '--b', label='$\\langle -U\'(\\hat{x})\\rangle$')
    plt.legend()
    plt.ylabel('force')
    plt.xlabel('time $t$ (a.u.)')


plt.subplot(221)
plt.title("No absorbing boundary time DEPENDENT potential")
test_Ehrenfest1(sys_no_abs_boundary)

plt.subplot(222)
plt.title("No absorbing boundary time DEPENDENT potential")
test_Ehrenfest2(sys_no_abs_boundary)

plt.subplot(223)
plt.title("Absorbing boundary time DEPENDENT potential")
test_Ehrenfest1(sys_with_abs_boundary)

plt.subplot(224)
plt.title("Absorbing boundary time DEPENDENT potential")
test_Ehrenfest2(sys_with_abs_boundary)

plt.show()

#########################################################################

def plot_spectrum(sys):
    """
    Plot the High Harmonic Generation spectrum
    """
    # Power spectrum emitted is calculated using the Larmor formula
    #   (https://en.wikipedia.org/wiki/Larmor_formula)
    # which says that the power emitted is proportional to the square of the acceleration
    # i.e., the RHS of the second Ehrenfest theorem

    N = len(sys.P_average_RHS)
    omegas = fftpack.fftshift(fftpack.fftfreq(N, sys.dt/(2*np.pi))) / omega_laser
    spectrum = np.abs(fftpack.fftshift(fftpack.fft(sys.P_average_RHS)))**2
    spectrum /= spectrum.max()

    plt.semilogy(omegas, spectrum)
    plt.ylabel('spectrum')
    plt.xlabel('frequency / $\\omega_L$')
    plt.xlim([0, 40.])
    plt.ylim([1e-10, 1.])

plt.subplot(121)
plt.title("No absorbing boundary High Harmonic Generation Spectrum")
plot_spectrum(sys_no_abs_boundary)

plt.subplot(122)
plt.title("With absorbing boundary High Harmonic Generation Spectrum")
plot_spectrum(sys_with_abs_boundary)

plt.show()