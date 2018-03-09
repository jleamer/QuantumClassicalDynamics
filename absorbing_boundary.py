"""
This script demonstrates the utility of the absorbing boundary,
which is an imaginary addition to the potential, to simulate the evolution of
bounded potentials (e.g., such as a soft-core Coulomb potential, a Morse potential that has a bounded arm).
"""
from imag_time_propagation import ImgTimePropagation, np, fftpack
from scipy.signal import blackman
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm # enable log color plot

# Define the strength of the laser field:
# Always add an envelop to the laser field to avoid all sorts of artifacts
# we use a sin**2 envelope, which resembles the Blackman filter
E_laser = "F * sin(omega_laser * t) * sin(pi * t / t_final) ** 2"

# specify parameters separately
sys_params = dict(
    dt=0.01,
    X_gridDIM=1024,
    X_amplitude=100.,

    # laser field frequency
    omega_laser = 0.06,

    # amplitude of the laser field strength
    F = 0.04,

    # the initial time
    t = 0,

    # the final time of propagation (= 8 periods of laser oscillations)
    t_final = 8 * 2. * np.pi / 0.06,

    K="0.5 * P ** 2",
    diff_K="P",

    # Define the  potential energy as a sum of the soft core Columb potential
    # and the laser field interaction in the dipole approximation
    V="-1. / sqrt(X ** 2 + 1.37) + X * ({})".format(E_laser),

    # the derivative of the potential to calculate Ehrenfest
    diff_V="X * (X ** 2 + 1.37) ** (-1.5) + ({})".format(E_laser),

    # add constant pi
    pi=np.pi,
)

#########################################################################
#
#   Define functions for testing and visualizing
#
#########################################################################


def test_propagation(sys):
    """
    Run tests for the specified propagators and plot the probability density
    of the time dependent propagation
    :param sys: class that propagates
    """
    # Set the ground state wavefunction as the initial condition
    sys.get_stationary_states(1)
    sys.set_wavefunction(sys.stationary_states[0])

    iterations = 748
    steps = int(round(sys.t_final / sys.dt / iterations))

    # display the propagator
    plt.imshow(
        [np.abs(sys.propagate(steps)) ** 2 for _ in range(iterations)],
        origin='lower',
        norm=LogNorm(vmin=1e-12, vmax=0.1),
        aspect=0.4, # image aspect ratio
        extent=[sys.X.min(), sys.X.max(), 0., sys.t]
    )
    plt.xlabel('coordinate $x$ (a.u.)')
    plt.ylabel('time $t$ (a.u.)')
    plt.colorbar()


def test_Ehrenfest1(sys):
    """
    Test the first Ehenfest theorem for the specified quantum system
    """
    times = sys.dt * np.arange(len(sys.X_average))

    dX_dt = np.gradient(sys.X_average, sys.dt)

    print("{:.2e}".format(np.linalg.norm(dX_dt - sys.X_average_RHS)))

    plt.plot(times, dX_dt, '-r', label='$d\\langle\\hat{x}\\rangle / dt$')
    plt.plot(times, sys.X_average_RHS, '--b', label='$\\langle\\hat{p}\\rangle$')
    plt.legend()
    plt.ylabel('momentum')
    plt.xlabel('time $t$ (a.u.)')


def test_Ehrenfest2(sys):
    """
    Test the second Ehenfest theorem for the specified quantum system
    """
    times = sys.dt * np.arange(len(sys.P_average))

    dP_dt = np.gradient(sys.P_average, sys.dt)

    print("{:.2e}".format(np.linalg.norm(dP_dt - sys.P_average_RHS)))

    plt.plot(times, dP_dt, '-r', label='$d\\langle\\hat{p}\\rangle / dt$')
    plt.plot(times, sys.P_average_RHS, '--b', label='$\\langle -U\'(\\hat{x})\\rangle$')
    plt.legend()
    plt.ylabel('force')
    plt.xlabel('time $t$ (a.u.)')

#########################################################################
#
#   Declare the propagators
#
#########################################################################


sys_no_abs_boundary = ImgTimePropagation(**sys_params)
sys_with_abs_boundary = ImgTimePropagation(
    abs_boundary="sin(0.5 * pi * (X + X_amplitude) / X_amplitude) ** (0.05 * dt)",
    **sys_params
)

#########################################################################
#
# Test propagation in the context of High Harmonic Generation
#
#########################################################################

plt.subplot(121)
plt.title("No absorbing boundary, $|\\Psi(x, t)|^2$")
test_propagation(sys_no_abs_boundary)

plt.subplot(122)
plt.title("Absorbing boundary, $|\\Psi(x, t)|^2$")
test_propagation(sys_with_abs_boundary)

plt.show()

#########################################################################

plt.subplot(221)
plt.title("No absorbing boundary")
print("\nNo absorbing boundary error in the first Ehrenfest relation: ")
test_Ehrenfest1(sys_no_abs_boundary)

plt.subplot(222)
plt.title("No absorbing boundary")
print("\nNo absorbing boundary error in the second Ehrenfest relation: ")
test_Ehrenfest2(sys_no_abs_boundary)

plt.subplot(223)
plt.title("Absorbing boundary")
print("\nWith absorbing boundary error in the first Ehrenfest relation: ")
test_Ehrenfest1(sys_with_abs_boundary)

plt.subplot(224)
plt.title("Absorbing boundary")
print("\nWith absorbing boundary error in the second Ehrenfest relation: ")
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
    k = np.arange(N)

    # frequency range
    omegas = (k - N / 2) * np.pi / (0.5 * sys.t)

    # spectra of the
    spectrum = np.abs(
        # used windows fourier transform to calculate the spectra
        # rhttp://docs.scipy.org/doc/scipy/reference/tutorial/fftpack.html
        fftpack.fft((-1) ** k * blackman(N) * sys.P_average_RHS)
    ) ** 2
    spectrum /= spectrum.max()

    plt.semilogy(omegas / sys.omega_laser, spectrum)
    plt.ylabel('spectrum (arbitrary units)')
    plt.xlabel('frequency / $\\omega_L$')
    plt.xlim([0, 45.])
    plt.ylim([1e-15, 1.])

plt.subplot(121)
plt.title("No absorbing boundary High Harmonic Generation Spectrum")
plot_spectrum(sys_no_abs_boundary)

plt.subplot(122)
plt.title("With absorbing boundary High Harmonic Generation Spectrum")
plot_spectrum(sys_with_abs_boundary)

plt.show()