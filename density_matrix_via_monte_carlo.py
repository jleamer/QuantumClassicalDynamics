"""
Compare the split operator density matrix propagator with the Monte Carlo wavefunction propagator
"""

from wavefunc_monte_carlo1D import WavefuncMonteCarloPoission, uniform, np, ne
from multiprocessing import Pool


def propagate_traj(args):
    """
    A function that propagates a single quantum trajectory
    :param params: dictionary of parameters to initialize the Monte Carlo propagator
    :param init_wavefunc:
    :param seed: the seed for random number generator
    :return: numpy.array contanning the final wavefunction
    """
    params, init_wavefunc, seed = args

    # Since there are many trajectories are run in parallel use only a single thread
    ne.set_num_threads(1)

    # Set the seed for random number generation to avoid the artifact described in
    #   https://stackoverflow.com/questions/24345637/why-doesnt-numpy-random-and-multiprocessing-play-nice
    # It is recommended that seeds be generate via the function get_seeds (see below)
    np.random.seed(seed)

    # initialize the propagator
    qsys = WavefuncMonteCarloPoission(**params).set_wavefunction(init_wavefunc)

    # propagate
    return qsys.propagate(params["ntsteps"])


def get_seeds(size):
    """
    Generate unique random seeds for subsequently seeding them into random number generators in multiprocessing simulations

    This utility is to avoid the following artifact:
        https://stackoverflow.com/questions/24345637/why-doesnt-numpy-random-and-multiprocessing-play-nice
    :param size: number of samples to generate
    :return: numpy.array of np.uint32
    """
    # Note that np.random.seed accepts 32 bit unsigned integers

    # get the maximum value of np.uint32 can take
    max_val = np.iinfo(np.uint32).max

    # A set of unique and random np.uint32
    seeds = set()

    # generate random numbers until we have sufficiently many nonrepeating numbers
    while len(seeds) < size:
        seeds.update(
            np.random.randint(max_val, size=size, dtype=np.uint32)
        )

    # make sure we do not return more numbers that we are asked for
    return np.fromiter(seeds, np.uint32, size)


if __name__ == '__main__':

    from split_op_denisty_matrix import DensityMatrix

    ####################################################################################
    #
    #   Set the parameters
    #
    ####################################################################################

    # parameters for the density matrix propagator
    rho_params = dict(
        t=0.,
        dt=0.01,
        X_gridDIM=256,
        X_amplitude=5.,

        # how many time steps to propagate
        ntsteps=500,

        # displacement of the initial state
        X0=uniform(-2., 2.),

        omega=uniform(1, 2),
        V="0.5 * (omega * {X})** 4",
        K="0.5 * {P} ** 2",

        gamma=uniform(1., 2.),
        A="gamma * {X}",
    )

    # parameters for the wave function (including the Monte Carlo propagator
    def apply_A(self):
        """
        Apply X onto the wavefunction
        :param self:
        :return: None
        """
        self.wavefunction *= self.X

    wave_params = rho_params.copy()
    wave_params.update({
        'V':rho_params['V'].format(X='X'),
        'K':rho_params['K'].format(P='P'),
        'apply_A':(apply_A,),
        'AdaggerA_X':("(gamma * X) ** 2",),
    })

    # set the initial states
    init_wave = "exp(-(X - X0) ** 2)"

    # the corresponding density matrix
    init_rho = "exp(-(X - X0) ** 2 -(X_prime - X0) ** 2)"

    ####################################################################################
    #
    #   Propagate the density matrix via split operator
    #
    ####################################################################################

    print("Density matrix propagator starts...\n")

    rho_prop = DensityMatrix(**rho_params).set_rho(init_rho)
    rho_prop.propagate(rho_prop.ntsteps)

    print(".. ended\n")

    ####################################################################################
    #
    #   Obtain the density matrix via the wave function Monte Carlo propagation
    #
    ####################################################################################

    print("Monte Carlo simulation starts...\n")

    # allocate memory for additional matrices
    monte_carlo_rho = np.zeros_like(rho_prop.rho)
    rho_traj = np.zeros_like(monte_carlo_rho) # monte_carlo_rho is average of rho_traj

    # the iterator to launch 1000 trajectories
    iter_trajs = ((wave_params, init_wave, seed) for seed in get_seeds(1000))

    # run each Monte Carlo trajectories on multiple cores
    with Pool() as pool:
        # index counting the trajectory needed to calculate the mean iteratively
        t = 0
        for psi in pool.imap_unordered(propagate_traj, iter_trajs, chunksize=100):

            # form the density matrix out of the wavefunctions
            np.outer(psi.conj(), psi, out=rho_traj)

            # Calculate the iterative mean following http://www.heikohoffmann.de/htmlthesis/node134.html
            rho_traj -= monte_carlo_rho
            rho_traj /= (t + 1)
            monte_carlo_rho += rho_traj

            # increment the trajectory counter
            t += 1

    print("...ended")

    ####################################################################################
    #
    #   Plot
    #
    ####################################################################################

    # Plotting facility
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    img_params = dict(
        extent=[rho_prop.X.min(), rho_prop.X.max(), rho_prop.X.min(), rho_prop.X.max()],
        origin='lower',
        norm=LogNorm(1e-4, 1.),
        cmap='jet',
    )

    plt.subplot(121)
    plt.title("$\\rho$ via the density matrix propagator")
    plt.imshow(np.abs(rho_prop.rho), **img_params)
    plt.xlabel("$x$ (a.u.)")
    plt.ylabel("$x$ (a.u.)")
    plt.colorbar()

    plt.subplot(122)
    plt.title("$\\rho$ via Monte Carlo")
    plt.imshow(np.abs(monte_carlo_rho), **img_params)
    plt.xlabel("$x$ (a.u.)")
    plt.ylabel("$x$ (a.u.)")
    plt.colorbar()

    plt.show()