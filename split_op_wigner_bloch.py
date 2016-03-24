import numpy as np

# numpy.fft has better implementation of real fourier transform
# necessary for real split operator propagator
from numpy import fft


class SplitOpWignerBloch:
    """
    Find the Wigner function of the Maxwell-Gibbs canonical state [rho = exp(-H/kT)]
    by split-operator propagation of the Bloch equation in phase space.
    The Hamiltonian should be of the form H = K(p) + V(x).

    This implementation follows split_op_wigner_moyal.py
    """
    def __init__(self, **kwargs):
        """
        The following parameters must be specified
            X_gridDIM - the coordinate grid size
            X_amplitude - maximum value of the coordinates
            P_gridDIM - the momentum grid size
            P_amplitude - maximum value of the momentum
            V(x) - potential energy (as a function)
            K(p) - momentum dependent part of the hamiltonian (as a function)
            kT  - the temperature (if kT = 0, then the ground state Wigner function will be obtained.)
            dbeta - (optional) 1/kT step size
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
            self.kT
        except AttributeError:
            raise AttributeError("Temperature (kT) was not specified")

        if self.kT > 0:
            try:
                self.dbeta
            except AttributeError:
                # if dbeta is not defined, just choose some value
                self.dbeta = 0.01

            # get number of dbeta steps to reach the desired Gibbs state
            self.num_beta_steps = 1. / (self.kT*self.dbeta)

            if round(self.num_beta_steps) <> self.num_beta_steps:
                # Changing self.dbeta so that num_beta_steps is an exact integer
                self.num_beta_steps = round(self.num_beta_steps)
                self.dbeta = 1. / (self.kT*self.num_beta_steps)

            self.num_beta_steps = int(self.num_beta_steps)
        else:
            raise NotImplemented("The calculation of the ground state Wigner function has not been implemnted")

        ###################################################################################
        #
        #   Generate grids
        #
        ###################################################################################

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

        ###################################################################################
        #
        # Pre-calculate exponents used for the split operator propagation
        #
        ###################################################################################

        # Get the sum of the potential energy contributions
        self.expV = self.V(self.X - 0.5*self.Theta) + self.V(self.X + 0.5*self.Theta)
        self.expV *= -0.5*self.dbeta

        # Make sure that the largest value is zero
        self.expV -= self.expV.max()
        # such that the following exponent is never larger then one
        np.exp(self.expV, out=self.expV)

        # Get the sum of the kinetic energy contributions
        self.expK = self.K(self.P + 0.5*self.Lambda) + self.K(self.P - 0.5*self.Lambda)
        self.expK *= -0.5*self.dbeta

        # Make sure that the largest value is zero
        self.expK -= self.expK.max()
        # such that the following exponent is never larger then one
        np.exp(self.expK, out=self.expK)

    def single_step_propagation(self):
        """
        Perform single step propagation. The final Wigner function is not normalized.
        :return: self.wignerfunction
        """
        # p x -> theta x
        self.wignerfunction = fft.rfft(self.wignerfunction, axis=0)
        self.wignerfunction *= self.expV

        # theta x  ->  p x
        self.wignerfunction = fft.irfft(self.wignerfunction, axis=0)

        # p x  ->  p lambda
        self.wignerfunction = fft.rfft(self.wignerfunction, axis=1)
        self.wignerfunction *= self.expK

        # p lambda  ->  p x
        self.wignerfunction = fft.irfft(self.wignerfunction, axis=1)

        return self.wignerfunction

    def get_gibbs_state(self):
        """
        Calculate the Boltzmann-Gibbs state and save it in self.gibbs_state
        :return:
        """
        # pre-compute the volume element in phase space
        dXdP = self.dX * self.dP

        # Initialize the Wigner function as the infinite temperature Gibbs state
        self.wignerfunction = 0.*self.X + 0.*self.P + 1.

        for _ in xrange(self.num_beta_steps):
            # propagate by dbeta
            self.single_step_propagation()

            # normalization
            self.wignerfunction /= self.wignerfunction.sum() * dXdP

        return self.wignerfunction

##############################################################################
#
#   Run some examples
#
##############################################################################

if __name__ == '__main__':

    print(SplitOpWignerBloch.__doc__)

    import matplotlib.pyplot as plt

    qsys_params = dict(
        t=0.,
        dt=0.005,

        X_gridDIM=256,
        X_amplitude=6.,

        P_gridDIM=256,
        P_amplitude=9.,

        kT=0.1,

        # kinetic energy part of the hamiltonian
        K=lambda p: 0.5*p**2,

        # potential energy part of the hamiltonian
        V=lambda x: 0.5*x**4
    )

    print("Calculating the Gibbs state...")
    gibbs_state = SplitOpWignerBloch(**qsys_params).get_gibbs_state()

    # Propagate this state via the Wigner-Moyal equation
    from split_op_wigner_moyal import SplitOpWignerMoyal

    print("Check that the obtained Gibbs state is stationary under the Wigner-Moyal propagation...")
    propagator = SplitOpWignerMoyal(**qsys_params)
    final_state = propagator.set_wignerfunction(gibbs_state).propagate(3000)

    ##############################################################################
    #
    #   Plot the results
    #
    ##############################################################################

    from wigner_normalize import WignerSymLogNorm

    # save common plotting parameters
    plot_params = dict(
        origin='lower',
        extent=[propagator.X.min(), propagator.X.max(), propagator.P.min(), propagator.P.max()],
        cmap='seismic',
        # make a logarithmic color plot (see, e.g., http://matplotlib.org/users/colormapnorms.html)
        norm=WignerSymLogNorm(linthresh=1e-13, vmin=-0.01, vmax=0.1)
    )
    plt.subplot(121)

    plt.title("The Gibbs state (initial state)")
    plt.imshow(gibbs_state, **plot_params)
    plt.colorbar()
    plt.xlabel('$x$ (a.u.)')
    plt.ylabel('$p$ (a.u.)')

    plt.subplot(122)

    plt.title("The Gibbs state after propagation")
    plt.imshow(final_state, **plot_params)
    plt.colorbar()
    plt.xlabel('$x$ (a.u.)')
    plt.ylabel('$p$ (a.u.)')

    plt.show()