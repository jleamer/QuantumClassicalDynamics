"""
Demonstration of the imaginary time propagation to obtain the ground and excited state.
"""

import numpy as np
from split_op_schrodinger1D import SplitOpSchrodinger1D # Previously developed propagator
from scipy import linalg # Linear algebra for dense matrix
import matplotlib.pyplot as plt # Plotting facility


# Let's find the ground state of argon atom with the single active electron approximation

# specify parameters separately
# note the time step was not included
atom_params = dict(
    X_gridDIM=1024,
    X_amplitude=10.,
    K="0.5 * P ** 2", #lambda p: 0.5*p**2,
    V="-1. / sqrt(X ** 2 + 1.37)",#"#lambda x: -1./np.sqrt(x**2 + 1.37) # the soft core Coulomb potential
)

# the amplitude of time step
dt = 0.005

# initialize the imaginary time propagator
atom_sys = SplitOpSchrodinger1D(dt=-1j * dt, **atom_params)

# make a guess for the ground state
# it is a good idea to make it nodeless
atom_sys.set_wavefunction("exp(-X ** 2)")

##################################################################################################
#
# Ground state calculations
#
##################################################################################################

# perform the imaginary time propagation (the longer, the better)
ground_state = atom_sys.propagate(4000)

# get "exact" ground state by diagonalizing the MUB hamiltonian
from mub_qhamiltonian import MUBQHamiltonian
ground_state_exact = MUBQHamiltonian(**atom_params).get_eigenstate(0)


# check the quality of ground state by propagating it in the real time
ground_state_after = SplitOpSchrodinger1D(dt=dt, **atom_params) \
                        .set_wavefunction(ground_state) \
                        .propagate(3000)

# during the real time propagation the wave function may pick up the phase
ground_state_after = np.abs(ground_state_after)

plt.title("Ground state calculation for argon within the single active electron approximation")
# to plot potential energy
# plt.plot(atom_sys.X_range, atom_sys.V(atom_sys.X_range), label='potential energy')
plt.semilogy(atom_sys.X, ground_state, 'r-', label='ground state')
plt.semilogy(atom_sys.X, ground_state_after, 'b--', label='ground state after propagation')
plt.semilogy(atom_sys.X, ground_state_exact, 'g-.', label='ground state via diagonalization')
plt.xlabel("$x$ (a.u.)")
plt.legend(loc='lower center')
plt.show()

##################################################################################################
#
# First excited state calculations
#
##################################################################################################

# Set the initial guess for the first excited state
# anything with a node should do the job
atom_sys.set_wavefunction(
    atom_sys.X * ground_state
)

for _ in xrange(4000):
    # take a step in the imaginary time propagation
    excited1_state = atom_sys.propagate()

    # project out the ground state
    excited1_state -= ground_state * np.dot(ground_state, excited1_state) * atom_sys.dX

# Finally, normalize the wavefunction
excited1_state /= linalg.norm(excited1_state) * np.sqrt(atom_sys.dX)

# check the quality of the first excited state by propagating it in the real time
excited1_state_after = SplitOpSchrodinger1D(dt=dt, **atom_params) \
                        .set_wavefunction(excited1_state) \
                        .propagate(3000)

# get the first excited state by diagonalizing the MUB hamiltonian
excited1_state_exact = MUBQHamiltonian(**atom_params).get_eigenstate(1)

plt.title("First eited state calculation of argon")
plt.semilogy(atom_sys.X, np.abs(excited1_state), 'r-', label='exited state')
plt.semilogy(atom_sys.X, np.abs(excited1_state_after), 'b--', label='excited state after propagation')
plt.semilogy(atom_sys.X, np.abs(excited1_state_exact), 'g-.', label='excited state via diagonalization')
plt.ylim([1e-4, 1e0])
plt.xlabel("$x$ (a.u.)")
plt.legend(loc='lower center')
plt.show()