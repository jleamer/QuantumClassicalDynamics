"""
Demonstration of the imaginary time propagation to obtain the ground and excited state.
"""
import numpy as np
from split_op_schrodinger1D import SplitOpSchrodinger1D # import previously developed propagator
from scipy import linalg # Linear algebra for dense matrix
import matplotlib.pyplot as plt # Plotting facility


# Let's find the ground state of argon atom with the single active electron approximation

# specify parameters separately
# note the time step was not included
atom_params = dict(
    X_gridDIM=1024,
    X_amplitude=10.,
    K=lambda p: 0.5*p**2,
    V=lambda x: -1./np.sqrt(x**2 + 1.37) # the soft core Coulomb potential
)

# the amplitude of time step
dt = 0.005

# initialize the imaginary time propagator
atom_sys = SplitOpSchrodinger1D(dt=-1j*dt, **atom_params)

# make a guess for the ground state
# it is a good idea to make it nodeless
atom_sys.set_wavefunction(
    np.exp(-atom_sys.X_range**2)
)

# perform the imaginary time propagation (the longer, the better)
ground_state = atom_sys.propagate(4000)

print(linalg.norm(ground_state.imag))

# check the quality of ground state by propagating it in the real time
ground_state_after = SplitOpSchrodinger1D(dt=dt, **atom_params) \
                        .set_wavefunction(ground_state) \
                        .propagate(3000)

# during the real time propagation the wave function may pick up the phase
ground_state_after = np.abs(ground_state_after)

# to plot potential energy
# plt.plot(atom_sys.X_range, atom_sys.V(atom_sys.X_range), label='potential energy')
plt.semilogy(atom_sys.X_range, ground_state, 'r-', label='ground state')
plt.semilogy(atom_sys.X_range, ground_state_after, 'b--', label='ground state after propagation')
plt.legend()
plt.show()








