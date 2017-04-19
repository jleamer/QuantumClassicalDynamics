import matplotlib.pyplot as plt
from mub_qhamiltonian import MUBQHamiltonian, np

__doc__ = """
Band structure calculation for a 1D quantum system
"""

# Parameters of systems to reproduce Fig. 1 from M. Wu et al. Phys. Rev A 91, 043839 (2015)
sys_params = dict(
    X_gridDIM=256,

    # the lattice constant is 2 * X_amplitude
    X_amplitude=4.,

    # lattice height
    V0=0.37,

    # the kinetic energy (the value of quasimomentum will be assigned bellow)
    K="0.5 * (P + kq) ** 2",

    # Exactly solvable Mathieu-type periodic system
    V="-V0 * (1 + cos(pi * X / X_amplitude))",

    pi=np.pi,
)

# range of quasimomenta
k_quasimomenta = np.pi / sys_params["X_amplitude"] * np.linspace(-0.5, 0.5, 100)

# band structure
bands = np.array([
    MUBQHamiltonian(kq=kq, **sys_params).get_energy(slice(0, 4)) for kq in k_quasimomenta
])

# Conversion factor of atomic units to eV
au2eV = 27.

# plot bands
plt.title("Band structure of 1D (Mathieu-type) solid state system\nReproduced Fig. 1 from M. Wu et al. Phys. Rev A 91, 043839 (2015)")

for E in bands.T:
    plt.plot(sys_params["X_amplitude"] / np.pi * k_quasimomenta, au2eV * E)

plt.xlabel("$k$ (units of $2\pi/ a_0$)")
plt.ylabel('$\\varepsilon(k)$ (eV)')
plt.show()