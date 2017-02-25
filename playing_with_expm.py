__doc__ = """
Let's find a matrix A for which the Taylor approximation gives a different result from the Pade approximation
"""

import numpy as np
from scipy.linalg import norm, expm, expm3

error = 0.

while error < 1e-3:
    A = np.random.rand(10, 10)
    error = norm(expm(A) - expm3(A))

print("\nHere is the matrix for which the Taylor approximation gives a different result from the Pade approximation:")
print(A)