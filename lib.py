import numpy as np
import matplotlib.pyplot as plt


hbar = 1.05457e-34  # [J/s] - Reduced Planck's constant
me = 9.10938e-31  # [kg] - Electron mass
intervalLength = 1.  # [m] - Length of interval
acc = 1000  # [1] - Amount of iterations
xvalues = np.linspace(0, intervalLength, acc)

dx = intervalLength / acc  # [m] - Increment length

def V1(x):
	return 0

def V2(x):
	return x**2

def Hamilton(V):
	"""

	:param V: function(x); the potential as a function of x
	:return: np.array (acc x acc); the matrix Hamilton operator :: Hamilton * [psi[1], ..., psi[acc]] = E * psi
	"""

	# Make matrix for V
	v = np.zeros((acc, acc))
	for i in range(acc):
		v[i][i] = V(xvalues[i])

	# Make the matrix itself
	H = -2 * np.eye(acc, dtype=float) + np.eye(acc, k=1, dtype=float) + np.eye(acc, k=-1, dtype=float)

	# Scale by the right amount
	H *= -(hbar)**2 / (2*me*(dx)**2)

	# Add potential
	H += v

	return H


H = Hamilton(V1)

E, psi = np.linalg.eig(H)
# psi[:, i] is the eigenvector corresponding to E[i]
