import numpy as np

hbar = 1.05457e-34  # [Js] - Reduced Planck's constant
me = 9.10938e-31  # [kg] - Electron mass
eV = 1.619e-19  # [J] (1 electron Volt)

# Creating the space and the time-interval
N = 1000
dx = 1e-10
x = np.array([n * dx for n in range(N)])
x0 = x[N - 1] * 1 / 4  # Wave starts at 1/4 of the whole way


def V1(x):
	return 0


def V2(x):
	return 1e-3*(x - 2*x0) ** 2


def Hamilton(V):
	"""
	:param V: function(x); the potential as a function of x
	:return: np.array (N x N); the matrix Hamilton operator :: Hamilton * [psi[1], ..., psi[N]] = E * psi
	"""

	# Make matrix for V
	v = np.zeros((N, N))
	for i in range(N):
		v[i][i] = V(x[i])

	# Make the matrix itself
	H = -2 * np.eye(N, dtype=float) + np.eye(N, k=1, dtype=float) + np.eye(N, k=-1, dtype=float)

	# Scale by the right amount
	H *= -(hbar) ** 2 / (2 * me * (dx) ** 2)

	# Add potential
	H += v

	return H


def startingState(k0, sigma):
	"""
	:param x0: expectation of x at t = 0
	:param k0: wave number
	:return: starting state evaluated at every position over the interval (complex numpy array)
	"""
	normFac = (2 * np.pi * sigma ** 2) ** (-1 / 4)
	gauss = np.exp(-((x - x0) ** 2) / (4 * sigma ** 2))
	planeWave = np.exp(1j * k0 * x)

	# Return the starting state (gauss wave)
	return normFac * gauss * planeWave


def developCoeff(k0,sigma,psiJ):
	"""
	:param psiJ: array containing psi_J evaluated at every x_n
	:return: the coefficient c_j for the solution
	"""

	# Check if psiJ and x have equal lengths
	if len(psiJ) == N:
		# return the summation over all x (Dot-product of the vectors)
		return np.dot(np.conjugate(psiJ), startingState(k0,sigma))
	else:
		print("**error** psiJ and x have different lengths")
		return 0


def expectation(F, psiVec):
	"""
	:param F: Operator for F as an array (define it in advance), containing F operated on psi(x,t) for all x at time t
	:param psiVec: vector of psi(x,t) evaluated at every x_n at a time t
	:return: the expectation value of X at time t
	"""
	# Expectation
	a = np.dot(np.conjugate(psiVec), F)
	# Discard the negligible imaginary part due to computation errors.
	return np.real(a) * dx


def sigmaAnalytical(t, sigma):
	"""

	:param t: Float; the discret time we evaluate at
	:param sigma: Float; standard deviation of the Gauss wave at t=0
	:return: Float; analytical value of the standard deviation at the given time
	"""
	return np.sqrt((sigma) ** 2 + ((hbar * t) / (2 * me * sigma)) ** 2)
