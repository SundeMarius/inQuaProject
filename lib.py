import numpy as np


hbar = 1.05457e-34  # [Js] - Reduced Planck's constant
me = 9.10938e-31  # [kg] - Electron mass
dx = 1e-11
E = hbar**2/(2*me*(1200*dx)**2)
# Sigma, expectation values at t = 0
sigma = dx*10
x0 = sigma*45  #Put the mean x at t=0 away from the walls
k0 = np.sqrt(2*me*E)/hbar  # wave number k [1/m]

def V1(x):
	return 0


def V2(x):
	return (x-x0*1.4)** 2


def Hamilton(V, positions):
	"""
	:param V: function(x); the potential as a function of x
	:param positions: array of all the x-values
	:return: np.array (N x N); the matrix Hamilton operator :: Hamilton * [psi[1], ..., psi[N]] = E * psi
	"""

	# Make matrix for V
	Nt = len(positions)
	v = np.zeros((Nt, Nt))
	for i in range(Nt):
		v[i][i] = V(positions[i])

	# Make the matrix itself
	H = -2 * np.eye(Nt, dtype=float) + np.eye(Nt, k=1, dtype=float) + np.eye(Nt, k=-1, dtype=float)

	# Scale by the right amount
	H *= -(hbar) ** 2 / (2 * me * (dx) ** 2)

	# Add potential
	H += v

	return H


def startingState(positions):
	"""
	:param positions: numpy array of the positions over the interval
	:return: starting state evaluated at every position over the interval (complex numpy array)
	"""
	Nt = len(positions)  # Number of panels over the interval

	normFac = (2*np.pi*sigma**2)**(-1/4)
	gauss = np.exp(-((positions - x0)**2)/(4*sigma**2))
	planeWave = np.exp(1j * k0 * positions)

	# Return the starting state (gauss wave)
	return normFac * gauss * planeWave


def developCoeff(psiJ, positions):
	"""
	:param psiJ: array containing psi_J evaluated at every x_n
	:param positions: array of x_n
	:return: the coefficient c_j for the solution
	"""
	Nt = len(positions)  # Number of panels over the interval

	# Check if psiJ and positions have equal lengths
	if len(psiJ) == Nt:
		# return the summation over all positions (Dot-product of the vectors)
		return np.dot(np.conjugate(psiJ), startingState(positions))
	else:
		print("**error** psiJ and positions have different lengths")
		return 0


def expectation(X, positions, coeff, psiMatrix, E, t):
	"""
	:param X: expectation-variable as a function (define it in advance)
	:param positions: numpy array of x_n
	:param coeff: numpy array of all the develop coefficients j (from j = 1 to N)
	:param psiMatrix: Matrix of psiJ as colum-vectors
	:param E: numpy array of all the eigenvalues (each possible energy)
	:param t: at time t (number)
	:return: the expecetation value of X at time t
	"""
	# A psi-vector containing psi(x,t) for all positions at time t
	Efac = np.exp(-1j * E * t / hbar)
	psiVec = np.matmul(psiMatrix,Efac*coeff)

	# Expectation
	return np.dot(X(positions), np.absolute(psiVec)**2)*dx


def sigmaAnalytical(t):
	return np.sqrt((sigma)**2 + ((hbar*t)/(2*me*sigma))**2)


