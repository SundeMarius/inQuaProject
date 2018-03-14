import numpy as np
import matplotlib.pyplot as plt


hbar = 1.05457e-34  # [Js] - Reduced Planck's constant
me = 9.10938e-31  # [kg] - Electron mass
v0 = 0.5 # [m/s] - initial speed
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

def startingState(positions,v0):
    """
    :param positions: numpy array of the positions over the interval
    :param v0: initial speed
    :return: starting state evaluated at every position over the interval (complex numpy array)
    """
    N = len(positions)  # Number of panels over the interval

    #Sigma, expectation values at t = 0
    sigma = 10*N*dx
    x0 = N*dx/2 #Midway on the interval
    k0 = me*v0/hbar #speed 0.1 m/s (this is the wave-number)
    normFac = (2*np.pi*sigma)**(-0.25)
    gauss = np.exp(-(positions-x0)**2/(4*sigma**2))
    planeWave = np.exp(1j*k0*positions)

    #Return the starting state (gauss wave)
    return np.array(normFac*gauss*planeWave)

def developCoeff(psiJ,positions):
    """
    :param psiJ: array containing psi_J evaluated at every x_n
    :param positions: array of x_n
    :return: the coefficient c_j for the solution
    """
    N = len(positions) # Number of panels over the interval

    #Check if psiJ and positions have equal lengths
    if len(psiJ) == N:
        #return the summation over all positions (Dot-product of the vectors)
        return np.dot(np.conjugate(psiJ),startingState(positions,v0))*dx
    else:
        print("**error** psiJ and positions have different lengths")
        return 0

def expectation(X,positions,developCoeff,psiMatrix,E,t):
    """
    :param X: expectation-variable as a function (define it in advance)
    :param developCoeff: numpy array of all the develop coefficients j (from j = 1 to N)
    :param positions: numpy array of x_n
    :param psiMatrix: numpy array containing psi_J-vectors for J = 1 to N (Matrix of psiJ as row-vectors)
    :param E: numpy array of all the eigenvalues (each possible energy)
    :param t: at time t (number)
    :return: the expecetation value of X at time t
    """
    #Define psi(x,t) at a position x_n
    def psiXn(n,t):
        """
        :param n: position-index n (from 0 to N-1)
        :param t: time t
        :return:  psi(x,t) at position x at time t
        """
        energyCoeff = np.exp(-1j * E * t / hbar)*developCoeff #numpy array of energy factors with developCoeff. for each state j
        states = np.transpose(psiMatrix)  # Matrix with row vectors as the wave-values for each state j at position x_n

        return np.dot(states[n],energyCoeff)

    #A psi-vector containing psi(x,t) for all positions at time t (and its conjugate)
    N = len(positions) #Number of subintervals
    psiVec = np.array([psiXn(n,t) for n in range(N)])
    psiVecConj = np.conjugate(psiVec)

    #Expectation
    return np.dot(psiVec*X(positions),psiVecConj)*dx

