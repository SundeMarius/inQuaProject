import numpy as np

# constants
me = 9.1e-31
hbar = (6.63e-34) / 2 / np.pi

def startingState(positions):
    """
    :param positions: numpy array of the positions over the interval
    :return: starting state evaluated at every position over the interval (complex numpy array)
    """
    N = len(positions)  # Number of panels over the interval

    # find delta x (only if N > 1)
    try:
        dx = (positions[N] - positions[0]) /(N-1)
    except ZeroDivisionError:
        print("Zero division error! there must be at least 1 position in 'positions'.")

    #Sigma, expectation values at t = 0
    sigma = 10*N*dx
    x0 = N*dx/2 #Midway on the interval
    k0 = me*0.1/hbar #speed 0.1 m/s (this is the wave-number)
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

    #find delta x (only if N > 1)
    try:
        dx = (positions[N]-positions[0])/(N-1)
    except ZeroDivisionError:
        print("Zero division error! there must be at least 1 position in 'positions'.")

    #Check if psiJ and positions have equal lengths
    if len(psiJ) == N:
        #return the summation over all positions (Dot-product of the vectors)
        return np.dot(np.conjugate(psiJ),startingState(positions))*dx
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

    #find delta x (only if N > 1)
    try:
        dx = (positions[N]-positions[0])/(N-1)
    except ZeroDivisionError:
        print("Zero division error! there must be at least 2 position in 'positions'.")

    #Expectation
    return np.dot(psiVec*X(positions),psiVecConj)*dx

N = 1000
x = np.linspace(0,1,N)

def X(x): return x

