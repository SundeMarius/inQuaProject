import numpy as np, lib as l
import matplotlib.pyplot as plt
from matplotlib import animation as ani

# Define some things for plotting
font = {'family': 'normal', 'weight': 'bold', 'size': 12}
plt.rc('font', **font)

#Variables and interval
N = 1000#Number of panels over the interval
x = np.array([n*l.dx for n in range(N)])
v0 = l.k0*l.hbar/l.me
t = np.array([n*l.dx/v0/N for n in range(N)])

#Solve the Shrodinger equation
H = l.Hamilton(l.V2,x)
#Solution (E and psiMatrix)
E, psiMatrix = np.linalg.eigh(H)

#Develop starting state in eigen-functions; calculate coefficients c_j
coeff = np.array([l.developCoeff(psiMatrix[:,j],x) for j in range(len(x))]) #Vector with c_j for all j
#Expectation-arrays
(EX,EX2,EP,EP2) = (np.zeros(N),np.zeros(N),np.zeros(N),np.zeros(N))

for i in range(len(t)):
    # A psi-vector containing psi(x,t) for all positions at time T
    Efac = np.exp(-1j * E * t[i] / l.hbar)
    psiVec = np.matmul(psiMatrix,Efac*coeff)

    #Operators for x, x**2, p and p**2, as arrays with operated psi(x,t) for all x at time t
    X = x*psiVec
    X2 = (x**2)*psiVec
    P = [l.hbar/1j/l.dx*(psiVec[n+1]-psiVec[n]) for n in range(N-1)] + [l.hbar/1j/l.dx*(psiVec[N-1]-psiVec[N-2])]
    P2 = [l.hbar/1j/(l.dx**2)*(psiVec[2]-2*psiVec[1]+psiVec[0])] + \
         [l.hbar/1j/(l.dx**2)*(psiVec[n+1]-2*psiVec[n]+psiVec[n-1]) for n in range(1,N-1)] + \
         [l.hbar/1j/(l.dx**2)*(psiVec[N-1]-2*psiVec[N-2]+psiVec[N-3])]

    #calculate expectation-values
    EX[i] = l.expectation(X,psiVec)
    EX2[i] = l.expectation(X2, psiVec)
    EP[i] = l.expectation(P, psiVec)
    EP2[i] = l.expectation(P2, psiVec)
    print("it:", i + 1, " Expectation:", EX[i])

#Calculate deltaX and deltaP
stdX = np.sqrt(abs(EX2-EX**2))
stdP = np.sqrt(abs(EP2-EP**2))

plt.plot(t,stdP*stdX,label=r"$\Delta X\Delta P (t)$",c="crimson",lw=0.5)
plt.axhline(l.hbar/2,c="blue")
#plt.ylim(0,2*l.hbar)
plt.title("Uskarphetsproduktet ")
plt.legend(loc="best")
plt.grid()
plt.show()


