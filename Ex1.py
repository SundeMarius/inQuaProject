import numpy as np, lib as l
import matplotlib.pyplot as plt

#Variables and interval
N = 1200#Number of panels over the interval
x = np.array([n*l.dx for n in range(N)])
v0 = l.k0*l.hbar/l.me
t = np.array([n*l.dx/v0/N for n in range(N)])

#Solve the Shrodinger equation
H = l.Hamilton(l.V1,x)

#Solution (E and psiMatrix)
E, psiMatrix = np.linalg.eigh(H)

#Develop starting state in eigen-functions; calculate coefficients c_j
coeff = np.array([l.developCoeff(psiMatrix[:,j],x) for j in range(len(x))]) #Vector with c_j for all j

#Expectation of x and x**2
def X(x): return x
def X2(x): return x**2

EX = np.zeros(len(t))
EX2 = np.zeros(len(t))
for i in range(len(t)):
    EX[i] = l.expectation(X,x,coeff,psiMatrix,E,t[i])
    EX2[i] = l.expectation(X2,x,coeff,psiMatrix,E,t[i])
    print("it:",i+1," Expectation:",EX[i])

#Calculate uncertainty delta x (and analytical)
stdX = np.sqrt(EX2-EX**2)
anaDx = l.sigmaAnalytical(t)

#Plot uncertainity of x
plt.plot(t,stdX,label=r"$\Delta X(t)$")
plt.plot(t,anaDx,label=r"$\Delta X(t) - analytical$")
plt.legend(loc="best")
plt.xlabel("Time t [s]")
plt.ylabel(r"$\sigma$ [m]")
plt.grid()
plt.show()


# start = l.startingState(x)
# start2 = np.matmul(psiMatrix,coeff)
#
# plt.plot(x,np.absolute(start)**2,c="r")
# #plt.plot(x,np.absolute(start2)**2,c="g")
# plt.grid()
# plt.show()



