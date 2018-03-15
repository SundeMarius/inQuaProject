import numpy as np, lib as l
import matplotlib.pyplot as plt

#Variables and interval
v0 = 0.2 #Initial speed of electron
N = 2000 #Number of panels over the interval
a = 0 # [m] left end of interval
b = 1 # [m] right end of interval
x = np.linspace(a,b,N)
dx = (b-a)/N
t = np.array([n*dx/v0 for n in range(N)])

#Solve the Shrodinger equation
H = l.Hamilton(l.V1,x)

#Solution (E and psiMatrix)
E, psiMatrix = np.linalg.eigh(H)

#Develop starting state in eigen-functions; calculate coefficients c_j
coeff = np.array([l.developCoeff(psiMatrix[j],x,v0) for j in range(N)]) #Vector with c_j for all j

#Expectation of x and x**2
def X(x): return x
def X2(x): return x**2

# EX = np.zeros(len(t))
# EX2 = np.zeros(len(t))
# for i in range(len(t)):
#     EX[i] = l.expectation(X,x,coeff,psiMatrix,E,t[i])
#     EX2[i] = l.expectation(X2,x,coeff,psiMatrix,E,t[i])
#     print("it:",i+1," Expectation:",EX[i])
#
# #Calculate uncertainty delta x
# stdX = np.sqrt(EX2-EX**2)
# #Plot expectation of x
# plt.plot(t,stdX)
# plt.grid()
# plt.show()

start = l.startingState(x,v0)
start2 = np.matmul(psiMatrix,coeff)
print(coeff)
for i in range(N):
    print("x=",x[i],"\tstart1=",np.real(start[i]*np.conjugate(start[i]))
          ,"\tstart2=",np.real(start2[i]*np.conjugate(start2[i])))

