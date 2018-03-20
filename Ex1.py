import numpy as np, lib as l
import matplotlib.pyplot as plt
from matplotlib import animation as ani

# Define some things for plotting
font = {'family': 'normal', 'weight': 'bold', 'size': 16}
plt.rc('font', **font)

#Variables and time interval
E = 1*l.eV
sigma = 10*l.dx
k0 = np.sqrt(2*l.me*E)/l.hbar
v0 = np.sqrt(2*E/l.me)
t = np.array([3/5*n*l.dx/v0 for n in range(l.N)])

#Solve the Shrodinger equation
H = l.Hamilton(l.V1)

#Solution (E and psiMatrix)
E, psiMatrix = np.linalg.eigh(H)

#Develop starting state in eigen-functions; calculate coefficients c_j
coeff = np.array([l.developCoeff(k0,sigma,psiMatrix[:,j]) for j in range(len(l.x))]) #Vector with c_j for all j
#Expectation-arrays
(EX,EX2) = (np.zeros(l.N),np.zeros(l.N))

for i in range(len(t)):
    # A psi-vector containing psi(x,t) for all positions at time T
    Efac = np.exp(-1j * E * t[i] / l.hbar)
    psiVec = np.matmul(psiMatrix,Efac*coeff)

    #Operators for x and x**2 as arrays with operated psi(x,t) for all x at time t
    X = l.x*psiVec
    X2 = (l.x**2)*psiVec

    #calculate expectation-values
    EX[i] = l.expectation(X,psiVec)
    EX2[i] = l.expectation(X2, psiVec)
    print("it:", i + 1, " Expectation:", EX[i])

#Calculate deltaX and deltaP
stdX = np.sqrt(abs(EX2-EX**2))
anaDx = l.sigmaAnalytical(t,sigma)

#Plot uncertainity of x
plt.plot(t,stdX,label=r"$\Delta x(t) - numerisk$")
plt.plot(t,anaDx,label=r"$\Delta x(t) - analytisk$")
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

#Animation of wavepacket
fig = plt.figure('Wave packet animation', figsize=(16,8))
ymax = 1e8
ax = plt.axes(xlim=(0,l.N*l.dx),ylim=(0,ymax))
line, = ax.plot([],[],lw=1)

#Initilizing the background
def init():
    line.set_data([],[])
    return line,

#Declaring timestep
dt = 3e-15

#Animation function
def animate(i):
    T = dt*i

    #Calculating psi(x,t)
    Efac = np.exp(-1j * E * T / l.hbar)
    Psi_t = np.matmul(psiMatrix, Efac * coeff)

    #probability-density
    rho_t = np.abs(Psi_t)**2
    line.set_data(l.x,rho_t)
    return line,

#Add the potential-plot in animation (only for V2)
plt.plot(l.x,l.V2(l.x)*ymax/(l.V2(l.N*l.dx)))
plt.xlabel(r'$x$ [m]', fontsize=20)

#Call the animator, frames =  number of pictures (max i), interval = duration of each picture (in ms)
anim = ani.FuncAnimation(fig,animate,init_func=init,repeat=False,frames=3000,interval=1,blit=True)
plt.show()



