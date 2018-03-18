import numpy as np, lib as l
import matplotlib.pyplot as plt
from matplotlib import animation as ani

#Variables and interval
N = 1200#Number of panels over the interval
x = np.array([n*l.dx for n in range(N)])
v0 = l.k0*l.hbar/l.me
t = np.array([n*l.dx/v0/N for n in range(N)])

#Solve the Shrodinger equation
H = l.Hamilton(l.V2,x)

#Solution (E and psiMatrix)
E, psiMatrix = np.linalg.eigh(H)

#Develop starting state in eigen-functions; calculate coefficients c_j
coeff = np.array([l.developCoeff(psiMatrix[:,j],x) for j in range(len(x))]) #Vector with c_j for all j

# #Expectation of x and x**2
# def X(x): return x
# def X2(x): return x**2
#
# EX = np.zeros(len(t))
# EX2 = np.zeros(len(t))
# for i in range(len(t)):
#     EX[i] = l.expectation(X,x,coeff,psiMatrix,E,t[i])
#     EX2[i] = l.expectation(X2,x,coeff,psiMatrix,E,t[i])
#     print("it:",i+1," Expectation:",EX[i])
#
# #Calculate uncertainty delta x (and analytical)
# stdX = np.sqrt(EX2-EX**2)
# anaDx = l.sigmaAnalytical(t)
#
# #Plot uncertainity of x
# plt.plot(t,stdX,label=r"$\Delta X(t)$")
# plt.plot(t,anaDx,label=r"$\Delta X(t) - analytical$")
# plt.legend(loc="best")
# plt.xlabel("Time t [s]")
# plt.ylabel(r"$\sigma$ [m]")
# plt.grid()
# plt.show()


# start = l.startingState(x)
# start2 = np.matmul(psiMatrix,coeff)
#
# plt.plot(x,np.absolute(start)**2,c="r")
# #plt.plot(x,np.absolute(start2)**2,c="g")
# plt.grid()
# plt.show()

#Animation of wavepacket
fig = plt.figure('Wave packet animation', figsize=(16,8))
ymax = 5e9
ax = plt.axes(xlim=(0,N*l.dx),ylim=(0,ymax))
line, = ax.plot([],[],lw=3)

#Initilizing the background
def init():
    line.set_data([],[])
    return line,

#Declaring timestep
dt = 1e-17

#Animation function
def animate(i):
    T = dt*i

    #Calculating psi(x,t)
    Efac = np.exp(-1j * E * T / l.hbar)
    Psi_t = np.matmul(psiMatrix, Efac * coeff)

    #probability-density
    rho_t = np.abs(Psi_t)**2
    line.set_data(x,rho_t)
    return line,

#Add the potential-plot in animation (only for V2)
plt.plot(x,l.V2(x)*ymax/(l.V2(N*l.dx)))
plt.xlabel(r'$x$ [m]', fontsize=20)

#Call the animator, frames =  number of pictures (max i), interval = duration of each picture (in ms)
anim = ani.FuncAnimation(fig,animate,init_func=init,repeat=False,frames=3000,interval=1,blit=True)
plt.show()



