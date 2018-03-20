import lib as l
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani

# Define some things for plotting
font = {'family': 'normal', 'weight': 'bold', 'size': 16}
plt.rc('font', **font)
plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble'] = [r'\boldmath']

#Variables and time interval
E = l.eV
sigma = l.dx*62.5

k0 = np.sqrt(2*l.me*E)/l.hbar
v0 = np.sqrt(2*E/l.me)
t = np.array([2/4*n*l.dx/v0 for n in range(l.N)])

#Solve the Shrodinger equation
H = l.Hamilton(l.V1)

#Solution (E and psiMatrix)
E, psiMatrix = np.linalg.eigh(H)

#Develop starting state in eigen-functions - calculate coefficients c_j (a vector of c_j for all j)
coeff = np.array([l.developCoeff(k0,sigma,psiMatrix[:, j]) for j in range(len(l.x))])
#Initialize arrays for expectation-values
(EX, EX2) = (np.zeros(l.N), np.zeros(l.N))


#Iterate over each time step
for i in range(len(t)):
    # A psi-vector containing psi(x,t) for all positions at time T
    Efac = np.exp(-1j * E * t[i] / l.hbar)
    psiVec = np.matmul(psiMatrix, Efac*coeff)

    #Operators for x and x**2 as arrays with operated psi(x,t) for all x at time t
    X = l.x*psiVec
    X2 = (l.x**2)*psiVec

    #calculate expectation-values
    EX[i] = l.expectation(X, psiVec)
    EX2[i] = l.expectation(X2, psiVec)
    print("it:", i + 1, " Expectation:", EX[i])

#Calculate deltaX and deltaP
stdX = np.sqrt(abs(EX2-EX**2))
anaDx = l.sigmaAnalytical(t,sigma)



#Plot uncertainity of x
plt.plot(t, stdX, label=r"$\Delta x(t)$ - numerisk", lw=3, color='crimson')
plt.plot(t, anaDx, label=r"$\Delta x(t)$ - analytisk", lw=3, color='blue')
plt.legend(loc="best")
plt.xlabel("$t$ [s]")
plt.ylabel(r"$\sigma$ [m]")
plt.grid()
plt.show()

##Plot starting state (to check that the end of the interval does not interfere with the electron)
#plt.plot(t, np.abs(l.startingState(k0, sigma))**2)
#plt.show()

#Animation of wavepacket
fig = plt.figure('Wave packet animation', figsize=(16, 8))
ymax = 1e8
ax = plt.axes(xlim=(0, l.N*l.dx), ylim=(0, ymax))
line, = ax.plot([], [], lw=3)
#Initilizing the background
def init():
    line.set_data([], [])
    return line,
#Declaring timestep
dt = 1e-15

#Animation function
def animate(i):
    T = dt*i

    #Calculating psi(x,t)
    Efac = np.exp(-1j * E * T / l.hbar)
    Psi_t = np.matmul(psiMatrix, Efac * coeff)

    #probability-density
    rho_t = np.abs(Psi_t)**2
    line.set_data(l.x, rho_t)
    return line,

#Call the animator, frames =  number of pictures (max i), interval = duration of each picture (in ms)
anim = ani.FuncAnimation(fig, animate, init_func=init, repeat=False, frames=3000, interval=1, blit=True)
plt.show()

