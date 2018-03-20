import lib as l
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani

# Define some things for plotting
font = {'family': 'normal', 'weight': 'bold', 'size': 16}
plt.rc('font', **font)
plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble'] = [r'\boldmath']

#Variables and different starting states
(E1,E2) = (l.eV,1000*l.eV)
(sigma1,sigma2) = l.dx*62.5,l.dx*62.5
(k01,k02) = (np.sqrt(2*l.me*E1)/l.hbar,np.sqrt(2*l.me*E2)/l.hbar)
(v01,v02) = (k01*l.hbar/l.me,k02*l.hbar/l.me)
t1 = np.array([2/4*n*l.dx/v01 for n in range(l.N)])
t2 = np.array([3/5*n*l.dx/v02 for n in range(l.N)])

#Solve the Shrodinger equation
H = l.Hamilton(l.V2)
#Solution (E and psiMatrix)
E, psiMatrix = np.linalg.eigh(H)

#Develop starting state in eigen-functions; calculate coefficients c_j
coeff1 = np.array([l.developCoeff(k01,sigma1,psiMatrix[:,j]) for j in range(len(l.x))])
coeff2 = np.array([l.developCoeff(k02,sigma2,psiMatrix[:,j]) for j in range(len(l.x))])

#Expectation-arrays
(EX, EX2, EP, EP2) = (np.zeros(l.N),np.zeros(l.N),np.zeros(l.N),np.zeros(l.N))

for i in range(l.N):
    # A psi-vector containing psi(x,t) for all positions at time T
    Efac = np.exp(-1j * E * t1[i] / l.hbar)
    psiVec = np.matmul(psiMatrix,Efac*coeff1)

    #Operators for x, x**2, p and p**2, as arrays with operated psi(x,t) for all x at time t
    X = l.x*psiVec
    X2 = (l.x**2)*psiVec
    P = [l.hbar/1j/l.dx*(psiVec[n+1]-psiVec[n]) for n in range(l.N-1)] + [l.hbar/1j/l.dx*(psiVec[l.N-1]-psiVec[l.N-2])]
    P2 = [l.hbar/1j/(l.dx**2)*(psiVec[2]-2*psiVec[1]+psiVec[0])] + \
         [l.hbar/1j/(l.dx**2)*(psiVec[n+1]-2*psiVec[n]+psiVec[n-1]) for n in range(1,l.N-1)] + \
         [l.hbar/1j/(l.dx**2)*(psiVec[l.N-1]-2*psiVec[l.N-2]+psiVec[l.N-3])]

    #calculate expectation-values
    EX[i] = l.expectation(X,psiVec)
    EX2[i] = l.expectation(X2, psiVec)
    EP[i] = l.expectation(P, psiVec)
    EP2[i] = l.expectation(P2, psiVec)
    print("it:", i + 1, " Expectation:", EX[i])

#Calculate deltaX and deltaP
stdX = np.sqrt(abs(EX2-EX**2))
stdP = np.sqrt(abs(EP2-EP**2))

plt.plot(t1,stdP*stdX,label=r"$\Delta X\Delta P (t)$",c="crimson",lw=0.5)
plt.axhline(l.hbar/2,c="blue")
#plt.ylim(0,2*l.hbar)
plt.title("Uskarphetsproduktet ")
plt.legend(loc="best")
plt.grid()
plt.show()


#Animation of wavepacket
fig = plt.figure('Wave packet animation', figsize=(16,8))
ymax = 5e8
ax = plt.axes(xlim=(0,l.N*l.dx),ylim=(0,ymax))
line, = ax.plot([],[],lw=1)

#Initilizing the background
def init():
    line.set_data([],[])
    return line,

#Declaring timestep
dt = 2e-15

#Animation function
def animate(i):
    T = dt*i

    #Calculating psi(x,t)
    Efac = np.exp(-1j * E * T / l.hbar)
    Psi_t = np.matmul(psiMatrix, Efac * coeff1)

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
