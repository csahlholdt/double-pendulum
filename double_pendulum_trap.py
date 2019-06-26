from time import time

import numpy as np
import scipy.integrate as integrate

import matplotlib.style as mpls
mpls.use('classic')
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class TrappedDoublePendulum(object):
    '''
    Trapped double pendulum class.

    Initialised with the parameters:
    
    init_state
        Initial angle and angular velocity for the two pendulum arms
        in the form [phi1, omega1, phi2, omega2].

    L1 and L2:
        Lengths of the two pendulum arms.

    M1 and M2:
        Masses of the two bobs.

    g:
        Gravitaional acceleration.

    R0:
        Radius of the trapping circle.

    alpha and U1:
        Parameters of the trapping circle potential.
        I honestly don't remember exactly what they represent, but
        alpha is the strength (better not touch these values).
    '''
    def __init__(self,
                 init_state=[np.pi/2, 0, np.pi/2, 0],
                 L1=0.50, L2=0.50,
                 M1=1.0, M2=1.0,
                 g=9.82,
                 R0=1.50,
                 alpha=200,
                 U1=2.0):

        self.state = np.asarray(init_state, dtype='float')
        self.params = (L1, L2, M1, M2, g, alpha, U1)
        self.R0 = R0

    def get_xy(self):
        L1, L2 = self.params[:2]
        phi1, phi2 = self.state[::2]

        x = np.array([0, L1*np.sin(phi1), L1*np.sin(phi1) + L2*np.sin(phi2)])
        y = np.array([0, -L1*np.cos(phi1), -L1*np.cos(phi1) - L2*np.cos(phi2)])

        return x, y

    def second_bob_radius(self):
        x, y = self.get_xy()
        x = x[2]
        y = y[2]

        return np.sqrt(x**2 + y**2)

    def energy(self):
        L1, L2, M1, M2, g = self.params[:5]
        phi1, omega1, phi2, omega2 = self.state

        x = np.cumsum([L1 * np.sin(phi1),
                       L2 * np.sin(phi2)])
        y = np.cumsum([-L1 * np.cos(phi1),
                       -L2 * np.cos(phi2)])
        vx = np.cumsum([L1 * omega1 * np.cos(phi1),
                        L2 * omega2 * np.cos(phi2)])
        vy = np.cumsum([L1 * omega1 * np.sin(phi1),
                        L2 * omega2 * np.sin(phi2)])

        U = g * (M1 * y[0] + M2 * y[1])
        K = 0.5 * (M1 * (vx[0]**2 + vy[0]**2) + \
                   M2 * (vx[1]**2 + vy[1]**2))

        return U + K

    def dstate_dt(self, state, t):
        L1, L2, M1, M2, g, alpha, U1 = self.params
        R0 = self.R0
        phi1, omega1, phi2, omega2 = state
        
        dydx = np.zeros_like(state)
        dydx[0] = omega1
        dydx[2] = omega2

        M = np.array([[(M1+M2)*L1**2, M2*L1*L2*np.cos(phi1-phi2)],
                      [M2*L1*L2*np.cos(phi1-phi2), M2*L2**2]])

        K = np.array([[(M1+M2)*g*L1, 0],
                      [0, M2*g*L2]])

        omega_squared = np.array([[-omega2**2], [omega1**2]])
        sin_phi = np.array([[np.sin(phi1)], [np.sin(phi2)]])

        U_del = alpha*L1*L2*(U1/R0**alpha)*np.sin(phi1-phi2)*\
                (L1**2 + L2**2 + 2*L1*L2*np.cos(phi1-phi2))**(alpha/2-1)

        U_vector = np.array([[U_del], [-U_del]])

        aux_vector = M2*L1*L2*np.sin(phi1-phi2) * omega_squared - np.dot(K, sin_phi) + U_vector
        omega_dot = np.linalg.solve(M, aux_vector)

        dydx[1] = omega_dot[0]
        dydx[3] = omega_dot[1]

        return dydx

    def step(self, dt):
        self.state = integrate.odeint(self.dstate_dt, self.state, [0, dt])[1]


# Choose wether to use the default parameters or define your own below
change_default_params = True
# If decrease_R0 is True, the trapping circle will slowly shrink until
# a minimum value after which it springs back to R0 (and repeats)
decrease_R0 = True

# These parameters are used if change_default_params = True
L1, L2, M1, M2 = (0.5, 0.5, 1, 1)
R0 = L1 + L2 + 0.1
init_state = [np.pi, 0, np.pi/2, 5]

dt = 1./50
if change_default_params:
    TDP = TrappedDoublePendulum(init_state=init_state,
                                L1=L1, L2=L2, M1=M1, M2=M2,
                                R0=R0)
else:
    TDP = TrappedDoublePendulum()

L1, L2, M1, M2 = TDP.params[:4]
R0 = TDP.R0

if decrease_R0:
    if L2 > 2*L1:
        R0_min = L2 + 0.2
    else:
        R0_min = L1 + 0.2
else:
    R0_min = R0

def animate(i):
    '''perform animation step'''

    if decrease_R0:
        new_R0 = R0 - (R0 - R0_min)*(i/500)
        if TDP.second_bob_radius() < new_R0 - 0.1:
            TDP.R0 = new_R0
    TDP.step(dt)

    line.set_data(*TDP.get_xy())
    circle.set_radius(TDP.R0)
    return line, circle

ax_lim = L1 + L2 + 0.2

fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                     xlim=(-ax_lim, ax_lim), ylim=(-ax_lim, ax_lim))
ax.grid()
line, = ax.plot([], [], 'o-', lw=2)

circle = plt.Circle((0, 0), R0, color='b', fill=False)
ax.add_artist(circle)

# choose the interval based on dt and the time to animate one step
t0 = time()
animate(0)
t1 = time()
interval = 1000 * dt - (t1 - t0)

ani = animation.FuncAnimation(fig, animate, frames=500, interval=interval)

#ani.save('TDP.gif', writer='imagemagick', fps=30)
plt.show()
