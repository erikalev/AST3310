import FVis
from derivatives import *
import numpy as np
import matplotlib.pyplot as plt

kb = 1.38064852e-23
mu = 1.6605e-27
n = 100
nx = 100
u = 0.6
dx = 1.0        
rho = np.zeros((nx, n)) 
P = np.zeros((nx, n))
rhoux = np.zeros((nx, n))
ux = np.zeros((nx, n))

x_arr = [i*dx for i in range(n)]

T = 1e-5
#self.T = 1e-5
rho[:][0] = 1
rho[50:][0] = 10
P[:][0] = rho[:][0]/(u*mu)*kb*T        
cs = np.sqrt(5.0/3.0*P[:][0]/rho[:][0])

for k in xrange(n-1):
    dt = min(0.3/cs)
    d_rho_dt = -rho[:][k]*upwind(ux[:][k], ux[:][k], dx, nx) -ux[:][k]*upwind(ux[:][k], rho[:][k], dx, nx)
    d_ux_dx = upwind(ux[:][k], ux[:][k], dx, nx)
    d_P_dx = upwind(ux[:][k], P[:][k], dx, nx)
    d_ux_dt = -ux[:][k]*d_ux_dx - d_P_dx/rho[:][k]

    rho[:][k+1] = rho[:][k] + dt*d_rho_dt            
    ux[:][k+1] = ux[:][k] + dt*d_ux_dt
    P[:][k+1] = rho[:][k]/(u*mu)*kb*T
    cs = np.sqrt(5.0/3.0*P[:][k]/rho[:][k])



import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig, ax = plt.subplots()

x = x_arr        # x-array
line, = ax.plot(x, rho[:][0])

def animate(i):
    line.set_xdata(x_arr) 
    line.set_ydata(rho[:][i])  # update the data
    return line,

#Init only required for blitting to give a clean slate.
def init():
    line.set_ydata(np.ma.array(x, mask=True))
    return line,

ani = animation.FuncAnimation(fig, animate, np.arange(1, 200), init_func=init,
    interval=25, blit=True)
plt.show()

"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
 
def simData():
# this function is called as the argument for
# the simPoints function. This function contains
# (or defines) and iterator---a device that computes
# a value, passes it back to the main program, and then
# returns to exactly where it left off in the function upon the
# next call. I believe that one has to use this method to animate
# a function using the matplotlib animation package.
#
    t_max = 100.0
    dt = 1
    x = 0.0
    t = 0.0
    count = 0
    while t < t_max:
        x = rho[:][count]
        t = t + dt
        yield x, t
 
def simPoints(simData):
    x, t = simData[0], simData[1]
    time_text.set_text(time_template%(t))
    line.set_data(t, x)
    return line, time_text
 
##
##   set up figure for plotting:
##
fig = plt.figure()
ax = fig.add_subplot(111)
# I'm still unfamiliar with the following line of code:
line, = ax.plot([], [], 'bo', ms=10)
ax.set_ylim(0, 15)
ax.set_xlim(0, 100)
##
time_template = 'Time = %.1f s'    # prints running simulation time
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
## Now call the animation package: (simData is the user function
## serving as the argument for simPoints):
ani = animation.FuncAnimation(fig, simPoints, simData, blit=False,\
     interval=10, repeat=True)
plt.show()
"""
"""
vis = FVis.FluidVisualiser()
vis.save_data(1000, solver.step, rho=solver.rho, u=solver.ux, P=
solver.P, sim_fps=0.5)  

vis.animate_1D("rho")
vis.delete_current_data()
"""
