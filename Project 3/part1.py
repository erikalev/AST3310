from derivatives import *
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

kb =1.38064852e-23
nx = 100
n = nx
T = 1e3
u = 0.6
dx = 1.0    
gamma = 5.0/3.0
m_u = 1.6605e-27


p = np.zeros((n, nx))           # momentum
rho = np.zeros((n, nx))         # density
ux = np.zeros((n, nx))          # velocity
P = np.zeros((n, nx))           # Pressure
cs = np.zeros((n, nx))          # speed of sound
rho[0, 0] = 1.0
ux[0, 0] = 0
P[0, 0] = rho[0, 0]/(u*m_u)*kb*T
p[0, 0] = rho[0, 0]*ux[0, 0]
cs = np.sqrt(gamma*P[0, 0]/rho[0, 0])

dt = 0.3/cs
dt1 = dt

def find_dt(ux, rho, dx):
    # Establishing variables to perform the variable step length
    p = 0.3
    test = np.zeros(2*nx)
    for i in xrange(len(ux)):
        drho_dt = -upwind(ux, rho, dx, i, nx)
        drho_ux_dt = -upwind(ux, rho*ux*ux, dx, i, nx) - upwind(ux, P, dx, i, nx)
        test[i*2] = abs(p*rho[i]/(cs*drho_dt))
        test[i*2 + 1] = abs(p*P[i]/(cs*drho_ux_dt))
    #test[0] = abs(p*rho[i]/(cs*drho_dt))
    #test[1] = abs(p*P[i]/(cs*drho_ux_dt))

    dt = -min(test)        

    # Resetting initial dm to avoid to large values
    if dt < dt1:
        dt = dt1

    return dt

for i in xrange(nx-1):
    for j in xrange(nx-1):
        if i < 50:
            rho[j] = 1.0
        else:
            rho[j] = 10
        rho[j+1, i] = rho[j, i] + dt*upwind(ux[:,i], rho[:, i], dx, j, nx)
        ux[j+1, i] = ux[j, i] + dt*(upwind(ux[:,i], rho[:,i]*ux[:,i]*ux[:,i], dx, j, nx))
        P[j+1, i] = rho[j+1, i]/(u*m_u)*kb*T

    dt = find_dt(ux[:,i], rho[:,i], P[:,i], dx)
    rho[i+1, j] = rho[i, j] -upwind(ux, rho, dx, j, nx)*dt
    
