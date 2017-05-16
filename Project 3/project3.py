import FVis
from derivatives import *
import numpy as np
import matplotlib.pyplot as plt

class Project3:
    def __init__(self, n, nx):   
        self.kb = 1.38064852e-23
        self.mu = 1.6605e-27
        self.n = n
        self.nx = nx
        self.u = 0.6
        self.dx = 1.0        
        self.rho = np.zeros(nx) 
        self.P = np.zeros(nx)
        self.ux = np.zeros(nx)
    
        self.T = 1e-8
        #self.T = 1e-5

    def initial_conditions(self):
        self.rho[:] = 1
        self.rho[50:] = 10
        self.P = self.rho/(self.u*self.mu)*self.kb*self.T        

    def rel_phi(self, phi, dphi):
        return abs(dphi/phi)
    
    def rel_v(self, v, dv):
        return abs(v/dv)

    def variable_step_length(self, ux, rho, d_rho_dt, dt1, d_ux_dx):
        rel_rho = self.rel_phi(rho, d_rho_dt)
        #rel_ux = self.rel_phi(ux,)
        #rel_uz = self.rel_phi(uz, d_uz_dt)
        rel_ux_v = self.rel_v(ux, self.dx)
        delta = max(np.amax(rel_rho), np.amax(rel_ux_v))
        if delta == 0.0:
            dt = dt1
        else:
            dt = 0.1/delta  
        return dt

    def step(self):
        self.cs = np.sqrt(5.0/3.0*self.P/self.rho)
        dt = min(0.3/self.cs)/4
        #print dt
        d_rho_dt = -self.rho*central_diff(self.ux, self.rho*self.ux, self.dx, self.nx) -self.ux*upwind(self.ux, self.rho, self.dx, self.nx)
        d_ux_dx = upwind(self.ux, self.ux, self.dx, self.nx)
        d_P_dx = central_diff(self.ux, self.P, self.dx, self.nx)
        d_ux_dt = -self.ux*d_ux_dx - d_P_dx/self.rho

        dt = self.variable_step_length(self.ux, self.rho, d_rho_dt, dt, d_ux_dx)
        self.rho[:] = self.rho + dt*d_rho_dt            
        self.ux[:] = self.ux + dt*d_ux_dt
        self.P[:] = self.rho/(self.u*self.mu)*self.kb*self.T

        return dt


solver = Project3(100, 100)
solver.initial_conditions()

vis = FVis.FluidVisualiser()
vis.save_data(1000, solver.step, rho=solver.rho, u=solver.ux, P=
solver.P, sim_fps=0.5)

vis.animate_1D("rho")
vis.delete_current_data()

