import FVis
import numpy as np
import matplotlib.pyplot as plt

class Project3:    
    def __init__(self, n, nx):   
        self.count = 0
        self.kb = 1.38064852e-23
        self.mu = 1.6605e-27
        self.n = n
        self.nx = nx
        self.u = 0.6
        self.dx = 1.0        
        self.rho = np.zeros(nx) 
        self.P = np.zeros(nx)
        self.ux = np.zeros(nx)
        self.X = [self.dx*i for i in range(self.nx)]
        self.T = 1e-5
        self.time = 0

    def initial_conditions(self):
        self.rho[:] = 1
        self.rho[self.nx/2:] = 10
        self.P = self.rho/(self.u*self.mu)*self.kb*self.T        

    def rel_phi(self, phi, dphi):
        return abs(dphi/phi)
    
    def rel_v(self, v, dv):
        return abs(v/dv)

    def central_diff(self, phi):
        der = np.zeros(self.nx)
        der = (np.roll(phi, -1) - np.roll(phi, 1))/(2*self.dx)    
        return der

    def upwind(self, phi, u):
        der = np.zeros(self.nx)
        derivate_negative_u = (np.roll(phi, -1) - phi)/self.dx
        derivate_positive_u = (phi - np.roll(phi, 1))/self.dx
        der[np.where(u < 0)] = derivate_negative_u[np.where(u < 0)]
        der[np.where(u >= 0)] = derivate_positive_u[np.where(u >= 0)]
        return der

    def variable_step_length(self, ux, rho, d_rho_dt, d_ux_dx):
        rel_rho = self.rel_phi(rho, d_rho_dt)
        rel_ux_v = self.rel_v(ux, self.dx)
        delta = max(np.amax(rel_rho), np.amax(rel_ux_v))
        if delta == 0.0:
            dt = 0.1
        else:
            dt = 0.1/delta  
        return dt

    def step(self):
        drhodxu = self.upwind(self.rho, self.ux)
        duxdxc = self.central_diff(self.ux)
        d_rho_dt = -self.ux*drhodxu - self.rho*duxdxc
        
        d_ux_dx = self.upwind(self.ux, self.ux)
        d_P_dx = self.central_diff(self.P)
        d_ux_dxu = self.upwind(self.ux, self.ux)
        d_ux_dt = -self.ux*d_ux_dxu - d_P_dx/self.rho

        dt = self.variable_step_length(self.ux, self.rho, d_rho_dt, d_ux_dx)
        self.rho[:] = self.rho + dt*d_rho_dt            
        self.ux[:] = self.ux + dt*d_ux_dt
        self.P[:] = self.rho/(self.u*self.mu)*self.kb*self.T
        if self.count%100 == 0:
            plt.plot(self.X, self.rho)
            plt.ylabel("density [$kg/m^3$]")
            plt.xlabel("x [m]")
            #plt.legend(["%.2f [s]" % self.time])
            plt.legend(["0 s", "20 s", "42 s", "65 s"])
            plt.hold("on")
            print self.time
        self.count += 1
        
        self.time += dt
        return dt

solver = Project3(100, 100)
solver.initial_conditions()

vis = FVis.FluidVisualiser()
vis.save_data(80, solver.step, rho=solver.rho, u=solver.ux, P=
solver.P, sim_fps=0.5)

vis.animate_1D("rho")
vis.plot_avg("rho")
vis.delete_current_data()
plt.show()
