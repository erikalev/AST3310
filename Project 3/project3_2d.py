import FVis
import numpy as np
import matplotlib.pyplot as plt

class Project3:
    def __init__(self, nx, nz):   
        self.kb = 1.38064852e-23
        self.mu = 1.6605e-27
        self.nz = nz
        self.nx = nx

        self.u = 0.6
        self.dx = 1.0        
        self.dz = 1.0
        self.n = 0

        self.rho = np.zeros((nx, nz)) 
        self.P = np.zeros((nx, nz))
        self.ux = np.zeros((nx, nz))
        self.uz = np.zeros((nx,  nz))
        self.T = 1e-7
        self.G = 6.67408e-11
        self.M_sun = 1
        self.R_sun = 695700000
        
    def initial_conditions(self):
        self.rho[:][:] = 1
        center1 = np.asarray((75, 75))
        center2 = np.asarray((125, 125))
        for i in xrange(self.nx):
            for j in xrange(self.nz):
                if np.linalg.norm(center1 - np.asarray((i, j))) <= 5.0:
                    self.rho[i, j] = 10.0
                if np.linalg.norm(center2 - np.asarray((i, j))) <= 5.0:
                    self.rho[i, j] = 10.0
        self.P = self.rho/(self.u*self.mu)*self.kb*self.T        


    def upwind_x(self, phi, u):
        if np.size(u) == self.nx*self.nz:
            der = np.zeros((self.nx, self.nz))
            derivate_negative_u = (np.roll(phi, -1, axis = 1) - phi)/self.dx
            derivate_positive_u = (phi - np.roll(phi, 1, axis = 1))/self.dx

            der[np.where(u < 0)] = derivate_negative_u[np.where(u < 0)]
            der[np.where(u >= 0)] = derivate_positive_u[np.where(u >= 0)]
        
        else:
            der = np.zeros(self.nx)
            derivate_negative_u = (np.roll(phi, -1) - phi)/self.dx
            derivate_positive_u = (phi - np.roll(phi, 1))/self.dx
            der[np.where(u < 0)] = derivate_negative_u[np.where(u < 0)]
            der[np.where(u >= 0)] = derivate_positive_u[np.where(u >= 0)]

        return der

    def upwind_z(self, phi, u):
        if np.size(u) == self.nx*self.nz:
            der = np.zeros((self.nx, self.nz))        
            derivate_negative_u = (np.roll(phi, -1, axis= 0 ) - phi)/self.dz
            derivate_positive_u = (phi - np.roll(phi, 1, axis= 0 ))/self.dz

            der[np.where(u < 0)] = derivate_negative_u[np.where(u < 0)]
            der[np.where(u >= 0)] = derivate_positive_u[np.where(u >= 0)]

        else:
            print "You have chosen dz without having a z-dimention!"
        return der                
            
    def central_diff_x(self, phi):
        if np.size(self.ux) == self.nx*self.nz:
            der = np.zeros((self.nx, self.nz))
            der = (np.roll(phi, -1, axis = 1) - np.roll(phi, 1, axis = 1))/(2*self.dx)    
        else:
            der = np.zeros(self.nx)
            der = (np.roll(phi, -1) - np.roll(phi, 1))/(2*self.dx)    
        return der

    def central_diff_z(self, phi):
        if np.size(self.ux) == self.nx*self.nz:
            der = np.zeros((self.nx, self.nz))        
            der = (np.roll(phi, -1, axis = 0) - np.roll(phi, 1, axis = 0))/(2*self.dz)    
        else:
            print "You have chosen dz without having a z-dimention!"
        return der

    def viscocity(self, n, rho, d_ux_dz, d_uz_dx):
        return -n*rho*(d_ux_dz + d_uz_dx)


    def rel_phi(self, phi, dphi):
        return abs(dphi/phi)
    
    def rel_v(self, v, dx):
        return abs(v/dx)
 
    def variable_step_length(self, ux, uz, rho, d_rho_dt, d_ux_dt, d_uz_dt, dt1, d_ux_dx, d_uz_dz):

        rel_rho = self.rel_phi(rho, d_rho_dt)
        #rel_ux = self.rel_phi(ux,)
        #rel_uz = self.rel_phi(uz, d_uz_dt)
        rel_ux_v = self.rel_v(ux, self.dx)
        rel_uz_v = self.rel_v(uz, self.dz)
        delta = max(np.amax(rel_rho), np.amax(rel_ux_v), np.amax(rel_uz_v))
        if delta == 0.0:
            dt = dt1
        else:
            dt =0.1/delta  
        return dt
    
    def step(self):
        self.cs = np.sqrt(5.0/3.0*self.P/self.rho)
        dt = 0.3/np.amax(self.cs)
        
        # x-derivatives to use in eq. of motion
        d_ux_dx = self.central_diff_x(self.ux)
        d_P_dx = self.central_diff_x(self.P)
        d_uz_dx = self.central_diff_x(self.uz)
        d_rho_dx = self.upwind_x(self.rho, self.ux)


        # z-derivatives to use in eq. of motion
        d_uz_dz = self.central_diff_z(self.uz)
        d_ux_dz = self.central_diff_z(self.ux)
        d_P_dz = self.central_diff_z(self.P)
        d_rho_dz = self.upwind_z(self.rho, self.uz)

        d_ux_dt = -self.ux*d_ux_dx - d_P_dx/self.rho - self.uz*d_ux_dz - self.central_diff_z(self.viscocity(self.n, self.rho, d_uz_dx, d_ux_dz))/self.rho

        d_uz_dt = -self.uz*d_uz_dz - d_P_dz/self.rho - self.ux*d_uz_dx - self.central_diff_x(self.viscocity(self.n, self.rho, d_uz_dx, d_ux_dz))/self.rho

        # derivatives to use in eq. of motion

        d_rho_dt = -self.rho*d_ux_dx -self.ux*d_rho_dx - self.rho*d_uz_dz -self.uz*d_rho_dz        
        dt = self.variable_step_length(self.ux, self.uz, self.rho, d_rho_dt, d_ux_dt, d_uz_dt, dt, d_ux_dx, d_uz_dz)
        self.rho[:,:] = self.rho + dt*d_rho_dt            
        self.ux[:,:] = self.ux + dt*d_ux_dt
        self.uz[:,:] = self.uz + dt*d_uz_dt
        self.P[:,:] = self.rho/(self.u*self.mu)*self.kb*self.T
        
        return dt


solver = Project3(200, 200)
solver.initial_conditions()
#print solver.step()

vis = FVis.FluidVisualiser()
vis.save_data(1000, solver.step, rho=solver.rho, u=solver.ux, w = solver.uz, P=
solver.P, sim_fps=0.5)

vis.animate_2D("rho")
vis.delete_current_data()

