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
        self.dx = 15e6/(nx-1)        
        self.dz = 5e6/(nz-1)
        self.n = 0

        self.rho = np.zeros((nx, nz)) 
        self.P = np.zeros((nx, nz))
        self.ux = np.zeros((nx, nz))
        self.uz = np.zeros((nx,  nz))
        self.T = np.zeros((nx,  nz))
        self.e = np.zeros((nx,  nz))
        self.T_innit = np.zeros((nx,  nz))
        self.nabla = 3.0/5.0
        self.G = 6.67408e-11
        self.M_sun = 1.989e30
        self.R_sun = 695700000
        self.g = self.G*self.M_sun/self.R_sun**2

    def initial_conditions(self):
        self.P[-1,:] = 1.8e8         # solar photosphere [SI]        
        self.T[-1,:] = 5778          # solar photosphere [SI]
        self.rho[-1,:] = self.P[-1,:]/self.T[-1,:]/self.kb*self.u*self.mu/self.nabla
        for i in xrange(self.nx-1, 0, -1):
            dP_dz = self.rho[i-self.nx,:]*self.g
            dT_dz = self.nabla*self.T[i-self.nx,:]/self.P[i,:]*dP_dz
            self.T[i-1,:] = self.T[i,:] + self.dz*dT_dz
            self.P[i-1,:] = self.P[i,:] + self.dz*dP_dz
            self.rho[i-1,:] = self.P[i-1,:]/self.T[i-1,:]/self.nabla*self.u*self.mu/self.kb
        self.e[:,:] = self.rho*self.T*self.kb/(self.u*self.mu)
        self.T_top = self.T[-1,:]
        self.T_bot = self.T[0,:]

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

    def upwind_z_u(self, phi, u):
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
 
    def variable_step_length(self, ux, uz, rho, e, P, d_rho_dt, d_ux_dx, d_uz_dz, d_e_dt):
        rel_rho = np.amax(self.rel_phi(rho, d_rho_dt))
        rel_ux_v = np.amax(self.rel_v(ux, self.dx))
        rel_uz_v = np.amax(self.rel_v(uz, self.dz))
        rel_e = np.amax(self.rel_phi(e, d_e_dt))
        delta = max(rel_rho, rel_ux_v, rel_uz_v, rel_e)
        #print rel_rho, rel_ux_v, rel_uz_v, rel_e
        if delta == 0.0:
            self.cs = np.sqrt(5.0/3.0*P/rho)
            dt = 0.3*self.dx/np.amax(self.cs)
            
        else:
            dt =0.1/delta

        return dt
    
    def step(self):    
        # x-derivatives to use in eq. of motion
        d_ux_dx = self.central_diff_x(self.ux)
        d_P_dx = self.central_diff_x(self.P)
        d_uz_dx = self.central_diff_x(self.uz)
        d_rho_dx = self.upwind_x(self.rho, self.ux)
        d_e_dx = self.upwind_x(self.e, self.ux)
        d_e_dz = self.upwind_z(self.e, self.uz)


        # z-derivatives to use in eq. of motion
        d_uz_dz = self.central_diff_z(self.uz)
        d_ux_dz = self.central_diff_z(self.ux)
        
        
        d_ux_dz[0,:] = 0.0
        d_ux_dz[-1,:] = 0.0

        d_P_dz = self.central_diff_z(self.P)
        d_P_dz[0,:] = self.g*self.rho[0,:]
        d_P_dz[-1,:] = self.g*self.rho[-1,:]
        rho_boundary = d_P_dz*1/self.g
        #d_e_dz = rho_boundary/(self.u*self.mu)*self.kb*self.T
        d_rho_dz = self.upwind_z(self.rho, self.uz)

        d_ux_dt = -self.ux*d_ux_dx - d_P_dx/self.rho - self.uz*d_ux_dz - self.central_diff_z(self.viscocity(self.n, self.rho, d_uz_dx, d_ux_dz))/self.rho

        d_uz_dt = -self.uz*d_uz_dz - d_P_dz/self.rho - self.ux*d_uz_dx - self.central_diff_x(self.viscocity(self.n, self.rho, d_uz_dx, d_ux_dz))/self.rho - self.g/self.rho
        
        # derivatives to use in eq. of motion
        d_e_dt = -self.ux*d_e_dx -self.e*d_ux_dx - self.uz*d_e_dz - self.e*d_uz_dz - self.P*(d_ux_dx + d_uz_dz)
        d_rho_dt = -self.rho*d_ux_dx -self.ux*d_rho_dx - self.rho*d_uz_dz -self.uz*d_rho_dz        
        dt = self.variable_step_length(self.ux, self.uz, self.rho, self.e, self.P, d_rho_dt, d_ux_dx, d_uz_dz, d_e_dt)
        self.rho[:,:] = self.rho + dt*d_rho_dt            
        self.ux[:,:] = self.ux + dt*d_ux_dt
        self.uz[:,:] = self.uz + dt*d_uz_dt
        self.e[:,:] = self.e + dt*d_e_dt
        
        self.rho[0,:] = d_P_dz[0,:]/self.g
        self.rho[-1,:] = d_P_dz[-1,:]/self.g

        self.e[0,:] = d_P_dz[0,:]/self.g/(self.u*self.mu)*self.kb*self.T[0,:]
        self.e[-1,:] = d_P_dz[-1,:]/self.g/(self.u*self.mu)*self.kb*self.T[-1,:]

        self.T[:,:] = self.e/self.rho/self.kb*self.u*self.mu
        
        self.uz[0,:] = 0.0
        self.uz[-1,:] = 0.0

        self.T[0,:] = 1.1*self.T_bot
        self.T[-1,:] = 0.9*self.T_top
        self.P[:,:] = self.rho/(self.u*self.mu)*self.kb*self.T
        return dt


solver = Project3(10, 10)
solver.initial_conditions()
#print solver.step()

vis = FVis.FluidVisualiser()
vis.save_data(100, solver.step, rho=solver.rho, u=solver.ux, w = solver.uz, P=
solver.P, sim_fps=0.5)

vis.animate_2D("rho")
vis.delete_current_data()

