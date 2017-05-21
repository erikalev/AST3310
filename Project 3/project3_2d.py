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
        self.T = 1e-4


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
            derivate_negative_u = (np.roll(phi, -1, axis = 0) - phi)/self.dx
            derivate_positive_u = (phi - np.roll(phi, 1, axis = 0))/self.dx

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
            derivate_negative_u = (np.roll(phi, -1, axis= 1 ) - phi)/self.dz
            derivate_positive_u = (phi - np.roll(phi, 1, axis= 1 ))/self.dz

            der[np.where(u < 0)] = derivate_negative_u[np.where(u < 0)]
            der[np.where(u >= 0)] = derivate_positive_u[np.where(u >= 0)]

        else:
            print "You have chosen dz without having a z-dimention!"
        return der                
            
    def central_diff_x(self, phi):
        if np.size(self.ux) == self.nx*self.nz:
            der = np.zeros((self.nx, self.nz))
            der = (np.roll(phi, -1, axis = 0) - np.roll(phi, 1, axis = 0))/(2*self.dx)    
        else:
            der = np.zeros(self.nx)
            der = (np.roll(phi, -1) - np.roll(phi, 1))/(2*self.dx)    
        return der

    def central_diff_z(self, phi):
        if np.size(self.ux) == self.nx*self.nz:
            der = np.zeros((self.nx, self.nz))        
            der = (np.roll(phi, -1, axis = 1) - np.roll(phi, 1, axis = 1))/(2*self.dz)    
        else:
            print "You have chosen dz without having a z-dimention!"
        return der

    def viscocity(self, rho, d_ux_dz, d_uz_dx):
        return -self.n*rho*(d_ux_dz + d_uz_dx)


    def rel_phi(self, phi, dphi):
        return abs(dphi/phi)
    
    def rel_v(self, v, dx):
        return abs(v/dx)
 
    def variable_step_length(self, ux, uz, rho, d_rho_dt, d_ux_dt, d_uz_dt, d_ux_dx, d_uz_dz):
        rel_rho = self.rel_phi(rho, d_rho_dt)        
        if 0.0 in ux:
            rel_ux = np.array([0, 0])
        else:
            rel_ux = self.rel_phi(ux, d_ux_dt)
        if 0.0 in uz:
            rel_uz = np.array([0, 0])
        else:
            rel_uz = self.rel_phi(uz, d_uz_dt)

        rel_ux_v = self.rel_v(ux, self.dx)
        rel_uz_v = self.rel_v(uz, self.dz)
        delta = max(rel_rho.max(), rel_ux_v.max(), rel_uz_v.max(), rel_uz.max(), rel_ux.max())

        if delta == 0.0:
            dt = 0.1

        else:
            dt =0.1/delta

        return dt
    def step(self):
        # Calculate RHS drho/dt
        dux_xc = self.central_diff_x(self.ux)
        duz_zc = self.central_diff_z(self.uz)
        drho_xu = self.upwind_x(self.rho, self.ux)  # give ux to check
        drho_zu = self.upwind_z(self.rho, self.uz)  # give uz
        drho_t = -self.rho*(dux_xc + duz_zc) - self.ux*drho_xu - self.uz*drho_zu

        # Calculate RHS dux/dt
        dux_xu = self.upwind_x(self.ux, self.ux)
        drho_xc = self.central_diff_x(self.rho)
        dux_zu = self.upwind_z(self.ux, self.uz)

        duz_xc = self.central_diff_x(self.uz)
        dux_zc =self.central_diff_z(self.ux)

        dux_t = - (self.ux*dux_xu + drho_xc/self.rho + self.uz*dux_zu) -self.central_diff_z(self.viscocity(self.rho, duz_xc, dux_zc))/self.rho
        
        # Calculate RHS duz/dt
        duz_zu = self.upwind_z(self.uz, self.uz)
        drho_zc = self.central_diff_z(self.rho)
        duz_xu = self.upwind_x(self.uz, self.ux)
        duz_t = - (self.uz*duz_zu + drho_zc/self.rho + self.ux*duz_xu) - self.central_diff_x(self.viscocity(self.rho, duz_xc, dux_zc))/self.rho

        #d_rho_dt = -self.rho*(d_ux_dx + d_uz_dz) - self.ux*d_rho_dx -self.uz*d_rho_dz        
        dt = self.variable_step_length(self.ux, self.uz, self.rho, drho_t, dux_t, duz_t, dux_xc, duz_zc)

        self.rho[:,:] = self.rho + dt*drho_t            
        self.ux[:,:] = self.ux + dt*dux_t
        #print duz_t[74:76, 74:76]
        self.uz[:,:] = self.uz + dt*duz_t
        self.P[:,:] = self.rho/(self.u*self.mu)*self.kb*self.T

        return dt


solver = Project3(200, 200)
#solver.init_c(1e-4)
solver.initial_conditions()
#print solver.step()

vis = FVis.FluidVisualiser()
vis.save_data(200, solver.step, rho=solver.rho, u=solver.ux, w = solver.uz, P=
solver.P, sim_fps=1)
vis.animate_2D("rho")#  , save=True)
vis.delete_current_data()

