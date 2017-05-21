import FVis
import numpy as np
import matplotlib.pyplot as plt

class Project3:
    def __init__(self, nx, nz):   
        self.kb = 1.38064852e-23            # Boltzmann constant
        self.mu = 1.6605e-27                # Average nuclear weight
        self.nz = nz                        # Grid points in z-direction
        self.nx = nx                        # Grid points in x-direction
        self.u = 0.6                        # Atomic mass unit
        
        self.dx = 15e6/(nx-1)               # Step size in x-direction
        self.dz = 5e6/(nz-1)                # Step size in x-direction
        self.n = 0                          # Viscocity constant

        self.rho = np.zeros((nz, nx))       # Array for density
        self.P = np.zeros((nz, nx))         # Array for pressure
        self.ux = np.zeros((nz, nx))        # Array for x-velocity
        self.uz = np.zeros((nz,  nx))       # Array for z-velocity
        self.T = np.zeros((nz,  nx))        # Array for temperature
        self.e = np.zeros((nz,  nx))        # Array for energy

        self.nabla = 3./5.0                 # Teperature gradient
        self.G = 6.67408e-11                # Gravitational constant
        self.M_sun = 1.989e30               # Mass sun
        self.R_sun = 695700000              # Radius sun
        self.g = self.G*self.M_sun/self.R_sun**2    # Gravitational force

    def initial_conditions(self):                
        """ This method sets the initial conditions of the system aswell as saves values
            to be used as boundary conditions"""
        self.P[-1,:] = 1.8e8                # solar photosphere [SI]        
        self.T[-1,:] = 5778                 # solar photosphere [SI]
        self.rho[-1,:] = 2e-4               # solar photosphere [SI]
        
        # Have to turn the matrix to fit the plot
        for i in xrange(self.nx-1, 0, -1):
            """ Setting the initial conditions for the spatial grid. 
                Using normal euler method for the first step and then
                central difference for the rest """
            if i == self.nx-1:
                dP_dz = self.rho[i,:]*self.g                                            # Hydrostatic equilibrium
                dT_dz = self.nabla*self.T[i,:]/self.P[i,:]*dP_dz                        # From nabla
                self.T[i-1,:] = self.T[i,:] + self.dz*dT_dz
                self.P[i-1,:] = self.P[i,:] + self.dz*dP_dz
                self.rho[i-1,:] = self.P[i-1,:]/self.T[i-1,:]*self.u*self.mu/self.kb    # From EOS
            else:                        
                dP_dz = self.rho[i,:]*self.g
                dT_dz = self.nabla*self.T[i,:]/self.P[i,:]*dP_dz
                self.P[i-1,:] = self.P[i+1,:] + 2*self.dz*dP_dz
                self.T[i-1,:] = self.T[i+1,:] + 2*self.dz*dT_dz
                self.rho[i-1,:] = self.P[i-1,:]/self.T[i-1,:]*self.u*self.mu/self.kb

        
        
        dT = self.mu*self.u*self.g*self.nabla/self.kb*self.dz                           # From nabla definition

        self.center1 = np.asarray((self.nz*(1.0/10.0), self.nx/2))                      # Density perturbation center
        for i in xrange(self.nx):
            for j in xrange(self.nz):
                """ Setting the density perturbation """
                if np.linalg.norm(self.center1 - np.asarray((i, j))) <= 5.0:
                    self.rho[i, j] *= 5.2

        
        self.e[:,:] = self.rho*self.T*self.kb/(self.u*self.mu)      # From the energy equaiton
        self.T_init = np.array([1.1*self.T[0],0.9*self.T[-1]])      # lower and upper init conditions scaled +- 10%


        """ Top and bottom values as described in the rapport """
        dP_t = -self.g*self.rho[-1]*self.dz; 
        dP_b = self.g*self.rho[0]*self.dz  
        dT_t = self.T_init[1] - self.T[-1]; 
        dT_b = self.T_init[0] - self.T[0]
        
        self.rho_t = self.rho[-1] + self.u*self.mu/self.kb*(self.T[-1]*dP_t - self.P[-1]*dT_t)/self.T[-1]**2
        self.rho_b = self.rho[0] + self.u*self.mu/self.kb*(self.T[0]*dP_b - self.P[0]*dT_b )/self.T[0]**2
        
        self.e_t = self.kb/(self.u*self.mu)*self.rho_t*self.T_init[1]
        self.e_b = self.kb/(self.u*self.mu)*self.rho_b*self.T_init[0]
        
        self.P_t = self.P[-1] - self.g*self.rho[-1]*self.dz
        self.P_b = self.P[0] + self.g*self.rho[0]*self.dz
        

    def upwind_x(self, phi, u):
        """ Differentiates in x-direction """
        if np.size(u) == self.nx*self.nz:
            der = np.zeros((self.nx, self.nz))
            derivate_negative_u = (np.roll(phi, -1, axis = 1) - phi)/self.dx
            derivate_positive_u = (phi - np.roll(phi, 1, axis = 1))/self.dx

            der[np.where(u < 0)] = derivate_negative_u[np.where(u < 0)]
            der[np.where(u >= 0)] = derivate_positive_u[np.where(u >= 0)]
        
        return der
    
    def central_diff_x(self, phi):
        """ Differentiates in x-direction """
        der = np.zeros((self.nx, self.nz))
        der = (np.roll(phi, -1, axis = 1) - np.roll(phi, 1, axis = 1))/(2*self.dx)    
        return der
           
    def upwind_z(self, phi, u, param ='u'):
        """ Differentiates in z-direction and sets boundary conditions """
        der = np.zeros((self.nx, self.nz))

        if param == 'rho' or param == 'e' or param == 'P':
            """ Setting boundary conditions """
            phi_neg = np.roll(phi, 1, axis = 0)
            phi_pos = np.roll(phi, -1, axis = 0)
            if param == 'rho':
                phi_pos[-1] = self.rho_t
                phi_neg[0] = self.rho_b
            if param == 'e':
                phi_pos[-1] = self.e_t
                phi_neg[0] = self.e_b
            if param == 'P':
                phi_pos[-1] = self.P_t
                phi_neg[0] = self.P_b
            
            derivate_negative_u = (phi_pos - phi)/self.dx
            derivate_positive_u = (phi - phi_neg)/self.dx
            der[np.where(u < 0)] = derivate_negative_u[np.where(u < 0)]
            der[np.where(u >= 0)] = derivate_positive_u[np.where(u >= 0)]
                 
        else: # Then velocity   
            phi_neg = np.roll(phi, 1, axis = 0)       # funcL = func[i-1] for all i
            phi_pos = np.roll(phi, -1, axis = 0)

            """ Boundary conditions"""
            phi_neg[0,:] = 0
            phi_pos[-1,:] = 0

            derivate_negative_u = (phi_pos - phi)/self.dx
            derivate_positive_u = (phi - phi_neg)/self.dx
            der[np.where(u < 0)] = derivate_negative_u[np.where(u < 0)]
            der[np.where(u >= 0)] = derivate_positive_u[np.where(u >= 0)]

            """ Boundary condition for dux/dz"""            
            if param == "ux":
                der[-1,:] = 0
                der[0,:] = 0                

        return der
    
    def central_diff_z(self, phi, param = 'u'):
        """ Differentiates in z-direction and sets boundary conditions """
        if param == 'rho' or param == 'e' or param == 'P':
            phi_pos = np.roll(phi, -1, axis = 0)
            phi_neg = np.roll(phi, 1, axis = 0)
            if param == 'rho':
                phi_pos[-1,:] = self.rho_t
                phi_neg[0,:] = self.rho_b
            if param == 'e':
                phi_pos[-1,:] = self.e_t
                phi_neg[0,:] = self.e_b
            if param == 'P':
                phi_pos[-1,:] = self.P_t
                phi_neg[0,:] = self.P_b
            der = (phi_pos - phi_neg)/(2*self.dz)

        else:
            phi_pos = np.roll(phi, -1, axis = 0)
            phi_neg = np.roll(phi, 1, axis = 0)

            """ Boundary conditions"""
            phi_pos[-1,:] = 0
            phi_neg[0,:] = 0

            der = (phi_pos - phi_neg)/(2*self.dz)

            """ Boundary condition for dux/dz"""            
            if param == 'ux':
                der[-1,:] = 0
                der[0,:] = 0  
        return der    

    def rel_phi(self, phi, dphi):
        """Computes delta for time derivatives"""
        return abs(dphi/phi)
    
    def rel_v(self, v, dx):
        """Computes delta for spatial derivatives"""
        return abs(v/dx)
 
    def variable_step_length(self, ux, uz, rho, e, P, d_rho_dt, d_ux_dx, d_uz_dz, d_e_dt):
        """Computes the optimal time step """
        rel_rho = np.amax(self.rel_phi(rho, d_rho_dt))
        rel_ux_v = np.amax(self.rel_v(ux, self.dx))
        rel_uz_v = np.amax(self.rel_v(uz, self.dz))
        rel_e = np.amax(self.rel_phi(e, d_e_dt))
        delta = max(rel_rho, rel_ux_v, rel_uz_v, rel_e)

        """Only used for the first time step"""
        if delta == 0.0:
            dt =  0.1

        else:
            # P = 0.1
            dt =0.1/delta
        return dt
    
    def step(self):  
        """ Computing all the different derivatives needed """  

        dux_xc = self.central_diff_x(self.ux)
        duz_zc = self.central_diff_z(self.uz)
        drho_xu = self.upwind_x(self.rho, self.ux)  
        drho_zu = self.upwind_z(self.rho, self.uz, "rho")  
        drho_t = - self.rho*(dux_xc + duz_zc) - self.ux*drho_xu - self.uz*drho_zu
        
        dux_xu = self.upwind_x(self.ux, self.ux)
        drho_xc = self.central_diff_x(self.rho)
        dux_zu = self.upwind_z(self.ux, self.uz, "ux")
        dP_xc = self.central_diff_x(self.P)

        duz_zu = self.upwind_z(self.uz, self.uz)        
        drho_zc = self.central_diff_z(self.rho, "rho")
        duz_xu = self.upwind_x(self.uz, self.ux)

        dP_zc = self.central_diff_z(self.P, "P")
        
        dux_t = - self.ux*dux_xu - dP_xc/self.rho - self.uz*dux_zu
        duz_t = - self.uz*duz_zu - dP_zc/self.rho - self.ux*duz_xu + self.g/self.rho

        #Viscosity
        drho_xc = self.central_diff_x(self.rho)
        drho_zc = self.central_diff_z(self.rho, 'rho')
        dux_zc = self.central_diff_z(self.ux, 'ux')
        duz_xc = self.central_diff_x(self.uz,)
        visc = dux_zc + duz_xc
        
        dtau_z = self.n*drho_zc*visc
        dtau_x = self.n*drho_xc*visc
        dux_t += dtau_z/self.rho
        duz_t += dtau_x/self.rho

        # Energy
        d_e_dx = self.upwind_x(self.e, self.ux)
        d_e_dz = self.upwind_z(self.e, self.uz, "e")
        d_e_dt = -self.ux*d_e_dx -self.e*dux_xc - self.uz*d_e_dz - self.e*duz_zc - self.P*(dux_xc + duz_zc)

        dt = self.variable_step_length(self.ux, self.uz, self.rho, self.e, self.P, drho_t, dux_xc, duz_zc, d_e_dt)

        self.rho[:,:] = self.rho + dt*drho_t            
        self.ux[:,:] = self.ux + dt*dux_t
        self.uz[:,:] = self.uz + dt*duz_t
        self.e[:,:] = self.e + dt*d_e_dt
        self.T[:,:] = self.e/self.rho/self.kb*self.u*self.mu
        self.P[:,:] = self.rho/(self.u*self.mu)*self.kb*self.T
        return dt


solver = Project3(100, 100)
solver.initial_conditions()

vis = FVis.FluidVisualiser()
vis.save_data(100, solver.step, T = solver.T, e = solver.e, rho=solver.rho, u=solver.ux, w = solver.uz, P=
solver.P, sim_fps=0.5)
vis.animate_2D("rho")#, snapshots=[70, 140, 210, 280])
vis.delete_current_data()
