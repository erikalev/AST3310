from scipy import interpolate as inter
import numpy as np
import matplotlib.pyplot as plt
import sys
from numpy.polynomial import Polynomial as P

class project2:
    def __init__(self, L_0, M_0, R_0, rho_0, T_0, X, Y, Y3, Y4, Z, Z7Li, Z7Be):   
        self.R_0 = R_0                          # initial radius
        self.L_0 = L_0                          # initial luminosity
        self.M_0 = M_0                          # initial mass
        self.rho_0 = rho_0                      # initial density
        self.T_0 = T_0                          # initial temperature

        self.L_sun = 3.846e26        # Sun's Luminosity [W]
        self.R_sun = 6.96e8          # Sun's radius [m]
        self.M_sun = 1.989e30        # Sun's mass [kg]
        self.rho_star_avg = 1.408e3  # Sun's average density [kg m^-3]

        self.X, self.Y, self.Y3, self.Y4, self.Z, self.Z7Li, self.Z7Be = X, Y, Y3, Y4, Z, Z7Li, Z7Be # initial mass fractions        
        self.c = 299792458                      # Speed of light [m/s]
        self.sigma = 5.670367e-8                # Stefan-Boltzmann constant [J m^-2 s^-1 K^-4]
        self.k_b = 1.38064852e-23               # Boltzmann constant [m^2 kg s^-2 K^-1]
        self.u = 1.0/(2*X + 3.0/4.0*Y + Z/2.0)  # Mean molecular weight "dimensionless"
        self.m_u = 1.6605e-27                   # Atomic mass unit "dimensionless"
        self.G = 6.67408e-11                    # Gravitational constant [m^3 kg-1 s-2]
        self.Cp = 5./2.*self.k_b/(self.u*self.m_u)
        self.delta = 1

        """ Performing the linear 2D interpolation o6f the opacity values. These will be used 
        later on but performing the interpolation in the innit function allows us to just 
        call the ready function later on. We therefore only have to interpolate once which 
        greatly improves the time performance of the programme"""

        data = np.genfromtxt("/home/erik/FAG/AST3310/opacity.txt")  # Collecting all data in one array
        logT = data[1:,0]                                           # logT-values 
        logR = data[0,1:]                                           # logR-values
        n_T = len(logT)                                             # Number of logT-values 
        n_R = len(logR)                                             # Number of logR-values
        self.f = inter.interp2d(logR, logT, data[1:, 1:], bounds_error = False)           # Performing the interpolation

    def opacity(self, T, rho):
        """ Function which takes emperature and density in SI-units and returns the 
        opacity value corresponding to these values from the interpolation-function 
        in the innit-function"""

        rho = rho/1000.0                           # Converting the density to cgs
        R = np.log10(rho/((T/1e6)**3))             # As defined by Appendix D
        T = np.log10(T)        
        
        return float(10**(self.f(R, T))/10.0)     # Returning the opacity in SI-units

    def PP1_chain(self, r33, Q33):
        """Last reaction in the PP1 chain. The first 2 reaction are the same
        for both PP1 and PP2 and can therefore be calculated outside this function.
        Takes arguments reaction rates [kg s^-1] and energy produced in unit [MeV] """
        return r33*Q33

    def PP2_chain(self, r34, r17, re7, Q34, Q17, Qe7):
        """Last three reactions in the PP2 chain. The first 2 reaction are calculated 
        outside this function. Takes arguments reaction rates [kg s^-1] for all three 
        reaction and energy produced in units [MeV] for all three reactions."""
        return r34*Q34 + r17*Q17 + re7*Qe7

    def energy_produced(self, T, rho):
        """ Method for calculating the energy production in the PP1 and PP2 chains.
        Takes arguments temperature and density, both in SI units, and returns
        the energy pr second pr kg [J kg^-1 s^-1]"""

        MeVToJoule = 1.602e-13          # Converting MeV to Joule
        NA = 6.022e23                   # Avogadros number (dimensionless)
        T9 = float(T)/1e9                      # Units used to find the lambda-values [K]

        # Proportionality functions [m^-3 s^-1]
        lambda_pp = (4.01e-15*T9**(-2.0/3)*np.exp(-3.380*T9**(-1.0/3))*(1 + 0.123*T9**(1.0/3.0) + 1.09*T9**(2.0/3.0) + 0.938*T9))/NA/1e6
        lambda_33 = (6.04e10*T9**(-2.0/3.0)*np.exp(-12.276*T9**(-1.0/3.0))*(1 + 0.034*T9**(1.0/3.0) - 0.522*T9**(2.0/3.0) - 0.124*T9 + 0.353*T9**(4.0/3.0) + 0.213*T9**(-5.0/3.0)))/NA/1e6;
        T9star = T9/(1.0 + 4.95e-2*T9);
        lambda_34 = (5.61e6*T9star**(5.0/6.0)*T9**(-3.0/2.0)*np.exp(-12.826*T9star**(-1.0/3.0)))/NA/1e6;
        lambda_e7 = (1.34e-10*T9**(-1.0/2.0)*(1 - 0.537*T9**(1.0/3.0) + 3.86*T9**(2.0/3.0) + 0.0027/T9*np.exp(2.515e-3/T9)))/NA/1e6;
        T9star2 = T9/(1 + 0.759*T9);
        lambda_17 = (1.096e9*T9**(-2.0/3.0)*np.exp(-8.472*T9**(-1.0/3.0)) - 4.830e8*T9star2**(5.0/6.0)*T9**(-3.0/2.0)*np.exp(-8.472*T9star2**(-1.0/3.0)) + 1.06e10*T9**(-3.0/2.0)*np.exp(-30.442*T9**(-1)))/NA/1e6;

        # Number densities
        n_p = self.X*rho/self.m_u
        n_He3 = self.Y3*rho/(3*self.m_u)
        n_He4 = self.Y4*rho/(4*self.m_u)
        n_Be7 = self.Z7Be*rho/(7*self.m_u)
        n_Li7 = self.Z7Li*rho/(7*self.m_u)
        n_e = n_p + 2*n_He3 + 2*n_He4 + self.Z/2.0

        # If-test to apply the upper limit of Be7
        if (T < 1e6):
            if (NA*lambda_e7 > 1.57e-7/n_e):
                #print "Upper limit for Be7 was acchieved"
                lambda_e7 = 1.57e-7/(n_e*NA)

        #Reaction rates pr unit mass. These are really multiplied with a factor rho. [s^(-1) m^(-3)]
        rpp = n_p*n_p/2.0*lambda_pp
        r33 = n_He3*n_He3/2.0*lambda_33
        r34 = n_He3*n_He4*lambda_34
        r17 = n_Li7*n_p*lambda_17
        re7 = n_Be7*n_e*lambda_e7

        #Energys from the different reactions in PP1 and PP2 in units [MeV]
        Qpp = 0.15 + 1.02
        Qpd = 5.49
        Q33 = 12.86
        Q34 = 1.59
        Qe7 = 0.05
        Q17 = 17.35

        # Total power/volume to return
        E = 0
        PP1_E = 0
        PP2_E = 0
        #First 2 reaction in PP1 and PP2 combined
        First_energy_step = rpp*(Qpp + Qpd)*MeVToJoule
        E += First_energy_step
        # PP1 and PP2 Chain
        
        if (2*r33 + r34) > rpp:
            scale = rpp/(2*r33 + r34)
            r33 = scale*r33
            r34 = scale*r34

        if re7 > r34:
            re7 = r34

        if r17 > re7:
            r17 = re7
        
        PP1_E = self.PP1_chain(r33, Q33)*MeVToJoule
        PP2_E += self.PP2_chain(r34, r17, re7, Q34, Q17, Qe7)*MeVToJoule
        E += PP1_E + PP2_E 

        """
        if (rpp > 2*r33 + r34):
            # Do we produce more He3 than we consume            
            PP1_E = self.PP1_chain(r33, Q33)*MeVToJoule
            E += PP1_E
            if (r34 > re7):
            # Do we produce more Be7 than we consume
                if (re7 > r17):
                    # Do we produce more Li7 than we consume
                    PP2_E += self.PP2_chain(r34, r17, re7, Q34, Q17, Qe7)*MeVToJoule                            
                    E += PP2_E

                else:
                    # Production of He4 is goverened by the rate of production for Li7
                    PP2_e = self.PP2_chain(r34, re7, re7, Q34, Q17, Qe7)*MeVToJoule
                    E += PP2_E
            else:
                # Production of Li7 is goverend by the rate of production for Be7
                PP2_E = self.PP2_chain(r34, r34, r34, Q34, Q17, Qe7)*MeVToJoule
                E += PP2_E        
        else:
            # All energy production is goverened by the rate of productino for Deuterium
            scale = rpp/(2*r33 + r34)
            PP1_E = self.PP1_chain(rpp, Q33)*MeVToJoule
                        
            PP2_E = self.PP2_chain(r34*scale, r17*scale, re7*scale, Q34, Q17, Qe7)*MeVToJoule                
            E += PP1_E + PP2_E
        """
        return E, PP1_E, PP2_E

    def find_Pr(self, T):
        """ Method which takes temperature as argument and  
            returns the radiative pressure in the star """ 
        
        return 4*self.sigma/(3*self.c)*T*T*T*T

    def find_rho(self, P, T):
        """ Method which takes total pressure and temperature as arguments and  
            returns the density in the star when taking into 
            account an ideal gas and P = P_gass + P_radiation""" 

        Pg = P - self.find_Pr(T)
        return Pg*self.u*self.m_u/(self.k_b*T)

    def find_P(self, rho, T):
        """ Method which takes density and temperature as arguments and  
            returns the total pressure in the star when taking into 
            account an ideal gas and P = P_gass + P_radiation 
        """ 
        return rho/(self.u*self.m_u)*self.k_b*T + self.find_Pr(T)


    def plot(self, M, r, L, T, rho):        
        """Method which plots some of the achieved values"""
        
        #plotting the initial value sin subplots
        fig = plt.figure()
        ax = plt.subplot("221")
        ax.set_xlabel("Mass/$M_0$")
        ax.set_ylabel("R/$R_0$")
        ax.plot(M/M[0], r/r[0])

        ax = plt.subplot("222")
        ax.set_xlabel("Mass/$M_0$")
        ax.set_ylabel("L/$L_0$")
        ax.plot(M/M[0], L/L[0])

        ax = plt.subplot("223")
        ax.set_xlabel("Mass/$M_0$")
        ax.set_ylabel("T[MK]")
        ax.plot(M/M[0], T/1e6)

        ax = plt.subplot("224")
        #ax.set_ylim(10**0, 10**1)
        ax.set_xlabel("Mass/$M_0$")
        ax.set_ylabel("$\\rho/rho_0$")
        ax.plot(M/M[0], rho/rho[0])
        plt.subplots_adjust(hspace = .5)
        plt.subplots_adjust(wspace = .5)
        plt.show()

    def DSS(self, dm, dm1, r, rho, M, E, L, T, K, P, dTdm):
        """ Method to calculate the variable step length. Takes the old and
         the initial dm (dm and dm1) as input together with the variable values 
        and returns the new dm
        """      
        
        # Variable step length constant          
        p = 1e-2;

        # Establishing variables to perform the variable step length
        
        drdm = 1.0/(4*np.pi*r**2*rho)
        dPdm = -self.G*M/(4*np.pi*r**4)
        dLdm = E
        if dTdm == 0:
            dTdm =  -3*K*L/(256*np.pi**2*self.sigma*r**4*T**3)    
        test = np.zeros(5)
        test[0] = abs(p*r/drdm)
        test[1] = abs(p*P/dPdm)
        test[2] = abs(p*L/dLdm)
        test[3] = abs(p*T/dTdm)
        test[4] = abs(p*M)
        dm = -min(test)        
        if dm < dm1:
            dm = dm1

        return dm, dTdm
    
    def print_int(self, i, dm, rho, L, M, r, P, E, T):    
        """Method that just prints the values every designated time step """
        print "i = ", i
        print "dm = ", dm
        print "rho = ", rho
        print "L = ", L
        print "M = ", M
        print "r = ", r
        print "P = ", P
        print "E = ", E
        print "T = ", T
        print " "
        print "L/L_0= ", L/self.L_0
        print "M/M_0= ", M/self.M_0
        print "R/R_0= ", r/self.R_0
        print " "
    

    def test_small_enough_values(self, L, M, r):
        small_enough_values = False
        if (L/self.L_0 < 0.049):
            if (M/self.M_0 < 0.049):
                if (r/self.R_0 < 0.049):
                   small_enough_values = True
         
        return small_enough_values


    def test_negative_values(self, M, r, L, i):
        if (M[i] < 0):
            L_lim = L[i-1]/self.L_0
            M_lim = M[i-1]/self.M_0
            R_lim = r[i-1]/self.R_0

            print "Negative mass achieved"
            print "i = ", i
            print "L/L_0 = ", L_lim
            print "M/M_0 = ", M_lim
            print "R/R_0 = ", R_lim
            negative_values = True
            return negative_values        

        elif (r[i] < 0): 
            L_lim = L[i-1]/self.L_0
            M_lim = M[i-1]/self.M_0
            R_lim = r[i-1]/self.R_0

            print "Negative radius achieved"
            print "i = ", i
            print "L/L_0 = ", L_lim
            print "M/M_0 = ", M_lim
            print "R/R_0 = ", R_lim
            negative_values = True
            return negative_values        

        elif (L[i] < 0):
            L_lim = L[i-1]/self.L_0
            M_lim = M[i-1]/self.M_0
            R_lim = r[i-1]/self.R_0

            print "Negative luminosity achieved"
            print "i = ", i
            print "L/L_0 = ", L_lim
            print "M/M_0 = ", M_lim
            print "R/R_0 = ", R_lim
            negative_values = True
            return negative_values        
        else:
            negative_values = False
            return negative_values        
        

    
    def nabla_ad(self, P, delta, T, rho):
        """Method which returns the temperature gradient for the 
           adiabatic parcel """
        return P*delta/(T*rho*self.Cp)

    def nabla_st(self, L, K, rho, Hp, r, T):
        """Method which returns the temperature gradient for the 
           parcel at stable conditions"""
        return 3*L*K*rho*Hp/(64*np.pi*r**2*self.sigma*T**4)

    def xi(self, U, lm, nabla_ad, nabla_st):
        """Method which returns the solution to the third degree polynomial as described in
        the solution to exercise 5.13"""        
        B = U/(lm**2)
        K = 4*B
        p = np.poly1d([1./B, 1.0, K, nabla_ad-nabla_st])
        r = np.roots(p)
        r = r[np.isreal(r)]
        return np.linalg.norm(r)        
        
    def nabla_star(self, U, lm, nabla_ad, nabla_st):
        """Method which returns the temperature gradient for the 
           star """
        xi = self.xi(U, lm, nabla_ad, nabla_st)
        return xi**2 + xi*(4*U/lm**2) + nabla_ad

    def Fc(self, U, lm, nabla_ad, nabla_st, rho, Cp, T, g, delta, Hp):
        """ Method which return the convective flux """
        xi = self.xi(U, lm, nabla_ad, nabla_st)
        return rho*Cp*T*np.sqrt(g*delta)*Hp**(-3./2.)*(lm/2.)**2*xi**3

    def Fr(self, T, K, M, rho, Hp, nabla_star):
        """ Method whoch returns the rediative flux """
        return 16*self.sigma*T**4/(3*K*rho*Hp)*nabla_star
    
    def calculate_convection(self, U, lm, nabla_ad, nabla_st, nabla_star, rho, T, g, Hp, K, M, P, r):
        """ This method was basically just to make the code a little bit more understandable. 
            It only call self.Fc and self.Fr """
        Fc = self.Fc(U, lm, nabla_ad, nabla_st, rho, self.Cp, T, g, self.delta, Hp)
        Fr  = self.Fr(T, K, M, rho, Hp, nabla_star)
        return Fc, Fr

    def integrate(self, dm, N, dynamic_step_size = False):
        """ Method which takes the the mass step length dm, number of integration 
            points N and the dynamic step size boolean and integrates over N dm-steps. 
            If none declared in the call, dymamic step size is automatically turned 
            off. The method also checks if there is is negative mass, distance or 
            luminosity. If so an error message is printed out and th plot-method is 
            called to plot the final results. 
        """ 
        K = self.opacity(self.T_0, self.rho_0)    # Initial Kappa value 
        #Initializing vectors to be filled in with values 
        L = np.zeros(N)                 # Luminosity
        rho = np.zeros(N)               # Density
        T = np.zeros(N)                 # Temperature
        r = np.zeros(N)                 # Radial distance
        P = np.zeros(N)                 # Pressure
        M = np.zeros(N)                 # Mass
        E = np.zeros(N)                 # Power/volume
        nabla_st = np.zeros(N)          # Temperature gradient stable
        nabla_star = np.zeros(N)        # Temperature gradient star
        nabla_ad = np.zeros(N)          # Temperature gradient adiabatic
        Fr_frac = np.zeros(N)           # Radiative flux fraction
        Fc_frac = np.zeros(N)           # Convective flux fraction
        PP1_E = np.zeros(N)             # Energy from PP1
        PP2_E = np.zeros(N)             # Energy from PP2
        PP1_frac = np.zeros(N)          # Energy fraction from PP1
        PP2_frac = np.zeros(N)          # Energy fraction from PP1

        # Setting initial values for vectors
        L[0] = self.L_0
        rho[0] = self.rho_0
        T[0] = self.T_0
        r[0] = self.R_0
        M[0] = self.M_0
        E[0] = self.energy_produced(self.T_0, self.rho_0)[0]/self.rho_0
        P[0] = self.find_P(self.rho_0, self.T_0)
        PP1_E[0] = self.energy_produced(self.T_0, self.rho_0)[1]
        PP2_E[0] = self.energy_produced(self.T_0, self.rho_0)[2]

        PP1_frac[0] = PP1_E[0]/(PP1_E[0] + PP2_E[0])
        PP2_frac[0] = PP2_E[0]/(PP1_E[0] + PP2_E[0])

        dm1 = dm                  # controle variable which is used to set dm back to it's original value

        negative_values = False
        small_enough_values = False

        for i in range(N-1):
            # updating the opacity value
            K = self.opacity(T[i], rho[i])

            # Updating values needed for the temperature gradients          
            g = self.G*M[i]/(r[i]**2)
            alpha = 1
            Hp = P[i]/(g*rho[i])    
            lm = Hp*alpha
            U = 64*self.sigma*T[i]**3/(3*K*rho[i]**2*self.Cp)*np.sqrt(Hp/(g*self.delta))
            
            # Temperature gradients
            nabla_ad[i] = self.nabla_ad(P[i], self.delta, T[i], rho[i])
            nabla_st[i] =  self.nabla_st(L[i], K, rho[i], Hp, r[i], T[i])
            nabla_star[i] = self.nabla_star(U, lm, nabla_ad[i], nabla_st[i])

            if nabla_st[i] > nabla_ad[i]:
                """If true then we take convection into account """ 
                Fc, Fr = self.calculate_convection(U, lm, nabla_ad[i], nabla_st[i], nabla_star[i], rho[i], T[i], g, Hp, K, M[i], P[i], r[i])
                dTdm = -T[i]*self.G*M[i]/(4*P[i]*r[i]**4*np.pi)*nabla_star[i]   
                Fc_frac[i] = Fc/(Fr + Fc)
                Fr_frac[i] = Fr/(Fr + Fc)
            
            else:
                """Continue with the same algorythms as in project 1 """
                nabla_star[i] = nabla_st[i]
                dTdm = 0
                Fc_frac[i] = 0
                Fr_frac[i] = 1
            
            # Tests to check that we dont achieve any negative values in M, r or L            
            if self.test_negative_values(M, r, L, i) == True:
                break
            
            # Restriction for the outer convection layer
            """
            if i < N/2:
                if Fc_frac[i-1] != 0:
                    if Fc_frac[i] == 0:
                        if r[i]/r[0] > 0.85:
                            break
            """

            # Restriction to the T(r) plot
            """
            if r[i]/r[0] < 0.3:
                if T[i]/T[0] < 0.1:
                    break
            """
         
            # Restriction of the value of the resulting variables
            """         
            if self.test_small_enough_values(L[i], M[i], r[i]) == True:                
                break
            """
            if dynamic_step_size == True:            
                dm, dTdm = self.DSS(dm, dm1, r[i], rho[i], M[i], E[i], L[i], T[i], K, P[i], dTdm)
            
            # Euler solvers
            M[i+1] = M[i] + dm
            r[i+1] = r[i] + dm*1.0/(4*np.pi*r[i]**2*rho[i])

            # Possible restriction of the lowest allowed radius fraction
            """
            if r[i+1]/r[0] < 0.04:
                break
            """

            P[i+1] = P[i] + dm*((-self.G*M[i])/(4*np.pi*r[i]**4))
            L[i+1] = L[i] + E[i]*dm
            T[i+1] = T[i] + dm*((-3*K*L[i])/(256*np.pi**2*self.sigma*r[i]**4*T[i]**3))
                
            # Updating the temperature for convection 
            if nabla_st[i] > nabla_ad[i]:
                T[i+1] = T[i] + dm*dTdm
        
            # Updating density and energy 
            rho[i+1] = self.find_rho(P[i+1], T[i+1])
            E[i+1] = self.energy_produced(T[i+1], rho[i+1])[0]/rho[i+1]

            # Energy from PP1 and PP2
            PP1_E[i+1] = self.energy_produced(T[i+1], rho[i+1])[1]/rho[i+1]
            PP2_E[i+1] = self.energy_produced(T[i+1], rho[i+1])[2]/rho[i+1]

            # Fraction from PP1 and PP2
            PP1_frac[i+1] = PP1_E[i+1]/(PP1_E[i+1] + PP2_E[i+1])
            PP2_frac[i+1] = PP2_E[i+1]/(PP1_E[i+1] + PP2_E[i+1])


            # Printing out the values with an appropriate interval         
            if (i%1==0):
                self.print_int(i, dm, rho[i], L[i], M[i], r[i], P[i], E[i], T[i])                                
    
        return r[:i], M[:i], L[:i], rho[:i], P[:i], T[:i], E[:i], nabla_star[:i], nabla_st[:i], nabla_ad[:i], PP1_E[:i], PP2_E[:i], PP1_frac[:i], PP2_frac[:i], Fc_frac[:i], Fr_frac[:i]
        
if __name__ == "__main__":        
    # Setting initial parameters in SI-units
    L_sun = 3.846e26        # Sun's Luminosity [W]
    R_sun = 6.96e8          # Sun's radius [m]
    M_sun = 1.989e30        # Sun's mass [kg]
    rho_star_avg = 1.408e3  # Sun's average density [kg m^-3]

    #Setting initial parameters for bottom of sun's convection zone 
    L_0 = 1.0*L_sun
    R_0 = 1.0*R_sun
    M_0 = 1.0*M_sun
    rho_0 = 1.42e-7*rho_star_avg
    T_0 = 5770
    # Setting mass fractions
    X = 0.7
    Y3 = 1e-10
    Y = 0.29
    Y4 = Y - Y3
    Z = 0.01
    Z7Li = 1e-13
    Z7Be = 1e-13

    # Number of integration points
    N = 25000

    # Calling the class and the integrate function
    A = project2(L_0, M_0, 1.65*R_0, 40*rho_0, 1.10175*T_0, X, Y, Y3, Y4, Z, Z7Li, Z7Be)
    r, M, L, rho, P, T, E, nabla_star, nabla_st, nabla_ad, PP1_E, PP2_E, PP1_frac, PP2_frac, Fc_frac, Fr_frac = A.integrate(-1e26, N, dynamic_step_size = True)
    
    print "PP1 frac = ", abs(np.trapz(PP1_frac, r/r[0]))
    print "PP2 frac = ", abs(np.trapz(PP2_frac, r/r[0]))

    # Plotting temperature gradients    
    plt.plot(r/r[0], nabla_st, "-r", r/r[0], nabla_star, r/r[0], nabla_ad)
    plt.legend(["$\\nabla_{st}$", "$\\nabla^{\\star}$", "$\\nabla_{ad}$"], loc="upper left")    
    plt.yscale("log")
    plt.xlabel("$R/R_0$")
    plt.ylabel("$log_{10}(\\nabla)$")
    plt.show()

    # Plotting contributions from PP1 and PP2 chains
    plt.plot(r/r[0], PP1_frac, r/r[0], PP2_frac, r/r[0], E/max(E))
    plt.ylabel("Energy fractions")
    plt.xlabel("$R/R_0$")
    plt.legend(["PP1", "PP2", "$\\epsilon (r)/\\epsilon _{max}$"], loc="center right")
    plt.show()
    
    # Plotting the radiation and convection contribution
    plt.plot(r/r[0], Fc_frac, r/r[0], Fr_frac)
    plt.ylabel("Flux fractions")
    plt.xlabel("$R/R_0$")
    plt.legend(["Flux from convection", "Flux from radiation"], loc="upper left")
    plt.show()    
    
    # Plotting the cross section   
    fig, ax = plt.subplots()
    circle1 = plt.Circle((0, 0), 0, color='red')
    circle2 = plt.Circle((0, 0), 0, color='yellow')
    circle3 = plt.Circle((0, 0), 0, color='cyan')
    circle4 = plt.Circle((0, 0), 0, color='blue')
    plt.legend([circle1, circle2, circle3, circle4], ["Convection outside core", "Radiation outside core", "Radiation inside core", "Convection inside core"], prop={'size':8})
    for i in xrange(len(r)):
        if i%10 == 0:        
            if nabla_st[i] > nabla_ad[i]:
                if L[i]/L[0] < 0.995:
                    circle1 = plt.Circle((0, 0), r[i]/r[0], color='blue', fill=False)
                    ax.add_artist(circle1)
                else:
                    circle2 = plt.Circle((0, 0), r[i]/r[0], color='red', fill=False)
                    ax.add_artist(circle2)
            else:
                if L[i]/L[0] < 0.995:
                    circle3 = plt.Circle((0, 0), r[i]/r[0], color='cyan', fill=False)
                    ax.add_artist(circle3)
                else:
                    circle4 = plt.Circle((0, 0), r[i]/r[0], color='yellow', fill=False)
                    ax.add_artist(circle4)
            plt.hold("on")
    plt.axis([-1.2, 1.2, -1.2, 1.2])
    plt.show()
    
    # Plotting T(r)
    plt.plot(r/r[0], T/max(T))
    plt.legend(["Temperature"])
    plt.xlabel("$R/R_0$")
    plt.ylabel("$T/T_{max}$")
    plt.show()

    # Plotting L(r)
    plt.plot(r/r[0], L/L[0])
    plt.legend(["Luminosity"])
    plt.xlabel("$R/R_0$")
    plt.ylabel("$L/L_0$")
    plt.show()
    
    # Plotting rho(r) and P(r)
    plt.plot(r/r[0], np.log10(rho/rho[0]), r/r[0], np.log10(P/P[0]))    
    plt.legend(["Density", "Pressure"])
    plt.xlabel("$R/R_0$")
    plt.ylabel("$\\rho/\\rho_0, P/P_0$)")
    plt.show()
    
    # Plotting M(r)
    plt.plot(r/r[0], M/M[0])
    plt.legend(["M(r)"])
    plt.xlabel("$R/R_0$")
    plt.ylabel("$M/M_0$)")
    plt.show()

    #A.plot(M, r, L, T, rho)
    
    def least_square(no_values, R_0_min, R_0_max, rho_0_min, rho_0_max, T_0_min, T_0_max, N, dm):
        """
        This function performes a least square method
        It takes the arguments for the min and max values of R_0, rho_0 and T_0 where no_values is
        the number of points in each list. It also takes the arguments number of integration 
        points N and step length dm.
        The funciton then prints out the sum of the procentages (last_value/first_value) as they 
        get closer and closer towards zero together with the factor in front  
        """

        R_0_list = np.linspace(R_0_min, R_0_max, no_values)      # list of R_0-values
        rho_0_list = np.linspace(rho_0_min, rho_0_max, no_values)# list of rho_0-values
        T_0_list = np.linspace(T_0_min, T_0_max, no_values)      # list of T_0-values

        # Variables to use in the least square method
        R_0_save = 0
        rho_0_save = 0
        T_0_save = 0
        least = 1e6
        for i in range(len(R_0_list)):
            R_01 = R_0_list[i]
            for j in range(len(T_0_list)):
                T_01 = T_0_list[j]
                for k in range(len(rho_0_list)):
                    rho_01 = rho_0_list[k]
                    # Integrating through the class
                    A = project2(L_0, M_0, R_01, rho_01, T_01, X, Y, Y3, Y4, Z, Z7Li, Z7Be)
                    r, M, L, rho, P, T, E, nabla_star, nabla_st, nabla_ad, PP1_E, PP2_E, PP1_frac, PP2_frac, Fc_frac, Fr_frac = A.integrate(-1e26, N, dynamic_step_size = True)
                    L_lim = L[-1]/L[0]
                    M_lim = M[-1]/M[0]
                    R_lim = r[-1]/r[0]
                    if (L_lim + M_lim + R_lim <= least):
                        # Checking if the new value is a better fit than the last
                        R_0_save = R_01
                        rho_0_save = rho_01
                        T_0_save = T_01
                        least = L_lim + M_lim + R_lim
                        print "Least seperate values = ", L_lim, M_lim, R_lim
                        print "Least total value = ", least
                        print "Parameters used = ", R_0_save/R_sun, rho_0_save/(1.42e-7*rho_star_avg), T_0_save/5770
                        print " "
    #least_square(10, 0.85*R_sun, 1.2*R_sun, 50.*rho_0,80*rho_0, 0.85*5770, 1.2*5770, N, dm=-1e26)
