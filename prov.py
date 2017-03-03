            """
            dm_test = np.zeros(4)
            dV = np.zeros(4) 
            V_test = np.zeros(4)
            
            
            dV[0] = dm*1.0/(4*np.pi*r[i]**2*rho[i])
            dV[1] = dm*(-self.G*M[i]/(4*np.pi*r[i]**4))
            dV[2] = dm*E[i]
            dV[3] = dm*(-(3*K*L[i])/(256*np.pi**2*self.sigma*r[i]**4*T[i]**3))
            
            V_test[0] = r[i]
            V_test[1] = P[i]
            V_test[2] = L[i]
            V_test[3] = T[i]
            
            dm_test[0] = abs(p*r[i]/(1.0/(4*np.pi*r[i]**2*rho[i])))
            dm_test[1] = abs(p*P[i]/((-self.G*M[i])/(4*np.pi*r[i]**4)))
            dm_test[2] = abs(p*L[i]/E[i])
            dm_test[3] = abs(p*T[i]/(-(3*K*L[i])/(256*np.pi**2*self.sigma*r[i]**4*T[i]**3)))
            
            for j in range(len(dV)):           
                if (abs(dV[j])/V_test[j] > p):
                    dm = -min(dm_test)
                    break
                else:
                    dm = dm1
            """
