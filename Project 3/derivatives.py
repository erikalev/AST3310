import numpy as np
import sys

def upwind(u, phi, dx, nx, nz = 0, dz = False):
    if dz == False:
        if np.size(u) == nx*nz:
            der = np.zeros((nx, nz))
            derivate_negative_u = (np.roll(phi, -1, axis = 1) - phi)/dx
            derivate_positive_u = (phi - np.roll(phi, 1, axis = 1))/dx

            der[np.where(u < 0)] = derivate_negative_u[np.where(u < 0)]
            der[np.where(u >= 0)] = derivate_positive_u[np.where(u >= 0)]
            
        else:
            der = np.zeros(nx)
            derivate_negative_u = (np.roll(phi, -1) - phi)/dx
            derivate_positive_u = (phi - np.roll(phi, 1))/dx
            der[np.where(u < 0)] = derivate_negative_u[np.where(u < 0)]
            der[np.where(u >= 0)] = derivate_positive_u[np.where(u >= 0)]

    else:
        if np.size(u) == nx*nz:
            der = np.zeros((nx, nz))        
            der = np.zeros((nx, nz))
            derivate_negative_u = (np.roll(phi, -1, axis= 0 ) - phi)/dx
            derivate_positive_u = (phi - np.roll(phi, 1, axis= 0 ))/dx

            der[np.where(u < 0)] = derivate_negative_u[np.where(u < 0)]
            der[np.where(u >= 0)] = derivate_positive_u[np.where(u >= 0)]
                
        else:
            print "You have chosen dz without having a z-dimention!"

    return der

def central_diff(u, phi, dx, nx, nz=0, dz = False):
    if dz == False:
        if np.size(u) == nx*nz:
            der = np.zeros((nx, nx))
            der = (np.roll(phi, -1, axis = 1) - np.roll(phi, 1, axis = 1))/(2*dx)    
        else:
            der = np.zeros(nx)
            der = (np.roll(phi, -1) - np.roll(phi, 1))/(2*dx)    
    else:
        if np.size(u) == nx*nz:
            der = np.zeros((nx, nx))        
            der = (np.roll(phi, -1, axis = 0) - np.roll(phi, 1, axis = 0))/(2*dx)    
        else:
            print "You have chosen dz without having a z-dimention!"

    return der

