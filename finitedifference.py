# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 13:56:51 2018

@author: james
"""

import numpy as np

def tridiag(a, b, c, k1=-1, k2=0, k3=1):
    # create a tridiagonal matrix on diagonals k1, k2, k3
    return np.diag(a, k1) + np.diag(b, k2) + np.diag(c, k3)

def backwardeuler(T, mx, mt, lmbda, u_0, lbc, rbc, f):
    # Construct the backward Euler matrix
    A_BE = tridiag((mx)*[-lmbda], (mx+1)*[1+2*lmbda], (mx)*[-lmbda])
    
    # Add the boundary conditions
    A_BE[0,0] = A_BE[mx,mx] = 1
    A_BE[0,1] = A_BE[mx,mx-1] = 0
    
    u_jp1 = np.zeros(u_0.size)      # u at next time step 
    u_j = u_0    # u at the current time step
    
    deltat = T / (mt+1)
    # Solve the PDE: loop over all time points
    for n in range(1, mt+1):    
        # Backward Euler timestep at inner mesh points
        u_jp1 = np.linalg.solve(A_BE, u_j)

        # Boundary conditions
        u_jp1[0] = lbc(n*deltat); u_jp1[mx] = rbc(n*deltat) 
            
        # Update u_j
        u_j[:] = u_jp1[:]
        #u_j[1:mt] += deltat*f(n*deltat)
   
    return u_j

def forwardeuler(T, mx, mt, lmbda, u_0, lbc, rbc, f):   
    # Construct the forward Euler matrix
    A_FE = tridiag((mx)*[lmbda], (mx+1)*[1-2*lmbda], (mx)*[lmbda])
    A_FE[0,0] = A_FE[mx,mx] = 1
    A_FE[0,1] = A_FE[mx,mx-1] = 0
    
    u_jp1 = np.zeros(u_0.size)      # u at next time step 
    u_j = u_0    # u at the current time step
    
    deltat = T / (mt+1)
    # Solve the PDE: loop over all time points
    for n in range(1, mt+1):
        # Forward Euler timestep at inner mesh points
        u_jp1 = np.dot(A_FE, u_j)
        
        # Boundary conditions
        u_jp1[0] = lbc(n*deltat); u_jp1[mx] = rbc(n*deltat)
              
        # Update u_j
        u_j[:] = u_jp1[:]
    
    return u_j

def cranknicholson(T, mx, mt, lmbda, u_0, lbc, rbc, f):
    # Construct the Crank-Nicholson matrices
    A_CN = tridiag((mx-2)*[-0.5*lmbda], (mx-1)*[1+lmbda], (mx-2)*[-0.5*lmbda])
    B_CN = tridiag((mx-2)*[0.5*lmbda], (mx-1)*[1-lmbda], (mx-2)*[0.5*lmbda])
    
    u_jp1 = np.zeros(u_0.size)      # u at next time step 
    u_j = u_0    # u at the current time step
        
    deltat = T / (mt+1)
    for n in range(1, mt+1):    
        # Crank-Nicholson timestep at inner mesh points
        u_jp1[1:-1] = np.linalg.solve(A_CN, B_CN.dot(u_j[1:-1]))

        # Boundary conditions
        u_jp1[0] = lbc(n*deltat); u_jp1[mx] = rbc(n*deltat)
            
        # Update u_j
        u_j[:] = u_jp1[:]    
        u_j[1:mt] += 0.5*f(n*deltat)
        
    return u_j  