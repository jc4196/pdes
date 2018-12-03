# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 13:56:51 2018

@author: james
"""

import numpy as np

def tridiag(a, b, c, k1=-1, k2=0, k3=1):
    # create a tridiagonal matrix on diagonals k1, k2, k3
    return np.diag(a, k1) + np.diag(b, k2) + np.diag(c, k3)

def backwardeuler(T, L, mx, mt, lmbda, u0, lbc, rbc, source):
    # Construct the backward Euler matrix
    A_BE = tridiag(mx*[-lmbda], (mx+1)*[1+2*lmbda], mx*[-lmbda])
    
    deltax = L / mx; deltat = T / mt
    
    # Add the boundary conditions to the matrix A
    alpha1, beta1 = lbc.get_params()
    alpha2, beta2 = rbc.get_params()
    
    A_BE[0,0] = alpha1*deltax - beta1
    A_BE[0,1] = beta1
    A_BE[mx,mx-1] = -beta2
    A_BE[mx,mx] = beta2 + alpha2*deltax
    
    u_jp1 = np.zeros(u0.size)      # u at next time step 
    u_j = u0.copy()                # u at current time step

    # Solve the PDE: loop over all time points
    for n in range(1, mt+1):  
        # Add boundary conditions and source to vector u_j
        u_j[0] = deltax*lbc.apply_rhs(n*deltat) 
        if n != 1:
            u_j[1:mx] += deltat*source.apply(deltax*np.arange(1,mx))
        u_j[mx] = deltax*rbc.apply_rhs(n*deltat)
        
        # Backward Euler timestep at inner mesh points
        u_jp1 = np.linalg.solve(A_BE, u_j)
 
        # Update u_j
        u_j[:] = u_jp1[:]
         
    return u_j

def forwardeuler(T, mx, mt, lmbda, u_0, lbc, rbc, source):   
    # Construct the forward Euler matrix
    A_FE = tridiag((mx)*[lmbda], (mx+1)*[1-2*lmbda], (mx)*[lmbda])
    A_FE[0,0] = A_FE[mx,mx] = 1
    A_FE[0,1] = A_FE[mx,mx-1] = 0
    
    u_jp1 = np.zeros(u_0.size)      # u at next time step 
    u_j = u_0.copy()    # u at the current time step
    
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

def cranknicholson(T, L, mx, mt, lmbda, u_0, lbc, rbc, source):
    # Construct the Crank-Nicholson matrices
    A_CN = tridiag(mx*[-0.5*lmbda], (mx+1)*[1+lmbda], mx*[-0.5*lmbda])
    B_CN = tridiag(mx*[0.5*lmbda], (mx+1)*[1-lmbda], mx*[0.5*lmbda])
    
    deltax = L / mx; deltat = T / mt
    
    # Add the boundary conditions to the matrix A
    alpha1, beta1 = lbc.get_params()
    alpha2, beta2 = rbc.get_params()
    
    A_CN[0,0] = alpha1*deltax - beta1
    A_CN[0,1] = beta1
    A_CN[mx,mx-1] = -beta2
    A_CN[mx,mx] = beta2 + alpha2*deltax
    
    B_CN[0,0] = B_CN[mx,mx] = 1
    B_CN[0,1] = B_CN[mx,mx-1] = 0
    
    u_jp1 = np.zeros(u_0.size)      # u at next time step 
    u_j = u_0.copy()   # u at the current time step

    for n in range(1, mt+1):  
        b = B_CN.dot(u_j)
        # Add boundary conditions and source to vector b
        b[0] = deltax*lbc.apply_rhs(n*deltat) 
        if n != 1:
            b[1:mx] += deltat*source.apply(deltax*np.arange(1,mx))
        b[mx] = deltax*rbc.apply_rhs(n*deltat)

        # Crank-Nicholson timestep at inner mesh points
        u_jp1 = np.linalg.solve(A_CN, b)

        # Update u_j
        u_j[:] = u_jp1[:]    
        
    return u_j  