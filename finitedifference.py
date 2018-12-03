# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 13:56:51 2018

@author: james
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

def tridiag(a, b, c, k1=-1, k2=0, k3=1):
    # create a tridiagonal matrix on diagonals k1, k2, k3
    return np.diag(a, k1) + np.diag(b, k2) + np.diag(c, k3)

def tridiag(mx, a, b, c):
    lower = np.zeros(mx)
    main = np.zeros(mx+1)
    upper = np.zeros(mx)
    
    lower[:] = a
    main[:] = b
    upper[:] = c
    
    return sparse.diags(diagonals = [main, lower, upper],
                        offsets = [0, -1, 1],
                        shape = (mx+1, mx+1))
 
def tridiag(N, A00, A01, ANm1N, ANN, m, l, u):
    """ Construct a sparse tridiagonal matrix of the form
    
    A00 A01 0 .....       0
    l   m   u  0..
    0   l   m   u
    ...       .
    ....        .
    ......    l    m      u
    ...            ANm1N  ANN
    
    """
    main = [A00] + (N-1)*[m] + [ANN]
    lower = (N-1)*[l] + [ANm1N]
    upper = [A01] + (N-1)*[u]
    
    return sparse.diags(diagonals = [main, lower, upper],
                        offsets = [0, -1, 1],
                        shape = (N+1, N+1),
                        format='csr')
    
def backwardeuler(T, L, mx, mt, lmbda, u0, lbc, rbc, source):
    # Parameters needed to construct the matrix
    deltax = L / mx; deltat = T / mt
    alpha1, beta1 = lbc.get_params()
    alpha2, beta2 = rbc.get_params()
    
    # Construct the backward Euler matrix
    A_BE = tridiag(mx,
                   alpha1* deltax - beta1,
                   beta1,
                   -beta2,
                   beta2 + alpha2*deltax,
                   1+2*lmbda,
                   -lmbda,
                   -lmbda)
    
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
        u_jp1 = spsolve(A_BE, u_j)
 
        # Update u_j
        u_j[:] = u_jp1[:]
         
    return u_j

def forwardeuler(T, L, mx, mt, lmbda, u_0, lbc, rbc, source):   
    # Construct the forward Euler matrix
    A_FE = tridiag(mx, 1, 0, 0, 1, 1-2*lmbda, lmbda, lmbda)
    
    u_jp1 = np.zeros(u_0.size)      # u at next time step 
    u_j = u_0.copy()    # u at the current time step
    
    # Check boundary conditions
    params = [*lbc.get_params(), *rbc.get_params()]
    if params != [1,0,1,0] or source.get_expr() != 0:
        print('General boundary conditions and sources not implemented for forward Euler scheme, use backward Euler or Crank-Nicholson')
        return
    
    deltat = T / (mt+1)
    # Solve the PDE: loop over all time points
    for n in range(1, mt+1):
        # Forward Euler timestep at inner mesh points
        u_jp1 = A_FE.dot(u_j)
        
        # Boundary conditions
        u_jp1[0] = lbc.apply_rhs(n*deltat)
        u_jp1[mx] = rbc.apply_rhs(n*deltat)
              
        # Update u_j
        u_j[:] = u_jp1[:]
    
    return u_j

def cranknicholson(T, L, mx, mt, lmbda, u_0, lbc, rbc, source):  
    # Parameters needed to construct the matrices
    deltax = L / mx; deltat = T / mt
    alpha1, beta1 = lbc.get_params()
    alpha2, beta2 = rbc.get_params()

    # Construct the Crank-Nicholson matrices
    A_CN = tridiag(mx,
                   alpha1* deltax - beta1,
                   beta1,
                   -beta2,
                   beta2 + alpha2*deltax,
                   1+lmbda,
                   -0.5*lmbda,
                   -0.5*lmbda)

    B_CN = tridiag(mx, 1, 0, 0, 1, 1-lmbda, 0.5*lmbda, 0.5*lmbda)

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
        u_jp1 = spsolve(A_CN, b)

        # Update u_j
        u_j[:] = u_jp1[:]    
        
    return u_j  