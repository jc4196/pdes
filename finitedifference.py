# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 13:56:51 2018

@author: james
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

def tridiag(N, main, lower, upper):
    """
    Construct (N+1)x(N+1) sparse tridiagonal matrix
    
    N    highest index
    """
    return sparse.diags(diagonals = [main, lower, upper],
                        offsets = [0, -1, 1],
                        shape = (N+1, N+1),
                        format='csr')
    
def tridiag_b(N, A00, A01, ANm1N, ANN, m, l, u):
    """
    Construct a sparse tridiagonal matrix of the form
    
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
    params = lbc.get_params() + rbc.get_params()

    if params != (1,0,1,0) or source.get_expr() != 0:
        print('General boundary conditions and sources not implemented for forward Euler scheme, use backward Euler or Crank-Nicholson')
        return
    
    deltat = T / mt
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

def forwardeuler(T, L, mx, mt, kappa, u_0, lbc, rbc, source):
    # Construct the forward euler matrix    
    #A_FE = tridiag(mx, 
    #               1-2*lmbda, 2*lmbda, 2*lmbda, 1-2*lmbda,
    #               1-2*lmbda, lmbda, lmbda)

    # Construct forward euler matrix with variable kappa
    deltat = T/ mt; deltax = L / mx; lmbda
    p = deltat / (deltax**2)
    main = [1 - p*(kappa(t-0.5) - kappa(t+0.5)) for t in range(mx+1)]
    lower = [p*kappa(t+0.5) for t in range(mx)]
    lower[-1] *= 2
    upper = lower.copy()
    upper[0] *= 2
    
    A_FE = tridiag(mx, main, lower, upper)
    
    u_jp1 = np.zeros(u_0.size)      # u at next time step 
    u_j = u_0.copy()    # u at the current time step
    
    
    
    for n in range(1, mt+1):
        
        if lbc.isDirichlet() and rbc.isDirichlet():
            # multiply inner entries by the matrix
            u_jp1[1:-1] = A_FE[1:-1,1:-1].dot(u_j[1:-1])
            
            # add source term
            u_jp1[1:-1] += deltat*source.apply(deltax*np.arange(1,mx))
            
            # modify terms on and next to the boundaries
            u_jp1[0] = lbc.apply_rhs(n*deltat)
            u_jp1[1] += lmbda*lbc.apply_rhs(n*deltat)
            u_jp1[-2] += lmbda*rbc.apply_rhs(n*deltat)
            u_jp1[-1] = rbc.apply_rhs(n*deltat)
            
        elif lbc.isNeumann() and rbc.isDirichlet():
            # include first row for Neumann condition
            u_jp1[:-1] = A_FE[:-1,:-1].dot(u_j[:-1])
            
            # add source term
            u_jp1[1:-1] += deltat*source.apply(deltax*np.arange(1,mx))
            
            # modify Neumann condition
            u_jp1[0] -= 2*lmbda*deltax*lbc.apply_rhs(n*deltat)
            
            # modify Dirichlet condition
            u_jp1[-2] += lmbda*rbc.apply_rhs(n*deltat)
            u_jp1[-1] = rbc.apply_rhs(n*deltat)
    
        elif lbc.isDirichlet() and rbc.isNeumann():
            u_jp1[1:] = A_FE[1:,1:].dot(u_j[1:])
            
            # add source term
            u_jp1[1:-1] += deltat*source.apply(deltax*np.arange(1,mx))
            
            # modify Dirichlet condition
            u_jp1[0] = lbc.apply_rhs(n*deltat)
            u_jp1[1] += lmbda*lbc.apply_rhs(n*deltat)
            
            # modify Neumann condition
            u_jp1[-1] += 2*lmbda*deltax*rbc.apply_rhs(n*deltat)
      
        elif lbc.isNeumann() and rbc.isNeumann():
            # use whole matrix
            u_jp1 = A_FE.dot(u_j)
            
            # add source term
            u_jp1[1:-1] += deltat*source.apply(deltax*np.arange(1,mx))
            
            # modify boundaries
            u_jp1[0] -= 2*lmbda*deltax*lbc.apply_rhs(n*deltat)
            u_jp1[-1] += 2*lmbda*deltax*rbc.apply_rhs(n*deltat)
        
        else:
            print('General boundary conditions not implemented')
            return
        
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