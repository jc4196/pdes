# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 13:56:51 2018

@author: james
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

   
def tridiag(N, A00, A01, ANm1N, ANN, m, l, u):
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
    
    A_BE = tridiag(mx,
                   1+2*lmbda, -2*lmbda, -2*lmbda, 1+2*lmbda,
                   1+2*lmbda, -lmbda, -lmbda)

    deltat = T/ mt; deltax = L / mx

    u_jp1 = np.zeros(u0.size)      # u at next time step 
    u_j = u0.copy()                # u at current time step

    # aliases 
    f = source
    p = lbc.apply_rhs
    q = rbc.apply_rhs
    
    # Solve the PDE: loop over all time points
    for n in range(1, mt+1):  
        if lbc.isDirichlet() and rbc.isDirichlet():
            # create b vector
            b = u_j[1:-1].copy()
            b[0] += lmbda*p((n+1)*deltat)
            b[-1] += lmbda*q((n+1)*deltat)
            
            # solve for next step
            u_jp1[1:-1] = spsolve(A_BE[1:-1,1:-1], b)
            
            # add source term
            u_jp1[1:-1] += deltat*f(deltax*np.arange(1,mx), n*deltat)
            
            # set boundaries
            u_jp1[0] = p((n+1)*deltat)
            u_jp1[-1] = q((n+1)*deltat) 
            
        elif lbc.isNeumann() and rbc.isDirichlet():
            # create b vector
            b = u_j[:-1].copy()
            b[0] -= 2*lmbda*deltax*p((n+1)*deltat)
            b[-1] += lmbda*q((n+1)*deltat)
            
            # solve for next step
            u_jp1[:-1] = spsolve(A_BE[:-1,:-1], b)
            
            # add source term
            u_jp1[1:-1] += deltat*f(deltax*np.arange(1,mx), n*deltat)
         
            # set boundary
            u_jp1[-1] = q((n+1)*deltat)
    
        elif lbc.isDirichlet() and rbc.isNeumann():
            # create b vector
            b = u_j[1:].copy()
            b[0] += lmbda*p((n+1)*deltat)
            b[-1] += 2*lmbda*deltax*q((n+1)*deltat)
            
            # solve for next step
            u_jp1[1:] = spsolve(A_BE[1:,1:], b)
            
            # add source term
            u_jp1[1:-1] += deltat*f(deltax*np.arange(1,mx), n*deltat)
            
            # set boundary
            u_jp1[0] = p((n+1)*deltat)

        elif lbc.isNeumann() and rbc.isNeumann():
            # create b vector
            b = u_j.copy()
            b[0] -= 2*lmbda*deltax*p((n+1)*deltat)
            b[-1] += 2*lmbda*deltax*q((n+1)*deltat)
            
            # solve for next step
            u_jp1 = spsolve(A_BE, u_j)
            
            # add source term
            u_jp1[1:-1] += deltat*f(deltax*np.arange(1,mx), n*deltat)
            
        else:
            print('General boundary conditions not implemented')
            return
            
        # Update u_j
        u_j[:] = u_jp1[:]
         
    return u_j

    
def forwardeuler(T, L, mx, mt, lmbda, u_0, lbc, rbc, source):
    # Construct the forward euler matrix    
    A_FE = tridiag(mx, 
                   1-2*lmbda, 2*lmbda, 2*lmbda, 1-2*lmbda,
                   1-2*lmbda, lmbda, lmbda)

    deltat = T/ mt; deltax = L / mx
    
    u_jp1 = np.zeros(u_0.size)      # u at next time step 
    u_j = u_0.copy()    # u at the current time step
    
    # aliases 
    f = source
    p = lbc.apply_rhs
    q = rbc.apply_rhs
    
    for n in range(1, mt+1):
        
        if lbc.isDirichlet() and rbc.isDirichlet():
            # multiply inner entries by the matrix
            u_jp1[1:-1] = A_FE[1:-1,1:-1].dot(u_j[1:-1])
            
            # add source term
            u_jp1[1:-1] += deltat*f(deltax*np.arange(1,mx), n*deltat)
            
            # modify terms on and next to the boundaries
            u_jp1[0] = p(n*deltat)
            u_jp1[1] += lmbda*p(n*deltat)
            u_jp1[-2] += lmbda*q(n*deltat)
            u_jp1[-1] = q(n*deltat)
            
        elif lbc.isNeumann() and rbc.isDirichlet():
            # include first row for Neumann condition
            u_jp1[:-1] = A_FE[:-1,:-1].dot(u_j[:-1])
            
            # add source term
            u_jp1[1:-1] += deltat*f(deltax*np.arange(1,mx), n*deltat)
            
            # modify Neumann condition
            u_jp1[0] -= 2*lmbda*deltax*p(n*deltat)
            
            # modify Dirichlet condition
            u_jp1[-2] += lmbda*q(n*deltat)
            u_jp1[-1] = q(n*deltat)
    
        elif lbc.isDirichlet() and rbc.isNeumann():
            u_jp1[1:] = A_FE[1:,1:].dot(u_j[1:])
            
            # add source term
            u_jp1[1:-1] += deltat*f(deltax*np.arange(1,mx), n*deltat)
            
            # modify Dirichlet condition
            u_jp1[0] = p(n*deltat)
            u_jp1[1] += lmbda*p(n*deltat)
            
            # modify Neumann condition
            u_jp1[-1] += 2*lmbda*deltax*q(n*deltat)
      
        elif lbc.isNeumann() and rbc.isNeumann():
            # use whole matrix
            u_jp1 = A_FE.dot(u_j)
            
            # add source term
            u_jp1[1:-1] += deltat*f(deltax*np.arange(1,mx), n*deltat)
            
            # modify boundaries
            u_jp1[0] -= 2*lmbda*deltax*p(n*deltat)
            u_jp1[-1] += 2*lmbda*deltax*q(n*deltat)
        
        else:
            print('General boundary conditions not implemented')
            return
        
        # Update u_j
        u_j[:] = u_jp1[:]
    
    return u_j


    
def cranknicholson(T, L, mx, mt, lmbda, u0, lbc, rbc, source):  
    A_CN = tridiag(mx,
                   1+lmbda, -lmbda, -lmbda, 1+lmbda,
                   1+lmbda, -0.5*lmbda, -0.5*lmbda)
    
    B_CN = tridiag(mx,
                   1-lmbda, lmbda, lmbda, 1-lmbda,
                   1-lmbda, 0.5*lmbda, 0.5*lmbda)
 
    deltat = T/ mt; deltax = L / mx

    u_jp1 = np.zeros(u0.size)      # u at next time step 
    u_j = u0.copy()                # u at current time step

    # aliases 
    f = source
    p = lbc.apply_rhs
    q = rbc.apply_rhs
    
    
    # Solve the PDE: loop over all time points
    for n in range(mt):  
        if lbc.isDirichlet() and rbc.isDirichlet():
            # create b vector
            b = B_CN[1:-1,1:-1].dot(u_j[1:-1])
            b[0] += 0.5*lmbda*(p(n*deltat) + p((n+1)*deltat))
            b[-1] += 0.5*lmbda*(q(n*deltat) + q((n+1)*deltat))
            
            # solve for next step
            u_jp1[1:-1] = spsolve(A_CN[1:-1,1:-1], b)
            
            # add source term
            u_jp1[1:-1] += deltat*f(deltax*np.arange(1,mx), n*deltat)
            
            # set boundaries
            u_jp1[0] = p(n*deltat)
            u_jp1[-1] = q(n*deltat) 
 
        elif lbc.isDirichlet() and rbc.isNeumann():
            # create b vector
            b = B_CN[1:,1:].dot(u_j[1:])
            b[0] += 0.5*lmbda*(p(n*deltat) + p((n+1)*deltat))
            b[-1] += lmbda*deltax*(q(n*deltat) + q((n+1)*deltat))
            
            # solve for next step
            u_jp1[1:-1] = spsolve(A_CN[1:,1:], b)
            
            # add source term
            u_jp1[1:-1] += deltat*f(deltax*np.arange(1,mx), n*deltat)
            
            # set left boundary
            u_jp1[0] = p(n*deltat)
            
        elif lbc.isNeumann() and rbc.isDirichlet():
            # create b vector
            b = B_CN[:-1,:-1].dot(u_j[:-1])
            b[0] -= lmbda*deltax*(p(n*deltat) + p((n+1)*deltat))
            b[-1] += 0.5*lmbda*(q(n*deltat) + q((n+1)*deltat))
            
            # solve for next step
            u_jp1[1:-1] = spsolve(A_CN[:-1,:-1], b)
            
            # add source term
            u_jp1[1:-1] += deltat*f(deltax*np.arange(1,mx), n*deltat)
            
            # set right boundary
            u_jp1[-1] = q(n*deltat) 

        elif lbc.isNeumann() and rbc.isNeumann():
            # create b vector
            b = B_CN.dot(u_j)
            b[0] -= lmbda*deltax*(p(n*deltat) + p((n+1)*deltat))
            b[-1] += lmbda*deltax*(q(n*deltat) + q((n+1)*deltat))
            
            u_jp1 = spsolve(A_CN, b)
            
            # add source term
            u_jp1[1:-1] += deltat*f(deltax*np.arange(1,mx), n*deltat)
            
        else:
            print('General boundary conditions not implemented')
    
        # Update u_j
        u_j[:] = u_jp1[:]
         
    return u_j

def cranknicholson2(T, L, mx, mt, lmbda, u_0, lbc, rbc, source):  
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

    # aliases 
    f = source
    p = lbc.apply_rhs
    q = rbc.apply_rhs
    
    for n in range(1, mt+1):  
        b = B_CN.dot(u_j)
        # Add boundary conditions and source to vector b
        b[0] = deltax*p(n*deltat) 
        if n != 1:
            b[1:mx] += deltat*f(deltax*np.arange(1,mx), n*deltat)
        b[mx] = deltax*q(n*deltat)

        # Crank-Nicholson timestep at inner mesh points
        u_jp1 = spsolve(A_CN, b)

        # Update u_j
        u_j[:] = u_jp1[:]    
        
    return u_j 

def backwardeuler2(T, L, mx, mt, lmbda, u0, lbc, rbc, source):
    # Parameters needed to construct the matrix
    deltax = L / mx; deltat = T / mt
    alpha1, beta1 = lbc.get_params()
    alpha2, beta2 = rbc.get_params()
    
    # Construct the backward Euler matrix
    A_BE = tridiag(mx,
                   alpha1*deltax - beta1,
                   beta1,
                   -beta2,
                   beta2 + alpha2*deltax,
                   1+2*lmbda,
                   -lmbda,
                   -lmbda)
    
    u_jp1 = np.zeros(u0.size)      # u at next time step 
    u_j = u0.copy()                # u at current time step

    # aliases 
    f = source
    p = lbc.apply_rhs
    q = rbc.apply_rhs
    
    # Solve the PDE: loop over all time points
    for n in range(1, mt+1):  
        # Add boundary conditions and source to vector u_j
        u_j[0] = deltax*p(n*deltat) 
        if n != 1:
            u_j[1:mx] += deltat*f(deltax*np.arange(1,mx), n*deltat)
        u_j[mx] = deltax*q(n*deltat)
        
        # Backward Euler timestep at inner mesh points
        u_jp1 = spsolve(A_BE, u_j)
 
        # Update u_j
        u_j[:] = u_jp1[:]
         
    return u_j 


def discrete_diffusion(dp, T, mx, mt, lmbda, scheme):
    deltat = T/ mt; deltax = dp.L / mx   # time and space steps

    u_jp1 = np.zeros(u0.size)        # u at next time step 
    u_j = dp.u0.get_initial_state()  # u at current time step

    # aliases 
    f = dp.source         # source function
    p = dp.lbc.apply_rhs  # left boundary function
    q = dp.rbc.apply_rhs  # right boundary function
    
    if dp.isDirichletDirichlet():
        for n in range(mt):
            u_jp1 = scheme(u_j, 'DD')
            u_j = u_jp1
    elif dp.isDirichletNeumann():
        pass
    elif dp.isNeumannDirichlet():
        pass
    elif dp.isNeumannNeumann():
        pass
    else:
        pass
    
    return u_j
    
    