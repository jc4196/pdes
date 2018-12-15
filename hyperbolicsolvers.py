# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 10:56:19 2018

@author: james
"""

from helpers import tridiag

def explicitsolve(mx, mt, L, T, scheme,
                  c, source,
                  ix, iv, lbc, rbc, lbctype, rbctype):
    """Solve a wave equation problem with the given spacing and scheme"""
    xs = np.linspace(0, L, mx+1)     # mesh points in space
    ts = np.linspace(0, T, mt+1)      # mesh points in time
    deltax = xs[1] - xs[0]            # gridspacing in x
    deltat = ts[1] - ts[0]            # gridspacing in t
    lmbda = c*deltat/deltax      # squared Courant number

    # Get matrices and vector for the particular scheme
    A = tridiag(mx, lmbda**2, 2-2*lmbda**2, lmbda**2)
    A[0,1] *= 2; A[-1,-2] *= 2
    
    a, b = matrix_rows(mx, lbctype, rbctype)
    # initial condition vectors
    U = ix(xs)
    V = iv(xs)
    
    # set first two time steps
    u_jm1 = U 
    
    u_j = np.zeros(xs.size)
    u_j[a:b] = 0.5*A[a:b,a:b].dot(u_jm1[a:b]) + deltat*V[a:b]
    
    # boundary conditions (may not match initial conditions)
    
    if lbctype == 'Dirichlet':
        u_j[0] = lbc(0)
        
    if rbctype == 'Dirichlet':
        u_j[mx] = rbc(0)     
    
    # initialise u at next time step
    u_jp1 = np.zeros(xs.size)        
    
    for n in range(2,mt+1):
        u_jp1[a:b] = A[a:b,a:b].dot(u_j[a:b]) - u_jm1[a:b]
        
        # boundary conditions
        if lbctype == 'Neumann':
            u_jp1[0] += -2*lmbda**2*deltax*lbc(n*deltat)
        else:
            u_jp1[0] = lbc(n*deltat)
            u_jp1[1] += lmbda**2*lbc(n*deltat)
            
        if rbctype == 'Neumann':
            u_jp1[mx] += 2*lmbda**2*deltax*rbc(n*deltat)
        else:
            u_jp1[mx-1] += lmbda**2*rbc(n*deltat) 
            u_jp1[mx] = rbc(n*deltat)
        
        # update u_jm1 and u_j
        u_jm1[:], u_j[:] = u_j[:], u_jp1[:]
    
    return xs, u_j