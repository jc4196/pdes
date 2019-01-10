# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 10:52:12 2018

@author: james
"""
from scipy import sparse
import numpy as np

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

    
    
def forwardeuler(mx, deltax, deltat, lmbda, p, q, boundaries):  
    A = sparse.identity(mx+1, format='csr')
    
    # create forward Euler matrix
    B = tridiag(mx, 
                1-2*lmbda, 2*lmbda, 2*lmbda, 1-2*lmbda,
                1-2*lmbda, lmbda, lmbda)
    
    lbctype, rbctype = boundaries
    
    if lbctype == 'D':
        left = [0, lmbda*p(t)]
    elif lbctype == 'N':
        left = [-2*lmbda*deltax*p(t), 0]
    else:
        raise Exception('boundary type not implemented for forward Euler')
        
    if rbctype == 'D':
        right = [lmbda*q(t), 0]
    elif rbctype == 'N':
        right = [0, 2*lmbda*deltax*q(t)]
    else:
        raise Exception('boundary type not implemented for forward Euler')
        
    def v(t):
        return np.concatenate((left, np.zeros(mx-3), right))
    
    return A, B, v

def forwardeuler(mx, deltax, deltat, lmbda, p, q):  
    A = sparse.identity(mx+1, format='csr')
    
    # create forward Euler matrix
    B = tridiag(mx, 
                1-2*lmbda, 2*lmbda, 2*lmbda, 1-2*lmbda,
                1-2*lmbda, lmbda, lmbda)
    
    def v(t):
        return [-2*lmbda*deltax*p(t), lmbda*p(t),
                lmbda*q(t), 2*lmbda*deltax*q(t)]
        
    return A, B, v

def backwardeuler(mx, deltax, deltat, lmbda, p, q):
    # create backward Euler matrix
    A = tridiag(mx,
                1+2*lmbda, -2*lmbda, -2*lmbda, 1+2*lmbda,
                1+2*lmbda, -lmbda, -lmbda)
    
    B = sparse.identity(mx+1, format='csr')
    
    def v(t):
        return np.concatenate(([-2*lmbda*deltax*p(t),lmbda*p(t)],
                                np.zeros(mx-3),
                                [lmbda*q(t), 2*lmbda*deltax*q(t)]))
    
    return A, B, v

def cranknicholson(mx, deltax, deltat, lmbda, p, q):
    # create Crank-Nicholson matrices
    A = tridiag(mx,
                1+lmbda, -lmbda, -lmbda, 1+lmbda,
                1+lmbda, -0.5*lmbda, -0.5*lmbda)
    
    B = tridiag(mx,
                1-lmbda, lmbda, lmbda, 1-lmbda,
                1-lmbda, 0.5*lmbda, 0.5*lmbda)
    
    def v(t):
        return np.concatenate(([-lmbda*deltax*(p(t) + p(t + deltat)),
                               0.5*lmbda*(p(t) + p(t + deltat))],
                               np.zeros(mx-3),
                               [0.5*lmbda*(q(t) + q(deltat)),
                                lmbda*deltax*(q(t) + q(t + deltat))]))
    
    return A, B, v
    
def solve_diffusion_pde(mx, mt, L, T, scheme, 
                        kappa, source,
                        ic, lbc, rbc, boundaries):
    """
    Solve a diffusion type problem with the given spacing and scheme
    
    Parameters
        mx         (mx+1 is) number of mesh points in space
        mt         (mt + 1 is ) number of mesh points in time
        L          length of interval
        T          time to solve to
        kappa      diffusion constant
        ic         initial condition function (vectorized)
        lbc        left boundary condition function (vectorized)
        rbc        right boundary condition function (vectorized)
        boundaries    signature of boundary types *to finish
        source     source function (vectorized)
        scheme     scheme to solve the pde eg. forwardeuler, backward euler..
    
    returns
        xs      mesh points in space
        uT      the numerical solution u at time T
    """
    xs = np.linspace(0, L, mx+1)     # mesh points in space
    ts = np.linspace(0, T, mt+1)          # mesh points in time
    deltax = xs[1] - xs[0]                # gridspacing in x
    deltat = ts[1] - ts[0]                # gridspacing in t
    lmbda = kappa*deltat/(deltax**2)    # mesh fourier number

    u_j = ic(xs)
    u_jp1 = np.zeros(xs.size)
    
    # Get matrices and vector for the particular scheme
    A, B, boundary_fns = scheme(mx, deltax, deltat, lmbda, lbc, rbc)
        
    a, b = matrix_indices(boundaries, mx)
    l, r = boundaries

    
    for n in range(1, mt+1):
        # Solve matrix equation A*u_{j+1} = B*u_j + v
        
        v = np.zeros(b-a)
        v[0] = boundary_fns(n*deltat)[0] if l == 'N' else boundary_fns(n*deltat)[1]
        v[-1] = boundary_fns(n*deltat)[2] if r == 'D' else boundary_fns(n*deltat)[3]
    
        u_jp1[a:b] = spsolve(A[a:b,a:b],
                             B[a:b,a:b].dot(u_j[a:b]) + v)
        
        # add source to inner terms
        u_jp1[1:-1] += deltat*source(xs[1:-1], n*deltat)
        
        # fix Dirichlet boundary conditions
        if a == 1:
            u_jp1[0] = lbc(n*deltat)
        if b == mx:
            u_jp1[-1] = rbc(n*deltat)
        
        u_j[:] = u_jp1[:]
    
    return xs, u_j
    