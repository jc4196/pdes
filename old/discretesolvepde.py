# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 16:36:50 2018

@author: james
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

import matplotlib.pylab as pl

def tridiag(N, A00, A01, ANm1N, ANN, l, m, u):
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
    lower = (N-1)*[l] + [ANm1N]
    main = [A00] + (N-1)*[m] + [ANN]
    upper = [A01] + (N-1)*[u]
    
    return sparse.diags(diagonals = [lower, main, upper],
                        offsets = [-1, 0, 1],
                        shape = (N+1, N+1),
                        format='csr')

def tridiag(N, lower, main, upper):
    return sparse.diags([lower,main,upper],
                        offsets=[-1,0,1],
                        shape=(N+1,N+1),
                        format='csr')

def matrix_indices(boundaries, mx):
    l, r = boundaries
    
    a = 1 if l == 'D' else 0
    b = mx if r == 'D' else mx + 1
    
    return a, b

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

def matrix_rows(mx, lbctype, rbctype):
    a, b = 0, mx+1
    
    if lbctype == 'Dirichlet':
        a = 1
    if rbctype == 'Dirichlet':
        b = mx
        
    return a, b
    
def solve_wave_pde(mx, mt, L, T, scheme,
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
    
def plot_solution(xs, uT, uexact=None, title='', uexacttitle=''):
    """Plot the solution uT to a PDE problem at time t"""
    try:
        pl.plot(xs,uT,'ro',label='numerical')
    except:
        pass
        
    if uexact:
        xx = np.linspace(xs[0], xs[-1], 250)

        pl.plot(xx, uexact(xx),'b-',label=uexacttitle)

    pl.xlabel('x')
    pl.ylabel('u(x,T)')
    pl.title(title)
    pl.legend(loc='best')
    pl.show()

def animate_solution(trange):
    """animate the solution to a PDE problem for a range of t values"""
    pass

def error(uT, uexact):
    """Calculate the error between a solution value of t and the exact solution"""
    return np.linalg.norm(uT - uexact)