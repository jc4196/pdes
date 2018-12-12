# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 16:36:50 2018

@author: james
"""

import numpy as np
from scipy.sparse.linalg import spsolve

import matplotlib.pylab as pl

def solve_diffusion_pde(mx, mt, L, T, 
                        kappa, source,
                        ic, lbc, rbc, boundaries, 
                        scheme):
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
    A, B, v = scheme(mx, deltax, deltat, lmbda, lbc, rbc)
        
    a, b = boundaries
        
    for n in range(1, mt+1):
        # Solve matrix equation A*u_{j+1} = B*u_j + v
        u_jp1[a:b] = spsolve(A[a:b,a:b],
                             B[a:b,a:b].dot(u_j[a:b]) + v(n*deltat)[a:b])
        
        # add source to inner terms
        u_jp1[1:-1] += deltat*source(xs[1:-1], n*deltat)
        
        # fix Dirichlet boundary conditions
        if a == 1:
            u_jp1[0] = lbc(n*deltat)
        if b == mx:
            u_jp1[-1] = rbc(n*deltat)
        
        u_j[:] = u_jp1[:]
    
    return xs, u_j

def solve_wave_pde(mx, mt, L, T,
                   c, source,
                   ix, iv, lbc, rbc, boundaries,
                   scheme):
    """Solve a wave equation problem with the given spacing and scheme"""
    xs = np.linspace(0, L, mx+1)     # mesh points in space
    ts = np.linspace(0, T, mt+1)      # mesh points in time
    deltax = xs[1] - xs[0]            # gridspacing in x
    deltat = ts[1] - ts[0]            # gridspacing in t
    lmbda = c*deltat/deltax      # squared Courant number

    # Get matrices and vector for the particular scheme
    A, B, v = scheme(mx, deltax, deltat, lmbda, lbc, rbc) 

    # set initial condition
    u_jm1 = ix(xs) 
    
    # first time step
    u_j = np.zeros(xs.size)
    u_j[1:-1] = 0.5*A[1:-1,1:-1].dot(u_jm1[1:-1]) + deltat*iv(xs)[1:-1]
    u_j[0] = 0; u_j[mx] = 0  # boundary condition     
    
    # u at next time step
    u_jp1 = np.zeros(xs.size)        
    
    for n in range(2,mt+1):
        u_jp1[1:-1] = A[1:-1,1:-1].dot(u_j[1:-1]) - u_jm1[1:-1]
        
        # boundary conditions
        u_jp1[0] = 0; u_jp1[mx] = 0
        
        # update u_jm1 and u_j
        u_jm1[:],u_j[:] = u_j[:],u_jp1[:]
    
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