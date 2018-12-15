# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 11:04:22 2018

@author: james
"""
import numpy as np
from helpers import tridiag


def initialise(mx, mt, L, T):
    xs = np.linspace(0, L, mx+1)     # mesh points in space
    ts = np.linspace(0, T, mt+1)          # mesh points in time
    deltax = xs[1] - xs[0]                # gridspacing in x
    deltat = ts[1] - ts[0]                # gridspacing in t
    lmbda = kappa*deltat/(deltax**2)    # mesh fourier number   
    
    return xs, ts, deltax, deltat, lmbda


def addboundaries(u, lbctype, rbctype, D0, D1, Dmxm1, Dmx, N0, Nmx):
    """Add Neumann or Dirichlet boundary conditions"""
    if lbctype == 'Neumann':
        u[0] += N0
    elif lbctype == 'Dirichlet':
        u[0] = D0
        u[1] += D1
    else:
        raise Exception('That boundary condition is not implemented')
    
    if rbctype == 'Neumann':
        u[-1] += Nmx
    elif lbctype == 'Dirichlet':
        u[-2] += Dmxm1
        u[-1] = Dmx  
    else:
        raise Exception('That boundary condition is not implemented')

def matrixrowrange(mx, lbctype, rbctype):
    a, b = 0, mx+1
    
    if lbctype == 'Dirichlet':
        a = 1
    if rbctype == 'Dirichlet':
        b = mx
        
    return a, b

def forwardeuler(mx, mt, L, T, 
                 kappa, source,
                 ic, lbc, rbc, lbctype, rbctype):
    """
    Solve a diffusion type problem with the given spacing and scheme
    
    Parameters
        mx         (mx+1 is) number of mesh points in space
        mt         (mt + 1 is ) number of mesh points in time
        L          length of interval
        T          time to solve to
        kappa      diffusion constant
        ic         initial condition function (vectorized)
        source     source function (vectorized)
        lbc        left boundary condition function (vectorized)
        rbc        right boundary condition function (vectorized)
        lbctype    string representation of left boundary type eg. 'Neumann'
        rbctype    string representation of right boundary type
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
    
    # if boundary conditions don't match initial conditions
    if lbctype == 'Dirichlet':
        u_j[0] = lbc(0)
    if rbctype == 'Dirichlet':
        u_j[mx] = rbc(0)
        

    u_jp1 = np.zeros(xs.size)
    
    # Construct forward Euler matrix
    A_FE = tridiag(mx, lmbda, 1-2*lmbda, lmbda)
    
    # modify first and last row for Neumann conditions
    A_FE[0,1] *= 2; A_FE[mx,mx-1] *= 2

    # range of rows of A_FE to use
    a, b = matrixrowrange(mx, lbctype, rbctype)
    print(a,b)
    # Solve the PDE
    for n in range(1, mt+1):
        u_jp1[a:b] = A_FE[a:b,a:b].dot(u_j[a:b])
        
        addboundaries(u_jp1, lbctype, rbctype,
                      lbc(n*deltat),
                      lmbda*lbc(n*deltat),
                      lmbda*rbc(n*deltat),
                      rbc(n*deltat),
                      -2*lmbda*deltax*lbc(n*deltat),
                      2*lmbda*deltax*rbc(n*deltat))
        
        # fix up boundary conditions
        #if lbctype == 'Neumann':
        #    u_jp1[0] += -2*lmbda*deltax*lbc(n*deltat)
        #elif lbctype == 'Dirichlet':
        #    u_jp1[0] = lbc(n*deltat)
        #    u_jp1[1] += lmbda*lbc(n*deltat)
        #else:
        #    raise Exception('Boundary condition not implemented')
       # 
       # if rbctype == 'Neumann:
        #    u_jp1[mx] += 2*lmbda*deltax*rbc(n*deltat)
        #elif lbctype == 'Dirichlet:
        #    u_jp1[mx-1] += lmbda*rbc(n*deltat)
        #    u_jp1[mx] = rbc(n*deltat)

        # add source to inner terms
        u_jp1[1:-1] += deltat*source(xs[1:-1], n*deltat)
        
        u_j[:] = u_jp1[:]
    
    return xs, u_j

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