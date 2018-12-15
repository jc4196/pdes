# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 11:04:22 2018

@author: james

Contains functions for solving parabolic PDE problems of the form

du/dt = kappa*d^2u/dx^2 + f(x)


Each solver has the interface

def solver(mx, mt, L, T, kappa, ic, source, lbc, rbc, lbctype, rbctype):
    
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
    
    returns
        xs      mesh points in space
        uT      the numerical solution u at time T

"""
import numpy as np
from helpers import tridiag

from scipy.sparse.linalg import spsolve


def initialise(mx, mt, L, T, kappa):
    xs = np.linspace(0, L, mx+1)          # mesh points in space
    ts = np.linspace(0, T, mt+1)          # mesh points in time
    deltax = xs[1] - xs[0]                # gridspacing in x
    deltat = ts[1] - ts[0]                # gridspacing in t
    lmbda = kappa*deltat/(deltax**2)      # mesh fourier number   
    
    return xs, ts, deltax, deltat, lmbda


def addboundaries(u, lbctype, rbctype, D1, Dmxm1, N0, Nmx):
    """Add Neumann or Dirichlet boundary conditions"""
    if lbctype == 'Neumann':
        u[0] += N0
    elif lbctype == 'Dirichlet':
        u[1] += D1
    else:
        raise Exception('That boundary condition is not implemented')
    
    if rbctype == 'Neumann':
        u[-1] += Nmx
    elif lbctype == 'Dirichlet':
        u[-2] += Dmxm1
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
    """Forward euler finite-difference scheme (explicit) for solving
    parabolic PDE problems. Values of lambda > 1/2 will cause the scheme
    to become unstable"""
    
    # initialise     
    xs, ts, deltax, deltat, lmbda = initialise(mx, mt, L, T, kappa)

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

    # Solve the PDE at each time step
    for j in ts[1:]:
        u_jp1[a:b] = A_FE[a:b,a:b].dot(u_j[a:b])
        
        addboundaries(u_jp1, lbctype, rbctype,
                      lmbda*lbc(j),
                      lmbda*rbc(j),
                      -2*lmbda*deltax*lbc(j),
                      2*lmbda*deltax*rbc(j))

        # fix Dirichlet boundary conditions
        if lbctype == 'Dirichlet':
            u_jp1[0] = lbc(j)
        if rbctype == 'Dirichlet':
            u_jp1[mx] = rbc(j)
        
        # add source to inner terms
        u_jp1[1:-1] += deltat*source(xs[1:-1], j)
        
        u_j[:] = u_jp1[:]
    
    return xs, u_j

def backwardeuler(mx, mt, L, T, 
                  kappa, source,
                  ic, lbc, rbc, lbctype, rbctype):
    """Backward Euler finite-difference scheme (implicit) for solving 
    parabolic PDE problems. Unconditionally stable"""
    
    # initialise     
    xs, ts, deltax, deltat, lmbda = initialise(mx, mt, L, T, kappa)

    u_j = ic(xs)
    
    # if boundary conditions don't match initial conditions
    if lbctype == 'Dirichlet':
        u_j[0] = lbc(0)
    if rbctype == 'Dirichlet':
        u_j[mx] = rbc(0)
        

    u_jp1 = np.zeros(xs.size)
    
    # Construct forward Euler matrix
    B_FE = tridiag(mx, -lmbda, 1+2*lmbda, -lmbda)
    
    # modify first and last row for Neumann conditions
    B_FE[0,1] *= 2; B_FE[mx,mx-1] *= 2

    # range of rows of B_FE to use
    a, b = matrixrowrange(mx, lbctype, rbctype)

    # Solve the PDE at each time step
    for j in ts[1:]:
        addboundaries(u_j, lbctype, rbctype,
                      lmbda*lbc(j),
                      lmbda*rbc(j),
                      -2*lmbda*deltax*lbc(j),
                      2*lmbda*deltax*rbc(j))

        u_jp1[a:b] = spsolve(B_FE[a:b,a:b], u_j[a:b])
        
        # fix Dirichlet boundary conditions
        if lbctype == 'Dirichlet':
            u_jp1[0] = lbc(j)
        if rbctype == 'Dirichlet':
            u_jp1[mx] = rbc(j)
        
        # add source to inner terms
        u_jp1[1:-1] += deltat*source(xs[1:-1], j)
        
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