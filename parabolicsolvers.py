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
from helpers import tridiag, vectorize_xfn, vectorize_xtfn, numpify_many
from visualizations import plot_solution

from scipy.sparse.linalg import spsolve


def initialise(mx, mt, L, T, kappa):
    xs = np.linspace(0, L, mx+1)          # mesh points in space
    ts = np.linspace(0, T, mt+1)          # mesh points in time
    deltax = xs[1] - xs[0]                # gridspacing in x
    deltat = ts[1] - ts[0]                # gridspacing in t
    lmbda = kappa*deltat/(deltax**2)      # mesh fourier number   
    
    return xs, ts, deltax, deltat, lmbda

def matrixrowrange(mx, lbctype, rbctype):
    a, b = 0, mx+1
    
    if lbctype == 'Dirichlet':
        a = 1
    if rbctype == 'Dirichlet':
        b = mx
        
    return a, b

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
    A_FE = tridiag(mx+1, lmbda, 1-2*lmbda, lmbda)
    
    # modify first and last row for Neumann conditions
    A_FE[0,1] *= 2; A_FE[mx,mx-1] *= 2

    # range of rows of A_FE to use
    a, b = matrixrowrange(mx, lbctype, rbctype)

    # Solve the PDE at each time step
    for t in ts[:-1]:
        u_jp1[a:b] = A_FE[a:b,a:b].dot(u_j[a:b])
        
        addboundaries(u_jp1, lbctype, rbctype,
                      lmbda*lbc(t),
                      lmbda*rbc(t),
                      -2*lmbda*deltax*lbc(t),
                      2*lmbda*deltax*rbc(t))

        # fix Dirichlet boundary conditions
        if lbctype == 'Dirichlet':
            u_jp1[0] = lbc(t + deltat)
        if rbctype == 'Dirichlet':
            u_jp1[mx] = rbc(t + deltat)
        
        # add source to inner terms
        u_jp1[1:-1] += deltat*source(xs[1:-1], t)
        
        u_j[:] = u_jp1[:]
    
    return xs, u_j

def backwardeuler(mx, mt, L, T, 
                  kappa, source,
                  ic, lbc, rbc, lbctype, rbctype):
    """Backward Euler finite-difference scheme (implicit) for solving 
    parabolic PDE problems. Unconditionally stable"""
    
    # initialise     
    xs, ts, deltax, deltat, lmbda = initialise(mx, mt, L, T, kappa)
 
    ic, lbc, rbc, source = numpify_many((ic,'x'), (lbc,'x'),
                                        (rbc, 'x'), (source, 'x t'))
    
    
    u_j = ic(xs)
    plot_solution(xs, u_j)
    # if boundary conditions don't match initial conditions
    if lbctype == 'Dirichlet':
        u_j[0] = lbc(0)
    if rbctype == 'Dirichlet':
        u_j[mx] = rbc(0)
        

    u_jp1 = np.zeros(xs.size)
    
    # Construct forward Euler matrix
    B_FE = tridiag(mx+1, -lmbda, 1+2*lmbda, -lmbda)
    
    # modify first and last row for Neumann conditions
    B_FE[0,1] *= 2; B_FE[mx,mx-1] *= 2

    # range of rows of B_FE to use
    a, b = matrixrowrange(mx, lbctype, rbctype)

    # Solve the PDE at each time step
    for t in ts[:-1]:
        addboundaries(u_j, lbctype, rbctype,
                      lmbda*lbc(t + deltat),
                      lmbda*rbc(t + deltat),
                      -2*lmbda*deltax*lbc(t + deltat),
                      2*lmbda*deltax*rbc(t + deltat))

        u_jp1[a:b] = spsolve(B_FE[a:b,a:b], u_j[a:b])
        
        # fix Dirichlet boundary conditions
        if lbctype == 'Dirichlet':
            u_jp1[0] = lbc(t + deltat)
        if rbctype == 'Dirichlet':
            u_jp1[mx] = rbc(t + deltat)
        
        # add source to inner terms
        u_jp1[1:-1] += deltat*source(xs[1:-1], t + deltat)
        
        u_j[:] = u_jp1[:]
    
    return xs, u_j

def cranknicholson(mx, mt, L, T, 
                   kappa, source,
                   ic, lbc, rbc, lbctype, rbctype):
    """Crank-Nicholson finite-difference scheme (implicit) for solving 
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
    v = np.zeros(xs.size)  
    
    # Construct Crank-Nicholson matrices
    A_CN = tridiag(mx+1, -0.5*lmbda, 1+lmbda, -0.5*lmbda)
    B_CN = tridiag(mx+1, 0.5*lmbda, 1-lmbda, 0.5*lmbda)
    # modify first and last row for Neumann conditions
    A_CN[0,1] *= 2; A_CN[mx,mx-1] *= 2; B_CN[0,1] *= 2; B_CN[mx,mx-1] *= 2

    # range of rows of matrices to use
    a, b = matrixrowrange(mx, lbctype, rbctype)

    # Solve the PDE at each time step
    for t in ts[:-1]:
        v[a:b] = B_CN[a:b,a:b].dot(u_j[a:b])
        
        addboundaries(v, lbctype, rbctype,
                      0.5*lmbda*(lbc(t) + lbc(t + deltat)),
                      0.5*lmbda*(rbc(t) + rbc(t + deltat)),
                      -lmbda*deltax*(lbc(t) + lbc(t + deltat)),
                      lmbda*deltax*(rbc(t) + rbc(t + deltat)))

        u_jp1[a:b] = spsolve(A_CN[a:b,a:b], v[a:b])
        
        # fix Dirichlet boundary conditions
        if lbctype == 'Dirichlet':
            u_jp1[0] = lbc(t)
        if rbctype == 'Dirichlet':
            u_jp1[mx] = rbc(t)
        
        # add source to inner terms
        u_jp1[1:-1] += 0.5*deltat*(source(xs[1:-1], t) + source(xs[1:-1], t + deltat))
        
        u_j[:] = u_jp1[:]
    
    return xs, u_j