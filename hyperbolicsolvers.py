# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 10:56:19 2018

@author: james
"""

import numpy as np
from scipy.sparse.linalg import spsolve
from scipy import sparse

from helpers import tridiag
from visualizations import plot_solution


def initialise(mx, mt, L, T, c):
    xs = np.linspace(0, L, mx+1)     # mesh points in space
    ts = np.linspace(0, T, mt+1)      # mesh points in time
    deltax = xs[1] - xs[0]            # gridspacing in x
    deltat = ts[1] - ts[0]            # gridspacing in t
    lmbda = c*deltat/deltax      # squared Courant number  
    
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
        #u[1] += D1
        pass
    elif lbctype == 'Open' or lbctype == 'Periodic':
        pass
    else:
        raise Exception('That boundary condition is not implemented')
    
    if rbctype == 'Neumann':
        u[-1] += Nmx
    elif rbctype == 'Dirichlet':
        #u[-2] += Dmxm1
        pass
    elif rbctype == 'Open' or rbctype == 'Periodic':
        pass
    else:
        raise Exception('That boundary condition is not implemented')


def explicitsolve(mx, mt, L, T,
                  c, source,
                  ix, iv,
                  lbc, rbc, lbctype, rbctype):
    """Solve a wave equation problem with the given spacing"""
    xs, ts, deltax, deltat, lmbda = initialise(mx, mt, L, T, c)
    print("lambda = %f" % lmbda)
    # Construct explicit wave matrix
    A_EW = tridiag(mx+1, lmbda**2, 2-2*lmbda**2, lmbda**2)
    
    ##### Put changes to the matrix into a separate function ######
    A_EW[0,1] *= 2; A_EW[mx,mx-1] *= 2
    
    # This was an attempt to include the open boundary condition in the matrix
    #if lbctype == 'Open':
    #    A_EW[0,0] = 1 + lmbda; A_EW[0,1] = -lmbda
    #if rbctype =='Open':
    #    A_EW[mx,mx-1] = 1 - lmbda; A_EW[mx,mx] = lmbda
    
    ### and perhaps include this ###
    a, b = matrixrowrange(mx, lbctype, rbctype)

    # initial condition vectors
    U = ix(xs)
    V = iv(xs)
    
    # set first two time steps
    u_jm1 = U 

    u_j = np.zeros(xs.size)
    u_j[a:b] = 0.5*A_EW[a:b,a:b].dot(U[a:b]) + deltat*V[a:b]

    # boundary conditions (may not match initial conditions)

    ### separate function for adding Dirichlet conditions ###
    if lbctype == 'Dirichlet':
        u_j[0] = lbc(0)
        
    if rbctype == 'Dirichlet':
        u_j[mx] = rbc(0)     
 
    # initialise u at next time step
    u_jp1 = np.zeros(xs.size)        
    
    for t in ts[1:-1]:
        u_jp1[a:b] = A_EW[a:b,a:b].dot(u_j[a:b]) - u_jm1[a:b]

        addboundaries(u_jp1, lbctype, rbctype,
                      lmbda**2*lbc(t - deltat),
                      lmbda**2*rbc(t - deltat),
                      -2*lmbda**2*deltax*lbc(t - deltat),
                      2*lmbda**2*deltax*rbc(t - deltat))

        # apply open boundary conditions
        if lbctype == 'Open':
            u_jp1[0] = (1-lmbda)*u_j[0] + lmbda*u_j[1]
        if rbctype == 'Open':
            u_jp1[mx] = lmbda*u_j[mx-1] + (1-lmbda)*u_j[mx]
        
        # fix Dirichlet boundary conditions
        if lbctype == 'Dirichlet':
            u_jp1[0] = lbc(t + deltat)
        if rbctype == 'Dirichlet':
            u_jp1[mx] = rbc(t + deltat)
        
        # add source to inner terms
        #u_jp1[1:-1] += deltat*source(xs[-1:1], t)
        
        # update u_jm1 and u_j
        u_jm1[:], u_j[:] = u_j[:], u_jp1[:]
    
    return xs, u_j


def explicitsolve(mx, mt, L, T,
                  c, source,
                  ix, iv,
                  lbc, rbc, lbctype, rbctype):
    """Solve a wave equation problem with the given spacing"""
    xs, ts, deltax, deltat, lmbda = initialise(mx, mt, L, T, c)
    print("lambda = %f" % lmbda)
    
    # Construct explicit wave matrix
    A_EW = tridiag(mx+1, lmbda**2, 2-2*lmbda**2, lmbda**2)    
    
    ##### Put changes to the matrix into a separate function ######
    if lbctype == 'Neumann':
        A_EW[0,1] *= 2
    elif lbctype == 'Open' or lbctype == 'Periodic':
        A_EW[0,0] = 2*(1 + lmbda - lmbda**2); A_EW[0,1] = 2*lmbda**2
    
    if rbctype == 'Neumann':
        A_EW[mx,mx-1] *= 2
    elif rbctype == 'Open' or rbctype == 'Periodic':
        A_EW[mx,mx-1] = 2*lmbda**2; A_EW[mx,mx] = 2*(1 + lmbda - lmbda**2)
    
    # This was an attempt to include the open boundary condition in the matrix
    #if lbctype == 'Open':
    #    A_EW[0,0] = 1 + lmbda; A_EW[0,1] = -lmbda
    #if rbctype =='Open':
    #    A_EW[mx,mx-1] = 1 - lmbda; A_EW[mx,mx] = lmbda
    
    ### and perhaps include this ###
    a, b = matrixrowrange(mx, lbctype, rbctype)

    # initial condition vectors
    U = ix(xs)
    V = iv(xs)
    
    # set first two time steps
    u_jm1 = U 

    u_j = np.zeros(xs.size)
    u_j = 0.5*A_EW.dot(U) + deltat*V

    # boundary conditions (may not match initial conditions)

    ### separate function for adding Dirichlet conditions ###
    if lbctype == 'Dirichlet':
        u_j[0] = lbc(0)
        
    if rbctype == 'Dirichlet':
        u_j[mx] = rbc(0)     
 
    # initialise u at next time step
    u_jp1 = np.zeros(xs.size)        
   
    zero_left, zero_right = True, True
    
    for t in ts[1:-1]:
        u_jp1 = A_EW.dot(u_j) - u_jm1

        addboundaries(u_jp1, lbctype, rbctype,
                      lmbda**2*lbc(t - deltat),
                      lmbda**2*rbc(t - deltat),
                      -2*lmbda**2*deltax*lbc(t - deltat),
                      2*lmbda**2*deltax*rbc(t - deltat))

        # apply open boundary conditions
        if lbctype == 'Open' or (lbctype == 'Periodic' and zero_right):
            #u_jp1[0] = (1-lmbda)*u_j[0] + lmbda*u_j[1]
            u_jp1[0] /= (1+2*lmbda)
        if rbctype == 'Open' or (rbctype == 'Periodic' and zero_left):
            #u_jp1[mx] = lmbda*u_j[mx-1] + (1-lmbda)*u_j[mx]
            u_jp1[mx] /= (1+2*lmbda)
        
        # fix Dirichlet boundary conditions
        if lbctype == 'Dirichlet':
            u_jp1[0] = lbc(t + deltat)
        if rbctype == 'Dirichlet':
            u_jp1[mx] = rbc(t + deltat)
        
        if zero_left:
            if u_jp1[0] > 1e-6:
                zero_left = False
        
        if zero_right:            
            if u_jp1[mx] > 1e-6:
                zero_right = False
        
        if lbctype == 'Periodic':
            if zero_right == False:
                u_jp1[0] = u_jp1[mx]
                
        if rbctype == 'Periodic':
            if zero_left == False:
                u_jp1[mx] = u_jp1[0]
                
        # add source to inner terms
        #u_jp1[1:-1] += deltat*source(xs[-1:1], t)
        
        # update u_jm1 and u_j
        u_jm1[:], u_j[:] = u_j[:], u_jp1[:]
    
    return xs, u_j

def tsunami_solve(mx, mt, L, T, h0, h, wave):
    """Variable wavespeed problem assumes periodic boundary on the left and 
    an open boundary on the right"""
    xs = np.linspace(0, L, mx+1)      # mesh points in space
    ts = np.linspace(0, T, mt+1)      # mesh points in time
    deltax = xs[1] - xs[0]            # gridspacing in x
    deltat = ts[1] - ts[0]            # gridspacing in t  
    delta = deltat/deltax
    lmbda = delta*np.sqrt(h0)
    print('lambda = {}'.format(lmbda))

    # construct explicit wave matrix for variable wave speed problem
    lower = [delta**2*h(i - 0.5*deltax) for i in xs[1:-1]] + [2*lmbda**2]
    main = [2*(1 + lmbda - lmbda**2)] + \
            [2 - delta**2*(h(i + 0.5*deltax) + h(i - 0.5*deltax)) for i in xs[1:-1]] + \
             [2*(1 + lmbda - lmbda**2)]
    upper = [2*lmbda**2] + [delta**2*h(i + 0.5*deltax) for i in xs[1:-1]]

    A_EW = sparse.diags([lower,main,upper], offsets=[-1,0,1], format='csr')
    print(A_EW.todense())

    plot_solution(xs, h(xs))
    
     # initial condition vectors

    U = wave(xs)
    
    # set first two time steps
    u_jm1 = U 

    u_j = np.zeros(xs.size)
    u_j = 0.5*A_EW.dot(U)
    
    u_j[0] /= (1+2*lmbda); u_j[mx] /= (1+2*lmbda)

    # initialise u at next time step
    u_jp1 = np.zeros(xs.size)        
   
    zero_right = True
    
    for t in ts[1:-1]:
        u_jp1 = A_EW.dot(u_j) - u_jm1
        
        u_jp1[mx] /= (1+2*lmbda)
        
        if zero_right and u_jp1[mx] > 1e-5:
            zero_right = False
            print('first non zero right at time {}'.format(t))
        
        if zero_right:
            u_jp1[0] /= (1+2*lmbda)
        else:
            u_jp1[0] = u_jp1[mx]
        
        
        # update u_jm1 and u_j
        u_jm1[:], u_j[:] = u_j[:], u_jp1[:]
    
    return xs, u_j    
    
    
    
    
def implicitsolve(mx, mt, L, T,
                  c, source,
                  ix, iv,
                  lbc, rbc, lbctype, rbctype):
    xs, ts, deltax, deltat, lmbda = initialise(mx, mt, L, T, c)
  
    # Get matrices and vector for the particular scheme
    A_IW = tridiag(mx+1, -0.5*lmbda**2, 1+lmbda**2, -0.5*lmbda**2)
    B_IW = tridiag(mx+1, 0.5*lmbda**2, -1-lmbda**2, 0.5*lmbda**2)
    
    # corrections for Neumann conditions
    A_IW[0,1] *= 2; A_IW[mx,mx-1] *= 2; B_IW[0,1] *= 2; B_IW[mx,mx-1] *= 2 

    a, b = matrixrowrange(mx, lbctype, rbctype)
    
    # initial condition vectors
    U = ix(xs)
    V = iv(xs)
    
    # set first two time steps
    u_jm1 = U 
    
    u_j = np.zeros(xs.size)
    w = np.zeros(xs.size)
    
    w[a:b] = U[a:b] - deltat*B_IW[a:b,a:b].dot(V[a:b])
    u_j[a:b] = spsolve(A_IW[a:b,a:b], w[a:b])
    
    # boundary conditions (may not match initial conditions)   
    if lbctype == 'Dirichlet':
        u_j[0] = lbc(0)
        
    if rbctype == 'Dirichlet':
        u_j[mx] = rbc(0)     
    

    # initialise u at next time step
    u_jp1 = np.zeros(xs.size)
    v = np.zeros(xs.size)        
    

    for t in ts[1:-1]:
        v[a:b] = B_IW[a:b,a:b].dot(u_jm1[a:b]) + 2*u_j[a:b]
 
        addboundaries(v, lbctype, rbctype,
                      0.5*lmbda**2*(lbc(t) + lbc(t + deltat)),
                      0.5*lmbda**2*(rbc(t) + rbc(t + deltat)),
                       -lmbda**2*deltax*(lbc(t) + lbc(t + deltat)),
                      lmbda**2*deltax*(rbc(t) + rbc(t + deltat)))

        u_jp1[a:b] = spsolve(A_IW[a:b,a:b], v[a:b])
        
        if lbctype == 'Dirichlet':
            u_jp1[0] = lbc(t + deltat)
        if rbctype == 'Dirichlet':
            u_jp1[mx] = rbc(t + deltat)
    
            
        # add source to inner terms
        u_jp1[1:-1] += deltat*source(xs[1:-1], t)
        
        # update u_jm1 and u_j
        u_jm1[:], u_j[:] = u_j[:], u_jp1[:]
    
    return xs, u_j


def implicitsolve(mx, mt, L, T,
                  c, source,
                  ix, iv,
                  lbc, rbc, lbctype, rbctype):
    xs, ts, deltax, deltat, lmbda = initialise(mx, mt, L, T, c)
  
    # Get matrices and vector for the particular scheme
    A_IW = tridiag(mx+1, -0.5*lmbda**2, 1+lmbda**2, -0.5*lmbda**2)
    B_IW = tridiag(mx+1, 0.5*lmbda**2, -1-lmbda**2, 0.5*lmbda**2)
    
    # corrections for Neumann conditions
    A_IW[0,1] *= 2; B_IW[0,1] *= 2; A_IW[mx,mx-1] *= 2; B_IW[mx,mx-1] *= 2


    a, b = matrixrowrange(mx, lbctype, rbctype)
    
    # initial condition vectors
    U = ix(xs)
    V = iv(xs)
    
    # set first two time steps
    u_jm1 = U 
    
    u_j = np.zeros(xs.size)
    w = np.zeros(xs.size)
    
    w = U - deltat*B_IW.dot(V)
    u_j = spsolve(A_IW, w)
    
    # boundary conditions (may not match initial conditions)   
    if lbctype == 'Dirichlet':
        u_j[0] = lbc(0)
        
    if rbctype == 'Dirichlet':
        u_j[mx] = rbc(0)     
    

    # initialise u at next time step
    u_jp1 = np.zeros(xs.size)
    v = np.zeros(xs.size)        
    

    for t in ts[1:-1]:
        v = B_IW.dot(u_jm1) + 2*u_j
 
        addboundaries(v, lbctype, rbctype,
                      0.5*lmbda**2*(lbc(t) + lbc(t + deltat)),
                      0.5*lmbda**2*(rbc(t) + rbc(t + deltat)),
                       -lmbda**2*deltax*(lbc(t) + lbc(t + deltat)),
                      lmbda**2*deltax*(rbc(t) + rbc(t + deltat)))

        u_jp1 = spsolve(A_IW, v)
        
        if lbctype == 'Dirichlet':
            u_jp1[0] = lbc(t + deltat)
        if rbctype == 'Dirichlet':
            u_jp1[mx] = rbc(t + deltat)
        
        # apply open boundary conditions
        if lbctype == 'Open':
            u_jp1[0] = (1-lmbda)*u_j[0] + lmbda*u_j[1]
        if rbctype == 'Open':
            u_jp1[mx] = lmbda*u_j[mx-1] + (1-lmbda)*u_j[mx]
            
        # add source to inner terms
        u_jp1[1:-1] += deltat*source(xs[1:-1], t)
        
        # update u_jm1 and u_j
        u_jm1[:], u_j[:] = u_j[:], u_jp1[:]
    
    return xs, u_j