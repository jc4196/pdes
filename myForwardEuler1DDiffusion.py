# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 10:08:35 2018

@author: jc4196
"""

# simple forward Euler solver for the 1D heat equation
#   u_t = kappa u_xx  0<x<L, 0<t<T
# with zero-temperature boundary conditions
#   u=0 at x=0,L, t>0
# and prescribed initial temperature
#   u=u_I(x) 0<=x<=L,t=0

import numpy as np
import pylab as pl
from math import pi

# set problem parameters/functions
kappa = 1.0   # diffusion constant
L=1.0         # length of spatial domain
T=0.5        # total time to solve for

def u_I(x):
    # initial temperature distribution
    y = np.sin(pi*x/L)
    return y

def u_exact(x,t):
    # the exact solution
    y = np.exp(-kappa*(pi**2/L**2)*t)*np.sin(pi*x/L)
    return y


def tridiag(a, b, c, k1=-1, k2=0, k3=1):
    # create a tridiagonal matrix on diagonals k1, k2, k3
    return np.diag(a, k1) + np.diag(b, k2) + np.diag(c, k3)

def FE_solve_to(T, mx, mt, lmbda, u_0):   
    # Construct the forward Euler matrix
    A_FE = tridiag((mx-2)*[lmbda], (mx-1)*[1-2*lmbda], (mx-2)*[lmbda])
    
    u_jp1 = np.zeros(u_0.size)      # u at next time step 
    u_j = u_0    # u at the current time step
    
    # Solve the PDE: loop over all time points
    for n in range(1, mt+1):
        # Forward Euler timestep at inner mesh points
        
        # for loop version
        # for i in range(1, mx):
        #     u_jp1[i] = u_j[i] + lmbda*(u_j[i-1] - 2*u_j[i] + u_j[i+1])
        
        # Matrix version
        u_jp1[1:-1] = np.dot(A_FE, u_j[1:-1])
        
        # Boundary conditions
        u_jp1[0] = 0; u_jp1[mx] = 0
            
        # Update u_j
        u_j[:] = u_jp1[:]
    
    return u_j

def BE_solve_to(T, mx, mt, lmbda, u_0):
    # Construct the backward Euler matrix
    A_BE = tridiag((mx-2)*[-lmbda], (mx-1)*[1+2*lmbda], (mx-2)*[-lmbda])
    
    u_jp1 = np.zeros(u_0.size)      # u at next time step 
    u_j = u_0    # u at the current time step
    
    for n in range(1, mt+1):    
        # Backward Euler timestep at inner mesh points
        u_jp1[1:-1] = np.linalg.solve(A_BE, u_j[1:-1])

        # Boundary conditions
        u_jp1[0] = 0; u_jp1[mx] = 0 # this seems unnecessary?
            
        # Update u_j
        u_j[:] = u_jp1[:]      
        
    return u_j

def forward_euler_diffusion(mx, mt):
    # set up the numerical environment variables
    x = np.linspace(0, L, mx+1)     # mesh points in space
    t = np.linspace(0, T, mt+1)     # mesh points in time
    deltax = x[1] - x[0]            # gridspacing in x
    deltat = t[1] - t[0]            # gridspacing in t
    lmbda = kappa*deltat/(deltax**2)    # mesh fourier number
    A_FE = tridiag((mx-2)*[lmbda], (mx-1)*[1-2*lmbda], (mx-2)*[lmbda])
    
    print("deltax =",deltax)
    print("deltat =",deltat)
    print("lambda =",lmbda)
    
    # set up the solution variables
    u_j = np.zeros(x.size)        # u at current time step
    u_jp1 = np.zeros(x.size)      # u at next time step    
    
    # Set initial condition
    for i in range(0, mx+1):
        u_j[i] = u_I(x[i])

    # Solve the PDE: loop over all time points
    for n in range(1, mt+1):
        # Forward Euler timestep at inner mesh points
        
        # with a for loop
        # for i in range(1, mx):
        #     u_jp1[i] = u_j[i] + lmbda*(u_j[i-1] - 2*u_j[i] + u_j[i+1])
        
        # matrix version
        u_jp1[1:-1] = np.dot(A_FE, u_j[1:-1])
        
        # Boundary conditions
        u_jp1[0] = 0; u_jp1[mx] = 0
            
        # Update u_j
        u_j[:] = u_jp1[:]
        
    return x, u_j


def backward_euler_diffusion(mx,mt):
    # set up the numerical environment variables
    x = np.linspace(0, L, mx+1)     # mesh points in space
    t = np.linspace(0, T, mt+1)     # mesh points in time
    deltax = x[1] - x[0]            # gridspacing in x
    deltat = t[1] - t[0]            # gridspacing in t
    lmbda = kappa*deltat/(deltax**2)    # mesh fourier number
    print("deltax =",deltax)
    print("deltat =",deltat)
    print("lambda =",lmbda)
    
    
    # Set initial condition
    for i in range(0, mx+1):
        u_j[i] = u_I(x[i])
        
    for n in range(1, mt+1):    
        # Backward Euler timestep at inner mesh points
        u_jp1[1:-1] = np.linalg.solve(A_BE, u_j[1:-1])

        # Boundary conditions
        u_jp1[0] = 0; u_jp1[mx] = 0
            
        # Update u_j
        u_j[:] = u_jp1[:]      
        
    return x, u_j
  
    
def solve_diffusion_ibvp(kappa, L, T, mx, mt, solver=BE_solve_to):
    # set up the numerical environment variables
    x = np.linspace(0, L, mx+1)     # mesh points in space
    t = np.linspace(0, T, mt+1)     # mesh points in time
    deltax = x[1] - x[0]            # gridspacing in x
    deltat = t[1] - t[0]            # gridspacing in t
    lmbda = kappa*deltat/(deltax**2)    # mesh fourier number
    
    print("deltax =",deltax)
    print("deltat =",deltat)
    print("lambda =",lmbda)
    
    u_0 = np.zeros(x.size)
    
    # Set initial condition
    for i in range(0, mx+1):
        u_0[i] = u_I(x[i])
        
    u_T = solver(T, mx, mt, lmbda, u_0)
    
    return x, u_T
    
# set numerical parameters
mx = 20     # number of gridpoints in space
mt = 1000   # number of gridpoints in time   


# result
x, u_T = solve_diffusion_ibvp(kappa, L, T, mx, mt)

# plot the final result and exact solution
pl.plot(x,u_T,'ro',label='num')
xx = np.linspace(0,L,250)
pl.plot(xx,u_exact(xx,T),'b-',label='exact')
pl.xlabel('x')
pl.ylabel('u(x,0.5)')
pl.legend(loc='upper right')
pl.show