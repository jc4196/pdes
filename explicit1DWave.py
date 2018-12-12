# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 11:22:13 2018

@author: jc4196
"""

# simple explicit finite difference solver for the 1D wave equation
#   u_tt = c^2 u_xx  0<x<L, 0<t<T
# with zero-displacement boundary conditions
#   u=0 at x=0,L, t>0
# and prescribed initial displacement and velocity
#   u=u_I(x), u_t=v_I(x)  0<=x<=L,t=0

import numpy as np
import pylab as pl
from math import pi

# Set problem parameters/functions
c=1.0         # wavespeed
L=1.0         # length of spatial domain
T=2.0         # total time to solve for
def u_I(x):
    # initial displacement
    y = np.sin(pi*x/L)
    return y

def v_I(x):
    # initial velocity
    y = np.zeros(x.size)
    return y

def u_exact(x,t):
    # the exact solution
    y = np.cos(pi*c*t/L)*np.sin(pi*x/L)
    return y

# Set numerical parameters
mx = 10     # number of gridpoints in space
mt = 50     # number of gridpoints in time


# Set up the numerical environment variables
x = np.linspace(0, L, mx+1)     # gridpoints in space
t = np.linspace(0, T, mt+1)     # gridpoints in time
deltax = x[1] - x[0]            # gridspacing in x
deltat = t[1] - t[0]            # gridspacing in t
lmbda = c*deltat/deltax         # squared courant number
print("lambda=",lmbda)

# set up the solution variables
u_jm1 = np.zeros(x.size)        # u at previous time step
u_j = np.zeros(x.size)          # u at current time step
u_jp1 = np.zeros(x.size)        # u at next time step

# Set initial condition
for i in range(0, mx+1):
    u_jm1[i] = u_I(x[i])    

# First timestep
for i in range(1,mx):
    u_j[i] = u_jm1[i] + 0.5*(lmbda**2)*(u_jm1[i-1] - 2*u_jm1[i] + u_jm1[i+1]) \
                + deltat*v_I(x[i])

# First timestep boundary condition
u_j[0] = 0; u_j[mx]=0;

# Solve the PDE: loop over all time points
for n in range(2, mt+1):
    # regular timestep at inner mesh points
    for i in range(1, mx):
        u_jp1[i] = 2*u_j[i] + (lmbda**2)*(u_j[i-1] - 2*u_j[i] + u_j[i+1]) \
                    - u_jm1[i]
        
    # boundary conditions
    u_jp1[0] = 0; u_jp1[mx] = 0
            
    # update u_jm1 and u_j
    u_jm1[:],u_j[:] = u_j[:],u_jp1[:]

# Plot the final result and exact solution
pl.plot(x,u_jp1,'ro',label='numeric')
xx = np.linspace(0,L,250)
pl.plot(xx,u_exact(xx,T),'b-',label='exact')
pl.xlabel('x')
pl.ylabel('u(x,T)')
pl.legend()
pl.show