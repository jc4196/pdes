# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 11:55:01 2018

@author: james
"""

import numpy as np
import matplotlib.pylab as pl
from math import pi

def tridiag(a, b, c, k1=-1, k2=0, k3=1):
    # create a tridiagonal matrix on diagonals k1, k2, k3
    return np.diag(a, k1) + np.diag(b, k2) + np.diag(c, k3)

def BE(T, mx, mt, lmbda, u_0, lbc, rbc, f):
    # Construct the backward Euler matrix
    A_BE = tridiag((mx)*[-lmbda], (mx+1)*[1+2*lmbda], (mx)*[-lmbda])
    
    # Add the boundary conditions
    A_BE[0,0] = A_BE[mx,mx] = 1
    A_BE[0,1] = A_BE[mx,mx-1] = 0

    deltat = T / (mt+1)
    
    u_jp1 = np.zeros(u_0.size)      # u at next time step 
    u_j = u_0    # u at the current time step
    
    for n in range(1, mt+1):    
        # Backward Euler timestep at inner mesh points
        u_jp1 = np.linalg.solve(A_BE, u_j)

        # Boundary conditions
        u_jp1[0] = lbc(n*deltat); u_jp1[mx] = rbc(n*deltat) 
            
        # Update u_j
        u_j[:] = u_jp1[:]
        u_j[1:mt] += deltat*f(n*deltat)
   
    return u_j

class DiffusionProblem:
    def __init__(self,
                 L = 1,
                 ic=lambda x:np.sin(pi*x),
                 lbc=lambda x: 0,
                 rbc=lambda x: 0,
                 f=lambda x: 0,
                 solver=BE):
        self.L = L
        self.ic = ic
        self.lbc = lbc
        self.rbc = rbc
        self.f = f
        self.solver = solver
        
    def solve_to(self, T, mx=30, mt=1000):
        x = np.linspace(0, self.L, mx+1)     # mesh points in space
        t = np.linspace(0, T, mt+1)     # mesh points in time
        deltax = x[1] - x[0]            # gridspacing in x
        deltat = t[1] - t[0]            # gridspacing in t
        lmbda = kappa*deltat/(deltax**2)    # mesh fourier number
    
        print("deltax =",deltax)
        print("deltat =",deltat)
        print("lambda =",lmbda)
    
        u_0 = self.ic(x)
        u_T = self.solver(T, mx, mt, lmbda, u_0,
                          self.lbc, self.rbc, self.f)
        pl.plot(x, u_T)
        return x, u_T
