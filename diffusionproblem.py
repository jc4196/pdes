# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 11:55:01 2018

@author: james
"""

import numpy as np
import matplotlib.pylab as pl
from math import pi

from finitedifference import backwardeuler, forwardeuler, cranknicholson

class DiffusionProblem:
    def __init__(self,
                 kappa=1,
                 L=1,
                 ic=lambda x: np.sin(pi*x),
                 lbc=lambda t: 0,
                 rbc=lambda t: 0,
                 f=lambda x, t: 0,
                 fd=backwardeuler):
        self.kappa = kappa   # Diffusion constant
        self.L = L           # Length of interval
        self.ic = ic         # Initial condition u(x,0)
        self.lbc = lbc       # Left boundary condition u(0,t)
        self.rbc = rbc       # Right boundary condition u(L,t)
        self.f = f           # Forcing function f
        self.fd = fd         # Finite Difference method (forward/backward euler..)
        
    def solve_to(self, T, mx, mt, full_output=False):
        x = np.linspace(0, self.L, mx+1)     # mesh points in space
        t = np.linspace(0, T, mt+1)     # mesh points in time
        deltax = x[1] - x[0]            # gridspacing in x
        deltat = t[1] - t[0]            # gridspacing in t
        lmbda = self.kappa*deltat/(deltax**2)    # mesh fourier number
    
        if full_output:
            print("deltax =",deltax)
            print("deltat =",deltat)
            print("lambda =",lmbda)
    
        if isinstance(self.ic, (int, float)):
            u0 = np.full(x.size, self.ic, dtype=np.float64)
        else:
            u0 = self.ic(x)
  
        uT = self.fd(T, mx, mt, lmbda, u0,
                     self.lbc, self.rbc, self.f)
        return x, uT

    def plot_at_T(self, T, mx=20, mt=1000, u_exact=None, title=''):
        x, uT = self.solve_to(T,mx,mt)
        pl.plot(x,uT,'ro',label='num')
        
        if u_exact:
            x = np.linspace(0, self.L, 250)
            pl.plot(x,u_exact(x,T),'b-',label='exact')
        pl.xlabel('x')
        pl.ylabel('u(x,{})'.format(T))
        pl.title(title)
        pl.legend(loc='upper right')
        pl.show()
