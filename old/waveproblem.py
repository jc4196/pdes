# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 11:25:12 2018

@author: jc4196
"""

import numpy as np
import sympy as sp
from sympy.abc import c, L, x, t
from scipy.sparse.linalg import spsolve
from diffusionproblem import Dirichlet, Neumann
from schemes import tridiag
from IPython.display import display
import matplotlib.pylab as pl


        
class WaveProblem():
    def __init__(self,
                 c=1,
                 L=1,
                 ix=sp.sin(sp.pi*x),
                 iv=0,
                 lbc=Dirichlet(0,0),
                 rbc=Dirichlet(1, 0),
                 source=0):
    
        self.c = c   # Wave speed
        self.L = L           # Length of interval
        self.lbc = lbc       # Left boundary condition as above
        self.rbc = rbc       # Right boundary condition as above
        
        # Initial condition function h(x)
        self.ix_expr = ix  # initial condition expression for printing
        self.ix = np.vectorize(sp.lambdify(x, ix, 'numpy'),
                               otypes=[np.float32])
        
        self.iv_expr = iv  # initial condition expression for printing
        self.iv = np.vectorize(sp.lambdify(x, iv, 'numpy'),
                               otypes=[np.float32])       
        
        # Source function f(x)
        self.source_expr = source # source expression for printing
        self.source = np.vectorize(sp.lambdify((x, t), source, 'numpy'),
                                   otypes=[np.float32])  
        
    def pprint(self, title=''):
        """Print the diffusion problem with latex"""
        print(title)
        u = sp.Function('u')
        x, t = sp.symbols('x t')
        display(sp.Eq(u(x,t).diff(t, 2),
                      c**2*u(x,t).diff(x,2) + self.source_expr))
        self.lbc.pprint()
        self.rbc.pprint()
        display(sp.Eq(u(x,0), self.ix_expr))
        display(sp.Eq(u(x,t).diff(x).subs(t,0), self.iv_expr))

    def solve_to(self, T, mx, mt, scheme, full_output=False):
        """Solve the diffusion problem forward to time T using the given
        scheme."""
        xs = np.linspace(0, self.L, mx+1)     # mesh points in space
        ts = np.linspace(0, T, mt+1)     # mesh points in time
        deltax = xs[1] - xs[0]            # gridspacing in x
        deltat = ts[1] - ts[0]            # gridspacing in t
        lmbda = self.c*deltat/deltax    # squared Courant number
    
        if full_output:
            print("deltax =",deltax)
            print("deltat =",deltat)
            print("lambda =",lmbda)
    
        # initialise explicit solver
        A_EW = tridiag(mx,0,0,0,0, 2-2*lmbda**2, lmbda**2, lmbda**2)

        # set initial condition
        u_jm1 = self.ix(xs) 
        
        # first time step
        u_j = np.zeros(xs.size)
        u_j[1:-1] = 0.5*A_EW[1:-1,1:-1].dot(u_jm1[1:-1]) + deltat*self.iv(xs)[1:-1]
        u_j[0] = 0; u_j[mx] = 0  # boundary condition     
        
        # u at next time step
        u_jp1 = np.zeros(xs.size)        
        
        for n in range(2,mt+1):
            u_jp1[1:-1] = A_EW[1:-1,1:-1].dot(u_j[1:-1]) - u_jm1[1:-1]
            
            # boundary conditions
            u_jp1[0] = 0; u_jp1[mx] = 0
            
            # update u_jm1 and u_j
            u_jm1[:],u_j[:] = u_j[:],u_jp1[:]
        
        return xs, u_j

    def plot_at_T(self,
                  T,
                  mx=30,
                  mt=60,
                  scheme=None,
                  u_exact=None,
                  title=''):
        """Plot the solution to the diffusion problem at time T.
        If the exact solution is known, plot that too and return the 
        error at time T."""
        xs, uT = self.solve_to(T, mx, mt, scheme, full_output=False)
        try:
            pl.plot(xs,uT,'ro',label='numerical')
        except:
            pass
            
        if u_exact:
            xx = np.linspace(0, self.L, 250)
            uTsym = u_exact.subs({c: self.c,
                                  L: self.L,
                                  t: T})
            u = sp.lambdify(x, uTsym)
            pl.plot(xx, u(xx),'b-',label=r'${}$'.format(sp.latex(uTsym)))
            error = np.linalg.norm(abs(u(xs) - uT))

        pl.xlabel('x')
        pl.ylabel('u(x,{})'.format(T))
        pl.title(title)
        pl.legend(loc='best')
        pl.show()
        
        if u_exact:
            return error   