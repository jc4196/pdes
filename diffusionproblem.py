# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 11:55:01 2018

@author: james
"""

import numpy as np
import sympy as sp
from sympy.abc import kappa, L, x, t, c
from scipy.sparse.linalg import spsolve

from IPython.display import display
import matplotlib.pylab as pl

from schemes import backwardeuler, cranknicholson, forwardeuler
from discretesolvepde import *

class BC:
    """General boundary condition for the diffusion problem of the form
    
    alpha*u(xb,t) + beta*du(xb,t)/dx = g(t)
    """
    def __init__(self, xb, params):
        self.u = sp.Function('u')
        self.xb = xb
        self.alpha, self.beta, self.rhs = params
        self.rhs_fn = sp.lambdify(t, self.rhs, 'numpy')
    
    def pprint(self):
        display(sp.Eq(self.alpha*self.u(x, t).subs(x,self.xb) + \
                      self.beta*self.u(x, t).diff(x).subs(x,self.xb),
                      self.rhs))
        
    def apply_rhs(self, t_step):
        return self.rhs_fn(t_step)
    
    def get_params(self):
        return self.alpha, self.beta
    
    def isDirichlet(self):
        return (self.alpha, self.beta) == (1, 0)
    
    def isNeumann(self):
        return (self.alpha, self.beta) == (0, 1)
    
class Dirichlet(BC):
    def __init__(self, xb, rhs):
        BC.__init__(self, xb, (1, 0, rhs))

class Neumann(BC):
    def __init__(self, xb, rhs):
        BC.__init__(self, xb, (0, 1, rhs))
        


class DiffusionProblem:
    """Object specifying a diffusion problem of the form
    
    du/dt = kappa*d^2u/dx^2 + f(x)
    
    BCs
    alpha1*u(0,t) + beta1*du(0,t)/dx = g_1(t)
    alpha2*u(L,t) + beta2*du(L,t)/dx = g_2(t)
    
    IC
    u(x,0) = h(x)
    """
    def __init__(self,
                 kappa=1,
                 L=1,
                 ic=sp.sin(sp.pi*x),
                 lbc=Dirichlet(0,0),
                 rbc=Dirichlet(1, 0),
                 source=0):
        self.kappa = kappa   # Diffusion constant
        self.L = L           # Length of interval
        self.lbc = lbc       # Left boundary condition as above
        self.rbc = rbc       # Right boundary condition as above
        
        # Initial condition function h(x)
        self.ic_expr = ic  # initial condition expression for printing
        self.ic = np.vectorize(sp.lambdify(x, ic, 'numpy'),
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
        display(sp.Eq(u(x,t).diff(t),
                      kappa*u(x,t).diff(x,2) + self.source_expr))
        self.lbc.pprint()
        self.rbc.pprint()
        display(sp.Eq(u(x,0), self.ic_expr))
    
    def boundarytype(self, mx):
        if self.lbc.isDirichlet() and self.rbc.isDirichlet():
            return 1, mx
        elif self.lbc.isDirichlet() and self.rbc.isNeumann():
            return 1, mx+1
        elif self.lbc.isDirichlet() and self.rbc.isNeumann():
            return 0, mx
        elif self.lbc.isNeumann() and self.rbc.isNeumann():
            return 0, mx+1
        else:
            raise Exception('Boundary type not recognised')
    
    def solve_at_T(self, T, mx, mt, scheme, plot=True, u_exact=None, title=''):
        xs, uT =  solve_diffusion_pde(mx, mt, self.L, T,
                                      self.kappa, self.source, self.ic,
                                      self.lbc.apply_rhs, self.rbc.apply_rhs,
                                      self.boundarytype(mx),
                                      scheme)
        
        if u_exact:
            uTsym = u_exact.subs({kappa: self.kappa,
                                  L: self.L,
                                  t: T})
            u = sp.lambdify(x, uTsym)
            error = np.linalg.norm(u(xs) - uT)            
            if plot:       
                plot_solution(xs, uT, u, title=title, uexacttitle=r'${}$'.format(sp.latex(uTsym)))            
        else:
            error = None
            if plot:
                plot_solution(xs, uT, title=title)            
        
        return uT, error

    def solve_at_Ts(self, t_range, mx, mt, scheme, animate=True):
        """Solve the diffusion problem for a range of times and animate the
        solution"""
        pass
    
    
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
        

    def boundarytype(self, mx):
        return 1, mx
    
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
        
    def solve_at_T(self, T, mx, mt, scheme, full_output=False):
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
    
    def solve_at_T(self, T, mx, mt, scheme, plot=True, u_exact=None, title=''):
        xs, uT = solve_wave_pde(mx, mt, self.L, T,
                                self.c, self.source, self.ix, self.iv,
                                self.lbc.apply_rhs, self.rbc.apply_rhs,
                                self.boundarytype(mx),
                                scheme)
        
        if u_exact:
            uTsym = u_exact.subs({c: self.c,
                                  L: self.L,
                                  t: T})
            u = sp.lambdify(x, uTsym)
            error = np.linalg.norm(u(xs) - uT)            
            if plot:       
                plot_solution(xs, uT, u, title=title, uexacttitle=r'${}$'.format(sp.latex(uTsym)))            
        else:
            error = None
            if plot:
                plot_solution(xs, uT, title=title)            
        
        return uT, error       