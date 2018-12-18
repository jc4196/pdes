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

from parabolicsolvers import forwardeuler
from hyperbolicsolvers import tsunami_solve
from visualizations import plot_solution


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
    
    def get_type(self):
        return 'Dirichlet'

class Neumann(BC):
    def __init__(self, xb, rhs):
        BC.__init__(self, xb, (0, 1, rhs))
        
    def get_type(self):
        return 'Neumann'
    
class Open(BC):
    def __init__(self, xb, c):
        self.u = sp.Function('u')
        self.xb = xb
        self.c = c
        self.rhs_fn = lambda t: 0
    
    def pprint(self):
        display(sp.Eq(self.u(x,t).diff(t).subs(x, self.xb)
                        + c*self.u(x,t).diff(x).subs(x, self.xb), 0))
        
    def get_type(self):
        return 'Open'
        
class Periodic(BC):
    def __init__(self, x1, x2):
        self.u = sp.Function('u')
        self.xb = x1
        self.rhs_fn = lambda t: 0
        
    def pprint(self):
        display(sp.Eq(self.u(x,t).subs(x, self.x1), self.u(x,t).subs(x,self.x2)))
        
    def get_type(self):
        return 'Periodic'

class ParabolicProblem:
    """Object specifying a diffusion type problem of the form
    
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
                      self.kappa*u(x,t).diff(x,2) + self.source_expr))
        self.lbc.pprint()
        self.rbc.pprint()
        display(sp.Eq(u(x,0), self.ic_expr))
    
    def solve_at_T(self, T, mx, mt, scheme, plot=True, u_exact=None, title=''):
        xs, uT =  scheme(mx, mt, self.L, T,
                         self.kappa, self.source, self.ic,
                         self.lbc.apply_rhs, self.rbc.apply_rhs,
                         self.lbc.get_type(), self.rbc.get_type())
        
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
    
    
class HyperbolicProblem():
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
        
        # Initial condition functions
        self.ix_expr = ix  # initial condition expression for printing
        self.ix = np.vectorize(sp.lambdify(x, ix, 'numpy'),
                                   otypes=[np.float32])
        
        self.iv_expr = iv  
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
                      self.c**2*u(x,t).diff(x,2) + self.source_expr))
        self.lbc.pprint()
        self.rbc.pprint()
        display(sp.Eq(u(x,0), self.ix_expr))
        display(sp.Eq(u(x,t).diff(x).subs(t,0), self.iv_expr))
        
    
    def solve_at_T(self, T, mx, mt, scheme, plot=True, u_exact=None, title=''):
        xs, uT = scheme(mx, mt, self.L, T,
                        self.c, self.source,
                        self.ix, self.iv,
                        self.lbc.apply_rhs, self.rbc.apply_rhs,
                        self.lbc.get_type(), self.rbc.get_type())
        
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
    
class TsunamiProblem:
    def __init__(self,
                 L=30,
                 h = 1,
                 ix= 2 + 0.5*sp.exp(-(x-2.5)**2/0.5),
                 iv=0):
        self.L = L
        self.h = np.vectorize(sp.lambdify(x, h, 'numpy'), otypes=[np.float32])
        self.ix = sp.lambdify(x, ix, 'numpy')
        self.iv = np.vectorize(sp.lambdify(x, iv, 'numpy'),
                               otypes = [np.float32])
        
    def solve_at_T(self, T, mx, mt):
        xs, uT = tsunami_solve(mx, mt, self.L, T, self.h, self.ix, self.iv)
        
        plot_solution(xs, uT, style='b-')
        
        return uT
        
        
    