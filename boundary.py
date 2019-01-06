# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 17:54:26 2019

@author: james
"""

import sympy as sp
from sympy.abc import x, t

from IPython.display import display

class BC:
    """General mixed boundary condition for the diffusion problem of the form
    alpha*u(xb,t) + beta*du(xb,t)/dx = rhs
    """
    def __init__(self, xb, params):
        self.u = sp.Function('u')
        self.xb = xb
        self.alpha, self.beta, self.rhs = params
        self.type = 'Mixed'
    
    def pprint(self):
        display(sp.Eq(self.alpha*self.u(x, t).subs(x, self.xb) + \
                      self.beta*self.u(x, t).diff(x).subs(x,self.xb),
                      self.rhs))

    
class Dirichlet(BC):
    """
    Dirichlet boundary condition of the form
    u(xb, t) = rhs
    """
    def __init__(self, xb, rhs):
        BC.__init__(self, xb, (1, 0, rhs))
        self.type = 'Dirichlet'
    

class Neumann(BC):
    """
    Neumann boundary condition of the form
    du(xb,t)/dx = rhs
    """
    def __init__(self, xb, rhs):
        BC.__init__(self, xb, (0, 1, rhs))
        self.type = 'Neumann'
        
    
class Open:
    """
    Open boundary condition either of the form
    left**
    or 
    right**
    """
    def __init__(self, xb, c):
        self.u = sp.Function('u')
        self.xb = xb
        self.c = c
        self.rhs = lambda t: 0
        self.type = 'Open'
    
    def pprint(self):
        # this only works for left open bcs at the moment
        display(sp.Eq(self.u(x,t).diff(t).subs(x, self.xb)
                        + c*self.u(x,t).diff(x).subs(x, self.xb), 0))

class Periodic:
    def __init__(self, x1, x2):
        self.u = sp.Function('u')
        self.xb = x1
        self.rhs_fn = lambda t: 0
        self.type = 'Periodic'
        
    def pprint(self):
        display(sp.Eq(self.u(x,t).subs(x, self.x1), self.u(x,t).subs(x,self.x2)))
        