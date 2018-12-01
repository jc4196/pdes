# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 11:55:01 2018

@author: james
"""

import numpy as np
import sympy as sp
from sympy.abc import kappa, L, x, t

from IPython.display import display
import matplotlib.pylab as pl

from finitedifference import backwardeuler, forwardeuler, cranknicholson


class IC:
    def __init__(self, expr):
        self.u = sp.Function('u')
        self.expr = expr
    
    def lambdify(self):
        return(sp.lambdify(x, self.expr, 'numpy'))
        
    def pprint(self):
        display(sp.Eq(self.u(x, 0), self.expr))

class Dirichlet:
    def __init__(self, xb, expr):
        self.xb = xb
        self.u = sp.Function('u')
        self.expr = expr
    
    def lambdify(self):
        return sp.lambdify(t, self.expr, 'numpy')
    
    def pprint(self):
        display(sp.Eq(self.u(self.xb, t), self.expr))
        

#t = sp.symbols('t')
#d = Dirichlet(0, sp.sin(t))
#d.pprint()


default_ic = IC(sp.sin(sp.pi*x))

class DiffusionProblem:
    def __init__(self,
                 kappa=1,
                 L=1,
                 ic=default_ic,
                 lbc=Dirichlet(0, 0),
                 rbc=Dirichlet(1, 0),
                 source=lambda x, t: 0,
                 fd=backwardeuler):
        self.kappa = kappa   # Diffusion constant
        self.L = L           # Length of interval
        self.ic = ic         # Initial condition u(x,0)
        self.lbc = lbc       # Left boundary condition u(0,t)
        self.rbc = rbc       # Right boundary condition u(L,t)
        self.source = source # Source function
 
    def pprint(self, title=''):
        print(title)
        u = sp.Function('u')
        x, t = sp.symbols('x t')
        display(sp.Eq(u(x,t).diff(t), kappa*u(x,t).diff(x,2)))
        self.lbc.pprint()
        self.rbc.pprint()
        self.ic.pprint()
    
    def solve_to(self, T, mx, mt, scheme=backwardeuler, full_output=False):
        xs = np.linspace(0, self.L, mx+1)     # mesh points in space
        ts = np.linspace(0, T, mt+1)     # mesh points in time
        deltax = xs[1] - xs[0]            # gridspacing in x
        deltat = ts[1] - ts[0]            # gridspacing in t
        lmbda = self.kappa*deltat/(deltax**2)    # mesh fourier number
    
        if full_output:
            print("deltax =",deltax)
            print("deltat =",deltat)
            print("lambda =",lmbda)
    
        u0 = self.ic.lambdify()(xs)

        uT = scheme(T, mx, mt, lmbda, u0,
                    self.lbc.lambdify(), self.rbc.lambdify(), self.source)
        return xs, uT

    def plot_at_T(self, T, mx=20, mt=1000, u_exact=None, title=''):
        xs, uT = self.solve_to(T,mx,mt)
        pl.plot(xs,uT,'ro',label='num')
        
        if u_exact:
            xs = np.linspace(0, self.L, 250)
            uTsym = u_exact.subs({kappa: self.kappa,
                                  L: self.L,
                                  t: T})
            u = sp.lambdify(x, uTsym)
            pl.plot(xs, u(xs),'b-',label=r'${}$'.format(sp.latex(uTsym)))
        pl.xlabel('x')
        pl.ylabel('u(x,{})'.format(T))
        pl.title(title)
        pl.legend(loc='best')
        pl.show()
