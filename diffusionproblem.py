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
        self.ic_fn = np.vectorize(sp.lambdify(x, self.expr, 'numpy'),
                                  otypes=[np.float32])
    
    def get_initial_state(self, xs):
        return self.ic_fn(xs)
        
    def pprint(self):
        display(sp.Eq(self.u(x, 0), self.expr))

class Source:
    def __init__(self, expr):
        self.expr = expr
        self.source_fn = sp.lambdify(x, self.expr, 'numpy')
    
    def apply(self, xs):
        return self.source_fn(xs)
    
    def get_expr(self):
        return self.expr
        

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
                 source=0,
                 fd=backwardeuler):
        self.kappa = kappa   # Diffusion constant
        self.L = L           # Length of interval
        self.ic = IC(ic)     # Initial condition u(x,0) = h(x)
        self.lbc = lbc       # Left boundary condition as above
        self.rbc = rbc       # Right boundary condition as above
        self.source = Source(source)  # Source function f(x)
 
    def pprint(self, title=''):
        """Print the diffusion problem with latex"""
        print(title)
        u = sp.Function('u')
        x, t = sp.symbols('x t')
        display(sp.Eq(u(x,t).diff(t),
                      kappa*u(x,t).diff(x,2) + self.source.get_expr()))
        self.lbc.pprint()
        self.rbc.pprint()
        self.ic.pprint()
    
    def solve_to(self, T, mx, mt, scheme, full_output=False):
        """Solve the diffusion problem forward to time T using the given
        scheme."""
        xs = np.linspace(0, self.L, mx+1)     # mesh points in space
        ts = np.linspace(0, T, mt+1)     # mesh points in time
        deltax = xs[1] - xs[0]            # gridspacing in x
        deltat = ts[1] - ts[0]            # gridspacing in t
        lmbda = self.kappa*deltat/(deltax**2)    # mesh fourier number
    
        if full_output:
            print("deltax =",deltax)
            print("deltat =",deltat)
            print("lambda =",lmbda)
    
        u0 = self.ic.get_initial_state(xs)
        
        uT = scheme(T, self.L, mx, mt, lmbda, u0,
                    self.lbc, self.rbc, self.source)
        
        return xs, uT
        
    def solve_diffusion_problem(self, t_range, mx, mt, scheme):
        """Solve the diffusion problem for a range of times"""
        pass
        
    
    def plot_at_T(self,
                  T,
                  mx=20,
                  mt=1000,
                  scheme=forwardeuler,
                  u_exact=None,
                  title=''):
        """Plot the solution to the diffusion problem at time T.
        If the exact solution is known, plot that too and return the 
        error at time T."""
        xs, uT = self.solve_to(T, mx, mt, scheme)
        try:
            pl.plot(xs,uT,'ro',label='numerical')
        except:
            pass
            
        if u_exact:
            xx = np.linspace(0, self.L, 250)
            uTsym = u_exact.subs({kappa: self.kappa,
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
    
    def animate(self, t_range, mx, mt):
        """Animate the solution of the diffusion problem at the given
        time frames"""
        pass
    
    def error_at_T(self, T, mx, mt, u_exact, scheme=backwardeuler):
        """Return the error (L2 norm) between the solution of the 
        equation at T and the exact solution"""
        xs, uT = self.solve_to(T, mx, mt, scheme)
        
        uTsym = u_exact.subs({kappa: self.kappa,
                                  L: self.L,
                                  t: T})
        u = sp.lambdify(x, uTsym)
        error = np.linalg.norm(abs(u(xs) - uT))
        return error