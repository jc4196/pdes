# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 11:55:01 2018

@author: james
"""

import numpy as np
import sympy as sp
from sympy.abc import kappa, L, x, t
from scipy.sparse.linalg import spsolve

from IPython.display import display
import matplotlib.pylab as pl

from schemes import backwardeuler, cranknicholson, forwardeuler
from discretesolvepde import solve_diffusion_pde, plot_solution

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
    
    def solve_to(self, T, mx, mt, scheme):
        xs = np.linspace(0, self.L, mx+1)     # mesh points in space
        ts = np.linspace(0, T, mt+1)     # mesh points in time
        deltax = xs[1] - xs[0]            # gridspacing in x
        deltat = ts[1] - ts[0]            # gridspacing in t
        lmbda = self.kappa*deltat/(deltax**2)    # mesh fourier number

        u_j = self.ic(xs)
        u_jp1 = np.zeros(xs.size)
        
        # Get matrices and vector for the particular scheme
        A, B, v = scheme(mx,
                         deltax,
                         deltat,
                         lmbda,
                         self.lbc.apply_rhs,
                         self.rbc.apply_rhs)
        
        a, b = self.boundarytype(mx)
        
        for n in range(1, mt+1):
            # Solve matrix equation A*u_{j+1} = B*u_j + v
            u_jp1[a:b] = spsolve(A[a:b,a:b],
                                 B[a:b,a:b].dot(u_j[a:b]) + v(n*deltat)[a:b])
            
            # add source to inner terms
            u_jp1[1:-1] += deltat*self.source(xs[1:-1], n*deltat)
            
            # fix Dirichlet boundary conditions
            if a == 1:
                u_jp1[0] = self.lbc.apply_rhs(n*deltat)
            if b == mx:
                u_jp1[-1] = self.rbc.apply_rhs(n*deltat)
            
            u_j[:] = u_jp1[:]
        
        return xs, u_j
    
    def solve_at_T(self, T, mx, mt, scheme):
        return solve_diffusion_pde(mx, mt, self.L, T,
                                   self.kappa, self.source, self.ic,
                                   self.lbc.apply_rhs, self.rbc.apply_rhs,
                                   self.boundarytype(mx),
                                   scheme)

    def solve_diffusion_problem(self, t_range, mx, mt, scheme):
        """Solve the diffusion problem for a range of times"""
        pass
        
    
    def plot_at_T(self,
                  T,
                  mx=100,
                  mt=1000,
                  scheme=backwardeuler,
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
    
    def plot_at_T(self,
                  T,
                  mx=100,
                  mt=1000,
                  scheme=backwardeuler,
                  u_exact=None,
                  title=''):
        
        xs, uT = self.solve_at_T(T, mx, mt, scheme)
        if u_exact:
            uTsym = u_exact.subs({kappa: self.kappa,
                                  L: self.L,
                                  t: T})
            u = sp.lambdify(x, uTsym)
            plot_solution(xs, uT, u, title, r'${}$'.format(sp.latex(uTsym)))
        else:
            plot_solution(xs, uT)

    def animate(self, t_range, mx, mt):
        """Animate the solution of the diffusion problem at the given
        time frames"""
        pass
    
    def error_at_T(self, T, mx, mt, u_exact, scheme):
        """Return the error (L2 norm) between the solution of the 
        equation at T and the exact solution"""
        xs, uT = self.solve_to(T, mx, mt, scheme)
        
        uTsym = u_exact.subs({kappa: self.kappa,
                                  L: self.L,
                                  t: T})
        u = sp.lambdify(x, uTsym)
        error = np.linalg.norm(abs(u(xs) - uT))
        return error