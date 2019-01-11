# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 17:52:20 2019

@author: james
"""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

import sympy as sp
from sympy.abc import x, t, L, c

from IPython.display import display

from helpers import tridiag, numpify_many, numpify, get_error
from boundary import Dirichlet
from visualizations import plot_solution, animate_tsunami


## Objects ##

class HyperbolicProblem:
    """Object specifying a wave equation type problem of the form
    d^2u/dt^2 = c^2 d^2u/dx^2 + f(x)
    """
    def __init__(self,
                 c=1,
                 L=1,
                 ix=sp.sin(sp.pi*x),
                 iv=0,
                 lbc=Dirichlet(0,0),
                 rbc=Dirichlet(1, 0),
                 source=0):
    
        self.c = c           # Wave speed
        self.L = L           # Length of interval
        self.lbc = lbc       # Left boundary condition as above
        self.rbc = rbc       # Right boundary condition as above
        self.ix = ix         # Initial displacement
        self.iv = iv         # Initial velocity
        self.source = source # source function f(x)

        
    def pprint(self, title=''):
        """Print the diffusion problem with LaTeX"""
        print(title)
        u = sp.Function('u')
        x, t = sp.symbols('x t')
        display(sp.Eq(u(x,t).diff(t, 2),
                      self.c**2*u(x,t).diff(x,2) + self.source))
        self.lbc.pprint()
        self.rbc.pprint()
        display(sp.Eq(u(x,0), self.ix))
        display(sp.Eq(u(x,t).diff(x).subs(t,0), self.iv))
        
    
    def solve_at_T(self, T, mx, mt, scheme,
                   plot=True, u_exact=None, norm='L2', title=''):
        """
        Solve the wave equation at time T with grid spacing mx x mt.
        
        Parameters
        T        time to stop the integration
        mx       number of grid points in space
        mt       number of grid points in time
        scheme   the solver to use e.g. explicitsolve, implicitsolve
        u_exact  the exact solution (sympy expression)
        plot     plot the results if True
        title    title for the plot
        
        Returns
        uT       the solution at time T
        err      the absolute error (if u_exact is given)
        """
        # solve the PDE at time T
        xs, uT, lmbda = (SCHEMES[scheme])(mx, mt, self.L, T,
                                   self.c, self.source,
                                   self.ix, self.iv,
                                   self.lbc.rhs, self.rbc.rhs,
                                   self.lbc.type, self.rbc.type)
        if u_exact:
            # substitute in the values of c, L and t
            uTsym = u_exact.subs({c: self.c,
                                  L: self.L,
                                  t: T})
            # calculate absolute error
            error = get_error(xs, uT, uTsym, norm=norm)
            if plot:
                plot_solution(xs, uT, uTsym, title=title,
                              uexacttitle=r'${}$'.format(sp.latex(uTsym)))            
        else:
            error = None
            if plot:
                plot_solution(xs, uT, title=title)            
        
        return uT, error, lmbda      

class TsunamiProblem:
    def __init__(self, L, h0, wave, seabed):
        self.L = L
        self.h0 = h0
        self.seabed = seabed - h0
        self.h = h0 - seabed
        self.wave = wave
   
    def solve_at_T(self, T, mx, mt, plot=False, animate=False):
        xs, u = tsunami_solve(mx, mt, self.L, T, self.h0, self.h, self.wave)
        
        if plot:
            plot_solution(xs, u[-1], uexact=self.seabed , style='r-')
        if animate:
            animate_tsunami(xs, u, self.L)
        
        return u[-1]

## Solvers ##

def explicitsolve(mx, mt, L, T,
                  c, source,
                  ix, iv,
                  lbc, rbc, lbctype, rbctype):
    """
    Solve a wave equation using an explicit scheme. Conditionally
    stable for lambda <= 1. lambda = 1 is the 'magic step'.
    """
    xs, ts, deltax, deltat, lmbda = initialise(mx, mt, L, T, c)
    
    # make sure these functions are vectorized
    ix, iv, lbc, rbc, source = numpify_many((ix, 'x'), (iv, 'x'), (lbc, 't'),
                                            (rbc, 't'), (source, 'x t'))

    # Construct explicit wave matrix
    A_EW = tridiag(mx+1, lmbda**2, 2-2*lmbda**2, lmbda**2)    
    
    # Changes for Neumann boundary conditions
    A_EW[0,1] *= 2; A_EW[mx,mx-1] *= 2

    # range of rows of A_EW to use
    a = 1 if lbctype == 'Dirichlet' else 0
    b = mx if rbctype == 'Dirichlet' else mx+1

    # initial condition vectors
    U = ix(xs).copy()
    V = iv(xs).copy()

    # set first two time steps
    u_jm1 = U 
    u_j = 0.5*A_EW.dot(U) + deltat*V   
 
    # initialise u at next time step
    u_jp1 = np.zeros(xs.size)        
    
    for j in ts[1:-1]:
        u_jp1[a:b] = A_EW[a:b,a:b].dot(u_j[a:b]) - u_jm1[a:b]

        addboundaries(u_jp1, lbctype, rbctype,
                      lmbda**2*lbc(j - deltat),
                      lmbda**2*rbc(j - deltat),
                      -2*lmbda**2*deltax*lbc(j - deltat),
                      2*lmbda**2*deltax*rbc(j - deltat))

        
        # fix Dirichlet boundary conditions
        if lbctype == 'Dirichlet':
            u_jp1[0] = lbc(j + deltat)
        if rbctype == 'Dirichlet':
            u_jp1[mx] = rbc(j + deltat)
                    
        # add source to inner terms
        u_jp1[1:-1] += deltat*source(xs[1:-1], j)
        
        # update u_jm1 and u_j
        u_jm1[:], u_j[:] = u_j[:], u_jp1[:]
    
    return xs, u_j, lmbda

def implicitsolve(mx, mt, L, T,
                  c, source,
                  ix, iv,
                  lbc, rbc, lbctype, rbctype):
    """
    Solve a wave equation using an implicit scheme. Unconditionally stable.
    """
    xs, ts, deltax, deltat, lmbda = initialise(mx, mt, L, T, c)
    
    # make sure these functions are vectorized
    ix, iv, lbc, rbc, source = numpify_many((ix, 'x'), (iv, 'x'), (lbc, 't'),
                                            (rbc, 't'), (source, 'x t'))
    
    # construct matrices for implicit scheme
    A_IW = tridiag(mx+1, -0.5*lmbda**2, 1+lmbda**2, -0.5*lmbda**2)
    B_IW = tridiag(mx+1, 0.5*lmbda**2, -1-lmbda**2, 0.5*lmbda**2)
    
    # corrections for Neumann conditions
    A_IW[0,1] *= 2; B_IW[0,1] *= 2; A_IW[mx,mx-1] *= 2; B_IW[mx,mx-1] *= 2

    # range of rows of A_EW to use
    a = 1 if lbctype == 'Dirichlet' else 0
    b = mx if rbctype == 'Dirichlet' else mx+1

    # initial condition vectors
    U = ix(xs).copy()
    V = iv(xs).copy()
    
    # set first two time steps
    u_jm1 = U  
    u_j = spsolve(A_IW, U - deltat*B_IW.dot(V))

    # initialise u at next time step
    u_jp1 = np.zeros(xs.size)
    v = np.zeros(xs.size)        
    
    for j in ts[1:-1]:
        v[a:b] = B_IW[a:b,a:b].dot(u_jm1[a:b]) + 2*u_j[a:b]
 
        addboundaries(v, lbctype, rbctype,
                      0.5*lmbda**2*(lbc(j) + lbc(j + deltat)),
                      0.5*lmbda**2*(rbc(j) + rbc(j + deltat)),
                       -lmbda**2*deltax*(lbc(j) + lbc(j + deltat)),
                      lmbda**2*deltax*(rbc(j) + rbc(j + deltat)))

        u_jp1[a:b] = spsolve(A_IW[a:b,a:b], v[a:b])
        
        if lbctype == 'Dirichlet':
            u_jp1[0] = lbc(j + deltat)
        if rbctype == 'Dirichlet':
            u_jp1[mx] = rbc(j + deltat)
    
        # add source to inner terms
        u_jp1[1:-1] += deltat*source(xs[1:-1], j)
        
        # update u_jm1 and u_j
        u_jm1[:], u_j[:] = u_j[:], u_jp1[:]
    
    return xs, u_j, lmbda

def tsunami_solve(mx, mt, L, T, h0, h, wave):
    """Variable wavespeed problem assumes periodic boundary on the left and 
    an open boundary on the right"""
    xs, ts, deltax, deltat, lmbda = initialise(mx, mt, L, T, np.sqrt(h0))
    
    h, wave = numpify_many((h, 'x'), (wave,'x'))
    delta = deltat/deltax

    # construct explicit wave matrix for variable wave speed problem
    # with open boundary conditions initially
    lower = [delta**2*h(i - 0.5*deltax) for i in xs[1:-1]] + [2*lmbda**2]
    main = [2*(1 + lmbda - lmbda**2)] + \
            [2 - delta**2*(h(i + 0.5*deltax) + h(i - 0.5*deltax)) for i in xs[1:-1]] + \
             [2*(1 + lmbda - lmbda**2)]
    upper = [2*lmbda**2] + [delta**2*h(i + 0.5*deltax) for i in xs[1:-1]]

    A_EW = tridiag(mx+1, lower, main, upper)
    
     # initial condition vectors
    u = [np.zeros(xs.size) for i in range(mt+1)]
    
    U = wave(xs).copy()
    
    # set first two time steps
    u[0] = U 
    u[1] = 0.5*A_EW.dot(U)     
   
    # keep track of the first time the right side becomes non-zero
    zero_right = True
    
    for j in range(1, mt):
        u[j+1] = A_EW.dot(u[j]) - u[j-1]
    
        # correction for open boundary condition
        u[j+1][mx] /= (1+2*lmbda)
        
        # switch from open to periodic boundary
        if zero_right and u[j+1][mx] > 1e-5:
            zero_right = False
        
        # correction for open boundary conditon on the left before 
        # it becomes a periodic condition
        if zero_right:
            u[j+1][0] /= (1+2*lmbda)
        else:
            # periodic condition
            u[j+1][0] = u[j+1][mx]
    
    return xs, u    

# key for accessing schemes
SCHEMES = {'E': explicitsolve,
           'I': implicitsolve}

## Extra Functions ##

def addboundaries(u, lbctype, rbctype, D1, Dmxm1, N0, Nmx):
    """Add Neumann or Dirichlet boundary conditions"""
    if lbctype == 'Neumann':
        u[0] += N0
    elif lbctype == 'Dirichlet':
        u[1] += D1
    else:
        raise Exception('That boundary condition is not implemented')
    
    if rbctype == 'Neumann':
        u[-1] += Nmx
    elif rbctype == 'Dirichlet': 
        u[-2] += Dmxm1
    else:
        raise Exception('That boundary condition is not implemented')

def initialise(mx, mt, L, T, c):
    xs = np.linspace(0, L, mx+1)     # mesh points in space
    ts = np.linspace(0, T, mt+1)      # mesh points in time
    deltax = xs[1] - xs[0]            # gridspacing in x
    deltat = ts[1] - ts[0]            # gridspacing in t
    lmbda = c*deltat/deltax      # squared Courant number  
    
    return xs, ts, deltax, deltat, lmbda