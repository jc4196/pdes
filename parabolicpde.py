# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 17:53:58 2019

@author: james
"""
import numpy as np
from scipy.sparse.linalg import spsolve

import sympy as sp
from sympy.abc import x, t, kappa, L

from IPython.display import display

from boundary import Dirichlet
from helpers import tridiag, numpify, numpify_many, get_error
from visualizations import plot_solution


class ParabolicProblem:
    """Object specifying a diffusion type problem of the form
    
    du/dt = kappa*d^2u/dx^2 + f(x)
    """
    def __init__(self,
                 kappa=1,
                 L=1,
                 ic=sp.sin(sp.pi*x),
                 lbc=Dirichlet(0,0),
                 rbc=Dirichlet(1, 0),
                 source=0):
        self.kappa = kappa    # Diffusion coefficient
        self.L = L            # Length of interval
        self.lbc = lbc        # Left boundary condition object
        self.rbc = rbc        # Right boundary condition object
        self.ic = ic          # Initial condition function h(x)
        self.source = source  # Source function f(x)
 
 
    def pprint(self, title=''):
        """Print the diffusion problem with LaTeX"""
        print(title)
        u = sp.Function('u')
        
        # show the diffusion equation
        display(sp.Eq(u(x,t).diff(t),
                      self.kappa*u(x,t).diff(x,2) + self.source))
        # show the boundary conditions
        self.lbc.pprint()
        self.rbc.pprint()
        
        # show the initial condition
        display(sp.Eq(u(x,0), self.ic))
    
    def solve_at_T(self, T, mx, mt, scheme,
                   u_exact=None, plot=True, norm='L2', title=''):
        """
        Solve the diffusion equation at time T with grid spacing mx x mt.
        
        Parameters
        T        time to stop the integration
        mx       number of grid points in space
        mt       number of grid points in time
        scheme   the solver to use e.g. forwardeuler, cranknicholson
        u_exact  the exact solution (sympy expression)
        plot     plot the results if True
        title    title for the plot
        
        Returns
        uT       the solution at time T
        err      the absolute error (if u_exact is given)
        """
        
        # solve the PDE by the given scheme
        xs, uT, lmbda =  (SCHEMES[scheme])(mx, mt, self.L, T,
                                           self.kappa, self.source, self.ic,
                                           self.lbc.rhs, self.rbc.rhs,
                                           self.lbc.type, self.rbc.type)
        
        if u_exact:
            # substitute in the values of kappa, L and T
            uTsym = u_exact.subs({kappa: self.kappa,
                                  L: self.L,
                                  t: T})
            # use L-inf norm to calculate absolute error
            error = get_error(xs, uT, uTsym, norm=norm)            
            if plot:       
                plot_solution(xs, uT, uTsym, title=title,
                              uexacttitle=r'${}$'.format(sp.latex(uTsym)))            
        else:
            error = None
            if plot:
                plot_solution(xs, uT, title=title)            
        
        return uT, error, lmbda
    
## Schemes ##
        
def forwardeuler(mx, mt, L, T, 
                 kappa, source,
                 ic, lbc, rbc, lbctype, rbctype):
    """Forward euler finite-difference scheme (explicit) for solving
    parabolic PDE problems. Values of lambda > 1/2 will cause the scheme
    to become unstable"""
    
    # initialise     
    xs, ts, deltax, deltat, lmbda = initialise(mx, mt, L, T, kappa)

    # make sure these functions are vectorized
    ic, lbc, rbc, source = numpify_many((ic, 'x'), (lbc, 't'),
                                        (rbc, 't'), (source, 'x t'))
    
    # Construct forward Euler matrix
    A_FE = tridiag(mx+1, lmbda, 1-2*lmbda, lmbda)
    
    # modify first and last row for Neumann conditions
    A_FE[0,1] *= 2; A_FE[mx,mx-1] *= 2

    # initialise first time steps
    u_j = ic(xs).copy()
    u_jp1 = np.zeros(xs.size)


    # range of rows of A_FE to use
    a = 1 if lbctype == 'Dirichlet' else 0
    b = mx if rbctype == 'Dirichlet' else mx+1

    # Solve the PDE at each time step
    for j in ts[:-1]:
        u_jp1[a:b] = A_FE[a:b,a:b].dot(u_j[a:b])
        
        addboundaries(u_jp1, lbctype, rbctype,
                      lmbda*lbc(j),
                      lmbda*rbc(j),
                      -2*lmbda*deltax*lbc(j),
                      2*lmbda*deltax*rbc(j))

        # fix Dirichlet boundary conditions
        if lbctype == 'Dirichlet':
            u_jp1[0] = lbc(j + deltat)
        if rbctype == 'Dirichlet':
            u_jp1[mx] = rbc(j + deltat)
        
        # add source to inner terms
        u_jp1[1:-1] += deltat*source(xs[1:-1], j)
        
        u_j[:] = u_jp1[:]
    
    return xs, u_j, lmbda

      
def backwardeuler(mx, mt, L, T, 
                  kappa, source,
                  ic, lbc, rbc, lbctype, rbctype):
    """Backward Euler finite-difference scheme (implicit) for solving 
    parabolic PDE problems. Unconditionally stable"""

    # initialise   parameters   
    xs, ts, deltax, deltat, lmbda = initialise(mx, mt, L, T, kappa)
 
    # make sure these functions are vectorized
    ic, lbc, rbc, source = numpify_many((ic, 'x'), (lbc, 't'),
                                        (rbc, 't'), (source, 'x t'))

    # Construct backward Euler matrix
    B_FE = tridiag(mx+1, -lmbda, 1+2*lmbda, -lmbda)

    # modify first and last row for Neumann conditions
    B_FE[0,1] *= 2; B_FE[mx,mx-1] *= 2
    
    # restrict range of B_FE to use in case of Dirichlet conditions
    a = 1 if lbctype == 'Dirichlet' else 0
    b = mx if rbctype == 'Dirichlet' else mx+1

    # initialise first time steps
    u_j = ic(xs).copy()
    u_jp1 = np.zeros(xs.size)

    # Solve the PDE at each time step
    for j in ts[:-1]:
        # modifications to the boundaries
        addboundaries(u_j, lbctype, rbctype,
                      lmbda*lbc(j + deltat),    # Dirichlet u[1]
                      lmbda*rbc(j + deltat),    # Dirichlet u[-2]
                      -2*lmbda*deltax*lbc(j + deltat),  # Neumann u[0]
                      2*lmbda*deltax*rbc(j + deltat))   # Neumann u[-1]

        u_jp1[a:b] = spsolve(B_FE[a:b,a:b], u_j[a:b])
        
        # fix Dirichlet boundary conditions
        if lbctype == 'Dirichlet':
            u_jp1[0] = lbc(j + deltat)
        if rbctype == 'Dirichlet':
            u_jp1[mx] = rbc(j + deltat)
        
        # add source to inner terms
        u_jp1[1:-1] += deltat*source(xs[1:-1], j + deltat)
        
        u_j[:] = u_jp1[:]
        
    return xs, u_j, lmbda


def backwardeuler2(mx, mt, L, T, 
                  kappa, source,
                  ic, lbc, rbc, lbc_ab, rbc_ab):
    """
    Second implementation of backward Euler that takes mixed boundary
    conditions
    """
    
    # initialise   parameters   
    xs, ts, deltax, deltat, lmbda = initialise(mx, mt, L, T, kappa)
 
    # make sure these functions are vectorized
    ic, lbc, rbc, source = numpify_many((ic, 'x'), (lbc, 't'),
                                        (rbc, 't'), (source, 'x t'))
    
    # Parameters needed to construct the matrix
    alpha1, beta1 = lbc_ab
    alpha2, beta2 = rbc_ab
    
    # Construct the backward Euler matrix, first and last row are 
    # boundary condition equations
    lower = (mx-1)*[-lmbda] + [-beta2]
    main = [alpha1*deltax - beta1] + (mx-1)*[1+2*lmbda] + \
                [beta2 + alpha2*deltax]
    upper = [beta1] + (mx-1)*[-lmbda]
    A_BE = tridiag(mx+1, lower, main, upper)

    # initialise the first time steps
    u_j = ic(xs).copy()
    u_jp1 = np.zeros(xs.size)          
    
    # Solve the PDE: loop over all time points
    for j in ts[:-1]:  
        # Add boundary conditions to vector u_j
        u_j[0] = deltax*lbc(j*deltat)            
        u_j[mx] = deltax*rbc(j*deltat)
        
        # Backward euler timestep
        u_jp1 = spsolve(A_BE, u_j)
 
        # add source function
        u_jp1[1:mx] += deltat*source(xs[1:-1], j*deltat)
        
        # Update u_j
        u_j[:] = u_jp1[:]
         
    return xs, u_j, lmbda 


def cranknicholson(mx, mt, L, T, 
                   kappa, source,
                   ic, lbc, rbc, lbctype, rbctype):
    """Crank-Nicholson finite-difference scheme (implicit) for solving 
    parabolic PDE problems. Unconditionally stable"""
    
    # initialise   parameters   
    xs, ts, deltax, deltat, lmbda = initialise(mx, mt, L, T, kappa)
 
    # make sure these functions are vectorized
    ic, lbc, rbc, source = numpify_many((ic, 'x'), (lbc, 't'),
                                        (rbc, 't'), (source, 'x t'))
    
    # Construct Crank-Nicholson matrices
    A_CN = tridiag(mx+1, -0.5*lmbda, 1+lmbda, -0.5*lmbda)
    B_CN = tridiag(mx+1, 0.5*lmbda, 1-lmbda, 0.5*lmbda)
    # modify first and last row for Neumann conditions
    A_CN[0,1] *= 2; A_CN[mx,mx-1] *= 2; B_CN[0,1] *= 2; B_CN[mx,mx-1] *= 2

    # restrict range of A_CN and B_CN in case of Dirichlet conditions
    a = 1 if lbctype == 'Dirichlet' else 0
    b = mx if rbctype == 'Dirichlet' else mx+1

    # initialise first time steps
    u_j = ic(xs).copy()
    u_jp1 = np.zeros(xs.size)
    v = np.zeros(xs.size)  
    
    # Solve the PDE at each time step
    for j in ts[:-1]:
        v[a:b] = B_CN[a:b,a:b].dot(u_j[a:b])
        
        addboundaries(v, lbctype, rbctype,
                      0.5*lmbda*(lbc(j) + lbc(j + deltat)),
                      0.5*lmbda*(rbc(j) + rbc(j + deltat)),
                      -lmbda*deltax*(lbc(j) + lbc(j + deltat)),
                      lmbda*deltax*(rbc(j) + rbc(j + deltat)))

        u_jp1[a:b] = spsolve(A_CN[a:b,a:b], v[a:b])
        
        # fix Dirichlet boundary conditions
        if lbctype == 'Dirichlet':
            u_jp1[0] = lbc(j)
        if rbctype == 'Dirichlet':
            u_jp1[mx] = rbc(j)
        
        # add source to inner terms
        u_jp1[1:-1] += 0.5*deltat*(
                source(xs[1:-1], j) + source(xs[1:-1], j + deltat))
        
        u_j[:] = u_jp1[:]
    
    return xs, u_j, lmbda

# key for accessing schemes
SCHEMES = {'FE': forwardeuler,
           'BE': backwardeuler,
           'BE2': backwardeuler2,  # takes mixed boundary conditions
           'CN': cranknicholson}

## Extra functions ##
    
def initialise(mx, mt, L, T, kappa):
    """initialise parameters that will be used for the solving the PDE"""
    xs = np.linspace(0, L, mx+1)          # mesh points in space
    ts = np.linspace(0, T, mt+1)          # mesh points in time
    deltax = xs[1] - xs[0]                # gridspacing in x
    deltat = ts[1] - ts[0]                # gridspacing in t
    lmbda = kappa*deltat/(deltax**2)      # mesh fourier number   
    
    return xs, ts, deltax, deltat, lmbda

def addboundaries(u, lbctype, rbctype, D1, Dmxm1, N0, Nmx):
    """Modifications to the boundary rows before/after matrix multiplication"""
    if lbctype == 'Neumann':
        u[0] += N0
    elif lbctype == 'Dirichlet':
        u[1] += D1
    else:
        raise Exception('That boundary condition is not implemented')
    
    if rbctype == 'Neumann':
        u[-1] += Nmx
    elif lbctype == 'Dirichlet':
        u[-2] += Dmxm1
    else:
        raise Exception('That boundary condition is not implemented')