# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 13:06:32 2019

@author: james
"""
import sympy as sp
from sympy.abc import x, y
import numpy as np

from boundary import Dirichlet
from helpers import numpify_many

from IPython.display import display

import matplotlib.pyplot as plt

class EllipticProblem:
    """Object specifying a diffusion type problem of the form
    
    du/dt = kappa*d^2u/dx^2 + f(x)
    """
    def __init__(self,
                 Lx=2.0,
                 Ly=1.0,
                 lbc=0,
                 rbc=0,
                 tbc=sp.sin(sp.pi*x/2),
                 bbc=0):
        self.Lx = Lx          # Length of interval in x direction
        self.Ly = Ly          # Length of interval in y direction
        self.lbc = lbc        # Left boundary condition
        self.rbc = rbc        # Right boundary condition
        self.tbc = tbc        # Top boundary condition
        self.bbc = bbc        # Bottom boundary condition
     
    def pprint(self, title=''):
        print(title)
        u = sp.Function('u')

        display(sp.Eq(u(x,y).diff(x,2) + u(x,y).diff(y, 2), 0))
        display(sp.Eq(u(0,y), self.lbc))
        display(sp.Eq(u(x,y).subs(x,self.Lx), self.rbc))
        display(sp.Eq(u(x,0), self.bbc))
        display(sp.Eq(u(x,y).subs(y,self.Ly), self.tbc))

    def solve(self, mx, my, maxcount, maxerr, omega, u_exact= None, plot=True):
        """Solve Laplace's equation on a rectangle with Dirichlet 
        boundary conditions"""
        return SOR(mx, my, self.Lx, self.Ly,
                   self.lbc, self.rbc, self.tbc,
                   self.bbc, maxcount, maxerr,
                   omega, u_exact=u_exact, plot=plot)

        
        
def SOR(mx, my, Lx, Ly,
        lbc, rbc, tbc, bbc,
        maxcount, maxerr, omega, u_exact, plot=True):
    """Solve Laplace's equation on a rectangle with Dirichlet boundary
    conditions, using the SOR method"""
    # set up the numerical environment variables
    xs = np.linspace(0, Lx, mx+1)     # mesh points in x
    ys = np.linspace(0, Ly, my+1)     # mesh points in y
    deltax = xs[1] - xs[0]             # gridspacing in x
    deltay = ys[1] - ys[0]             # gridspacing in y
    lambdasqr = (deltax/deltay)**2       # mesh number
    
    u_old = np.zeros((xs.size,ys.size))   # u at current time step
    u_new = np.zeros((xs.size,ys.size))   # u at next time step
    u_true = np.zeros((xs.size,ys.size))  # exact solution
    R = np.zeros((xs.size, ys.size))     # residual (goes to 0 with convergence)
    
    bbc, tbc, lbc, rbc, u_exact = numpify_many((bbc, 'x'), (tbc, 'x'),
                                               (lbc, 'x'), (rbc, 'x'),
                                               (u_exact, 'x y'))
    
    # intialise the boundary conditions, for both timesteps
    u_old[1:-1,0] = bbc(xs[1:-1])
    u_old[1:-1,-1] = tbc(xs[1:-1])
    u_old[0,1:-1] = lbc(ys[1:-1])
    u_old[-1,1:-1] = rbc(ys[1:-1])
    u_new[:]=u_old[:]
    
    if u_exact:
        # true solution values on the grid 
        for i in range(0,mx+1):
            for j in range(0,my+1):
                u_true[i,j] = u_exact(xs[i],ys[j])
            

    count = 1
    err = maxerr+1
    
    while err>maxerr and count<maxcount:
        for j in range(1,my):
            for i in range(1,mx):
                # calculate residual
                R[i,j] = (u_new[i-1,j] + u_old[i+1,j] - \
                     2*(1+lambdasqr)*u_old[i,j] + \
                     lambdasqr*(u_new[i,j-1]+u_old[i,j+1]) )/(2*(1+lambdasqr))
                # SOR step
                u_new[i,j] = u_old[i,j] + omega*R[i,j]
        
        err = np.max(np.abs(u_new-u_old))
        u_old[:] = u_new[:]
        count=count+1
    
    if u_exact:
        # calculate the error, compared to the true solution    
        err_true = np.max(np.abs(u_new[1:-1,1:-1]-u_true[1:-1,1:-1]))
    else:
        err_true = 0
    
    if plot:
        # plot the resulting solution
        xx = np.append(xs,xs[-1]+deltax)-0.5*deltax  # cell corners needed for pcolormesh
        yy = np.append(ys,ys[-1]+deltay)-0.5*deltay
        plt.pcolormesh(xx,yy,u_new.T)
        cb = plt.colorbar()
        cb.set_label('u(x,y)')
        plt.xlabel('x'); plt.ylabel('y')
        plt.show()
        
    return u_new, count, err, err_true
    
    