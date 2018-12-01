# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 21:34:47 2018

@author: james
"""

from diffusionproblem import DiffusionProblem, Dirichlet, Neumann, IC, BC
from sympy import *
from sympy.abc import x, t, L, kappa
init_printing()

def example1():
    # Example 1 (These options are the defaults)
    # 
    # BCs
    # u(0,t) = 0
    # u(1.t) = 0
    #
    # IC
    # u(x,0) = sin(pi*x)
    
    dp1 = DiffusionProblem()
    dp1.pprint('Diffusion Example 1')
    
    # exact solution
    u = exp(-kappa*(pi**2/L**2)*t)*sin(pi*x/L)
    dp1.plot_at_T(0.1, u_exact=u, title='Example 1')

def example2():
    # Example 2 (another frequency)
      
    dp2 = DiffusionProblem(ic=IC(sin(pi*x) + 0.5*sin(3*pi*x)))
    dp2.pprint('Diffusion Example 2')
    
    # exact solution
    u = exp(-kappa*(pi**2/L**2)*t)*sin(pi*x/L) + \
                0.5*exp(-kappa*9*(pi**2/L**2)*t)*sin(3*pi*x/L) 
    dp2.plot_at_T(0.01, u_exact=u, title='Example 2')

def example3():
    # Example 3 (new boundary condition)
    # u(1,t) = 1
    
    dp3 = DiffusionProblem(rbc=Dirichlet(1,1))
    dp3.pprint('Diffusion Example 3')
    dp3.plot_at_T(0.03, title='Example 3')

def example4():
    # Example 4 (Initial condition)
    
    dp4 = DiffusionProblem(ic=IC(x))
    dp4.pprint('Diffusion Problem 4')
    
    u = (2/pi)*exp(-pi**2*t)*sin(pi*x)
    dp4.plot_at_T(0.01, u_exact=u)

def example5():
    # Example 5 (Neumann boundary condition)

    #dp5 = DiffusionProblem(rbc=Neumann(1, 0))
    #dp5.pprint() 
    bc = BC(0,(0,1,sin(x)),'left')
    bc.pprint()
    
#example1()
#example2()
#example3()
#example4()
example5()