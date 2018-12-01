# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 21:34:47 2018

@author: james
"""

from diffusionproblem import DiffusionProblem, Dirichlet, IC
from sympy import *
from sympy.abc import x, t, L, kappa
init_printing()

# Example 1 (These options are the defaults)
# 
# BCs
# u(0,t) = 0
# u(1.t) = 0
#
# IC
# u(x,0) = sin(pi*x)
#
# Exact solution
# u(x,t) = e^{-kappa*(pi^2/L^2)*t}*sin(pi*x/L)

u = exp(-kappa*(pi**2/L**2)*t)*sin(pi*x/L)
#u = exp(-kappa*(pi**2/L**2)*t)*sin(pi*x/L)

dp1 = DiffusionProblem()
dp1.pprint()
dp1.plot_at_T(0.1, u_exact=u, title='Example 1')

# Example 2 (another frequency)

kappa = 1
L = 1

x = symbols('x')
ic = IC(sin(pi*x) + 0.5*sin(3*pi*x))
u = exp(-kappa*(pi**2/L**2)*t)*sin(pi*x/L) + \
            0.5*exp(-kappa*9*(pi**2/L**2)*t)*sin(3*pi*x/L) 
            
dp2 = DiffusionProblem(ic=ic)
dp2.plot_at_T(0.01, u_exact=u, title='Example 2')

# Example 3 (new boundary condition)
# u(1,t) = 1

dp3 = DiffusionProblem(rbc=Dirichlet(1,1))
dp3.plot_at_T(0.03, title='Example 3')