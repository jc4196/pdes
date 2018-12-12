# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 21:34:47 2018

@author: james
"""

from diffusionproblem import DiffusionProblem, WaveProblem, Dirichlet, Neumann
from schemes import *
from discretesolvepde import error

from sympy import *
from sympy.abc import x, t, L, kappa
import numpy as np

init_printing()

mx = 50
mt = 1000
scheme = cranknicholson

def example1():
    # Example 1 (These options are the defaults)
    dp1 = DiffusionProblem()
    dp1.pprint('Diffusion Example 1')
    
    # exact solution
    u = exp(-kappa*(pi**2/L**2)*t)*sin(pi*x/L)
    uT, err = dp1.solve_at_T(0.5, mx, mt, scheme, u_exact=u,plot=False, title='Example 1')
    print('{:.5f}'.format(err))
    # test of forward euler scheme
    # deltax = 0.1 -> mx = 10
    # deltat 
    #mt = 5*np.logspace(2, 5, 4, dtype=np.int32)
    #print(mt)
    #errors = [dp1.error_at_T(0.5, 10, n, u, scheme=forwardeuler) for n in mt]
    #print(errors)
    
def example2():
    # Example 2 (another frequency in the initial condition)
      
    dp2 = DiffusionProblem(ic=sin(pi*x) + 0.5*sin(3*pi*x))
    dp2.pprint('Diffusion Example 2')
    
    # exact solution
    u = exp(-kappa*(pi**2/L**2)*t)*sin(pi*x/L) + \
                0.5*exp(-kappa*9*(pi**2/L**2)*t)*sin(3*pi*x/L) 
    dp2.solve_at_T(0.6, mx, mt, scheme, u_exact=u, title='Example 2')

def example3():
    # Example 3 (new boundary condition)
    # u(1,t) = 1

    dp3 = DiffusionProblem(rbc= Dirichlet(1,1))
    dp3.pprint('Diffusion Example 3')
    dp3.solve_at_T(0.02, mx, mt, scheme, title='Example 3')
    
    
def example4():
    # Example 4 (Initial condition)
    
    dp4 = DiffusionProblem(ic=x)
    dp4.pprint('Diffusion Problem 4')
    
    u_first = (2/pi)*exp(-pi**2*t)*sin(pi*x)
    dp4.solve_at_T(0.3, mx, mt, scheme, u_exact=u_first, title='Example 4')

def example5():
    # Example 5 (Neumann boundary condition)

    dp5 = DiffusionProblem(lbc=Neumann(0,0), rbc=Neumann(1,0), ic=x)
    dp5.pprint('Diffusion Problem 5')
     
    u_first = 0.5 - (4/pi**2)*exp(-pi**2*t)*cos(pi*x)
    dp5.solve_at_T(1, mx, mt, scheme, u_exact=u_first, title='Example 5')

def example6():
    # Example 6 (constant source)
    dp6 = DiffusionProblem(source=1, rbc=Dirichlet(1,1), ic=0)
    dp6.pprint('Diffusion Problem 6')
    
    # steady state
    ss = -0.5*x**2 + 1.5*x
    dp6.solve_at_T(0.5, mx, mt, scheme, u_exact=ss)
    
def example7():
    # Example 6 (source variable in x)
    
    dp7 = DiffusionProblem(source=sin(3*pi*x), ic=sin(pi*x))
    dp7.pprint('Diffusion Problem 7')
    
    u = exp(-(pi**2)*t)*sin(pi*x) + \
            (1/(3*pi)**2)*(1 - exp(-9*pi**2*t))*sin(3*pi*x)
            
    dp7.solve_at_T(0.4, mx, mt, scheme, u_exact = u)

def example8():
    dp8 = DiffusionProblem(ic=4*sin(3*pi*x))
    dp8.pprint('Diffusion Problem 8')
    
    u = 4*sin(3*pi*x)*exp(-(3*pi)**2*t)
    dp8.solve_at_T(0.1, mx, mt, scheme, u_exact=u)

def example9():
    dp9 = WaveProblem()
    #dp9.pprint()
    dp9.solve_at_T(2, 50, 100, explicitwave, title='Wave Example 1')

example1()    
#example2()
#example3()
#example4()
#example5()
#example6()
#example7()
#example8()
example9()