# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 11:46:18 2019

@author: james
"""
import numpy as np
from sympy import *
from sympy.abc import x, t

from parabolicpde import backwardeuler

from pdeproblem import HyperbolicProblem
from parabolicsolvers import forwardeuler, backwardeuler, cranknicholson
from hyperbolicsolvers import explicitsolve, implicitsolve
from visualizations import plot_solution

## Worksheet 1 ##

## Q2 (a) ##

def solver_demonstration():
    mx = 30; mt = 1000; L = 1; T = 0.5
    
    def uI(x):
        return np.sin(np.pi*x/L)
    
    def u(x):
        return np.exp(-(np.pi**2)*T)*np.sin(np.pi*x)
      
    xs, uT = backwardeuler(mx, mt, L, T,
                           1, 0,
                           uI, 0, 0, 'Dirichlet', 'Dirichlet')

    plot_solution(xs, uT, uexact=u)
   
## Q3 ##
def error_analysis():
    pass

## Worksheet 2 ##
    
def W2q1():
    mx = 25; mt = 50; T = 0.51
    hp = HyperbolicProblem()
    
    u= cos(pi*t)*sin(pi*x)

    uT, error = hp.solve_at_T(T, mx, mt, implicitsolve, u_exact=u)
    print('Error = {}'.format(error))
    
W1q2a()
W2q1()

def tsunami():
    # travelling wave with variable seabed
    L=50
    h0=8
    wave = 20*exp(-(x-5)**2/3)
    seabed = 7*exp(-(x-25)**2/50)
    #seabed = 0
    
    wp17 = hp.TsunamiProblem(L, h0, wave, seabed)
    
    # T = 40 makes a good animation
    uT = wp17.solve_at_T(8.5, 250, 600, plot=True, animate=False)