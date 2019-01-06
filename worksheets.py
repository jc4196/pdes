# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 11:46:18 2019

@author: james
"""
import numpy as np
from sympy import *
from sympy.abc import x, t
from pdeproblem import HyperbolicProblem
from parabolicsolvers import forwardeuler, backwardeuler, cranknicholson
from hyperbolicsolvers import explicitsolve, implicitsolve
from visualizations import plot_solution

## Worksheet 1 ##

def W1q2a():
    mx = 30; mt = 1000; L = 1; T = 0.5
    
    def uI(x):
        return np.sin(np.pi*x/L)
    
    def u(x):
        return np.exp(-(np.pi**2)*T)*np.sin(np.pi*x)
      
    xs, uT = backwardeuler(mx, mt, L, T,
                           1, 0,
                           uI, 0, 0, 'Dirichlet', 'Dirichlet')

    plot_solution(xs, uT, uexact=u)
    
def W1q3():
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