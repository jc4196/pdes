# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 11:46:18 2019

@author: james
"""
import numpy as np
from sympy import *
from sympy.abc import x, t

from parabolicpde import ParabolicProblem, backwardeuler
from hyperbolicpde import HyperbolicProblem

from visualizations import plot_solution
import matplotlib.pyplot as plt

## Worksheet 1 ##

## Q2 (a) 

def solver_demonstration():
    mx = 30; mt = 1000; L = 1; T = 0.5
    
    def uI(x):
        return np.sin(np.pi*x/L)
    
    def u(x):
        return np.exp(-(np.pi**2)*T)*np.sin(np.pi*x)
      
    xs, uT, lmbda = backwardeuler(mx, mt, L, T,
                                  1, 0,
                                  uI, 0, 0, 'Dirichlet', 'Dirichlet')

    plot_solution(xs, uT, uexact=u)
   
## Q3 
    
def error_analysis():
    # comparison of varying deltat 
    mx = 1000
    mt = np.logspace(1,4,7)
    deltat = 0.5 / mt
    #mt = [200,300,400,500,600,700, 800, 900, 1000, 2000, 3000]
    dp = ParabolicProblem()
    errors1 = []; errors2 = []; lmbdas = []
    u = exp(-(pi**2)*t)*sin(pi*x)
    
    print('deltax = {}'.format(1 / mx))
    print('{:16}{:16}{:16}{:16}'.format('deltat', 'lambda', 'Backward Euler', 'Crank-Nicholson'))
    for n in mt:
        n = int(n)
        _,error1,lmbda1 = dp.solve_at_T(0.5, mx, n, 'BE', u_exact=u, plot=False)
        _,error2,lmbda2 = dp.solve_at_T(0.5, mx, n, 'CN', u_exact=u, plot=False)
        lmbdas.append(lmbda2)
        errors1.append(error1)
        errors2.append(error2)
        print('{:<16.8f}{:<16.2f}{:<16.8f}{:<16.8f}'.format(
                0.5/n, lmbda1, error1, error2))

    plt.loglog(deltat, errors1)
    plt.loglog(deltat, errors2)
    plt.ylabel('Error')
    plt.xlabel(r'$\Delta t$')
    
    plt.show()

## Worksheet 2 ##
   
## Q1 (a) 

def stability_check():
    mx = 5; mt = 30; T = 1
    hp = HyperbolicProblem()
    
    u= cos(pi*t)*sin(pi*x)

    uT, error, lmbda = hp.solve_at_T(T, mx, mt, 'I', u_exact=u)
    print('Error = {}, lambda = {}'.format(error, lmbda))


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
    
solver_demonstration()
#error_analysis()
#stability_check()