# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 11:46:18 2019

@author: james

This file is more attractively presented in the Jupyter notebook 

'Worksheets.ipynb'.

"""
import numpy as np
from sympy import *
from sympy.abc import x,y, t

from parabolicpde import ParabolicProblem, backwardeuler
from hyperbolicpde import HyperbolicProblem
from ellipticpde import EllipticProblem

from visualizations import plot_solution
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

## Worksheet 1 ##
## 
## Q2 (a) 

def solver_demonstration():
    mx = 30; mt = 1000; L = 1; T = 0.5
    
    def uI(x):
        return np.sin(np.pi*x/L)
    
      
    xs, uT, lmbda = backwardeuler(mx, mt, L, T,
                                  1, 0,
                                  uI, 0, 0, 'Dirichlet', 'Dirichlet')

    plot_solution(xs, uT)
   
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
    
def sor_test():
    # elliptic example
    mx = 40              # number of gridpoints in x
    my = 20              # number of gridpoints in y
    maxerr = 1e-4        # target error
    maxcount = 1000      # maximum number of iteration steps
    omega = 1.5
    ep1 = EllipticProblem()
    
    u = sin(pi*x/2)*sinh(pi*y/2)/sinh(pi/2)
    u_new, count, error, err_true = ep1.solve(mx, my, maxcount, maxerr, omega, u)
    print('Final iteration error = {}\nMax error = {}\nno. of iterations = {}'.format(error, err_true, count))
    
def omegas_test():
    omegas = np.linspace(1,2,30)
    mx = 40              # number of gridpoints in x
    my = 20              # number of gridpoints in y
    maxerr = 1e-4        # target error
    maxcount = 1000      # maximum number of iteration steps
    omega = 1.5
    ep = EllipticProblem()  
    counts = []
    
    for omega in omegas:
        _,count,_,_ = ep.solve(mx, my, maxcount, maxerr, omega, plot=False)
        counts.append(count)
    
    plt.plot(omegas, counts)
    plt.xlabel(r'$\omega$')
    plt.ylabel('No. of iterations')
    plt.show()
    
def min_omega():
    mx = 40              # number of gridpoints in x
    my = 20              # number of gridpoints in y
    maxerr = 1e-4        # target error
    maxcount = 1000      # maximum number of iteration steps
    ep = EllipticProblem()   
    
    def num_iterations(omega):
        _,count,_,_=ep.solve(mx, my, maxcount, maxerr, omega, plot=False)
        return count
    
    
    res = minimize_scalar(num_iterations, bracket=(1,2), tol=1e-3)
    print(res)
solver_demonstration()
#error_analysis()
#stability_check()
#sor_test()
#omegas_test()
#min_omega()