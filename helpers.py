# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 11:05:29 2018

@author: james
"""
from scipy import sparse
import numpy as np
import sympy as sp
from sympy.abc import x, t

def tridiag(N, lower, main, upper):
    """
    Create an NxN tridiagonal matrix with diagonals lower, upper and main.
    Use sparse representation.
    """
    return sparse.diags([lower,main,upper],
                        offsets=[-1,0,1],
                        shape=(N, N),
                        format='csr')

def numpify(fn, arg):
    """
    Create a vectorized function from either a constant or a sympy function.
    If a function is neither of these the original function is returned
    
    Parameters
    fn   function to be vectorized
    arg  'x' or 'x t' for functions of 1 or 2 variables respectively
    """
    # first check if f is constant
    if isinstance(fn, (int, float)):
        if arg == 'x':
            return np.vectorize(lambda y: fn,
                                otypes=[np.float32])
        elif arg == 'x t':
            return np.vectorize(lambda y, s: fn,
                                otypes=[np.float32])
        else:
            raise Exception(
                    'Only functions of x or x and t are permitted')
    else:
        # then assume a sympy function and try to vectorize it
        try:
            if arg == 'x':
                return sp.lambdify(x, fn, 'numpy')
                print('success')
            elif arg == 'x t':
                return sp.lambdify((x,t), fn, 'numpy')
            else:
                raise Exception(
                        'Only functions of x or x and t are permitted')
        except:
            # just return the original function
            return fn

def numpify_many(*fns):
    """
    Create a vectorized function from either a constant or a sympy function.
    If a function is neither of these the original function is returned
    
    Parameters
    fns    list of tuples of the form (fn, 'x') or (fn, 'x t') to be vectorized
    """
    np_fns = []
    for fn, arg in fns:
        np_fns.append(numpify(fn,arg))

    return np_fns
            
def vectorize_xfn(*xs):
    fns = []
    for x in xs:         
        if isinstance(x, (int, float)):
            fns.append(np.vectorize(lambda y: x, otypes=[np.float]))
        else:
            fns.append(x)
    return fns

def vectorize_xtfn(x):
    if isinstance(x, (int, float)):
        return np.vectorize(lambda y, t: x, otypes=[np.float])
    else:
        return x
    

