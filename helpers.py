# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 11:05:29 2018

@author: james
"""
from scipy import sparse
import numpy as np
import sympy as sp
from sympy.abc import x, t, y

def get_error(xs, uT, u_exact, norm='L2'):
    # calculate l-inf norm of calculated value of u and exact value
    u = numpify(u_exact, 'x')
    
    if norm == 'L2':
        return np.linalg.norm(u(xs)- uT)
    elif norm == 'Linf':
        return np.max(np.abs(u(xs)-uT))
    else:
        raise Exception('Norm type not implemented')

def tridiag(N, lower, main, upper):
    """
    Create an NxN tridiagonal matrix with diagonals lower, upper and main.
    Use sparse representation.
    """
    return sparse.diags([lower,main,upper],
                        offsets=[-1,0,1],
                        shape=(N, N),
                        format='csr')

def numpify(fn, args):
    """
    Create a vectorized function from either a constant or a sympy function.
    If a function is neither of these the original function is returned.
    
    Parameters
    fn        function to be vectorized
    num_args  1 or 2 variables
    """
    # first check if f is constant
    if isinstance(fn, (int, float)):
        if args == 'x' or args == 't':
            return np.vectorize(lambda y: fn,
                                otypes=[np.float32])
        elif args == 'x t' or args == 'x y':
            return np.vectorize(lambda y, s: fn,
                                otypes=[np.float32])
        else:
            raise Exception(
                    'Only functions of 1 or 2 variables are permitted')
    else:
        # then assume a sympy function and try to vectorize it
        try:
            if args == 'x':
                return sp.lambdify(x, fn, 'numpy')
            elif args == 't':
                return sp.lambdify(t, fn, 'numpy')
            elif args == 'x t':
                return sp.lambdify((x,t), fn, 'numpy')
            elif args == 'x y':
                return sp.lambdify((x,y), fn, 'numpy')
            else:
                raise Exception(
                        'Only functions of 1 or 2 variables are permitted')
        except:
            # otherwise just return the original function
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
            
    

