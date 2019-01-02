# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 11:05:29 2018

@author: james
"""
from scipy import sparse
import numpy as np

def tridiag(N, lower, main, upper):
    return sparse.diags([lower,main,upper],
                        offsets=[-1,0,1],
                        shape=(N, N),
                        format='csr')
    
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
    

