# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 10:52:12 2018

@author: james
"""
from scipy import sparse
import numpy as np

def tridiag(N, A00, A01, ANm1N, ANN, m, l, u):
    """
    Construct a sparse tridiagonal matrix of the form
    
    A00 A01 0 .....       0
    l   m   u  0..
    0   l   m   u
    ...       .
    ....        .
    ......    l    m      u
    ...            ANm1N  ANN
    
    """
    main = [A00] + (N-1)*[m] + [ANN]
    lower = (N-1)*[l] + [ANm1N]
    upper = [A01] + (N-1)*[u]
    
    return sparse.diags(diagonals = [main, lower, upper],
                        offsets = [0, -1, 1],
                        shape = (N+1, N+1),
                        format='csr')

def forwardeuler(mx, deltax, deltat, lmbda, p, q):
    # create forward Euler matrix
    A = sparse.identity(mx+1, format='csr')
    
    B = tridiag(mx, 
                1-2*lmbda, 2*lmbda, 2*lmbda, 1-2*lmbda,
                1-2*lmbda, lmbda, lmbda)
    
    def v(t):
        return np.concatenate(([-2*lmbda*deltax*p(t), lmbda*p(t)],
                               np.zeros(mx-3),
                               [lmbda*q(t), 2*lmbda*deltax*q(t)]))
    return A, B, v
    

def backwardeuler(mx, deltax, deltat, lmbda, p, q):
    # create backward Euler matrix
    A = tridiag(mx,
                1+2*lmbda, -2*lmbda, -2*lmbda, 1+2*lmbda,
                1+2*lmbda, -lmbda, -lmbda)
    
    B = sparse.identity(mx+1, format='csr')
    
    def v(t):
        return np.concatenate(([-2*lmbda*deltax*p(t),lmbda*p(t)],
                                np.zeros(mx-3),
                                [lmbda*q(t), 2*lmbda*deltax*q(t)]))
    
    return A, B, v

def cranknicholson(mx, deltax, deltat, lmbda, p, q):
    # create Crank-Nicholson matrices
    A = tridiag(mx,
                1+lmbda, -lmbda, -lmbda, 1+lmbda,
                1+lmbda, -0.5*lmbda, -0.5*lmbda)
    
    B = tridiag(mx,
                1-lmbda, lmbda, lmbda, 1-lmbda,
                1-lmbda, 0.5*lmbda, 0.5*lmbda)
    
    def v(t):
        return np.concatenate(([-lmbda*deltax*(p(t) + p(t + deltat)),
                               0.5*lmbda*(p(t) + p(t + deltat))],
                               np.zeros(mx-3),
                               [0.5*lmbda*(q(t) + q(deltat)),
                                lmbda*deltax*(q(t) + q(t + deltat))]))
    
    return A, B, v
    
    
    