# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 11:05:29 2018

@author: james
"""
from scipy import sparse

def tridiag(N, lower, main, upper):
    return sparse.diags([lower,main,upper],
                        offsets=[-1,0,1],
                        shape=(N+1,N+1),
                        format='csr')
    

