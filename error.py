# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 11:16:50 2018

@author: james
"""

def error(uT, uexact):
    """Calculate the error between a solution value of t and the exact solution"""
    return np.linalg.norm(uT - uexact)