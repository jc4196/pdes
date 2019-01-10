# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 11:46:56 2018

@author: jc4196
"""

from waveproblem import WaveProblem
from sympy import *
from sympy.abc import x, t, c, L
init_printing()

def example1():
    wp1 = WaveProblem()
    wp1.pprint()
    u = cos(pi*c*t/L)*sin(pi*x/L)
    wp1.plot_at_T(2, u_exact=u)
    
example1()