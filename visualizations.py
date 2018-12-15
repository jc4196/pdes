# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 11:15:35 2018

@author: james
"""

import numpy as np
import matplotlib.pyplot as pl

def plot_solution(xs, uT, uexact=None, title='', uexacttitle=''):
    """Plot the solution uT to a PDE problem at time t"""
    try:
        pl.plot(xs,uT,'ro',label='numerical')
    except:
        pass
        
    if uexact:
        xx = np.linspace(xs[0], xs[-1], 250)

        pl.plot(xx, uexact(xx),'b-',label=uexacttitle)

    pl.xlabel('x')
    pl.ylabel('u(x,T)')
    pl.title(title)
    pl.legend(loc='best')
    pl.show()

def animate_solution(trange):
    """animate the solution to a PDE problem for a range of t values"""
    pass