# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 11:15:35 2018

@author: james
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

from helpers import numpify

def plot_solution(xs, uT, uexact=None, title='', uexacttitle='', style='ro'):
    """Plot the solution uT to a PDE problem at time t"""
    try:
        plt.plot(xs,uT,style,label='numerical')
    except:
        pass
        
    if uexact:
        u = numpify(uexact, 'x')
        xx = np.linspace(xs[0], xs[-1], 250)

        plt.plot(xx, u(xx),'b-',label=uexacttitle)

    plt.xlabel('x')
    plt.ylabel('u(x,T)')
    plt.title(title)
    plt.legend(loc='best')
    plt.show()

def animate_tsunami(xs, u, L):
    """animate the solution to the tsunami problem
    from: https://jakevdp.github.io/blog/2012/08/18/matplotlib-animation-tutorial/"""
    fig = plt.figure()
    ax = plt.axes(xlim=(0, L), ylim=(-2, 25))
    line, = ax.plot([], [], lw=2)
    # initialization function: plot the background of each frame
    def init():
        line.set_data([], [])
        return line,

    # animation function.  This is called sequentially
    def animate(i):
        line.set_data(xs, u[i])
        return line,
    
        # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=600, interval=20, blit=True)
    
    # save the animation as an mp4.  This requires ffmpeg or mencoder to be
    # installed.  The extra_args ensure that the x264 codec is used, so that
    # the video can be embedded in html5.  You may need to adjust this for
    # your system: for more information, see
    # http://matplotlib.sourceforge.net/api/animation_api.html
    anim.save('tsunami.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
    
    plt.show()