# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 17:53:58 2019

@author: james
"""
import sympy as sp

class ParabolicProblem:
    """Object specifying a diffusion type problem of the form
    
    du/dt = kappa*d^2u/dx^2 + f(x)
    """
    def __init__(self,
                 kappa=1,
                 L=1,
                 ic=sp.sin(sp.pi*x),
                 lbc=Dirichlet(0,0),
                 rbc=Dirichlet(1, 0),
                 source=0):
        self.kappa = kappa    # Diffusion coefficient
        self.L = L            # Length of interval
        self.lbc = lbc        # Left boundary condition as above
        self.rbc = rbc        # Right boundary condition as above
        self.ic = ic          # Initial condition function h(x)
        self.source = source  # Source function f(x)
 
 
    def pprint(self, title=''):
        """Print the diffusion problem with latex"""
        print(title)
        u = sp.Function('u')
        x, t = sp.symbols('x t')
        display(sp.Eq(u(x,t).diff(t),
                      self.kappa*u(x,t).diff(x,2) + self.source))
        self.lbc.pprint()
        self.rbc.pprint()
        display(sp.Eq(u(x,0), self.ic))
    
    def solve_at_T(self, T, mx, mt, scheme, plot=True, u_exact=None, title=''):
        xs, uT =  scheme(mx, mt, self.L, T,
                         self.kappa, self.source, self.ic,
                         self.lbc.apply_rhs, self.rbc.apply_rhs,
                         self.lbc.get_type(), self.rbc.get_type())
        
        if u_exact:
            uTsym = u_exact.subs({kappa: self.kappa,
                                  L: self.L,
                                  t: T})
            #u = sp.lambdify(x, uTsym)
            u = numpify(uTsym, 'x')
            error = np.linalg.norm(u(xs) - uT)            
            if plot:       
                plot_solution(xs, uT, u, title=title,
                              uexacttitle=r'${}$'.format(sp.latex(uTsym)))            
        else:
            error = None
            if plot:
                plot_solution(xs, uT, title=title)            
        
        return uT, error