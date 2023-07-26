import numpy as np
from scipy import sparse
import time


def open_boundary_conditions(M00, M10, M01, V10):

    xR, cond = surface_function(M00, M10, M01)
    
    dM = M10 @ xR @ M01
    dV = M10 @ xR @ V10
    
    return xR, dM, dV, cond


def surface_function(M00, M10, M01):
    
    cond = float('inf')
    cond_limit = 1e-8
    max_iter = 125
    IC = 1
    
    alpha = M10
    beta = M01
    eps = M00
    eps_surf = M00.copy()
    
    while((cond>cond_limit) and (IC < max_iter)):
        
        inv_element = np.linalg.inv(eps)
        i_alpha = np.matmul(inv_element, alpha)
        i_beta = np.matmul(inv_element, beta)
        a_i_b = np.matmul(alpha, i_beta)
        b_i_a = np.matmul(beta, i_alpha)
        eps -= a_i_b + b_i_a
        eps_surf -= a_i_b
        alpha = np.matmul(alpha, i_alpha)
        beta = np.matmul(beta, i_beta)
        
        cond = (np.abs(alpha)+np.abs(beta)).sum()
        cond /= 2
        
        IC += 1
    #print(IC)
    #print(cond)
    if(cond > cond_limit):
        print('Warning: Surface function did not converge')
        print('Condition number: ', cond)
        cond = np.nan
        SF = None
        return SF, cond
    SF = np.linalg.inv(eps_surf)
    return SF, cond