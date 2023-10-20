# Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.

import numpy as np
from scipy.sparse import csr_matrix

from quatrex.OBC.beyn_cpu import beyn



def compute_open_boundary_condition(
    M : csr_matrix,
    imaginary_limit : float,
    contour_integration_radius : float,
    blocksize : int,
):
    
    imaginary_limit = 5e-4
    contour_integration_radius = 1000
    
    left_boundary_size  = blocksize
    
    
    _, success, left_gr_beyn, self_energy_left_boundary, _ = beyn(M[:left_boundary_size, :left_boundary_size].toarray(),
                                                                  M[:left_boundary_size, left_boundary_size: 2*left_boundary_size].toarray(),
                                                                  M[left_boundary_size:2*left_boundary_size, :left_boundary_size].toarray(),
                                                                  imaginary_limit,
                                                                  contour_integration_radius,
                                                                  'L',
                                                                  function='G')

    if np.isnan(success):
        print('Error: Beyn algorithm failed to compute the self-energy at the left boundary')
        exit()


    right_boundary_size = blocksize

    _, success, right_gr_beyn, self_energy_right_boundary, _ = beyn(M[-right_boundary_size:, -right_boundary_size:].toarray(),
                                                                    M[-2*right_boundary_size:-right_boundary_size, -right_boundary_size:].toarray(),
                                                                    M[-right_boundary_size:, -2*right_boundary_size: -right_boundary_size].toarray(),
                                                                    imaginary_limit,
                                                                    contour_integration_radius,
                                                                    'R',
                                                                    function='G')

    if np.isnan(success):
        print('Error: Beyn algorithm failed to compute the self-energy at the right boundary')
        exit()
        
        
    return {"left": self_energy_left_boundary, "right": self_energy_right_boundary}, {"left": left_gr_beyn, "right": right_gr_beyn}



def apply_obc_to_system_matrix(
    A,
    OBCs,
    blocksize
):

    A[:blocksize, :blocksize] -= OBCs["left"]
    A[-blocksize:, -blocksize:] -= OBCs["right"]
    
    