# Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.

import numpy as np
from scipy.sparse import csr_matrix

from quatrex.refactored_solvers.open_boundary_conditions import compute_open_boundary_condition



def screened_interaction_solver(
    Coulomb_matrix: csr_matrix,
    Polarization_greater: csr_matrix,
    Polarization_lesser: csr_matrix,
    Polarization_retarded: csr_matrix,
    energy_array: np.ndarray,
    blocksize: int,
):
    
    Coulomb_matrix_localcopy = Coulomb_matrix.copy()

    for i, energy in enumerate(energy_array):
        
        System_matrix = 1-Coulomb_matrix_localcopy @ Polarization_retarded

        # L = V @ P<> @ V^dagger
        L_lesser = np.zeros_like(System_matrix, dtype=System_matrix.dtype)
        L_greater = np.zeros_like(System_matrix, dtype=System_matrix.dtype)

        # Compute OBC
        
        # Apply OBC to the system matrix
        apply_boundary_conditions(System_matrix, 
                                  System_matrix_left_obc, 
                                  System_matrix_right_obc, 
                                  blocksize)
        
        # Apply OBC to L (That is RHS)
        apply_boundary_conditions(Coulomb_matrix_localcopy, 
                                  Coulomb_matrix_localcopy_left_obc, 
                                  Coulomb_matrix_localcopy_right_obc, 
                                  blocksize)
        
        apply_boundary_conditions(L_lesser, 
                                  L_lesser_left_obc, 
                                  L_lesser_right_obc, 
                                  blocksize)
        
        apply_boundary_conditions(L_greater, 
                                  L_greater_left_obc, 
                                  L_greater_right_obc, 
                                  blocksize)
        
        System_matrix_inv = np.linalg.inv(System_matrix)
        
        Screened_interaction_retarded = System_matrix_inv @ Coulomb_matrix_localcopy
        
        Screened_interaction_lesser, Screened_interaction_greater\
            = compute_screened_interaction_lesser_and_greater(System_matrix_inv, 
                                                              L_lesser, 
                                                              L_greater)
            
        return Screened_interaction_retarded, Screened_interaction_lesser, Screened_interaction_greater
    
    
    
def compute_open_boundary_conditions(
    
):
    pass

    # Call Bayne
    
    # Otherwise call Sancho-Rubio
    
    
def apply_boundary_conditions(
    A,
    Left_obc,
    Right_obc,
    blocksize
):

    A[:blocksize, :blocksize] += Left_obc
    A[-blocksize:, -blocksize:] += Right_obc
    
    
    
    

def compute_screened_interaction_lesser_and_greater(
    System_matrix_inv : np.ndarray, 
    L_lesser : np.ndarray, 
    L_greater : np.ndarray
):
    
    Screened_interaction_lesser = System_matrix_inv @ L_lesser @ System_matrix_inv.conj().T
    Screened_interaction_greater = System_matrix_inv @ L_greater @ System_matrix_inv.conj().T
    
    return Screened_interaction_lesser, Screened_interaction_greater