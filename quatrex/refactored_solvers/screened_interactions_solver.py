# Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.

import numpy as np
from scipy.sparse import csr_matrix

from quatrex.refactored_solvers.open_boundary_conditions import compute_open_boundary_condition



def screened_interactions_solver(
    Coulomb_matrix: csr_matrix,
    Polarization_greater: csr_matrix,
    Polarization_lesser: csr_matrix,
    Polarization_retarded: csr_matrix,
    energy_array: np.ndarray,
    blocksize: int,
):

    for i, energy in enumerate(energy_array):
        
        System_matrix = 1-Coulomb_matrix @ Polarization_retarded

        # Get modified system ready for OBC
        modify_system_for_obc(System_matrix, Coulomb_matrix, Polarization_retarded)

        # Compute W OBC
        
        # Applied W OBC
        
        Screened_interaction_retarded = np.linalg.inv(System_matrix) @ Coulomb_matrix
        
        compute_screened_interaction_lesser_and_greater(Screened_interaction_retarded, 
                                                        Polarization_lesser, 
                                                        Polarization_greater)
    
"""
Notes about variable naming:

- _ct is for the original matrix complex conjugated
- _mm how certain sizes after matrix multiplication, because a block tri diagonal gets two new non zero off diagonals
- _s stands for values related to the left/start/top contact block
- _e stands for values related to the right/end/bottom contact block
- _d stands for diagonal block (X00/NN in matlab)
- _u stands for upper diagonal block (X01/NN1 in matlab)
- _l stands for lower diagonal block (X10/N1N in matlab) 
- exception _l/_r can stand for left/right in context of condition of OBC
- _rgf are the not true inverse tensor (partial inverse) 
more standard notation would be small characters for partial and large for true
but this goes against python naming style guide
"""
    
    
def modify_system_for_obc(
    System_matrix,
    Coulomb_matrix,
    Polarization_retarded
):
    
    
    
    
def compute_open_boundary_conditions(
    
):
    pass

    # Call Bayne
    
    # Otherwise call Sancho-Rubio
    
    
def apply_boundary_conditions(
    
):
    pass

    # Correct OBC
    
    # Applied OBC
    
    

def compute_screened_interaction_lesser_and_greater(
    Screened_interaction_retarded : np.ndarray, 
    Polarization_lesser : np.ndarray, 
    Polarization_greater : np.ndarray
):
    
    Screened_interaction_lesser = Screened_interaction_retarded @ Polarization_lesser @ Screened_interaction_retarded.conj().T
    Screened_interaction_greater = Screened_interaction_retarded @ Polarization_greater @ Screened_interaction_retarded.conj().T
    
    return Screened_interaction_lesser, Screened_interaction_greater