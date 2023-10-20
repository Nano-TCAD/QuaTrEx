# Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.

import numpy as np
from scipy.sparse import csr_matrix

from quatrex.refactored_solvers.open_boundary_conditions import compute_open_boundary_condition
from quatrex.refactored_solvers.open_boundary_conditions import apply_obc_to_system_matrix
from quatrex.refactored_solvers.utils import csr_to_triple_array

from quatrex.OBC import dL_OBC_eigenmode_cpu



def screened_interaction_solver(
    Screened_interaction_retarded_diag_blocks : np.ndarray,
    Screened_interaction_retarded_upper_blocks : np.ndarray,
    Screened_interaction_lesser_diag_blocks : np.ndarray,
    Screened_interaction_lesser_upper_blocks : np.ndarray,
    Screened_interaction_greater_diag_blocks : np.ndarray,
    Screened_interaction_greater_upper_blocks : np.ndarray,
    Coulomb_matrix: csr_matrix,
    Polarization_greater: csr_matrix,
    Polarization_lesser: csr_matrix,
    Polarization_retarded: csr_matrix,
    energy_array: np.ndarray,
    blocksize: int,
):
    
    for i, energy in enumerate(energy_array):
        
        System_matrix = get_system_matrix(Coulomb_matrix, Polarization_retarded[i])
        
        blocksize_after_matmult = update_blocksize(blocksize, System_matrix)
        
        L_greater = get_L(Coulomb_matrix, Polarization_greater[i])
        L_lesser = get_L(Coulomb_matrix, Polarization_lesser[i])
        
        OBCs, beyn_gr = compute_open_boundary_condition(System_matrix,
                                                        imaginary_limit=5e-4,
                                                        contour_integration_radius=1000, 
                                                        blocksize=blocksize_after_matmult)
        
        apply_obc_to_system_matrix(System_matrix, OBCs, blocksize_after_matmult)
        
        System_matrix_inv = np.linalg.inv(System_matrix)
        
        L_correction_of_obc(L_greater, L_lesser, System_matrix, beyn_gr, blocksize)
        
        Screened_interaction_lesser = compute_screened_interaction(System_matrix_inv, L_lesser) 
        Screened_interaction_greater = compute_screened_interaction(System_matrix_inv, L_greater)  

        # TODO: modify the blocksize slicing 
        Screened_interaction_lesser_diag_blocks[i],
        Screened_interaction_lesser_upper_blocks[i] = csr_to_triple_array(Screened_interaction_lesser, blocksize_after_matmult)
        Screened_interaction_greater_diag_blocks[i],
        Screened_interaction_greater_upper_blocks[i] = csr_to_triple_array(Screened_interaction_greater, blocksize_after_matmult)

    
    
    
def get_system_matrix(
    Coulomb_matrix : csr_matrix,
    Polarization_retarded : csr_matrix,
    blocksize : int
):
    
    System_matrix = 1-Coulomb_matrix @ Polarization_retarded
    
    # correct system matrix for infitie contact
    System_matrix[0:blocksize, 0:blocksize] = -Coulomb_matrix[blocksize:2*blocksize, 0:blocksize] @\
                                               Polarization_retarded[0:blocksize, blocksize:2*blocksize]
                                               
    System_matrix[-blocksize:, -blocksize:] = -Coulomb_matrix[-2*blocksize:-blocksize, -blocksize:] @\
                                               Polarization_retarded[-blocksize:, -2*blocksize:-blocksize]                                           
    
    return System_matrix
    
    
def update_blocksize(
    blocksize : int,
    System_matrix : np.ndarray
):
    
    return blocksize*2
    
    
def get_L(
    Coulomb_matrix : csr_matrix,
    Polarization : csr_matrix,
    blocksize : int
):
    """ Derived from matrice multiplication V @ P @ V^T taking into accounts
    infinite contact (ie. repetitions ont the first/last slice of the matrix).
    """
    
    L = Coulomb_matrix @ Polarization @ Coulomb_matrix.conj().T    
    
    # Upper left corrections
    C1 = Coulomb_matrix[blocksize:2*blocksize, 0:blocksize] @\
         Polarization[0:blocksize, 0:blocksize] @\
         Coulomb_matrix[blocksize:2*blocksize, 0:blocksize].conj().T
         
    C2 = Coulomb_matrix[0:blocksize, 0:blocksize] @\
         Polarization[blocksize:2*blocksize, 0:blocksize] @\
         Coulomb_matrix[blocksize:2*blocksize, 0:blocksize].conj().T
         
    C3 = Coulomb_matrix[blocksize:2*blocksize, 0:blocksize] @\
         Polarization[0:blocksize, blocksize:2*blocksize] @\
         Coulomb_matrix[0:blocksize, 0:blocksize].conj().T
    
    L[0:blocksize, 0:blocksize] += C1+C2+C3
    
    C1 = Coulomb_matrix[blocksize:2*blocksize, 0:blocksize] @\
         Polarization[blocksize:2*blocksize, 0:blocksize] @\
         Coulomb_matrix[blocksize:2*blocksize, 0:blocksize].conj().T
                           
    L[blocksize:2*blocksize, 0:blocksize] += C1
    
    C1 = Coulomb_matrix[blocksize:2*blocksize, 0:blocksize] @\
         Polarization[0:blocksize, blocksize:2*blocksize] @\
         Coulomb_matrix[blocksize:2*blocksize, 0:blocksize].conj().T
    
    L[0:blocksize, blocksize:2*blocksize] += C1
    
    
    # Lower right corrections
    # V_{N-1,N} P_{N,N-1} V_{N,N} + V_{N,N} P_{N-1,N} V_{N-1,N} + V_{N-1,N} P_{N,N} V_{N-1,N}
    C1 = Coulomb_matrix[-2*blocksize:-blocksize, -blocksize:] @\
         Polarization[-blocksize:, -2*blocksize:-blocksize] @\
         Coulomb_matrix[-blocksize:, -blocksize:].conj().T
         
    C2 = Coulomb_matrix[-blocksize:, -blocksize:] @\
         Polarization[-2*blocksize:-blocksize, -blocksize:] @\
         Coulomb_matrix[-2*blocksize:-blocksize, -blocksize:].conj().T
         
    C3 = Coulomb_matrix[-2*blocksize:-blocksize, -blocksize:] @\
         Polarization[-blocksize:, -blocksize:] @\
         Coulomb_matrix[-2*blocksize:-blocksize, -blocksize:].conj().T
    
    L[-blocksize:, -blocksize:] += C1+C2+C3
    
    C1 = Coulomb_matrix[-2*blocksize:-blocksize, -blocksize:] @\
         Polarization[-2*blocksize:-blocksize, -blocksize:] @\
         Coulomb_matrix[-2*blocksize:-blocksize, -blocksize:].conj().T
                           
    L[-2*blocksize:-blocksize, -blocksize:] += C1
    
    C1 = Coulomb_matrix[-2*blocksize:-blocksize, -blocksize:] @\
         Polarization[-blocksize:, -2*blocksize:-blocksize] @\
         Coulomb_matrix[-2*blocksize:-blocksize, -blocksize:].conj().T
    
    L[-blocksize:,-2*blocksize:-blocksize] += C1
    
    return L
    
      
    
    
def compute_screened_interaction(
    System_matrix_inv : np.ndarray, 
    L : np.ndarray
):
    
    Screened_interaction = System_matrix_inv @ L @ System_matrix_inv.conj().T
    
    return Screened_interaction


def L_correction_of_obc(
    
):
    
    L_greater_left_OBC_block, L_lesser_left_OBC_block = dL_OBC_eigenmode_cpu.get_dl_obc_alt(
                                                            Chi_left_BC_block,
                                                            L_greater_left_diag_block.toarray(),
                                                            L_greater_left_upper_block.toarray(),
                                                            L_leser_left_diag_block.toarray(),
                                                            L_leser_left_upper_block.toarray(),
                                                            M_retarded_left_lower_block.toarray(),
                                                            blk="L")

    if np.isnan(L_lesser_left_OBC_block).any():
        cond_l = np.nan
    else:
        L_greater_left_BC_block += L_greater_left_OBC_block
        L_lesser_left_BC_block += L_lesser_left_OBC_block

    L_greater_right_OBC_block, L_lesser_right_OBC_block = dL_OBC_eigenmode_cpu.get_dl_obc_alt(
                                            Chi_right_BC_block,
                                            L_greater_right_diag_block.toarray(),
                                            L_greater_right_lower_block.toarray(),
                                            L_leser_right_diag_block.toarray(),
                                            L_leser_right_lower_block.toarray(),
                                            M_retarded_right_upper_block.toarray(),
                                            blk="R")

    if np.isnan(L_lesser_right_OBC_block).any():
        cond_r = np.nan
    else:
        L_greater_right_BC_block += L_greater_right_OBC_block
        L_lesser_right_BC_block += L_lesser_right_OBC_block