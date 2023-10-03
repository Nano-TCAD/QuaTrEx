# Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.

import numpy as np
from scipy.sparse import csr_matrix

from quatrex.OBC.beyn_cpu import beyn




# To move into solver method
def compute_observables(
    G_retarded_diagblocks : np.ndarray, 
    G_lesser_diagblocks : np.ndarray,  
    G_greater_diagblocks : np.ndarray
):
    number_of_blocks = G_retarded_diagblocks.shape[1]
    number_of_energy_points = G_retarded_diagblocks.shape[0]
    
    density_of_states = np.zeros((number_of_energy_points,number_of_blocks), dtype=G_retarded_diagblocks.dtype)
    electron_density = np.zeros((number_of_energy_points,number_of_blocks), dtype=G_retarded_diagblocks.dtype)
    hole_density = np.zeros((number_of_energy_points,number_of_blocks), dtype=G_retarded_diagblocks.dtype)
    current_density = np.zeros((number_of_energy_points,number_of_blocks), dtype=G_retarded_diagblocks.dtype)
    
    for i in range(number_of_energy_points):
        for j in range(number_of_blocks):
            density_of_states[i,j] = 1j * np.trace(G_retarded_diagblocks[i,j] - G_retarded_diagblocks[i,j].T.conj())
            electron_density[i,j] = -1j * np.trace(G_lesser_diagblocks[i,j])
            hole_density[i,j] = 1j * np.trace(G_greater_diagblocks[i,j])
            # current_density[i,j] = np.real(np.trace(G_lesser_diagblocks[i,j] - G_greater_diagblocks[i,j])) TODO Correct this formula




# greens_function_solver.py
def greens_function_solver(
        G_retarded_diagblocks : np.ndarray, 
        G_retarded_upper_blocks : np.ndarray, 
        G_lesser_diagblocks : np.ndarray, 
        G_lesser_upperblocks : np.ndarray, 
        G_greater_diagblocks : np.ndarray, 
        G_greater_upperblocks : np.ndarray,
        Hamiltonian : csr_matrix,
        Overlap_matrix : csr_matrix,
        Self_energy_retarded : csr_matrix,
        Self_energy_lesser : csr_matrix,
        Self_energy_greater : csr_matrix,
        energy_array : np.ndarray,
        fermi_distribution_left : np.ndarray,
        fermi_distribution_right : np.ndarray,
        blocksize : int
    ):

    G_retarded : csr_matrix
    G_lesser : csr_matrix
    G_greater : csr_matrix

    for i, energy in enumerate(energy_array):

        M = (energy + 1j * 1e-12)*Overlap_matrix - Hamiltonian - Self_energy_retarded[i]
    
        Self_energy_retarded_left_boundary, Self_energy_retarded_right_boundary = compute_open_boundary_condition(M, blocksize)
        

        
        apply_boundary_conditions(M, 
                                    Self_energy_lesser[i],
                                    Self_energy_greater[i],
                                    Self_energy_retarded_left_boundary,
                                    Self_energy_retarded_right_boundary,
                                    fermi_distribution_left[i],
                                    fermi_distribution_right[i],
                                    blocksize)
        
        G_retarded = np.linalg.inv(M.toarray())
        

        #cut_to_tridiag(G_retarded, blocksize)
                    

        

        G_lesser, G_greater = compute_greens_function_lesser_and_greater(G_retarded, Self_energy_lesser[i], Self_energy_greater[i], blocksize)

        # plt.matshow(np.abs(Self_energy_lesser[i].toarray()))          
        # plt.matshow(np.abs(Self_energy_greater[i].toarray()))
        # plt.matshow(np.abs(G_lesser))          
        # plt.matshow(np.abs(G_greater))
        # plt.show()            
        
        # while True:
        #     pass
        
        # Extract the blocks from G_retarded, G_lesser, G_greater and store them in the corresponding arrays
        number_of_blocks = int(Hamiltonian.shape[0] / blocksize)
        
        for j in range(number_of_blocks):
            G_retarded_diagblocks[i,j] = G_retarded[j * blocksize : (j + 1) * blocksize, j * blocksize : (j + 1) * blocksize]
            G_lesser_diagblocks[i,j] = G_lesser[j * blocksize : (j + 1) * blocksize, j * blocksize : (j + 1) * blocksize]
            G_greater_diagblocks[i,j] = G_greater[j * blocksize : (j + 1) * blocksize, j * blocksize : (j + 1) * blocksize]

        for j in range(number_of_blocks-1):
            G_retarded_upper_blocks[i,j] = G_retarded[j * blocksize : (j + 1) * blocksize, (j + 1) * blocksize : (j + 2) * blocksize]
            G_lesser_upperblocks[i,j] = G_lesser[j * blocksize : (j + 1) * blocksize, (j + 1) * blocksize : (j + 2) * blocksize]
            G_greater_upperblocks[i,j] = G_greater[j * blocksize : (j + 1) * blocksize, (j + 1) * blocksize : (j + 2) * blocksize]


def compute_open_boundary_condition(
    M : csr_matrix,
    blocksize : int,
):
    
    imaginary_limit = 5e-4
    contour_integration_radius = 1000
    
    left_boundary_size  = blocksize
    
    # import matplotlib.pyplot as plt
    # plt.matshow(np.abs(M[:left_boundary_size, :left_boundary_size].toarray()))
    # plt.matshow(np.abs(M[:left_boundary_size, left_boundary_size: 2*left_boundary_size].toarray()))
    # plt.matshow(np.abs(M[left_boundary_size:2*left_boundary_size, :left_boundary_size].toarray()))
    # plt.show()
    
    _, success, _, self_energy_left_boundary, _ = beyn(M[:left_boundary_size, :left_boundary_size].toarray(),
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

    _, success, _, self_energy_right_boundary, _ = beyn(M[-right_boundary_size:, -right_boundary_size:].toarray(),
                                                       M[-2*right_boundary_size:-right_boundary_size, -right_boundary_size:].toarray(),
                                                       M[-right_boundary_size:, -2*right_boundary_size: -right_boundary_size].toarray(),
                                                       imaginary_limit,
                                                       contour_integration_radius,
                                                       'R',
                                                       function='G')

    if np.isnan(success):
        print('Error: Beyn algorithm failed to compute the self-energy at the right boundary')
        exit()
        
        
    return self_energy_left_boundary, self_energy_right_boundary
        
        
        
def apply_boundary_conditions(
    M : csr_matrix,
    Self_energy_lesser : csr_matrix,
    Self_energy_greater : csr_matrix,
    Self_energy_retarded_left_boundary : csr_matrix,
    Self_energy_retarded_right_boundary : csr_matrix,
    fermi_distribution_left_at_current_energy : float,
    fermi_distribution_right_at_current_energy : float,
    blocksize : int
):
    
    left_boundary_size  = blocksize
    right_boundary_size = blocksize
    
    M[:left_boundary_size, :left_boundary_size] -= Self_energy_retarded_left_boundary
    M[-right_boundary_size:,-right_boundary_size:] -= Self_energy_retarded_right_boundary
    
    Gamma_left = 1j * (Self_energy_retarded_left_boundary - Self_energy_retarded_left_boundary.conj().T)
    Self_energy_lesser_left_boundary  = 1j * fermi_distribution_left_at_current_energy * Gamma_left
    Self_energy_greater_left_boundary = 1j * (fermi_distribution_left_at_current_energy - 1) * Gamma_left
    Self_energy_lesser[:left_boundary_size, :left_boundary_size]  += Self_energy_lesser_left_boundary
    Self_energy_greater[:left_boundary_size, :left_boundary_size] += Self_energy_greater_left_boundary
    
    Gamma_right = 1j * (Self_energy_retarded_right_boundary - Self_energy_retarded_right_boundary.conj().T)
    Self_energy_lesser_right_boundary = 1j * fermi_distribution_right_at_current_energy * Gamma_right
    Self_energy_greater_right_boundary = 1j * (fermi_distribution_right_at_current_energy - 1) * Gamma_right
    Self_energy_lesser[-right_boundary_size:,-right_boundary_size:]  += Self_energy_lesser_right_boundary
    Self_energy_greater[-right_boundary_size:,-right_boundary_size:] += Self_energy_greater_right_boundary
    



def cut_to_tridiag(
    A: np.ndarray,
    blocksize : int
):
    # Delete elements of G_retarded that are outside of the tridiagonal block structure
    number_of_blocks = int(A.shape[0] / blocksize)
    zero_block = np.zeros((blocksize, blocksize))
    for row in range(number_of_blocks):
        for col in range(number_of_blocks):
            if col < row - 1 or col > row + 1:
                A[row*blocksize:(row+1)*blocksize, col*blocksize:(col+1)*blocksize] = zero_block
    


def compute_greens_function_lesser_and_greater(
    G_retarded : np.ndarray, 
    Self_energy_lesser : np.ndarray, 
    Self_energy_greater : np.ndarray,
    blocksize : int
):
    
    """ G_lesser = G_retarded @ Self_energy_lesser
    #cut_to_tridiag(G_lesser, blocksize)
    G_lesser = G_lesser @ G_retarded.conj().T
    
    G_greater = G_retarded @ Self_energy_greater
    #cut_to_tridiag(G_greater, blocksize)
    G_greater = G_greater @ G_retarded.conj().T """
    
    
    G_lesser = G_retarded @ Self_energy_lesser @ G_retarded.conj().T
    G_greater = G_retarded @ Self_energy_greater @ G_retarded.conj().T
    
    return G_lesser, G_greater

