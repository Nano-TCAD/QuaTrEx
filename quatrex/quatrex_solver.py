# Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.

import quatrex.constants
import quatrex.solvers_parameters

from quatrex.refactored_solvers import greens_function_solver

import bsparse

import numpy as np

    
class QuatrexSolver:
    
    def __init__(
		self,
		Hamiltonian: bsparse,
		Overlap_matrix: bsparse,
		Coulomb_matrix: bsparse,
		Neighboring_matrix_indices: np.ndarray,
		energy_array: np.ndarray,
		fermi_levels: [float,float],
		conduction_band_energy: float,
		temperature: float,
		solver_mode: str,
	):
        self._load_solver_parameters() 
        self._compute_matmult_blocksize()
        
        if self._solver_mode == "gw":
            self._init_gw_storage()
    
    # ----- Interface -----
    def self_consistency_solve(
		self,
	):
        current_iteration = 0
        while(not self._self_current_converged() or current_iteration >= self._max_iterations):
            current_iteration += 1
            G_lesser, G_greater = compute_greens_functions()
            if self._solver_mode == "gw":
                Self_energy += compute_gw_self_energy(G_retarded, G_lesser, G_greater, Screened_interactions, GW_self_energy, coulomb_potential,)
        
        self._compute_current()
        self._compute_electron_density()
        self._compute_hole_density()
        self._compute_density_of_states()
        
        return status
        
    
    def get_current(
        self
    ):
        pass    
    
    
    def get_electron_density(
        self
    ):
        pass    
    
    
    def get_hole_density(
        self
    ):
        pass
    
    
    def get_density_of_states(
        self
    ):
        pass
    
    
    
    # ----- Private methods -----
    def _load_solver_parameters(
        self
    ):
        pass
        
        
    def _compute_matmult_blocksize(
        self
    ):
        pass
    
    
    def _compute_current(
        self,
        G_lesser: list[bsparse],
        G_greater: list[bsparse]
    ):
        # uses the meir wingreen formula
        # to calculate the current
        number_of_energy_points = self._energy_array.size
        number_of_blocks = G_lesser[0].bshape
        current_density = np.zeros(
            (number_of_energy_points, number_of_blocks), dtype=G_lesser[0].dtype)

        # middle blocks
        for i in range(number_of_energy_points):
            for j in range(1, number_of_blocks-1):
                current_density[i,j] = np.real(np.trace(G_lesser[i][j,j] - G_greater[i][j,j]))
        # first and last block
        for i in range(number_of_energy_points):
            current_density[i,0] = np.real(np.trace(G_lesser[i][0,0] - G_greater[i][0,0]))
            current_density[i,-1] = np.real(np.trace(G_lesser[i][-1,-1] - G_greater[i][-1,-1])) 
            
    def _compute_electron_density(
        self,
        G_lesser: list[bsparse]
    ):
        number_of_energy_points = self._energy_array.size
        number_of_blocks = G_lesser[0].bshape
        electron_density = np.zeros(
            (number_of_energy_points, number_of_blocks), dtype=G_lesser[0].dtype)
        for i in range(number_of_energy_points):
            for j in range(number_of_blocks):
                electron_density[i, j] = -1j * np.trace(G_lesser[i][j, j])
    
    
    def _compute_hole_density(
        self,
        G_greater: list[bsparse]
    ):
        number_of_energy_points = self._energy_array.size
        number_of_blocks = G_greater[0].bshape
        hole_density = np.zeros(
            (number_of_energy_points, number_of_blocks), dtype=G_greater[0].dtype)
        for i in range(number_of_energy_points):
            for j in range(number_of_blocks):
                hole_density[i, j] = 1j * np.trace(G_greater[i][j, j])
    
    
    def _compute_density_of_states(
        self,
        G_retarded: list[bsparse]
    ):
        number_of_energy_points = self._energy_array.size
        number_of_blocks = G_retarded[0].bshape
        density_of_states = np.zeros(
            (number_of_energy_points, number_of_blocks), dtype=G_retarded[0].dtype)
        for i in range(number_of_energy_points):
            for j in range(number_of_blocks):
                density_of_states[i, j] = 1j * np.trace(
                    G_retarded[i][j, j] - G_retarded[i][j, j].T.conj())


    def _current_converged(
        self,
        current_convergence_threshold : float
    ) -> bool:
        pass

    
    # ----- Private attributes -----
    
    _solver_mode: str
    _max_iterations: int