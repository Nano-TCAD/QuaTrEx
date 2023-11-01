# Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.

import quatrex.constantes
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
            G_retarded, G_lesser, G_greater = compute_greens_functions()
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
        self
    ):
        pass
    
    
    def _compute_electron_density(
        self
    ):
        pass
    
    
    def _compute_hole_density(
        self
    ):
        pass
    
    
    def _compute_density_of_states(
        self
    ):
        pass
    
    
    def _current_converged(
        self,
        current_convergence_threshold : float
    ) -> bool:
        pass

    
    # ----- Private attributes -----
    
    _solver_mode: str
    _max_iterations: int