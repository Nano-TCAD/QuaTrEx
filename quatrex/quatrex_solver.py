# Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.

import quatrex.constants
import quatrex.solvers_parameters

from quatrex.refactored_solvers.greens_function_solver import greens_function_solver
from quatrex.refactored_solvers.gw_solver import gw_solver

import bsparse

import numpy as np


class QuatrexSolver:

    def __init__(
        self,
        Hamiltonian: bsparse,
        Overlap_matrix: bsparse,
        Coulomb_matrix: bsparse,
        Neighboring_matrix_indices: dict[np.ndarray],
        energy_array: np.ndarray,
        fermi_levels: [float, float],
        conduction_band_energy: float,
        temperature: float,
        solver_mode: str,
    ):
        self._Hamiltonian = Hamiltonian
        self._Overlap_matrix = Overlap_matrix
        self._Coulomb_matrix = Coulomb_matrix
        self._Neighboring_matrix_indices = Neighboring_matrix_indices
        self._energy_array = energy_array
        self._fermi_levels = fermi_levels
        self._conduction_band_energy = conduction_band_energy
        self._temperature = temperature
        self._solver_mode = solver_mode

        self._base_type = np.complex128

        self._Coulomb_matrix_at_neighbor_indices = \
            self._Coulomb_matrix[self._Neighboring_matrix_indices["row"],
                                 self._Neighboring_matrix_indices["col"]]

        self._load_solver_parameters()
        self._compute_matmult_blocksize()

        if self._solver_mode == "gw":
            self._init_gw_storage()
        
        self._current_density = None
        self._electron_density = None
        self._hole_density = None
        self._density_of_states = None

    # ----- Interface -----
    def self_consistency_solve(
        self,
    ):

        for _ in range(self._max_iterations):
            # TODO remove blocksize argument
            G_lesser, G_greater = greens_function_solver(
                self._Hamiltonian,
                self._Overlap_matrix,
                self._Self_energy_retarded,
                self._Self_energy_lesser,
                self._Self_energy_greater,
                self._energy_array,
                self._fermi_levels,
                self._temperature,
                self._Neighboring_matrix_indices,
                self._blocksize)

            if self._solver_mode == "gw":
                # changes inplace the self energy and screened interaction
                # TODO remove blocksize argument and use pydantic for compute parameters
                gw_solver(
                    self._Screened_interaction_lesser,
                    self._Screened_interaction_greater,
                    self._Self_energy_retarded,
                    self._Self_energy_lesser,
                    self._Self_energy_greater,
                    G_lesser,
                    G_greater,
                    self._Coulomb_matrix,
                    self._Coulomb_matrix_at_neighbor_indices,
                    self._Neighboring_matrix_indices,
                    self._energy_array,
                    self._blocksize,
                    self._screened_interaction_stepping_factor,
                    self._self_energy_stepping_factor)

            if self._current_converged():
                break

        else:
            print("Maximum number of iterations reached")
            raise RuntimeError("The self consistency loop did not converge")

        # TODO add mode which computes additionally the retarded greens function
        G_retarded = greens_function_solver()

        self._compute_current(G_lesser, G_greater)
        self._compute_electron_density(G_lesser)
        self._compute_hole_density(G_greater)
        self._compute_density_of_states(G_retarded)

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

    def _init_gw_storage(
        self
    ):
        number_of_neighboring_matrix_indices = self._Neighboring_matrix_indices["row"].size
        number_of_energy_points = self._energy_array.size
        self._Self_energy_retarded = np.zeros(
            (number_of_energy_points, number_of_neighboring_matrix_indices),
            dtype=self._base_type)

        self._Self_energy_lesser = np.zeros(
            (number_of_energy_points, number_of_neighboring_matrix_indices),
            dtype=self._base_type)

        self._Self_energy_greater = np.zeros(
            (number_of_energy_points, number_of_neighboring_matrix_indices),
            dtype=self._base_type)

        self._Screened_interaction_greater = np.zeros(
            (number_of_energy_points, number_of_neighboring_matrix_indices),
            dtype=self._base_type)

        self._Screened_interaction_lesser = np.zeros(
            (number_of_energy_points, number_of_neighboring_matrix_indices),
            dtype=self._base_type)

    def _compute_current(
        self,
        G_lesser: list[bsparse],
        G_greater: list[bsparse]
    ):
        # uses the meir wingreen formula
        # to calculate the current
        number_of_energy_points = self._energy_array.size
        number_of_blocks = G_lesser[0].bshape
        self._current_density = np.zeros(
            (number_of_energy_points, number_of_blocks), dtype=G_lesser[0].dtype)

        # middle blocks
        for i in range(number_of_energy_points):
            for j in range(1, number_of_blocks-1):
                self._current_density[i, j] = np.real(
                    np.trace(G_lesser[i][j, j] - G_greater[i][j, j]))
        # first and last block
        for i in range(number_of_energy_points):
            self._current_density[i, 0] = np.real(
                np.trace(G_lesser[i][0, 0] - G_greater[i][0, 0]))
            self._current_density[i, -1] = np.real(
                np.trace(G_lesser[i][-1, -1] - G_greater[i][-1, -1]))

    def _compute_electron_density(
        self,
        G_lesser: list[bsparse]
    ):
        number_of_energy_points = self._energy_array.size
        number_of_blocks = G_lesser[0].bshape
        self._electron_density = np.zeros(
            (number_of_energy_points, number_of_blocks), dtype=G_lesser[0].dtype)
        for i in range(number_of_energy_points):
            for j in range(number_of_blocks):
                self._electron_density[i, j] = -1j * np.trace(G_lesser[i][j, j])

    def _compute_hole_density(
        self,
        G_greater: list[bsparse]
    ):
        number_of_energy_points = self._energy_array.size
        number_of_blocks = G_greater[0].bshape
        self._hole_density = np.zeros(
            (number_of_energy_points, number_of_blocks), dtype=G_greater[0].dtype)
        for i in range(number_of_energy_points):
            for j in range(number_of_blocks):
                self._hole_density[i, j] = 1j * np.trace(G_greater[i][j, j])

    def _compute_density_of_states(
        self,
        G_retarded: list[bsparse]
    ):
        number_of_energy_points = self._energy_array.size
        number_of_blocks = G_retarded[0].bshape
        self._density_of_states = np.zeros(
            (number_of_energy_points, number_of_blocks), dtype=G_retarded[0].dtype)
        for i in range(number_of_energy_points):
            for j in range(number_of_blocks):
                self._density_of_states[i, j] = 1j * np.trace(
                    G_retarded[i][j, j] - G_retarded[i][j, j].T.conj())

    def _current_converged(
        self
    ) -> bool:
        
        if self._current_density is None:
            return False
        current_left = np.sum(self._current_density[:, 0])
        current_right = np.sum(self._current_density[:, -1])
        current_convergence = np.abs(current_left - current_right)/np.abs(current_left)
        
        return current_convergence < current_convergence_threshold

    # ----- Private attributes -----

    _solver_mode: str
    _max_iterations: int
