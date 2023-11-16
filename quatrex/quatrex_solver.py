# Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.

from quatrex.refactored_solvers.greens_function_solver import greens_function_solver
from quatrex.refactored_solvers.gw_solver import gw_solver
from quatrex.solvers_parameters import SolverParameters
from quatrex.files_to_refactor.adjust_conduction_band_edge import adjust_conduction_band_edge

import bsparse
import scipy
import tomllib

import numpy as np


class QuatrexSolver:

    def __init__(
        self,
        Hamiltonian: bsparse.BSparse,
        Overlap_matrix: bsparse.BSparse,
        Coulomb_matrix: bsparse.BSparse,
        Neighboring_matrix_indices: dict[np.ndarray],
        energy_array: np.ndarray,
        fermi_levels: dict[float],
        conduction_band_energy: float,
        temperature: float,
        solver_mode: str,
        solver_parameter_path: str = None,
    ):
        self._Hamiltonian = Hamiltonian
        self._Overlap_matrix = Overlap_matrix
        self._Coulomb_matrix = Coulomb_matrix
        self._Neighboring_matrix_indices = Neighboring_matrix_indices
        self._energy_array = energy_array
        self._fermi_levels = fermi_levels

        # TODO: calibrate band edge
        # DFT result is from k=0,
        # but the band edge is not necessarily at k=0
        self._conduction_band_energy = conduction_band_energy
        self._energy_difference_fermi_conduction_band = \
            {"left": fermi_levels["left"] - conduction_band_energy,
             "right": fermi_levels["right"] - conduction_band_energy}

        self._temperature = temperature
        self._solver_mode = solver_mode
        self._base_type = np.complex128

        self._Coulomb_matrix_at_neighbor_indices = \
            self._Coulomb_matrix[self._Neighboring_matrix_indices["row"],
                                 self._Neighboring_matrix_indices["col"]]

        self._load_solver_parameters(solver_parameter_path)
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

        for i in range(self._solver_parameters.self_consistency_loop_max_iterations):


            # TODO remove blocksize argument
            G_lesser, G_greater, current_density = greens_function_solver(
                self._Hamiltonian,
                self._Overlap_matrix,
                self._Self_energy_retarded,
                self._Self_energy_lesser,
                self._Self_energy_greater,
                self._energy_array,
                self._fermi_levels,
                self._temperature,
                self._Neighboring_matrix_indices,
                self._blocksize,
                if_compute_current_density=(i % self._solver_parameters.check_convergence_every_n_iterations ==
                                            self._solver_parameters.check_convergence_every_n_iterations-1))

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
                    self._solver_parameters)

                # Adjust band edge to track band gap
                conduction_band_energy = adjust_conduction_band_edge(
                    self._conduction_band_energy,
                    self._energy_array,
                    self._Overlap_matrix,
                    self._Hamiltonian,
                    self._Self_energy_retarded,
                    self._Self_energy_lesser,
                    self._Self_energy_greater,
                    self._Neighboring_matrix_indices,
                    self._blocksize)

                # change fermi energy accordingly
                self._conduction_band_energy = conduction_band_energy
                self._fermi_levels = self._energy_difference_fermi_conduction_band = \
                    {"left": conduction_band_energy +
                     self._energy_difference_fermi_conduction_band["left"],
                     "right": conduction_band_energy +
                     self._energy_difference_fermi_conduction_band["right"]}

            if i % self._solver_parameters.check_convergence_every_n_iterations == 0:
                if self._current_converged(current_density):
                    break

        else:
            print("Maximum number of iterations reached")
            raise RuntimeError("The self consistency loop did not converge")

        # TODO add mode which computes additionally the retarded greens function
        # G_retarded = greens_function_solver()

        self._compute_current_density(G_lesser, G_greater)
        self._compute_electron_density(G_lesser)
        self._compute_hole_density(G_greater)
        # self._compute_density_of_states(G_retarded)

        #return status

    def get_current_density(
        self
    ):
        if self._current_density is None:
            raise RuntimeError("The current density has not been computed yet")
        return self._current_density

    def get_electron_density(
        self
    ):
        if self._electron_density is None:
            raise RuntimeError("The electron density has not been computed yet")
        return self._electron_density

    def get_hole_density(
        self
    ):
        if self._hole_density is None:
            raise RuntimeError("The hole density has not been computed yet")
        return self._hole_density

    def get_density_of_states(
        self
    ):
        if self._density_of_states is None:
            raise RuntimeError(
                "The density of states has not been computed yet")
        return self._density_of_states

    # ----- Private methods -----

    def _load_solver_parameters(
        self,
        solver_parameter_path: str
    ):
        if solver_parameter_path is None:
            self._solver_parameters = SolverParameters()
        else:
            with open(solver_parameter_path, "rb") as f:
                config = tomllib.load(f)
            self._solver_parameters = SolverParameters.model_validate(config)

    def _compute_matmult_blocksize(
        self
    ):
        """
        idea: compute the sparsity of the matrix product V @ P @ V.H
        and check if far away blocks are zero
        such that only the sparsity of the matrices matter
        and not the values
        """

        block_sizes = self._Coulomb_matrix.row_sizes

        # TODO easier to determine from a device
        # which is a repetition of unit cells
        # i.e. constant block size and same sparsity
        # in constant sparsity among the diagonal blocks
        # and off diagonal blocks

        # consider sparsity of the underlying data
        # create matrix with Neighboring_matrix_indices sparsity
        constant = 1000000
        Polarization_ones = scipy.sparse.coo_matrix(
            (constant*np.ones_like(self._Neighboring_matrix_indices["row"]),
                (self._Neighboring_matrix_indices["row"],
                 self._Neighboring_matrix_indices["col"])),
            shape=self._Hamiltonian.shape)
        # to bsparse
        Polarization_ones = bsparse.BCOO.from_sparray(Polarization_ones,
                                                      sizes=(block_sizes, block_sizes))
        # TODO hack: difficult to make a copy with only constant
        # assume that the elements are much smaller than the constant
        Coulomb_matrix_ones = self._Coulomb_matrix.copy() + constant


        L = Coulomb_matrix_ones @ Polarization_ones @ Coulomb_matrix_ones.T

        third_blocks_zero = True
        second_blocks_zero = True
        for i in range(L.bshape[0]):
            for j in range(L.bshape[1]):
                if abs(i-j) == 3:
                    if L[i, j].nnz != 0:
                        third_blocks_zero = False
                if abs(i-j) == 2:
                    if L[i, j].nnz != 0:
                        second_blocks_zero = False
        if (third_blocks_zero and second_blocks_zero):
            increase_range = 1
        elif third_blocks_zero:
            increase_range = 2
        else:
            increase_range = 3

        # determine new tilling strategy
        divisible = (block_sizes.size % increase_range) == 0
        new_sizes = np.zeros(
                block_sizes.size//increase_range, dtype=block_sizes.dtype)
        if divisible:
            for i in range(new_sizes.size):
                for j in range(increase_range):
                    new_sizes[i] += block_sizes[increase_range*i + j]
        else:
            # not divisible number of blocks
            # idea: fuse the addional blocks with the second to last block
            # make contact blocks larger
            for j in range(increase_range):
                new_sizes[0] += block_sizes[j]
                new_sizes[-1] += block_sizes[-j]
            for i in range(1, new_sizes.size-1):
                for j in range(increase_range):
                    new_sizes[i] = block_sizes[increase_range*i + j]
            # add the additional blocks to the second to last block
            # TODO: optimize such that the additional blocks are distributed
            # to the smallest blocks
            for i in range(block_sizes.size % increase_range):
                new_sizes[-2] += block_sizes[-increase_range-1 - i]
        return new_sizes

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

    def _compute_current_density(
        self,
        G_lesser,
        G_greater
    ):
        # depending on mode different current formula is used
        if self._solver_mode == "gw":
            # uses the meir wingreen formula
            # the calculation of the current density is intertwined
            # with the calculation of the lesser and greater greens function
            _, _, current_density = greens_function_solver(
                self._Hamiltonian,
                self._Overlap_matrix,
                self._Self_energy_retarded,
                self._Self_energy_lesser,
                self._Self_energy_greater,
                self._energy_array,
                self._fermi_levels,
                self._temperature,
                self._Neighboring_matrix_indices,
                self._blocksize,
                if_compute_current_density=True)

            self._current_density = current_density

        elif self._solver_mode == "gf":
            # TODO: use the landauer büttiker formula
            pass
        else:
            raise ValueError("The solver mode is not supported")

    def _compute_electron_density(
        self,
        G_lesser: list[bsparse.BSparse]
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
        G_greater: list[bsparse.BSparse]
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
        G_retarded: list[bsparse.BSparse]
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
        self,
        current_density: np.ndarray
    ) -> bool:

        if current_density is None:
            return False
        current_left = np.sum(self._current_density[:, 0])
        current_right = np.sum(self._current_density[:, -1])
        current_convergence = np.abs(
            current_left - current_right)/np.abs(current_left)

        return current_convergence < self._solver_parameters.current_convergence_threshold

    # ----- Private attributes -----

    _solver_mode: str
    _solver_parameters: SolverParameters
