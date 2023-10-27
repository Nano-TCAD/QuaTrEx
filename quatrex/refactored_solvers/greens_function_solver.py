# Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.

import numpy as np
from scipy.sparse import csr_matrix

from quatrex.refactored_solvers.open_boundary_conditions import compute_open_boundary_condition
from quatrex.refactored_solvers.open_boundary_conditions import apply_obc_to_system_matrix
from quatrex.refactored_utils.utils import csr_to_flattened, flattened_to_list_of_csr


# To move into solver method
def compute_observables(
    G_retarded_diag_blocks: np.ndarray,
    G_lesser_diag_blocks: np.ndarray,
    G_greater_diag_blocks: np.ndarray
):
    number_of_blocks = G_retarded_diag_blocks.shape[1]
    number_of_energy_points = G_retarded_diag_blocks.shape[0]

    density_of_states = np.zeros(
        (number_of_energy_points, number_of_blocks), dtype=G_retarded_diag_blocks.dtype)
    electron_density = np.zeros(
        (number_of_energy_points, number_of_blocks), dtype=G_retarded_diag_blocks.dtype)
    hole_density = np.zeros(
        (number_of_energy_points, number_of_blocks), dtype=G_retarded_diag_blocks.dtype)
    current_density = np.zeros(
        (number_of_energy_points, number_of_blocks), dtype=G_retarded_diag_blocks.dtype)

    for i in range(number_of_energy_points):
        for j in range(number_of_blocks):
            density_of_states[i, j] = 1j * np.trace(
                G_retarded_diag_blocks[i, j] - G_retarded_diag_blocks[i, j].T.conj())
            electron_density[i, j] = -1j * np.trace(G_lesser_diag_blocks[i, j])
            hole_density[i, j] = 1j * np.trace(G_greater_diag_blocks[i, j])
            # current_density[i,j] = np.real(np.trace(G_lesser_diag_blocks[i,j] - G_greater_diag_blocks[i,j])) TODO Correct this formula


def greens_function_solver(
    Hamiltonian: csr_matrix,
    Overlap_matrix: csr_matrix,
    Self_energy_retarded_flattened: np.ndarray,
    Self_energy_lesser_flattened: np.ndarray,
    Self_energy_greater_flattened: np.ndarray,
    energy_array: np.ndarray,
    fermi_distribution_left: np.ndarray,
    fermi_distribution_right: np.ndarray,
    row_indices_kept: np.ndarray,
    col_indices_kept: np.ndarray,
    blocksize: int
) -> tuple[np.ndarray, np.ndarray]:

    # transform the self energy from flattened to list
    Self_energy_retarded_list = flattened_to_list_of_csr(
        Self_energy_retarded_flattened,
        row_indices_kept,
        col_indices_kept,
        Hamiltonian.shape[0])
    Self_energy_lesser_list = flattened_to_list_of_csr(
        Self_energy_lesser_flattened,
        row_indices_kept,
        col_indices_kept,
        Hamiltonian.shape[0])
    Self_energy_greater_list = flattened_to_list_of_csr(
        Self_energy_greater_flattened,
        row_indices_kept,
        col_indices_kept,
        Hamiltonian.shape[0])

    (Self_energy_retarded_list,
     Self_energy_lesser_list,
     Self_energy_greater_list) = symmetrize_self_energy(
        Self_energy_retarded_list,
        Self_energy_lesser_list,
        Self_energy_greater_list)

    G_greater_flattened = np.zeros((energy_array.size,
                                    row_indices_kept.size),
                                   dtype=Self_energy_retarded_list[0].dtype)
    G_lesser_flattened = np.zeros((energy_array.size,
                                   row_indices_kept.size),
                                  dtype=Self_energy_retarded_list[0].dtype)

    for i, energy in enumerate(energy_array):

        System_matrix = (energy + 1j * 1e-12)*Overlap_matrix - \
            Hamiltonian - Self_energy_retarded_list[i]

        # TODO --> Modification suggestion
        # System_matrix = make_system_matrix(energy, Overlap_matrix, Hamiltonian, Self_energy_retarded_list[i])

        OBCs, _ = compute_open_boundary_condition(System_matrix,
                                                  imaginary_limit=5e-4,
                                                  contour_integration_radius=1000,
                                                  blocksize=blocksize,
                                                  caller_function_name="G")

        fermi_distribution = {
            "left": fermi_distribution_left[i], "right": fermi_distribution_right[i]}

        apply_obc_to_system_matrix(System_matrix, OBCs, blocksize)

        apply_obc_to_self_energy(Self_energy_lesser_list[i],
                                 Self_energy_greater_list[i],
                                 OBCs,
                                 fermi_distribution,
                                 blocksize)

        G_retarded = np.linalg.inv(System_matrix.toarray())

        G_lesser, G_greater = compute_greens_function_lesser_and_greater(
            G_retarded, Self_energy_lesser_list[i], Self_energy_greater_list[i])

        G_lesser_flattened[i] = csr_to_flattened(G_lesser, row_indices_kept, col_indices_kept)
        G_greater_flattened[i] = csr_to_flattened(G_greater, row_indices_kept, col_indices_kept)

    return G_greater_flattened, G_lesser_flattened


def apply_obc_to_self_energy(
    Self_energy_lesser: csr_matrix,
    Self_energy_greater: csr_matrix,
    OBCs: dict[csr_matrix],
    fermi_distribution: dict[float],
    blocksize: int
):

    Gamma_left = 1j * (OBCs["left"] - OBCs["left"].conj().T)
    Self_energy_lesser_left_boundary = 1j * \
        fermi_distribution["left"] * Gamma_left
    Self_energy_greater_left_boundary = 1j * \
        (fermi_distribution["left"] - 1) * Gamma_left
    Self_energy_lesser[:blocksize,
                            :blocksize] += Self_energy_lesser_left_boundary
    Self_energy_greater[:blocksize,
                             :blocksize] += Self_energy_greater_left_boundary

    Gamma_right = 1j * (OBCs["right"] - OBCs["right"].conj().T)
    Self_energy_lesser_right_boundary = 1j * \
        fermi_distribution["right"] * Gamma_right
    Self_energy_greater_right_boundary = 1j * \
        (fermi_distribution["right"] - 1) * Gamma_right
    Self_energy_lesser[-blocksize:, -
                            blocksize:] += Self_energy_lesser_right_boundary
    Self_energy_greater[-blocksize:, -
                             blocksize:] += Self_energy_greater_right_boundary


def cut_to_tridiag(
    A: np.ndarray,
    blocksize: int
):
    # Delete elements of G_retarded that are outside of the tridiagonal block structure
    number_of_blocks = int(A.shape[0] / blocksize)
    zero_block = np.zeros((blocksize, blocksize))
    for row in range(number_of_blocks):
        for col in range(number_of_blocks):
            if col < row - 1 or col > row + 1:
                A[row*blocksize:(row+1)*blocksize, col *
                  blocksize:(col+1)*blocksize] = zero_block


def compute_greens_function_lesser_and_greater(
    G_retarded: np.ndarray,
    Self_energy_lesser: np.ndarray,
    Self_energy_greater: np.ndarray
):

    G_lesser = G_retarded @ Self_energy_lesser @ G_retarded.conj().T
    G_greater = G_retarded @ Self_energy_greater @ G_retarded.conj().T

    return G_lesser, G_greater


def symmetrize_self_energy(
    Self_energy_retarded_list,
    Self_energy_lesser_list,
    Self_energy_greater_list
):

    for ie in range(len(Self_energy_lesser_list)):
        # symmetrize (lesser and greater have to be skewed symmetric)
        Self_energy_lesser_list[ie] = (Self_energy_lesser_list[ie] -
                                       Self_energy_lesser_list[ie].T.conj()) / 2
        Self_energy_greater_list[ie] = (Self_energy_greater_list[ie] -
                                        Self_energy_greater_list[ie].T.conj()) / 2
        Self_energy_retarded_list[ie] = (np.real(Self_energy_retarded_list[ie])
                                         + (Self_energy_greater_list[ie] - Self_energy_lesser_list[ie]) / 2)

    return Self_energy_retarded_list, Self_energy_lesser_list, Self_energy_greater_list
