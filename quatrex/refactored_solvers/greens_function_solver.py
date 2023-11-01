# Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.

import numpy as np
from scipy.sparse import csr_matrix

from quatrex.refactored_utils.constants import e, k
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
    energy_fermi: dict[float],
    temperature: float,
    indices_of_neighboring_matrix: dict[np.ndarray],
    blocksize: int
) -> tuple[np.ndarray, np.ndarray]:

    # transform the self energy from flattened to list
    Self_energy_retarded_list = flattened_to_list_of_csr(
        Self_energy_retarded_flattened,
        indices_of_neighboring_matrix,
        Hamiltonian.shape[0])
    Self_energy_lesser_list = flattened_to_list_of_csr(
        Self_energy_lesser_flattened,
        indices_of_neighboring_matrix,
        Hamiltonian.shape[0])
    Self_energy_greater_list = flattened_to_list_of_csr(
        Self_energy_greater_flattened,
        indices_of_neighboring_matrix,
        Hamiltonian.shape[0])

    (Self_energy_retarded_list,
     Self_energy_lesser_list,
     Self_energy_greater_list) = symmetrize_self_energy(
        Self_energy_retarded_list,
        Self_energy_lesser_list,
        Self_energy_greater_list)

    G_greater_flattened = np.zeros((energy_array.size,
                                    indices_of_neighboring_matrix["row"].size),
                                   dtype=Self_energy_retarded_list[0].dtype)
    G_lesser_flattened = np.zeros((energy_array.size,
                                   indices_of_neighboring_matrix["row"].size),
                                  dtype=Self_energy_retarded_list[0].dtype)

    fermi_distribution = compute_fermi_distribution(energy_array, energy_fermi, temperature)

    for i, energy in enumerate(energy_array):

        System_matrix = get_system_matrix(Hamiltonian,
                                          Overlap_matrix,
                                          Self_energy_retarded_list[i],
                                          energy)

        OBCs, _ = compute_open_boundary_condition(System_matrix,
                                                  imaginary_limit=5e-4,
                                                  contour_integration_radius=1000,
                                                  blocksize=blocksize,
                                                  caller_function_name="G")

        

        apply_obc_to_system_matrix(System_matrix, OBCs, blocksize)

        apply_obc_to_self_energy(Self_energy_lesser_list[i],
                                 Self_energy_greater_list[i],
                                 OBCs,
                                 fermi_distribution["left"][i],
                                 fermi_distribution["right"][i],
                                 blocksize)

        G_retarded = np.linalg.inv(System_matrix.toarray())

        G_lesser = compute_greens_function(
            G_retarded, Self_energy_lesser_list[i])

        G_greater = compute_greens_function(
            G_retarded, Self_energy_greater_list[i])

        G_lesser_flattened[i] = csr_to_flattened(
            G_lesser, indices_of_neighboring_matrix)
        G_greater_flattened[i] = csr_to_flattened(
            G_greater, indices_of_neighboring_matrix)

    return G_greater_flattened, G_lesser_flattened


def get_system_matrix(
    Hamiltonian: csr_matrix,
    Overlap_matrix: csr_matrix,
    Self_energy_retarded: csr_matrix,
    energy: float,
):
    # possible to optimize TODO
    System_matrix = (energy + 1j * 1e-12)*Overlap_matrix - \
        Hamiltonian - Self_energy_retarded

    return System_matrix


def apply_obc_to_self_energy(
    Self_energy_lesser: csr_matrix,
    Self_energy_greater: csr_matrix,
    OBCs: dict[csr_matrix],
    fermi_distribution_left: float,
    fermi_distribution_right: float,
    blocksize: int
):

    Gamma_left = 1j * (OBCs["left"] - OBCs["left"].conj().T)
    Self_energy_lesser_left_boundary = 1j * \
        fermi_distribution_left * Gamma_left
    Self_energy_greater_left_boundary = 1j * \
        (fermi_distribution_left - 1) * Gamma_left
    Self_energy_lesser[:blocksize,
                       :blocksize] += Self_energy_lesser_left_boundary
    Self_energy_greater[:blocksize,
                        :blocksize] += Self_energy_greater_left_boundary

    Gamma_right = 1j * (OBCs["right"] - OBCs["right"].conj().T)
    Self_energy_lesser_right_boundary = 1j * \
        fermi_distribution_right * Gamma_right
    Self_energy_greater_right_boundary = 1j * \
        (fermi_distribution_right - 1) * Gamma_right
    Self_energy_lesser[-blocksize:, -
                       blocksize:] += Self_energy_lesser_right_boundary
    Self_energy_greater[-blocksize:, -
                        blocksize:] += Self_energy_greater_right_boundary



def compute_greens_function(
    System_matrix_inv: np.ndarray,
    Self_energy: np.ndarray
):

    G = System_matrix_inv @ Self_energy @ System_matrix_inv.conj().T

    return G


def symmetrize_self_energy(
    Self_energy_retarded_list,
    Self_energy_lesser_list,
    Self_energy_greater_list
):
    Self_energy_retarded_sym = []
    Self_energy_lesser_sym = []
    Self_energy_greater_sym = []
    for ie in range(len(Self_energy_lesser_list)):
        # symmetrize (lesser and greater have to be skewed symmetric)
        Self_energy_lesser_sym.append((Self_energy_lesser_list[ie] -
                                       Self_energy_lesser_list[ie].T.conj()) / 2)
        Self_energy_greater_sym.append((Self_energy_greater_list[ie] -
                                        Self_energy_greater_list[ie].T.conj()) / 2)
        Self_energy_retarded_sym.append(np.real(Self_energy_retarded_list[ie])
                                        + (Self_energy_greater_sym[ie] - Self_energy_lesser_sym[ie]) / 2)

    return Self_energy_retarded_sym, Self_energy_lesser_sym, Self_energy_greater_sym

def compute_fermi_distribution(
    energy_array : np.ndarray,
    energy_fermi : dict[float],
    temperature : float
)-> dict[np.ndarray]:
    fermi_distribution_left = 1 / (1 + np.exp((energy_array - energy_fermi["left"]) * e
                            / (k * temperature)))
    fermi_distribution_right = 1 / (1 + np.exp((energy_array - energy_fermi["right"]) * e
                            / (k * temperature)))
    return {"left": fermi_distribution_left, "right": fermi_distribution_right}
