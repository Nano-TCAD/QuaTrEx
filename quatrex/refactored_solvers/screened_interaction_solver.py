# Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import identity as sp_identity

from quatrex.refactored_solvers.open_boundary_conditions import compute_open_boundary_condition
from quatrex.refactored_solvers.open_boundary_conditions import apply_obc_to_system_matrix
from quatrex.refactored_utils.utils import csr_to_flattened
from quatrex.refactored_utils.utils import flattened_to_list_of_csr

from quatrex.files_to_refactor import dL_OBC_eigenmode_cpu


def screened_interaction_solver(
    Coulomb_matrix: csr_matrix,
    Polarization_lesser_flattened: np.ndarray,
    Polarization_greater_flattened: np.ndarray,
    number_of_energy_points: int,
    indices_of_neighboring_matrix: dict[np.ndarray],
    blocksize: int
):

    Polarization_greater_list = flattened_to_list_of_csr(
        Polarization_greater_flattened,
        indices_of_neighboring_matrix,
        Coulomb_matrix.shape[0])
    Polarization_lesser_list = flattened_to_list_of_csr(
        Polarization_lesser_flattened,
        indices_of_neighboring_matrix,
        Coulomb_matrix.shape[0])

    (Polarization_retarded_list,
     Polarization_lesser_list,
     Polarization_greater_list) = symmetrize_polarization(
        Polarization_greater_list,
        Polarization_lesser_list)

    Screened_interaction_greater_flattened = np.zeros((number_of_energy_points,
                                                       indices_of_neighboring_matrix["row"].size),
                                                      dtype=Polarization_greater_list[0].dtype)
    Screened_interaction_lesser_flattened = np.zeros((number_of_energy_points,
                                                      indices_of_neighboring_matrix["row"].size),
                                                     dtype=Polarization_greater_list[0].dtype)

    # TODO: explanation why the first point is skipped
    for i in range(1, number_of_energy_points):

        System_matrix = get_system_matrix(
            Coulomb_matrix, Polarization_retarded_list[i], blocksize)

        # TODO: Modify how the overall workflow deal with the increased blocksize
        blocksize_after_matmult = update_blocksize(blocksize, System_matrix)

        L_greater = get_L(
            Coulomb_matrix, Polarization_greater_list[i], blocksize)
        L_lesser = get_L(Coulomb_matrix, Polarization_lesser_list[i], blocksize)

        OBCs, beyn_gr = compute_open_boundary_condition(System_matrix,
                                                        imaginary_limit=1e-4,
                                                        contour_integration_radius=1e6,
                                                        blocksize=blocksize_after_matmult,
                                                        caller_function_name="W")

        apply_obc_to_system_matrix(System_matrix, OBCs, blocksize_after_matmult)

        System_matrix_inv = np.linalg.inv(System_matrix.toarray())

        L_correction_of_obc(L_greater, L_lesser, System_matrix,
                            beyn_gr, blocksize_after_matmult)

        Screened_interaction_lesser = compute_screened_interaction(
            System_matrix_inv, L_lesser)
        Screened_interaction_greater = compute_screened_interaction(
            System_matrix_inv, L_greater)

        Screened_interaction_lesser_flattened[i] = csr_to_flattened(
            Screened_interaction_lesser, indices_of_neighboring_matrix)
        Screened_interaction_greater_flattened[i] = csr_to_flattened(
            Screened_interaction_greater, indices_of_neighboring_matrix)

    return  Screened_interaction_lesser_flattened, Screened_interaction_greater_flattened


def get_system_matrix(
    Coulomb_matrix: csr_matrix,
    Polarization_retarded: csr_matrix,
    blocksize: int
):

    System_matrix = sp_identity(
        Coulomb_matrix.shape[0]) - Coulomb_matrix @ Polarization_retarded

    # Correct system matrix for infinite contact
    System_matrix[0:blocksize, 0:blocksize] -= Coulomb_matrix[blocksize:2*blocksize, 0:blocksize] @\
        Polarization_retarded[0:blocksize, blocksize:2*blocksize]

    System_matrix[-blocksize:, -blocksize:] -= Coulomb_matrix[-2*blocksize:-blocksize, -blocksize:] @\
        Polarization_retarded[-blocksize:, -2*blocksize:-blocksize]

    return System_matrix


def update_blocksize(
    blocksize: int,
    System_matrix: np.ndarray
):

    return blocksize


def get_L(
    Coulomb_matrix: csr_matrix,
    Polarization: csr_matrix,
    blocksize: int
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

    # TODO: Test offdiagonal blocks because in reference example they are zero
    # due to the coulomb matrix being rather sparse
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

    # TODO: Test offdiagonal blocks because in reference example they are zero
    # due to the coulomb matrix being rather sparse
    C1 = Coulomb_matrix[-2*blocksize:-blocksize, -blocksize:] @\
        Polarization[-2*blocksize:-blocksize, -blocksize:] @\
        Coulomb_matrix[-2*blocksize:-blocksize, -blocksize:].conj().T

    L[-2*blocksize:-blocksize, -blocksize:] += C1

    C1 = Coulomb_matrix[-2*blocksize:-blocksize, -blocksize:] @\
        Polarization[-blocksize:, -2*blocksize:-blocksize] @\
        Coulomb_matrix[-2*blocksize:-blocksize, -blocksize:].conj().T

    L[-blocksize:, -2*blocksize:-blocksize] += C1

    return L


def compute_screened_interaction(
    System_matrix_inv: np.ndarray,
    L: np.ndarray
):

    Screened_interaction = System_matrix_inv @ L @ System_matrix_inv.conj().T

    return Screened_interaction


def L_correction_of_obc(
    L_greater,
    L_lesser,
    System_matrix,
    beyn_gr: dict[np.ndarray],
    blocksize
):

    L_greater_left_OBC_block, L_lesser_left_OBC_block = dL_OBC_eigenmode_cpu.get_dl_obc_alt(
        beyn_gr["left"],
        L_greater[:blocksize, :blocksize].toarray(),
        L_greater[:blocksize, blocksize:2*blocksize].toarray(),
        L_lesser[:blocksize, :blocksize].toarray(),
        L_lesser[:blocksize, blocksize:2*blocksize].toarray(),
        System_matrix[blocksize:2*blocksize, :blocksize].toarray(),
        blk="L")

    # TODO: Modify the handling of error
    if np.isnan(L_lesser_left_OBC_block).any():
        print(
            'Error: Algorithm failed to compute the self-energy for L at the left boundary')
        exit()
    L_greater[:blocksize, :blocksize] += L_greater_left_OBC_block
    L_lesser[:blocksize, :blocksize] += L_lesser_left_OBC_block

    L_greater_right_OBC_block, L_lesser_right_OBC_block = dL_OBC_eigenmode_cpu.get_dl_obc_alt(
        beyn_gr["right"],
        L_greater[-blocksize:, -blocksize:].toarray(),
        L_greater[-blocksize:, -2*blocksize:-blocksize].toarray(),
        L_lesser[-blocksize:, -blocksize:].toarray(),
        L_lesser[-blocksize:, -2*blocksize:-blocksize].toarray(),
        System_matrix[-2*blocksize:-blocksize, -blocksize:].toarray(),
        blk="R")

    # TODO: Modify the handling of error
    if np.isnan(L_lesser_right_OBC_block).any():
        print(
            'Error: Algorithm failed to compute the self-energy for L at the left boundary')
        exit()

    L_greater[-blocksize:, -blocksize:] += L_greater_right_OBC_block
    L_lesser[-blocksize:, -blocksize:] += L_lesser_right_OBC_block


def symmetrize_polarization(
    Polarization_greater_list,
    Polarization_lesser_list
):

    # Symmetrization of Polarization (TODO: check if this is needed)
    Polarization_retarded_list = []
    for ie in range(len(Polarization_greater_list)):
        # Anti-Hermitian symmetrizing of PL and PG
        Polarization_lesser_list[ie] = (Polarization_lesser_list[ie] -
                                        Polarization_lesser_list[ie].conj().T) / 2

        Polarization_greater_list[ie] = (Polarization_greater_list[ie] -
                                         Polarization_greater_list[ie].conj().T) / 2

        # PR has to be derived from PL and PG and then has to be symmetrized
        Polarization_retarded_list.append((Polarization_greater_list[ie] -
                                           Polarization_lesser_list[ie]) / 2)

    return Polarization_retarded_list, Polarization_lesser_list, Polarization_greater_list
