# Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.

import numpy as np
from scipy.sparse import csr_matrix

from quatrex.files_to_refactor.beyn_cpu import beyn
from quatrex.obc.sancho_rubio import sancho_rubio


def compute_open_boundary_condition(
    System_matrix: csr_matrix,
    imaginary_limit: float,
    contour_integration_radius: float,
    blocksize: int,
    caller_function_name: str
):

    _, success, surface_greens_function_left, _, _ = beyn(System_matrix[:blocksize, :blocksize].toarray(),
                                                          System_matrix[:blocksize, blocksize: 2 *
                                                                        blocksize].toarray(),
                                                          System_matrix[blocksize:2*blocksize,
                                                                        :blocksize].toarray(),
                                                          imaginary_limit,
                                                          contour_integration_radius,
                                                          'L',
                                                          function=caller_function_name)

    # TODO: Modify the handling of Beyn errors
    if np.isnan(success):
        surface_greens_function_left = sancho_rubio(
            System_matrix[:blocksize, :blocksize].toarray(),
            System_matrix[blocksize:2*blocksize, :blocksize].toarray(),
            System_matrix[:blocksize, blocksize: 2*blocksize].toarray())

    self_energy_left_boundary = compute_boundary_self_energy(
        surface_greens_function_left,
        System_matrix[blocksize:2*blocksize, :blocksize].toarray(),
        System_matrix[:blocksize, blocksize: 2*blocksize].toarray())


    _, success, surface_greens_function_right, _, _ = beyn(System_matrix[-blocksize:, -blocksize:].toarray(),
                                                           System_matrix[-2*blocksize:-blocksize, -
                                                                         blocksize:].toarray(),
                                                           System_matrix[-blocksize:, -2*blocksize: -
                                                                         blocksize].toarray(),
                                                           imaginary_limit,
                                                           contour_integration_radius,
                                                           'R',
                                                           function=caller_function_name)

    # TODO: Modify the handling of Beyn errors
    if np.isnan(success):
        surface_greens_function_right = sancho_rubio(
            System_matrix[-blocksize:, -blocksize:].toarray(),
            System_matrix[-2*blocksize:-blocksize, -blocksize:].toarray(),
            System_matrix[-blocksize:, -2*blocksize: -blocksize].toarray())

    self_energy_right_boundary = compute_boundary_self_energy(
        surface_greens_function_right,
        System_matrix[-2*blocksize:-blocksize, -blocksize:].toarray(),
        System_matrix[-blocksize:, -2*blocksize: -blocksize].toarray())


    return {"left": self_energy_left_boundary, "right": self_energy_right_boundary},\
           {"left": surface_greens_function_left, "right": surface_greens_function_right}


def compute_boundary_self_energy(
    surface_greens_function_left: np.ndarray,
    M_ij: np.ndarray,
    M_ji: np.ndarray = None
):
    if M_ji is None:
        M_ji = M_ij.conj().T
    return M_ij @ surface_greens_function_left @ M_ji


def apply_obc_to_system_matrix(
    A,
    OBCs,
    blocksize
):

    A[:blocksize, :blocksize] -= OBCs["left"]
    A[-blocksize:, -blocksize:] -= OBCs["right"]
