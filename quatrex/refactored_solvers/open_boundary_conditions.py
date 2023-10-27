# Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.

import numpy as np
from scipy.sparse import csr_matrix

from quatrex.files_to_refactor.beyn_cpu import beyn
from quatrex.files_to_refactor import sancho


def compute_open_boundary_condition(
    M: csr_matrix,
    imaginary_limit: float,
    contour_integration_radius: float,
    blocksize: int,
    caller_function_name: str
):

    _, success, left_gr_beyn, self_energy_left_boundary, _ = beyn(M[:blocksize, :blocksize].toarray(),
                                                                  M[:blocksize, blocksize: 2 *
                                                                      blocksize].toarray(),
                                                                  M[blocksize:2*blocksize,
                                                                      :blocksize].toarray(),
                                                                  imaginary_limit,
                                                                  contour_integration_radius,
                                                                  'L',
                                                                  function=caller_function_name)

    # TODO: Modify the handling of Beyn errors
    if np.isnan(success):
        # TODO: modify the open_boundary_conditions function, identity parsing is not correct
        # Coulomb matrix is used in Wr that we don't produce anyway
        left_gr_beyn, self_energy_left_boundary, _, success = sancho.open_boundary_conditions(
            M[:blocksize, :blocksize].toarray(),
            M[blocksize:2*blocksize, :blocksize].toarray(),
            M[:blocksize, blocksize: 2*blocksize].toarray(),
            np.identity(blocksize))
        if np.isnan(success):
            print("Error: Sancho open boundary conditions failed [left side]")
            exit()

    _, success, right_gr_beyn, self_energy_right_boundary, _ = beyn(M[-blocksize:, -blocksize:].toarray(),
                                                                    M[-2*blocksize:-blocksize, -
                                                                        blocksize:].toarray(),
                                                                    M[-blocksize:, -2*blocksize: -
                                                                        blocksize].toarray(),
                                                                    imaginary_limit,
                                                                    contour_integration_radius,
                                                                    'R',
                                                                    function=caller_function_name)

    # TODO: Modify the handling of Beyn errors
    if np.isnan(success):
        # TODO: modify the open_boundary_conditions function, identity parsing is not correct
        # Coulomb matrix is used in Wr that we don't produce anyway
        right_gr_beyn, self_energy_right_boundary, _, success = sancho.open_boundary_conditions(
            M[-blocksize:, -blocksize:].toarray(),
            M[-2*blocksize:-blocksize, -blocksize:].toarray(),
            M[-blocksize:, -2*blocksize: -blocksize].toarray(),
            np.identity(blocksize))
        if np.isnan(success):
            print("Error: Sancho open boundary conditions failed [right side]")
            exit()

    return {"left": self_energy_left_boundary, "right": self_energy_right_boundary}, {"left": left_gr_beyn, "right": right_gr_beyn}


def apply_obc_to_system_matrix(
    A,
    OBCs,
    blocksize
):

    A[:blocksize, :blocksize] -= OBCs["left"]
    A[-blocksize:, -blocksize:] -= OBCs["right"]
