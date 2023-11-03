# Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.

import numpy as np
import bsparse


def compute_current_meir_wingreen(
    System_matrix: bsparse.BSparse,
    G_lesser: bsparse,
    G_greater: bsparse,    
    Self_energy_lesser: bsparse.BSparse,
    Self_energy_greater: bsparse.BSparse,
    OBCs: dict[np.ndarray],
    fermi_distribution_right: float
) -> np.ndarray:
    """
    Computes the current of an interacting system with the meir wingreen formula
    from the right contact.
    """

    g_retarded = forward_pass_right_connected(System_matrix)
    _, Self_energy_lesser_boundary = forward_pass_lesser_greater_right_connected(
        System_matrix, Self_energy_lesser, g_retarded)
    _, Self_energy_greater_boundary = forward_pass_lesser_greater_right_connected(
        System_matrix, Self_energy_greater, g_retarded)

    # boundary self energy not compute in forward pass
    # given by the open boundary condition
    Gamma_right = 1j * (OBCs["right"] - OBCs["right"].conj().T)
    Self_energy_lesser_boundary[-1, -1] = 1j * \
        fermi_distribution_right * Gamma_right
    Self_energy_greater_boundary[-1, -1] = 1j * \
        (fermi_distribution_right - 1) * Gamma_right

    number_of_blocks = System_matrix.bshape[0]
    current = np.zeros(
        (number_of_blocks), dtype=System_matrix.dtype)
    # calculate the current
    for i in range(number_of_blocks):
        current[i] = np.real(np.trace(Self_energy_greater_boundary[i, i] @
                                      G_lesser[i, i] -
                                      G_greater[i, i] @
                                      Self_energy_lesser_boundary[i, i]))

    return current


def forward_pass_right_connected(
    System_matrix: bsparse.BSparse
) -> bsparse.BSparse:

    number_of_blocks = System_matrix.bshape[0]
    g_retarded = bsparse.diag([np.zeros((blocksize, blocksize))
                               for blocksize in range(number_of_blocks)])

    # forward pass for retarded RGF (right connected)
    g_retarded[-1, -1] = np.linalg.inv(System_matrix[-1:, -1:])
    for i in range(number_of_blocks - 2, -1, -1):
        g_retarded[i, i] = np.linalg.inv(System_matrix[i,  i] -
                                         System_matrix[i, i+1] @
                                         g_retarded[i+1, i+1] @
                                         System_matrix[i+1,  i])
    return g_retarded


def forward_pass_lesser_greater_right_connected(
    System_matrix: bsparse.BSparse,
    Self_energy_lesser_greater: bsparse.BSparse,
    g_retarded: bsparse.BSparse
) -> bsparse.BSparse:
    number_of_blocks = System_matrix.bshape[0]

    g_lesser_greater = bsparse.diag([np.zeros((blocksize, blocksize))
                                     for blocksize in range(number_of_blocks)])

    Self_energy_lesser_greater_boundary = bsparse.diag([np.zeros((blocksize, blocksize))
                                                        for blocksize in range(number_of_blocks)])

    g_lesser_greater[-1, -1] = g_retarded[-1, -1] @\
        Self_energy_lesser_greater[-1:, -1:] @\
        g_retarded[-1, -1].T.conj()

    for i in range(number_of_blocks - 2, -1, -1):
        A_lesser_greater = System_matrix[i, i+1] @\
            g_retarded[i+1, i+1] @\
            Self_energy_lesser_greater[i+1,  i]

        Self_energy_lesser_greater_boundary[i, i] = System_matrix[i, i+1] @\
            g_lesser_greater[i+1, i+1] @\
            System_matrix[i, i+1].T.conj() -\
            (A_lesser_greater - A_lesser_greater.T.conj())

        g_lesser_greater[i, i] = g_retarded[i, i] @\
            (Self_energy_lesser_greater[i,  i] +
             Self_energy_lesser_greater_boundary[i, i]) @\
            g_retarded[i, i].T.conj()

    return g_lesser_greater, Self_energy_lesser_greater_boundary
