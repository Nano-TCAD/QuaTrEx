# Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.

import numpy as np
import pytest

from quatrex.refactored_solvers.greens_function_solver import symmetrize_self_energy, compute_greens_function
from quatrex.test_utils import create_matrices
from quatrex.test_utils.matrix_modification import cut_to_tridiag


@pytest.mark.parametrize(
    "number_of_blocks",
    [(2),
     (5),
     ]
)
@pytest.mark.parametrize(
    "blocksize",
    [(1),
     (20),
     (73),
     ]
)
def test_compute_greens_function_lesser_and_greater(
    number_of_blocks: int,
    blocksize: int
):
    """
    Test the fundamental formula for the lesser and greater Green's function
    """
    rng = np.random.default_rng()
    matrix_size = number_of_blocks * blocksize
    A = rng.random((matrix_size, matrix_size)) + 1j * \
        rng.random((matrix_size, matrix_size))
    B = create_matrices.create_tridiagonal_matrix(number_of_blocks, blocksize)
    C_reference = A @ B @ A.conj().T
    cut_to_tridiag(C_reference, blocksize)
    C_test = compute_greens_function(A, B)
    assert np.allclose(C_test, C_reference)


@pytest.mark.parametrize(
    "number_of_blocks",
    [(2),
     (5),
     ]
)
@pytest.mark.parametrize(
    "blocksize",
    [(1),
     (20),
     (73),
     ]
)
def test_symmetrize_self_energy(
    number_of_blocks: int,
    blocksize: int
):
    """
    Test the correct skewed hermitian symmetrization of the self energy
    """
    A_list = [create_matrices.create_tridiagonal_matrix(
        number_of_blocks, blocksize)]
    B_list = [create_matrices.create_tridiagonal_matrix(
        number_of_blocks, blocksize)]
    C_list = [create_matrices.create_tridiagonal_matrix(
        number_of_blocks, blocksize)]

    A_sym_list, B_sym_list, C_sym_list = \
        symmetrize_self_energy(A_list, B_list, C_list)

    assert np.allclose(C_sym_list[0], -C_sym_list[0].conj().T)
    assert np.allclose(B_sym_list[0], -B_sym_list[0].conj().T)
    assert np.allclose(A_sym_list[0], np.real(
        A_list[0]) + (C_sym_list[0] - B_sym_list[0])/2)
