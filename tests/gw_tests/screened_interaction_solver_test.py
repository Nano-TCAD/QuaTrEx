# Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.

import pytest
import numpy as np
from quatrex.refactored_solvers import screened_interaction_solver
from quatrex.test_utils import create_matrices


@pytest.mark.parametrize(
    "number_of_blocks",
    [(2),
     (3),
     ]
)
@pytest.mark.parametrize(
    "blocksize",
    [(1),
     (20),
     (73),
     ]
)
def test_get_system_matrix(
    number_of_blocks: int,
    blocksize: int
):
    number_of_blocks += 2
    matrix_size = number_of_blocks * blocksize
    A_reference = create_matrices.create_periodic_matrix(
        number_of_blocks, blocksize)
    B_reference = create_matrices.create_periodic_matrix(
        number_of_blocks, blocksize)

    C_reference = np.eye(matrix_size, dtype=np.complex128)\
        - A_reference @ B_reference

    # cut out the device
    A_test = A_reference[blocksize:-blocksize, blocksize:-blocksize]
    B_test = B_reference[blocksize:-blocksize, blocksize:-blocksize]

    C_test = screened_interaction_solver.get_system_matrix(
        A_test, B_test, blocksize)
    assert np.allclose(
        C_test, C_reference[blocksize:-blocksize, blocksize:-blocksize])


@pytest.mark.parametrize(
    "number_of_blocks",
    [(2),
     (3),
     ]
)
@pytest.mark.parametrize(
    "blocksize",
    [(1),
     (20),
     (73),
     ]
)
def test_get_L(
    number_of_blocks: int,
    blocksize: int
):
    number_of_blocks += 4
    A_reference = create_matrices.create_periodic_matrix(
        number_of_blocks, blocksize)
    B_reference = create_matrices.create_periodic_matrix(
        number_of_blocks, blocksize)

    C_reference = A_reference @ B_reference @ A_reference.conj().T

    # cut out the device
    A_test = A_reference[2*blocksize:-2*blocksize, 2*blocksize:-2*blocksize]
    B_test = B_reference[2*blocksize:-2*blocksize, 2*blocksize:-2*blocksize]

    C_test = screened_interaction_solver.get_L(A_test, B_test, blocksize)
    assert np.allclose(C_test,
                       C_reference[2*blocksize:-2*blocksize, 2*blocksize:-2*blocksize])


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
def test_symmetrize_polarization(
    number_of_blocks: int,
    blocksize: int
):
    C_list = [create_matrices.create_tridiagonal_matrix(number_of_blocks, blocksize)]
    B_list = [create_matrices.create_tridiagonal_matrix(number_of_blocks, blocksize)]
    A_list , B_skewed_list , C_skewed_list = \
        screened_interaction_solver.symmetrize_polarization(C_list, B_list)
    assert np.allclose(C_skewed_list[0], -C_skewed_list[0].conj().T)
    assert np.allclose(B_skewed_list[0], -B_skewed_list[0].conj().T)
    assert np.allclose(A_list[0], (C_skewed_list[0] - B_skewed_list[0])/2 )
    assert np.allclose(A_list[0], -A_list[0].conj().T)

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
def test_compute_screened_interaction(
    number_of_blocks: int,
    blocksize: int
):
    A = create_matrices.create_tridiagonal_matrix(number_of_blocks, blocksize)
    B = create_matrices.create_tridiagonal_matrix(number_of_blocks, blocksize)
    C_reference = A @ B @ A.conj().T
    C_test = screened_interaction_solver.compute_screened_interaction(A, B)
    assert np.allclose(C_test, C_reference)



@pytest.mark.parametrize(
    "number_of_blocks",
    [(5),
     ]
)
@pytest.mark.parametrize(
    "blocksize",
    [(1),
     ]
)
def test_update_blocksize(
    number_of_blocks: int,
    blocksize: int
):
    # TODO: implement this test correctly
    # since update_blocksize is not yet implemented correctly

    # use sparse blocks instead of dense blocks to test functionality
    A = create_matrices.create_tridiagonal_matrix(number_of_blocks, blocksize)
    B = create_matrices.create_tridiagonal_matrix(number_of_blocks, blocksize)
    C = np.eye(A.shape[0], dtype=np.complex128) - A @ B
    blocksize_test = screened_interaction_solver.update_blocksize(C, blocksize)

    # tiling is possible
    assert A.shape[0] % blocksize_test == 0
    # assert that there are no elements outside the tridiagonal blocks
    for i in range(number_of_blocks):
        for j in range(number_of_blocks):
            if abs(i-j) > 1:
                assert np.allclose(A[i*blocksize_test:(i+1)*blocksize_test,
                                     j*blocksize_test:(j+1)*blocksize_test], 0)

