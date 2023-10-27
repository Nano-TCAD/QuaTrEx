# Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.

import numpy as np
import pytest

from quatrex.refactored_solvers.open_boundary_conditions import apply_obc_to_system_matrix


@pytest.mark.parametrize(
    "number_of_blocks",
    [(2),
     (4),
     ]
)
@pytest.mark.parametrize(
    "blocksize",
    [(1),
     (20),
     (73),
     ]
)
def test_apply_obc_to_system_matrix(
    number_of_blocks: int,
    blocksize: int
):
    rng = np.random.default_rng()
    matrix_size = number_of_blocks * blocksize
    A = rng.random((matrix_size, matrix_size)) + 1j * \
        rng.random((matrix_size, matrix_size))
    OBC_left = rng.random((blocksize, blocksize)) + 1j * \
        rng.random((blocksize, blocksize))
    OBC_right = rng.random((blocksize, blocksize)) + 1j * \
        rng.random((blocksize, blocksize))

    OBCs = {"left": OBC_left, "right": OBC_right}

    A_copy = A.copy()

    apply_obc_to_system_matrix(A, OBCs, blocksize)
    # assert that only the boundary blocks have changed
    assert np.allclose(A[:blocksize, :blocksize],
                       A_copy[:blocksize, :blocksize] - OBC_left)
    assert np.allclose(A[-blocksize:, -blocksize:],
                       A_copy[-blocksize:, -blocksize:] - OBC_right)
    for i in range(number_of_blocks):
        for j in range(number_of_blocks):
            if (i == 0 and j == 0) or (i == number_of_blocks - 1 and j == number_of_blocks - 1):
                continue
            else:
                assert np.allclose(A[i*blocksize:(i+1)*blocksize, j*blocksize:(j+1)*blocksize],
                                   A_copy[i*blocksize:(i+1)*blocksize, j*blocksize:(j+1)*blocksize])
