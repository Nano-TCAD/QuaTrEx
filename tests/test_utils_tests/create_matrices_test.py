# Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.

import numpy as np
import pytest
from scipy import sparse

from quatrex.test_utils import create_matrices, assert_properties

@pytest.mark.parametrize(
    "matrix_size",
    [(1),
     (21),
     (75),
     ]
)
@pytest.mark.parametrize(
    "density",
    [(0.5),
     (0.01),
     ]
)
def test_create_indices_to_keep(
    matrix_size: int,
    density: float
):
    row_indices_to_keep, col_indices_to_keep = \
        create_matrices.create_indices_to_keep(matrix_size, density)
    data = np.ones_like(row_indices_to_keep)
    A = sparse.coo_matrix((data, (row_indices_to_keep, col_indices_to_keep)),
                         shape=(matrix_size, matrix_size))
    assert np.allclose(A.toarray(), A.toarray().T)


@pytest.mark.parametrize(
    "blocksize",
    [(1),
     (21),
     (75),
     ]
)
def test_create_invertible_block(
    blocksize: int
):
    A = create_matrices.create_invertible_block(blocksize)
    assert A.shape == (blocksize, blocksize)
    A_inverse = np.linalg.inv(A)
    assert np.linalg.norm(A_inverse @ A - np.eye(blocksize)
                          ) / np.linalg.norm(A) < 1e-13
    assert np.allclose(A_inverse @ A, np.eye(blocksize))


@pytest.mark.parametrize(
    "number_of_blocks",
    [(1),
     (2),
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
def test_create_periodic_matrix(
        number_of_blocks: int,
        blocksize: int
):
    matrix_size = number_of_blocks * blocksize
    A = create_matrices.create_periodic_matrix(number_of_blocks, blocksize)
    assert A.shape == (matrix_size, matrix_size)

    assert_properties.assert_block_tridiagonal(A, blocksize)
    assert_properties.assert_periodic(A, blocksize)
    assert_properties.assert_invertible(A)


@pytest.mark.parametrize(
    "number_of_blocks",
    [(1),
     (2),
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
def test_create_tridiagonal_matrix(
        number_of_blocks: int,
        blocksize: int
):
    A = create_matrices.create_tridiagonal_matrix(number_of_blocks, blocksize)
    assert_properties.assert_block_tridiagonal(A, blocksize)
    assert_properties.assert_invertible(A)
