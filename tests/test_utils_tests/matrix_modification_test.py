# Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.

import numpy as np
import pytest

from quatrex.test_utils import matrix_modification, assert_properties


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
     (53),
     ]
)
def test_cut_to_diag(
        number_of_blocks: int,
        blocksize: int
):
    matrix_size = number_of_blocks * blocksize
    rng = np.random.default_rng()
    A = rng.uniform(size=(matrix_size, matrix_size)) + 1j *\
        rng.uniform(size=(matrix_size, matrix_size))
    matrix_modification.cut_to_diag(A, blocksize)
    assert_properties.assert_block_diagonal(A, blocksize)


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
     (53),
     ]
)
def test_cut_to_tridiag(
        number_of_blocks: int,
        blocksize: int
):
    matrix_size = number_of_blocks * blocksize
    rng = np.random.default_rng()
    A = rng.uniform(size=(matrix_size, matrix_size)) + 1j * \
        rng.uniform(size=(matrix_size, matrix_size))
    matrix_modification.cut_to_tridiag(A, blocksize)
    assert_properties.assert_block_tridiagonal(A, blocksize)


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
     (53),
     ]
)
def test_cut_to_upper_half(
        number_of_blocks: int,
        blocksize: int
):
    matrix_size = number_of_blocks * blocksize
    rng = np.random.default_rng()
    A = rng.uniform(size=(matrix_size, matrix_size)) + 1j * \
        rng.uniform(size=(matrix_size, matrix_size))
    matrix_modification.cut_to_upper_half(A, blocksize)
    zero_block = np.zeros((blocksize, blocksize))
    for i in range(number_of_blocks):
        for j in range(number_of_blocks):
            if i > j:
                assert np.allclose(
                    A[i*blocksize:(i+1)*blocksize, j*blocksize:(j+1)*blocksize], zero_block)


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
     (53),
     ]
)
def test_cut_to_lower_half(
        number_of_blocks: int,
        blocksize: int
):
    matrix_size = number_of_blocks * blocksize
    rng = np.random.default_rng()
    A = rng.uniform(size=(matrix_size, matrix_size)) + 1j * \
        rng.uniform(size=(matrix_size, matrix_size))
    matrix_modification.cut_to_lower_half(A, blocksize)
    zero_block = np.zeros((blocksize, blocksize))
    for i in range(number_of_blocks):
        for j in range(number_of_blocks):
            if i < j:
                assert np.allclose(
                    A[i*blocksize:(i+1)*blocksize, j*blocksize:(j+1)*blocksize], zero_block)
