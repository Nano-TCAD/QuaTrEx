# Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.

import numpy as np

def create_invertible_block(
    blocksize: int
):
    rng = np.random.default_rng()
    A = rng.uniform(size=(blocksize, blocksize)) + 1j * rng.uniform(size=(blocksize, blocksize))
    value_diag = np.sum(np.abs(A), axis=1)
    np.fill_diagonal(A, value_diag)
    return A

def create_periodic_matrix(
    number_of_blocks: int,
    blocksize: int
):
    matrix_size = number_of_blocks * blocksize
    A = np.zeros((matrix_size, matrix_size), dtype=np.complex128)
    A_diag = create_invertible_block(blocksize)
    A_upper = create_invertible_block(blocksize)
    A_lower = create_invertible_block(blocksize)
    for i in range(number_of_blocks):
        i_ = slice(i*blocksize, (i+1)*blocksize)
        A[i_, i_] = A_diag
    for i in range(number_of_blocks-1):
        i_ = slice(i*blocksize, (i+1)*blocksize)
        i_plus_one_ = slice((i+1)*blocksize, (i+2)*blocksize)
        A[i_, i_plus_one_] = A_upper
        A[i_plus_one_, i_] = A_lower
    return A

def create_tridiagonal_matrix(
        number_of_blocks: int,
        blocksize: int
):
    matrix_size = number_of_blocks * blocksize
    A = np.zeros((matrix_size, matrix_size), dtype=np.complex128)
    for i in range(number_of_blocks):
        i_ = slice(i*blocksize, (i+1)*blocksize)
        A[i_, i_] = create_invertible_block(blocksize)
    for i in range(number_of_blocks-1):
        i_ = slice(i*blocksize, (i+1)*blocksize)
        i_plus_one_ = slice((i+1)*blocksize, (i+2)*blocksize)
        A[i_, i_plus_one_] = np.random.rand(blocksize, blocksize) + 1j * np.random.rand(blocksize, blocksize)
        A[i_plus_one_, i_] = np.random.rand(blocksize, blocksize) + 1j * np.random.rand(blocksize, blocksize)
    return A
