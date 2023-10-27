# Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.

import numpy as np

def assert_block_tridiagonal(
        A: np.ndarray,
        blocksize: int
):
    number_of_blocks = int(A.shape[0] / blocksize)
    zero_block = np.zeros((blocksize, blocksize))
    for row in range(number_of_blocks):
        for col in range(number_of_blocks):
            if col < row - 1 or col > row + 1:
                assert np.allclose(A[row*blocksize:(row+1)*blocksize, col*blocksize:(col+1)*blocksize], zero_block)

def assert_block_diagonal(
    A: np.ndarray,
    blocksize: int
):
    number_of_blocks = int(A.shape[0] / blocksize)
    zero_block = np.zeros((blocksize, blocksize))
    for row in range(number_of_blocks):
        for col in range(number_of_blocks):
            if col != row:
                assert np.allclose(A[row*blocksize:(row+1)*blocksize, col*blocksize:(col+1)*blocksize], zero_block)

def assert_periodic(
    A: np.ndarray,
    blocksize: int
):
    number_of_blocks = int(A.shape[0] / blocksize)
    for i in range(number_of_blocks):
        i_ = slice(i*blocksize, (i+1)*blocksize)
        assert np.allclose(A[i_, i_], A[:blocksize, :blocksize])
    for i in range(number_of_blocks-1):
        i_ = slice(i*blocksize, (i+1)*blocksize)
        i_plus_one_ = slice((i+1)*blocksize, (i+2)*blocksize)
        assert np.allclose(A[i_, i_plus_one_], A[:blocksize, blocksize:2*blocksize])
        assert np.allclose(A[i_plus_one_, i_], A[blocksize:2*blocksize, :blocksize])

def assert_invertible(
        A: np.ndarray
):
    matrix_size = A.shape[0]
    A_inverse = np.linalg.inv(A)
    assert np.linalg.norm(A_inverse @ A - np.eye(matrix_size)) / np.linalg.norm(A) < 1e-13
