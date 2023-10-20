# Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.

import numpy as np
from scipy.sparse import csr_matrix



def csr_to_triple_array(
    A : csr_matrix,
    blocksize : int,
) -> [np.ndarray, np.ndarray, np.ndarray]:
    
    number_of_blocks = int(A.shape[0] / blocksize)
    
    A_diag_blocks = np.zeros((number_of_blocks, blocksize, blocksize), dtype=A.dtype)
    A_upper_blocks = np.zeros((number_of_blocks-1, blocksize, blocksize), dtype=A.dtype)
    
    for j in range(number_of_blocks):
        A_diag_blocks[j] = A[j * blocksize : (j + 1) * blocksize, j * blocksize : (j + 1) * blocksize]
    for j in range(number_of_blocks-1):
        A_upper_blocks[j] = A[j * blocksize : (j + 1) * blocksize, (j + 1) * blocksize : (j + 2) * blocksize]
        
    return A_diag_blocks, A_upper_blocks
