import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import inv
import time

def fullInversion(A):
    """
        Invert a matrix using:
            numpy dense matrix: numpy.linalg.inv
            scipy CSC matrix: scipy.sparse.linalg.inv
    """
    if type(A) == np.ndarray:
        tic = time.perf_counter()
        A_inv = np.linalg.inv(A)
        toc = time.perf_counter()

        print(f"Numpy: Full inversion took {toc - tic:0.4f} seconds")
        
        return A_inv

    elif type(A) == csc_matrix:
        tic = time.perf_counter()
        A_inv = inv(A)
        toc = time.perf_counter()

        print(f"Scipy CSC: Full inversion took {toc - tic:0.4f} seconds")
        
        return A_inv
    
    

def rgf(A_bloc_diag, A_bloc_upper, A_bloc_lower):
    """
        Block-tridiagonal selected inversion using RGF algorithm.
    """
    nblocks = A_bloc_diag.shape[0]
    blockSize = A_bloc_diag.shape[1]
    size = A_bloc_diag.shape[0] * A_bloc_diag.shape[1]

    # Incomplete forward substitution
    g_diag  = np.zeros((nblocks, blockSize, blockSize))

    # Full backward substitution
    G_diag  = np.zeros((nblocks, blockSize, blockSize))
    G_upper = np.zeros((nblocks-1, blockSize, blockSize))
    G_lower = np.zeros((nblocks-1, blockSize, blockSize))

    tic = time.perf_counter()
    # Initialisation of g
    g_diag[0, ] = np.linalg.inv(A_bloc_diag[0, ])

    # Forward substitution
    for i in range(1, nblocks):
        g_diag[i, ] = np.linalg.inv(A_bloc_diag[i, ] - A_bloc_lower[i-1, ] @ g_diag[i-1, ] @ A_bloc_upper[i-1, ])

    # Initialisation of last element of G
    G_diag[-1, ] = g_diag[-1, ]

    # Backward substitution
    for i in range(nblocks-2, -1, -1): 
        G_diag[i, ]  = g_diag[i, ] @ (np.identity(blockSize) + A_bloc_upper[i, ] @ G_diag[i+1, ] @ A_bloc_lower[i, ] @ g_diag[i, ])
        G_upper[i, ] = -g_diag[i, ] @ A_bloc_upper[i, ] @ G_diag[i+1, ]
        G_lower[i, ] = -g_diag[i, ] @ A_bloc_lower[i, ] @ G_diag[i+1, ]
    toc = time.perf_counter()

    print(f"RGF: Inversion took {toc - tic:0.4f} seconds")

    return G_diag, G_upper, G_lower

