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
        Block-diagonal selected inversion using RGF algorithm.
    """
    nblocks = A_bloc_diag.shape[0]
    blockSize = A_bloc_diag.shape[1]
    size = A_bloc_diag.shape[0] * A_bloc_diag.shape[1]

    g = np.zeros((nblocks, blockSize, blockSize))

    tic = time.perf_counter()
    # Initialisation of g
    g[0, ] = np.linalg.inv(A_bloc_diag[0, ])

    # Forward substitution
    for i in range(1, nblocks):
        g[i, ] = np.linalg.inv(A_bloc_diag[i, ] - A_bloc_lower[i-1, ] @ g[i-1, ] @ A_bloc_upper[i-1, ])

    # Backward substitution
    for i in range(nblocks-2, -1, -1): 
        g[i, ] = g[i, ] @ (np.identity(blockSize) + A_bloc_upper[i, ] @ g[i+1, ] @ A_bloc_lower[i, ] @ g[i, ])
    toc = time.perf_counter()

    print(f"RGF: Inversion took {toc - tic:0.4f} seconds")

    # Reshape g to a 2D matrix
    G = np.zeros((size, size))

    for i in range(nblocks):
        G[i*blockSize:(i+1)*blockSize, i*blockSize:(i+1)*blockSize] = g[i, ]

    # Only return the diagonal
    G_diag = np.array([G[i, i] for i in range(size)])

    return G, G_diag

