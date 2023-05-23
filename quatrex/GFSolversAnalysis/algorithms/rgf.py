import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import inv
import time



def rgf(A_bloc_diag, A_bloc_upper, A_bloc_lower, reverse : bool = False):
    """
        Block-tridiagonal selected inversion using RGF algorithm.
    """
    nblocks = A_bloc_diag.shape[0]
    blockSize = A_bloc_diag.shape[1]

    # Storage for the incomplete forward substitution
    g_diag_left_to_right  = np.zeros((nblocks, blockSize, blockSize))
    #g_diag_right_to_left  = np.zeros((nblocks, blockSize, blockSize))

    # Storage for the full backward substitution 
    G_diag_left_to_right  = np.zeros((nblocks, blockSize, blockSize))
    #G_diag_right_to_left  = np.zeros((nblocks, blockSize, blockSize))

    G_upper_1 = np.zeros((nblocks-1, blockSize, blockSize))
    #G_upper_2 = np.zeros((nblocks-1, blockSize, blockSize))

    G_lower_1 = np.zeros((nblocks-1, blockSize, blockSize))
    #G_lower_2 = np.zeros((nblocks-1, blockSize, blockSize))


    tic = time.perf_counter() # -----------------------------
    # 1. Initialisation of g
    g_diag_left_to_right[0, ] = np.linalg.inv(A_bloc_diag[0, ])
    #g_diag_right_to_left[-1, ] = np.linalg.inv(A_bloc_diag[-1, ])


    # 2. Forward substitution
    # From left to right
    for i in range(1, nblocks):
        g_diag_left_to_right[i, ] = np.linalg.inv(A_bloc_diag[i, ] - A_bloc_lower[i-1, ] @ g_diag_left_to_right[i-1, ] @ A_bloc_upper[i-1, ])

    # From right to left
    #for i in range(nblocks-2, -1, -1):
    #    g_diag_right_to_left[i, ] = np.linalg.inv(A_bloc_diag[i, ] - A_bloc_upper[i, ] @ g_diag_right_to_left[i+1, ] @ A_bloc_lower[i, ])


    # 3. Initialisation of last element of G
    G_diag_left_to_right[-1, ] = g_diag_left_to_right[-1, ]
    #G_diag_right_to_left[0, ]  = g_diag_right_to_left[0, ]


    # 4. Backward substitution
    # From right to left
    for i in range(nblocks-2, -1, -1): 
        G_diag_left_to_right[i, ] =  g_diag_left_to_right[i, ] @ (np.identity(blockSize) + A_bloc_upper[i, ] @ G_diag_left_to_right[i+1, ] @ A_bloc_lower[i, ] @ g_diag_left_to_right[i, ])
        G_upper_1[i, ]            = -g_diag_left_to_right[i, ] @ A_bloc_upper[i, ] @ G_diag_left_to_right[i+1, ]
        G_lower_1[i, ]            = G_upper_1[i, ].T

    # From left to right
    #for i in range(1, nblocks):
    #    G_diag_right_to_left[i, ] =  g_diag_right_to_left[i, ] @ (np.identity(blockSize) + A_bloc_lower[i-1, ] @ G_diag_right_to_left[i-1, ] @ A_bloc_upper[i-1, ] @ g_diag_right_to_left[i, ])
    #    G_lower_2[i-1, ]          = -g_diag_right_to_left[i, ] @ A_bloc_lower[i-1, ] @ G_diag_right_to_left[i-1, ]
    #    G_upper_2[i-1, ]          = G_lower_2[i-1, ].T
    toc = time.perf_counter() # -----------------------------


    """ if np.allclose(G_diag_left_to_right, G_diag_right_to_left):
        print("RGF: G_diag_left_to_right and G_diag_right_to_left are close")

    if np.allclose(G_upper_1, G_upper_2):
        print("RGF: G_upper_1 and G_upper_2 are close")

    if np.allclose(G_lower_1, G_lower_2):
        print("RGF: G_lower_1 and G_lower_2 are close") """

    print(f"RGF: Inversion took {toc - tic:0.4f} seconds")
    return G_diag_left_to_right, G_upper_1, G_lower_1

