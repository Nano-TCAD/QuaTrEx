import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import inv
import time



def rgf_Gr(A_bloc_diag, A_bloc_upper, A_bloc_lower, rightToLeft : bool = False):
    """
        RGF algorithm performing block-tridiagonal inversion of the given matrix.
    """
    nblocks = A_bloc_diag.shape[0]
    blockSize = A_bloc_diag.shape[1]

    # Storage for the incomplete forward substitution
    g_diag_blocks  = np.zeros((nblocks, blockSize, blockSize), dtype=A_bloc_diag.dtype)

    # Storage for the full backward substitution 
    G_diag_blocks  = np.zeros((nblocks, blockSize, blockSize), dtype=A_bloc_diag.dtype)
    G_upper_blocks = np.zeros((nblocks-1, blockSize, blockSize), dtype=A_bloc_upper.dtype)
    G_lower_blocks = np.zeros((nblocks-1, blockSize, blockSize), dtype=A_bloc_lower.dtype)


    tic = time.perf_counter() # -----------------------------
    if rightToLeft:
        # 1. Initialisation of g
        g_diag_blocks[-1, ] = np.linalg.inv(A_bloc_diag[-1, ])

        # From right to left
        for i in range(nblocks-2, -1, -1):
            g_diag_blocks[i, ] = np.linalg.inv(A_bloc_diag[i, ] - A_bloc_upper[i, ] @ g_diag_blocks[i+1, ] @ A_bloc_lower[i, ])

        # 3. Initialisation of last element of G
        G_diag_blocks[0, ] = g_diag_blocks[0, ]

        # From left to right
        for i in range(1, nblocks):
            G_diag_blocks[i, ]    =  g_diag_blocks[i, ] @ (np.identity(blockSize) + A_bloc_lower[i-1, ] @ G_diag_blocks[i-1, ] @ A_bloc_upper[i-1, ] @ g_diag_blocks[i, ])
            G_lower_blocks[i-1, ] = -g_diag_blocks[i, ] @ A_bloc_lower[i-1, ] @ G_diag_blocks[i-1, ]
            G_upper_blocks[i-1, ] = G_lower_blocks[i-1, ].T
    else:
        # 1. Initialisation of g
        g_diag_blocks[0, ] = np.linalg.inv(A_bloc_diag[0, ])

        # 2. Forward substitution
        # From left to right
        for i in range(1, nblocks):
            g_diag_blocks[i, ] = np.linalg.inv(A_bloc_diag[i, ] - A_bloc_lower[i-1, ] @ g_diag_blocks[i-1, ] @ A_bloc_upper[i-1, ])

        # 3. Initialisation of last element of G
        G_diag_blocks[-1, ] = g_diag_blocks[-1, ]

        # 4. Backward substitution
        # From right to left
        for i in range(nblocks-2, -1, -1): 
            G_diag_blocks[i, ]  =  g_diag_blocks[i, ] @ (np.identity(blockSize) + A_bloc_upper[i, ] @ G_diag_blocks[i+1, ] @ A_bloc_lower[i, ] @ g_diag_blocks[i, ])
            G_upper_blocks[i, ] = -g_diag_blocks[i, ] @ A_bloc_upper[i, ] @ G_diag_blocks[i+1, ]
            G_lower_blocks[i, ] =  G_upper_blocks[i, ].T
    toc = time.perf_counter() # -----------------------------


    print(f"RGF Gr: Inversion took {toc - tic:0.4f} seconds")
    return G_diag_blocks, G_upper_blocks, G_lower_blocks

