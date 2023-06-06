"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@author: Anders Winka (awinka@iis.ee.ethz.ch)
@date: 2023-05

Based on initial idea and work from: Anders Winka

@reference: https://doi.org/10.1007/s10825-013-0458-7

Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.
"""

import numpy as np
import time

from scipy.sparse import csc_matrix
from scipy.sparse.linalg import inv
from mpi4py import MPI



def rgf_leftprocess(A_bloc_diag_leftprocess, A_bloc_upper_leftprocess, A_bloc_lower_leftprocess):
    """
        Left process of the 2-sided RGF algorithm.
            - Array traversal is done from left to right
    """
    comm = MPI.COMM_WORLD

    nblocks   = A_bloc_diag_leftprocess.shape[0]
    blockSize = A_bloc_diag_leftprocess.shape[1]

    g_diag_leftprocess = np.zeros((nblocks+1, blockSize, blockSize), dtype=A_bloc_diag_leftprocess.dtype)
    G_diag_blocks_leftprocess  = np.zeros((nblocks, blockSize, blockSize), dtype=A_bloc_diag_leftprocess.dtype)
    G_upper_blocks_leftprocess = np.zeros((nblocks, blockSize, blockSize), dtype=A_bloc_upper_leftprocess.dtype)
    G_lower_blocks_leftprocess = np.zeros((nblocks, blockSize, blockSize), dtype=A_bloc_lower_leftprocess.dtype)


    # Initialisation of g
    g_diag_leftprocess[0, ] = np.linalg.inv(A_bloc_diag_leftprocess[0, ])

    # Forward substitution
    for i in range(1, nblocks):
        g_diag_leftprocess[i, ] = np.linalg.inv(A_bloc_diag_leftprocess[i, ]\
                                                - A_bloc_lower_leftprocess[i-1, ] @ g_diag_leftprocess[i-1, ] @ A_bloc_upper_leftprocess[i-1, ])

    # Communicate the left connected block and receive the right connected block
    comm.send(g_diag_leftprocess[nblocks-1, ], dest=1, tag=0)
    comm.barrier()
    g_diag_leftprocess[nblocks, ] = comm.recv(source=1, tag=0)

    # Connection from both sides of the full G
    G_diag_blocks_leftprocess[nblocks-1, ] = np.linalg.inv(A_bloc_diag_leftprocess[nblocks-1, ]\
                                                         - A_bloc_lower_leftprocess[nblocks-2, ] @ g_diag_leftprocess[nblocks-2, ] @ A_bloc_upper_leftprocess[nblocks-2, ]\
                                                         - A_bloc_upper_leftprocess[nblocks-1, ] @ g_diag_leftprocess[nblocks, ] @ A_bloc_lower_leftprocess[nblocks-1, ])

    # Compute the shared off-diagonal upper block
    G_upper_blocks_leftprocess[nblocks-1, ] = -G_diag_blocks_leftprocess[nblocks-1, ] @ A_bloc_upper_leftprocess[nblocks-1, ] @ g_diag_leftprocess[nblocks, ]
    G_lower_blocks_leftprocess[nblocks-1, ] = G_upper_blocks_leftprocess[nblocks-1, ].T

    # Backward substitution
    for i in range(nblocks-2, -1, -1):
        G_diag_blocks_leftprocess[i, ]  = g_diag_leftprocess[i, ] @ (np.identity(blockSize) + A_bloc_upper_leftprocess[i, ] @ G_diag_blocks_leftprocess[i+1, ] @ A_bloc_lower_leftprocess[i, ] @ g_diag_leftprocess[i, ])
        G_upper_blocks_leftprocess[i, ] = -g_diag_leftprocess[i, ] @ A_bloc_upper_leftprocess[i, ] @ G_diag_blocks_leftprocess[i+1, ]
        G_lower_blocks_leftprocess[i, ] =  G_upper_blocks_leftprocess[i, ].T


    return G_diag_blocks_leftprocess, G_upper_blocks_leftprocess, G_lower_blocks_leftprocess



def rgf_rightprocess(A_bloc_diag_rightprocess, A_bloc_upper_rightprocess, A_bloc_lower_rightprocess):
    """
        Right process of the 2-sided RGF algorithm.
            - Array traversal is done from right to left
    """
    comm = MPI.COMM_WORLD

    nblocks   = A_bloc_diag_rightprocess.shape[0]
    blockSize = A_bloc_diag_rightprocess.shape[1]

    g_diag_rightprocess = np.zeros((nblocks+1, blockSize, blockSize), dtype=A_bloc_diag_rightprocess.dtype)
    G_diag_blocks_rightprocess  = np.zeros((nblocks, blockSize, blockSize), dtype=A_bloc_diag_rightprocess.dtype)
    G_upper_blocks_rightprocess = np.zeros((nblocks, blockSize, blockSize), dtype=A_bloc_upper_rightprocess.dtype)
    G_lower_blocks_rightprocess = np.zeros((nblocks, blockSize, blockSize), dtype=A_bloc_lower_rightprocess.dtype)


    # Initialisation of g
    g_diag_rightprocess[-1, ] = np.linalg.inv(A_bloc_diag_rightprocess[-1, ])

    # Forward substitution
    for i in range(nblocks-1, 0, -1):
        g_diag_rightprocess[i, ] = np.linalg.inv(A_bloc_diag_rightprocess[i-1, ]\
                                                 - A_bloc_upper_rightprocess[i, ] @ g_diag_rightprocess[i+1, ] @ A_bloc_lower_rightprocess[i, ])

    # Communicate the right connected block and receive the left connected block
    comm.barrier()
    g_diag_rightprocess[0, ] = comm.recv(source=0, tag=0)
    comm.send(g_diag_rightprocess[1, ], dest=0, tag=0)

    # Connection from both sides of the full G
    G_diag_blocks_rightprocess[0, ] = np.linalg.inv(A_bloc_diag_rightprocess[0, ]\
                                             - A_bloc_lower_rightprocess[0, ] @ g_diag_rightprocess[0, ] @ A_bloc_upper_rightprocess[0, ]\
                                             - A_bloc_upper_rightprocess[1, ] @ g_diag_rightprocess[2, ] @ A_bloc_lower_rightprocess[1, ])

    # Compute the shared off-diagonal upper block
    G_upper_blocks_rightprocess[0, ] = g_diag_rightprocess[0, ] @ A_bloc_upper_rightprocess[0, ] @ G_diag_blocks_rightprocess[0, ]
    G_lower_blocks_rightprocess[0, ] = G_upper_blocks_rightprocess[0, ].T

    # Backward substitution
    for i in range(1, nblocks):
        G_diag_blocks_rightprocess[i, ]  = g_diag_rightprocess[i+1, ] @ (np.identity(blockSize) + A_bloc_lower_rightprocess[i, ] @ G_diag_blocks_rightprocess[i-1, ] @ A_bloc_upper_rightprocess[i, ] @ g_diag_rightprocess[i+1, ])
        G_lower_blocks_rightprocess[i, ] = -g_diag_rightprocess[i+1, ] @ A_bloc_lower_rightprocess[i, ] @ G_diag_blocks_rightprocess[i-1, ]
        G_upper_blocks_rightprocess[i, ] = G_lower_blocks_rightprocess[i, ].T


    return G_diag_blocks_rightprocess, G_upper_blocks_rightprocess, G_lower_blocks_rightprocess



def rgf2sided_Gr(A_bloc_diag, A_bloc_upper, A_bloc_lower):
    """
        Block-tridiagonal selected inversion using 2-sided RGF algorithm.
            - Using MPI for multiprocessing
            - Rank 0 will agregate the final result
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    nblocks   = A_bloc_diag.shape[0]
    nblocks_2 = int(nblocks/2)
    blockSize = A_bloc_diag.shape[1]

    G_diag_blocks = np.zeros((nblocks, blockSize, blockSize), dtype=A_bloc_diag.dtype)
    G_upper_blocks = np.zeros((nblocks-1, blockSize, blockSize), dtype=A_bloc_upper.dtype)
    G_lower_blocks = np.zeros((nblocks-1, blockSize, blockSize), dtype=A_bloc_lower.dtype)


    tic = time.perf_counter() # -----------------------------
    if rank == 0:
        G_diag_blocks[0:nblocks_2, ]\
        , G_upper_blocks[0:nblocks_2, ]\
        , G_lower_blocks[0:nblocks_2, ] = rgf_leftprocess(A_bloc_diag[0:nblocks_2, ], A_bloc_upper[0:nblocks_2, ], A_bloc_lower[0:nblocks_2, ])
        
        comm.barrier()

        G_diag_blocks[nblocks_2:, ]  = comm.recv(source=1, tag=0)
        G_upper_blocks[nblocks_2:, ] = comm.recv(source=1, tag=1)
        G_lower_blocks[nblocks_2:, ] = comm.recv(source=1, tag=2)

    elif rank == 1:
        G_diag_blocks[nblocks_2:, ]\
        , G_upper_blocks[nblocks_2-1:, ]\
        , G_lower_blocks[nblocks_2-1:, ] = rgf_rightprocess(A_bloc_diag[nblocks_2:, ], A_bloc_upper[nblocks_2-1:, ], A_bloc_lower[nblocks_2-1:, ])
        
        comm.send(G_diag_blocks[nblocks_2:, ], dest=0, tag=0)
        comm.send(G_upper_blocks[nblocks_2:, ], dest=0, tag=1)
        comm.send(G_lower_blocks[nblocks_2:, ], dest=0, tag=2)

        comm.barrier()
    
    comm.barrier()
    toc = time.perf_counter() # -----------------------------


    timing = toc - tic

    return G_diag_blocks, G_upper_blocks, G_lower_blocks, timing

