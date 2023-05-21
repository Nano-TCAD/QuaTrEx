import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import inv
import time

from mpi4py import MPI



def rgf_leftprocess(A_bloc_diag_leftprocess, A_bloc_upper_leftprocess, A_bloc_lower_leftprocess):
    """
        Left process of the 2-sided RGF algorithm.
            - Array traversal is done from left to right
    """
    comm = MPI.COMM_WORLD

    nblocks   = A_bloc_diag_leftprocess.shape[0]
    blockSize = A_bloc_diag_leftprocess.shape[1]

    g_diag_leftprocess = np.zeros((nblocks+1, blockSize, blockSize))
    G_diag_leftprocess = np.zeros((nblocks+1, blockSize, blockSize))

    # Initialisation of g
    g_diag_leftprocess[0, ] = np.linalg.inv(A_bloc_diag_leftprocess[0, ])

    # Forward substitution
    for i in range(1, nblocks):
        g_diag_leftprocess[i, ] = np.linalg.inv(A_bloc_diag_leftprocess[i, ] - A_bloc_lower_leftprocess[i-1, ] @ g_diag_leftprocess[i-1, ] @ A_bloc_upper_leftprocess[i-1, ])

    # Communicate the left connected block and receive the right connected block
    comm.send(g_diag_leftprocess[nblocks-1, ], dest=1, tag=0)
    comm.barrier()
    g_diag_leftprocess[nblocks, ] = comm.recv(source=1, tag=0)

    # Initialisation of last G
    G_diag_leftprocess[nblocks, ] = g_diag_leftprocess[nblocks, ]

    # Backward substitution
    for i in range(nblocks, 0, -1):
        G_diag_leftprocess[i-1, ]  =  g_diag_leftprocess[i-1, ] @ (np.identity(blockSize) + A_bloc_upper_leftprocess[i-1, ] @ G_diag_leftprocess[i, ] @ A_bloc_lower_leftprocess[i-1, ] @ g_diag_leftprocess[i-1, ])

    return G_diag_leftprocess[:nblocks, ]



def rgf_rightprocess(A_bloc_diag_rightprocess, A_bloc_upper_rightprocess, A_bloc_lower_rightprocess):
    """
        Right process of the 2-sided RGF algorithm.
            - Array traversal is done from right to left
    """
    comm = MPI.COMM_WORLD

    nblocks   = A_bloc_diag_rightprocess.shape[0]
    blockSize = A_bloc_diag_rightprocess.shape[1]

    g_diag_rightprocess = np.zeros((nblocks+1, blockSize, blockSize))
    G_diag_rightprocess = np.zeros((nblocks+1, blockSize, blockSize))

    # Initialisation of g
    g_diag_rightprocess[nblocks, ] = np.linalg.inv(A_bloc_diag_rightprocess[-1, ])

    # Forward substitution
    for i in range(nblocks-1, 0, -1):
        g_diag_rightprocess[i, ] = np.linalg.inv(A_bloc_diag_rightprocess[i, ] - A_bloc_lower_rightprocess[i, ] @ g_diag_rightprocess[i+1, ] @ A_bloc_upper_rightprocess[i, ])

    # Communicate the left connected block and receive the right connected block
    comm.barrier()
    g_diag_rightprocess[0, ] = comm.recv(source=0, tag=0)
    comm.send(g_diag_rightprocess[1, ], dest=0, tag=0)

    # Initialisation of last G
    G_diag_rightprocess[0, ] = g_diag_rightprocess[0, ]

    # Backward substitution
    for i in range(1, nblocks+1):
        G_diag_rightprocess[i, ]  =  g_diag_rightprocess[i, ] @ (np.identity(blockSize) + A_bloc_upper_rightprocess[i-1, ] @ G_diag_rightprocess[i-1, ] @ A_bloc_lower_rightprocess[i-1, ] @ g_diag_rightprocess[i, ])

    return G_diag_rightprocess[1:nblocks+1, ]



def rgf2sided(A_bloc_diag, A_bloc_upper, A_bloc_lower):
    """
        Block-tridiagonal selected inversion using 2-sided RGF algorithm.
            - Using MPI for multiprocessing
            - Rank 0 will agregate the final result
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    nblocks   = A_bloc_diag.shape[0]
    print("nblocks: ", nblocks)
    nblocks_2 = int(nblocks/2)
    print("nblocks_2: ", nblocks_2)
    blockSize = A_bloc_diag.shape[1]

    G_diag = np.zeros((nblocks, blockSize, blockSize))

    print()

    tic = time.perf_counter() # -----------------------------
    if rank == 0:
        G_diag[0:nblocks_2, ] = rgf_leftprocess(A_bloc_diag[0:nblocks_2, ], A_bloc_upper[0:nblocks_2, ], A_bloc_lower[0:nblocks_2, ])
        #print("Process 0 own: G_diag[:nblocks_2, ]", G_diag)
        comm.barrier()
        G_diag[nblocks_2:, ] = comm.recv(source=1, tag=0)
        #print("Process 0 recv: G_diag[nblocks_2:, ]", G_diag[nblocks_2:, ])
        #print("Process 0 own: G_diag[:nblocks_2, ]", G_diag)


    elif rank == 1:
        G_diag[nblocks_2:, ] = rgf_rightprocess(A_bloc_diag[nblocks_2:, ], A_bloc_upper[nblocks_2-1:, ], A_bloc_lower[nblocks_2-1:, ])
        #print("Process 1 own: G_diag[:nblocks_2, ]", G_diag)
        #print("Process 1 send: G_diag[nblocks_2:, ]", G_diag[nblocks_2:, ])
        comm.send(G_diag[nblocks_2:, ], dest=0, tag=0)
        comm.barrier()
    
    comm.barrier()
    toc = time.perf_counter() # -----------------------------

    if rank == 0:
        print(f"RGF 2-Sided: Inversion took {toc - tic:0.4f} seconds")
    return G_diag

