import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import inv
import time

from mpi4py import MPI


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

    # Storage for the incomplete forward substitution
    g_diag  = np.zeros((nblocks, blockSize, blockSize))

    # Storage for the full backward substitution 
    G_diag  = np.zeros((nblocks, blockSize, blockSize))
    G_upper = np.zeros((nblocks-1, blockSize, blockSize))
    G_lower = np.zeros((nblocks-1, blockSize, blockSize))


    tic = time.perf_counter() # -----------------------------
    # Initialisation of g
    g_diag[0, ] = np.linalg.inv(A_bloc_diag[0, ])

    # Forward substitution
    for i in range(1, nblocks):
        g_diag[i, ] = np.linalg.inv(A_bloc_diag[i, ] - A_bloc_lower[i-1, ] @ g_diag[i-1, ] @ A_bloc_upper[i-1, ])

    # Initialisation of last element of G
    G_diag[-1, ] = g_diag[-1, ]

    # Backward substitution
    for i in range(nblocks-2, -1, -1): 
        G_diag[i, ]  =  g_diag[i, ] @ (np.identity(blockSize) + A_bloc_upper[i, ] @ G_diag[i+1, ] @ A_bloc_lower[i, ] @ g_diag[i, ])
        G_upper[i, ] = -g_diag[i, ] @ A_bloc_upper[i, ] @ G_diag[i+1, ]
        G_lower[i, ] = -g_diag[i, ] @ A_bloc_lower[i, ] @ G_diag[i+1, ]
    toc = time.perf_counter() # -----------------------------


    print(f"RGF: Inversion took {toc - tic:0.4f} seconds")
    return G_diag, G_upper, G_lower



def rgf_leftprocess(A_bloc_diag_leftprocess, A_bloc_upper_leftprocess, A_bloc_lower_leftprocess):
    """
        Left process of the 2-sided RGF algorithm.
            - Array traversal is done from left to right
    """
    comm = MPI.COMM_WORLD

    nblocks   = A_bloc_diag_leftprocess.shape[0]
    nblocks_2 = int(nblocks/2)
    blockSize = A_bloc_diag_leftprocess.shape[1]

    g_diag_leftprocess = np.zeros((nblocks_2+1, blockSize, blockSize))
    G_diag_leftprocess = np.zeros((nblocks_2, blockSize, blockSize))

    # Initialisation of g
    g_diag_leftprocess[0, ] = np.linalg.inv(A_bloc_diag_leftprocess[0, ])

    # Forward substitution
    for i in range(1, nblocks_2):
        g_diag_leftprocess[i, ] = np.linalg.inv(A_bloc_diag_leftprocess[i, ] - A_bloc_lower_leftprocess[i-1, ] @ g_diag_leftprocess[i-1, ] @ A_bloc_upper_leftprocess[i-1, ])

    # Communicate the left connected block and receive the right connected block
    comm.send(g_diag_leftprocess[nblocks_2, ], dest=1, tag=0)
    comm.barrier()
    g_diag_leftprocess[-1, ] = comm.recv(source=1, tag=0)

    # Backward substitution
    for i in range(nblocks_2-1, 1, -1):
        G_diag_leftprocess[i, ]  =  g_diag_leftprocess[i, ] @ (np.identity(blockSize) + A_bloc_upper_leftprocess[i, ] @ G_diag_leftprocess[i+1, ] @ A_bloc_lower_leftprocess[i, ] @ g_diag_leftprocess[i, ])

    return G_diag_leftprocess



def rgf_rightprocess(A_bloc_diag_rightprocess, A_bloc_upper_rightprocess, A_bloc_lower_rightprocess):
    """
        Right process of the 2-sided RGF algorithm.
            - Array traversal is done from right to left
    """
    comm = MPI.COMM_WORLD

    nblocks   = A_bloc_diag_rightprocess.shape[0]
    nblocks_2 = int(nblocks/2)
    blockSize = A_bloc_diag_rightprocess.shape[1]

    g_diag_rightprocess = np.zeros((nblocks_2+1, blockSize, blockSize))
    G_diag_rightprocess = np.zeros((nblocks_2, blockSize, blockSize))

    # Initialisation of g
    g_diag_rightprocess[-1, ] = np.linalg.inv(A_bloc_diag_rightprocess[-1, ])

    # Forward substitution
    for i in range(nblocks_2-1, -1, -1):
        g_diag_rightprocess[i, ] = np.linalg.inv(A_bloc_diag_rightprocess[i, ] - A_bloc_lower_rightprocess[i, ] @ g_diag_rightprocess[i+1, ] @ A_bloc_upper_rightprocess[i, ])

    # Communicate the left connected block and receive the right connected block
    comm.barrier()
    g_diag_rightprocess[0, ] = comm.recv(source=0, tag=0)
    comm.send(g_diag_rightprocess[1, ], dest=0, tag=0)

    # Backward substitution
    for i in range(1, nblocks_2):
        G_diag_rightprocess[i, ]  =  g_diag_rightprocess[i, ] @ (np.identity(blockSize) + A_bloc_upper_rightprocess[i, ] @ G_diag_rightprocess[i+1, ] @ A_bloc_lower_rightprocess[i, ] @ g_diag_rightprocess[i, ])

    return G_diag_rightprocess



def rgf2sided(A_bloc_diag, A_bloc_upper, A_bloc_lower):
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

    G_diag = np.zeros((nblocks, blockSize, blockSize))


    tic = time.perf_counter() # -----------------------------
    if rank == 0:
        G_diag[0:nblocks_2, ] = rgf_leftprocess(A_bloc_diag[0:nblocks_2, ], A_bloc_upper[0:nblocks_2, ], A_bloc_lower[0:nblocks_2, ])
        comm.barrier()
        G_diag[nblocks_2:, ] = comm.recv(source=1, tag=0)

    elif rank == 1:
        G_diag[nblocks_2:, ] = rgf_rightprocess(A_bloc_diag[nblocks_2:, ], A_bloc_upper[nblocks_2:, ], A_bloc_lower[nblocks_2:, ])
        comm.send(G_diag[nblocks_2:, ], dest=0, tag=0)
        comm.barrier()
    
    comm.barrier()
    toc = time.perf_counter() # -----------------------------

    if rank == 0:
        print(f"RGF 2-Sided: Inversion took {toc - tic:0.4f} seconds")
    return G_diag

