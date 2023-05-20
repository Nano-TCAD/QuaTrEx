import numpy as np
from scipy.sparse import csc_matrix
from mpi4py import MPI


def generateDenseMatrix(size, seed=None):
    """
        Generate a dense matrix of shape: (size x size) filled with random numbers.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        print("Generating dense matrix of size: ", size)

    if seed is not None:
        np.random.seed(seed)

    return np.random.rand(size, size)



def generateSparseMatrix(size, density, seed=None):
    """
        Generate a sparse matrix of shape: (size x size), densisty of non-zero elements: density,
        filled with random numbers.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        print("Generating sparse matrix of size: ", size, " and density: ", density)

    if seed is not None:
        np.random.seed(seed)

    A = np.random.rand(size, size)
    A[A < (1-density)] = 0
    return A
    


def generateBandedDiagonalMatrix(size, bandwidth, seed=None):
    """
        Generate a banded diagonal matrix of shape: (size x size), bandwidth: bandwidth,
        filled with random numbers.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        print("Generating banded diagonal matrix of size: ", size, " and bandwidth: ", bandwidth)

    if seed is not None:
        np.random.seed(seed)

    A = np.random.rand(size, size)
    
    for i in range(size):
        for j in range(size):
            if i - j > bandwidth or j - i > bandwidth:
                A[i, j] = 0
    return A



def denseToSparseStorage(A):
    """
        Convert a numpy dense matrix to a sparse matrix into scipy.csc format.
    """
    return csc_matrix(A)



def denseToBlocksTriDiagStorage(A, blockSize):
    """
        Converte a numpy dense matrix to 3 dimensional numpy array of blocks. Handling
        the diagonal blocks, upper diagonal blocks and lower diagonal blocks separately.
    """
    nBlocks = int(np.ceil(A.shape[0]/blockSize))

    A_bloc_diag = np.zeros((nBlocks, blockSize, blockSize))
    A_bloc_upper = np.zeros((nBlocks-1, blockSize, blockSize))
    A_bloc_lower = np.zeros((nBlocks-1, blockSize, blockSize))

    for i in range(nBlocks):
        A_bloc_diag[i, ] = A[i*blockSize:(i+1)*blockSize, i*blockSize:(i+1)*blockSize]
        if i < nBlocks-1:
            A_bloc_upper[i, ] = A[i*blockSize:(i+1)*blockSize, (i+1)*blockSize:(i+2)*blockSize]
            A_bloc_lower[i, ] = A[(i+1)*blockSize:(i+2)*blockSize, i*blockSize:(i+1)*blockSize]

    return A_bloc_diag, A_bloc_upper, A_bloc_lower



def makeSymmetric(A):
    """
        Make a matrix symmetric by adding its transpose to itself.
    """
    return A + A.T