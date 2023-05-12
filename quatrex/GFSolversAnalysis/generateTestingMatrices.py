import numpy as np
from scipy.sparse import csc_matrix

def generateDenseMatrix(size):
    """
        Generate a dense matrix of shape: (size x size) filled with random numbers.
    """
    print("Generating dense matrix of size: ", size)

    return np.random.rand(size, size)



def generateSparseMatrix(size, density):
    """
        Generate a sparse matrix of shape: (size x size), densisty of non-zero elements: density,
        filled with random numbers.
    """
    print("Generating sparse matrix of size: ", size, " and density: ", density)

    A = np.random.rand(size, size)
    A[A < (1-density)] = 0
    return A
    


def generateBandedDiagonalMatrix(size, bandwidth):
    """
        Generate a banded diagonal matrix of shape: (size x size), bandwidth: bandwidth,
        filled with random numbers.
    """
    print("Generating banded diagonal matrix of size: ", size, " and bandwidth: ", bandwidth)

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


def denseToBlockStorage(A, nBlocks):
    """
        Converte a numpy dense matrix to 3 dimensional numpy array of blocks. Handling
        the diagonal blocks, upper diagonal blocks and lower diagonal blocks separately.
    """
    blockSize = int(np.ceil(A.shape[0]/nBlocks))

    A_bloc_diag = np.zeros((nBlocks, blockSize, blockSize))
    A_bloc_upper = np.zeros((nBlocks-1, blockSize, blockSize))
    A_bloc_lower = np.zeros((nBlocks-1, blockSize, blockSize))

    for i in range(nBlocks):
        A_bloc_diag[i, ] = A[i*blockSize:(i+1)*blockSize, i*blockSize:(i+1)*blockSize]
        if i < nBlocks-1:
            A_bloc_upper[i, ] = A[i*blockSize:(i+1)*blockSize, (i+1)*blockSize:(i+2)*blockSize]
            A_bloc_lower[i, ] = A[(i+1)*blockSize:(i+2)*blockSize, i*blockSize:(i+1)*blockSize]

    return A_bloc_diag, A_bloc_upper, A_bloc_lower