from matplotlib import pyplot as plt
import numpy as np

def vizualiseDenseMatrixFlat(mat, legend=""):
    """
        Visualise a dense matrix in a 2D plot. Third dimension is represented by color.
    """
    plt.title('Dense matrix hotmat: ' + legend)
    plt.imshow(abs(mat), cmap='hot', interpolation='nearest')
    plt.show()


def vizualiseCSCMatrixFlat(mat, legend=""):
    """
        Visualise a CSC matrix in a 2D plot. Third dimension is represented by color.
    """
    # add a title and axis labels
    plt.title('CSC matrix hotmat: '+ legend)
    plt.imshow(abs(mat.toarray()), cmap='hot', interpolation='nearest')
    plt.show()


def vizualiseDenseMatrixFromBlocks(A_bloc_diag, A_bloc_upper, A_bloc_lower, legend="", wastedStorage=False):
    """
        Visualise a dense matrix from its diagonals, upper and lower blocks.
    """

    # Recreate a dense matrix from its diagonals, upper and lower blocks
    nBlocks = A_bloc_diag.shape[0]
    blockSize = A_bloc_diag.shape[1]

    A = np.zeros((nBlocks*blockSize, nBlocks*blockSize))

    # Modifing the coloration to vizualiase wasted storage
    if wastedStorage:
        A_bloc_diag[A_bloc_diag > 0] = 1
        A_bloc_diag[A_bloc_diag == 0] = 0.5 # Wasted storage from diag blocks
        A_bloc_upper[A_bloc_upper > 0] = 0.7
        A_bloc_upper[A_bloc_upper == 0] = 0.3 # Wasted storage from upper blocks
        A_bloc_lower[A_bloc_lower > 0] = 0.7
        A_bloc_lower[A_bloc_lower == 0] = 0.3 # Wasted storage from lower blocks

    for i in range(nBlocks):
        A[i*blockSize:(i+1)*blockSize, i*blockSize:(i+1)*blockSize] = A_bloc_diag[i, ]

    for i in range(nBlocks-1):
        A[i*blockSize:(i+1)*blockSize, (i+1)*blockSize:(i+2)*blockSize] = A_bloc_upper[i, ]
        A[(i+1)*blockSize:(i+2)*blockSize, i*blockSize:(i+1)*blockSize] = A_bloc_lower[i, ]

    plt.title('Block matrix hotmat: '+ legend)
    plt.imshow(abs(A), cmap='hot', interpolation='nearest')
    plt.show()
