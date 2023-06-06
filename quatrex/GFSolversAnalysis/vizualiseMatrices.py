"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@date: 2023-05

Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.
"""

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

    A = np.zeros((nBlocks*blockSize, nBlocks*blockSize), dtype=A_bloc_diag.dtype)

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



def compareDenseMatrixFromBlocks(A_ref_bloc_diag, A_ref_bloc_upper, A_ref_bloc_lower,
                                 A_bloc_diag, A_bloc_upper, A_bloc_lower, legend=""):
    """
        Compare two dense matrices from their diagonals, upper and lower blocks.
    """

    # Recreate a dense matrix from its diagonals, upper and lower blocks
    nBlocksARef = A_ref_bloc_diag.shape[0]
    blockSizeARef = A_ref_bloc_diag.shape[1]

    nBlocksA = A_bloc_diag.shape[0]
    blockSizeA = A_bloc_diag.shape[1]

    A_ref = np.zeros((nBlocksARef*blockSizeARef, nBlocksARef*blockSizeARef), dtype=A_ref_bloc_diag.dtype)
    A     = np.zeros((nBlocksA*blockSizeA, nBlocksA*blockSizeA), dtype=A_bloc_diag.dtype)

    for i in range(nBlocksARef):
        A_ref[i*blockSizeARef:(i+1)*blockSizeARef, i*blockSizeARef:(i+1)*blockSizeARef] = A_ref_bloc_diag[i, ]

    for i in range(nBlocksARef-1):
        A_ref[i*blockSizeARef:(i+1)*blockSizeARef, (i+1)*blockSizeARef:(i+2)*blockSizeARef] = A_ref_bloc_upper[i, ]
        A_ref[(i+1)*blockSizeARef:(i+2)*blockSizeARef, i*blockSizeARef:(i+1)*blockSizeARef] = A_ref_bloc_lower[i, ]

    for i in range(nBlocksA):
        A[i*blockSizeA:(i+1)*blockSizeA, i*blockSizeA:(i+1)*blockSizeA] = A_bloc_diag[i, ]

    for i in range(nBlocksA-1):
        A[i*blockSizeA:(i+1)*blockSizeA, (i+1)*blockSizeA:(i+2)*blockSizeA] = A_bloc_upper[i, ]
        A[(i+1)*blockSizeA:(i+2)*blockSizeA, i*blockSizeA:(i+1)*blockSizeA] = A_bloc_lower[i, ]

    plt.subplot(1, 2, 1)
    plt.title('Reference solution')
    plt.imshow(abs(A_ref), cmap='hot', interpolation='nearest')

    plt.subplot(1, 2, 2)
    plt.title(legend)
    plt.imshow(abs(A), cmap='hot', interpolation='nearest')

    plt.show()



def showBenchmark(greenRetardedBenchtiming, greenLesserBenchtiming, nBlocks, blockSize):
    plt.subplot(1, 2, 1)
    plt.bar(range(len(greenRetardedBenchtiming)), list(greenRetardedBenchtiming.values()), align='center')
    plt.xticks(range(len(greenRetardedBenchtiming)), list(greenRetardedBenchtiming.keys()))
    plt.title("Retarded Green's function benchmark")
    plt.ylabel("Time (s)")

    plt.subplot(1, 2, 2)
    plt.bar(range(len(greenLesserBenchtiming)), list(greenLesserBenchtiming.values()), align='center')
    plt.xticks(range(len(greenLesserBenchtiming)), list(greenLesserBenchtiming.keys()))
    plt.title("Lesser Green's function benchmark")
    plt.ylabel("Time (s)")

    plt.suptitle(f"matrixSize={(int)(nBlocks*blockSize)}, nBlocks={(int)(nBlocks)}, blockSize={blockSize}")

    plt.show()