import generateTestingMatrices as gen
import vizualiseMatrix as viz
import inversionsAlgorithms as inv
import verifyResults as verif

import numpy as np

if __name__ == "__main__":
    size = 2000
    density = 0.3
    bandwidth = 100

    # Dense and sparse initial matrices
    A = gen.generateBandedDiagonalMatrix(size, bandwidth)
    A_csc = gen.denseToSparseStorage(A)

    # Dense and sparse reference solutions (Full inversons)
    A_refsol_np = inv.fullInversion(A)
    A_refsol_csc = inv.fullInversion(A_csc)

    # Dense and sparse reference solutions (Only diagonal banded elements extracted)
    A_refsol_np_bandextracted = verif.extractDiagonalBandedElements(A_refsol_np, bandwidth)
    A_refsol_csc_bandextracted = verif.extractDiagonalBandedElements(A_refsol_csc, bandwidth)

    if not verif.verifResults(A_refsol_np, A_refsol_csc) or not verif.verifResults(A_refsol_np_bandextracted, A_refsol_csc_bandextracted):
        print("Error: reference solutions are different.")
        exit()



    # Alg 1: RGF
    A_bloc_diag, A_bloc_upper, A_bloc_lower = gen.denseToBlockStorage(A, 100)

    G, G_diag = inv.rgf(A_bloc_diag, A_bloc_upper, A_bloc_lower)

    print("RGF diag validation: ", verif.verifResultsDiag(A_refsol_np, G))

    #viz.vizualiseDenseMatrixFromBlocks(A_bloc_diag, A_bloc_upper, A_bloc_lower)
    """ viz.vizualiseDenseMatrixFlat(A_refsol_np_bandextracted, "Reference solution")
    viz.vizualiseDenseMatrixFlat(G, "RGF solution") """