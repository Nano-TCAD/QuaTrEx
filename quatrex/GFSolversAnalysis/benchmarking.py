import generateTestingMatrices as gen
import vizualiseMatrix as viz
import inversionsAlgorithms as inv
import verifyResults as verif


if __name__ == "__main__":
    size = 10
    density = 0.3
    bandwidth = 1

    # Dense and sparse initial matrices
    A = gen.generateBandedDiagonalMatrix(size, bandwidth)
    A_csc = gen.denseToSparseStorage(A)

    # Dense and sparse reference solutions (Full inversons)
    A_refsol_np = inv.fullInversion(A)
    A_refsol_csc = inv.fullInversion(A_csc)

    # Dense and sparse reference solutions (Only diagonal banded elements extracted)
    A_refsol_np_bandextracted = verif.extractDiagonalBandedElements(A_refsol_np, bandwidth)
    A_refsol_csc_bandextracted = verif.extractDiagonalBandedElements(A_refsol_csc, bandwidth)

    if not verif.verifResultsDense(A_refsol_np, A_refsol_csc) or not verif.verifResultsDense(A_refsol_np_bandextracted, A_refsol_csc_bandextracted):
        print("Error: reference solutions are different.")
        exit()


    viz.vizualiseDenseMatrixFlat(A, "Initial dense matrix")