import generateTestingMatrices as gen
import vizualiseMatrix as viz
import inversionsAlgorithms as inv


if __name__ == "__main__":
    size = 100
    density = 0.3
    bandwidth = 10

    A = gen.generateBandedDiagonalMatrix(size, bandwidth)
    A_refsol_np = inv.fullInversion(A)

    A_csr = gen.denseToSparseStorage(A)
    A_refsol_csr = inv.fullInversion(A_csr)


    

    