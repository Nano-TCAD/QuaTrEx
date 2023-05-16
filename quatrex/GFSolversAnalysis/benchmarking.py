import generateMatrices as gen
import vizualiseMatrices as viz
import inversionsAlgorithms as inv
import verifyResults as verif

import numpy as np

if __name__ == "__main__":
    size = 12   
    blocksize = 1
    density = blocksize**2/size**2
    bandwidth = np.ceil(blocksize/2)

    # Dense and sparse initial matrices
    A = gen.generateBandedDiagonalMatrix(size, bandwidth)
    A_csc = gen.denseToSparseStorage(A)

    # Dense and sparse reference solutions (Full inversons)
    A_refsol_np = inv.fullInversion(A)
    A_refsol_csc = inv.fullInversion(A_csc)

    if not verif.verifResults(A_refsol_np, A_refsol_csc):
        print("Error: reference solutions are different.")
        exit()
    else:
        A_refsol_bloc_diag, A_refsol_bloc_upper, A_refsol_bloc_lower = gen.denseToBlocksTriDiagStorage(A_refsol_np, blocksize)
        

    # --- Alg 1: RGF ---
    A_bloc_diag, A_bloc_upper, A_bloc_lower = gen.denseToBlocksTriDiagStorage(A, blocksize)

    G_rgf_diag, G_rgf_upper, G_rgf_lower = inv.rgf(A_bloc_diag, A_bloc_upper, A_bloc_lower)

    print("RGF validation: ", verif.verifResultsBlocksTri(A_refsol_bloc_diag, 
                                                          A_refsol_bloc_upper, 
                                                          A_refsol_bloc_lower, 
                                                          G_rgf_diag, 
                                                          G_rgf_upper, 
                                                          G_rgf_lower)) 
    
    print("RGF diag validation: ", verif.verifResultsBlocksTri(A_refsol_bloc_diag, 
                                                               A_refsol_bloc_upper, 
                                                               A_refsol_bloc_lower, 
                                                               G_rgf_diag, 
                                                               A_refsol_bloc_upper, 
                                                               A_refsol_bloc_lower)) 
    
    print("RGF upper validation: ", verif.verifResultsBlocksTri(A_refsol_bloc_diag, 
                                                                A_refsol_bloc_upper, 
                                                                A_refsol_bloc_lower, 
                                                                A_refsol_bloc_diag, 
                                                                G_rgf_upper, 
                                                                A_refsol_bloc_lower)) 

    print("RGF lower validation: ", verif.verifResultsBlocksTri(A_refsol_bloc_diag, 
                                                                A_refsol_bloc_upper, 
                                                                A_refsol_bloc_lower, 
                                                                A_refsol_bloc_diag, 
                                                                A_refsol_bloc_upper, 
                                                                G_rgf_lower)) 

    viz.vizualiseDenseMatrixFromBlocks(A_refsol_bloc_diag, A_refsol_bloc_upper, A_refsol_bloc_lower, "Reference solution")
    viz.vizualiseDenseMatrixFromBlocks(G_rgf_diag, G_rgf_upper, G_rgf_lower, "RGF solution")
    """ viz.vizualiseDenseMatrixFlat(A_refsol_np_bandextracted, "Reference solution")
    viz.vizualiseDenseMatrixFlat(G, "RGF solution") """
