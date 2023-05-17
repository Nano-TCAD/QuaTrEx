import generateMatrices as gen
import vizualiseMatrices as viz
import inversionsAlgorithms as inv
import verifyResults as verif

import numpy as np
from mpi4py import MPI


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()


    size = 12   
    blocksize = 2
    density = blocksize**2/size**2
    bandwidth = np.ceil(blocksize/2)

    # Dense and sparse initial matrices
    A = gen.generateBandedDiagonalMatrix(size, bandwidth, 63)
    A_csc = gen.denseToSparseStorage(A)

    # Dense and sparse reference solutions (Full inversons)
    A_refsol_np = inv.fullInversion(A)
    A_refsol_csc = inv.fullInversion(A_csc)

    if not verif.verifResults(A_refsol_np, A_refsol_csc):
        print("Error: reference solutions are different.")
        exit()
    else:
        A_refsol_bloc_diag, A_refsol_bloc_upper, A_refsol_bloc_lower = gen.denseToBlocksTriDiagStorage(A_refsol_np, blocksize)

    A_bloc_diag, A_bloc_upper, A_bloc_lower = gen.denseToBlocksTriDiagStorage(A, blocksize)



    # --- Alg 1: RGF ---
    if rank == 0:
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

        """ print("A_refsol_bloc_lower", A_refsol_bloc_lower[-1])
        print("G_rgf_lower", G_rgf_lower[-1])

        print("A_refsol_bloc_upper", A_refsol_bloc_upper[-1])
        print("G_rgf_upper", G_rgf_upper[-1])

        viz.vizualiseDenseMatrixFromBlocks(A_refsol_bloc_diag, A_refsol_bloc_upper, A_refsol_bloc_lower, "Reference solution")
        viz.vizualiseDenseMatrixFromBlocks(G_rgf_diag, G_rgf_upper, G_rgf_lower, "RGF solution") """



    # --- Alg 2: RGF 2-sided ---
    # mpiexec -n 2 python benchmarking.py
    G_rgf2sided_diag = inv.rgf2sided(A_bloc_diag, A_bloc_upper, A_bloc_lower)

    if rank == 0:
        print("RGF 2-sided validation: ", verif.verifResultsBlocksTri(A_refsol_bloc_diag, 
                                                                      A_refsol_bloc_upper, 
                                                                      A_refsol_bloc_lower, 
                                                                      G_rgf2sided_diag[0], 
                                                                      A_refsol_bloc_upper[1], 
                                                                      A_refsol_bloc_lower[2])) 

        viz.vizualiseDenseMatrixFromBlocks(A_refsol_bloc_diag, A_refsol_bloc_upper, A_refsol_bloc_lower, "Reference solution")
        viz.vizualiseDenseMatrixFromBlocks(G_rgf2sided_diag, A_refsol_bloc_upper, A_refsol_bloc_lower, "RGF 2-sided solution")
    
