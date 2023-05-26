import generateMatrices as gen
import vizualiseMatrices as viz
import verifyResults as verif

import algorithms.fullInversion as inv
import algorithms.rgf as rgf
import algorithms.rgf2sided as rgf2sided

import numpy as np
from mpi4py import MPI


if __name__ == "__main__":
    # ---------------------------------------------------------------------------------------------
    # Initialization of the problem and computation of the reference solution
    # ---------------------------------------------------------------------------------------------
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    size = 12
    blocksize = 2
    density = blocksize**2/size**2
    bandwidth = np.ceil(blocksize/2)

    # Dense and sparse initial matrices
    A = gen.generateBandedDiagonalMatrix(size, bandwidth, 63)
    A = gen.makeSymmetric(A)
    A_csc = gen.denseToSparseStorage(A)

    # Dense and sparse reference solutions (Full inversons)
    A_refsol_np  = inv.fullInversion(A)
    A_refsol_csc = inv.fullInversion(A_csc)

    if not verif.verifResults(A_refsol_np, A_refsol_csc):
        print("Error: reference solutions are different.")
        exit()
    else:
        A_refsol_bloc_diag, A_refsol_bloc_upper, A_refsol_bloc_lower = gen.denseToBlocksTriDiagStorage(A_refsol_np, blocksize)

    A_bloc_diag, A_bloc_upper, A_bloc_lower = gen.denseToBlocksTriDiagStorage(A, blocksize)



    A_debug_diag, A_debug_upper, A_debug_lower = gen.denseToBlocksTriDiagStorage(A, blocksize)


    comm.barrier()
    # ---------------------------------------------------------------------------------------------
    # 1. RGF  
    # ---------------------------------------------------------------------------------------------

    if rank == 0: # Single process algorithm
        G_rgf_diag, G_rgf_upper, G_rgf_lower = rgf.rgf(A_bloc_diag, A_bloc_upper, A_bloc_lower)

        print("RGF validation: ", verif.verifResultsBlocksTri(A_refsol_bloc_diag, 
                                                            A_refsol_bloc_upper, 
                                                            A_refsol_bloc_lower, 
                                                            G_rgf_diag, 
                                                            G_rgf_upper, 
                                                            G_rgf_lower)) 

    comm.barrier()
    # ---------------------------------------------------------------------------------------------
    # 2. RGF 2-sided 
    # ---------------------------------------------------------------------------------------------
    # mpiexec -n 2 python benchmarking.py

"""     G_rgf2sided_diag = rgf2sided.rgf2sided(A_bloc_diag, A_bloc_upper, A_bloc_lower)

    if rank == 0: # Results agregated on 1st process and compared to reference solution
        print("RGF 2-sided validation: ", verif.verifResultsBlocksTri(A_refsol_bloc_diag, 
                                                                      A_refsol_bloc_upper, 
                                                                      A_refsol_bloc_lower, 
                                                                      G_rgf2sided_diag, 
                                                                      A_refsol_bloc_upper, 
                                                                      A_refsol_bloc_lower)) 

        viz.vizualiseDenseMatrixFromBlocks(A_refsol_bloc_diag, A_refsol_bloc_upper, A_refsol_bloc_lower, "Reference solution")
        viz.vizualiseDenseMatrixFromBlocks(G_rgf2sided_diag, A_refsol_bloc_upper, A_refsol_bloc_lower, "RGF 2-sided solution")
    
 """