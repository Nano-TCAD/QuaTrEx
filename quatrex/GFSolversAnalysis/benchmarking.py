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

    # Problem parameters
    size = 12
    blocksize = 2
    density = blocksize**2/size**2
    bandwidth = np.ceil(blocksize/2)

    seed = 63
    isComplex = True

    # Retarded Green's function initial matrix
    A = gen.generateBandedDiagonalMatrix(size, bandwidth, seed, isComplex)
    A = gen.makeSymmetric(A)
    A_csc = gen.denseToCSC(A)

    # Retarded Green's function references solutions (Full inversons)
    GreenRetarded_refsol_np  = inv.fullInversion(A)
    GreenRetarded_refsol_csc = inv.fullInversion(A_csc)

    if not verif.verifResults(GreenRetarded_refsol_np, GreenRetarded_refsol_csc):
        print("Error: reference solutions are different.")
        exit()
    else:
        # Extract the blocks from the retarded Green's function reference solution
        GreenRetarded_refsol_bloc_diag\
        , GreenRetarded_refsol_bloc_upper\
        , GreenRetarded_refsol_bloc_lower = gen.denseToBlocksTriDiagStorage(GreenRetarded_refsol_np, blocksize)

    A_bloc_diag, A_bloc_upper, A_bloc_lower = gen.denseToBlocksTriDiagStorage(A, blocksize)



    comm.barrier()
    # ---------------------------------------------------------------------------------------------
    # 1. RGF  
    # ---------------------------------------------------------------------------------------------

    if rank == 0: # Single process algorithm
        GreenRetarded_rgf_diag, GreenRetarded_rgf_upper, GreenRetarded_rgf_lower = rgf.rgf(A_bloc_diag, A_bloc_upper, A_bloc_lower)

        print("RGF validation: ", verif.verifResultsBlocksTri(GreenRetarded_refsol_bloc_diag, 
                                                              GreenRetarded_refsol_bloc_upper, 
                                                              GreenRetarded_refsol_bloc_lower, 
                                                              GreenRetarded_rgf_diag, 
                                                              GreenRetarded_rgf_upper, 
                                                              GreenRetarded_rgf_lower)) 



    comm.barrier()
    # ---------------------------------------------------------------------------------------------
    # 2. RGF 2-sided 
    # ---------------------------------------------------------------------------------------------
    # mpiexec -n 2 python benchmarking.py

    GreenRetarded_rgf2sided_diag = rgf2sided.rgf2sided(A_bloc_diag, A_bloc_upper, A_bloc_lower)

    if rank == 0: # Results agregated on 1st process and compared to reference solution
        print("RGF 2-sided validation: ", verif.verifResultsBlocksTri(GreenRetarded_refsol_bloc_diag, 
                                                                      GreenRetarded_refsol_bloc_upper, 
                                                                      GreenRetarded_refsol_bloc_lower, 
                                                                      GreenRetarded_rgf2sided_diag, 
                                                                      GreenRetarded_refsol_bloc_upper, 
                                                                      GreenRetarded_refsol_bloc_lower)) 


        viz.compareDenseMatrixFromBlocks(GreenRetarded_refsol_bloc_diag, 
                                         GreenRetarded_refsol_bloc_upper, 
                                         GreenRetarded_refsol_bloc_lower,
                                         GreenRetarded_rgf2sided_diag, 
                                         GreenRetarded_refsol_bloc_upper, 
                                         GreenRetarded_refsol_bloc_lower, "RGF 2-sided solution")
