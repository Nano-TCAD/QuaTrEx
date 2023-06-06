"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@date: 2023-05

Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.
"""

import generateMatrices as genMat
import vizualiseMatrices as vizMat
import verifyResults as verif

import algorithms.fullInversion as inv
import algorithms.rgf as rgf
import algorithms.rgf2sided as rgf2sided

import numpy as np
import time

from mpi4py import MPI



if __name__ == "__main__":
    # ---------------------------------------------------------------------------------------------
    # Initialization of the problem and computation of the reference solution
    # ---------------------------------------------------------------------------------------------
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    greenRetardedBenchtiming = {"np": 0, "csc": 0, "rgf": 0, "rgf2sided": 0}
    greenLesserBenchtiming   = {"np": 0, "csc": 0, "rgf": 0, "rgf2sided": 0}

    # Problem parameters
    size = 12
    blocksize = 2
    density = blocksize**2/size**2
    bandwidth = np.ceil(blocksize/2)

    isComplex = True
    seed = 63



    # Retarded Green's function initial matrix
    A = genMat.generateBandedDiagonalMatrix(size, bandwidth, isComplex, seed)
    A = genMat.makeSymmetric(A)
    A_csc = genMat.denseToCSC(A)

    # Retarded Green's function references solutions (Full inversions)
    GreenRetarded_refsol_np, greenRetardedBenchtiming["np"]   = inv.numpyInversion(A)
    GreenRetarded_refsol_csc, greenRetardedBenchtiming["csc"] = inv.scipyCSCInversion(A_csc)

    if not verif.verifResults(GreenRetarded_refsol_np, GreenRetarded_refsol_csc):
        print("Error: Green retarded references solutions are different.")
        exit()
    else:
        # Extract the blocks from the retarded Green's function reference solution
        GreenRetarded_refsol_block_diag\
        , GreenRetarded_refsol_block_upper\
        , GreenRetarded_refsol_block_lower = genMat.denseToBlocksTriDiagStorage(GreenRetarded_refsol_np, blocksize)

    A_block_diag, A_block_upper, A_block_lower = genMat.denseToBlocksTriDiagStorage(A, blocksize)



    # Lesser Green's function initial matrix
    GreenAdvanced_refsol_np = np.conjugate(np.transpose(GreenRetarded_refsol_np)) 

    SigmaLesser = genMat.generateBandedDiagonalMatrix(size, bandwidth, isComplex, seed)
    SigmaLesser = genMat.makeSymmetric(SigmaLesser)

    # Lesser Green's function references solutions (Full inversions)
    # 1. Dense matrix
    tic = time.perf_counter()
    B = A @ SigmaLesser @ GreenAdvanced_refsol_np
    toc = time.perf_counter()

    timing = toc - tic

    GreenLesser_refsol_np, greenLesserBenchtiming["np"]   = inv.numpyInversion(B)
    greenLesserBenchtiming["np"] += timing

    # 2. CSC matrix
    GreenAdvanced_refsol_csc = genMat.denseToCSC(GreenAdvanced_refsol_np)
    SigmaLesser_csc = genMat.denseToCSC(SigmaLesser)

    tic = time.perf_counter()
    B_csc = A_csc @ SigmaLesser_csc @ GreenAdvanced_refsol_csc
    toc = time.perf_counter()

    timing = toc - tic

    GreenLesser_refsol_csc, greenLesserBenchtiming["csc"] = inv.scipyCSCInversion(B_csc)
    greenLesserBenchtiming["csc"] += timing

    if not verif.verifResults(GreenLesser_refsol_np, GreenLesser_refsol_csc):
        print("Error: Green lesser references solutions are different.")
        exit()
    else:
        # Extract the blocks from the retarded Green's function reference solution
        GreenAdvanced_refsol_block_diag\
        , GreenAdvanced_refsol_block_upper\
        , GreenAdvanced_refsol_block_lower = genMat.denseToBlocksTriDiagStorage(GreenLesser_refsol_np, blocksize)




    comm.barrier()
    # ---------------------------------------------------------------------------------------------
    # 1. RGF  
    # ---------------------------------------------------------------------------------------------

    if rank == 0: # Single process algorithm
        GreenRetarded_rgf_diag\
        , GreenRetarded_rgf_upper\
        , GreenRetarded_rgf_lower\
        , greenRetardedBenchtiming["rgf"] = rgf.rgf_Gr(A_block_diag, A_block_upper, A_block_lower)

        print("RGF: Gr validation: ", verif.verifResultsBlocksTri(GreenRetarded_refsol_block_diag, 
                                                                 GreenRetarded_refsol_block_upper, 
                                                                 GreenRetarded_refsol_block_lower, 
                                                                 GreenRetarded_rgf_diag, 
                                                                 GreenRetarded_rgf_upper, 
                                                                 GreenRetarded_rgf_lower)) 



    comm.barrier()
    # ---------------------------------------------------------------------------------------------
    # 2. RGF 2-sided 
    # ---------------------------------------------------------------------------------------------
    # mpiexec -n 2 python benchmarking.py

    GreenRetarded_rgf2sided_diag\
    , GreenRetarded_rgf2sided_upper\
    , GreenRetarded_rgf2sided_lower\
    , greenRetardedBenchtiming["rgf2sided"] = rgf2sided.rgf2sided_Gr(A_block_diag, A_block_upper, A_block_lower)

    if rank == 0: # Results agregated on 1st process and compared to reference solution
        print("RGF 2-sided: Gr validation: ", verif.verifResultsBlocksTri(GreenRetarded_refsol_block_diag, 
                                                                          GreenRetarded_refsol_block_upper, 
                                                                          GreenRetarded_refsol_block_lower, 
                                                                          GreenRetarded_rgf2sided_diag, 
                                                                          GreenRetarded_rgf2sided_upper, 
                                                                          GreenRetarded_rgf2sided_lower)) 


        vizMat.compareDenseMatrixFromBlocks(GreenRetarded_refsol_block_diag, 
                                            GreenRetarded_refsol_block_upper, 
                                            GreenRetarded_refsol_block_lower,
                                            GreenRetarded_rgf2sided_diag, 
                                            GreenRetarded_rgf2sided_upper, 
                                            GreenRetarded_rgf2sided_lower, "RGF 2-sided solution")



    

    # ---------------------------------------------------------------------------------------------
    # X. Data plotting
    # ---------------------------------------------------------------------------------------------
    #if rank == 0:
        #vizMat.showBenchmark(greenRetardedBenchtiming, greenLesserBenchtiming, size/blocksize, blocksize)


