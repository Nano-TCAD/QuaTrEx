#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 18:02:03 2023

@author: leonard
"""
import concurrent.futures
from itertools import repeat

import time

import numpy as np
from scipy import sparse
import mkl

from utils.linalg import invert
from utils.matrix_creation import initialize_block_G, mat_assembly_fullG
from GreensFunction.fermi import fermi_function
from block_tri_solvers.rgf_W import rgf_W
from operator import mul

import copy


def calc_W_pool(DH, E, PG, PL, PR, V, w_mask, mkl_threads = 1, worker_num = 1):
    
    NE = E.shape[0]
    NA = DH.NA

    dNP = 50 # number of points to smooth the edges of the Green's Function
    
    factor = np.ones(NE)  
    factor[NE-dNP-1:NE] = (np.cos(np.pi*np.linspace(0, 1, dNP+1)) + 1)/2
    factor[0:dNP+1] = (np.cos(np.pi*np.linspace(1, 0, dNP+1)) + 1)/2

    factor[np.where(np.invert(w_mask))[0]] = 0.0

    nb = DH.Bmin.shape[0]
    NT = DH.Bmax[-1] + 1
    Bsize = np.max(DH.Bmax - DH.Bmin + 1)
    bmax = DH.Bmax - 1
    bmin = DH.Bmin - 1

    # fix nbc to 2 for the given solution
    # todo calculate it
    nbc = 2

    # block sizes after matrix multiplication
    bmax_ref = bmax[nbc-1:nb:nbc]
    bmin_ref = bmin[0:nb:nbc]
    # number of blocks after matrix multiplication
    nb_mm = bmax_ref.size
    # larges block length after matrix multiplication 
    lb_max_mm = np.max(bmax_ref - bmin_ref + 1)
    

    (WR_3D_E, WRnn1_3D_E, WL_3D_E, WLnn1_3D_E, WG_3D_E, WGnn1_3D_E) = initialize_block_G(NE, nb_mm, lb_max_mm)
    XR_3D_E = np.zeros((NE, nb_mm, lb_max_mm, lb_max_mm), dtype = np.cfloat)
    mkl.set_num_threads(mkl_threads)

    print("MKL_THREADS: ", mkl_threads)
    print("NUM_WORKERS: ", worker_num)
    
    index_e = np.arange(NE)

    tic = time.perf_counter()
    # Create a process pool with 4 workers
    with concurrent.futures.ThreadPoolExecutor(max_workers=worker_num) as executor:
        # Use the map function to apply the inv_matrices function to each pair of matrices in parallel
        # Use partial function application to bind the constant arguments to inv_matrices
        # Pass in an additional argument to inv_matrices that contains the index of the matrices pair
        #results = list(executor.map(lambda args: inv_matrices(args[0], const_arg1, const_arg2, args[1]), ((matrices_pairs[i], i) for i in range(len(matrices_pairs)))))
        executor.map(rgf_W, repeat(V), PG, PL, PR, repeat(bmax), repeat(bmin),  WG_3D_E, WGnn1_3D_E,
                                            WL_3D_E, WLnn1_3D_E, WR_3D_E, WRnn1_3D_E, XR_3D_E, repeat(nbc), index_e, factor)
    toc = time.perf_counter()

    print("Total Time for parallel W section: " + "%.2f" % (toc-tic) + " [s]" )
    
    
    # with concurrent.futures.ThreadPoolExecutor(max_workers=worker_num) as executor:    
    #     executor.map(assemble_full_G_smoothing, GR, factor, GR_3D_E, GRnn1_3D_E, repeat(DH.Bmin), repeat(DH.Bmax), repeat('sparse'), repeat('R'))
    #     executor.map(assemble_full_G_smoothing, GL, factor, GL_3D_E, GLnn1_3D_E, repeat(DH.Bmin), repeat(DH.Bmax), repeat('sparse'), repeat('L'))
    #     executor.map(assemble_full_G_smoothing, GG, factor, GG_3D_E, GGnn1_3D_E, repeat(DH.Bmin), repeat(DH.Bmax), repeat('sparse'), repeat('G'))

    return WR_3D_E, WRnn1_3D_E, WL_3D_E, WLnn1_3D_E, WG_3D_E, WGnn1_3D_E, nb_mm, lb_max_mm
