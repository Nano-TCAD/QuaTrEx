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
    # factor[0:dNP+1] = (np.cos(np.pi*np.linspace(1, 0, dNP+1)) + 1)/2
    #factor[np.where(np.invert(w_mask))[0]] = 0.0

    nb = DH.Bmin.shape[0]
    NT = DH.Bmax[-1]
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
    XR_3D_E = np.zeros((NE, nb_mm, lb_max_mm, lb_max_mm), dtype = np.complex128)
    mkl.set_num_threads(mkl_threads)

    print("MKL_THREADS: ", mkl_threads)
    print("NUM_WORKERS: ", worker_num)
    
    index_e = np.arange(NE)

    # dosw not needed, but as a placeholder
    dosw = np.zeros(shape=(NE,nb_mm), dtype = np.complex128)
    nEw = np.zeros(shape=(NE,nb_mm), dtype = np.complex128)
    nPw = np.zeros(shape=(NE,nb_mm), dtype = np.complex128)

    tic = time.perf_counter()
    # Create a process pool with 4 workers
    with concurrent.futures.ThreadPoolExecutor(max_workers=worker_num) as executor:
        # Use the map function to apply the inv_matrices function to each pair of matrices in parallel
        # Use partial function application to bind the constant arguments to inv_matrices
        # Pass in an additional argument to inv_matrices that contains the index of the matrices pair
        #results = list(executor.map(lambda args: inv_matrices(args[0], const_arg1, const_arg2, args[1]), ((matrices_pairs[i], i) for i in range(len(matrices_pairs)))))
        executor.map(
                    rgf_W,
                    repeat(V),
                    PG, PL, PR,
                    repeat(bmax), repeat(bmin),
                    WG_3D_E, WGnn1_3D_E,
                    WL_3D_E, WLnn1_3D_E,
                    WR_3D_E, WRnn1_3D_E,
                    XR_3D_E, dosw, nEw, nPw, repeat(nbc),
                    index_e, factor)
    toc = time.perf_counter()

    print("Total Time for parallel W section: " + "%.2f" % (toc-tic) + " [s]" )

    # Calculate F1, F2, which are the relative errors of GR-GA = GG-GL 
    F1 = np.max(np.abs(dosw - (nEw + nPw)) / (np.abs(dosw) + 1e-6), axis=1)
    F2 = np.max(np.abs(dosw - (nEw + nPw)) / (np.abs(nEw + nPw) + 1e-6), axis=1)

    # Remove individual peaks
    # dDOSm = np.concatenate(([0], np.max(np.abs(dosw[1:NE-1, :] / (dosw[0:NE-2, :] + 1)), axis=1), [0]))
    # dDOSp = np.concatenate(([0], np.max(np.abs(dosw[1:NE-1, :] / (dosw[2:NE, :] + 1)), axis=1), [0]))

    # Find indices of elements satisfying the conditions
    ind_zeros = np.where((F1 > 0.1) | (F2 > 0.1))[0]
    
    # Remove the identified peaks and errors
    for index in ind_zeros:
        WR_3D_E[index, :, :, :] = 0
        WRnn1_3D_E[index, :, :, :] = 0
        WL_3D_E[index, :, :, :] = 0
        WLnn1_3D_E[index, :, :, :] = 0
        WG_3D_E[index, :, :, :] = 0
        WGnn1_3D_E[index, :, :, :] = 0
    
    # with concurrent.futures.ThreadPoolExecutor(max_workers=worker_num) as executor:    
    #     executor.map(assemble_full_G_smoothing, GR, factor, GR_3D_E, GRnn1_3D_E, repeat(DH.Bmin), repeat(DH.Bmax), repeat('sparse'), repeat('R'))
    #     executor.map(assemble_full_G_smoothing, GL, factor, GL_3D_E, GLnn1_3D_E, repeat(DH.Bmin), repeat(DH.Bmax), repeat('sparse'), repeat('L'))
    #     executor.map(assemble_full_G_smoothing, GG, factor, GG_3D_E, GGnn1_3D_E, repeat(DH.Bmin), repeat(DH.Bmax), repeat('sparse'), repeat('G'))

    return WR_3D_E, WRnn1_3D_E, WL_3D_E, WLnn1_3D_E, WG_3D_E, WGnn1_3D_E, nb_mm, lb_max_mm
