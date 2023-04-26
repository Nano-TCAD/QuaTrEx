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
from block_tri_solvers.rgf_GF import rgf_GF

from operator import mul

import copy


def calc_GF_pool(DH, E, SigR, SigL, SigG, Efl, Efr, Temp, mkl_threads = 1, worker_num = 1):
    kB = 1.38e-23
    q = 1.6022e-19
    
    UT = kB * Temp / q
    
    vfermi = np.vectorize(fermi_function)
    fL = vfermi(E, Efl, UT)
    fR = vfermi(E, Efr, UT)
    
    NE = E.shape[0]
    NA = DH.NA

    dNP = 50 # number of points to smooth the edges of the Green's Function
    
    factor = np.ones(NE)  
    factor[NE-dNP-1:NE] = (np.cos(np.pi*np.linspace(0, 1, dNP+1)) + 1)/2
    factor[0:dNP+1] = (np.cos(np.pi*np.linspace(1, 0, dNP+1)) + 1)/2

    NB = DH.Bmin.shape[0]
    NT = DH.Bmax[-1] + 1
    Bsize = np.max(DH.Bmax - DH.Bmin + 1)
    

    (GR_3D_E, GRnn1_3D_E, GL_3D_E, GLnn1_3D_E, GG_3D_E, GGnn1_3D_E) = initialize_block_G(NE, NB, Bsize)

    mkl.set_num_threads(mkl_threads)

    print("MKL_THREADS: ", mkl_threads)
    print("NUM_WORKERS: ", worker_num)

    rgf_M = generator_rgf_Hamiltonian(E, DH, SigR)

    M_par = np.ndarray(shape = (E.shape[0],),dtype = object)

    for IE in range(NE):
        M_par[IE] = (E[IE] + 1j*1e-12) * DH.Overlap['H_4'] - DH.Hamiltonian['H_4'] - SigR[IE]
    
    index_e = np.arange(NE)
    Bmin = DH.Bmin.copy()
    Bmax = DH.Bmax.copy()

    tic = time.perf_counter()
    # Create a process pool with 4 workers
    with concurrent.futures.ThreadPoolExecutor(max_workers=worker_num) as executor:
        # Use the map function to apply the inv_matrices function to each pair of matrices in parallel
        # Use partial function application to bind the constant arguments to inv_matrices
        # Pass in an additional argument to inv_matrices that contains the index of the matrices pair
        #results = list(executor.map(lambda args: inv_matrices(args[0], const_arg1, const_arg2, args[1]), ((matrices_pairs[i], i) for i in range(len(matrices_pairs)))))
        executor.map(rgf_GF, rgf_M, SigL, SigG, 
                                         GR_3D_E, GRnn1_3D_E, GL_3D_E, GLnn1_3D_E, GG_3D_E, GGnn1_3D_E, fL, fR,
                                         repeat(Bmin), repeat(Bmax), factor, index_e)
    toc = time.perf_counter()

    print("Total Time for parallel section: " + "%.2f" % (toc-tic) + " [s]" )
    
    return GR_3D_E, GRnn1_3D_E, GL_3D_E, GLnn1_3D_E, GG_3D_E, GGnn1_3D_E

def generator_rgf_Hamiltonian(E, DH, SigR):
    for i in range(E.shape[0]):
        yield (E[i] + 1j*1e-12) * DH.Overlap['H_4'] - DH.Hamiltonian['H_4'] - SigR[i]


def assemble_full_G_smoothing(G, factor, G_block, Gnn1_block, Bmin, Bmax, format = 'sparse', type = 'R'):
    G_temp  = factor * mat_assembly_fullG(G_block, Gnn1_block, Bmin, Bmax, format = format, type = type)
    G[:,:] = G_temp[:,:]