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


def calc_W(DH, E, PG, PL, PR, V, w_mask, mkl_threads = 1, worker_num = 1):
    
    NE = E.shape[0]
    NA = DH.NA

    dNP = 50 # number of points to smooth the edges of the Green's Function
    
    factor = np.ones(NE)  
    factor[NE-dNP-1:NE] = (np.cos(np.pi*np.linspace(0, 1, dNP+1)) + 1)/2
    #factor[0:dNP+1] = (np.cos(np.pi*np.linspace(1, 0, dNP+1)) + 1)/2

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
    

    (WR_3D_E,WRnn1_3D_E, WL_3D_E, WLnn1_3D_E, WG_3D_E, WGnn1_3D_E) = initialize_block_G(NE, nb_mm, lb_max_mm)
    XR_3D_E = np.zeros((NE, nb_mm, lb_max_mm, lb_max_mm), dtype = np.cfloat)

    #mkl.set_num_threads(mkl_threads)

    print("MKL_THREADS: ", mkl_threads)
    print("NUM_WORKERS: ", worker_num)
    
    index_e = np.arange(NE)

    for IE in range(NE):
        print(IE)
        print(factor[IE])
        rgf_W(V, PG[IE], PL[IE], PR[IE], bmax, bmin, WG_3D_E[IE], WGnn1_3D_E[IE],
                               WL_3D_E[IE], WLnn1_3D_E[IE], WR_3D_E[IE], WRnn1_3D_E[IE], XR_3D_E[IE], nbc, IE, factor[IE], ref_flag = False)
        

    return WR_3D_E, WRnn1_3D_E, WL_3D_E, WLnn1_3D_E, WG_3D_E, WGnn1_3D_E, nb_mm, lb_max_mm
