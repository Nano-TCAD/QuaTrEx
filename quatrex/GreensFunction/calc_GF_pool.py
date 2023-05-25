# Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.
import concurrent.futures
from itertools import repeat

import time

import numpy as np
import numpy.typing as npt
from scipy import sparse
import mkl

from utils.matrix_creation import initialize_block_G, mat_assembly_fullG
from GreensFunction.fermi import fermi_function
from block_tri_solvers.rgf_GF import rgf_GF

from operator import mul

import copy


def calc_GF_pool(DH, E, SigR, SigL, SigG, Efl, Efr, Temp, DOS, nE, nP, idE, mkl_threads = 1, worker_num = 1):
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

    rgf_M = generator_rgf_Hamiltonian(E, DH, SigR)

    M_par = np.ndarray(shape = (E.shape[0],),dtype = object)

    for IE in range(NE):
        SigL[IE] = (SigL[IE] - SigL[IE].T.conj()) / 2
        SigG[IE] = (SigG[IE] - SigG[IE].T.conj()) / 2
    
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
                                         GR_3D_E, GRnn1_3D_E, GL_3D_E, GLnn1_3D_E, GG_3D_E, GGnn1_3D_E, DOS, nE, nP, idE, fL, fR,
                                         repeat(Bmin), repeat(Bmax), factor, index_e)
    toc = time.perf_counter()

    print("Total Time for parallel section: " + "%.2f" % (toc-tic) + " [s]" )
    # Calculate F1, F2, which are the relative errors of GR-GA = GG-GL 
    F1 = np.max(np.abs(DOS - (nE + nP)) / (np.abs(DOS) + 1e-6), axis=1)
    F2 = np.max(np.abs(DOS - (nE + nP)) / (np.abs(nE + nP) + 1e-6), axis=1)

    # Remove individual peaks
    dDOSm = np.concatenate(([0], np.max(np.abs(DOS[1:NE-1, :] / (DOS[0:NE-2, :] + 1)), axis=1), [0]))
    dDOSp = np.concatenate(([0], np.max(np.abs(DOS[1:NE-1, :] / (DOS[2:NE, :] + 1)), axis=1), [0]))

    # Find indices of elements satisfying the conditions
    ind_zeros = np.where((F1 > 0.1) | (F2 > 0.1) | ((dDOSm > 5) & (dDOSp > 5)))[0]

    for index in ind_zeros:
        GR_3D_E[index, :, :, :] = 0
        GRnn1_3D_E[index, :, :, :] = 0
        GL_3D_E[index, :, :, :] = 0
        GLnn1_3D_E[index, :, :, :] = 0
        GG_3D_E[index, :, :, :] = 0
        GGnn1_3D_E[index, :, :, :] = 0

    return GR_3D_E, GRnn1_3D_E, GL_3D_E, GLnn1_3D_E, GG_3D_E, GGnn1_3D_E

def calc_GF_pool_mpi(
        DH,
        energy: npt.NDArray[np.float64],
        SigR,
        SigL,
        SigG,
        Efl,
        Efr,
        Temp,
        DOS,
        nE,
        nP,
        idE,
        factor: npt.NDArray[np.float64],
        mkl_threads: int = 1,
        worker_num: int = 1
):

    kB = 1.38e-23
    q = 1.6022e-19
    
    UT = kB * Temp / q
    
    vfermi = np.vectorize(fermi_function)
    fL = vfermi(energy, Efl, UT)
    fR = vfermi(energy, Efr, UT)

    # initialize the Green's function in block format with zero
    # number of energy points
    ne = energy.shape[0]
    # number of blocks
    nb = DH.Bmin.shape[0]
    # length of the largest block
    lb = np.max(DH.Bmax - DH.Bmin + 1)
    # init
    (GR_3D_E, GRnn1_3D_E, GL_3D_E, GLnn1_3D_E, GG_3D_E, GGnn1_3D_E) = initialize_block_G(ne, nb, lb)

    mkl.set_num_threads(mkl_threads)

    for ie in range(ne):
        SigL[ie] = 1j * np.imag(SigL[ie])
        SigG[ie] = 1j * np.imag(SigG[ie])  

        SigL[ie] = (SigL[ie] - SigL[ie].T.conj()) / 2
        SigG[ie] = (SigG[ie] - SigG[ie].T.conj()) / 2
        SigR[ie] = np.real(SigR[ie]) + 1j * np.imag(SigG[ie] - SigL[ie])/2  
        SigR[ie] = (SigR[ie] + SigR[ie].T) / 2

    rgf_M = generator_rgf_Hamiltonian(energy, DH, SigR)
    
    index_e = np.arange(ne)
    bmin = DH.Bmin.copy()
    bmax = DH.Bmax.copy()

    # Create a process pool with 4 workers
    with concurrent.futures.ThreadPoolExecutor(max_workers=worker_num) as executor:
        # Use the map function to apply the inv_matrices function to each pair of matrices in parallel
        # Use partial function application to bind the constant arguments to inv_matrices
        # Pass in an additional argument to inv_matrices that contains the index of the matrices pair
        #results = list(executor.map(lambda args: inv_matrices(args[0], const_arg1, const_arg2, args[1]), ((matrices_pairs[i], i) for i in range(len(matrices_pairs)))))
        executor.map(rgf_GF, rgf_M, SigL, SigG, 
                                         GR_3D_E, GRnn1_3D_E, GL_3D_E, GLnn1_3D_E, GG_3D_E, GGnn1_3D_E, DOS, nE, nP, idE, fL, fR,
                                         repeat(bmin), repeat(bmax), factor, index_e)
    
    # Calculate F1, F2, which are the relative errors of GR-GA = GG-GL 
    F1 = np.max(np.abs(DOS - (nE + nP)) / (np.abs(DOS) + 1e-6), axis=1)
    F2 = np.max(np.abs(DOS - (nE + nP)) / (np.abs(nE + nP) + 1e-6), axis=1)

    # Remove individual peaks (To-Do: improve this part by sending boundary elements to the next process)
    dDOSm = np.concatenate(([0], np.max(np.abs(DOS[1:ne-1, :] / (DOS[0:ne-2, :] + 1)), axis=1), [0]))
    dDOSp = np.concatenate(([0], np.max(np.abs(DOS[1:ne-1, :] / (DOS[2:ne, :] + 1)), axis=1), [0]))

    # Find indices of elements satisfying the conditions
    ind_zeros = np.where((F1 > 0.1) | (F2 > 0.1) | ((dDOSm > 5) & (dDOSp > 5)))[0]

    for index in ind_zeros:
        GR_3D_E[index, :, :, :] = 0
        GRnn1_3D_E[index, :, :, :] = 0
        GL_3D_E[index, :, :, :] = 0
        GLnn1_3D_E[index, :, :, :] = 0
        GG_3D_E[index, :, :, :] = 0
        GGnn1_3D_E[index, :, :, :] = 0

    return GR_3D_E, GRnn1_3D_E, GL_3D_E, GLnn1_3D_E, GG_3D_E, GGnn1_3D_E


def calc_GF_mpi(
        DH,
        energy: npt.NDArray[np.float64],
        SigR,
        SigL,
        SigG,
        Efl,
        Efr,
        Temp,
        DOS,
        factor: npt.NDArray[np.float64],
        mkl_threads: int = 1,
        worker_num: int = 1
):

    kB = 1.38e-23
    q = 1.6022e-19
    
    UT = kB * Temp / q
    
    vfermi = np.vectorize(fermi_function)
    fL = vfermi(energy, Efl, UT)
    fR = vfermi(energy, Efr, UT)

    # initialize the Green's function in block format with zero
    # number of energy points
    ne = energy.shape[0]
    # number of blocks
    nb = DH.Bmin.shape[0]
    # length of the largest block
    lb = np.max(DH.Bmax - DH.Bmin + 1)
    # init
    (GR_3D_E, GRnn1_3D_E, GL_3D_E, GLnn1_3D_E, GG_3D_E, GGnn1_3D_E) = initialize_block_G(ne, nb, lb)

    mkl.set_num_threads(mkl_threads)

    rgf_M = generator_rgf_Hamiltonian(energy, DH, SigR)
    
    index_e = np.arange(ne)
    bmin = DH.Bmin.copy()
    bmax = DH.Bmax.copy()

    # Create a process pool with 4 workers
    for ie in range(ne):
        rgf_GF(next(rgf_M), SigL[ie], SigG[ie],
                          GR_3D_E[ie], GRnn1_3D_E[ie],
                          GL_3D_E[ie], GLnn1_3D_E[ie],
                          GG_3D_E[ie], GGnn1_3D_E[ie],
                          DOS[ie], fL[ie], fR[ie],
                          bmin, bmax, factor[ie], index_e[ie])
    
    return GR_3D_E, GRnn1_3D_E, GL_3D_E, GLnn1_3D_E, GG_3D_E, GGnn1_3D_E

def generator_rgf_Hamiltonian(E, DH, SigR):
    for i in range(E.shape[0]):
        yield (E[i] + 1j*1e-12) * DH.Overlap['H_4'] - DH.Hamiltonian['H_4'] - SigR[i]


def assemble_full_G_smoothing(G, factor, G_block, Gnn1_block, Bmin, Bmax, format = 'sparse', type = 'R'):
    G_temp  = factor * mat_assembly_fullG(G_block, Gnn1_block, Bmin, Bmax, format = format, type = type)
    G[:,:] = G_temp[:,:]