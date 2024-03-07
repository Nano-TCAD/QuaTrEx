# Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.

import concurrent.futures
from itertools import repeat

import time

import numpy as np
import numpy.typing as npt
import mkl

import cupy as cp
import cupyx as cpx

from quatrex.utils.matrix_creation import homogenize_matrix_Rnosym, extract_small_matrix_blocks
from quatrex.GreensFunction.fermi import fermi_function
from quatrex.GreensFunction.self_energy_preprocess import self_energy_preprocess_2d

import quatrex.block_tri_solvers.rgf_GF_GPU_combo as rgf_GF_GPU_combo
from quatrex.OBC.obc_gf_cpu import obc_GF_cpu
from quatrex.OBC.obc_gf_gpu import obc_GF_gpu

from operator import mul


import time


def calc_GF_pool_mpi_split_memopt(
    DH,
    energy: npt.NDArray[np.float64],
    SigR,
    SigL,
    SigG,
    SigR_ephn,
    SigL_ephn,
    SigG_ephn,
    sr_h2g,
    sl_h2g,
    sg_h2g,
    sr_phn,
    sl_phn,
    sg_phn,
    gr_h2g,
    gl_h2g,
    gg_h2g,
    map_diag,
    map_upper,
    map_lower,
    rows,
    columns,
    ij2ji,
    Efl,
    Efr,
    Temp,
    DOS,
    nE,
    nP,
    idE,
    factor: npt.NDArray[np.float64],
    comm,
    rank,
    size,
    homogenize=True,
    NCpSC: int = 1,
    mkl_threads: int = 1,
    worker_num: int = 1        
):
    comm.Barrier()
    if rank == 0:
        time_pre_OBC = -time.perf_counter()
    
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

    mkl.set_num_threads(mkl_threads)

    index_e = np.arange(ne)
    bmin = DH.Bmin.copy()
    bmax = DH.Bmax.copy()

    bmin_fi = bmin -1
    bmax_fi = bmax -1

    LBsize = bmax[0] - bmin[0] + 1
    RBsize = bmax[nb - 1] - bmin[nb - 1] + 1

    SigRBL = cpx.zeros_pinned((ne, LBsize, LBsize), dtype = np.complex128)
    SigRBR = cpx.zeros_pinned((ne, RBsize, RBsize), dtype = np.complex128)
    SigLBL = cpx.zeros_pinned((ne, LBsize, LBsize), dtype = np.complex128)
    SigLBR = cpx.zeros_pinned((ne, RBsize, RBsize), dtype = np.complex128)
    SigGBL = cpx.zeros_pinned((ne, LBsize, LBsize), dtype = np.complex128)
    SigGBR = cpx.zeros_pinned((ne, RBsize, RBsize), dtype = np.complex128)
    condl = np.zeros((ne), dtype = np.float64)
    condr = np.zeros((ne), dtype = np.float64)


    rgf_M_0 = generator_rgf_GF(energy, DH)
    index_e = np.arange(ne)
    bmin = DH.Bmin.copy()
    bmax = DH.Bmax.copy()

    # for ie in range(ne):
    #     #SigL[ie] = 1j * np.imag(SigL[ie])
    #     #SigG[ie] = 1j * np.imag(SigG[ie])

    #     SigL[ie] = (SigL[ie] - SigL[ie].T.conj()) / 2
    #     SigG[ie] = (SigG[ie] - SigG[ie].T.conj()) / 2
    #     #SigR[ie] = np.real(SigR[ie]) + 1j * np.imag(SigG[ie] - SigL[ie]) / 2
    #     SigR[ie] = np.real(SigR[ie]) + (SigG[ie] - SigL[ie]) / 2
    #     #SigR[ie] = (SigR[ie] + SigR[ie].T) / 2

    #     SigL[ie] += SigL_ephn[ie]
    #     SigG[ie] += SigG_ephn[ie]
    #     SigR[ie] += SigR_ephn[ie]

    #     if homogenize:
    #         (SigR00, SigR01, SigR10, _) = extract_small_matrix_blocks(SigR[ie][bmin[0] - 1:bmax[0], bmin[0] - 1:bmax[0]],\
    #                                                                     SigR[ie][bmin[0] - 1:bmax[0], bmin[1] - 1:bmax[1]], \
    #                                                                     SigR[ie][bmin[1] - 1:bmax[1], bmin[0] - 1:bmax[0]], NCpSC, 'L')
    #         SigR[ie] = homogenize_matrix_Rnosym(SigR00,
    #                                             SigR01, 
    #                                             SigR10, 
    #                                             len(bmax))
    #         (SigL00, SigL01, SigL10, _) = extract_small_matrix_blocks(SigL[ie][bmin[0] - 1:bmax[0], bmin[0] - 1:bmax[0]],\
    #                                                                     SigL[ie][bmin[0] - 1:bmax[0], bmin[1] - 1:bmax[1]], \
    #                                                                     SigL[ie][bmin[1] - 1:bmax[1], bmin[0] - 1:bmax[0]], NCpSC, 'L')
    #         SigL[ie] = homogenize_matrix_Rnosym(SigL00,
    #                                         SigL01,
    #                                         SigL10,
    #                                         len(bmax))
    #         (SigG00, SigG01, SigG10, _) = extract_small_matrix_blocks(SigG[ie][bmin[0] - 1:bmax[0], bmin[0] - 1:bmax[0]],\
    #                                                                     SigG[ie][bmin[0] - 1:bmax[0], bmin[1] - 1:bmax[1]], \
    #                                                                     SigG[ie][bmin[1] - 1:bmax[1], bmin[0] - 1:bmax[0]], NCpSC, 'L')
    #         SigG[ie] = homogenize_matrix_Rnosym(SigG00,
    #                                         SigG01, SigG10, len(bmax))
    comm.Barrier() 

    if rank == 0:
        time_pre_OBC += time.perf_counter()
        print("Time for pre-processing OBC: %.3f s" % time_pre_OBC, flush = True)
        time_OBC = -time.perf_counter()
                
    with concurrent.futures.ThreadPoolExecutor(max_workers=worker_num) as executor:
        # executor.map(self_energy_preprocess, SigL, SigG, SigR, SigL_ephn, SigG_ephn, SigR_ephn,
        #              repeat(NCpSC), repeat(bmin), repeat(bmax), repeat(homogenize))
        #results = 
        executor.map(obc_GF_gpu, rgf_M_0,
           SigR,
           fL,
           fR,
           SigRBL, SigRBR, SigLBL, SigLBR, SigGBL, SigGBR,
           repeat(bmin),
           repeat(bmax),
           repeat(NCpSC))
        #for idx, res in enumerate(results):
        #    condl[idx] = res[0]
        #    condr[idx] = res[1]
        
    # for ie in range(ne):
    #     condl[ie], condr[ie] = obc_GF_gpu(next(rgf_M_0),
    #         SigR[ie],
    #         fL[ie],
    #         fR[ie],
    #         SigRBL[ie], SigRBR[ie], SigLBL[ie], SigLBR[ie], SigGBL[ie], SigGBR[ie],
    #         bmin,
    #         bmax,
    #         NCpSC)
    
    comm.Barrier()
    if rank == 0:
        time_OBC += time.perf_counter()
        print("Time for OBC: %.3f s" % time_OBC, flush = True)
        time_GF_trafo = -time.perf_counter()

    l_defect = np.count_nonzero(np.isnan(condl))
    r_defect = np.count_nonzero(np.isnan(condr))

    if l_defect > 0 or r_defect > 0:
        print("Warning: %d left and %d right boundary conditions are not satisfied." % (l_defect, r_defect))

    sl_rgf, sg_rgf, sr_rgf = self_energy_preprocess_2d(sl_h2g, sg_h2g, sr_h2g, sl_phn, sg_phn, sr_phn, rows, columns, ij2ji,  NCpSC, bmin, bmax, False)
    mapping_diag = rgf_GF_GPU_combo.map_to_mapping(map_diag, nb)
    mapping_upper = rgf_GF_GPU_combo.map_to_mapping(map_upper, nb-1)
    mapping_lower = rgf_GF_GPU_combo.map_to_mapping(map_lower, nb-1)

    hamiltonian_diag, hamiltonian_upper, hamiltonian_lower = rgf_GF_GPU_combo.csr_to_block_tridiagonal_csr(DH.Hamiltonian['H_4'], bmin - 1, bmax)
    overlap_diag, overlap_upper, overlap_lower = rgf_GF_GPU_combo.csr_to_block_tridiagonal_csr(DH.Overlap['H_4'], bmin - 1, bmax)
    input_stream = cp.cuda.stream.Stream(non_blocking=True)

    comm.Barrier()
    if rank == 0:
        time_GF_trafo += time.perf_counter()
        print("Time for GF transformation: %.3f s" % time_GF_trafo, flush = True)
        time_SE = -time.perf_counter()


    energy_batchsize = 10
    energy_batch = np.arange(0, ne, energy_batchsize)

    comm.Barrier()
    if rank == 0:
        time_SE += time.perf_counter()
        print("Time for SE subtraction: %.3f s" % time_SE, flush = True)
        time_GF = -time.perf_counter()
    for ie in energy_batch:
        rgf_GF_GPU_combo.rgf_batched_GPU(energy[ie:ie+energy_batchsize],
                            mapping_diag, mapping_upper, mapping_lower,
                            hamiltonian_diag, hamiltonian_upper, hamiltonian_lower,
                            overlap_diag, overlap_upper, overlap_lower,
                            sr_rgf[ie:ie+energy_batchsize, :], sl_rgf[ie:ie+energy_batchsize, :], sg_rgf[ie:ie+energy_batchsize, :],
                            SigRBL[ie:ie+energy_batchsize, :, :], SigRBR[ie:ie+energy_batchsize, :, :],
                            SigLBL[ie:ie+energy_batchsize, :, :], SigLBR[ie:ie+energy_batchsize, :, :],
                            SigGBL[ie:ie+energy_batchsize, :, :], SigGBR[ie:ie+energy_batchsize, :, :],
                            gr_h2g[ie:ie+energy_batchsize, :], gl_h2g[ie:ie+energy_batchsize, :], gg_h2g[ie:ie+energy_batchsize, :],
                            DOS[ie:ie+energy_batchsize, :], nE[ie:ie+energy_batchsize, :],
                            nP[ie:ie+energy_batchsize, :], idE[ie:ie+energy_batchsize, :], bmin, bmax, solve = True,
                            input_stream = input_stream)
        
    comm.Barrier()
    if rank == 0:
        time_GF += time.perf_counter()
        print("Time for GF: %.3f s" % time_GF, flush = True)
        time_post_proc = -time.perf_counter()

    
    # Calculate F1, F2, which are the relative errors of GR-GA = GG-GL
    F1 = np.max(np.abs(DOS - (nE + nP)) / (np.abs(DOS) + 1e-6), axis=1)
    F2 = np.max(np.abs(DOS - (nE + nP)) / (np.abs(nE + nP) + 1e-6), axis=1)

    buf_recv_r = np.empty((DOS.shape[1]), dtype=np.complex128)
    buf_send_r = np.empty((DOS.shape[1]), dtype=np.complex128)
    buf_recv_l = np.empty((DOS.shape[1]), dtype=np.complex128)
    buf_send_l = np.empty((DOS.shape[1]), dtype=np.complex128)
    if size > 1:
        if rank == 0:
            buf_send_r[:] = DOS[ne - 1, :]
            comm.Sendrecv(sendbuf=buf_send_r, dest=rank + 1, recvbuf=buf_recv_r, source=rank + 1)

        elif rank == size - 1:
            buf_send_l[:] = DOS[0, :]
            comm.Sendrecv(sendbuf=buf_send_l, dest=rank - 1, recvbuf=buf_recv_l, source=rank - 1)
        else:
            buf_send_r[:] = DOS[ne - 1, :]
            buf_send_l[:] = DOS[0, :]
            comm.Sendrecv(sendbuf=buf_send_r, dest=rank + 1, recvbuf=buf_recv_r, source=rank + 1)
            comm.Sendrecv(sendbuf=buf_send_l, dest=rank - 1, recvbuf=buf_recv_l, source=rank - 1)

    # Remove individual peaks (To-Do: improve this part by sending boundary elements to the next process)
    if size == 1:
        dDOSm = np.concatenate(([0], np.max(np.abs(DOS[1:ne - 1, :] / (DOS[0:ne - 2, :] + 1)),
                                            axis=1), [np.max(np.abs(DOS[ne - 1, :] / (DOS[ne - 2, :] + 1)))]))
        dDOSp = np.concatenate(
            ([np.max(np.abs(DOS[0, :] / (DOS[1, :] + 1)))], np.max(np.abs(DOS[1:ne - 1, :] / (DOS[2:ne, :] + 1)),
                                                                   axis=1), [0]))
    elif rank == 0:
        dDOSm = np.concatenate(([0], np.max(np.abs(DOS[1:ne - 1, :] / (DOS[0:ne - 2, :] + 1)),
                                            axis=1), [np.max(np.abs(DOS[ne - 1, :] / (DOS[ne - 2, :] + 1)))]))
        dDOSp = np.concatenate(([np.max(np.abs(DOS[0, :] / (DOS[1, :] + 1)))],
                                np.max(np.abs(DOS[1:ne - 1, :] / (DOS[2:ne, :] + 1)),
                                       axis=1), [np.max(np.abs(DOS[ne - 1, :] / (buf_recv_r + 1)))]))
    elif rank == size - 1:
        dDOSm = np.concatenate(([np.max(np.abs(DOS[0, :] / (buf_recv_l + 1)))],
                                np.max(np.abs(DOS[1:ne - 1, :] / (DOS[0:ne - 2, :] + 1)),
                                       axis=1), [np.max(np.abs(DOS[ne - 1, :] / (DOS[ne - 2, :] + 1)))]))
        dDOSp = np.concatenate(
            ([np.max(np.abs(DOS[0, :] / (DOS[1, :] + 1)))], np.max(np.abs(DOS[1:ne - 1, :] / (DOS[2:ne, :] + 1)),
                                                                   axis=1), [0]))
    else:
        dDOSm = np.concatenate(([np.max(np.abs(DOS[0, :] / (buf_recv_l + 1)))],
                                np.max(np.abs(DOS[1:ne - 1, :] / (DOS[0:ne - 2, :] + 1)),
                                       axis=1), [np.max(np.abs(DOS[ne - 1, :] / (DOS[ne - 2, :] + 1)))]))
        dDOSp = np.concatenate(([np.max(np.abs(DOS[0, :] / (DOS[1, :] + 1)))],
                                np.max(np.abs(DOS[1:ne - 1, :] / (DOS[2:ne, :] + 1)),
                                       axis=1), [np.max(np.abs(DOS[ne - 1, :] / (buf_recv_r + 1)))]))

    # Find indices of elements satisfying the conditions
    ind_zeros = np.where((F1 > 0.1) | (F2 > 0.1) | ((dDOSm > 5) & (dDOSp > 5)))[0]

    for index in ind_zeros:
        gr_h2g[index, :] = 0
        gl_h2g[index, :] = 0
        gg_h2g[index, :] = 0
        
    comm.Barrier()
    if rank == 0:
        time_post_proc += time.perf_counter()
        print("Time for post-processing: %.3f s" % time_post_proc, flush = True)
    
def generator_rgf_GF(E, DH):
    for i in range(E.shape[0]):
        yield (E[i] + 1j * 1e-12) * DH.Overlap['H_4'] - DH.Hamiltonian['H_4']
