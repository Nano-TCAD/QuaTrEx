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
from quatrex.OBC.obc_gf_gpu_2 import obc_GF_gpu
from quatrex.OBC.beyn_batched import beyn_batched_gpu_3 as beyn_gpu

from operator import mul


import time


def calc_GF_pool_mpi_split_memopt(
    DH,
    hamiltonian_diag, hamiltonian_upper, hamiltonian_lower,
    overlap_diag, overlap_upper, overlap_lower,
    energy: npt.NDArray[np.float64],
    sr_rgf_dev,
    sl_rgf_dev,
    sg_rgf_dev,
    gr_h2g,
    gl_h2g,
    gg_h2g,
    mapping_diag_dev,
    mapping_upper_dev,
    mapping_lower_dev,
    # rows,
    # columns,
    # ij2ji,
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
    fL = cp.asarray(vfermi(energy, Efl, UT)).reshape(len(energy), 1, 1)
    fR = cp.asarray(vfermi(energy, Efr, UT)).reshape(len(energy), 1, 1)

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


    # rgf_M_0 = generator_rgf_GF(energy, DH)
    # index_e = np.arange(ne)
    bmin = DH.Bmin.copy()
    bmax = DH.Bmax.copy()

    # sl_rgf_dev = cp.asarray(sl_h2g)
    # sg_rgf_dev = cp.asarray(sg_h2g)
    # sr_rgf_dev = cp.asarray(sr_h2g)
    # sl_phn_dev = cp.asarray(sl_phn)
    # sg_phn_dev = cp.asarray(sg_phn)
    # sr_phn_dev = cp.asarray(sr_phn)
    # rgf_GF_GPU_combo.self_energy_preprocess_2d(sl_rgf_dev, sg_rgf_dev, sr_rgf_dev, sl_phn_dev, sg_phn_dev, sr_phn_dev, cp.asarray(rows), cp.asarray(columns), cp.asarray(ij2ji))
    # # NCpSC, bmin, bmax, False)
    # mapping_diag = rgf_GF_GPU_combo.map_to_mapping(map_diag, nb)
    # mapping_upper = rgf_GF_GPU_combo.map_to_mapping(map_upper, nb-1)
    # mapping_lower = rgf_GF_GPU_combo.map_to_mapping(map_lower, nb-1)
    # mapping_diag_dev = cp.asarray(mapping_diag)
    # mapping_upper_dev = cp.asarray(mapping_upper)
    # mapping_lower_dev = cp.asarray(mapping_lower)

    # hamiltonian_diag, hamiltonian_upper, hamiltonian_lower = rgf_GF_GPU_combo.csr_to_block_tridiagonal_csr(DH.Hamiltonian['H_4'], bmin - 1, bmax)
    # overlap_diag, overlap_upper, overlap_lower = rgf_GF_GPU_combo.csr_to_block_tridiagonal_csr(DH.Overlap['H_4'], bmin - 1, bmax)
    input_stream = cp.cuda.stream.Stream(non_blocking=True)

    M00_left = cp.empty((ne, LBsize, LBsize), dtype = np.complex128)
    M01_left = cp.empty((ne, LBsize, RBsize), dtype = np.complex128)
    M10_left = cp.empty((ne, RBsize, LBsize), dtype = np.complex128)
    M00_right = cp.empty((ne, RBsize, RBsize), dtype = np.complex128)
    M01_right = cp.empty((ne, RBsize, LBsize), dtype = np.complex128)
    M10_right = cp.empty((ne, LBsize, RBsize), dtype = np.complex128)
    rgf_GF_GPU_combo._get_dense_block_batch(sr_rgf_dev, mapping_diag_dev, 0, M00_left)
    rgf_GF_GPU_combo._get_dense_block_batch(sr_rgf_dev, mapping_upper_dev, 0, M01_left)
    rgf_GF_GPU_combo._get_dense_block_batch(sr_rgf_dev, mapping_lower_dev, 0, M10_left)
    rgf_GF_GPU_combo._get_dense_block_batch(sr_rgf_dev, mapping_diag_dev, nb-1, M00_right)
    rgf_GF_GPU_combo._get_dense_block_batch(sr_rgf_dev, mapping_upper_dev, nb-2, M01_right)
    rgf_GF_GPU_combo._get_dense_block_batch(sr_rgf_dev, mapping_lower_dev, nb-2, M10_right)

    csr_matrix = rgf_GF_GPU_combo.csr_matrix
    hdtype = np.complex128
    block_size = max(LBsize, RBsize)

    H_diag_buffer = [csr_matrix(cp.empty(block_size * block_size, hdtype),
                                cp.empty(block_size * block_size, cp.int32),
                                cp.empty(block_size + 1, cp.int32),) for _ in range(2)]
    H_upper_buffer = [csr_matrix(cp.empty(block_size * block_size, hdtype),
                                 cp.empty(block_size * block_size, cp.int32),
                                 cp.empty(block_size + 1, cp.int32),) for _ in range(2)]
    H_lower_buffer = [csr_matrix(cp.empty(block_size * block_size, hdtype),
                                 cp.empty(block_size * block_size, cp.int32),
                                 cp.empty(block_size + 1, cp.int32),) for _ in range(2)]
    S_diag_buffer = [csr_matrix(cp.empty(block_size * block_size, hdtype),
                                cp.empty(block_size * block_size, cp.int32),
                                cp.empty(block_size + 1, cp.int32),) for _ in range(2)]
    S_upper_buffer = [csr_matrix(cp.empty(block_size * block_size, hdtype),
                                 cp.empty(block_size * block_size, cp.int32),
                                 cp.empty(block_size + 1, cp.int32),) for _ in range(2)]
    S_lower_buffer = [csr_matrix(cp.empty(block_size * block_size, hdtype),
                                 cp.empty(block_size * block_size, cp.int32),
                                 cp.empty(block_size + 1, cp.int32),) for _ in range(2)]

    batch_size = len(energy)
    block_size = LBsize
    num_threads = min(1024, block_size)
    num_thread_blocks = batch_size * block_size
    energy_dev = cp.asarray(energy)
    hd, sd = H_diag_buffer[0], S_diag_buffer[0]
    hu, su = H_upper_buffer[0], S_upper_buffer[0]
    hl, sl = H_lower_buffer[0], S_lower_buffer[0]
    rgf_GF_GPU_combo._copy_csr_to_gpu(hamiltonian_diag[0], hd)
    rgf_GF_GPU_combo._copy_csr_to_gpu(overlap_diag[0], sd)
    rgf_GF_GPU_combo._copy_csr_to_gpu(hamiltonian_upper[0], hu)
    rgf_GF_GPU_combo._copy_csr_to_gpu(overlap_upper[0], su)
    rgf_GF_GPU_combo._copy_csr_to_gpu(hamiltonian_lower[0], hl)
    rgf_GF_GPU_combo._copy_csr_to_gpu(overlap_lower[0], sl)
    block_size = LBsize
    num_threads = min(1024, block_size)
    num_thread_blocks = batch_size * block_size
    rgf_GF_GPU_combo._get_system_matrix[num_thread_blocks, num_threads](
        energy_dev, hd.data, hd.indices, hd.indptr, sd.data, sd.indices, sd.indptr, M00_left, M00_left, batch_size, block_size)
    rgf_GF_GPU_combo._get_system_matrix[num_thread_blocks, num_threads](
        energy_dev, hu.data, hu.indices, hu.indptr, su.data, su.indices, su.indptr, M01_left, M01_left, batch_size, block_size)
    rgf_GF_GPU_combo._get_system_matrix[num_thread_blocks, num_threads](
        energy_dev, hl.data, hl.indices, hl.indptr, sl.data, sl.indices, sl.indptr, M10_left, M10_left, batch_size, block_size)
    hd, sd = H_diag_buffer[-1], S_diag_buffer[-1]
    hu, su = H_upper_buffer[-1], S_upper_buffer[-1]
    hl, sl = H_lower_buffer[-1], S_lower_buffer[-1]
    rgf_GF_GPU_combo._copy_csr_to_gpu(hamiltonian_diag[-1], hd)
    rgf_GF_GPU_combo._copy_csr_to_gpu(overlap_diag[-1], sd)
    rgf_GF_GPU_combo._copy_csr_to_gpu(hamiltonian_upper[-1], hu)
    rgf_GF_GPU_combo._copy_csr_to_gpu(overlap_upper[-1], su)
    rgf_GF_GPU_combo._copy_csr_to_gpu(hamiltonian_lower[-1], hl)
    rgf_GF_GPU_combo._copy_csr_to_gpu(overlap_lower[-1], sl)
    block_size = RBsize
    num_threads = min(1024, block_size)
    num_thread_blocks = batch_size * block_size
    rgf_GF_GPU_combo._get_system_matrix[num_thread_blocks, num_threads](
        energy_dev, hd.data, hd.indices, hd.indptr, sd.data, sd.indices, sd.indptr, M00_right, M00_right, batch_size, block_size)
    rgf_GF_GPU_combo._get_system_matrix[num_thread_blocks, num_threads](
        energy_dev, hu.data, hu.indices, hu.indptr, su.data, su.indices, su.indptr, M01_right, M01_right, batch_size, block_size)
    rgf_GF_GPU_combo._get_system_matrix[num_thread_blocks, num_threads](
        energy_dev, hl.data, hl.indices, hl.indptr, sl.data, sl.indices, sl.indptr, M10_right, M10_right, batch_size, block_size)
    cp.cuda.Stream.null.synchronize()

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
        executor.map(obc_GF_gpu, M00_left, M01_left, M10_left, M00_right, M01_right, M10_right,
           fL,
           fR,
           SigRBL, SigRBR, SigLBL, SigLBR, SigGBL, SigGBR,
           repeat(bmin),
           repeat(bmax),
           repeat(NCpSC))
    
    # imag_lim = 5e-4
    # R = 1000
    # SigRBL_gpu, _, condL, _ = beyn_gpu(NCpSC, M00_left, M01_left, M10_left, imag_lim, R, 'L')
    # assert not any(np.isnan(cond) for cond in condL)
    # GammaL = 1j * (SigRBL_gpu - SigRBL_gpu.transpose(0, 2, 1).conj())
    # (1j * fL * GammaL).get(out=SigLBL)
    # (1j * (fL - 1) * GammaL).get(out=SigGBL)
    # SigRBL_gpu.get(out=SigRBL)
    # SigRBR_gpu, _, condR, _ = beyn_gpu(NCpSC, M00_right, M01_right, M10_right, imag_lim, R, 'R')
    # assert not any(np.isnan(cond) for cond in condR)
    # GammaR = 1j * (SigRBR_gpu - SigRBR_gpu.transpose(0, 2, 1).conj())
    # (1j * fR * GammaR).get(out=SigLBR)
    # (1j * (fR - 1) * GammaR).get(out=SigGBR)
    # SigRBR_gpu.get(out=SigRBR)
    
    comm.Barrier()
    if rank == 0:
        time_OBC += time.perf_counter()
        print("Time for OBC: %.3f s" % time_OBC, flush = True)
        time_GF_trafo = -time.perf_counter()

    l_defect = np.count_nonzero(np.isnan(condl))
    r_defect = np.count_nonzero(np.isnan(condr))

    if l_defect > 0 or r_defect > 0:
        print("Warning: %d left and %d right boundary conditions are not satisfied." % (l_defect, r_defect))

    # sl_rgf, sg_rgf, sr_rgf = self_energy_preprocess_2d(sl_h2g, sg_h2g, sr_h2g, sl_phn, sg_phn, sr_phn, rows, columns, ij2ji,  NCpSC, bmin, bmax, False)
    # mapping_diag = rgf_GF_GPU_combo.map_to_mapping(map_diag, nb)
    # mapping_upper = rgf_GF_GPU_combo.map_to_mapping(map_upper, nb-1)
    # mapping_lower = rgf_GF_GPU_combo.map_to_mapping(map_lower, nb-1)

    # hamiltonian_diag, hamiltonian_upper, hamiltonian_lower = rgf_GF_GPU_combo.csr_to_block_tridiagonal_csr(DH.Hamiltonian['H_4'], bmin - 1, bmax)
    # overlap_diag, overlap_upper, overlap_lower = rgf_GF_GPU_combo.csr_to_block_tridiagonal_csr(DH.Overlap['H_4'], bmin - 1, bmax)
    # input_stream = cp.cuda.stream.Stream(non_blocking=True)

    comm.Barrier()
    if rank == 0:
        time_GF_trafo += time.perf_counter()
        print("Time for GF transformation: %.3f s" % time_GF_trafo, flush = True)
        time_SE = -time.perf_counter()


    energy_batchsize = 7
    energy_batch = np.arange(0, ne, energy_batchsize)

    comm.Barrier()
    if rank == 0:
        time_SE += time.perf_counter()
        print("Time for SE subtraction: %.3f s" % time_SE, flush = True)
        time_GF = -time.perf_counter()
    for ie in energy_batch:
        rgf_GF_GPU_combo.rgf_batched_GPU(energy[ie:ie+energy_batchsize],
                            mapping_diag_dev, mapping_upper_dev, mapping_lower_dev,
                            hamiltonian_diag, hamiltonian_upper, hamiltonian_lower,
                            overlap_diag, overlap_upper, overlap_lower,
                            sr_rgf_dev[ie:ie+energy_batchsize, :], sl_rgf_dev[ie:ie+energy_batchsize, :], sg_rgf_dev[ie:ie+energy_batchsize, :],
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
