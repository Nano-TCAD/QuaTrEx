# Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.

import concurrent.futures
from itertools import repeat

import time

import numpy as np
import numpy.typing as npt
from scipy import sparse
import mkl

from quatrex.utils.matrix_creation import initialize_block_G, mat_assembly_fullG, homogenize_matrix, \
    homogenize_matrix_Rnosym, extract_small_matrix_blocks
from quatrex.GreensFunction.fermi import fermi_function
from quatrex.GreensFunction.self_energy_preprocess import self_energy_preprocess
from quatrex.block_tri_solvers.rgf_GF import rgf_GF, rgf_standaloneGF
from quatrex.OBC.obc_gf_cpu import obc_GF_cpu

from operator import mul

import copy


def calc_GF_pool_mpi_split(
    DH,
    energy: npt.NDArray[np.float64],
    SigR,
    SigL,
    SigG,
    SigR_ephn,
    SigL_ephn,
    SigG_ephn,
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
    return_sigma_boundary=False,
    mkl_threads: int = 1,
    worker_num: int = 1,
    block_inv: bool = False,
    use_dace: bool = False,
    validate_dace: bool = False,

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
    (GR_3D_E, GRnn1_3D_E, GL_3D_E, GLnn1_3D_E, GG_3D_E,
     GGnn1_3D_E) = initialize_block_G(ne, nb, lb)

    mkl.set_num_threads(mkl_threads)

    index_e = np.arange(ne)
    bmin = DH.Bmin.copy()
    bmax = DH.Bmax.copy()

    LBsize = bmax[0] - bmin[0] + 1
    RBsize = bmax[nb - 1] - bmin[nb - 1] + 1

    SigRBL = np.zeros((ne, LBsize, LBsize), dtype=np.complex128)
    SigRBR = np.zeros((ne, RBsize, RBsize), dtype=np.complex128)
    SigLBL = np.zeros((ne, LBsize, LBsize), dtype=np.complex128)
    SigLBR = np.zeros((ne, RBsize, RBsize), dtype=np.complex128)
    SigGBL = np.zeros((ne, LBsize, LBsize), dtype=np.complex128)
    SigGBR = np.zeros((ne, RBsize, RBsize), dtype=np.complex128)
    condl = np.zeros((ne), dtype=np.float64)
    condr = np.zeros((ne), dtype=np.float64)

    rgf_M_0 = generator_rgf_GF(energy, DH)
    index_e = np.arange(ne)
    bmin = DH.Bmin.copy()
    bmax = DH.Bmax.copy()

    for ie in range(ne):
        # SigL[ie] = 1j * np.imag(SigL[ie])
        # SigG[ie] = 1j * np.imag(SigG[ie])

        SigL[ie] = (SigL[ie] - SigL[ie].T.conj()) / 2
        SigG[ie] = (SigG[ie] - SigG[ie].T.conj()) / 2
        # SigR[ie] = np.real(SigR[ie]) + 1j * np.imag(SigG[ie] - SigL[ie]) / 2
        SigR[ie] = np.real(SigR[ie]) + (SigG[ie] - SigL[ie]) / 2
        # SigR[ie] = (SigR[ie] + SigR[ie].T) / 2

        SigL[ie] += SigL_ephn[ie]
        SigG[ie] += SigG_ephn[ie]
        SigR[ie] += SigR_ephn[ie]

        if homogenize:
            (SigR00, SigR01, SigR10, _) = extract_small_matrix_blocks(SigR[ie][bmin[0] - 1:bmax[0], bmin[0] - 1:bmax[0]],
                                                                      SigR[ie][bmin[0] - 1:bmax[0],
                                                                               bmin[1] - 1:bmax[1]],
                                                                      SigR[ie][bmin[1] - 1:bmax[1], bmin[0] - 1:bmax[0]], NCpSC, 'L')
            SigR[ie] = homogenize_matrix_Rnosym(SigR00,
                                                SigR01,
                                                SigR10,
                                                len(bmax))
            (SigL00, SigL01, SigL10, _) = extract_small_matrix_blocks(SigL[ie][bmin[0] - 1:bmax[0], bmin[0] - 1:bmax[0]],
                                                                      SigL[ie][bmin[0] - 1:bmax[0],
                                                                               bmin[1] - 1:bmax[1]],
                                                                      SigL[ie][bmin[1] - 1:bmax[1], bmin[0] - 1:bmax[0]], NCpSC, 'L')
            SigL[ie] = homogenize_matrix_Rnosym(SigL00,
                                                SigL01,
                                                SigL10,
                                                len(bmax))
            (SigG00, SigG01, SigG10, _) = extract_small_matrix_blocks(SigG[ie][bmin[0] - 1:bmax[0], bmin[0] - 1:bmax[0]],
                                                                      SigG[ie][bmin[0] - 1:bmax[0],
                                                                               bmin[1] - 1:bmax[1]],
                                                                      SigG[ie][bmin[1] - 1:bmax[1], bmin[0] - 1:bmax[0]], NCpSC, 'L')
            SigG[ie] = homogenize_matrix_Rnosym(SigG00,
                                                SigG01, SigG10, len(bmax))

    comm.Barrier()

    if rank == 0:
        time_pre_OBC += time.perf_counter()
        print("Time for pre-processing OBC: %.3f s" % time_pre_OBC, flush=True)
        time_OBC = -time.perf_counter()

    with concurrent.futures.ThreadPoolExecutor(max_workers=worker_num) as executor:
        results = executor.map(obc_GF_cpu, rgf_M_0,
                               SigR,
                               fL,
                               fR,
                               SigRBL, SigRBR, SigLBL, SigLBR, SigGBL, SigGBR,
                               repeat(bmin),
                               repeat(bmax),
                               repeat(NCpSC))
        for idx, res in enumerate(results):
            condl[idx] = res[0]
            condr[idx] = res[1]
    comm.Barrier()
    if rank == 0:
        time_OBC += time.perf_counter()
        print("Time for OBC: %.3f s" % time_OBC, flush=True)
        time_GF_trafo = -time.perf_counter()

    rgf_M = generator_rgf_Hamiltonian(energy, DH, SigR)
    rgf_H = generator_rgf_currentdens_Hamiltonian(energy, DH)

    comm.Barrier()
    if rank == 0:
        time_GF_trafo += time.perf_counter()
        print("Time for GF transformation: %.3f s" % time_GF_trafo, flush=True)
        time_GF = -time.perf_counter()

    with concurrent.futures.ThreadPoolExecutor(max_workers=worker_num) as executor:
        executor.map(rgf_standaloneGF, rgf_M, rgf_H, SigL, SigG,
                     SigRBL, SigRBR, SigLBL, SigLBR, SigGBL, SigGBR,
                     condl, condr,
                     GR_3D_E, GRnn1_3D_E, GL_3D_E, GLnn1_3D_E, GG_3D_E, GGnn1_3D_E,
                     DOS, nE, nP, idE, repeat(bmin), repeat(
                         bmax), factor, index_e, repeat(NCpSC), repeat(block_inv),
                     repeat(use_dace), repeat(validate_dace))

    comm.Barrier()
    if rank == 0:
        time_GF += time.perf_counter()
        print("Time for GF: %.3f s" % time_GF, flush=True)
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
            comm.Sendrecv(sendbuf=buf_send_r, dest=rank + 1,
                          recvbuf=buf_recv_r, source=rank + 1)

        elif rank == size - 1:
            buf_send_l[:] = DOS[0, :]
            comm.Sendrecv(sendbuf=buf_send_l, dest=rank - 1,
                          recvbuf=buf_recv_l, source=rank - 1)
        else:
            buf_send_r[:] = DOS[ne - 1, :]
            buf_send_l[:] = DOS[0, :]
            comm.Sendrecv(sendbuf=buf_send_r, dest=rank + 1,
                          recvbuf=buf_recv_r, source=rank + 1)
            comm.Sendrecv(sendbuf=buf_send_l, dest=rank - 1,
                          recvbuf=buf_recv_l, source=rank - 1)

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
    ind_zeros = np.where((F1 > 0.1) | (F2 > 0.1) |
                         ((dDOSm > 5) & (dDOSp > 5)))[0]

    for index in ind_zeros:
        GR_3D_E[index, :, :, :] = 0
        GRnn1_3D_E[index, :, :, :] = 0
        GL_3D_E[index, :, :, :] = 0
        GLnn1_3D_E[index, :, :, :] = 0
        GG_3D_E[index, :, :, :] = 0
        GGnn1_3D_E[index, :, :, :] = 0

    comm.Barrier()
    if rank == 0:
        time_post_proc += time.perf_counter()
        print("Time for post-processing: %.3f s" % time_post_proc, flush=True)

    if (return_sigma_boundary):
        return GR_3D_E, GRnn1_3D_E, GL_3D_E, GLnn1_3D_E, GG_3D_E, GGnn1_3D_E, SigRBL, SigRBR
    else:
        return GR_3D_E, GRnn1_3D_E, GL_3D_E, GLnn1_3D_E, GG_3D_E, GGnn1_3D_E


def calc_GF_pool_mpi(
    DH,
    idx_k,
    energy: npt.NDArray[np.float64],
    SigR,
    SigL,
    SigG,
    SigR_ephn,
    SigL_ephn,
    SigG_ephn,
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
    return_sigma_boundary=False,
    mkl_threads: int = 1,
    worker_num: int = 1,
    block_inv: bool = False,
    use_dace: bool = False,
    validate_dace: bool = False,
):
    """k-points implemented. Can cause problems when merging"""
    kB = 1.38e-23
    q = 1.6022e-19

    UT = kB * Temp / q

    vfermi = np.vectorize(fermi_function)
    fL = vfermi(energy, Efl, UT)
    fR = vfermi(energy, Efr, UT)

    # initialize the Green's function in block format with zero
    # number of energy points x kpoints
    ne = energy.shape[0]
    # number of blocks
    nb = DH.Bmin.shape[0]
    # length of the largest block
    lb = np.max(DH.Bmax - DH.Bmin + 1)
    # init
    (GR_3D_E, GRnn1_3D_E, GL_3D_E, GLnn1_3D_E, GG_3D_E,
     GGnn1_3D_E) = initialize_block_G(ne, nb, lb)

    mkl.set_num_threads(mkl_threads)

    index_e = np.arange(ne)
    bmin = DH.Bmin.copy()
    bmax = DH.Bmax.copy()
    if (return_sigma_boundary):
        LBsize = bmax[0] - bmin[0] + 1
        RBsize = bmax[nb - 1] - bmin[nb - 1] + 1

        SigRBL = np.zeros((ne, LBsize, LBsize), dtype=np.complex128)
        SigRBR = np.zeros((ne, RBsize, RBsize), dtype=np.complex128)

    for ie in range(ne):
        # SigL[ie] = 1j * np.imag(SigL[ie])
        # SigG[ie] = 1j * np.imag(SigG[ie])

        SigL[ie] = (SigL[ie] - SigL[ie].T.conj()) / 2
        SigG[ie] = (SigG[ie] - SigG[ie].T.conj()) / 2
        # SigR[ie] = np.real(SigR[ie]) + 1j * np.imag(SigG[ie] - SigL[ie]) / 2
        SigR[ie] = np.real(SigR[ie]) + (SigG[ie] - SigL[ie]) / 2
        # SigR[ie] = (SigR[ie] + SigR[ie].T) / 2

        SigL[ie] += SigL_ephn[ie]
        SigG[ie] += SigG_ephn[ie]
        SigR[ie] += SigR_ephn[ie]

        if homogenize:
            (SigR00, SigR01, SigR10, _) = extract_small_matrix_blocks(SigR[ie][bmin[0] - 1:bmax[0], bmin[0] - 1:bmax[0]],
                                                                      SigR[ie][bmin[0] - 1:bmax[0],
                                                                               bmin[1] - 1:bmax[1]],
                                                                      SigR[ie][bmin[1] - 1:bmax[1], bmin[0] - 1:bmax[0]], NCpSC, 'L')
            SigR[ie] = homogenize_matrix_Rnosym(SigR00,
                                                SigR01,
                                                SigR10,
                                                len(bmax))
            (SigL00, SigL01, SigL10, _) = extract_small_matrix_blocks(SigL[ie][bmin[0] - 1:bmax[0], bmin[0] - 1:bmax[0]],
                                                                      SigL[ie][bmin[0] - 1:bmax[0],
                                                                               bmin[1] - 1:bmax[1]],
                                                                      SigL[ie][bmin[1] - 1:bmax[1], bmin[0] - 1:bmax[0]], NCpSC, 'L')
            SigL[ie] = homogenize_matrix_Rnosym(SigL00,
                                                SigL01,
                                                SigL10,
                                                len(bmax))
            (SigG00, SigG01, SigG10, _) = extract_small_matrix_blocks(SigG[ie][bmin[0] - 1:bmax[0], bmin[0] - 1:bmax[0]],
                                                                      SigG[ie][bmin[0] - 1:bmax[0],
                                                                               bmin[1] - 1:bmax[1]],
                                                                      SigG[ie][bmin[1] - 1:bmax[1], bmin[0] - 1:bmax[0]], NCpSC, 'L')
            SigG[ie] = homogenize_matrix_Rnosym(SigG00,
                                                SigG01, SigG10, len(bmax))

    rgf_M = generator_rgf_Hamiltonian(energy, idx_k, DH, SigR)
    rgf_H = generator_rgf_currentdens_Hamiltonian(energy, idx_k, DH)
    index_e = np.arange(ne)
    bmin = DH.Bmin.copy()
    bmax = DH.Bmax.copy()

    # Create a process pool with 4 workers
    with concurrent.futures.ThreadPoolExecutor(max_workers=worker_num) as executor:
        # Use the map function to apply the inv_matrices function to each pair of matrices in parallel
        # Use partial function application to bind the constant arguments to inv_matrices
        # Pass in an additional argument to inv_matrices that contains the index of the matrices pair
        # results = list(executor.map(lambda args: inv_matrices(args[0], const_arg1, const_arg2, args[1]), ((matrices_pairs[i], i) for i in range(len(matrices_pairs)))))
        # results = executor.map(rgf_GF, rgf_M, rgf_H, SigL, SigG,
        if (return_sigma_boundary):
            results = executor.map(rgf_GF, rgf_M, rgf_H, SigL, SigG, GR_3D_E, GRnn1_3D_E, GL_3D_E, GLnn1_3D_E, GG_3D_E, GGnn1_3D_E,
                                   DOS, nE, nP, idE, fL, fR, repeat(bmin), repeat(
                                       bmax), factor, index_e, repeat(NCpSC), repeat(block_inv),
                                   repeat(use_dace), repeat(validate_dace))
            for idx, res in enumerate(results):
                SigRBL[idx, :, :] = res[0]
                SigRBR[idx, :, :] = res[1]
        else:
            executor.map(rgf_GF, rgf_M, rgf_H, SigL, SigG, GR_3D_E, GRnn1_3D_E, GL_3D_E, GLnn1_3D_E, GG_3D_E, GGnn1_3D_E,
                         DOS, nE, nP, idE, fL, fR, repeat(bmin), repeat(
                             bmax), factor, index_e, repeat(NCpSC), repeat(block_inv),
                         repeat(use_dace), repeat(validate_dace))
        # for res in results:
        #    assert res == 0

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
            comm.Sendrecv(sendbuf=buf_send_r, dest=rank + 1,
                          recvbuf=buf_recv_r, source=rank + 1)

        elif rank == size - 1:
            buf_send_l[:] = DOS[0, :]
            comm.Sendrecv(sendbuf=buf_send_l, dest=rank - 1,
                          recvbuf=buf_recv_l, source=rank - 1)
        else:
            buf_send_r[:] = DOS[ne - 1, :]
            buf_send_l[:] = DOS[0, :]
            comm.Sendrecv(sendbuf=buf_send_r, dest=rank + 1,
                          recvbuf=buf_recv_r, source=rank + 1)
            comm.Sendrecv(sendbuf=buf_send_l, dest=rank - 1,
                          recvbuf=buf_recv_l, source=rank - 1)

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
    ind_zeros = np.where((F1 > 0.1) | (F2 > 0.1) |
                         ((dDOSm > 5) & (dDOSp > 5)))[0]

    for index in ind_zeros:
        GR_3D_E[index, :, :, :] = 0
        GRnn1_3D_E[index, :, :, :] = 0
        GL_3D_E[index, :, :, :] = 0
        GLnn1_3D_E[index, :, :, :] = 0
        GG_3D_E[index, :, :, :] = 0
        GGnn1_3D_E[index, :, :, :] = 0

    if (return_sigma_boundary):
        return GR_3D_E, GRnn1_3D_E, GL_3D_E, GLnn1_3D_E, GG_3D_E, GGnn1_3D_E, SigRBL, SigRBR
    else:
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
    nE,
    nP,
    idE,
    factor: npt.NDArray[np.float64],
    comm,
    rank,
    size,
    mkl_threads: int = 1,
    worker_num: int = 1,
    block_inv: bool = False,
    use_dace: bool = False,
    validate_dace: bool = False,
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
    (GR_3D_E, GRnn1_3D_E, GL_3D_E, GLnn1_3D_E, GG_3D_E,
     GGnn1_3D_E) = initialize_block_G(ne, nb, lb)

    mkl.set_num_threads(mkl_threads)

    rgf_M = generator_rgf_Hamiltonian(energy, DH, SigR)
    rgf_H = generator_rgf_currentdens_Hamiltonian(energy, DH)

    index_e = np.arange(ne)
    bmin = DH.Bmin.copy()
    bmax = DH.Bmax.copy()

    # Create a process pool with 4 workers
    for ie in range(ne):
        rgf_GF(next(rgf_M),
               next(rgf_H),
               SigL[ie],
               SigG[ie],
               GR_3D_E[ie],
               GRnn1_3D_E[ie],
               GL_3D_E[ie],
               GLnn1_3D_E[ie],
               GG_3D_E[ie],
               GGnn1_3D_E[ie],
               DOS[ie],
               nE,
               nP,
               idE,
               fL[ie],
               fR[ie],
               bmin,
               bmax,
               factor[ie],
               index_e[ie],
               block_inv=block_inv,
               use_dace=use_dace,
               validate_dace=validate_dace)

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
            comm.Sendrecv(sendbuf=buf_send_r, dest=rank + 1,
                          recvbuf=buf_recv_r, source=rank + 1)

        elif rank == size - 1:
            buf_send_l[:] = DOS[0, :]
            comm.Sendrecv(sendbuf=buf_send_l, dest=rank - 1,
                          recvbuf=buf_recv_l, source=rank - 1)
        else:
            buf_send_r[:] = DOS[ne - 1, :]
            buf_send_l[:] = DOS[0, :]
            comm.Sendrecv(sendbuf=buf_send_r, dest=rank + 1,
                          recvbuf=buf_recv_r, source=rank + 1)
            comm.Sendrecv(sendbuf=buf_send_l, dest=rank - 1,
                          recvbuf=buf_recv_l, source=rank - 1)

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
    ind_zeros = np.where((F1 > 0.1) | (F2 > 0.1) |
                         ((dDOSm > 5) & (dDOSp > 5)))[0]

    for index in ind_zeros:
        GR_3D_E[index, :, :, :] = 0
        GRnn1_3D_E[index, :, :, :] = 0
        GL_3D_E[index, :, :, :] = 0
        GLnn1_3D_E[index, :, :, :] = 0
        GG_3D_E[index, :, :, :] = 0
        GGnn1_3D_E[index, :, :, :] = 0

    return GR_3D_E, GRnn1_3D_E, GL_3D_E, GLnn1_3D_E, GG_3D_E, GGnn1_3D_E


def generator_rgf_Hamiltonian(E, idx_k, DH, SigR):
    for i, ik in enumerate(idx_k):
        kp = tuple(DH.kp[ik])
        yield (E[i] + 1j * 1e-12) * DH.Overlap[(0, 0, 0)] - DH.k_Hamiltonian[kp] - SigR[i]


def generator_rgf_GF(E, DH):
    for i in range(E.shape[0]):
        yield (E[i] + 1j * 1e-12) * DH.Overlap['H_4'] - DH.Hamiltonian['H_4']


def generator_rgf_currentdens_Hamiltonian(E, idx_k, DH):
    for i, ik in enumerate(idx_k):
        kp = tuple(DH.kp[ik])
        yield DH.k_Hamiltonian[kp] - (E[i]) * DH.Overlap[(0, 0, 0)]


def assemble_full_G_smoothing(G, factor, G_block, Gnn1_block, Bmin, Bmax, format='sparse', type='R'):
    G_temp = factor * \
        mat_assembly_fullG(G_block, Gnn1_block, Bmin,
                           Bmax, format=format, type=type)
    G[:, :] = G_temp[:, :]
