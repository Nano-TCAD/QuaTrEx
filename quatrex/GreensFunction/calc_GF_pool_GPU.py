# Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.

import concurrent.futures
from itertools import repeat

import time

import numpy as np
import numpy.typing as npt
from scipy import sparse
import mkl

from quatrex.utils import change_format
from quatrex.utils.matrix_creation import initialize_block_G, mat_assembly_fullG, homogenize_matrix, \
                                            homogenize_matrix_Rnosym, extract_small_matrix_blocks
from quatrex.GreensFunction.fermi import fermi_function
from quatrex.GreensFunction.self_energy_preprocess import self_energy_preprocess, self_energy_preprocess_2d
from quatrex.block_tri_solvers.rgf_GF import rgf_GF, rgf_standaloneGF
from quatrex.block_tri_solvers.rgf_GF_GPU import rgf_standaloneGF_GPU
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
    sr_h2g,
    sl_h2g,
    sg_h2g,
    sr_phn,
    sl_phn,
    sg_phn,
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
    return_sigma_boundary = False,
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
    (GR_3D_E, GRnn1_3D_E, GL_3D_E, GLnn1_3D_E, GG_3D_E, GGnn1_3D_E) = initialize_block_G(ne, nb, lb)
    (GR_3D_E_btch, GRnn1_3D_E_btch, GL_3D_E_btch, GLnn1_3D_E_btch, GG_3D_E_btch, GGnn1_3D_E_btch) = initialize_block_G(ne, nb, lb)

    mkl.set_num_threads(mkl_threads)

    index_e = np.arange(ne)
    bmin = DH.Bmin.copy()
    bmax = DH.Bmax.copy()

    LBsize = bmax[0] - bmin[0] + 1
    RBsize = bmax[nb - 1] - bmin[nb - 1] + 1

    SigRBL = np.zeros((ne, LBsize, LBsize), dtype = np.complex128)
    SigRBR = np.zeros((ne, RBsize, RBsize), dtype = np.complex128)
    SigLBL = np.zeros((ne, LBsize, LBsize), dtype = np.complex128)
    SigLBR = np.zeros((ne, RBsize, RBsize), dtype = np.complex128)
    SigGBL = np.zeros((ne, LBsize, LBsize), dtype = np.complex128)
    SigGBR = np.zeros((ne, RBsize, RBsize), dtype = np.complex128)
    condl = np.zeros((ne), dtype = np.float64)
    condr = np.zeros((ne), dtype = np.float64)


    rgf_M_0 = generator_rgf_GF(energy, DH)
    index_e = np.arange(ne)
    bmin = DH.Bmin.copy()
    bmax = DH.Bmax.copy()

    with concurrent.futures.ThreadPoolExecutor(max_workers=worker_num) as executor:
        executor.map(self_energy_preprocess, SigL, SigG, SigR, SigL_ephn, SigG_ephn, SigR_ephn,
                     repeat(NCpSC), repeat(bmin), repeat(bmax), repeat(homogenize))
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

    rgf_M = generator_rgf_Hamiltonian(energy, DH, SigR)
    rgf_H = generator_rgf_currentdens_Hamiltonian(energy, DH)

    l_defect = np.count_nonzero(np.isnan(condl))
    r_defect = np.count_nonzero(np.isnan(condr))

    if l_defect > 0 or r_defect > 0:
        print("Warning: %d left and %d right boundary conditions are not satisfied." % (l_defect, r_defect))

    sl_rgf, sg_rgf, sr_rgf = self_energy_preprocess_2d(sl_h2g, sg_h2g, sr_h2g, sl_phn, sg_phn, sr_phn, rows, columns, ij2ji,  NCpSC, bmin, bmax, homogenize)

    blocked_hamiltonian_diag = np.zeros((nb, ne, lb, lb), dtype=np.complex128)
    blocked_hamiltonian_upper = np.zeros((nb-1, ne, lb, lb), dtype=np.complex128)
    blocked_hamiltonian_lower = np.zeros((nb-1, ne, lb, lb), dtype=np.complex128)
    change_format.sparse2block_energyhamgen_no_map(DH.Hamiltonian['H_4'], DH.Overlap['H_4'], blocked_hamiltonian_diag, blocked_hamiltonian_upper, blocked_hamiltonian_lower, bmax-1, bmin -1, energy)
    sr_blco_diag, sr_blco_upper, sr_blco_lower = change_format.sparse2block_energy_forbatchedblockwise(map_diag, map_upper, map_lower, sr_rgf, bmax-1, bmin -1)
    sl_blco_diag, sl_blco_upper, sl_blco_lower  = change_format.sparse2block_energy_forbatchedblockwise(map_diag, map_upper, map_lower, sl_rgf, bmax-1, bmin -1)
    sg_blco_diag, sg_blco_upper, sg_blco_lower  = change_format.sparse2block_energy_forbatchedblockwise(map_diag, map_upper, map_lower, sg_rgf, bmax-1, bmin -1)

    sr_blco_diag[0,:,:,:] += SigRBL
    sr_blco_diag[-1,:,:,:] += SigRBR
    sl_blco_diag[0,:,:,:] += SigLBL
    sl_blco_diag[-1,:,:,:] += SigLBR
    sg_blco_diag[0,:,:,:] += SigGBL
    sg_blco_diag[-1,:,:,:] += SigGBR

    blocked_hamiltonian_diag -= sr_blco_diag
    blocked_hamiltonian_upper -= sr_blco_upper
    blocked_hamiltonian_lower -= sr_blco_lower

    print("Start solving the Green's function.")

    energy_batchsize = 2
    energy_batch = np.arange(0, ne, energy_batchsize)
    # for ie in energy_batch:
    #     rgf_standaloneGF_GPU(blocked_hamiltonian_diag[:, ie:ie+energy_batchsize, :, :],
    #                          blocked_hamiltonian_upper[:, ie:ie+energy_batchsize, :, :],
    #                          blocked_hamiltonian_lower[:, ie:ie+energy_batchsize, :, :],
    #                          sg_blco_diag[:, ie:ie+energy_batchsize, :, :],
    #                          sg_blco_upper[:, ie:ie+energy_batchsize, :, :],
    #                          sg_blco_lower[:, ie:ie+energy_batchsize, :, :],
    #                          sl_blco_diag[:, ie:ie+energy_batchsize, :, :],
    #                          sl_blco_upper[:, ie:ie+energy_batchsize, :, :],
    #                          sl_blco_lower[:, ie:ie+energy_batchsize, :, :],
    #                          GR_3D_E_btch[:,ie:ie+energy_batchsize, :, :],
    #                          GRnn1_3D_E_btch[:, ie:ie+energy_batchsize, :, :],
    #                          GL_3D_E_btch[:, ie:ie+energy_batchsize,  :, :],
    #                          GLnn1_3D_E_btch[:, ie:ie+energy_batchsize,  :, :],
    #                          GG_3D_E_btch[:, ie:ie+energy_batchsize,  :, :],
    #                          GGnn1_3D_E_btch[:, ie:ie+energy_batchsize,  :, :],
    #                          DOS[:, ie + energy_batchsize], nE[:, ie+energy_batchsize],
    #                           nP[:, ie + energy_batchsize], idE[:, ie + energy_batchsize], bmin, bmax, factor)

        
    with concurrent.futures.ThreadPoolExecutor(max_workers=worker_num) as executor:
        executor.map(rgf_standaloneGF, rgf_M, rgf_H, SigL, SigG,\
                     SigRBL, SigRBR, SigLBL, SigLBR, SigGBL, SigGBR,\
                     condl, condr,
                     GR_3D_E, GRnn1_3D_E, GL_3D_E, GLnn1_3D_E, GG_3D_E, GGnn1_3D_E,
                     DOS, nE, nP, idE, repeat(bmin), repeat(bmax), factor, index_e, repeat(NCpSC), repeat(block_inv),
                     repeat(use_dace), repeat(validate_dace))

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
        GR_3D_E[index, :, :, :] = 0
        GRnn1_3D_E[index, :, :, :] = 0
        GL_3D_E[index, :, :, :] = 0
        GLnn1_3D_E[index, :, :, :] = 0
        GG_3D_E[index, :, :, :] = 0
        GGnn1_3D_E[index, :, :, :] = 0
        
    if(return_sigma_boundary):
        return GR_3D_E, GRnn1_3D_E, GL_3D_E, GLnn1_3D_E, GG_3D_E, GGnn1_3D_E, SigRBL, SigRBR
    else:   
        return GR_3D_E, GRnn1_3D_E, GL_3D_E, GLnn1_3D_E, GG_3D_E, GGnn1_3D_E
    

def generator_rgf_Hamiltonian(E, DH, SigR):
    for i in range(E.shape[0]):
        yield (E[i] + 1j * 1e-12) * DH.Overlap['H_4'] - DH.Hamiltonian['H_4'] - SigR[i]

def generator_rgf_GF(E, DH):
    for i in range(E.shape[0]):
        yield (E[i] + 1j * 1e-12) * DH.Overlap['H_4'] - DH.Hamiltonian['H_4']


def generator_rgf_currentdens_Hamiltonian(E, DH):
    for i in range(E.shape[0]):
        yield DH.Hamiltonian['H_4'] - (E[i]) * DH.Overlap['H_4']


def assemble_full_G_smoothing(G, factor, G_block, Gnn1_block, Bmin, Bmax, format='sparse', type='R'):
    G_temp = factor * mat_assembly_fullG(G_block, Gnn1_block, Bmin, Bmax, format=format, type=type)
    G[:, :] = G_temp[:, :]
