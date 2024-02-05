# Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.
""" Functions to calculate the screened interaction on the cpu. See README.md for more information. """

import concurrent.futures
from itertools import repeat
import numpy as np
import mkl
import typing
import numpy.typing as npt
from scipy import sparse
from quatrex.utils import matrix_creation
from quatrex.utils import change_format
from quatrex.utils.matrix_creation import homogenize_matrix, homogenize_matrix_Rnosym, \
    extract_small_matrix_blocks
from quatrex.block_tri_solvers import rgf_W
# from quatrex.block_tri_solvers import matrix_inversion_w
from quatrex.OBC import obc_w_cpu
import time


def p2w_pool_mpi_cpu_split(
    hamiltionian_obj: object,
    energy: npt.NDArray[np.float64],
    pg: npt.NDArray[np.complex128],
    pl: npt.NDArray[np.complex128],
    pr: npt.NDArray[np.complex128],
    vh: npt.NDArray[np.complex128],
    dosw: npt.NDArray[np.complex128],
    new: npt.NDArray[np.complex128],
    npw: npt.NDArray[np.complex128],
    idx_e: npt.NDArray[np.int32],
    factor: npt.NDArray[np.float64],
    comm,
    rank,
    size,
    nbc,
    homogenize: bool = False,
    NCpSC: int = 1,
    mkl_threads: int = 1,
    worker_num: int = 1,
    block_inv: bool = False,
    use_dace: bool = False,
    validate_dace: bool = False
) -> typing.Tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128], npt.NDArray[np.complex128],
                  npt.NDArray[np.complex128], npt.NDArray[np.complex128], npt.NDArray[np.complex128], int, int]:
    """Calculates the screened interaction on the cpu.
    Uses mkl threading and pool threads.

    Args:
        hamiltionian_obj (object): Class containing the hamiltonian information
        energy (npt.NDArray[np.float64]): energy points
        pg (npt.NDArray[np.complex128]): Greater polarization, vector of sparse matrices
        pl (npt.NDArray[np.complex128]): Lesser polarization, vector of sparse matrices
        pr (npt.NDArray[np.complex128]): Retarded polarization, vector of sparse matrices
        vh (npt.NDArray[np.complex128]): Vh sparse matrix
        dosw (npt.NDArray[np.complex128]): density of state
        new (npt.NDArray[np.complex128]): density of state
        npw (npt.NDArray[np.complex128]): density of state
        factor (npt.NDArray[np.float64]): Smoothing factor
        mkl_threads (int, optional): Number of mkl threads used. Defaults to 1.
        worker_num(int, optional): Number of pool workers used. Defaults to 1.

    Returns:
        typing.Tuple[npt.NDArray[np.complex128],
                    npt.NDArray[np.complex128],
                    npt.NDArray[np.complex128],
                    npt.NDArray[np.complex128],
                    npt.NDArray[np.complex128],
                    npt.NDArray[np.complex128],
                    int, int ]:
    Diagonal/Upper block tensor (#blocks, blocksize, blocksize) of greater, lesser, retarded screened interaction.
    Number of blocks and block size after matrix multiplication
    """
    comm.Barrier()
    if rank == 0:
        time_pre_OBC = -time.perf_counter()
    # number of energy points
    ne = energy.shape[0]

    # number of blocks
    nb = hamiltionian_obj.Bmin.shape[0]
    # start and end index of each block in python indexing
    bmax = hamiltionian_obj.Bmax - 1
    bmin = hamiltionian_obj.Bmin - 1

    # fix nbc to 2 for the given solution
    # todo calculate it
    # nbc = 2
    lb_vec = bmax - bmin + 1
    lb_start = lb_vec[0]
    lb_end = lb_vec[nb - 1]

    # dvh_sd = np.zeros((ne, lb_start, lb_start), dtype=np.complex128)
    # dvh_ed = np.zeros((ne, lb_end, lb_end), dtype=np.complex128)
    # dmr_sd = np.zeros((ne, lb_start, lb_start), dtype=np.complex128)
    # dmr_ed = np.zeros((ne, lb_end, lb_end), dtype=np.complex128)
    # dlg_sd = np.zeros((ne, lb_start, lb_start), dtype=np.complex128)
    # dlg_ed = np.zeros((ne, lb_end, lb_end), dtype=np.complex128)
    # dll_sd = np.zeros((ne, lb_start, lb_start), dtype=np.complex128)
    # dll_ed = np.zeros((ne, lb_end, lb_end), dtype=np.complex128)
    # condl = np.zeros((ne), dtype = np.float64)
    # condr = np.zeros((ne), dtype = np.float64)

    # block sizes after matrix multiplication
    bmax_mm = bmax[nbc - 1:nb:nbc]
    bmin_mm = bmin[0:nb:nbc]
    # number of blocks after matrix multiplication
    nb_mm = bmax_mm.size
    # larges block length after matrix multiplication
    lb_max_mm = np.max(bmax_mm - bmin_mm + 1)
    lb_vec_mm = bmax_mm - bmin_mm + 1
    lb_start_mm = lb_vec_mm[0]
    lb_end_mm = lb_vec_mm[nb_mm - 1]

    dvh_sd = np.zeros((ne, lb_start_mm, lb_start_mm), dtype=np.complex128)
    dvh_ed = np.zeros((ne, lb_end_mm, lb_end_mm), dtype=np.complex128)
    dmr_sd = np.zeros((ne, lb_start_mm, lb_start_mm), dtype=np.complex128)
    dmr_ed = np.zeros((ne, lb_end_mm, lb_end_mm), dtype=np.complex128)
    dlg_sd = np.zeros((ne, lb_start_mm, lb_start_mm), dtype=np.complex128)
    dlg_ed = np.zeros((ne, lb_end_mm, lb_end_mm), dtype=np.complex128)
    dll_sd = np.zeros((ne, lb_start_mm, lb_start_mm), dtype=np.complex128)
    dll_ed = np.zeros((ne, lb_end_mm, lb_end_mm), dtype=np.complex128)
    condl = np.zeros((ne), dtype=np.float64)
    condr = np.zeros((ne), dtype=np.float64)

    # create empty buffer for screened interaction
    # in block format
    wr_diag, wr_upper, wl_diag, wl_upper, wg_diag, wg_upper = matrix_creation.initialize_block_G(
        ne, nb_mm, lb_max_mm)
    xr_diag = np.zeros((ne, nb_mm, lb_max_mm, lb_max_mm), dtype=np.complex128)
    # set number of mkl threads
    mkl.set_num_threads(mkl_threads)

    for ie in range(ne):
        # Anti-Hermitian symmetrizing of PL and PG
        # pl[ie] = 1j * np.imag(pl[ie])
        pl[ie] = (pl[ie] - pl[ie].conj().T) / 2

        # pg[ie] = 1j * np.imag(pg[ie])
        pg[ie] = (pg[ie] - pg[ie].conj().T) / 2

        # PR has to be derived from PL and PG and then has to be symmetrized
        pr[ie] = (pg[ie] - pl[ie]) / 2
        # pr[ie] = (pr[ie] + pr[ie].T) / 2
        if homogenize:
            (PR00, PR01, PR10, _) = extract_small_matrix_blocks(pr[ie][bmin[0]:bmax[0]+1, bmin[0]:bmax[0]+1],
                                                                pr[ie][bmin[0]:bmax[0] +
                                                                       1, bmin[1]:bmax[1]+1],
                                                                pr[ie][bmin[1]:bmax[1]+1, bmin[0]:bmax[0]+1], NCpSC, 'L')
            pr[ie] = homogenize_matrix_Rnosym(PR00,
                                              PR01,
                                              PR10, len(bmax))
            (PL00, PL01, PL10, _) = extract_small_matrix_blocks(pl[ie][bmin[0]:bmax[0]+1, bmin[0]:bmax[0]+1],
                                                                pl[ie][bmin[0]:bmax[0] +
                                                                       1, bmin[1]:bmax[1]+1],
                                                                pl[ie][bmin[1]:bmax[1]+1, bmin[0]:bmax[0]+1], NCpSC, 'L')
            pl[ie] = homogenize_matrix_Rnosym(PL00,
                                              PL01,
                                              PL10,
                                              len(bmax))
            (PG00, PG01, PG10, _) = extract_small_matrix_blocks(pg[ie][bmin[0]:bmax[0]+1, bmin[0]:bmax[0]+1],
                                                                pg[ie][bmin[0]:bmax[0] +
                                                                       1, bmin[1]:bmax[1]+1],
                                                                pg[ie][bmin[1]:bmax[1]+1, bmin[0]:bmax[0]+1], NCpSC, 'L')
            pg[ie] = homogenize_matrix_Rnosym(PG00,
                                              PG01,
                                              PG10,
                                              len(bmax))

    # Create a process pool with num_worker workers
    comm.Barrier()
    if rank == 0:
        time_pre_OBC += time.perf_counter()
        print("Time for pre-processing OBC: %.3f s" % time_pre_OBC, flush=True)
        time_OBC = -time.perf_counter()

    ref_flag = False
    with concurrent.futures.ThreadPoolExecutor(max_workers=worker_num) as executor:
        results = executor.map(obc_w_cpu.obc_w_cpu, repeat(vh),
                               pg, pl, pr,
                               repeat(bmax), repeat(bmin),
                               dvh_sd, dvh_ed,
                               dmr_sd, dmr_ed,
                               dlg_sd, dlg_ed,
                               dll_sd, dll_ed,
                               repeat(nbc),
                               repeat(NCpSC),
                               repeat(block_inv),
                               repeat(use_dace),
                               repeat(validate_dace), repeat(ref_flag)
                               )
        for idx, res in enumerate(results):
            condl[idx] = res[0]
            condr[idx] = res[1]

    l_defect = np.count_nonzero(np.isnan(condl))
    r_defect = np.count_nonzero(np.isnan(condr))

    if l_defect > 0 or r_defect > 0:
        print("Warning: %d left and %d right boundary conditions are not satisfied." % (
            l_defect, r_defect))

    comm.Barrier()
    if rank == 0:
        time_OBC += time.perf_counter()
        print("Time for OBC: %.3f s" % time_OBC, flush=True)
        time_GF_trafo = -time.perf_counter()

    comm.Barrier()
    if rank == 0:
        time_GF_trafo += time.perf_counter()
        print("Time for W transformation: %.3f s" % time_GF_trafo, flush=True)
        time_GF = -time.perf_counter()

    with concurrent.futures.ThreadPoolExecutor(max_workers=worker_num) as executor:
        # Use the map function to apply the inv_matrices function to each pair of matrices in parallel
        # executor.map(
        results = executor.map(
            rgf_W.rgf_w_opt_standalone,
            repeat(vh),
            pg, pl, pr,
            repeat(bmax), repeat(bmin),
            wg_diag, wg_upper,
            wl_diag, wl_upper,
            wr_diag, wr_upper,
            xr_diag,
            dvh_sd, dvh_ed,
            dmr_sd, dmr_ed,
            dlg_sd, dlg_ed,
            dll_sd, dll_ed,
            dosw, new, npw, repeat(nbc),
            idx_e, factor,
            repeat(NCpSC),
            repeat(block_inv),
            repeat(use_dace),
            repeat(validate_dace), repeat(ref_flag))
        # for res in results:
        #    assert isinstance(res, np.ndarray)
    comm.Barrier()
    if rank == 0:
        time_GF += time.perf_counter()
        print("Time for W: %.3f s" % time_GF, flush=True)
        time_post_proc = -time.perf_counter()

    # with concurrent.futures.ThreadPoolExecutor(max_workers=worker_num) as executor:
    #     # Use the map function to apply the inv_matrices function to each pair of matrices in parallel
    #     #executor.map(
    #     results = executor.map(
    #                 rgf_W.rgf_w_opt,
    #                 repeat(vh),
    #                 pg, pl, pr,
    #                 repeat(bmax), repeat(bmin),
    #                 wg_diag, wg_upper,
    #                 wl_diag, wl_upper,
    #                 wr_diag, wr_upper,
    #                 xr_diag, dosw, new, npw, repeat(nbc),
    #                 idx_e, factor,
    #                 repeat(NCpSC),
    #                 repeat(block_inv),
    #                 repeat(use_dace),
    #                 repeat(validate_dace),repeat(ref_flag))
    #     for res in results:
    #        assert isinstance(res, np.ndarray)

    # Calculate F1, F2, which are the relative errors of GR-GA = GG-GL
    F1 = np.max(np.abs(dosw - (new + npw)) / (np.abs(dosw) + 1e-6), axis=1)
    F2 = np.max(np.abs(dosw - (new + npw)) /
                (np.abs(new + npw) + 1e-6), axis=1)

    buf_recv_r = np.empty((dosw.shape[1]), dtype=np.complex128)
    buf_send_r = np.empty((dosw.shape[1]), dtype=np.complex128)
    buf_recv_l = np.empty((dosw.shape[1]), dtype=np.complex128)
    buf_send_l = np.empty((dosw.shape[1]), dtype=np.complex128)
    if size > 1:
        if rank == 0:
            buf_send_r[:] = dosw[ne - 1, :]
            comm.Sendrecv(sendbuf=buf_send_r, dest=rank + 1,
                          recvbuf=buf_recv_r, source=rank + 1)

        elif rank == size - 1:
            buf_send_l[:] = dosw[0, :]
            comm.Sendrecv(sendbuf=buf_send_l, dest=rank - 1,
                          recvbuf=buf_recv_l, source=rank - 1)
        else:
            buf_send_r[:] = dosw[ne - 1, :]
            buf_send_l[:] = dosw[0, :]
            comm.Sendrecv(sendbuf=buf_send_r, dest=rank + 1,
                          recvbuf=buf_recv_r, source=rank + 1)
            comm.Sendrecv(sendbuf=buf_send_l, dest=rank - 1,
                          recvbuf=buf_recv_l, source=rank - 1)

    # Remove individual peaks (To-Do: improve this part by sending boundary elements to the next process)
    if size == 1:
        dDOSm = np.concatenate(([0], np.max(np.abs(dosw[1:ne - 1, :] / (dosw[0:ne - 2, :] + 1)),
                                            axis=1), [np.max(np.abs(dosw[ne - 1, :] / (dosw[ne - 2, :] + 1)))]))
        dDOSp = np.concatenate(([np.max(np.abs(dosw[0, :] / (dosw[1, :] + 1)))],
                                np.max(np.abs(dosw[1:ne - 1, :] / (dosw[2:ne, :] + 1)), axis=1), [0]))
    elif rank == 0:
        dDOSm = np.concatenate(([0], np.max(np.abs(dosw[1:ne - 1, :] / (dosw[0:ne - 2, :] + 1)),
                                            axis=1), [np.max(np.abs(dosw[ne - 1, :] / (dosw[ne - 2, :] + 1)))]))
        dDOSp = np.concatenate(([np.max(np.abs(dosw[0, :] / (dosw[1, :] + 1)))],
                                np.max(np.abs(dosw[1:ne - 1, :] / (dosw[2:ne, :] + 1)),
                                       axis=1), [np.max(np.abs(dosw[ne - 1, :] / (buf_recv_r + 1)))]))
    elif rank == size - 1:
        dDOSm = np.concatenate(([np.max(np.abs(dosw[0, :] / (buf_recv_l + 1)))],
                                np.max(np.abs(dosw[1:ne - 1, :] / (dosw[0:ne - 2, :] + 1)),
                                       axis=1), [np.max(np.abs(dosw[ne - 1, :] / (dosw[ne - 2, :] + 1)))]))
        dDOSp = np.concatenate(([np.max(np.abs(dosw[0, :] / (dosw[1, :] + 1)))],
                                np.max(np.abs(dosw[1:ne - 1, :] / (dosw[2:ne, :] + 1)), axis=1), [0]))
    else:
        dDOSm = np.concatenate(([np.max(np.abs(dosw[0, :] / (buf_recv_l + 1)))],
                                np.max(np.abs(dosw[1:ne - 1, :] / (dosw[0:ne - 2, :] + 1)),
                                       axis=1), [np.max(np.abs(dosw[ne - 1, :] / (dosw[ne - 2, :] + 1)))]))
        dDOSp = np.concatenate(([np.max(np.abs(dosw[0, :] / (dosw[1, :] + 1)))],
                                np.max(np.abs(dosw[1:ne - 1, :] / (dosw[2:ne, :] + 1)),
                                       axis=1), [np.max(np.abs(dosw[ne - 1, :] / (buf_recv_r + 1)))]))

    # Find indices of elements satisfying the conditions
    ind_zeros = np.where((F1 > 0.1) | (F2 > 0.1) |
                         ((dDOSm > 5) & (dDOSp > 5)))[0]

    if idx_e[0] == 0:
        ind_zeros = np.concatenate(([0], ind_zeros))

    # Remove the identified peaks and errors
    for index in ind_zeros:
        wr_diag[index, :, :, :] = 0
        wr_upper[index, :, :, :] = 0
        wl_diag[index, :, :, :] = 0
        wl_upper[index, :, :, :] = 0
        wg_diag[index, :, :, :] = 0
        wg_upper[index, :, :, :] = 0

    comm.Barrier()
    if rank == 0:
        time_post_proc += time.perf_counter()
        print("Time for post-processing: %.3f s" % time_post_proc, flush=True)

    return wg_diag, wg_upper, wl_diag, wl_upper, wr_diag, wr_upper, nb_mm, lb_max_mm, ind_zeros


def p2w_pool_mpi_cpu(
    hamiltionian_obj: object,
    energy: npt.NDArray[np.float64],
    pg: npt.NDArray[np.complex128],
    pl: npt.NDArray[np.complex128],
    pr: npt.NDArray[np.complex128],
    vh: npt.NDArray[np.complex128],
    dosw: npt.NDArray[np.complex128],
    new: npt.NDArray[np.complex128],
    npw: npt.NDArray[np.complex128],
    idx_e: npt.NDArray[np.int32],
    factor: npt.NDArray[np.float64],
    comm,
    rank,
    size,
    nbc,
    homogenize: bool = False,
    NCpSC: int = 1,
    mkl_threads: int = 1,
    worker_num: int = 1,
    block_inv: bool = False,
    use_dace: bool = False,
    validate_dace: bool = False
) -> typing.Tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128], npt.NDArray[np.complex128],
                  npt.NDArray[np.complex128], npt.NDArray[np.complex128], npt.NDArray[np.complex128], int, int]:
    """Calculates the screened interaction on the cpu.
    Uses mkl threading and pool threads.

    Args:
        hamiltionian_obj (object): Class containing the hamiltonian information
        energy (npt.NDArray[np.float64]): energy points
        pg (npt.NDArray[np.complex128]): Greater polarization, vector of sparse matrices
        pl (npt.NDArray[np.complex128]): Lesser polarization, vector of sparse matrices
        pr (npt.NDArray[np.complex128]): Retarded polarization, vector of sparse matrices
        vh (npt.NDArray[np.complex128]): Vh sparse matrix
        dosw (npt.NDArray[np.complex128]): density of state
        new (npt.NDArray[np.complex128]): density of state
        npw (npt.NDArray[np.complex128]): density of state
        factor (npt.NDArray[np.float64]): Smoothing factor
        mkl_threads (int, optional): Number of mkl threads used. Defaults to 1.
        worker_num(int, optional): Number of pool workers used. Defaults to 1.

    Returns:
        typing.Tuple[npt.NDArray[np.complex128],
                    npt.NDArray[np.complex128],
                    npt.NDArray[np.complex128],
                    npt.NDArray[np.complex128],
                    npt.NDArray[np.complex128],
                    npt.NDArray[np.complex128],
                    int, int ]:
    Diagonal/Upper block tensor (#blocks, blocksize, blocksize) of greater, lesser, retarded screened interaction.
    Number of blocks and block size after matrix multiplication
    """
    # number of energy points
    ne = energy.shape[0]

    # number of blocks
    nb = hamiltionian_obj.Bmin.shape[0]
    # start and end index of each block in python indexing
    bmax = hamiltionian_obj.Bmax - 1
    bmin = hamiltionian_obj.Bmin - 1

    # fix nbc to 2 for the given solution
    # todo calculate it
    # nbc = 2

    # block sizes after matrix multiplication
    bmax_mm = bmax[nbc - 1:nb:nbc]
    bmin_mm = bmin[0:nb:nbc]
    # number of blocks after matrix multiplication
    nb_mm = bmax_mm.size
    # larges block length after matrix multiplication
    lb_max_mm = np.max(bmax_mm - bmin_mm + 1)

    # create empty buffer for screened interaction
    # in block format
    wr_diag, wr_upper, wl_diag, wl_upper, wg_diag, wg_upper = matrix_creation.initialize_block_G(
        ne, nb_mm, lb_max_mm)
    xr_diag = np.zeros((ne, nb_mm, lb_max_mm, lb_max_mm), dtype=np.complex128)
    # set number of mkl threads
    mkl.set_num_threads(mkl_threads)

    for ie in range(ne):
        # Anti-Hermitian symmetrizing of PL and PG
        # pl[ie] = 1j * np.imag(pl[ie])
        pl[ie] = (pl[ie] - pl[ie].conj().T) / 2

        # pg[ie] = 1j * np.imag(pg[ie])
        pg[ie] = (pg[ie] - pg[ie].conj().T) / 2

        # PR has to be derived from PL and PG and then has to be symmetrized
        pr[ie] = (pg[ie] - pl[ie]) / 2
        # pr[ie] = (pr[ie] + pr[ie].T) / 2
        if homogenize:
            (PR00, PR01, PR10, _) = extract_small_matrix_blocks(pr[ie][bmin[0]:bmax[0]+1, bmin[0]:bmax[0]+1],
                                                                pr[ie][bmin[0]:bmax[0] +
                                                                       1, bmin[1]:bmax[1]+1],
                                                                pr[ie][bmin[1]:bmax[1]+1, bmin[0]:bmax[0]+1], NCpSC, 'L')
            pr[ie] = homogenize_matrix_Rnosym(PR00,
                                              PR01,
                                              PR10, len(bmax))
            (PL00, PL01, PL10, _) = extract_small_matrix_blocks(pl[ie][bmin[0]:bmax[0]+1, bmin[0]:bmax[0]+1],
                                                                pl[ie][bmin[0]:bmax[0] +
                                                                       1, bmin[1]:bmax[1]+1],
                                                                pl[ie][bmin[1]:bmax[1]+1, bmin[0]:bmax[0]+1], NCpSC, 'L')
            pl[ie] = homogenize_matrix_Rnosym(PL00,
                                              PL01,
                                              PL10,
                                              len(bmax))
            (PG00, PG01, PG10, _) = extract_small_matrix_blocks(pg[ie][bmin[0]:bmax[0]+1, bmin[0]:bmax[0]+1],
                                                                pg[ie][bmin[0]:bmax[0] +
                                                                       1, bmin[1]:bmax[1]+1],
                                                                pg[ie][bmin[1]:bmax[1]+1, bmin[0]:bmax[0]+1], NCpSC, 'L')
            pg[ie] = homogenize_matrix_Rnosym(PG00,
                                              PG01,
                                              PG10,
                                              len(bmax))

    # Create a process pool with num_worker workers

    ref_flag = False
    with concurrent.futures.ThreadPoolExecutor(max_workers=worker_num) as executor:
        # Use the map function to apply the inv_matrices function to each pair of matrices in parallel
        # executor.map(
        results = executor.map(
            rgf_W.rgf_w_opt,
            repeat(vh),
            pg, pl, pr,
            repeat(bmax), repeat(bmin),
            wg_diag, wg_upper,
            wl_diag, wl_upper,
            wr_diag, wr_upper,
            xr_diag, dosw, new, npw, repeat(nbc),
            idx_e, factor,
            repeat(NCpSC),
            repeat(block_inv),
            repeat(use_dace),
            repeat(validate_dace), repeat(ref_flag))
        for res in results:
            assert isinstance(res, np.ndarray)

    # Calculate F1, F2, which are the relative errors of GR-GA = GG-GL
    F1 = np.max(np.abs(dosw - (new + npw)) / (np.abs(dosw) + 1e-6), axis=1)
    F2 = np.max(np.abs(dosw - (new + npw)) /
                (np.abs(new + npw) + 1e-6), axis=1)

    buf_recv_r = np.empty((dosw.shape[1]), dtype=np.complex128)
    buf_send_r = np.empty((dosw.shape[1]), dtype=np.complex128)
    buf_recv_l = np.empty((dosw.shape[1]), dtype=np.complex128)
    buf_send_l = np.empty((dosw.shape[1]), dtype=np.complex128)
    if size > 1:
        if rank == 0:
            buf_send_r[:] = dosw[ne - 1, :]
            comm.Sendrecv(sendbuf=buf_send_r, dest=rank + 1,
                          recvbuf=buf_recv_r, source=rank + 1)

        elif rank == size - 1:
            buf_send_l[:] = dosw[0, :]
            comm.Sendrecv(sendbuf=buf_send_l, dest=rank - 1,
                          recvbuf=buf_recv_l, source=rank - 1)
        else:
            buf_send_r[:] = dosw[ne - 1, :]
            buf_send_l[:] = dosw[0, :]
            comm.Sendrecv(sendbuf=buf_send_r, dest=rank + 1,
                          recvbuf=buf_recv_r, source=rank + 1)
            comm.Sendrecv(sendbuf=buf_send_l, dest=rank - 1,
                          recvbuf=buf_recv_l, source=rank - 1)

    # Remove individual peaks (To-Do: improve this part by sending boundary elements to the next process)
    if size == 1:
        dDOSm = np.concatenate(([0], np.max(np.abs(dosw[1:ne - 1, :] / (dosw[0:ne - 2, :] + 1)),
                                            axis=1), [np.max(np.abs(dosw[ne - 1, :] / (dosw[ne - 2, :] + 1)))]))
        dDOSp = np.concatenate(([np.max(np.abs(dosw[0, :] / (dosw[1, :] + 1)))],
                                np.max(np.abs(dosw[1:ne - 1, :] / (dosw[2:ne, :] + 1)), axis=1), [0]))
    elif rank == 0:
        dDOSm = np.concatenate(([0], np.max(np.abs(dosw[1:ne - 1, :] / (dosw[0:ne - 2, :] + 1)),
                                            axis=1), [np.max(np.abs(dosw[ne - 1, :] / (dosw[ne - 2, :] + 1)))]))
        dDOSp = np.concatenate(([np.max(np.abs(dosw[0, :] / (dosw[1, :] + 1)))],
                                np.max(np.abs(dosw[1:ne - 1, :] / (dosw[2:ne, :] + 1)),
                                       axis=1), [np.max(np.abs(dosw[ne - 1, :] / (buf_recv_r + 1)))]))
    elif rank == size - 1:
        dDOSm = np.concatenate(([np.max(np.abs(dosw[0, :] / (buf_recv_l + 1)))],
                                np.max(np.abs(dosw[1:ne - 1, :] / (dosw[0:ne - 2, :] + 1)),
                                       axis=1), [np.max(np.abs(dosw[ne - 1, :] / (dosw[ne - 2, :] + 1)))]))
        dDOSp = np.concatenate(([np.max(np.abs(dosw[0, :] / (dosw[1, :] + 1)))],
                                np.max(np.abs(dosw[1:ne - 1, :] / (dosw[2:ne, :] + 1)), axis=1), [0]))
    else:
        dDOSm = np.concatenate(([np.max(np.abs(dosw[0, :] / (buf_recv_l + 1)))],
                                np.max(np.abs(dosw[1:ne - 1, :] / (dosw[0:ne - 2, :] + 1)),
                                       axis=1), [np.max(np.abs(dosw[ne - 1, :] / (dosw[ne - 2, :] + 1)))]))
        dDOSp = np.concatenate(([np.max(np.abs(dosw[0, :] / (dosw[1, :] + 1)))],
                                np.max(np.abs(dosw[1:ne - 1, :] / (dosw[2:ne, :] + 1)),
                                       axis=1), [np.max(np.abs(dosw[ne - 1, :] / (buf_recv_r + 1)))]))

    # Find indices of elements satisfying the conditions
    ind_zeros = np.where((F1 > 0.1) | (F2 > 0.1) |
                         ((dDOSm > 5) & (dDOSp > 5)))[0]

    # Remove the identified peaks and errors
    for index in ind_zeros:
        wr_diag[index, :, :, :] = 0
        wr_upper[index, :, :, :] = 0
        wl_diag[index, :, :, :] = 0
        wl_upper[index, :, :, :] = 0
        wg_diag[index, :, :, :] = 0
        wg_upper[index, :, :, :] = 0

    return wg_diag, wg_upper, wl_diag, wl_upper, wr_diag, wr_upper, nb_mm, lb_max_mm, ind_zeros


def p2w_pool_mpi_cpu_kpoint(
    coulomb_obj: object,
    energy: npt.NDArray[np.float64],
    pg: npt.NDArray[np.complex128],
    pl: npt.NDArray[np.complex128],
    pr: npt.NDArray[np.complex128],
    dosw: npt.NDArray[np.complex128],
    new: npt.NDArray[np.complex128],
    npw: npt.NDArray[np.complex128],
    idx_k: npt.NDArray[np.int32],
    idx_e: npt.NDArray[np.int32],
    factor: npt.NDArray[np.float64],
    comm,
    rank,
    size,
    nbc,
    homogenize: bool = False,
    mkl_threads: int = 1,
    worker_num: int = 1,
    block_inv: bool = False,
    use_dace: bool = False,
    validate_dace: bool = False
) -> typing.Tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128], npt.NDArray[np.complex128],
                  npt.NDArray[np.complex128], npt.NDArray[np.complex128], npt.NDArray[np.complex128], int, int]:
    """Calculates the screened interaction on the cpu.
    Uses mkl threading and pool threads.

    k-points included for this function.

    Args:
        energy (npt.NDArray[np.float64]): energy points
        pg (npt.NDArray[np.complex128]): Greater polarization, vector of sparse matrices
        pl (npt.NDArray[np.complex128]): Lesser polarization, vector of sparse matrices
        pr (npt.NDArray[np.complex128]): Retarded polarization, vector of sparse matrices
        coulomb_obj (npt.NDArray[np.complex128]): Class containing the Coulomb integral information
        dosw (npt.NDArray[np.complex128]): density of state
        new (npt.NDArray[np.complex128]): density of state
        npw (npt.NDArray[np.complex128]): density of state
        factor (npt.NDArray[np.float64]): Smoothing factor
        mkl_threads (int, optional): Number of mkl threads used. Defaults to 1.
        worker_num(int, optional): Number of pool workers used. Defaults to 1.

    Returns:
        typing.Tuple[npt.NDArray[np.complex128],
                    npt.NDArray[np.complex128],
                    npt.NDArray[np.complex128],
                    npt.NDArray[np.complex128],
                    npt.NDArray[np.complex128],
                    npt.NDArray[np.complex128],
                    int, int ]:
    Diagonal/Upper block tensor (#blocks, blocksize, blocksize) of greater, lesser, retarded screened interaction.
    Number of blocks and block size after matrix multiplication
    """
    # number of energy points
    ne = energy.shape[0]

    # number of blocks
    nb = coulomb_obj.NBlocks
    # start and end index of each block in python indexing
    bmax = coulomb_obj.Bmax
    bmin = coulomb_obj.Bmin

    # fix nbc to 2 for the given solution
    # todo calculate it
    #nbc = 2

    # block sizes after matrix multiplication
    bmax_mm = bmax[nbc - 1:nb:nbc]
    bmin_mm = bmin[0:nb:nbc]
    # number of blocks after matrix multiplication
    nb_mm = bmax_mm.size
    # larges block length after matrix multiplication
    lb_max_mm = np.max(bmax_mm - bmin_mm + 1)

    # create empty buffer for screened interaction
    # in block format
    wr_diag, wr_upper, wl_diag, wl_upper, wg_diag, wg_upper = matrix_creation.initialize_block_G(ne, nb_mm, lb_max_mm)
    xr_diag = np.zeros((ne, nb_mm, lb_max_mm, lb_max_mm), dtype=np.complex128)
    # set number of mkl threads
    mkl.set_num_threads(mkl_threads)

    for ie in range(ne):
        # Anti-Hermitian symmetrizing of PL and PG
        #pl[ie] = 1j * np.imag(pl[ie])
        pl[ie] = (pl[ie] - pl[ie].conj().T) / 2

        #pg[ie] = 1j * np.imag(pg[ie])
        pg[ie] = (pg[ie] - pg[ie].conj().T) / 2

        # PR has to be derived from PL and PG and then has to be symmetrized
        pr[ie] = (pg[ie] - pl[ie]) / 2
        #pr[ie] = (pr[ie] + pr[ie].T) / 2
        if homogenize:
            (PR00, PR01, PR10, _) = extract_small_matrix_blocks(pr[ie][bmin[0]:bmax[0]+1, bmin[0]:bmax[0]+1],
                                                                pr[ie][bmin[0]:bmax[0] +
                                                                       1, bmin[1]:bmax[1]+1],
                                                                pr[ie][bmin[1]:bmax[1]+1, bmin[0]:bmax[0]+1], NCpSC, 'L')
            pr[ie] = homogenize_matrix_Rnosym(PR00,
                                              PR01,
                                              PR10, len(bmax))
            (PL00, PL01, PL10, _) = extract_small_matrix_blocks(pl[ie][bmin[0]:bmax[0]+1, bmin[0]:bmax[0]+1],
                                                                pl[ie][bmin[0]:bmax[0] +
                                                                       1, bmin[1]:bmax[1]+1],
                                                                pl[ie][bmin[1]:bmax[1]+1, bmin[0]:bmax[0]+1], NCpSC, 'L')
            pl[ie] = homogenize_matrix_Rnosym(PL00,
                                              PL01,
                                              PL10,
                                              len(bmax))
            (PG00, PG01, PG10, _) = extract_small_matrix_blocks(pg[ie][bmin[0]:bmax[0]+1, bmin[0]:bmax[0]+1],
                                                                pg[ie][bmin[0]:bmax[0] +
                                                                       1, bmin[1]:bmax[1]+1],
                                                                pg[ie][bmin[1]:bmax[1]+1, bmin[0]:bmax[0]+1], NCpSC, 'L')
            pg[ie] = homogenize_matrix_Rnosym(PG00,
                                              PG01,
                                              PG10,
                                              len(bmax))

    # Here I need a generator as for the calculation of the retarded Green's function
    rgf_Coul = generator_rgf_Coulomb(idx_k, coulomb_obj)
    # Create a process pool with num_worker workers
    ref_flag = False
    with concurrent.futures.ThreadPoolExecutor(max_workers=worker_num) as executor:
        # Use the map function to apply the inv_matrices function to each pair of matrices in parallel
        #executor.map(
        results = executor.map(
                    rgf_W.rgf_w_opt,
                    rgf_Coul,
                    pg, pl, pr,
                    repeat(bmax), repeat(bmin),
                    wg_diag, wg_upper,
                    wl_diag, wl_upper,
                    wr_diag, wr_upper,
                    xr_diag, dosw, new, npw, repeat(nbc),
                    idx_e, factor,
                    repeat(block_inv),
                    repeat(use_dace),
                    repeat(validate_dace),repeat(ref_flag))
        for res in results:
           assert isinstance(res, np.ndarray)

    # Calculate F1, F2, which are the relative errors of GR-GA = GG-GL
    F1 = np.max(np.abs(dosw - (new + npw)) / (np.abs(dosw) + 1e-6), axis=1)
    F2 = np.max(np.abs(dosw - (new + npw)) / (np.abs(new + npw) + 1e-6), axis=1)

    buf_recv_r = np.empty((dosw.shape[1]), dtype=np.complex128)
    buf_send_r = np.empty((dosw.shape[1]), dtype=np.complex128)
    buf_recv_l = np.empty((dosw.shape[1]), dtype=np.complex128)
    buf_send_l = np.empty((dosw.shape[1]), dtype=np.complex128)
    if size > 1:
        if rank == 0:
            buf_send_r[:] = dosw[ne - 1, :]
            comm.Sendrecv(sendbuf=buf_send_r, dest=rank + 1, recvbuf=buf_recv_r, source=rank + 1)

        elif rank == size - 1:
            buf_send_l[:] = dosw[0, :]
            comm.Sendrecv(sendbuf=buf_send_l, dest=rank - 1, recvbuf=buf_recv_l, source=rank - 1)
        else:
            buf_send_r[:] = dosw[ne - 1, :]
            buf_send_l[:] = dosw[0, :]
            comm.Sendrecv(sendbuf=buf_send_r, dest=rank + 1, recvbuf=buf_recv_r, source=rank + 1)
            comm.Sendrecv(sendbuf=buf_send_l, dest=rank - 1, recvbuf=buf_recv_l, source=rank - 1)

    # Remove individual peaks (To-Do: improve this part by sending boundary elements to the next process)
    if size == 1:
        dDOSm = np.concatenate(([0], np.max(np.abs(dosw[1:ne - 1, :] / (dosw[0:ne - 2, :] + 1)),
                                            axis=1), [np.max(np.abs(dosw[ne - 1, :] / (dosw[ne - 2, :] + 1)))]))
        dDOSp = np.concatenate(([np.max(np.abs(dosw[0, :] / (dosw[1, :] + 1)))],
                                np.max(np.abs(dosw[1:ne - 1, :] / (dosw[2:ne, :] + 1)), axis=1), [0]))
    elif rank == 0:
        dDOSm = np.concatenate(([0], np.max(np.abs(dosw[1:ne - 1, :] / (dosw[0:ne - 2, :] + 1)),
                                            axis=1), [np.max(np.abs(dosw[ne - 1, :] / (dosw[ne - 2, :] + 1)))]))
        dDOSp = np.concatenate(([np.max(np.abs(dosw[0, :] / (dosw[1, :] + 1)))],
                                np.max(np.abs(dosw[1:ne - 1, :] / (dosw[2:ne, :] + 1)),
                                       axis=1), [np.max(np.abs(dosw[ne - 1, :] / (buf_recv_r + 1)))]))
    elif rank == size - 1:
        dDOSm = np.concatenate(([np.max(np.abs(dosw[0, :] / (buf_recv_l + 1)))],
                                np.max(np.abs(dosw[1:ne - 1, :] / (dosw[0:ne - 2, :] + 1)),
                                       axis=1), [np.max(np.abs(dosw[ne - 1, :] / (dosw[ne - 2, :] + 1)))]))
        dDOSp = np.concatenate(([np.max(np.abs(dosw[0, :] / (dosw[1, :] + 1)))],
                                np.max(np.abs(dosw[1:ne - 1, :] / (dosw[2:ne, :] + 1)), axis=1), [0]))
    else:
        dDOSm = np.concatenate(([np.max(np.abs(dosw[0, :] / (buf_recv_l + 1)))],
                                np.max(np.abs(dosw[1:ne - 1, :] / (dosw[0:ne - 2, :] + 1)),
                                       axis=1), [np.max(np.abs(dosw[ne - 1, :] / (dosw[ne - 2, :] + 1)))]))
        dDOSp = np.concatenate(([np.max(np.abs(dosw[0, :] / (dosw[1, :] + 1)))],
                                np.max(np.abs(dosw[1:ne - 1, :] / (dosw[2:ne, :] + 1)),
                                       axis=1), [np.max(np.abs(dosw[ne - 1, :] / (buf_recv_r + 1)))]))

    # Find indices of elements satisfying the conditions
    ind_zeros = np.where((F1 > 0.1) | (F2 > 0.1) | ((dDOSm > 5) & (dDOSp > 5)))[0]

    # Remove the identified peaks and errors
    for index in ind_zeros:
        wr_diag[index, :, :, :] = 0
        wr_upper[index, :, :, :] = 0
        wl_diag[index, :, :, :] = 0
        wl_upper[index, :, :, :] = 0
        wg_diag[index, :, :, :] = 0
        wg_upper[index, :, :, :] = 0

    return wg_diag, wg_upper, wl_diag, wl_upper, wr_diag, wr_upper, nb_mm, lb_max_mm, ind_zeros


def generator_rgf_Coulomb(idx_k, DH):
    for ik in idx_k:
        kp = tuple(DH.kp[ik])
        yield DH.k_Coulomb_matrix[kp]


def p2w_mpi_cpu(
    hamiltionian_obj: object,
    energy: npt.NDArray[np.float64],
    pg: npt.NDArray[np.complex128],
    pl: npt.NDArray[np.complex128],
    pr: npt.NDArray[np.complex128],
    vh: npt.NDArray[np.complex128],
    dosw: npt.NDArray[np.complex128],
    new: npt.NDArray[np.complex128],
    npw: npt.NDArray[np.complex128],
    factor: npt.NDArray[np.float64],
    comm,
    rank,
    size,
    nbc,
    mkl_threads: int = 1,
    block_inv: bool = False,
    use_dace: bool = False,
    validate_dace: bool = False
) -> typing.Tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128], npt.NDArray[np.complex128],
                  npt.NDArray[np.complex128], npt.NDArray[np.complex128], npt.NDArray[np.complex128], int, int]:
    """Calculates the screened interaction on the cpu.
    Uses only mkl threading.

    Args:
        hamiltionian_obj (object): Class containing the hamiltonian information
        energy (npt.NDArray[np.float64]): energy points
        pg (npt.NDArray[np.complex128]): Greater polarization, vector of sparse matrices
        pl (npt.NDArray[np.complex128]): Lesser polarization, vector of sparse matrices
        pr (npt.NDArray[np.complex128]): Retarded polarization, vector of sparse matrices
        vh (npt.NDArray[np.complex128]): Vh sparse matrix
        dosw (npt.NDArray[np.complex128]): density of state
        new (npt.NDArray[np.complex128]): todo
        npw (npt.NDArray[np.complex128]): todo
        factor (npt.NDArray[np.float64]): Smoothing factor
        mkl_threads (int, optional): Number of mkl threads used. Defaults to 1.

    Returns:
        typing.Tuple[npt.NDArray[np.complex128],
                    npt.NDArray[np.complex128],
                    npt.NDArray[np.complex128],
                    npt.NDArray[np.complex128],
                    npt.NDArray[np.complex128],
                    npt.NDArray[np.complex128],
                    int, int ]:
    Diagonal/Upper block tensor (#blocks, blocksize, blocksize) of greater, lesser, retarded screened interaction.
    Number of blocks and block size after matrix multiplication
    """

    # create timing vector
    times = np.zeros((10), dtype=np.float64)

    # number of energy points
    ne = energy.shape[0]

    # number of blocks
    nb = hamiltionian_obj.Bmin.shape[0]
    # start and end index of each block in python indexing
    bmax = hamiltionian_obj.Bmax - 1
    bmin = hamiltionian_obj.Bmin - 1

    # fix nbc to 2 for the given solution
    # todo calculate it
    # nbc = 2

    # block sizes after matrix multiplication
    bmax_mm = bmax[nbc - 1:nb:nbc]
    bmin_mm = bmin[0:nb:nbc]
    # number of blocks after matrix multiplication
    nb_mm = bmax_mm.size
    # larges block length after matrix multiplication
    lb_max_mm = np.max(bmax_mm - bmin_mm + 1)

    # create empty buffer for screened interaction
    # in block format
    wr_diag, wr_upper, wl_diag, wl_upper, wg_diag, wg_upper = matrix_creation.initialize_block_G(
        ne, nb_mm, lb_max_mm)
    xr_diag = np.zeros((ne, nb_mm, lb_max_mm, lb_max_mm), dtype=np.complex128)
    # set number of mkl threads
    mkl.set_num_threads(mkl_threads)

    # todo remove this
    # not used inside rgf_W
    index_e = np.arange(ne)

    for ie in range(ne):
        out = rgf_W.rgf_w_opt(vh,
                              pg[ie],
                              pl[ie],
                              pr[ie],
                              bmax,
                              bmin,
                              wg_diag[ie],
                              wg_upper[ie],
                              wl_diag[ie],
                              wl_upper[ie],
                              wr_diag[ie],
                              wr_upper[ie],
                              xr_diag[ie],
                              dosw[ie],
                              new[ie],
                              npw[ie],
                              nbc,
                              index_e[ie],
                              factor[ie],
                              block_inv=block_inv,
                              use_dace=use_dace,
                              validate_dace=validate_dace)
        times += out
    print("Time symmetrize: ", times[0])
    print("Time sr,lg,ll arrays: ", times[1])
    print("Time scattering obc: ", times[2])
    print("Time beyn obc: ", times[3])
    print("Time dl obc: ", times[4])
    print("Time inversion: ", times[5])

    # Calculate F1, F2, which are the relative errors of GR-GA = GG-GL
    F1 = np.max(np.abs(dosw - (new + npw)) / (np.abs(dosw) + 1e-6), axis=1)
    F2 = np.max(np.abs(dosw - (new + npw)) /
                (np.abs(new + npw) + 1e-6), axis=1)

    buf_recv_r = np.empty((dosw.shape[1]), dtype=np.complex128)
    buf_send_r = np.empty((dosw.shape[1]), dtype=np.complex128)
    buf_recv_l = np.empty((dosw.shape[1]), dtype=np.complex128)
    buf_send_l = np.empty((dosw.shape[1]), dtype=np.complex128)
    if size > 1:
        if rank == 0:
            buf_send_r[:] = dosw[ne - 1, :]
            comm.Sendrecv(sendbuf=buf_send_r, dest=rank + 1,
                          recvbuf=buf_recv_r, source=rank + 1)

        elif rank == size - 1:
            buf_send_l[:] = dosw[0, :]
            comm.Sendrecv(sendbuf=buf_send_l, dest=rank - 1,
                          recvbuf=buf_recv_l, source=rank - 1)
        else:
            buf_send_r[:] = dosw[ne - 1, :]
            buf_send_l[:] = dosw[0, :]
            comm.Sendrecv(sendbuf=buf_send_r, dest=rank + 1,
                          recvbuf=buf_recv_r, source=rank + 1)
            comm.Sendrecv(sendbuf=buf_send_l, dest=rank - 1,
                          recvbuf=buf_recv_l, source=rank - 1)

    # Remove individual peaks (To-Do: improve this part by sending boundary elements to the next process)
    if size == 1:
        dDOSm = np.concatenate(([0], np.max(np.abs(dosw[1:ne - 1, :] / (dosw[0:ne - 2, :] + 1)),
                                            axis=1), [np.max(np.abs(dosw[ne - 1, :] / (dosw[ne - 2, :] + 1)))]))
        dDOSp = np.concatenate(([np.max(np.abs(dosw[0, :] / (dosw[1, :] + 1)))],
                                np.max(np.abs(dosw[1:ne - 1, :] / (dosw[2:ne, :] + 1)), axis=1), [0]))
    elif rank == 0:
        dDOSm = np.concatenate(([0], np.max(np.abs(dosw[1:ne - 1, :] / (dosw[0:ne - 2, :] + 1)),
                                            axis=1), [np.max(np.abs(dosw[ne - 1, :] / (dosw[ne - 2, :] + 1)))]))
        dDOSp = np.concatenate(([np.max(np.abs(dosw[0, :] / (dosw[1, :] + 1)))],
                                np.max(np.abs(dosw[1:ne - 1, :] / (dosw[2:ne, :] + 1)),
                                       axis=1), [np.max(np.abs(dosw[ne - 1, :] / (buf_recv_r + 1)))]))
    elif rank == size - 1:
        dDOSm = np.concatenate(([np.max(np.abs(dosw[0, :] / (buf_recv_l + 1)))],
                                np.max(np.abs(dosw[1:ne - 1, :] / (dosw[0:ne - 2, :] + 1)),
                                       axis=1), [np.max(np.abs(dosw[ne - 1, :] / (dosw[ne - 2, :] + 1)))]))
        dDOSp = np.concatenate(([np.max(np.abs(dosw[0, :] / (dosw[1, :] + 1)))],
                                np.max(np.abs(dosw[1:ne - 1, :] / (dosw[2:ne, :] + 1)), axis=1), [0]))
    else:
        dDOSm = np.concatenate(([np.max(np.abs(dosw[0, :] / (buf_recv_l + 1)))],
                                np.max(np.abs(dosw[1:ne - 1, :] / (dosw[0:ne - 2, :] + 1)),
                                       axis=1), [np.max(np.abs(dosw[ne - 1, :] / (dosw[ne - 2, :] + 1)))]))
        dDOSp = np.concatenate(([np.max(np.abs(dosw[0, :] / (dosw[1, :] + 1)))],
                                np.max(np.abs(dosw[1:ne - 1, :] / (dosw[2:ne, :] + 1)),
                                       axis=1), [np.max(np.abs(dosw[ne - 1, :] / (buf_recv_r + 1)))]))

    # Find indices of elements satisfying the conditions
    ind_zeros = np.where((F1 > 0.1) | (F2 > 0.1) |
                         ((dDOSm > 5) & (dDOSp > 5)))[0]

    # Remove the identified peaks and errors
    for index in ind_zeros:
        wr_diag[index, :, :, :] = 0
        wr_upper[index, :, :, :] = 0
        wl_diag[index, :, :, :] = 0
        wl_upper[index, :, :, :] = 0
        wg_diag[index, :, :, :] = 0
        wg_upper[index, :, :, :] = 0

    return wg_diag, wg_upper, wl_diag, wl_upper, wr_diag, wr_upper, nb_mm, lb_max_mm


def p2w_mpi_cpu_alt(
    hamiltionian_obj: object,
    ij2ji: npt.NDArray[np.int32],
    rows: npt.NDArray[np.int32],
    columns: npt.NDArray[np.int32],
    pg: npt.NDArray[np.complex128],
    pl: npt.NDArray[np.complex128],
    pr: npt.NDArray[np.complex128],
    vh: npt.NDArray[np.complex128],
    factors: npt.NDArray[np.float64],
    map_diag_mm2m: npt.NDArray[np.int32],
    map_upper_mm2m: npt.NDArray[np.int32],
    map_lower_mm2m: npt.NDArray[np.int32],
    nbc: np.int32,
    mkl_threads: int = 1
) -> typing.Tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128], npt.NDArray[np.complex128]]:
    """
    Calculates the screened interaction on the cpu.
    Uses only mkl threading.
    Splits up the screened interaction calculations into six parts:
    - Symmetrization of the polarization.
    - Change of the format into vectors of sparse matrices
    + Calculation of helper variables.
    + calculation of the blocks for boundary conditions.
    - calculation of the beyn boundary conditions.
    - calculation of the dl boundary conditions.
    - Inversion
    - back transformation into 2D format

    Args:
        hamiltionian_obj (object): Class containing the hamiltonian information
        ij2ji (npt.NDArray[np.int32]): Vector to transpose in orbitals
        rows (npt.NDArray[np.int32]): Non zero row indices of input matrices
        columns (npt.NDArray[np.int32]): Non zero column indices of input matrices
        pg (npt.NDArray[np.complex128]): Greater polarization, (#energy, #nnz)
        pl (npt.NDArray[np.complex128]): Greater polarization, (#energy, #nnz)
        pr (npt.NDArray[np.complex128]): Greater polarization, (#energy, #nnz)
        vh (npt.NDArray[np.complex128]): Effective interaction, (#nnz)
        factors (npt.NDArray[np.float64]): Factors to smooth in energy, multiplied at the end, (#energy)
        map_diag_mm2m (npt.NDArray[np.int32]): Map from dense block diagonal to 2D
        map_upper_mm2m (npt.NDArray[np.int32]): Map from dense block upper to 2D
        map_upper_mm2m (npt.NDArray[np.int32]): Map from dense block lower to 2D
        mkl_threads (int, optional): Number of mkl threads to use. Defaults to 1.

    Returns:
        typing.Tuple[ npt.NDArray[np.complex128], npt.NDArray[np.complex128], npt.NDArray[np.complex128] ]: Greater, lesser, retarded screened interaction
    """

    # create timing vector
    times = np.zeros((10))

    # number of energy points and nonzero elements
    ne = pg.shape[0]
    no = pg.shape[1]
    nao = np.max(hamiltionian_obj.Bmax)

    # number of blocks
    nb = hamiltionian_obj.Bmin.shape[0]
    # start and end index of each block in python indexing
    bmax = hamiltionian_obj.Bmax - 1
    bmin = hamiltionian_obj.Bmin - 1
    # block lengths
    lb = bmax - bmin + 1
    # fix nbc to 2 for the given solution
    # todo calculate it
    # nbc = 2
    # block sizes after matrix multiplication
    bmax_mm = bmax[nbc - 1:nb:nbc]
    bmin_mm = bmin[0:nb:nbc]
    # number of blocks after matrix multiplication
    nb_mm = bmax_mm.size
    # larges block length after matrix multiplication
    lb_max_mm = np.max(bmax_mm - bmin_mm + 1)

    # create empty buffer for screened interaction
    # diagonal blocks
    xr_diag = np.zeros((ne, nb_mm, lb_max_mm, lb_max_mm), dtype=np.complex128)
    wg_diag = np.zeros((ne, nb_mm, lb_max_mm, lb_max_mm), dtype=np.complex128)
    wl_diag = np.zeros((ne, nb_mm, lb_max_mm, lb_max_mm), dtype=np.complex128)
    wr_diag = np.zeros((ne, nb_mm, lb_max_mm, lb_max_mm), dtype=np.complex128)
    # upper diagonal blocks
    wg_upper = np.zeros(
        (ne, nb_mm - 1, lb_max_mm, lb_max_mm), dtype=np.complex128)
    wl_upper = np.zeros(
        (ne, nb_mm - 1, lb_max_mm, lb_max_mm), dtype=np.complex128)
    wr_upper = np.zeros(
        (ne, nb_mm - 1, lb_max_mm, lb_max_mm), dtype=np.complex128)

    # create buffer for boundary conditions
    mr_sf = np.zeros((ne, 3, nbc * lb[0], nbc * lb[0]), dtype=np.complex128)
    mr_ef = np.zeros((ne, 3, nbc * lb[0], nbc * lb[0]), dtype=np.complex128)
    lg_sf = np.zeros((ne, 3, nbc * lb[0], nbc * lb[0]), dtype=np.complex128)
    lg_ef = np.zeros((ne, 3, nbc * lb[0], nbc * lb[0]), dtype=np.complex128)
    ll_sf = np.zeros((ne, 3, nbc * lb[0], nbc * lb[0]), dtype=np.complex128)
    ll_ef = np.zeros((ne, 3, nbc * lb[0], nbc * lb[0]), dtype=np.complex128)
    dxr_sf = np.zeros((ne, 1, nbc * lb[0], nbc * lb[0]), dtype=np.complex128)
    dxr_ef = np.zeros((ne, 1, nbc * lb[0], nbc * lb[0]), dtype=np.complex128)
    dmr_sf = np.zeros((ne, 1, nbc * lb[0], nbc * lb[0]), dtype=np.complex128)
    dmr_ef = np.zeros((ne, 1, nbc * lb[0], nbc * lb[0]), dtype=np.complex128)
    dlg_sf = np.zeros((ne, 1, nbc * lb[0], nbc * lb[0]), dtype=np.complex128)
    dlg_ef = np.zeros((ne, 1, nbc * lb[0], nbc * lb[0]), dtype=np.complex128)
    dll_sf = np.zeros((ne, 1, nbc * lb[0], nbc * lb[0]), dtype=np.complex128)
    dll_ef = np.zeros((ne, 1, nbc * lb[0], nbc * lb[0]), dtype=np.complex128)
    vh_sf = np.zeros((ne, 1, nbc * lb[0], nbc * lb[0]), dtype=np.complex128)
    vh_ef = np.zeros((ne, 1, nbc * lb[0], nbc * lb[0]), dtype=np.complex128)
    dvh_sf = np.zeros((ne, 1, nbc * lb[0], nbc * lb[0]), dtype=np.complex128)
    dvh_ef = np.zeros((ne, 1, nbc * lb[0], nbc * lb[0]), dtype=np.complex128)

    # set number of mkl threads
    mkl.set_num_threads(mkl_threads)

    times[0] = -time.perf_counter()
    # todo try to not additionally symmetrize
    # Anti-Hermitian symmetrizing of pl and pg
    pg = 1j * np.imag(pg)
    pg = (pg - pg[:, ij2ji].conjugate()) / 2
    pl = 1j * np.imag(pl)
    pl = (pl - pl[:, ij2ji].conjugate()) / 2
    # pr has to be derived from pl and pg and then has to be symmetrized
    pr = 1j * np.imag(pg - pl) / 2
    pr = (pr + pr[:, ij2ji]) / 2
    times[0] += time.perf_counter()

    # compute helper arrays
    times[1] = -time.perf_counter()
    vh_sparse = sparse.csr_array(
        (vh, (rows, columns)), shape=(nao, nao), dtype=np.complex128)
    mr_vec = np.ndarray((ne, ), dtype=object)
    lg_vec = np.ndarray((ne, ), dtype=object)
    ll_vec = np.ndarray((ne, ), dtype=object)
    for i in range(ne):
        pg_s = sparse.csr_array((pg[i, :, ], (rows, columns)), shape=(
            nao, nao), dtype=np.complex128)
        pl_s = sparse.csr_array((pl[i, :, ], (rows, columns)), shape=(
            nao, nao), dtype=np.complex128)
        pr_s = sparse.csr_array((pr[i, :, ], (rows, columns)), shape=(
            nao, nao), dtype=np.complex128)
        mr_vec[i], lg_vec[i], ll_vec[i] = obc_w_cpu.obc_w_sl(
            vh_sparse, pg_s, pl_s, pr_s, nao)
        obc_w_cpu.obc_w_sc(pg_s, pl_s, pr_s, vh_sparse, mr_sf[i], mr_ef[i], lg_sf[i], lg_ef[i], ll_sf[i], ll_ef[i],
                           dmr_sf[i], dmr_ef[i], dlg_sf[i], dlg_ef[i], dll_sf[i], dll_ef[i], vh_sf[i], vh_ef[i], bmax,
                           bmin, nbc)
    times[1] += time.perf_counter()

    times[2] = -time.perf_counter()
    cond_l = np.zeros((ne), dtype=bool)
    cond_r = np.zeros((ne), dtype=bool)
    for i in range(ne):
        cond_r[i], cond_l[i] = obc_w_cpu.obc_w_beyn(dxr_sf[i], dxr_ef[i], mr_sf[i], mr_ef[i], vh_sf[i], vh_ef[i],
                                                    dmr_sf[i], dmr_ef[i], dvh_sf[i], dvh_ef[i])

    times[2] += time.perf_counter()

    times[3] = -time.perf_counter()
    for i in range(ne):
        if cond_r[i] and cond_l[i]:
            # if not np.isnan(cond_r[i]) and not np.isnan(cond_l[i]):
            obc_w_cpu.obc_w_dl(dxr_sf[i], dxr_ef[i], lg_sf[i], lg_ef[i], ll_sf[i], ll_ef[i], mr_sf[i], mr_ef[i],
                               dlg_sf[i], dll_sf[i], dlg_ef[i], dll_ef[i])
    times[3] += time.perf_counter()

    times[4] = -time.perf_counter()
    # calculate the inversion for every energy point
    for i in range(ne):
        if cond_r[i] and cond_l[i]:
            # if not np.isnan(cond_r[i]) and not np.isnan(cond_l[i]):
            matrix_inversion_w.rgf(bmax_mm, bmin_mm, vh_sparse, mr_vec[i], lg_vec[i], ll_vec[i], factors[i], wg_diag[i],
                                   wg_upper[i], wl_diag[i], wl_upper[i], wr_diag[i], wr_upper[i], xr_diag[i],
                                   dmr_ef[i, 0], dlg_ef[i, 0], dll_ef[i,
                                                                      0], dvh_ef[i, 0], dmr_sf[i, 0], dlg_sf[i, 0],
                                   dll_sf[i, 0], dvh_sf[i, 0])
    times[4] += time.perf_counter()

    times[5] = -time.perf_counter()
    # lower blocks from identity
    wg_lower = -wg_upper.conjugate().transpose((0, 1, 3, 2))
    wl_lower = -wl_upper.conjugate().transpose((0, 1, 3, 2))
    wr_lower = wr_upper.transpose((0, 1, 3, 2))

    # tranform to 2D format
    wg = change_format.block2sparse_energy_alt(map_diag_mm2m,
                                               map_upper_mm2m,
                                               map_lower_mm2m,
                                               wg_diag,
                                               wg_upper,
                                               wg_lower,
                                               no,
                                               ne,
                                               energy_contiguous=False)
    wl = change_format.block2sparse_energy_alt(map_diag_mm2m,
                                               map_upper_mm2m,
                                               map_lower_mm2m,
                                               wl_diag,
                                               wl_upper,
                                               wl_lower,
                                               no,
                                               ne,
                                               energy_contiguous=False)
    wr = change_format.block2sparse_energy_alt(map_diag_mm2m,
                                               map_upper_mm2m,
                                               map_lower_mm2m,
                                               wr_diag,
                                               wr_upper,
                                               wr_lower,
                                               no,
                                               ne,
                                               energy_contiguous=False)
    times[5] += time.perf_counter()

    print("Time symmetrize: ", times[0])
    print("Time to list + sr,lg,ll arrays + scattering obc: ", times[1])
    print("Time beyn obc: ", times[2])
    print("Time dl obc: ", times[3])
    print("Time inversion: ", times[4])
    print("Time block: ", times[5])
    return wg, wl, wr
