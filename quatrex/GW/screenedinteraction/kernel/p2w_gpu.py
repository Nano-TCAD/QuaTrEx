# Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.
""" Functions to calculate the screened interaction on the gpu See README.md for more information. """
import concurrent.futures
from itertools import repeat
import numpy as np
import numpy.typing as npt
import cupy as cp
from scipy import sparse
from cupyx.scipy import sparse as cusparse
import mkl
import typing
from quatrex.utils import change_format
from quatrex.utils import matrix_creation
from quatrex.utils.matrix_creation import homogenize_matrix_Rnosym, \
                                            extract_small_matrix_blocks
from block_tri_solvers import matrix_inversion_w, rgf_W, rgf_W_GPU
from quatrex.OBC import obc_w_gpu
from quatrex.OBC import obc_w_cpu
import time

def p2w_pool_mpi_gpu_split(
    hamiltionian_obj: object,
    energy: npt.NDArray[np.float64],
    pg: npt.NDArray[np.complex128],
    pl: npt.NDArray[np.complex128],
    pr: npt.NDArray[np.complex128],
    vh: npt.NDArray[np.complex128],
    map_diag,
    map_upper,
    map_lower,
    rows,
    columns,
    ij2ji,
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
    validate_dace: bool = False):
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
    #nbc = 2
    lb_vec = bmax - bmin + 1
    lb_start = lb_vec[0]
    lb_end = lb_vec[nb - 1]

    dvh_sd = np.zeros((ne, lb_start, lb_start), dtype=np.complex128)
    dvh_ed = np.zeros((ne, lb_end, lb_end), dtype=np.complex128)
    dmr_sd = np.zeros((ne, lb_start, lb_start), dtype=np.complex128)
    dmr_ed = np.zeros((ne, lb_end, lb_end), dtype=np.complex128)
    dlg_sd = np.zeros((ne, lb_start, lb_start), dtype=np.complex128)
    dlg_ed = np.zeros((ne, lb_end, lb_end), dtype=np.complex128)
    dll_sd = np.zeros((ne, lb_start, lb_start), dtype=np.complex128)
    dll_ed = np.zeros((ne, lb_end, lb_end), dtype=np.complex128)
    condl = np.zeros((ne), dtype = np.float64)
    condr = np.zeros((ne), dtype = np.float64)


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
            (PR00, PR01, PR10, _) = extract_small_matrix_blocks(pr[ie][bmin[0]:bmax[0]+1, bmin[0]:bmax[0]+1],\
                                                                      pr[ie][bmin[0]:bmax[0]+1, bmin[1]:bmax[1]+1], \
                                                                      pr[ie][bmin[1]:bmax[1]+1, bmin[0]:bmax[0]+1], NCpSC, 'L')
            pr[ie] = homogenize_matrix_Rnosym(PR00,
                                        PR01,
                                        PR10, len(bmax))
            (PL00, PL01, PL10, _) = extract_small_matrix_blocks(pl[ie][bmin[0]:bmax[0]+1, bmin[0]:bmax[0]+1],\
                                                                        pl[ie][bmin[0]:bmax[0]+1, bmin[1]:bmax[1]+1], \
                                                                        pl[ie][bmin[1]:bmax[1]+1, bmin[0]:bmax[0]+1], NCpSC, 'L')
            pl[ie] = homogenize_matrix_Rnosym(PL00,
                                        PL01,
                                        PL10,
                                        len(bmax))
            (PG00, PG01, PG10, _) = extract_small_matrix_blocks(pg[ie][bmin[0]:bmax[0]+1, bmin[0]:bmax[0]+1],\
                                                                        pg[ie][bmin[0]:bmax[0]+1, bmin[1]:bmax[1]+1], \
                                                                        pg[ie][bmin[1]:bmax[1]+1, bmin[0]:bmax[0]+1], NCpSC, 'L')
            pg[ie] = homogenize_matrix_Rnosym(PG00,
                                        PG01,
                                        PG10,
                                        len(bmax))

    # Create a process pool with num_worker workers
    comm.Barrier()
    if rank == 0:
        time_pre_OBC += time.perf_counter()
        print("Time for pre-processing OBC: %.3f s" % time_pre_OBC, flush = True)
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
                               repeat(validate_dace),repeat(ref_flag)
                                )
        for idx, res in enumerate(results):
            condl[idx] = res[0]
            condr[idx] = res[1]
    

    l_defect = np.count_nonzero(np.isnan(condl))
    r_defect = np.count_nonzero(np.isnan(condr))

    if l_defect > 0 or r_defect > 0:
        print("Warning: %d left and %d right boundary conditions are not satisfied." % (l_defect, r_defect))

    comm.Barrier()
    if rank == 0:
        time_OBC += time.perf_counter()
        print("Time for OBC: %.3f s" % time_OBC, flush = True)
        time_GF_trafo = -time.perf_counter()

    comm.Barrier()
    if rank == 0:
        time_GF_trafo += time.perf_counter()
        print("Time for W transformation: %.3f s" % time_GF_trafo, flush = True)
        #time_GF = -time.perf_counter()
        time_spmm = -time.perf_counter()

    vh_gpu = cp.sparse.csr_matrix(vh)
    vh_ct_gpu = vh_gpu.conj().transpose()
    nao = vh.shape[0]
    #mr_gpu = cp.ndarray((ne,), dtype=object)
    mr_gpu = []
    lg_gpu = []
    ll_gpu = []
    for i in range(ne):
        pg_gpu = cp.sparse.csr_matrix(pg[i])
        pl_gpu = cp.sparse.csr_matrix(pl[i])
        pr_gpu = cp.sparse.csr_matrix(pr[i])
        mr_gpu.append(cp.sparse.identity(nao) - vh_gpu @ pr_gpu)
        lg_gpu.append(vh_gpu @ pg_gpu @ vh_ct_gpu)
        ll_gpu.append(vh_gpu @ pl_gpu @ vh_ct_gpu)
    cp.cuda.runtime.deviceSynchronize()

    comm.Barrier()
    if rank == 0:
        time_spmm += time.perf_counter()
        print("Time for spmm: %.3f s" % time_spmm, flush = True)
        time_GF = -time.perf_counter()

    with concurrent.futures.ThreadPoolExecutor(max_workers=worker_num) as executor:
        # Use the map function to apply the inv_matrices function to each pair of matrices in parallel
        #executor.map(
        results = executor.map(
                    rgf_W_GPU.rgf_w_opt_standalone,
                    repeat(vh),
                    lg_gpu, ll_gpu, mr_gpu,
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
                    repeat(validate_dace),repeat(ref_flag))
        # for res in results:
        #    assert isinstance(res, np.ndarray)
    comm.Barrier()
    if rank == 0:
        time_GF += time.perf_counter()
        print("Time for W: %.3f s" % time_GF, flush = True)
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
        print("Time for post-processing: %.3f s" % time_post_proc, flush = True)

    return wg_diag, wg_upper, wl_diag, wl_upper, wr_diag, wr_upper, nb_mm, lb_max_mm, ind_zeros

def p2w_mpi_gpu(
    hamiltionian_obj: object, ij2ji: npt.NDArray[np.int32], rows: npt.NDArray[np.int32], columns: npt.NDArray[np.int32],
    pg: npt.NDArray[np.complex128], pl: npt.NDArray[np.complex128], pr: npt.NDArray[np.complex128],
    vh: npt.NDArray[np.complex128], factors: npt.NDArray[np.float64], map_diag_mm2m: npt.NDArray[np.int32],
    map_upper_mm2m: npt.NDArray[np.int32], map_lower_mm2m: npt.NDArray[np.int32]
) -> typing.Tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128], npt.NDArray[np.complex128]]:
    """
    Calculates the screened interaction on the cpu.
    Uses the gpu for most of the calculations.
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
    nbc = 2
    # block sizes after matrix multiplication
    bmax_mm = bmax[nbc - 1:nb:nbc]
    bmin_mm = bmin[0:nb:nbc]
    # number of blocks after matrix multiplication
    nb_mm = bmax_mm.size
    # larges block length after matrix multiplication
    lb_max_mm = np.max(bmax_mm - bmin_mm + 1)

    # create empty buffer for screened interaction
    # diagonal blocks
    wg_diag = np.zeros((ne, nb_mm, lb_max_mm, lb_max_mm), dtype=np.complex128)
    wl_diag = np.zeros((ne, nb_mm, lb_max_mm, lb_max_mm), dtype=np.complex128)
    wr_diag = np.zeros((ne, nb_mm, lb_max_mm, lb_max_mm), dtype=np.complex128)
    # upper diagonal blocks
    wg_upper = np.zeros((ne, nb_mm - 1, lb_max_mm, lb_max_mm), dtype=np.complex128)
    wl_upper = np.zeros((ne, nb_mm - 1, lb_max_mm, lb_max_mm), dtype=np.complex128)
    wr_upper = np.zeros((ne, nb_mm - 1, lb_max_mm, lb_max_mm), dtype=np.complex128)
    xr_diag_gpu = cp.zeros((ne, nb_mm, lb_max_mm, lb_max_mm), dtype=cp.complex128)
    wg_diag_gpu = cp.zeros((ne, nb_mm, lb_max_mm, lb_max_mm), dtype=cp.complex128)
    wl_diag_gpu = cp.zeros((ne, nb_mm, lb_max_mm, lb_max_mm), dtype=cp.complex128)
    wr_diag_gpu = cp.zeros((ne, nb_mm, lb_max_mm, lb_max_mm), dtype=cp.complex128)
    # upper diagonal blocks
    wg_upper_gpu = cp.zeros((ne, nb_mm - 1, lb_max_mm, lb_max_mm), dtype=cp.complex128)
    wl_upper_gpu = cp.zeros((ne, nb_mm - 1, lb_max_mm, lb_max_mm), dtype=cp.complex128)
    wr_upper_gpu = cp.zeros((ne, nb_mm - 1, lb_max_mm, lb_max_mm), dtype=cp.complex128)

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

    # vh = np.copy(vh)
    pg = np.copy(pg)
    pl = np.copy(pl)
    pr = np.copy(pr)
    vh_gpu = cp.empty_like(vh)
    pg_gpu = cp.empty_like(pg)
    pl_gpu = cp.empty_like(pl)
    pr_gpu = cp.empty_like(pr)
    ij2ji_gpu = cp.empty_like(ij2ji)
    rows_gpu = cp.empty_like(rows)
    columns_gpu = cp.empty_like(columns)
    mr_sf_gpu = cp.empty_like(mr_sf)
    mr_ef_gpu = cp.empty_like(mr_ef)
    lg_sf_gpu = cp.empty_like(lg_sf)
    lg_ef_gpu = cp.empty_like(lg_ef)
    ll_sf_gpu = cp.empty_like(ll_sf)
    ll_ef_gpu = cp.empty_like(ll_ef)
    dmr_sf_gpu = cp.empty_like(dmr_sf)
    dmr_ef_gpu = cp.empty_like(dmr_ef)
    dlg_sf_gpu = cp.empty_like(dlg_sf)
    dlg_ef_gpu = cp.empty_like(dlg_ef)
    dll_sf_gpu = cp.empty_like(dll_sf)
    dll_ef_gpu = cp.empty_like(dll_ef)
    vh_sf_gpu = cp.empty_like(vh_sf)
    vh_ef_gpu = cp.empty_like(vh_ef)
    dvh_sf_gpu = cp.empty_like(dvh_sf)
    dvh_ef_gpu = cp.empty_like(dvh_ef)

    times[0] = -time.perf_counter()

    # load data to the gpu
    vh_gpu.set(vh)
    pg_gpu.set(pg)
    pl_gpu.set(pl)
    pr_gpu.set(pr)
    ij2ji_gpu.set(ij2ji)
    rows_gpu.set(rows)
    columns_gpu.set(columns)

    # todo try to not additionally symmetrize
    # Anti-Hermitian symmetrizing of pl and pg
    pg_gpu = 1j * cp.imag(pg_gpu)
    pg_gpu = (pg_gpu - pg_gpu[:, ij2ji_gpu].conjugate()) / 2
    pl_gpu = 1j * cp.imag(pl_gpu)
    pl_gpu = (pl_gpu - pl_gpu[:, ij2ji_gpu].conjugate()) / 2
    # pr has to be derived from pl and pg and then has to be symmetrized
    pr_gpu = 1j * cp.imag(pg_gpu - pl_gpu) / 2
    pr_gpu = (pr_gpu + pr_gpu[:, ij2ji_gpu]) / 2

    # unload
    # pg_gpu.get(out=pg)
    # pl_gpu.get(out=pl)
    # pr_gpu.get(out=pr)

    times[0] += time.perf_counter()

    # compute helper arrays
    times[1] = -time.perf_counter()
    vh_sparse = cusparse.csr_matrix((vh_gpu, (rows_gpu, columns_gpu)), shape=(nao, nao), dtype=cp.complex128)
    mr_vec = []
    lg_vec = []
    ll_vec = []
    for i in range(ne):
        pg_s = cusparse.csr_matrix((pg_gpu[i, :, ], (rows_gpu, columns_gpu)), shape=(nao, nao), dtype=cp.complex128)
        pl_s = cusparse.csr_matrix((pl_gpu[i, :, ], (rows_gpu, columns_gpu)), shape=(nao, nao), dtype=cp.complex128)
        pr_s = cusparse.csr_matrix((pr_gpu[i, :, ], (rows_gpu, columns_gpu)), shape=(nao, nao), dtype=cp.complex128)
        mr, lg, ll = obc_w_gpu.obc_w_sl(vh_sparse, pg_s, pl_s, pr_s, nao)
        mr_vec.append(mr)
        lg_vec.append(lg)
        ll_vec.append(ll)
        obc_w_gpu.obc_w_sc(pg_s, pl_s, pr_s, vh_sparse, mr_sf_gpu[i], mr_ef_gpu[i], lg_sf_gpu[i], lg_ef_gpu[i],
                           ll_sf_gpu[i], ll_ef_gpu[i], dmr_sf_gpu[i], dmr_ef_gpu[i], dlg_sf_gpu[i], dlg_ef_gpu[i],
                           dll_sf_gpu[i], dll_ef_gpu[i], vh_sf_gpu[i], vh_ef_gpu[i], bmax, bmin, nbc)
    # unload
    mr_sf_gpu.get(out=mr_sf)
    mr_ef_gpu.get(out=mr_ef)
    lg_sf_gpu.get(out=lg_sf)
    lg_ef_gpu.get(out=lg_ef)
    ll_sf_gpu.get(out=ll_sf)
    ll_ef_gpu.get(out=ll_ef)
    dmr_sf_gpu.get(out=dmr_sf)
    dmr_ef_gpu.get(out=dmr_ef)
    dlg_sf_gpu.get(out=dlg_sf)
    dlg_ef_gpu.get(out=dlg_ef)
    dll_sf_gpu.get(out=dll_sf)
    dll_ef_gpu.get(out=dll_ef)
    vh_sf_gpu.get(out=vh_sf)
    vh_ef_gpu.get(out=vh_ef)

    times[1] += time.perf_counter()

    times[2] = -time.perf_counter()
    cond_l = np.zeros((ne), dtype=np.float64)
    cond_r = np.zeros((ne), dtype=np.float64)
    for i in range(ne):
        cond_r[i], cond_l[i] = obc_w_cpu.obc_w_beyn(dxr_sf[i], dxr_ef[i], mr_sf[i], mr_ef[i], vh_sf[i], vh_ef[i],
                                                    dmr_sf[i], dmr_ef[i], dvh_sf[i], dvh_ef[i])

    times[2] += time.perf_counter()

    times[3] = -time.perf_counter()
    for i in range(ne):
        if not np.isnan(cond_r[i]) and not np.isnan(cond_l[i]):
            obc_w_cpu.obc_w_dl(dxr_sf[i], dxr_ef[i], lg_sf[i], lg_ef[i], ll_sf[i], ll_ef[i], mr_sf[i], mr_ef[i],
                               dlg_sf[i], dll_sf[i], dlg_ef[i], dll_ef[i])
    times[3] += time.perf_counter()

    times[4] = -time.perf_counter()

    dmr_ef_gpu.set(dmr_ef)
    dlg_ef_gpu.set(dlg_ef)
    dll_ef_gpu.set(dll_ef)
    dvh_ef_gpu.set(dvh_ef)
    dmr_sf_gpu.set(dmr_sf)
    dlg_sf_gpu.set(dlg_sf)
    dll_sf_gpu.set(dll_sf)
    dvh_sf_gpu.set(dvh_sf)
    # calculate the inversion for every energy point
    for i in range(ne):
        if not np.isnan(cond_r[i]) and not np.isnan(cond_l[i]):
            matrix_inversion_w.rgf_gpu(bmax_mm, bmin_mm, vh_sparse, mr_vec[i], lg_vec[i], ll_vec[i], factors[i],
                                       wg_diag_gpu[i], wg_upper_gpu[i], wl_diag_gpu[i], wl_upper_gpu[i], wr_diag_gpu[i],
                                       wr_upper_gpu[i], xr_diag_gpu[i], dmr_ef_gpu[i, 0], dlg_ef_gpu[i, 0],
                                       dll_ef_gpu[i, 0], dvh_ef_gpu[i, 0], dmr_sf_gpu[i, 0], dlg_sf_gpu[i, 0],
                                       dll_sf_gpu[i, 0], dvh_sf_gpu[i, 0])
    times[4] += time.perf_counter()

    times[5] = -time.perf_counter()
    # unload
    wg_diag_gpu.get(out=wg_diag)
    wg_upper_gpu.get(out=wg_upper)
    wl_diag_gpu.get(out=wl_diag)
    wl_upper_gpu.get(out=wl_upper)
    wr_diag_gpu.get(out=wr_diag)
    wr_upper_gpu.get(out=wr_upper)

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


def p2w_mpi_gpu_alt(
    hamiltionian_obj: object, ij2ji: npt.NDArray[np.int32], rows: npt.NDArray[np.int32], columns: npt.NDArray[np.int32],
    pg: npt.NDArray[np.complex128], pl: npt.NDArray[np.complex128], pr: npt.NDArray[np.complex128],
    vh: npt.NDArray[np.complex128], factors: npt.NDArray[np.float64], map_diag_mm2m: npt.NDArray[np.int32],
    map_upper_mm2m: npt.NDArray[np.int32], map_lower_mm2m: npt.NDArray[np.int32]
) -> typing.Tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128], npt.NDArray[np.complex128]]:
    """
    Calculates the screened interaction on the cpu.
    Uses the gpu for most of the calculations.
    Splits up the screened interaction calculations into six parts:
    - Symmetrization of the polarization.
    - Change of the format into vectors of sparse matrices
      + Calculation of helper variables.
      + calculation of the blocks for boundary conditions.
    - calculation of the beyn boundary conditions.
    - calculation of the dl boundary conditions.
    - Inversion
    - back transformation into 2D format
    Alternative only one for loop

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
    nbc = 2
    # block sizes after matrix multiplication
    bmax_mm = bmax[nbc - 1:nb:nbc]
    bmin_mm = bmin[0:nb:nbc]
    # number of blocks after matrix multiplication
    nb_mm = bmax_mm.size
    # larges block length after matrix multiplication
    lb_max_mm = np.max(bmax_mm - bmin_mm + 1)

    # create empty buffer for screened interaction
    # diagonal blocks
    wg_diag = np.zeros((ne, nb_mm, lb_max_mm, lb_max_mm), dtype=np.complex128)
    wl_diag = np.zeros((ne, nb_mm, lb_max_mm, lb_max_mm), dtype=np.complex128)
    wr_diag = np.zeros((ne, nb_mm, lb_max_mm, lb_max_mm), dtype=np.complex128)
    # upper diagonal blocks
    wg_upper = np.zeros((ne, nb_mm - 1, lb_max_mm, lb_max_mm), dtype=np.complex128)
    wl_upper = np.zeros((ne, nb_mm - 1, lb_max_mm, lb_max_mm), dtype=np.complex128)
    wr_upper = np.zeros((ne, nb_mm - 1, lb_max_mm, lb_max_mm), dtype=np.complex128)
    xr_diag_gpu = cp.zeros((ne, nb_mm, lb_max_mm, lb_max_mm), dtype=cp.complex128)
    wg_diag_gpu = cp.zeros((ne, nb_mm, lb_max_mm, lb_max_mm), dtype=cp.complex128)
    wl_diag_gpu = cp.zeros((ne, nb_mm, lb_max_mm, lb_max_mm), dtype=cp.complex128)
    wr_diag_gpu = cp.zeros((ne, nb_mm, lb_max_mm, lb_max_mm), dtype=cp.complex128)
    # upper diagonal blocks
    wg_upper_gpu = cp.zeros((ne, nb_mm - 1, lb_max_mm, lb_max_mm), dtype=cp.complex128)
    wl_upper_gpu = cp.zeros((ne, nb_mm - 1, lb_max_mm, lb_max_mm), dtype=cp.complex128)
    wr_upper_gpu = cp.zeros((ne, nb_mm - 1, lb_max_mm, lb_max_mm), dtype=cp.complex128)

    # create buffer for boundary conditions
    mr_sf = np.zeros((3, nbc * lb[0], nbc * lb[0]), dtype=np.complex128)
    mr_ef = np.zeros((3, nbc * lb[0], nbc * lb[0]), dtype=np.complex128)
    lg_sf = np.zeros((3, nbc * lb[0], nbc * lb[0]), dtype=np.complex128)
    lg_ef = np.zeros((3, nbc * lb[0], nbc * lb[0]), dtype=np.complex128)
    ll_sf = np.zeros((3, nbc * lb[0], nbc * lb[0]), dtype=np.complex128)
    ll_ef = np.zeros((3, nbc * lb[0], nbc * lb[0]), dtype=np.complex128)
    dxr_sf = np.zeros((1, nbc * lb[0], nbc * lb[0]), dtype=np.complex128)
    dxr_ef = np.zeros((1, nbc * lb[0], nbc * lb[0]), dtype=np.complex128)
    dmr_sf = np.zeros((1, nbc * lb[0], nbc * lb[0]), dtype=np.complex128)
    dmr_ef = np.zeros((1, nbc * lb[0], nbc * lb[0]), dtype=np.complex128)
    dlg_sf = np.zeros((1, nbc * lb[0], nbc * lb[0]), dtype=np.complex128)
    dlg_ef = np.zeros((1, nbc * lb[0], nbc * lb[0]), dtype=np.complex128)
    dll_sf = np.zeros((1, nbc * lb[0], nbc * lb[0]), dtype=np.complex128)
    dll_ef = np.zeros((1, nbc * lb[0], nbc * lb[0]), dtype=np.complex128)
    vh_sf = np.zeros((1, nbc * lb[0], nbc * lb[0]), dtype=np.complex128)
    vh_ef = np.zeros((1, nbc * lb[0], nbc * lb[0]), dtype=np.complex128)
    dvh_sf = np.zeros((1, nbc * lb[0], nbc * lb[0]), dtype=np.complex128)
    dvh_ef = np.zeros((1, nbc * lb[0], nbc * lb[0]), dtype=np.complex128)

    # vh = np.copy(vh)
    pg = np.copy(pg)
    pl = np.copy(pl)
    pr = np.copy(pr)
    vh_gpu = cp.empty_like(vh)
    pg_gpu = cp.empty_like(pg)
    pl_gpu = cp.empty_like(pl)
    pr_gpu = cp.empty_like(pr)
    ij2ji_gpu = cp.empty_like(ij2ji)
    rows_gpu = cp.empty_like(rows)
    columns_gpu = cp.empty_like(columns)
    mr_sf_gpu = cp.empty_like(mr_sf)
    mr_ef_gpu = cp.empty_like(mr_ef)
    lg_sf_gpu = cp.empty_like(lg_sf)
    lg_ef_gpu = cp.empty_like(lg_ef)
    ll_sf_gpu = cp.empty_like(ll_sf)
    ll_ef_gpu = cp.empty_like(ll_ef)
    dmr_sf_gpu = cp.empty_like(dmr_sf)
    dmr_ef_gpu = cp.empty_like(dmr_ef)
    dlg_sf_gpu = cp.empty_like(dlg_sf)
    dlg_ef_gpu = cp.empty_like(dlg_ef)
    dll_sf_gpu = cp.empty_like(dll_sf)
    dll_ef_gpu = cp.empty_like(dll_ef)
    vh_sf_gpu = cp.empty_like(vh_sf)
    vh_ef_gpu = cp.empty_like(vh_ef)
    dvh_sf_gpu = cp.empty_like(dvh_sf)
    dvh_ef_gpu = cp.empty_like(dvh_ef)

    times[0] = -time.perf_counter()

    # load data to the gpu
    vh_gpu.set(vh)
    pg_gpu.set(pg)
    pl_gpu.set(pl)
    pr_gpu.set(pr)
    ij2ji_gpu.set(ij2ji)
    rows_gpu.set(rows)
    columns_gpu.set(columns)

    # todo try to not additionally symmetrize
    # Anti-Hermitian symmetrizing of pl and pg
    pg_gpu = 1j * cp.imag(pg_gpu)
    pg_gpu = (pg_gpu - pg_gpu[:, ij2ji_gpu].conjugate()) / 2
    pl_gpu = 1j * cp.imag(pl_gpu)
    pl_gpu = (pl_gpu - pl_gpu[:, ij2ji_gpu].conjugate()) / 2
    # pr has to be derived from pl and pg and then has to be symmetrized
    pr_gpu = 1j * cp.imag(pg_gpu - pl_gpu) / 2
    pr_gpu = (pr_gpu + pr_gpu[:, ij2ji_gpu]) / 2

    times[0] += time.perf_counter()

    vh_sparse = cusparse.csr_matrix((vh_gpu, (rows_gpu, columns_gpu)), shape=(nao, nao), dtype=cp.complex128)
    cond_l = 0.0
    cond_r = 0.0

    for i in range(ne):
        times[1] += -time.perf_counter()
        pg_s = cusparse.csr_matrix((pg_gpu[i, :, ], (rows_gpu, columns_gpu)), shape=(nao, nao), dtype=cp.complex128)
        pl_s = cusparse.csr_matrix((pl_gpu[i, :, ], (rows_gpu, columns_gpu)), shape=(nao, nao), dtype=cp.complex128)
        pr_s = cusparse.csr_matrix((pr_gpu[i, :, ], (rows_gpu, columns_gpu)), shape=(nao, nao), dtype=cp.complex128)
        mr, lg, ll = obc_w_gpu.obc_w_sl(vh_sparse, pg_s, pl_s, pr_s, nao)

        obc_w_gpu.obc_w_sc(pg_s, pl_s, pr_s, vh_sparse, mr_sf_gpu, mr_ef_gpu, lg_sf_gpu, lg_ef_gpu, ll_sf_gpu,
                           ll_ef_gpu, dmr_sf_gpu, dmr_ef_gpu, dlg_sf_gpu, dlg_ef_gpu, dll_sf_gpu, dll_ef_gpu, vh_sf_gpu,
                           vh_ef_gpu, bmax, bmin, nbc)
        times[1] += time.perf_counter()
        # unload
        mr_sf_gpu.get(out=mr_sf)
        mr_ef_gpu.get(out=mr_ef)
        lg_sf_gpu.get(out=lg_sf)
        lg_ef_gpu.get(out=lg_ef)
        ll_sf_gpu.get(out=ll_sf)
        ll_ef_gpu.get(out=ll_ef)
        dmr_sf_gpu.get(out=dmr_sf)
        dmr_ef_gpu.get(out=dmr_ef)
        dlg_sf_gpu.get(out=dlg_sf)
        dlg_ef_gpu.get(out=dlg_ef)
        dll_sf_gpu.get(out=dll_sf)
        dll_ef_gpu.get(out=dll_ef)
        vh_sf_gpu.get(out=vh_sf)
        vh_ef_gpu.get(out=vh_ef)

        times[2] += -time.perf_counter()

        cond_r, cond_l = obc_w_cpu.obc_w_beyn(dxr_sf, dxr_ef, mr_sf, mr_ef, vh_sf, vh_ef, dmr_sf, dmr_ef, dvh_sf,
                                              dvh_ef)

        times[2] += time.perf_counter()

        times[3] += -time.perf_counter()
        if not np.isnan(cond_r) and not np.isnan(cond_l):
            obc_w_cpu.obc_w_dl(dxr_sf, dxr_ef, lg_sf, lg_ef, ll_sf, ll_ef, mr_sf, mr_ef, dlg_sf, dll_sf, dlg_ef, dll_ef)
        times[3] += time.perf_counter()

        times[4] += -time.perf_counter()

        if not np.isnan(cond_r) and not np.isnan(cond_l):
            dmr_ef_gpu.set(dmr_ef)
            dlg_ef_gpu.set(dlg_ef)
            dll_ef_gpu.set(dll_ef)
            dvh_ef_gpu.set(dvh_ef)
            dmr_sf_gpu.set(dmr_sf)
            dlg_sf_gpu.set(dlg_sf)
            dll_sf_gpu.set(dll_sf)
            dvh_sf_gpu.set(dvh_sf)
            matrix_inversion_w.rgf_gpu(bmax_mm, bmin_mm, vh_sparse, mr, lg, ll, factors[i], wg_diag_gpu[i],
                                       wg_upper_gpu[i], wl_diag_gpu[i], wl_upper_gpu[i], wr_diag_gpu[i],
                                       wr_upper_gpu[i], xr_diag_gpu[i], dmr_ef_gpu[0], dlg_ef_gpu[0], dll_ef_gpu[0],
                                       dvh_ef_gpu[0], dmr_sf_gpu[0], dlg_sf_gpu[0], dll_sf_gpu[0], dvh_sf_gpu[0])
        times[4] += time.perf_counter()

    times[5] = -time.perf_counter()
    # unload
    wg_diag_gpu.get(out=wg_diag)
    wg_upper_gpu.get(out=wg_upper)
    wl_diag_gpu.get(out=wl_diag)
    wl_upper_gpu.get(out=wl_upper)
    wr_diag_gpu.get(out=wr_diag)
    wr_upper_gpu.get(out=wr_upper)

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
