import concurrent.futures
import time
from itertools import repeat

import cupy as cp
import cupyx as cpx
class dummy:
    def __init__(self):
        pass
    def set_num_threads(self, n):
        pass

try:
    import mkl
except (ImportError, ModuleNotFoundError):
    mkl = dummy()
import numpy as np
import numpy.typing as npt

from quatrex.block_tri_solvers import (
    # matrix_inversion_w,
    rgf_GF_GPU_combo,
    # rgf_W_GPU,
    rgf_W_GPU_combo,
)
from quatrex.OBC import obc_w_gpu
from quatrex.OBC.beyn_batched import beyn_new_batched_gpu_3 as beyn_gpu
# from quatrex.utils.matrix_creation import (
#     extract_small_matrix_blocks,
#     homogenize_matrix_Rnosym,
# )

from quatrex.GW.screenedinteraction.polarization_preprocess import polarization_preprocess_2d


@cpx.jit.rawkernel()
def _toarray(data, indices, indptr, out, srow, erow, scol, ecol):
    tid = cpx.jit.threadIdx.x
    bid = cpx.jit.blockIdx.x
    row = srow + bid
    if row < erow:
        num_threads = cpx.jit.blockDim.x
        sidx = indptr[row]
        eidx = indptr[row + 1]
        block_size = ecol - scol

        buf = cpx.jit.shared_memory(cp.complex128, 832)
        for i in range(tid, block_size, num_threads):
            buf[i] = 0
        cpx.jit.syncthreads()

        # for i in range(sidx + tid, eidx, num_threads):
        #     j = indices[i]
        #     if j >= scol and j < ecol:
        #         out[bid, j - scol] = data[i]
        # cpx.jit.syncthreads()

        i = sidx + tid
        while i < eidx:
            j = indices[i]
            if j >= scol and j < ecol:
                buf[j - scol] = data[i]
            i += num_threads
        cpx.jit.syncthreads()

        for i in range(tid, block_size, num_threads):
            out[bid, i] = buf[i]


def spgemm(A, B, rows: int = 8192):
    C = None
    for i in range(0, A.shape[0], rows):
        A_block = A[i:min(A.shape[0], i+rows)]
        C_block = A_block @ B
        if C is None:
            C = C_block
        else:
            C = cpx.scipy.sparse.vstack([C, C_block], format="csr")
    return C


def spgemm_direct(A, B, C, rows: int = 8192):
    idx = 0
    for i in range(0, A.shape[0], rows):
        A_block = A[i:min(A.shape[0], i+rows)]
        C_block = A_block @ B
        C[idx : idx + C_block.nnz] = C_block.data
        idx += C_block.nnz


def sp_mm_gpu(pr_rgf, pg_rgf, pl_rgf, rows, columns, vh_dev, mr, lg, ll, nao):
    """Matrix multiplication with sparse matrices """
    # vh_dev = cp.sparse.csr_matrix(vh_cpu)
    vh_ct_dev = vh_dev.T.conj(copy=False)
    pr_dev = cp.asarray(pr_rgf)
    pg_dev = cp.asarray(pg_rgf)
    pl_dev = cp.asarray(pl_rgf)
    #mr[:] = (cp.sparse.identity(nao) - spgemm(vh_dev, cp.sparse.csr_matrix(pr))).data
    mr[:] = (cp.sparse.identity(nao) - spgemm(vh_dev, cp.sparse.csr_matrix((pr_dev, (rows, columns)), shape = (nao, nao)))).data
    #spgemm_direct(spgemm(vh_dev, cp.sparse.csr_matrix(pg)), vh_ct_dev, lg)
    spgemm_direct(spgemm(vh_dev, cp.sparse.csr_matrix((pg_dev, (rows, columns)), shape = (nao, nao))), vh_ct_dev, lg)
    #spgemm_direct(spgemm(vh_dev, cp.sparse.csr_matrix(pl)), vh_ct_dev, ll)
    spgemm_direct(spgemm(vh_dev, cp.sparse.csr_matrix((pl_dev, (rows, columns)), shape = (nao, nao))), vh_ct_dev, ll)


def calc_W_pool_mpi_split(
    # Hamiltonian object.
    hamiltonian_obj,
    # Energy vector.
    energy,
    # Polarization.
    pg,
    pl,
    pr,
    # polarization 2D format
    pg_p2w,
    pl_p2w,
    pr_p2w,
    # Coulomb matrix.
    vh,
    vh_diag, vh_upper, vh_lower,
    # Output Green's functions.
    wg_p2w,
    wl_p2w,
    wr_p2w,
    # Sparse-to-dense Mappings.
    # map_diag_mm,
    # map_upper_mm,
    # map_lower_mm,
    # map_diag_m,
    # map_upper_m,
    # map_lower_m,
    # map_diag_l,
    # map_upper_l,
    # map_lower_l,
    mapping_diag_mm,
    mapping_upper_mm,
    mapping_lower_mm,
    mapping_diag_m,
    mapping_upper_m,
    mapping_lower_m,
    mapping_diag_l,
    mapping_upper_l,
    mapping_lower_l,
    # P indices
    rows_dev,
    columns_dev,
    # P transposition
    ij2ji_dev,
    # M and L indices.
    rows_m,
    columns_m,
    rows_l,
    columns_l,
    # M and L transposition indices.
    ij2ji_m,
    ij2ji_l,
    # Output observables.
    dosw,
    new,
    npw,
    idx_e,
    # Some factor, idk.
    factor,
    # MPI communicator info.
    comm,
    rank,
    size,
    # Number of connected blocks.
    nbc,
    # Options.
    homogenize=True,
    NCpSC: int = 1,
    return_sigma_boundary=False,
    mkl_threads: int = 1,
    worker_num: int = 1,
    compute_mode: int = 0,
    block_inv: bool = False,
    use_dace: bool = False,
    validate_dace: bool = False,
):
    """Memory-optimized version of the p2w_gpu function.

    This is implemented in analogy to the callc_GF_pool_mpi_split_memopt
    function combined with the p2w_pool_mpi_split function.

    """

    mempool = cp.get_default_memory_pool()

    comm.Barrier()
    if rank == 0:
        time_pre_OBC = -time.perf_counter()
        print(f"Used bytes: {mempool.used_bytes()}", flush=True)
        print(f"Total bytes: {mempool.total_bytes()}", flush=True)

    # --- Preprocessing ------------------------------------------------

    input_stream = cp.cuda.stream.Stream(non_blocking=True)

    # Number of energy points.
    ne = energy.shape[0]

    # Number of blocks.
    nb = hamiltonian_obj.Bmin.shape[0]
    # Start and end indices of the blocks.
    bmax = hamiltonian_obj.Bmax - 1
    bmin = hamiltonian_obj.Bmin - 1

    # fix nbc to 2 for the given solution
    # todo calculate it (?)
    # nbc = 2
    lb_vec = bmax - bmin + 1
    lb_start = lb_vec[0]
    lb_end = lb_vec[nb - 1]

    # Block sizes after matrix multiplication.
    bmax_mm = bmax[nbc - 1 : nb : nbc]
    bmin_mm = bmin[0:nb:nbc]
    # Number of blocks after matrix multiplication.
    nb_mm = bmax_mm.size
    # larges block length after matrix multiplication
    lb_max_mm = np.max(bmax_mm - bmin_mm + 1)
    lb_vec_mm = bmax_mm - bmin_mm + 1
    lb_start_mm = lb_vec_mm[0]
    lb_end_mm = lb_vec_mm[nb_mm - 1]

    # mapping_diag_l = rgf_GF_GPU_combo.map_to_mapping(map_diag_l, nb_mm)
    # mapping_upper_l = rgf_GF_GPU_combo.map_to_mapping(map_upper_l, nb_mm - 1)
    # mapping_lower_l = rgf_GF_GPU_combo.map_to_mapping(map_lower_l, nb_mm - 1)

    # mapping_diag_m = rgf_GF_GPU_combo.map_to_mapping(map_diag_m, nb_mm)
    # mapping_upper_m = rgf_GF_GPU_combo.map_to_mapping(map_upper_m, nb_mm - 1)
    # mapping_lower_m = rgf_GF_GPU_combo.map_to_mapping(map_lower_m, nb_mm - 1)

    # mapping_diag_mm = rgf_GF_GPU_combo.map_to_mapping(map_diag_mm, nb_mm)
    # mapping_upper_mm = rgf_GF_GPU_combo.map_to_mapping(map_upper_mm, nb_mm - 1)
    # mapping_lower_mm = rgf_GF_GPU_combo.map_to_mapping(map_lower_mm, nb_mm - 1)

    # vh_diag, vh_upper, vh_lower = rgf_GF_GPU_combo.csr_to_block_tridiagonal_csr(
    #     vh, bmin_mm, bmax_mm + 1
    # )

    obc_w_batchsize = 8

    # Pinned memory buffers for the OBC.
    # dxr_sd = cpx.zeros_pinned((obc_w_batchsize, lb_start_mm, lb_start_mm), dtype=np.complex128)
    # dxr_ed = cpx.zeros_pinned((obc_w_batchsize, lb_end_mm, lb_end_mm), dtype=np.complex128)
    # dvh_sd = cpx.zeros_pinned((obc_w_batchsize, lb_start_mm, lb_start_mm), dtype=np.complex128)
    # dvh_ed = cpx.zeros_pinned((obc_w_batchsize, lb_end_mm, lb_end_mm), dtype=np.complex128)
    # dmr_sd = cpx.zeros_pinned((obc_w_batchsize, lb_start_mm, lb_start_mm), dtype=np.complex128)
    # dmr_ed = cpx.zeros_pinned((obc_w_batchsize, lb_end_mm, lb_end_mm), dtype=np.complex128)
    # dlg_sd = cpx.zeros_pinned((obc_w_batchsize, lb_start_mm, lb_start_mm), dtype=np.complex128)
    # dlg_ed = cpx.zeros_pinned((obc_w_batchsize, lb_end_mm, lb_end_mm), dtype=np.complex128)
    # dll_sd = cpx.zeros_pinned((obc_w_batchsize, lb_start_mm, lb_start_mm), dtype=np.complex128)
    # dll_ed = cpx.zeros_pinned((obc_w_batchsize, lb_end_mm, lb_end_mm), dtype=np.complex128)

    # Dense buffers for the OBC matmul.
    # mr_s0 = cpx.empty_like_pinned(dmr_sd)
    # mr_s1 = cpx.empty_like_pinned(dmr_sd)
    # mr_s2 = cpx.empty_like_pinned(dmr_sd)
    # mr_e0 = cpx.empty_like_pinned(dmr_ed)
    # mr_e1 = cpx.empty_like_pinned(dmr_ed)
    # mr_e2 = cpx.empty_like_pinned(dmr_ed)
    # lg_s0 = cpx.empty_like_pinned(dlg_sd)
    # lg_s1 = cpx.empty_like_pinned(dlg_sd)
    # lg_e0 = cpx.empty_like_pinned(dlg_ed)
    # lg_e1 = cpx.empty_like_pinned(dlg_ed)
    # ll_s0 = cpx.empty_like_pinned(dll_sd)
    # ll_s1 = cpx.empty_like_pinned(dll_sd)
    # ll_e0 = cpx.empty_like_pinned(dll_ed)
    # ll_e1 = cpx.empty_like_pinned(dll_ed)
    # vh_s = cpx.empty_pinned((obc_w_batchsize, lb_start_mm, lb_start_mm), dtype=np.complex128)
    # vh_e = cpx.empty_pinned((obc_w_batchsize, lb_start_mm, lb_start_mm), dtype=np.complex128)
    # mb00 = cpx.empty_pinned(
    #     (
    #         obc_w_batchsize,
    #         2 * nbc * NCpSC + 1,
    #         lb_start_mm // (nbc * NCpSC),
    #         lb_start_mm // (nbc * NCpSC),
    #     ),
    #     dtype=np.complex128,
    # )
    # mbNN = cpx.empty_pinned(
    #     (
    #         obc_w_batchsize,
    #         2 * nbc * NCpSC + 1,
    #         lb_start_mm // (nbc * NCpSC),
    #         lb_start_mm // (nbc * NCpSC),
    #     ),
    #     dtype=np.complex128,
    # )

    comm.Barrier()
    if rank == 0:
        time_pre_OBC += time.perf_counter()
        print("Time for pre-processing OBC: %.3f s" % time_pre_OBC, flush=True)
        time_OBC = -time.perf_counter()

    # --- Boundary conditions ------------------------------------------

    # pl_rgf, pg_rgf, pr_rgf = polarization_preprocess_2d(pl_p2w, pg_p2w, pr_p2w, rows_dev, columns_dev, ij2ji_dev, NCpSC, bmin, bmax, homogenize)
    polarization_preprocess_2d(pl_p2w, pg_p2w, pr_p2w, rows_dev, columns_dev, ij2ji_dev, NCpSC, bmin, bmax, homogenize)
    pl_rgf, pg_rgf, pr_rgf = pl_p2w, pg_p2w, pr_p2w

    nao = vh.shape[0]
    vh_dev = cp.sparse.csr_matrix(vh)
    vh_ct_dev = vh_dev.T.conj(copy=False)
    identity = cp.sparse.identity(nao)

    # slice block start diagonal and slice block start off diagonal
    slb_sd = slice(0, lb_start)
    slb_so = slice(lb_start, 2 * lb_start)
    # slice block end diagonal and slice block end off diagonal
    slb_ed = slice(nao - lb_end, nao)
    slb_eo = slice(nao - 2 * lb_end, nao - lb_end)

    mr_dev = cp.empty((obc_w_batchsize, len(rows_m)), dtype=np.complex128)
    lg_dev = cp.empty((obc_w_batchsize, len(rows_l)), dtype=np.complex128)
    ll_dev = cp.empty((obc_w_batchsize, len(rows_l)), dtype=np.complex128)
    mr_host = mr_dev
    lg_host = lg_dev
    ll_host = ll_dev

    # print(f"Used bytes: {mempool.used_bytes()}", flush=True)
    # print(f"Total bytes: {mempool.total_bytes()}", flush=True)

    # vh_s1 = cp.zeros((ne, lb_start, lb_start), dtype=np.complex128)
    # vh_s2 = cp.zeros((ne, lb_start, lb_start), dtype=np.complex128)
    # pg_s1 = cp.zeros((ne, lb_start, lb_start), dtype=np.complex128)
    # pg_s2 = cp.zeros((ne, lb_start, lb_start), dtype=np.complex128)
    # pl_s1 = cp.zeros((ne, lb_start, lb_start), dtype=np.complex128)
    # pl_s2 = cp.zeros((ne, lb_start, lb_start), dtype=np.complex128)
    # pr_s1 = cp.zeros((ne, lb_start, lb_start), dtype=np.complex128)
    # pr_s2 = cp.zeros((ne, lb_start, lb_start), dtype=np.complex128)
    # pr_s3 = cp.zeros((ne, lb_start, lb_start), dtype=np.complex128)

    # vh_e1 = cp.zeros((ne, lb_end, lb_end), dtype=np.complex128)
    # vh_e2 = cp.zeros((ne, lb_end, lb_end), dtype=np.complex128)
    # pg_e1 = cp.zeros((ne, lb_end, lb_end), dtype=np.complex128)
    # pg_e2 = cp.zeros((ne, lb_end, lb_end), dtype=np.complex128)
    # pl_e1 = cp.zeros((ne, lb_end, lb_end), dtype=np.complex128)
    # pl_e2 = cp.zeros((ne, lb_end, lb_end), dtype=np.complex128)
    # pr_e1 = cp.zeros((ne, lb_end, lb_end), dtype=np.complex128)
    # pr_e2 = cp.zeros((ne, lb_end, lb_end), dtype=np.complex128)
    # pr_e3 = cp.zeros((ne, lb_end, lb_end), dtype=np.complex128)

    vh_s1 = cp.empty((obc_w_batchsize, lb_start, lb_start), dtype=np.complex128)
    vh_s2 = cp.empty((obc_w_batchsize, lb_start, lb_start), dtype=np.complex128)
    pg_s1 = cp.empty((obc_w_batchsize, lb_start, lb_start), dtype=np.complex128)
    pg_s2 = cp.empty((obc_w_batchsize, lb_start, lb_start), dtype=np.complex128)
    pl_s1 = cp.empty((obc_w_batchsize, lb_start, lb_start), dtype=np.complex128)
    pl_s2 = cp.empty((obc_w_batchsize, lb_start, lb_start), dtype=np.complex128)
    pr_s1 = cp.empty((obc_w_batchsize, lb_start, lb_start), dtype=np.complex128)
    pr_s2 = cp.empty((obc_w_batchsize, lb_start, lb_start), dtype=np.complex128)
    pr_s3 = cp.empty((obc_w_batchsize, lb_start, lb_start), dtype=np.complex128)

    vh_e1 = cp.empty((obc_w_batchsize, lb_end, lb_end), dtype=np.complex128)
    vh_e2 = cp.empty((obc_w_batchsize, lb_end, lb_end), dtype=np.complex128)
    pg_e1 = cp.empty((obc_w_batchsize, lb_end, lb_end), dtype=np.complex128)
    pg_e2 = cp.empty((obc_w_batchsize, lb_end, lb_end), dtype=np.complex128)
    pl_e1 = cp.empty((obc_w_batchsize, lb_end, lb_end), dtype=np.complex128)
    pl_e2 = cp.empty((obc_w_batchsize, lb_end, lb_end), dtype=np.complex128)
    pr_e1 = cp.empty((obc_w_batchsize, lb_end, lb_end), dtype=np.complex128)
    pr_e2 = cp.empty((obc_w_batchsize, lb_end, lb_end), dtype=np.complex128)
    pr_e3 = cp.empty((obc_w_batchsize, lb_end, lb_end), dtype=np.complex128)

    pr_dev = None
    pg_dev = None
    pl_dev = None

    condL, condR = [], []

    comm.Barrier()

    start_spgemm = time.perf_counter()
    time_copy_in = 0
    time_spgemm = 0
    time_copy_out = 0
    time_dense = 0
    time_obc_mm = 0
    time_beyn_w = 0
    time_obc_l = 0
    time_w = 0

    for i in range(0, ne, obc_w_batchsize):
        j = min(i + obc_w_batchsize, ne)

        # for ie in range(ne):
        for ie in range(i, j):

            time_copy_in -= time.perf_counter()

            if pr_dev is None:
                pr_dev = cp.sparse.csr_matrix((cp.asarray(pr_rgf[ie]), (rows_dev, columns_dev)), shape = (nao, nao))
                pg_dev = cp.sparse.csr_matrix((cp.asarray(pg_rgf[ie]), (rows_dev, columns_dev)), shape = (nao, nao))
                pl_dev = cp.sparse.csr_matrix((cp.asarray(pl_rgf[ie]), (rows_dev, columns_dev)), shape = (nao, nao))
            else:
                pr_tmp = cp.asarray(pr_rgf[ie])
                pr_dev.data[:] = pr_tmp[ij2ji_dev]
                pg_tmp = cp.asarray(pg_rgf[ie])
                pg_dev.data[:] = pg_tmp[ij2ji_dev]
                pl_tmp = cp.asarray(pl_rgf[ie])
                pl_dev.data[:] = pl_tmp[ij2ji_dev]
            
            time_copy_in += time.perf_counter()

            time_spgemm -= time.perf_counter()

            cp.cuda.Stream.null.synchronize()

            mr_dev[ie-i] = (identity - spgemm(vh_dev, pr_dev)).data
            spgemm_direct(spgemm(vh_dev, pg_dev), vh_ct_dev, lg_dev[ie-i])
            spgemm_direct(spgemm(vh_dev, pl_dev), vh_ct_dev, ll_dev[ie-i])

            time_spgemm += time.perf_counter()

            time_dense -= time.perf_counter()

            num_threads = min(1024, lb_start)
            num_thread_blocks = lb_start
            _toarray[num_thread_blocks, num_threads](vh_dev.data, vh_dev.indices, vh_dev.indptr, vh_s1[ie-i], slb_sd.start, slb_sd.stop, slb_sd.start, slb_sd.stop)
            _toarray[num_thread_blocks, num_threads](vh_dev.data, vh_dev.indices, vh_dev.indptr, vh_s2[ie-i], slb_sd.start, slb_sd.stop, slb_so.start, slb_so.stop)
            _toarray[num_thread_blocks, num_threads](pg_dev.data, pg_dev.indices, pg_dev.indptr, pg_s1[ie-i], slb_sd.start, slb_sd.stop, slb_sd.start, slb_sd.stop)
            _toarray[num_thread_blocks, num_threads](pg_dev.data, pg_dev.indices, pg_dev.indptr, pg_s2[ie-i], slb_sd.start, slb_sd.stop, slb_so.start, slb_so.stop)
            _toarray[num_thread_blocks, num_threads](pl_dev.data, pl_dev.indices, pl_dev.indptr, pl_s1[ie-i], slb_sd.start, slb_sd.stop, slb_sd.start, slb_sd.stop)
            _toarray[num_thread_blocks, num_threads](pl_dev.data, pl_dev.indices, pl_dev.indptr, pl_s2[ie-i], slb_sd.start, slb_sd.stop, slb_so.start, slb_so.stop)
            _toarray[num_thread_blocks, num_threads](pr_dev.data, pr_dev.indices, pr_dev.indptr, pr_s1[ie-i], slb_sd.start, slb_sd.stop, slb_sd.start, slb_sd.stop)
            _toarray[num_thread_blocks, num_threads](pr_dev.data, pr_dev.indices, pr_dev.indptr, pr_s2[ie-i], slb_sd.start, slb_sd.stop, slb_so.start, slb_so.stop)
            _toarray[num_thread_blocks, num_threads](pr_dev.data, pr_dev.indices, pr_dev.indptr, pr_s3[ie-i], slb_so.start, slb_so.stop, slb_sd.start, slb_sd.stop)

            num_threads = min(1024, lb_end)
            num_thread_blocks = lb_end
            _toarray[num_thread_blocks, num_threads](vh_dev.data, vh_dev.indices, vh_dev.indptr, vh_e1[ie-i], slb_ed.start, slb_ed.stop, slb_ed.start, slb_ed.stop)
            _toarray[num_thread_blocks, num_threads](vh_dev.data, vh_dev.indices, vh_dev.indptr, vh_e2[ie-i], slb_ed.start, slb_ed.stop, slb_eo.start, slb_eo.stop)
            _toarray[num_thread_blocks, num_threads](pg_dev.data, pg_dev.indices, pg_dev.indptr, pg_e1[ie-i], slb_ed.start, slb_ed.stop, slb_ed.start, slb_ed.stop)
            _toarray[num_thread_blocks, num_threads](pg_dev.data, pg_dev.indices, pg_dev.indptr, pg_e2[ie-i], slb_ed.start, slb_ed.stop, slb_eo.start, slb_eo.stop)
            _toarray[num_thread_blocks, num_threads](pl_dev.data, pl_dev.indices, pl_dev.indptr, pl_e1[ie-i], slb_ed.start, slb_ed.stop, slb_ed.start, slb_ed.stop)
            _toarray[num_thread_blocks, num_threads](pl_dev.data, pl_dev.indices, pl_dev.indptr, pl_e2[ie-i], slb_ed.start, slb_ed.stop, slb_eo.start, slb_eo.stop)
            _toarray[num_thread_blocks, num_threads](pr_dev.data, pr_dev.indices, pr_dev.indptr, pr_e1[ie-i], slb_ed.start, slb_ed.stop, slb_ed.start, slb_ed.stop)
            _toarray[num_thread_blocks, num_threads](pr_dev.data, pr_dev.indices, pr_dev.indptr, pr_e2[ie-i], slb_eo.start, slb_eo.stop, slb_ed.start, slb_ed.stop)
            _toarray[num_thread_blocks, num_threads](pr_dev.data, pr_dev.indices, pr_dev.indptr, pr_e3[ie-i], slb_ed.start, slb_ed.stop, slb_eo.start, slb_eo.stop)

            vh_e2[ie-i] = vh_e2[ie-i].T.conj()
            pg_e2[ie-i] = -pg_e2[ie-i].T.conj()
            pl_e2[ie-i] = -pl_e2[ie-i].T.conj()

            time_dense += time.perf_counter()
        
        
        # start_obc_mm = time.perf_counter()

        # obc_w_gpu.obc_w_mm_batched_gpu(vh_s1, vh_s2, pg_s1, pg_s2, pl_s1, pl_s2, pr_s1, pr_s2, pr_s3,
        #                                vh_e1, vh_e2, pg_e1, pg_e2, pl_e1, pl_e2, pr_e1, pr_e2, pr_e3,
        #                                dmr_sd, dmr_ed, dlg_sd, dlg_ed, dll_sd, dll_ed,
        #                             #    mr_s, mr_e, lg_s, lg_e, ll_s, ll_e,
        #                                mr_s0, mr_s1, mr_s2, mr_e0, mr_e1, mr_e2, lg_s0, lg_s1, lg_e0, lg_e1, ll_s0, ll_s1, ll_e0, ll_e1,
        #                                vh_s, vh_e, mb00, mbNN, nbc, NCpSC)
            
        time_obc_mm -= time.perf_counter()

        # obc_w_gpu.obc_w_mm_batched_gpu(vh_s1[:j-i], vh_s2[:j-i], pg_s1[:j-i], pg_s2[:j-i], pl_s1[:j-i], pl_s2[:j-i], pr_s1[:j-i], pr_s2[:j-i], pr_s3[:j-i],
        #                             vh_e1[:j-i], vh_e2[:j-i], pg_e1[:j-i], pg_e2[:j-i], pl_e1[:j-i], pl_e2[:j-i], pr_e1[:j-i], pr_e2[:j-i], pr_e3[:j-i],
        #                             dmr_sd[:j-i], dmr_ed[:j-i], dlg_sd[:j-i], dlg_ed[:j-i], dll_sd[:j-i], dll_ed[:j-i],
        #                             mr_s0[:j-i], mr_s1[:j-i], mr_s2[:j-i], mr_e0[:j-i], mr_e1[:j-i], mr_e2[:j-i],
        #                             lg_s0[:j-i], lg_s1[:j-i], lg_e0[:j-i], lg_e1[:j-i], ll_s0[:j-i], ll_s1[:j-i], ll_e0[:j-i], ll_e1[:j-i],
        #                             vh_s[:j-i], vh_e[:j-i], mb00[:j-i], mbNN[:j-i], nbc, NCpSC)
        (
            dmr_sd, dmr_ed, dlg_sd, dlg_ed, dll_sd, dll_ed,
            mr_s0, mr_s1, mr_s2, mr_e0, mr_e1, mr_e2,
            lg_s0, lg_s1, lg_e0, lg_e1, ll_s0, ll_s1, ll_e0, ll_e1,
            vh_s, vh_e, mb00, mbNN
        ) = obc_w_gpu.obc_w_mm_batched_gpu(vh_s1[:j-i], vh_s2[:j-i], pg_s1[:j-i], pg_s2[:j-i], pl_s1[:j-i], pl_s2[:j-i], pr_s1[:j-i], pr_s2[:j-i], pr_s3[:j-i],
                                    vh_e1[:j-i], vh_e2[:j-i], pg_e1[:j-i], pg_e2[:j-i], pl_e1[:j-i], pl_e2[:j-i], pr_e1[:j-i], pr_e2[:j-i], pr_e3[:j-i],
                                    nbc, NCpSC)
        
        time_obc_mm += time.perf_counter()

        time_beyn_w -= time.perf_counter()

        imag_lim = 1e-4
        R = 1e4
        # matrix_blocks_left = cp.asarray(mb00[:j-i])
        # M00_left = cp.asarray(mr_s0[:j-i])
        # M01_left = cp.asarray(mr_s1[:j-i])
        # M10_left = cp.asarray(mr_s2[:j-i])
        matrix_blocks_left = mb00
        M00_left = mr_s0
        M01_left = mr_s1
        M10_left = mr_s2
        dmr, dxr_sd_gpu, condl, _ = beyn_gpu(nbc * NCpSC, matrix_blocks_left, M00_left, M01_left, M10_left, imag_lim, R, 'L')
        # dxr_sd_gpu.get(out=dxr_sd[:j-i])
        dxr_sd = dxr_sd_gpu
        # dmr_sd[:j-i] -= dmr.get()
        dmr_sd -= dmr
        # (M10_left @ dxr_sd_gpu @ cp.asarray(vh_s[:j-i])).get(out=dvh_sd[:j-i])
        dvh_sd = M10_left @ dxr_sd_gpu @ vh_s
        # matrix_blocks_right = cp.asarray(mbNN[:j-i])
        # M00_right = cp.asarray(mr_e0[:j-i])
        # M01_right = cp.asarray(mr_e1[:j-i])
        # M10_right = cp.asarray(mr_e2[:j-i])
        matrix_blocks_right = mbNN
        M00_right = mr_e0
        M01_right = mr_e1
        M10_right = mr_e2
        dmr, dxr_ed_gpu, condr, _ = beyn_gpu(nbc * NCpSC, matrix_blocks_right, M00_right, M01_right, M10_right, imag_lim, R, 'R')
        # dxr_ed_gpu.get(out=dxr_ed[:j-i])
        dxr_ed = dxr_ed_gpu
        # dmr_ed[:j-i] -= dmr.get()
        dmr_ed -= dmr
        # (M01_right @ dxr_ed_gpu @ cp.asarray(vh_e[:j-i])).get(out=dvh_ed[:j-i])
        dvh_ed = M01_right @ dxr_ed_gpu @ vh_e

        condL.extend(condl)
        condR.extend(condr)

        time_beyn_w += time.perf_counter()

        time_obc_l -= time.perf_counter()

        with concurrent.futures.ThreadPoolExecutor(max_workers=obc_w_batchsize) as executor:
            executor.map(obc_w_gpu.obc_w_L_lg_2,
                dlg_sd[:j-i],
                dlg_ed[:j-i],
                dll_sd[:j-i],
                dll_ed[:j-i],
                mr_s0[:j-i], mr_s1[:j-i], mr_s2[:j-i],
                mr_e0[:j-i], mr_e1[:j-i], mr_e2[:j-i],
                lg_s0[:j-i], lg_s1[:j-i],
                lg_e0[:j-i], lg_e1[:j-i],
                ll_s0[:j-i], ll_s1[:j-i],
                ll_e0[:j-i], ll_e1[:j-i],
                dxr_sd[:j-i],
                dxr_ed[:j-i],
            )

        mr_s0, mr_s1, mr_s2 = None, None, None
        mr_e0, mr_e1, mr_e2 = None, None, None
        lg_s0, lg_s1 = None, None
        lg_e0, lg_e1 = None, None
        ll_s0, ll_s1 = None, None
        ll_e0, ll_e1 = None, None
        dxr_sd, dxr_ed = None, None
        mb00, mbNN = None, None
        matrix_blocks_left, matrix_blocks_right = None, None
        M00_left, M01_left, M10_left = None, None, None
        M00_right, M01_right, M10_right = None, None, None
        
        time_obc_l += time.perf_counter()

        time_w -= time.perf_counter()

        rgf_W_GPU_combo.rgf_batched_GPU(
            energies=energy[i:j],
            map_diag_mm=mapping_diag_mm,
            map_upper_mm=mapping_upper_mm,
            map_lower_mm=mapping_lower_mm,
            map_diag_m=mapping_diag_m,
            map_upper_m=mapping_upper_m,
            map_lower_m=mapping_lower_m,
            map_diag_l=mapping_diag_l,
            map_upper_l=mapping_upper_l,
            map_lower_l=mapping_lower_l,
            vh_diag_host=vh_diag,
            vh_upper_host=vh_upper,
            vh_lower_host=vh_lower,
            mr_host=mr_host[:j-i],
            ll_host=ll_host[:j-i],
            lg_host=lg_host[:j-i],
            dvh_left_host=dvh_sd[:j-i],
            dvh_right_host=dvh_ed[:j-i],
            dmr_left_host=dmr_sd[:j-i],
            dmr_right_host=dmr_ed[:j-i],
            dlg_left_host=dlg_sd[:j-i],
            dlg_right_host=dlg_ed[:j-i],
            dll_left_host=dll_sd[:j-i],
            dll_right_host=dll_ed[:j-i],
            wr_host=wr_p2w[i:j],
            wl_host=wl_p2w[i:j],
            wg_host=wg_p2w[i:j],
            dosw=dosw[i:j],
            nEw=new[i:j],
            nPw=npw[i:j],
            bmax=bmax_mm,
            bmin=bmin_mm,
            solve=False,
            input_stream=input_stream,
        )

        time_w += time.perf_counter()

        # print(f"Used bytes: {mempool.used_bytes()}", flush=True)
        # print(f"Total bytes: {mempool.total_bytes()}", flush=True)
    

    vh_s1 = None
    vh_s2 = None
    pg_s1 = None
    pg_s2 = None
    pl_s1 = None
    pl_s2 = None
    pr_s1 = None
    pr_s2 = None
    pr_s3 = None

    vh_e1 = None
    vh_e2 = None
    pg_e1 = None
    pg_e2 = None
    pl_e1 = None
    pl_e2 = None
    pr_e1 = None
    pr_e2 = None
    pr_e3 = None

    # comm.Barrier()
    # finish_obc_mm = time.perf_counter()
    # if rank == 0:
    #     print("    Time for obc mm: %.3f s" % (finish_obc_mm - start_obc_mm), flush=True)

    comm.Barrier()
    finish_spgemm = time.perf_counter()
    if rank == 0:
        print("Screened Interaction region", flush=True)
        print("    Time for overall W loop: %.3f s" % (finish_spgemm - start_spgemm), flush=True)
        print("        Time for CopyToDevice: %.3f s" % time_copy_in, flush=True)
        print("        Time for SpGEMM: %.3f s" % time_spgemm, flush=True)
        print("        Time for CopyToHost: %.3f s" % time_copy_out, flush=True)
        print("        Time for Densification: %.3f s" % time_dense, flush=True)
        print("        Time for OBC MM: %.3f s" % time_obc_mm, flush=True)
        print("        Time for Beyn W: %.3f s" % time_beyn_w, flush=True)
        print("        Time for OBC L: %.3f s" % time_obc_l, flush=True)
        print("        Time for RGF W: %.3f s" % time_w, flush=True)

        print(f"Used bytes: {mempool.used_bytes()}", flush=True)
        print(f"Total bytes: {mempool.total_bytes()}", flush=True)

    # comm.Barrier()
    start_beyn_w = time.perf_counter()

    # imag_lim = 1e-4
    # R = 1e4
    # matrix_blocks_left = cp.asarray(mb00)
    # M00_left = cp.asarray(mr_s0)
    # M01_left = cp.asarray(mr_s1)
    # M10_left = cp.asarray(mr_s2)
    # dmr, dxr_sd_gpu, condL, _ = beyn_gpu(nbc * NCpSC, matrix_blocks_left, M00_left, M01_left, M10_left, imag_lim, R, 'L')
    # dxr_sd_gpu.get(out=dxr_sd)
    # dmr_sd -= dmr.get()
    # (M10_left @ dxr_sd_gpu @ cp.asarray(vh_s)).get(out=dvh_sd)
    # matrix_blocks_right = cp.asarray(mbNN)
    # M00_right = cp.asarray(mr_e0)
    # M01_right = cp.asarray(mr_e1)
    # M10_right = cp.asarray(mr_e2)
    # dmr, dxr_ed_gpu, condR, _ = beyn_gpu(NCpSC, matrix_blocks_right, M00_right, M01_right, M10_right, imag_lim, R, 'R')
    # dxr_ed_gpu.get(out=dxr_ed)
    # dmr_ed -= dmr.get()
    # (M01_right @ dxr_ed_gpu @ cp.asarray(vh_e)).get(out=dvh_ed)

    comm.Barrier()
    finish_beyn_w = time.perf_counter()
    if rank == 0:
        print("    Time for beyn w: %.3f s" % (finish_beyn_w - start_beyn_w), flush=True)
        
    comm.Barrier()
    start_obc_l = time.perf_counter()

    # with concurrent.futures.ThreadPoolExecutor(max_workers=worker_num) as executor:
    #     executor.map(obc_w_gpu.obc_w_L_lg_2,
    #         dlg_sd,
    #         dlg_ed,
    #         dll_sd,
    #         dll_ed,
    #         mr_s0, mr_s1, mr_s2,
    #         mr_e0, mr_e1, mr_e2,
    #         lg_s0, lg_s1,
    #         lg_e0, lg_e1,
    #         ll_s0, ll_s1,
    #         ll_e0, ll_e1,
    #         dxr_sd,
    #         dxr_ed,
    #     )
    
    comm.Barrier()
    finish_obc_l = time.perf_counter()
    if rank == 0:
        print("    Time for obc l: %.3f s" % (finish_obc_l - start_obc_l), flush=True)

    # l_defect = np.count_nonzero(np.isnan(condL))
    # r_defect = np.count_nonzero(np.isnan(condR))

    # if l_defect > 0 or r_defect > 0:
    #     print(
    #         "Warning: %d left and %d right boundary conditions are not satisfied."
    #         % (l_defect, r_defect)
    #     )

    comm.Barrier()
    if rank == 0:
        time_OBC += time.perf_counter()
        print("Time for OBC: %.3f s" % time_OBC, flush=True)
        time_spmm = -time.perf_counter()

    comm.Barrier()
    if rank == 0:
        time_spmm += time.perf_counter()
        print("Time for spgemm: %.3f s" % time_spmm, flush=True)
        time_GF_trafo = -time.perf_counter()

    # --- Transform L to the (NE, NNZ) format --------------------------

    # mapping_diag_l = rgf_GF_GPU_combo.map_to_mapping(map_diag_l, nb_mm)
    # mapping_upper_l = rgf_GF_GPU_combo.map_to_mapping(map_upper_l, nb_mm - 1)
    # mapping_lower_l = rgf_GF_GPU_combo.map_to_mapping(map_lower_l, nb_mm - 1)

    # mapping_diag_m = rgf_GF_GPU_combo.map_to_mapping(map_diag_m, nb_mm)
    # mapping_upper_m = rgf_GF_GPU_combo.map_to_mapping(map_upper_m, nb_mm - 1)
    # mapping_lower_m = rgf_GF_GPU_combo.map_to_mapping(map_lower_m, nb_mm - 1)

    # mapping_diag_mm = rgf_GF_GPU_combo.map_to_mapping(map_diag_mm, nb_mm)
    # mapping_upper_mm = rgf_GF_GPU_combo.map_to_mapping(map_upper_mm, nb_mm - 1)
    # mapping_lower_mm = rgf_GF_GPU_combo.map_to_mapping(map_lower_mm, nb_mm - 1)

    # vh_diag, vh_upper, vh_lower = rgf_GF_GPU_combo.csr_to_block_tridiagonal_csr(
    #     vh, bmin_mm, bmax_mm + 1
    # )

    comm.Barrier()
    if rank == 0:
        time_GF_trafo += time.perf_counter()
        print("Time for W transformation: %.3f s" % time_GF_trafo, flush=True)
        # time_GF = -time.perf_counter()
        time_GF = -time.perf_counter()

    # --- Compute screened interaction ---------------------------------

    # input_stream = cp.cuda.stream.Stream(non_blocking=True)

    # energy_batchsize = 16
    # energy_batch = np.arange(0, ne, energy_batchsize)

    # for ie in energy_batch:
    #     rgf_W_GPU_combo.rgf_batched_GPU(
    #         energies=energy[ie : ie + energy_batchsize],
    #         map_diag_mm=mapping_diag_mm,
    #         map_upper_mm=mapping_upper_mm,
    #         map_lower_mm=mapping_lower_mm,
    #         map_diag_m=mapping_diag_m,
    #         map_upper_m=mapping_upper_m,
    #         map_lower_m=mapping_lower_m,
    #         map_diag_l=mapping_diag_l,
    #         map_upper_l=mapping_upper_l,
    #         map_lower_l=mapping_lower_l,
    #         vh_diag_host=vh_diag,
    #         vh_upper_host=vh_upper,
    #         vh_lower_host=vh_lower,
    #         mr_host=mr_host[ie : ie + energy_batchsize, :],
    #         ll_host=ll_host[ie : ie + energy_batchsize, :],
    #         lg_host=lg_host[ie : ie + energy_batchsize, :],
    #         dvh_left_host=dvh_sd[ie : ie + energy_batchsize, :, :],
    #         dvh_right_host=dvh_ed[ie : ie + energy_batchsize, :, :],
    #         dmr_left_host=dmr_sd[ie : ie + energy_batchsize, :, :],
    #         dmr_right_host=dmr_ed[ie : ie + energy_batchsize, :, :],
    #         dlg_left_host=dlg_sd[ie : ie + energy_batchsize, :, :],
    #         dlg_right_host=dlg_ed[ie : ie + energy_batchsize, :, :],
    #         dll_left_host=dll_sd[ie : ie + energy_batchsize, :, :],
    #         dll_right_host=dll_ed[ie : ie + energy_batchsize, :, :],
    #         wr_host=wr_p2w[ie : ie + energy_batchsize, :],
    #         wl_host=wl_p2w[ie : ie + energy_batchsize, :],
    #         wg_host=wg_p2w[ie : ie + energy_batchsize, :],
    #         dosw=dosw[ie : ie + energy_batchsize, :],
    #         nEw=new[ie : ie + energy_batchsize, :],
    #         nPw=npw[ie : ie + energy_batchsize, :],
    #         bmax=bmax_mm,
    #         bmin=bmin_mm,
    #         solve=True,
    #         input_stream=input_stream,
    #     )

    comm.Barrier()
    if rank == 0:
        time_GF += time.perf_counter()
        print("Time for W: %.3f s" % time_GF, flush=True)
        time_post_proc = -time.perf_counter()

    # --- Post-processing ----------------------------------------------
    
    l_defect = np.count_nonzero(np.isnan(condL))
    r_defect = np.count_nonzero(np.isnan(condR))

    if l_defect > 0 or r_defect > 0:
        print(
            "Warning: %d left and %d right boundary conditions are not satisfied."
            % (l_defect, r_defect)
        )

    # Calculate F1, F2, which are the relative errors of GR-GA = GG-GL
    F1 = np.max(np.abs(dosw - (new + npw)) / (np.abs(dosw) + 1e-6), axis=1)
    F2 = np.max(np.abs(dosw - (new + npw)) / (np.abs(new + npw) + 1e-6), axis=1)

    buf_recv_r = np.empty((dosw.shape[1]), dtype=np.complex128)
    # buf_send_r = np.empty((dosw.shape[1]), dtype=np.complex128)
    buf_recv_l = np.empty((dosw.shape[1]), dtype=np.complex128)
    # buf_send_l = np.empty((dosw.shape[1]), dtype=np.complex128)
    if size > 1:
        if rank == 0:
            # buf_send_r[:] = dosw[ne - 1, :]
            buf_send_r = dosw[ne - 1, :]
            comm.Sendrecv(
                sendbuf=buf_send_r, dest=rank + 1, recvbuf=buf_recv_r, source=rank + 1
            )

        elif rank == size - 1:
            # buf_send_l[:] = dosw[0, :]
            buf_send_l = dosw[0, :]
            comm.Sendrecv(
                sendbuf=buf_send_l, dest=rank - 1, recvbuf=buf_recv_l, source=rank - 1
            )
        else:
            # buf_send_r[:] = dosw[ne - 1, :]
            # buf_send_l[:] = dosw[0, :]
            buf_send_r = dosw[ne - 1, :]
            buf_send_l = dosw[0, :]
            comm.Sendrecv(
                sendbuf=buf_send_r, dest=rank + 1, recvbuf=buf_recv_r, source=rank + 1
            )
            comm.Sendrecv(
                sendbuf=buf_send_l, dest=rank - 1, recvbuf=buf_recv_l, source=rank - 1
            )

    # Remove individual peaks (To-Do: improve this part by sending boundary elements to the next process)
    if size == 1:
        dDOSm = np.concatenate(
            (
                [0],
                np.max(np.abs(dosw[1 : ne - 1, :] / (dosw[0 : ne - 2, :] + 1)), axis=1),
                [np.max(np.abs(dosw[ne - 1, :] / (dosw[ne - 2, :] + 1)))],
            )
        )
        dDOSp = np.concatenate(
            (
                [np.max(np.abs(dosw[0, :] / (dosw[1, :] + 1)))],
                np.max(np.abs(dosw[1 : ne - 1, :] / (dosw[2:ne, :] + 1)), axis=1),
                [0],
            )
        )
    elif rank == 0:
        dDOSm = np.concatenate(
            (
                [0],
                np.max(np.abs(dosw[1 : ne - 1, :] / (dosw[0 : ne - 2, :] + 1)), axis=1),
                [np.max(np.abs(dosw[ne - 1, :] / (dosw[ne - 2, :] + 1)))],
            )
        )
        dDOSp = np.concatenate(
            (
                [np.max(np.abs(dosw[0, :] / (dosw[1, :] + 1)))],
                np.max(np.abs(dosw[1 : ne - 1, :] / (dosw[2:ne, :] + 1)), axis=1),
                [np.max(np.abs(dosw[ne - 1, :] / (buf_recv_r + 1)))],
            )
        )
    elif rank == size - 1:
        dDOSm = np.concatenate(
            (
                [np.max(np.abs(dosw[0, :] / (buf_recv_l + 1)))],
                np.max(np.abs(dosw[1 : ne - 1, :] / (dosw[0 : ne - 2, :] + 1)), axis=1),
                [np.max(np.abs(dosw[ne - 1, :] / (dosw[ne - 2, :] + 1)))],
            )
        )
        dDOSp = np.concatenate(
            (
                [np.max(np.abs(dosw[0, :] / (dosw[1, :] + 1)))],
                np.max(np.abs(dosw[1 : ne - 1, :] / (dosw[2:ne, :] + 1)), axis=1),
                [0],
            )
        )
    else:
        dDOSm = np.concatenate(
            (
                [np.max(np.abs(dosw[0, :] / (buf_recv_l + 1)))],
                np.max(np.abs(dosw[1 : ne - 1, :] / (dosw[0 : ne - 2, :] + 1)), axis=1),
                [np.max(np.abs(dosw[ne - 1, :] / (dosw[ne - 2, :] + 1)))],
            )
        )
        dDOSp = np.concatenate(
            (
                [np.max(np.abs(dosw[0, :] / (dosw[1, :] + 1)))],
                np.max(np.abs(dosw[1 : ne - 1, :] / (dosw[2:ne, :] + 1)), axis=1),
                [np.max(np.abs(dosw[ne - 1, :] / (buf_recv_r + 1)))],
            )
        )

    # Find indices of elements satisfying the conditions
    ind_zeros = np.where((F1 > 0.1) | (F2 > 0.1) | ((dDOSm > 5) & (dDOSp > 5)) | (np.isnan(condL) | np.isnan(condR)))[0]


    if idx_e[0] == 0:
        ind_zeros = np.concatenate(([0], ind_zeros))


    # Remove the identified peaks and errors
    for index in ind_zeros:
        wr_p2w[index, :] = 0
        wl_p2w[index, :] = 0
        wg_p2w[index, :] = 0


    comm.Barrier()
    if rank == 0:
        time_post_proc += time.perf_counter()
        print("Time for post-processing: %.3f s" % time_post_proc, flush=True)


if __name__ == "__main__":

    A_dev = cp.sparse.random(1000, 1000, density=0.01, dtype=np.float64, format="csr")
    A_dev = A_dev + 1j * A_dev
    print(f"A_dev.nnz = {A_dev.nnz}, A_dev.shape = {A_dev.shape}, format = {A_dev.format}", flush=True)

    bsize = 100
    num_threads = min(1024, bsize)
    num_thread_blocks = bsize
    for row in range(0, A_dev.shape[0], bsize):
        for col in range(0, A_dev.shape[1], bsize):
            ref = A_dev[row : row + bsize, col : col + bsize].toarray()
            val = cp.zeros((bsize, bsize), dtype=np.complex128)
            _toarray[num_thread_blocks, num_threads](A_dev.data, A_dev.indices, A_dev.indptr, val, row, row+bsize, col, col+bsize)
            print(cp.linalg.norm(val - ref) / cp.linalg.norm(ref), flush=True)
            assert cp.allclose(val, ref)
