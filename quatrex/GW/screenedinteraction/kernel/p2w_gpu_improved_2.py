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
    matrix_inversion_w,
    rgf_GF_GPU_combo,
    rgf_W_GPU,
    rgf_W_GPU_combo,
)
from quatrex.OBC import obc_w_cpu, obc_w_gpu
from quatrex.OBC.beyn_batched import beyn_new_batched_gpu_3 as beyn_gpu
from quatrex.utils.matrix_creation import (
    extract_small_matrix_blocks,
    homogenize_matrix_Rnosym,
)

from quatrex.GW.screenedinteraction.polarization_preprocess import polarization_preprocess_2d


@cpx.jit.rawkernel()
def _toarray(data, indices, indptr, out, srow, erow, scol, ecol):
    tid = int(cpx.jit.threadIdx.x)
    bid = int(cpx.jit.blockIdx.x)
    row = srow + bid
    if row < erow:
        num_threads = int(cpx.jit.blockDim.x)
        sidx = indptr[row]
        eidx = indptr[row + 1]
        for i in range(sidx + tid, eidx, num_threads):
            j = indices[i]
            if j >= scol and j < ecol:
                out[bid, j - scol] = data[i]


def spgemm(A, B, rows: int = 5408):
    C = None
    for i in range(0, A.shape[0], rows):
        A_block = A[i:min(A.shape[0], i+rows)]
        C_block = A_block @ B
        if C is None:
            C = C_block
        else:
            C = cpx.scipy.sparse.vstack([C, C_block], format="csr")
    return C


def spgemm_direct(A, B, C, rows: int = 5408):
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
    # Output Green's functions.
    wg_p2w,
    wl_p2w,
    wr_p2w,
    # Sparse-to-dense Mappings.
    map_diag_mm,
    map_upper_mm,
    map_lower_mm,
    map_diag_m,
    map_upper_m,
    map_lower_m,
    map_diag_l,
    map_upper_l,
    map_lower_l,
    # P indices
    rows,
    columns,
    # P transposition
    ij2ji,
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
    start_index_W = 0,
    NCpSC: int = 1,
    return_sigma_boundary=False,
    mkl_threads: int = 1,
    worker_num: int = 1,
    compute_mode: int = 0,
    block_inv: bool = False,
    use_dace: bool = False,
    validate_dace: bool = False,
    iter : int = 117
):
    """Memory-optimized version of the p2w_gpu function.

    This is implemented in analogy to the callc_GF_pool_mpi_split_memopt
    function combined with the p2w_pool_mpi_split function.

    """

    comm.Barrier()
    if rank == 0:
        time_pre_OBC = -time.perf_counter()

    # --- Preprocessing ------------------------------------------------

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

    # Pinned memory buffers for the OBC.
    dxr_sd = cpx.zeros_pinned((ne, lb_start_mm, lb_start_mm), dtype=np.complex128)
    dxr_ed = cpx.zeros_pinned((ne, lb_end_mm, lb_end_mm), dtype=np.complex128)
    dvh_sd = cpx.zeros_pinned((ne, lb_start_mm, lb_start_mm), dtype=np.complex128)
    dvh_ed = cpx.zeros_pinned((ne, lb_end_mm, lb_end_mm), dtype=np.complex128)
    dmr_sd = cpx.zeros_pinned((ne, lb_start_mm, lb_start_mm), dtype=np.complex128)
    dmr_ed = cpx.zeros_pinned((ne, lb_end_mm, lb_end_mm), dtype=np.complex128)
    dlg_sd = cpx.zeros_pinned((ne, lb_start_mm, lb_start_mm), dtype=np.complex128)
    dlg_ed = cpx.zeros_pinned((ne, lb_end_mm, lb_end_mm), dtype=np.complex128)
    dll_sd = cpx.zeros_pinned((ne, lb_start_mm, lb_start_mm), dtype=np.complex128)
    dll_ed = cpx.zeros_pinned((ne, lb_end_mm, lb_end_mm), dtype=np.complex128)
    condl = np.zeros((ne), dtype=np.float64)
    condr = np.zeros((ne), dtype=np.float64)

    # Dense buffers for the OBC matmul.
    # mr_s = np.ndarray((ne,), dtype=object)
    # mr_e = np.ndarray((ne,), dtype=object)
    # lg_s = np.ndarray((ne,), dtype=object)
    # lg_e = np.ndarray((ne,), dtype=object)
    # ll_s = np.ndarray((ne,), dtype=object)
    # ll_e = np.ndarray((ne,), dtype=object)
    mr_s0 = cpx.empty_like_pinned(dmr_sd)
    mr_s1 = cpx.empty_like_pinned(dmr_sd)
    mr_s2 = cpx.empty_like_pinned(dmr_sd)
    mr_e0 = cpx.empty_like_pinned(dmr_ed)
    mr_e1 = cpx.empty_like_pinned(dmr_ed)
    mr_e2 = cpx.empty_like_pinned(dmr_ed)
    lg_s0 = cpx.empty_like_pinned(dlg_sd)
    lg_s1 = cpx.empty_like_pinned(dlg_sd)
    lg_e0 = cpx.empty_like_pinned(dlg_ed)
    lg_e1 = cpx.empty_like_pinned(dlg_ed)
    ll_s0 = cpx.empty_like_pinned(dll_sd)
    ll_s1 = cpx.empty_like_pinned(dll_sd)
    ll_e0 = cpx.empty_like_pinned(dll_ed)
    ll_e1 = cpx.empty_like_pinned(dll_ed)
    vh_s = np.ndarray((ne, lb_start_mm, lb_start_mm), dtype=np.complex128)
    vh_e = np.ndarray((ne, lb_start_mm, lb_start_mm), dtype=np.complex128)
    mb00 = np.ndarray(
        (
            ne,
            2 * nbc * NCpSC + 1,
            lb_start_mm // (nbc * NCpSC),
            lb_start_mm // (nbc * NCpSC),
        ),
        dtype=np.complex128,
    )
    mbNN = np.ndarray(
        (
            ne,
            2 * nbc * NCpSC + 1,
            lb_start_mm // (nbc * NCpSC),
            lb_start_mm // (nbc * NCpSC),
        ),
        dtype=np.complex128,
    )

    # dvh_sd_ref = cpx.zeros_pinned((ne, lb_start_mm, lb_start_mm), dtype=np.complex128)
    # dvh_ed_ref = cpx.zeros_pinned((ne, lb_end_mm, lb_end_mm), dtype=np.complex128)
    # dmr_sd_ref = cpx.zeros_pinned((ne, lb_start_mm, lb_start_mm), dtype=np.complex128)
    # dmr_ed_ref = cpx.zeros_pinned((ne, lb_end_mm, lb_end_mm), dtype=np.complex128)
    # dlg_sd_ref = cpx.zeros_pinned((ne, lb_start_mm, lb_start_mm), dtype=np.complex128)
    # dlg_ed_ref = cpx.zeros_pinned((ne, lb_end_mm, lb_end_mm), dtype=np.complex128)
    # dll_sd_ref = cpx.zeros_pinned((ne, lb_start_mm, lb_start_mm), dtype=np.complex128)
    # dll_ed_ref = cpx.zeros_pinned((ne, lb_end_mm, lb_end_mm), dtype=np.complex128)

    # # Dense buffers for the OBC matmul.
    # mr_s_ref = np.ndarray((ne,), dtype=object)
    # mr_e_ref = np.ndarray((ne,), dtype=object)
    # lg_s_ref = np.ndarray((ne,), dtype=object)
    # lg_e_ref = np.ndarray((ne,), dtype=object)
    # ll_s_ref = np.ndarray((ne,), dtype=object)
    # ll_e_ref = np.ndarray((ne,), dtype=object)
    # vh_s_ref = np.ndarray((ne, lb_start_mm, lb_start_mm), dtype=np.complex128)
    # vh_e_ref = np.ndarray((ne, lb_start_mm, lb_start_mm), dtype=np.complex128)
    # mb00_ref = np.ndarray(
    #     (
    #         ne,
    #         2 * nbc * NCpSC + 1,
    #         lb_start_mm // (nbc * NCpSC),
    #         lb_start_mm // (nbc * NCpSC),
    #     ),
    #     dtype=np.complex128,
    # )
    # mbNN_ref = np.ndarray(
    #     (
    #         ne,
    #         2 * nbc * NCpSC + 1,
    #         lb_start_mm // (nbc * NCpSC),
    #         lb_start_mm // (nbc * NCpSC),
    #     ),
    #     dtype=np.complex128,
    # )

    # for ie in range(ne):
    #     # Anti-Hermitian symmetrizing of PL and PG
    #     # pl[ie] = 1j * np.imag(pl[ie])
    #     pl[ie] = (pl[ie] - pl[ie].conj().T) / 2

    #     # pg[ie] = 1j * np.imag(pg[ie])
    #     pg[ie] = (pg[ie] - pg[ie].conj().T) / 2

    #     # PR has to be derived from PL and PG and then has to be symmetrized
    #     pr[ie] = (pg[ie] - pl[ie]) / 2
    #     # pr[ie] = (pr[ie] + pr[ie].T) / 2
    #     if homogenize:
    #         (PR00, PR01, PR10, _) = extract_small_matrix_blocks(
    #             pr[ie][bmin[0] : bmax[0] + 1, bmin[0] : bmax[0] + 1],
    #             pr[ie][bmin[0] : bmax[0] + 1, bmin[1] : bmax[1] + 1],
    #             pr[ie][bmin[1] : bmax[1] + 1, bmin[0] : bmax[0] + 1],
    #             NCpSC,
    #             "L",
    #         )
    #         pr[ie] = homogenize_matrix_Rnosym(PR00, PR01, PR10, len(bmax))
    #         (PL00, PL01, PL10, _) = extract_small_matrix_blocks(
    #             pl[ie][bmin[0] : bmax[0] + 1, bmin[0] : bmax[0] + 1],
    #             pl[ie][bmin[0] : bmax[0] + 1, bmin[1] : bmax[1] + 1],
    #             pl[ie][bmin[1] : bmax[1] + 1, bmin[0] : bmax[0] + 1],
    #             NCpSC,
    #             "L",
    #         )
    #         pl[ie] = homogenize_matrix_Rnosym(PL00, PL01, PL10, len(bmax))
    #         (PG00, PG01, PG10, _) = extract_small_matrix_blocks(
    #             pg[ie][bmin[0] : bmax[0] + 1, bmin[0] : bmax[0] + 1],
    #             pg[ie][bmin[0] : bmax[0] + 1, bmin[1] : bmax[1] + 1],
    #             pg[ie][bmin[1] : bmax[1] + 1, bmin[0] : bmax[0] + 1],
    #             NCpSC,
    #             "L",
    #         )
    #         pg[ie] = homogenize_matrix_Rnosym(PG00, PG01, PG10, len(bmax))

    # Create a process pool with num_worker workers
    comm.Barrier()
    if rank == 0:
        time_pre_OBC += time.perf_counter()
        print("Time for pre-processing OBC: %.3f s" % time_pre_OBC, flush=True)
        time_OBC = -time.perf_counter()

    # --- Boundary conditions ------------------------------------------

    ref_flag = False
    # with concurrent.futures.ThreadPoolExecutor(max_workers=worker_num) as executor:
    #     #results = executor.map(obc_w_cpu.obc_w_cpu, repeat(vh),
    #     executor.map(obc_w_cpu.obc_w_cpu, repeat(vh),
    #                            pg, pl, pr,
    #                            repeat(bmax), repeat(bmin),
    #                            dvh_sd, dvh_ed,
    #                            dmr_sd, dmr_ed,
    #                            dlg_sd, dlg_ed,
    #                            dll_sd, dll_ed,
    #                            repeat(nbc),
    #                            repeat(NCpSC),
    #                            repeat(block_inv),
    #                            repeat(use_dace),
    #                            repeat(validate_dace),repeat(ref_flag)
    #                             )
    #     # for idx, res in enumerate(results):
    #     #     condl[idx] = res[0]
    #     #     condr[idx] = res[1]

    # for ie in range(ne):
    #     mr_s[ie] = tuple(
    #         np.zeros((lb_start_mm, lb_start_mm), dtype=np.complex128) for __ in range(3)
    #     )
    #     mr_e[ie] = tuple(
    #         np.zeros((lb_end_mm, lb_end_mm), dtype=np.complex128) for __ in range(3)
    #     )
    #     lg_s[ie] = tuple(
    #         np.zeros((lb_start_mm, lb_start_mm), dtype=np.complex128) for __ in range(2)
    #     )
    #     lg_e[ie] = tuple(
    #         np.zeros((lb_end_mm, lb_end_mm), dtype=np.complex128) for __ in range(2)
    #     )
    #     ll_s[ie] = tuple(
    #         np.zeros((lb_start_mm, lb_start_mm), dtype=np.complex128) for __ in range(2)
    #     )
    #     ll_e[ie] = tuple(
    #         np.zeros((lb_end_mm, lb_end_mm), dtype=np.complex128) for __ in range(2)
    #     )

        # mr_s_ref[ie] = tuple(
        #     np.zeros((lb_start_mm, lb_start_mm), dtype=np.complex128) for __ in range(3)
        # )
        # mr_e_ref[ie] = tuple(
        #     np.zeros((lb_end_mm, lb_end_mm), dtype=np.complex128) for __ in range(3)
        # )
        # lg_s_ref[ie] = tuple(
        #     np.zeros((lb_start_mm, lb_start_mm), dtype=np.complex128) for __ in range(2)
        # )
        # lg_e_ref[ie] = tuple(
        #     np.zeros((lb_end_mm, lb_end_mm), dtype=np.complex128) for __ in range(2)
        # )
        # ll_s_ref[ie] = tuple(
        #     np.zeros((lb_start_mm, lb_start_mm), dtype=np.complex128) for __ in range(2)
        # )
        # ll_e_ref[ie] = tuple(
        #     np.zeros((lb_end_mm, lb_end_mm), dtype=np.complex128) for __ in range(2)
        # )

    mr_host = cpx.empty_pinned((ne, len(rows_m)), dtype = np.complex128)
    lg_host = cpx.empty_pinned((ne, len(rows_l)), dtype = np.complex128)
    ll_host = cpx.empty_pinned((ne, len(rows_l)), dtype = np.complex128)

    ij2ji_dev = cp.asarray(ij2ji)
    rows_dev = cp.asarray(rows)
    columns_dev = cp.asarray(columns)
    pl_rgf, pg_rgf, pr_rgf = polarization_preprocess_2d(pl_p2w, pg_p2w, pr_p2w, rows, columns, ij2ji, NCpSC, bmin, bmax, homogenize)

    nao = vh.shape[0]
    vh_dev = cp.sparse.csr_matrix(vh)
    vh_ct_dev = vh_dev.T.conj(copy=False)
    identity = cp.sparse.identity(nao)

    mr_dev = cp.empty((1, len(rows_m)), dtype=np.complex128)
    lg_dev = cp.empty((1, len(rows_l)), dtype=np.complex128)
    ll_dev = cp.empty((1, len(rows_l)), dtype=np.complex128)

    # slice block start diagonal and slice block start off diagonal
    slb_sd = slice(0, lb_start)
    slb_so = slice(lb_start, 2 * lb_start)
    # slice block end diagonal and slice block end off diagonal
    slb_ed = slice(nao - lb_end, nao)
    slb_eo = slice(nao - 2 * lb_end, nao - lb_end)

    vh_s1 = cp.zeros((ne, lb_start, lb_start), dtype=np.complex128)
    vh_s2 = cp.zeros((ne, lb_start, lb_start), dtype=np.complex128)
    pg_s1 = cp.zeros((ne, lb_start, lb_start), dtype=np.complex128)
    pg_s2 = cp.zeros((ne, lb_start, lb_start), dtype=np.complex128)
    pl_s1 = cp.zeros((ne, lb_start, lb_start), dtype=np.complex128)
    pl_s2 = cp.zeros((ne, lb_start, lb_start), dtype=np.complex128)
    pr_s1 = cp.zeros((ne, lb_start, lb_start), dtype=np.complex128)
    pr_s2 = cp.zeros((ne, lb_start, lb_start), dtype=np.complex128)
    pr_s3 = cp.zeros((ne, lb_start, lb_start), dtype=np.complex128)

    vh_e1 = cp.zeros((ne, lb_end, lb_end), dtype=np.complex128)
    vh_e2 = cp.zeros((ne, lb_end, lb_end), dtype=np.complex128)
    pg_e1 = cp.zeros((ne, lb_end, lb_end), dtype=np.complex128)
    pg_e2 = cp.zeros((ne, lb_end, lb_end), dtype=np.complex128)
    pl_e1 = cp.zeros((ne, lb_end, lb_end), dtype=np.complex128)
    pl_e2 = cp.zeros((ne, lb_end, lb_end), dtype=np.complex128)
    pr_e1 = cp.zeros((ne, lb_end, lb_end), dtype=np.complex128)
    pr_e2 = cp.zeros((ne, lb_end, lb_end), dtype=np.complex128)
    pr_e3 = cp.zeros((ne, lb_end, lb_end), dtype=np.complex128)

    pr_dev = None
    pg_dev = None
    pl_dev = None

    cp.cuda.Stream.null.synchronize()
    comm.Barrier()

    start_spgemm = time.perf_counter()
    time_copy_in = 0
    time_spgemm = 0
    time_copy_out = 0
    time_dense = 0

    for ie in range(ne):

        cp.cuda.Stream.null.synchronize()
        time_copy_in -= time.perf_counter()

        # if pr_dev is None:
        pr_dev = cp.sparse.csr_matrix((cp.asarray(pr_rgf[ie]), (rows_dev, columns_dev)), shape = (nao, nao))
        pg_dev = cp.sparse.csr_matrix((cp.asarray(pg_rgf[ie]), (rows_dev, columns_dev)), shape = (nao, nao))
        pl_dev = cp.sparse.csr_matrix((cp.asarray(pl_rgf[ie]), (rows_dev, columns_dev)), shape = (nao, nao))
        # else:
        #     pr_tmp = cp.asarray(pr_rgf[ie])
        #     pr_dev.data[:] = pr_tmp[ij2ji]
        #     pg_tmp = cp.asarray(pg_rgf[ie])
        #     pg_dev.data[:] = pg_tmp[ij2ji]
        #     pl_tmp = cp.asarray(pl_rgf[ie])
        #     pl_dev.data[:] = pl_tmp[ij2ji]
         
        # pr_dev = cp.asarray(pr_rgf[ie])
        # pg_dev = cp.asarray(pg_rgf[ie])
        # pl_dev = cp.asarray(pl_rgf[ie])
        # pr_dev = cp.sparse.csr_matrix((pr_dev, (rows_dev, columns_dev)), shape = (nao, nao))
        # pg_dev = cp.sparse.csr_matrix((pg_dev, (rows_dev, columns_dev)), shape = (nao, nao))
        # pl_dev = cp.sparse.csr_matrix((pl_dev, (rows_dev, columns_dev)), shape = (nao, nao))
            
        cp.cuda.Stream.null.synchronize()
        time_copy_in += time.perf_counter()

        time_spgemm -= time.perf_counter()

        mr_dev[0] = (identity - spgemm(vh_dev, pr_dev)).data
        spgemm_direct(spgemm(vh_dev, pg_dev), vh_ct_dev, lg_dev[0])
        spgemm_direct(spgemm(vh_dev, pl_dev), vh_ct_dev, ll_dev[0])

        cp.cuda.Stream.null.synchronize()
        time_spgemm += time.perf_counter()

        # #sp_mm_gpu(pr[ie], pg[ie], pl[ie], vh, mr_dev[0], lg_dev[0], ll_dev[0], nao)
        # sp_mm_gpu(pr_rgf[ie], pg_rgf[ie], pl_rgf[ie], rows_dev, columns_dev, vh_dev, mr_dev[0], lg_dev[0], ll_dev[0], nao)
    
        time_copy_out -= time.perf_counter()
        
        mr_host[ie, :] = cp.asnumpy(mr_dev[0])
        lg_host[ie, :] = cp.asnumpy(lg_dev[0])
        ll_host[ie, :] = cp.asnumpy(ll_dev[0])

        cp.cuda.Stream.null.synchronize()
        time_copy_out += time.perf_counter()

        # vh_s1[ie] = cp.ascontiguousarray(vh_dev[slb_sd, slb_sd].toarray(order="C"))
        # vh_s2[ie] = cp.ascontiguousarray(vh_dev[slb_sd, slb_so].toarray(order="C"))
        # pg_s1[ie] = cp.ascontiguousarray(pg_dev[slb_sd, slb_sd].toarray(order="C"))
        # pg_s2[ie] = cp.ascontiguousarray(pg_dev[slb_sd, slb_so].toarray(order="C"))
        # pl_s1[ie] = cp.ascontiguousarray(pl_dev[slb_sd, slb_sd].toarray(order="C"))
        # pl_s2[ie] = cp.ascontiguousarray(pl_dev[slb_sd, slb_so].toarray(order="C"))
        # pr_s1[ie] = cp.ascontiguousarray(pr_dev[slb_sd, slb_sd].toarray(order="C"))
        # pr_s2[ie] = cp.ascontiguousarray(pr_dev[slb_sd, slb_so].toarray(order="C"))
        # pr_s3[ie] = cp.ascontiguousarray(pr_dev[slb_so, slb_sd].toarray(order="C"))

        # vh_e1[ie] = cp.ascontiguousarray(vh_dev[slb_ed, slb_ed].toarray(order="C"))
        # vh_e2[ie] = cp.ascontiguousarray(vh_dev[slb_ed, slb_eo].conjugate().transpose().toarray(order="C"))
        # pg_e1[ie] = cp.ascontiguousarray(pg_dev[slb_ed, slb_ed].toarray(order="C"))
        # pg_e2[ie] = cp.ascontiguousarray(-pg_dev[slb_ed, slb_eo].conjugate().transpose().toarray(order="C"))
        # pl_e1[ie] = cp.ascontiguousarray(pl_dev[slb_ed, slb_ed].toarray(order="C"))
        # pl_e2[ie] = cp.ascontiguousarray(-pl_dev[slb_ed, slb_eo].conjugate().transpose().toarray(order="C"))
        # pr_e1[ie] = cp.ascontiguousarray(pr_dev[slb_ed, slb_ed].toarray(order="C"))
        # pr_e2[ie] = cp.ascontiguousarray(pr_dev[slb_eo, slb_ed].toarray(order="C"))
        # pr_e3[ie] = cp.ascontiguousarray(pr_dev[slb_ed, slb_eo].toarray(order="C"))

        time_dense -= time.perf_counter()

        num_threads = min(1024, lb_start)
        num_thread_blocks = lb_start
        _toarray[num_thread_blocks, num_threads](vh_dev.data, vh_dev.indices, vh_dev.indptr, vh_s1[ie], slb_sd.start, slb_sd.stop, slb_sd.start, slb_sd.stop)
        _toarray[num_thread_blocks, num_threads](vh_dev.data, vh_dev.indices, vh_dev.indptr, vh_s2[ie], slb_sd.start, slb_sd.stop, slb_so.start, slb_so.stop)
        _toarray[num_thread_blocks, num_threads](pg_dev.data, pg_dev.indices, pg_dev.indptr, pg_s1[ie], slb_sd.start, slb_sd.stop, slb_sd.start, slb_sd.stop)
        _toarray[num_thread_blocks, num_threads](pg_dev.data, pg_dev.indices, pg_dev.indptr, pg_s2[ie], slb_sd.start, slb_sd.stop, slb_so.start, slb_so.stop)
        _toarray[num_thread_blocks, num_threads](pl_dev.data, pl_dev.indices, pl_dev.indptr, pl_s1[ie], slb_sd.start, slb_sd.stop, slb_sd.start, slb_sd.stop)
        _toarray[num_thread_blocks, num_threads](pl_dev.data, pl_dev.indices, pl_dev.indptr, pl_s2[ie], slb_sd.start, slb_sd.stop, slb_so.start, slb_so.stop)
        _toarray[num_thread_blocks, num_threads](pr_dev.data, pr_dev.indices, pr_dev.indptr, pr_s1[ie], slb_sd.start, slb_sd.stop, slb_sd.start, slb_sd.stop)
        _toarray[num_thread_blocks, num_threads](pr_dev.data, pr_dev.indices, pr_dev.indptr, pr_s2[ie], slb_sd.start, slb_sd.stop, slb_so.start, slb_so.stop)
        _toarray[num_thread_blocks, num_threads](pr_dev.data, pr_dev.indices, pr_dev.indptr, pr_s3[ie], slb_so.start, slb_so.stop, slb_sd.start, slb_sd.stop)

        # vh_s1[ie] = vh_dev[slb_sd, slb_sd].toarray()
        # vh_s2[ie] = vh_dev[slb_sd, slb_so].toarray()
        # pg_s1[ie] = pg_dev[slb_sd, slb_sd].toarray()
        # pg_s2[ie] = pg_dev[slb_sd, slb_so].toarray()
        # pl_s1[ie] = pl_dev[slb_sd, slb_sd].toarray()
        # pl_s2[ie] = pl_dev[slb_sd, slb_so].toarray()
        # pr_s1[ie] = pr_dev[slb_sd, slb_sd].toarray()
        # pr_s2[ie] = pr_dev[slb_sd, slb_so].toarray()
        # pr_s3[ie] = pr_dev[slb_so, slb_sd].toarray()

        num_threads = min(1024, lb_end)
        num_thread_blocks = lb_end
        _toarray[num_thread_blocks, num_threads](vh_dev.data, vh_dev.indices, vh_dev.indptr, vh_e1[ie], slb_ed.start, slb_ed.stop, slb_ed.start, slb_ed.stop)
        _toarray[num_thread_blocks, num_threads](vh_dev.data, vh_dev.indices, vh_dev.indptr, vh_e2[ie], slb_ed.start, slb_ed.stop, slb_eo.start, slb_eo.stop)
        _toarray[num_thread_blocks, num_threads](pg_dev.data, pg_dev.indices, pg_dev.indptr, pg_e1[ie], slb_ed.start, slb_ed.stop, slb_ed.start, slb_ed.stop)
        _toarray[num_thread_blocks, num_threads](pg_dev.data, pg_dev.indices, pg_dev.indptr, pg_e2[ie], slb_ed.start, slb_ed.stop, slb_eo.start, slb_eo.stop)
        _toarray[num_thread_blocks, num_threads](pl_dev.data, pl_dev.indices, pl_dev.indptr, pl_e1[ie], slb_ed.start, slb_ed.stop, slb_ed.start, slb_ed.stop)
        _toarray[num_thread_blocks, num_threads](pl_dev.data, pl_dev.indices, pl_dev.indptr, pl_e2[ie], slb_ed.start, slb_ed.stop, slb_eo.start, slb_eo.stop)
        _toarray[num_thread_blocks, num_threads](pr_dev.data, pr_dev.indices, pr_dev.indptr, pr_e1[ie], slb_ed.start, slb_ed.stop, slb_ed.start, slb_ed.stop)
        _toarray[num_thread_blocks, num_threads](pr_dev.data, pr_dev.indices, pr_dev.indptr, pr_e2[ie], slb_eo.start, slb_eo.stop, slb_ed.start, slb_ed.stop)
        _toarray[num_thread_blocks, num_threads](pr_dev.data, pr_dev.indices, pr_dev.indptr, pr_e3[ie], slb_ed.start, slb_ed.stop, slb_eo.start, slb_eo.stop)

        vh_e2[ie] = vh_e2[ie].T.conj()
        pg_e2[ie] = -pg_e2[ie].T.conj()
        pl_e2[ie] = -pl_e2[ie].T.conj()

        # vh_e1[ie] = vh_dev[slb_ed, slb_ed].toarray()
        # vh_e2[ie] = vh_dev[slb_ed, slb_eo].conjugate().transpose().toarray()
        # pg_e1[ie] = pg_dev[slb_ed, slb_ed].toarray()
        # pg_e2[ie] = -pg_dev[slb_ed, slb_eo].conjugate().transpose().toarray()
        # pl_e1[ie] = pl_dev[slb_ed, slb_ed].toarray()
        # pl_e2[ie] = -pl_dev[slb_ed, slb_eo].conjugate().transpose().toarray()
        # pr_e1[ie] = pr_dev[slb_ed, slb_ed].toarray()
        # pr_e2[ie] = pr_dev[slb_eo, slb_ed].toarray()
        # pr_e3[ie] = pr_dev[slb_ed, slb_eo].toarray()

        cp.cuda.Stream.null.synchronize()
        time_dense += time.perf_counter()

        # obc_w_gpu.obc_w_mm_gpu_2(vh_dev,
        #                          pg_dev, pl_dev, pr_dev,
        #                          bmax, bmin,
        #                          dvh_sd_ref[ie], dvh_ed_ref[ie],
        #                          dmr_sd_ref[ie], dmr_ed_ref[ie],
        #                          dlg_sd_ref[ie], dlg_ed_ref[ie],
        #                          dll_sd_ref[ie], dll_ed_ref[ie],
        #                          mr_s_ref[ie], mr_e_ref[ie],
        #                          lg_s_ref[ie], lg_e_ref[ie],
        #                          ll_s_ref[ie], ll_e_ref[ie],
        #                          vh_s_ref[ie], vh_e_ref[ie],
        #                          mb00_ref[ie], mbNN_ref[ie],
        #                          rows_dev, columns_dev,
        #                          nbc, NCpSC, block_inv, use_dace, validate_dace, ref_flag)
    
    cp.cuda.Stream.null.synchronize()
    comm.Barrier()
    finish_spgemm = time.perf_counter()
    if rank == 0:
        print("OBC-W region", flush=True)
        print("    Time for overall spgemm loop: %.3f s" % (finish_spgemm - start_spgemm), flush=True)
        print("        Time for copy in: %.3f s" % time_copy_in, flush=True)
        print("        Time for spgemm: %.3f s" % time_spgemm, flush=True)
        print("        Time for copy out: %.3f s" % time_copy_out, flush=True)
        print("        Time for dense: %.3f s" % time_dense, flush=True)
    
    start_obc_mm = time.perf_counter()

    obc_w_gpu.obc_w_mm_batched_gpu(vh_s1, vh_s2, pg_s1, pg_s2, pl_s1, pl_s2, pr_s1, pr_s2, pr_s3,
                                   vh_e1, vh_e2, pg_e1, pg_e2, pl_e1, pl_e2, pr_e1, pr_e2, pr_e3,
                                   dmr_sd, dmr_ed, dlg_sd, dlg_ed, dll_sd, dll_ed,
                                #    mr_s, mr_e, lg_s, lg_e, ll_s, ll_e,
                                   mr_s0, mr_s1, mr_s2, mr_e0, mr_e1, mr_e2, lg_s0, lg_s1, lg_e0, lg_e1, ll_s0, ll_s1, ll_e0, ll_e1,
                                   vh_s, vh_e, mb00, mbNN, nbc, NCpSC)
    cp.cuda.Stream.null.synchronize()
    comm.Barrier()
    finish_obc_mm = time.perf_counter()
    if rank == 0:
        print("    Time for obc mm: %.3f s" % (finish_obc_mm - start_obc_mm), flush=True)
    
    def _norm(val, ref):
        return np.linalg.norm(val - ref) / np.linalg.norm(ref)
    
    # print("dmr_sd", _norm(dmr_sd, dmr_sd_ref))
    # print("dmr_ed", _norm(dmr_ed, dmr_ed_ref))
    # print("dlg_sd", _norm(dlg_sd, dlg_sd_ref))
    # print("dlg_ed", _norm(dlg_ed, dlg_ed_ref))
    # print("dll_sd", _norm(dll_sd, dll_sd_ref))
    # print("dll_ed", _norm(dll_ed, dll_ed_ref))
    # print("vh_s", _norm(vh_s, vh_s_ref))
    # print("vh_e", _norm(vh_e, vh_e_ref))
    # print("mb00", _norm(mb00, mb00_ref))
    # print("mbNN", _norm(mbNN, mbNN_ref))

    # for ie in range(ne):
    #     print("mr_s[0]", _norm(mr_s[ie][0], mr_s_ref[ie][0]))
    #     print("mr_s[1]", _norm(mr_s[ie][1], mr_s_ref[ie][1]))
    #     print("mr_s[2]", _norm(mr_s[ie][2], mr_s_ref[ie][2]))
    #     print("mr_e[0]", _norm(mr_e[ie][0], mr_e_ref[ie][0]))
    #     print("mr_e[1]", _norm(mr_e[ie][1], mr_e_ref[ie][1]))
    #     print("mr_e[2]", _norm(mr_e[ie][2], mr_e_ref[ie][2]))
    #     print("lg_s[0]", _norm(lg_s[ie][0], lg_s_ref[ie][0]))
    #     print("lg_s[1]", _norm(lg_s[ie][1], lg_s_ref[ie][1]))
    #     print("lg_e[0]", _norm(lg_e[ie][0], lg_e_ref[ie][0]))
    #     print("lg_e[1]", _norm(lg_e[ie][1], lg_e_ref[ie][1]))
    #     print("ll_s[0]", _norm(ll_s[ie][0], ll_s_ref[ie][0]))
    #     print("ll_s[1]", _norm(ll_s[ie][1], ll_s_ref[ie][1]))
    #     print("ll_e[0]", _norm(ll_e[ie][0], ll_e_ref[ie][0]))
    #     print("ll_e[1]", _norm(ll_e[ie][1], ll_e_ref[ie][1]))

    # dmr_sd = dmr_sd_ref
    # dmr_ed = dmr_ed_ref
    # dlg_sd = dlg_sd_ref
    # dlg_ed = dlg_ed_ref
    # dll_sd = dll_sd_ref
    # dll_ed = dll_ed_ref
    # mr_s = mr_s_ref
    # mr_e = mr_e_ref
    # lg_s = lg_s_ref
    # lg_e = lg_e_ref
    # ll_s = ll_s_ref
    # ll_e = ll_e_ref
    # vh_s = vh_s_ref
    # vh_e = vh_e_ref
    # mb00 = mb00_ref
    # mbNN = mbNN_ref

    # with concurrent.futures.ThreadPoolExecutor(max_workers=worker_num) as executor:
    #     executor.map(obc_w_gpu.obc_w_mm_gpu_2,
    #         repeat(vh_dev),
    #         pg_rgf,
    #         pl_rgf,
    #         pr_rgf,
    #         repeat(bmax),
    #         repeat(bmin),
    #         dvh_sd,
    #         dvh_ed,
    #         dmr_sd,
    #         dmr_ed,
    #         dlg_sd,
    #         dlg_ed,
    #         dll_sd,
    #         dll_ed,
    #         mr_s,
    #         mr_e,
    #         lg_s,
    #         lg_e,
    #         ll_s,
    #         ll_e,
    #         vh_s,
    #         vh_e,
    #         mb00,
    #         mbNN,
    #         repeat(rows_dev),
    #         repeat(columns_dev),
    #         repeat(nbc),
    #         repeat(NCpSC),
    #         repeat(block_inv),
    #         repeat(use_dace),
    #         repeat(validate_dace),
    #         repeat(ref_flag),
    #     )

    if compute_mode == 0:

        with concurrent.futures.ThreadPoolExecutor(max_workers=worker_num) as executor:
            # results = executor.map(obc_w_cpu.obc_w_cpu, repeat(vh),
            executor.map(
                obc_w_cpu.obc_w_cpu_excludingmm,
                repeat(vh),
                pg,
                pl,
                pr,
                repeat(bmax),
                repeat(bmin),
                dvh_sd,
                dvh_ed,
                dmr_sd,
                dmr_ed,
                dlg_sd,
                dlg_ed,
                dll_sd,
                dll_ed,
                mr_s,
                mr_e,
                lg_s,
                lg_e,
                ll_s,
                ll_e,
                vh_s,
                vh_e,
                mb00,
                mbNN,
                repeat(nbc),
                repeat(NCpSC),
                repeat(block_inv),
                repeat(use_dace),
                repeat(validate_dace),
                repeat(ref_flag),
            )

    else:

        comm.Barrier()
        start_save = time.perf_counter()

        # if (rank % 40) == 0 and (iter % 100) < 5:
        #     path = '/capstor/scratch/cscs/ldeuschl/block_results/blocks_longSi_NW_14400_54_GVG_realV_eps5_ps1_mems50/'
        #     filename = path + "W_energy_%.4f" % energy[0] + "iter_" + str(iter) + "_rank_" + str(rank) + ".npz"
        #     np.savez(filename, MB_left = mb00, MB_right = mbNN, M00_left = mr_s0, M01_left = mr_s1, M10_left = mr_s2, M00_right = mr_s0, M01_right = mr_e1, M10_right = mr_e2)



        comm.Barrier()
        finish_save = time.perf_counter()

        if rank == 0:
            print("    Time for saving boundary blocks w: %.3f s" % (finish_save - start_save), flush=True)


        comm.Barrier()
        start_beyn_w = time.perf_counter()

        with concurrent.futures.ThreadPoolExecutor(max_workers=worker_num) as executor:
            # results = executor.map(obc_w_cpu.obc_w_cpu, repeat(vh),
            executor.map(
                obc_w_gpu.obc_w_gpu_beynonly_2,
                dxr_sd,
                dxr_ed,
                dvh_sd,
                dvh_ed,
                dmr_sd,
                dmr_ed,
                mr_s0, mr_s1, mr_s2,
                mr_e0, mr_e1, mr_e2,
                vh_s,
                vh_e,
                mb00,
                mbNN,
                repeat(nbc),
                repeat(NCpSC),
                repeat(block_inv),
                repeat(use_dace),
                repeat(validate_dace),
                repeat(ref_flag),
            )

        # imag_lim = 1e-4
        # R = 1e4
        # matrix_blocks_left = cp.asarray(mb00)
        # # M00_left = cp.empty((ne, lb_start_mm, lb_start_mm), dtype=np.complex128)
        # # M01_left = cp.empty_like(M00_left)
        # # M10_left = cp.empty_like(M00_left)
        # # for ie in range(ne):
        # #     M00_left[ie].set(mr_s[ie][0])
        # #     M01_left[ie].set(mr_s[ie][1])
        # #     M10_left[ie].set(mr_s[ie][2])
        # M00_left = cp.asarray(mr_s0)
        # M01_left = cp.asarray(mr_s1)
        # M10_left = cp.asarray(mr_s2)
        # dmr, dxr_sd_gpu, condL, _ = beyn_gpu(nbc * NCpSC, matrix_blocks_left, M00_left, M01_left, M10_left, imag_lim, R, 'L')
        # assert not any(np.isnan(cond) for cond in condL)
        # dxr_sd_gpu.get(out=dxr_sd)
        # dmr_sd -= dmr.get()
        # (M10_left @ dxr_sd_gpu @ cp.asarray(vh_s)).get(out=dvh_sd)
        # matrix_blocks_right = cp.asarray(mbNN)
        # # M00_right = cp.empty((ne, lb_end_mm, lb_end_mm), dtype=np.complex128)
        # # M01_right = cp.empty_like(M00_right)
        # # M10_right = cp.empty_like(M00_right)
        # # for ie in range(ne):
        # #     M00_right[ie].set(mr_e[ie][0])
        # #     M01_right[ie].set(mr_e[ie][1])
        # #     M10_right[ie].set(mr_e[ie][2])
        # M00_right = cp.asarray(mr_e0)
        # M01_right = cp.asarray(mr_e1)
        # M10_right = cp.asarray(mr_e2)
        # dmr, dxr_ed_gpu, condR, _ = beyn_gpu(nbc * NCpSC, matrix_blocks_right, M00_right, M01_right, M10_right, imag_lim, R, 'R')
        # assert not any(np.isnan(cond) for cond in condR)
        # dxr_ed_gpu.get(out=dxr_ed)
        # dmr_ed -= dmr.get()
        # (M01_right @ dxr_ed_gpu @ cp.asarray(vh_e)).get(out=dvh_ed)

        cp.cuda.Stream.null.synchronize()
        comm.Barrier()
        finish_beyn_w = time.perf_counter()
        if rank == 0:
            print("    Time for beyn w: %.3f s" % (finish_beyn_w - start_beyn_w), flush=True)


        # with concurrent.futures.ThreadPoolExecutor(max_workers=worker_num) as executor:
        #     executor.map(obc_w_gpu.obc_w_L_lg,
        #         dlg_sd,
        #         dlg_ed,
        #         dll_sd,
        #         dll_ed,
        #         mr_s,
        #         mr_e,
        #         lg_s,
        #         lg_e,
        #         ll_s,
        #         ll_e,
        #         dxr_sd,
        #         dxr_ed,
        #     )

        comm.Barrier()
        start_save = time.perf_counter()

        # if (rank % 40) == 0 and (iter % 100) < 5:
        #     path = '/capstor/scratch/cscs/ldeuschl/block_results/blocks_longSi_NW_14400_54_GVG_realV_eps5_ps1_mems50/'
        #     filename = path + "W<>_energy_%.4f" % energy[0] + "iter_" + str(iter) + "_rank_" + str(rank) + ".npz"
        #     np.savez(filename, LG00_left = lg_s0, LG01_left = lg_s1, LL00_left = ll_s0, LL01_left = ll_s1, LG00_right = lg_e0, LG01_right = lg_e1, LL00_right = ll_e0, LL01_right = ll_e1)



        comm.Barrier()
        finish_save = time.perf_counter()

        if rank == 0:
            print("    Time for saving boundary blocks w<>: %.3f s" % (finish_save - start_save), flush=True)
            
        comm.Barrier()
        start_obc_l = time.perf_counter()

        with concurrent.futures.ThreadPoolExecutor(max_workers=worker_num) as executor:
            executor.map(obc_w_gpu.obc_w_L_lg_2,
                dlg_sd,
                dlg_ed,
                dll_sd,
                dll_ed,
                mr_s0, mr_s1, mr_s2,
                mr_e0, mr_e1, mr_e2,
                lg_s0, lg_s1,
                lg_e0, lg_e1,
                ll_s0, ll_s1,
                ll_e0, ll_e1,
                dxr_sd,
                dxr_ed,
            )
        
        cp.cuda.Stream.null.synchronize()
        comm.Barrier()
        finish_obc_l = time.perf_counter()
        if rank == 0:
            print("    Time for obc l: %.3f s" % (finish_obc_l - start_obc_l), flush=True)

    l_defect = np.count_nonzero(np.isnan(condl))
    r_defect = np.count_nonzero(np.isnan(condr))

    if l_defect > 0 or r_defect > 0:
        print(
            "Warning: %d left and %d right boundary conditions are not satisfied."
            % (l_defect, r_defect)
        )

    comm.Barrier()
    if rank == 0:
        time_OBC += time.perf_counter()
        print("Time for OBC: %.3f s" % time_OBC, flush=True)
        time_spmm = -time.perf_counter()

    # # --- Sparse matrix multiplication ---------------------------------
    # # NOTE: This is what currently kills the performance. The CPU
    # # implementation is not efficient. The GPU implementation is not
    # # memory efficient.
        
    # mr_host = np.empty((ne, len(rows_m)), dtype = np.complex128)
    # lg_host = np.empty((ne, len(rows_l)), dtype = np.complex128)
    # ll_host = np.empty((ne, len(rows_l)), dtype = np.complex128)

    # pl_rgf, pg_rgf, pr_rgf = polarization_preprocess_2d(pl_p2w, pg_p2w, pr_p2w, rows, columns, ij2ji, NCpSC, bmin, bmax, homogenize)
    # rows_dev = cp.asarray(rows)
    # columns_dev = cp.asarray(columns)

    # use_gpu = True
    # if use_gpu:
    #     nao = vh.shape[0]

    #     for ie in range(ne):
    #         mr_dev = cp.empty((1, len(rows_m)), dtype=np.complex128)
    #         lg_dev = cp.empty((1, len(rows_l)), dtype=np.complex128)
    #         ll_dev = cp.empty((1, len(rows_l)), dtype=np.complex128)
    #         #sp_mm_gpu(pr[ie], pg[ie], pl[ie], vh, mr_dev[0], lg_dev[0], ll_dev[0], nao)
    #         sp_mm_gpu(pr_rgf[ie], pg_rgf[ie], pl_rgf[ie], rows_dev, columns_dev, vh, mr_dev[0], lg_dev[0], ll_dev[0], nao)
        
    #         mr_host[ie, :] = cp.asnumpy(mr_dev[0])
    #         lg_host[ie, :] = cp.asnumpy(lg_dev[0])
    #         ll_host[ie, :] = cp.asnumpy(ll_dev[0])
        
    # # use_gpu = True
    # # if use_gpu:
    # #     nao = vh.shape[0]
    # #     mr_dev = cp.empty((ne, len(rows_m)), dtype=np.complex128)
    # #     lg_dev = cp.empty((ne, len(rows_l)), dtype=np.complex128)
    # #     ll_dev = cp.empty((ne, len(rows_l)), dtype=np.complex128)

    # #     for ie in range(ne):
    # #         sp_mm_gpu(pr[ie], pg[ie], pl[ie], vh, mr_dev[ie], lg_dev[ie], ll_dev[ie], nao)

    # #     mr_host = cp.asnumpy(mr_dev)
    # #     lg_host = cp.asnumpy(lg_dev)
    # #     ll_host = cp.asnumpy(ll_dev)

    # else:
    #     vh_ct = vh.conj().transpose()
    #     nao = vh.shape[0]
    #     mr_host = np.empty((ne, len(rows_m)), dtype=np.complex128)
    #     lg_host = np.empty((ne, len(rows_l)), dtype=np.complex128)
    #     ll_host = np.empty((ne, len(rows_l)), dtype=np.complex128)

    #     with concurrent.futures.ThreadPoolExecutor(max_workers=worker_num) as executor:
    #         results = executor.map(
    #             rgf_W_GPU.sp_mm_cpu, pr, pg, pl, repeat(vh), repeat(vh_ct), repeat(nao)
    #         )
    #         for ie, res in enumerate(results):
    #             mr_host[ie] = res[0][rows_m, columns_m]
    #             lg_host[ie] = res[1][rows_l, columns_l]
    #             ll_host[ie] = res[2][rows_l, columns_l]

    comm.Barrier()
    if rank == 0:
        time_spmm += time.perf_counter()
        print("Time for spgemm: %.3f s" % time_spmm, flush=True)
        time_GF_trafo = -time.perf_counter()

    # --- Transform L to the (NE, NNZ) format --------------------------

    mapping_diag_l = rgf_GF_GPU_combo.map_to_mapping(map_diag_l, nb_mm)
    mapping_upper_l = rgf_GF_GPU_combo.map_to_mapping(map_upper_l, nb_mm - 1)
    mapping_lower_l = rgf_GF_GPU_combo.map_to_mapping(map_lower_l, nb_mm - 1)

    mapping_diag_m = rgf_GF_GPU_combo.map_to_mapping(map_diag_m, nb_mm)
    mapping_upper_m = rgf_GF_GPU_combo.map_to_mapping(map_upper_m, nb_mm - 1)
    mapping_lower_m = rgf_GF_GPU_combo.map_to_mapping(map_lower_m, nb_mm - 1)

    mapping_diag_mm = rgf_GF_GPU_combo.map_to_mapping(map_diag_mm, nb_mm)
    mapping_upper_mm = rgf_GF_GPU_combo.map_to_mapping(map_upper_mm, nb_mm - 1)
    mapping_lower_mm = rgf_GF_GPU_combo.map_to_mapping(map_lower_mm, nb_mm - 1)

    vh_diag, vh_upper, vh_lower = rgf_GF_GPU_combo.csr_to_block_tridiagonal_csr(
        vh, bmin_mm, bmax_mm + 1
    )
    # mr_rgf = np.empty((ne, len(columns_m)), dtype=np.complex128)
    # lg_rgf = np.empty((ne, len(columns_l)), dtype=np.complex128)
    # ll_rgf = np.empty((ne, len(columns_l)), dtype=np.complex128)

    # Extract the data from the sparse matrices.
    # NOTE: Kinda inefficient but ll_rgf can be ragged so this is safer.
    # def _get_data(x, x_rgf, rows, columns):
    #     x_rgf[:] = x[rows, columns]

    # with concurrent.futures.ThreadPoolExecutor(max_workers=worker_num) as executor:
    #     executor.map(_get_data, mr, mr_rgf, repeat(rows_m), repeat(columns_m))
    #     executor.map(_get_data, lg, lg_rgf, repeat(rows_l), repeat(columns_l))
    #     executor.map(_get_data, ll, ll_rgf, repeat(rows_l), repeat(columns_l))

    # Sanity checks.
    # assert lg_rgf.shape[1] == rows_l.size
    # assert ll_rgf.shape[1] == rows_l.size
    # assert mr_rgf.shape[1] == rows_m.size

    cp.cuda.Stream.null.synchronize()
    comm.Barrier()
    if rank == 0:
        time_GF_trafo += time.perf_counter()
        print("Time for W transformation: %.3f s" % time_GF_trafo, flush=True)
        # time_GF = -time.perf_counter()
        time_GF = -time.perf_counter()

    # --- Compute screened interaction ---------------------------------

    input_stream = cp.cuda.stream.Stream(non_blocking=True)

    energy_batchsize = 4
    energy_batch = np.arange(0, ne, energy_batchsize)

    for ie in energy_batch:
        rgf_W_GPU_combo.rgf_batched_GPU(
            energies=energy[ie : ie + energy_batchsize],
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
            mr_host=mr_host[ie : ie + energy_batchsize, :],
            ll_host=ll_host[ie : ie + energy_batchsize, :],
            lg_host=lg_host[ie : ie + energy_batchsize, :],
            dvh_left_host=dvh_sd[ie : ie + energy_batchsize, :, :],
            dvh_right_host=dvh_ed[ie : ie + energy_batchsize, :, :],
            dmr_left_host=dmr_sd[ie : ie + energy_batchsize, :, :],
            dmr_right_host=dmr_ed[ie : ie + energy_batchsize, :, :],
            dlg_left_host=dlg_sd[ie : ie + energy_batchsize, :, :],
            dlg_right_host=dlg_ed[ie : ie + energy_batchsize, :, :],
            dll_left_host=dll_sd[ie : ie + energy_batchsize, :, :],
            dll_right_host=dll_ed[ie : ie + energy_batchsize, :, :],
            wr_host=wr_p2w[ie : ie + energy_batchsize, :],
            wl_host=wl_p2w[ie : ie + energy_batchsize, :],
            wg_host=wg_p2w[ie : ie + energy_batchsize, :],
            dosw=dosw[ie : ie + energy_batchsize, :],
            nEw=new[ie : ie + energy_batchsize, :],
            nPw=npw[ie : ie + energy_batchsize, :],
            bmax=bmax_mm,
            bmin=bmin_mm,
            solve=True,
            input_stream=input_stream,
        )

    cp.cuda.Stream.null.synchronize()
    comm.Barrier()
    if rank == 0:
        time_GF += time.perf_counter()
        print("Time for W: %.3f s" % time_GF, flush=True)
        time_post_proc = -time.perf_counter()

    # --- Post-processing ----------------------------------------------

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
            comm.Sendrecv(
                sendbuf=buf_send_r, dest=rank + 1, recvbuf=buf_recv_r, source=rank + 1
            )

        elif rank == size - 1:
            buf_send_l[:] = dosw[0, :]
            comm.Sendrecv(
                sendbuf=buf_send_l, dest=rank - 1, recvbuf=buf_recv_l, source=rank - 1
            )
        else:
            buf_send_r[:] = dosw[ne - 1, :]
            buf_send_l[:] = dosw[0, :]
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
    ind_zeros = np.where((F1 > 0.1) | (F2 > 0.1) | ((dDOSm > 5) & (dDOSp > 5)))[0]

    # if not((np.sum(np.isnan(F1)) == 0) and (np.sum(np.isinf(F1)) == 0)):
    #     print("encountered invalid value in F1", flush = True)
    # if not((np.sum(np.isnan(F2)) == 0) and (np.sum(np.isinf(F2))==0)):
    #     print("encountered invalid value in F2", flush = True)
    # if not((np.sum(np.isnan(dDOSm)) == 0) and (np.sum(np.isinf(dDOSm)) == 0)):
    #     print("encountered invalid value in dDOSm", flush = True)
    # if not((np.sum(np.isnan(dDOSp)) == 0) and (np.sum(np.isinf(dDOSp)) == 0)):
    #     print("encountered invalid value in dDOSp", flush = True)

    # if np.sum(np.isnan(dosw)) == 0:
    #     ind_zeros_nan_dosw = np.array([], dtype = int)
    # else:
    #     ind_zeros_nan_dosw = np.argwhere(np.isnan(dosw))[0]
    #     print("encountered nan is dosw", flush = True)

    # ind_zeros = np.concatenate((ind_zeros_nan_dosw, ind_zeros))

    # if np.sum(np.isinf(dosw)) == 0:
    #     ind_zeros_inf_dosw = np.array([], dtype = int)
    # else:
    #     ind_zeros_inf_dosw = np.argwhere(np.isinf(dosw))[0]
    #     print("encountered inf is dosw", flush = True)

    # ind_zeros = np.concatenate((ind_zeros_inf_dosw, ind_zeros))

    # if np.sum(np.isnan(new)) == 0:
    #     ind_zeros_nan_new = np.array([], dtype = int)
    # else:
    #     ind_zeros_nan_new = np.argwhere(np.isnan(new))[0]
    #     print("encountered nan in new", flush = True)
    
    # ind_zeros = np.concatenate((ind_zeros_nan_new, ind_zeros))

    # if np.sum(np.isinf(new)) == 0:
    #     ind_zeros_inf_new = np.array([], dtype = int)
    # else:
    #     ind_zeros_inf_new = np.argwhere(np.isinf(new))[0]
    #     print("encountered inf is new", flush = True)

    # ind_zeros = np.concatenate((ind_zeros_inf_new, ind_zeros))

    # if np.sum(np.isnan(npw)) == 0:
    #     ind_zeros_nan_npw = np.array([], dtype = int)
    # else:
    #     ind_zeros_nan_npw = np.argwhere(np.isnan(npw))[0]
    #     print("encountered nan in npw", flush = True)
    
    # ind_zeros = np.concatenate((ind_zeros_nan_npw, ind_zeros))

    # if np.sum(np.isinf(npw)) == 0:
    #     ind_zeros_inf_npw = np.array([], dtype = int)
    # else:
    #     ind_zeros_inf_npw = np.argwhere(np.isinf(npw))[0]
    #     print("encountered inf is npw", flush = True)

    # ind_zeros = np.concatenate((ind_zeros_inf_npw, ind_zeros))

    if idx_e[0] == 0:
        ind_zeros = np.concatenate(([0], ind_zeros))

    start_indices = idx_e[idx_e < start_index_W] - idx_e[0]
    ind_zeros = np.concatenate((start_indices, ind_zeros))
    
    # Remove the identified peaks and errors
    for index in ind_zeros:
        wr_p2w[index, :] = 0
        wl_p2w[index, :] = 0
        wg_p2w[index, :] = 0

    if np.sum(np.isnan(wr_p2w)) == 0:
        ind_zeros_w = np.array([], dtype = int)
    else:
        print("There are still nans inside wr", flush = True)
        ind_zeros_w = np.argwhere(np.isnan(wr_p2w)).T[0]
        ind_unique = np.unique(ind_zeros_w)
        wr_p2w[ind_unique, :] = 0.0

    if np.sum(np.isnan(wl_p2w)) == 0:
        ind_zeros_w = np.array([], dtype = int)
    else:
        print("There are still nans inside wl", flush = True)
        ind_zeros_w = np.argwhere(np.isnan(wl_p2w)).T[0]
        ind_unique = np.unique(ind_zeros_w)
        wl_p2w[ind_unique, :] = 0.0

    if np.sum(np.isnan(wg_p2w)) == 0:
        ind_zeros_w = np.array([], dtype = int)
    else:
        print("There are still nans inside wg", flush = True)
        ind_zeros_w = np.argwhere(np.isnan(wg_p2w)).T[0]
        ind_unique = np.unique(ind_zeros_w)
        wg_p2w[ind_unique, :] = 0.0

    assert((np.sum(np.isnan(wr_p2w)) + np.sum(np.isnan(wl_p2w)) + np.sum(np.isnan(wg_p2w))) == 0)

    cp.cuda.Stream.null.synchronize()
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