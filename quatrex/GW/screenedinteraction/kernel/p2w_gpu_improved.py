import concurrent.futures
import time
from itertools import repeat

import cupy as cp
import cupyx as cpx
import mkl
import numpy as np
import numpy.typing as npt

from quatrex.block_tri_solvers import (
    matrix_inversion_w,
    rgf_GF_GPU_combo,
    rgf_W_GPU,
    rgf_W_GPU_combo,
)
from quatrex.OBC import obc_w_cpu, obc_w_gpu
from quatrex.utils.matrix_creation import (
    extract_small_matrix_blocks,
    homogenize_matrix_Rnosym,
)


def calc_W_pool_mpi_split(
    # Hamiltonian object.
    hamiltonian_obj,
    # Energy vector.
    energy,
    # Polarization.
    pg,
    pl,
    pr,
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
    mr_s = np.ndarray((ne,), dtype=object)
    mr_e = np.ndarray((ne,), dtype=object)
    lg_s = np.ndarray((ne,), dtype=object)
    lg_e = np.ndarray((ne,), dtype=object)
    ll_s = np.ndarray((ne,), dtype=object)
    ll_e = np.ndarray((ne,), dtype=object)
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
            (PR00, PR01, PR10, _) = extract_small_matrix_blocks(
                pr[ie][bmin[0] : bmax[0] + 1, bmin[0] : bmax[0] + 1],
                pr[ie][bmin[0] : bmax[0] + 1, bmin[1] : bmax[1] + 1],
                pr[ie][bmin[1] : bmax[1] + 1, bmin[0] : bmax[0] + 1],
                NCpSC,
                "L",
            )
            pr[ie] = homogenize_matrix_Rnosym(PR00, PR01, PR10, len(bmax))
            (PL00, PL01, PL10, _) = extract_small_matrix_blocks(
                pl[ie][bmin[0] : bmax[0] + 1, bmin[0] : bmax[0] + 1],
                pl[ie][bmin[0] : bmax[0] + 1, bmin[1] : bmax[1] + 1],
                pl[ie][bmin[1] : bmax[1] + 1, bmin[0] : bmax[0] + 1],
                NCpSC,
                "L",
            )
            pl[ie] = homogenize_matrix_Rnosym(PL00, PL01, PL10, len(bmax))
            (PG00, PG01, PG10, _) = extract_small_matrix_blocks(
                pg[ie][bmin[0] : bmax[0] + 1, bmin[0] : bmax[0] + 1],
                pg[ie][bmin[0] : bmax[0] + 1, bmin[1] : bmax[1] + 1],
                pg[ie][bmin[1] : bmax[1] + 1, bmin[0] : bmax[0] + 1],
                NCpSC,
                "L",
            )
            pg[ie] = homogenize_matrix_Rnosym(PG00, PG01, PG10, len(bmax))

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

    for ie in range(ne):
        mr_s[ie] = tuple(
            np.zeros((lb_start_mm, lb_start_mm), dtype=np.complex128) for __ in range(3)
        )
        mr_e[ie] = tuple(
            np.zeros((lb_end_mm, lb_end_mm), dtype=np.complex128) for __ in range(3)
        )
        lg_s[ie] = tuple(
            np.zeros((lb_start_mm, lb_start_mm), dtype=np.complex128) for __ in range(2)
        )
        lg_e[ie] = tuple(
            np.zeros((lb_end_mm, lb_end_mm), dtype=np.complex128) for __ in range(2)
        )
        ll_s[ie] = tuple(
            np.zeros((lb_start_mm, lb_start_mm), dtype=np.complex128) for __ in range(2)
        )
        ll_e[ie] = tuple(
            np.zeros((lb_end_mm, lb_end_mm), dtype=np.complex128) for __ in range(2)
        )

        obc_w_gpu.obc_w_mm_gpu(
            vh,
            pg[ie],
            pl[ie],
            pr[ie],
            bmax,
            bmin,
            dvh_sd[ie],
            dvh_ed[ie],
            dmr_sd[ie],
            dmr_ed[ie],
            dlg_sd[ie],
            dlg_ed[ie],
            dll_sd[ie],
            dll_ed[ie],
            mr_s[ie],
            mr_e[ie],
            lg_s[ie],
            lg_e[ie],
            ll_s[ie],
            ll_e[ie],
            vh_s[ie],
            vh_e[ie],
            mb00[ie],
            mbNN[ie],
            nbc,
            NCpSC,
            block_inv,
            use_dace,
            validate_dace,
            ref_flag,
        )

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
        with concurrent.futures.ThreadPoolExecutor(max_workers=worker_num) as executor:
            # results = executor.map(obc_w_cpu.obc_w_cpu, repeat(vh),
            executor.map(
                obc_w_cpu.obc_w_cpu_beynonly,
                dxr_sd,
                dxr_ed,
                dvh_sd,
                dvh_ed,
                dmr_sd,
                dmr_ed,
                mr_s,
                mr_e,
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

        for ie in range(ne):
            obc_w_gpu.obc_w_L_lg(
                dlg_sd[ie],
                dlg_ed[ie],
                dll_sd[ie],
                dll_ed[ie],
                mr_s[ie],
                mr_e[ie],
                lg_s[ie],
                lg_e[ie],
                ll_s[ie],
                ll_e[ie],
                dxr_sd[ie],
                dxr_ed[ie],
            )

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

    # --- Sparse matrix multiplication ---------------------------------
    # NOTE: This is what currently kills the performance. The CPU
    # implementation is not efficient. The GPU implementation is not
    # memory efficient.

    # TODO: Have someone figure out which are the most memory-efficient
    # calls to spmm on the GPU and implement them.

    vh_ct = vh.conj().transpose()
    nao = vh.shape[0]
    mr = []
    lg = []
    ll = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=worker_num) as executor:
        results = executor.map(
            rgf_W_GPU.sp_mm_cpu, pr, pg, pl, repeat(vh), repeat(vh_ct), repeat(nao)
        )
        for res in results:
            mr.append(res[0])
            lg.append(res[1])
            ll.append(res[2])

    comm.Barrier()
    if rank == 0:
        time_spmm += time.perf_counter()
        print("Time for spmm: %.3f s" % time_spmm, flush=True)
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

    # Canonicalize the sparse matrices.
    [m.sum_duplicates() for m in mr]
    [l.sum_duplicates() for l in lg]
    [l.sum_duplicates() for l in ll]
    assert all(m.has_canonical_format for m in mr)
    assert all(l.has_canonical_format for l in lg)
    assert all(l.has_canonical_format for l in ll)

    # Extract the data from the sparse matrices.
    # NOTE: Kinda inefficient but ll_rgf can be ragged so this is safer.
    mr_rgf = np.array([m[rows_m, columns_m] for m in mr])
    lg_rgf = np.array([l[rows_l, columns_l] for l in lg])
    ll_rgf = np.array([l[rows_l, columns_l] for l in ll])

    # Sanity checks.
    assert lg_rgf.shape[1] == rows_l.size
    assert ll_rgf.shape[1] == rows_l.size
    # NOTE: For some reason mr_rgf has an unneeded dimension. Remove it.
    mr_rgf = mr_rgf[:, 0, :]
    assert mr_rgf.shape[1] == rows_m.size

    vh_diag, vh_upper, vh_lower = rgf_GF_GPU_combo.csr_to_block_tridiagonal_csr(
        vh, bmin_mm, bmax_mm
    )

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
            energy[ie : ie + energy_batchsize],
            mapping_diag_mm,
            mapping_upper_mm,
            mapping_lower_mm,
            mapping_diag_m,
            mapping_upper_m,
            mapping_lower_m,
            mapping_diag_l,
            mapping_upper_l,
            mapping_lower_l,
            vh_diag,
            vh_upper,
            vh_lower,
            mr_rgf[ie : ie + energy_batchsize, :],
            ll_rgf[ie : ie + energy_batchsize, :],
            lg_rgf[ie : ie + energy_batchsize, :],
            dvh_sd[ie : ie + energy_batchsize, :, :],
            dvh_ed[ie : ie + energy_batchsize, :, :],
            dmr_sd[ie : ie + energy_batchsize, :, :],
            dmr_ed[ie : ie + energy_batchsize, :, :],
            dlg_sd[ie : ie + energy_batchsize, :, :],
            dlg_ed[ie : ie + energy_batchsize, :, :],
            dll_sd[ie : ie + energy_batchsize, :, :],
            dll_ed[ie : ie + energy_batchsize, :, :],
            wr_p2w[ie : ie + energy_batchsize, :],
            wl_p2w[ie : ie + energy_batchsize, :],
            wg_p2w[ie : ie + energy_batchsize, :],
            dosw[ie : ie + energy_batchsize, :],
            new[ie : ie + energy_batchsize, :],
            npw[ie : ie + energy_batchsize, :],
            bmax_mm,
            bmin_mm,
            solve=True,
            input_stream=input_stream,
        )

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