import cupy as cp
import numpy as np

import cupy as cp
import cupyx as cpx
import numpy as np

from quatrex.block_tri_solvers.rgf_GF_GPU_combo import (
    _copy_csr_to_gpu,
    _get_dense_block_batch,
    _store_compressed,
    csr_matrix,
)


@cpx.jit.rawkernel()
def _get_coulomb_batch(
    vh_data,
    vh_indices,
    vh_indptr,
    out,
    batch_size,
    block_size,
):

    tid = cpx.jit.threadIdx.x
    if tid < block_size:
        num_threads = cpx.jit.blockDim.x
        bid = cpx.jit.blockIdx.x
        ie = bid // block_size
        ir = bid % block_size

        # NOTE: The buffer size here should ideally not be hardcoded.
        buf = cpx.jit.shared_memory(cp.complex128, 512)
        for i in range(tid, block_size, num_threads):
            buf[i] = 0
        cpx.jit.syncthreads()

        start = vh_indptr[ir]
        end = vh_indptr[ir + 1]
        i = start + tid
        while i < end:
            j = vh_indices[i]
            buf[j] += vh_data[i]
            i += num_threads
        cpx.jit.syncthreads()

        for i in range(tid, block_size, num_threads):
            out[ie, ir, i] = buf[i]


def rgf_batched_GPU(
    # Energy vector, dense format
    energies,
    # Mappings (NE, NNZ) format to (NB, NE, BS, BS) format
    map_diag_mm,
    map_upper_mm,
    map_lower_mm,
    map_diag_m,
    map_upper_m,
    map_lower_m,
    map_diag_l,
    map_upper_l,
    map_lower_l,
    # Coulomb matrix diagonals.
    vh_diag_host,
    vh_upper_host,
    vh_lower_host,
    # System matrix and self-energy in (NE, NNZ) format.
    mr_host,
    ll_host,
    lg_host,
    # Boundary conditions in dense format.
    dvh_left_host,
    dvh_right_host,
    dmr_left_host,
    dmr_right_host,
    dlg_left_host,
    dlg_right_host,
    dll_left_host,
    dll_right_host,
    # Output Green's Functions, (NE, NNZ) format.
    wg_host,
    wl_host,
    wr_host,
    # Output Observables.
    dosw,
    nEw,
    nPw,
    # Block indices.
    bmax,
    bmin,
    # Options.
    solve: bool = True,
    input_stream: cp.cuda.Stream = None,
):
    # start = time.time()
    # print(f"Starting RGF: {start}")

    # Sizes
    batch_size = len(energies)
    num_blocks = len(vh_diag_host)
    block_size = max(bmax - bmin + 1)
    dtype = np.complex128
    # hdtype = np.float64
    hdtype = np.complex128

    num_threads = min(1024, block_size)
    num_thread_blocks = batch_size * block_size

    map_diag_mm_dev = [cp.empty_like(mm) for mm in map_diag_mm]
    map_upper_mm_dev = [cp.empty_like(mm) for mm in map_upper_mm]
    map_lower_mm_dev = [cp.empty_like(mm) for mm in map_lower_mm]

    map_diag_m_dev = [cp.empty_like(m) for m in map_diag_m]
    map_upper_m_dev = [cp.empty_like(m) for m in map_upper_m]
    map_lower_m_dev = [cp.empty_like(m) for m in map_lower_m]

    map_diag_l_dev = [cp.empty_like(l) for l in map_diag_l]
    map_upper_l_dev = [cp.empty_like(l) for l in map_upper_l]
    map_lower_l_dev = [cp.empty_like(l) for l in map_lower_l]

    vd_batch = cp.empty((batch_size, block_size, block_size), dtype=dtype)
    vu_batch = cp.empty((batch_size, block_size, block_size), dtype=dtype)
    vl_batch = cp.empty((batch_size, block_size, block_size), dtype=dtype)

    vh_diag_buffer = [
        csr_matrix(
            cp.empty(block_size * block_size, hdtype),
            cp.empty(block_size * block_size, cp.int32),
            cp.empty(block_size + 1, cp.int32),
        )
        for _ in range(2)
    ]
    vh_upper_buffer = [
        csr_matrix(
            cp.empty(block_size * block_size, hdtype),
            cp.empty(block_size * block_size, cp.int32),
            cp.empty(block_size + 1, cp.int32),
        )
        for _ in range(2)
    ]
    vh_lower_buffer = [
        csr_matrix(
            cp.empty(block_size * block_size, hdtype),
            cp.empty(block_size * block_size, cp.int32),
            cp.empty(block_size + 1, cp.int32),
        )
        for _ in range(2)
    ]

    mr_diag_buffer = cp.empty((2, batch_size, block_size, block_size), dtype=dtype)
    mr_upper_buffer = cp.empty((2, batch_size, block_size, block_size), dtype=dtype)
    mr_lower_buffer = cp.empty((2, batch_size, block_size, block_size), dtype=dtype)
    ll_diag_buffer = cp.empty((2, batch_size, block_size, block_size), dtype=dtype)
    ll_upper_buffer = cp.empty((2, batch_size, block_size, block_size), dtype=dtype)
    ll_lower_buffer = cp.empty((2, batch_size, block_size, block_size), dtype=dtype)
    lg_diag_buffer = cp.empty((2, batch_size, block_size, block_size), dtype=dtype)
    lg_upper_buffer = cp.empty((2, batch_size, block_size, block_size), dtype=dtype)
    lg_lower_buffer = cp.empty((2, batch_size, block_size, block_size), dtype=dtype)

    computation_stream = cp.cuda.Stream.null
    input_stream = input_stream or cp.cuda.Stream(non_blocking=True)
    input_events = [cp.cuda.Event() for _ in range(2)]
    backward_events = [cp.cuda.Event() for _ in range(2)]

    xR_gpu = cp.empty(
        (num_blocks, batch_size, block_size, block_size), dtype=dtype
    )  # Retarded (right)
    wL_gpu = cp.empty(
        (num_blocks, batch_size, block_size, block_size), dtype=dtype
    )  # Lesser (right)
    wG_gpu = cp.empty(
        (num_blocks, batch_size, block_size, block_size), dtype=dtype
    )  # Greater (right)
    dll_gpu = cp.empty(
        (num_blocks - 1, batch_size, block_size, block_size), dtype=dtype
    )  # Lesser boundary self-energy
    dlg_gpu = cp.empty(
        (num_blocks - 1, batch_size, block_size, block_size), dtype=dtype
    )  # Greater boundary self-energy
    DOS_gpu = cp.empty((batch_size, num_blocks), dtype=dtype)
    nE_gpu = cp.empty((batch_size, num_blocks), dtype=dtype)
    nP_gpu = cp.empty((batch_size, num_blocks), dtype=dtype)
    # idE_gpu = cp.empty((batch_size, num_blocks), dtype=idE.dtype)

    XR_gpu = cp.empty(
        (2, batch_size, block_size, block_size), dtype=dtype
    )  # Retarded (right)
    XRnn1_gpu = cp.empty(
        (1, batch_size, block_size, block_size), dtype=dtype
    )  # Retarded (right)
    XRn1n_gpu = cp.empty(
        (2, batch_size, block_size, block_size), dtype=dtype
    )  # Retarded (right)
    WR_gpu = cp.empty(
        (2, batch_size, block_size, block_size), dtype=dtype
    )  # Retarded (right)
    WL_gpu = cp.empty(
        (2, batch_size, block_size, block_size), dtype=dtype
    )  # Lesser (right)
    WLnn1_gpu = cp.empty(
        (1, batch_size, block_size, block_size), dtype=dtype
    )  # Lesser (right)
    WG_gpu = cp.empty(
        (2, batch_size, block_size, block_size), dtype=dtype
    )  # Greater (right)
    WGnn1_gpu = cp.empty(
        (1, batch_size, block_size, block_size), dtype=dtype
    )  # Greater (right)

    WR_compressed = cp.zeros_like(wr_host)
    WL_compressed = cp.zeros_like(wl_host)
    WG_compressed = cp.zeros_like(wg_host)

    mr_dev = cp.empty_like(mr_host)
    ll_dev = cp.empty_like(ll_host)
    lg_dev = cp.empty_like(lg_host)
    dmr_dev = [None for _ in range(num_blocks)]
    dvh_dev = [None for _ in range(num_blocks)]
    dll_dev = [None for _ in range(num_blocks)]
    dlg_dev = [None for _ in range(num_blocks)]

    # med = time.time()
    # print(f"Memory allocation: {med - start}")

    # Backward pass IB \in {NB - 1, ..., 0}

    # First iteration IB = NB - 1
    IB = num_blocks - 1
    nIB = IB - 1
    idx = IB % 2
    nidx = nIB % 2
    NN = bmax[-1] - bmin[-1] + 1

    dmr_dev[0] = cp.asarray(dmr_left_host)
    dmr_dev[-1] = cp.asarray(dmr_right_host)

    dvh_dev[0] = cp.asarray(dvh_left_host)
    dvh_dev[-1] = cp.asarray(dvh_right_host)

    dll_dev[0] = cp.asarray(dll_left_host)
    dll_dev[-1] = cp.asarray(dll_right_host)
    dlg_dev[0] = cp.asarray(dlg_left_host)
    dlg_dev[-1] = cp.asarray(dlg_right_host)

    with input_stream:

        for i in range(num_blocks):
            map_diag_mm_dev[i].set(map_diag_mm[i])
            map_diag_m_dev[i].set(map_diag_m[i])
            map_diag_l_dev[i].set(map_diag_l[i])
            if i < num_blocks - 1:
                map_upper_mm_dev[i].set(map_upper_mm[i])
                map_upper_m_dev[i].set(map_upper_m[i])
                map_upper_l_dev[i].set(map_upper_l[i])
                map_lower_mm_dev[i].set(map_lower_mm[i])
                map_lower_m_dev[i].set(map_lower_m[i])
                map_lower_l_dev[i].set(map_lower_l[i])

        mr_dev.set(mr_host)
        ll_dev.set(ll_host)
        lg_dev.set(lg_host)
        input_events[idx].record(stream=input_stream)

        if num_blocks > 1:

            nmrd = mr_diag_buffer[nidx]
            nmru = mr_upper_buffer[nidx]
            nmrl = mr_lower_buffer[nidx]
            nlld = ll_diag_buffer[nidx]
            nlll = ll_lower_buffer[nidx]
            nlgd = lg_diag_buffer[nidx]
            nlgl = lg_lower_buffer[nidx]

            ndmr = dmr_dev[nIB]
            ndll = dll_dev[nIB]
            ndlg = dlg_dev[nIB]

            # NOTE: The next block can already be at the boundary!
            _get_dense_block_batch(mr_dev, map_diag_m_dev, nIB, nmrd, ndmr)
            _get_dense_block_batch(mr_dev, map_upper_m_dev, nIB, nmru)
            _get_dense_block_batch(mr_dev, map_lower_m_dev, nIB, nmrl)
            _get_dense_block_batch(ll_dev, map_diag_l_dev, nIB, nlld, ndll)
            _get_dense_block_batch(ll_dev, map_lower_l_dev, nIB, nlll)
            _get_dense_block_batch(lg_dev, map_diag_l_dev, nIB, nlgd, ndlg)
            _get_dense_block_batch(lg_dev, map_lower_l_dev, nIB, nlgl)

            input_events[nidx].record(stream=input_stream)



    computation_stream.wait_event(event=input_events[idx])

    mrd = mr_diag_buffer[idx]
    lld = ll_diag_buffer[idx]
    lgd = lg_diag_buffer[idx]

    xr = xR_gpu[IB]
    wl = wL_gpu[IB]
    wg = wG_gpu[IB]

    dmr = dmr_dev[IB]
    dll = dll_dev[IB]
    dlg = dlg_dev[IB]

    _get_dense_block_batch(mr_dev, map_diag_m_dev, IB, mrd, dmr)
    _get_dense_block_batch(ll_dev, map_diag_l_dev, IB, lld, dll)
    _get_dense_block_batch(lg_dev, map_diag_l_dev, IB, lgd, dlg)

    if solve:
        gpu_identity = cp.identity(NN, dtype=mrd.dtype)
        gpu_identity_batch = cp.repeat(
            gpu_identity[cp.newaxis, :, :], batch_size, axis=0
        )
        xr[:, 0:NN, 0:NN] = cp.linalg.solve(mrd[:, 0:NN, 0:NN], gpu_identity_batch)
    else:
        xr[:, 0:NN, 0:NN] = cp.linalg.inv(mrd[:, 0:NN, 0:NN])
    xr_h = cp.conjugate(xr[:, 0:NN, 0:NN].transpose((0, 2, 1)))
    cp.matmul(
        xr[:, 0:NN, 0:NN] @ lld[:, 0:NN, 0:NN],
        xr_h[:, 0:NN, 0:NN],
        out=wl[:, 0:NN, 0:NN],
    )
    cp.matmul(
        xr[:, 0:NN, 0:NN] @ lgd[:, 0:NN, 0:NN],
        xr_h[:, 0:NN, 0:NN],
        out=wg[:, 0:NN, 0:NN],
    )

    backward_events[idx].record(stream=computation_stream)

    # Rest iterations IB \in {NB - 2, ..., 0}
    for IB in range(num_blocks - 2, -1, -1):

        pIB = IB + 1
        nIB = IB - 1
        idx = IB % 2
        nidx = nIB % 2
        pidx = pIB % 2
        NI = bmax[IB] - bmin[IB] + 1
        NP = bmax[IB + 1] - bmin[IB + 1] + 1

        with input_stream:

            input_stream.wait_event(event=backward_events[pidx])
            if nIB >= 0:

                nmrd = mr_diag_buffer[nidx]
                nmru = mr_upper_buffer[nidx]
                nmrl = mr_lower_buffer[nidx]
                nlld = ll_diag_buffer[nidx]
                nlll = ll_lower_buffer[nidx]
                nlgd = lg_diag_buffer[nidx]
                nlgl = lg_lower_buffer[nidx]

                ndmr = dmr_dev[nIB]
                ndll = dll_dev[nIB]
                ndlg = dlg_dev[nIB]

                _get_dense_block_batch(mr_dev, map_diag_m_dev, nIB, nmrd, ndmr)
                _get_dense_block_batch(mr_dev, map_upper_m_dev, nIB, nmru)
                _get_dense_block_batch(mr_dev, map_lower_m_dev, nIB, nmrl)
                _get_dense_block_batch(ll_dev, map_diag_l_dev, nIB, nlld, ndll)
                _get_dense_block_batch(ll_dev, map_lower_l_dev, nIB, nlll)
                _get_dense_block_batch(lg_dev, map_diag_l_dev, nIB, nlgd, ndlg)
                _get_dense_block_batch(lg_dev, map_lower_l_dev, nIB, nlgl)

            else:  # nIB < 0

                nllu = ll_upper_buffer[idx]
                nlgu = lg_upper_buffer[idx]

                _get_dense_block_batch(ll_dev, map_upper_l_dev, IB, nllu)
                _get_dense_block_batch(lg_dev, map_upper_l_dev, IB, nlgu)

                _copy_csr_to_gpu(vh_diag_host[IB], vh_diag_buffer[idx])
                _copy_csr_to_gpu(vh_lower_host[IB], vh_lower_buffer[idx])

            input_events[nidx].record(stream=input_stream)

        mrd = mr_diag_buffer[idx]
        mru = mr_upper_buffer[idx]
        mrl = mr_lower_buffer[idx]
        lld = ll_diag_buffer[idx]
        lll = ll_lower_buffer[idx]
        lgd = lg_diag_buffer[idx]
        lgl = lg_lower_buffer[idx]

        xr = xR_gpu[IB]
        pxr = xR_gpu[IB + 1]
        wl = wL_gpu[IB]
        pwl = wL_gpu[IB + 1]
        wg = wG_gpu[IB]
        pwg = wG_gpu[IB + 1]
        dmr = dmr_dev[IB]
        dll = dll_gpu[IB]
        dlg = dlg_gpu[IB]

        if IB == 0:
            xr = XR_gpu[0]
            wl = WL_gpu[0]
            wg = WG_gpu[0]

        computation_stream.wait_event(event=input_events[idx])

        mru_h = cp.conjugate(mru[:, 0:NI, 0:NP].transpose((0, 2, 1)))
        mru_x_pxr = mru[:, 0:NI, 0:NP] @ pxr[:, 0:NP, 0:NP]
        al = mru_x_pxr @ lll[:, 0:NP, 0:NI]
        ag = mru_x_pxr @ lgl[:, 0:NP, 0:NI]
        inv_arg = mrd[:, 0:NI, 0:NI] - mru_x_pxr @ mrl[:, :NP, 0:NI]
        if solve:
            gpu_identity = cp.identity(NI, dtype=inv_arg.dtype)
            gpu_identity_batch = cp.repeat(
                gpu_identity[cp.newaxis, :, :], batch_size, axis=0
            )
            xr[:, 0:NI, 0:NI] = cp.linalg.solve(inv_arg, gpu_identity_batch)
        else:
            xr[:, 0:NI, 0:NI] = cp.linalg.inv(inv_arg)
        xr_h = cp.conjugate(xr[:, 0:NI, 0:NI].transpose((0, 2, 1)))
        cp.subtract(
            mru[:, 0:NI, 0:NP] @ pwl[:, 0:NP, 0:NP] @ mru_h[:, 0:NP, 0:NI],
            al - cp.conjugate(al.transpose((0, 2, 1))),
            out=dll[:, 0:NI, 0:NI],
        )  # SLB must change
        cp.subtract(
            mru[:, 0:NI, 0:NP] @ pwg[:, 0:NP, 0:NP] @ mru_h[:, 0:NP, 0:NI],
            ag - cp.conjugate(ag.transpose((0, 2, 1))),
            out=dlg[:, 0:NI, 0:NI],
        )  # SGB must change
        cp.matmul(
            xr[:, 0:NI, 0:NI] @ (lld[:, 0:NI, 0:NI] + dll[:, 0:NI, 0:NI]),
            xr_h[:, 0:NI, 0:NI],
            out=wl[:, 0:NI, 0:NI],
        )
        cp.matmul(
            xr[:, 0:NI, 0:NI] @ (lgd[:, 0:NI, 0:NI] + dlg[:, 0:NI, 0:NI]),
            xr_h[:, 0:NI, 0:NI],
            out=wg[:, 0:NI, 0:NI],
        )

        backward_events[idx].record(stream=computation_stream)

    # return xr.get(), wl.get(), wg.get()
    
    # Forward pass

    # First iteration
    IB = 0
    nIB = IB + 1
    idx = IB % 2
    nidx = nIB % 2

    with input_stream:

        input_stream.wait_event(event=backward_events[idx])
        if nIB < num_blocks:

            npmru = mr_upper_buffer[nidx]
            npmrl = mr_lower_buffer[nidx]

            nplll = ll_lower_buffer[nidx]
            nplgl = lg_lower_buffer[nidx]

            _get_dense_block_batch(mr_dev, map_upper_m_dev, IB, npmru)
            _get_dense_block_batch(mr_dev, map_lower_m_dev, IB, npmrl)

            _get_dense_block_batch(ll_dev, map_lower_l_dev, IB, nplll)
            _get_dense_block_batch(lg_dev, map_lower_l_dev, IB, nplgl)

            _copy_csr_to_gpu(vh_upper_host[IB], vh_upper_buffer[nidx])
            _copy_csr_to_gpu(vh_diag_host[nIB], vh_diag_buffer[nidx])

            if nIB < num_blocks - 1:

                nmru = mr_diag_buffer[nidx]  # NOTE: This is not a mistake.
                nmrl = ll_diag_buffer[nidx]
                nllu = ll_upper_buffer[nidx]
                nlgu = lg_upper_buffer[nidx]

                _get_dense_block_batch(mr_dev, map_upper_m_dev, nIB, nmru)
                _get_dense_block_batch(mr_dev, map_lower_m_dev, nIB, nmrl)
                _get_dense_block_batch(ll_dev, map_upper_l_dev, nIB, nllu)
                _get_dense_block_batch(lg_dev, map_upper_l_dev, nIB, nlgu)

                _copy_csr_to_gpu(vh_lower_host[nIB], vh_lower_buffer[nidx])

            input_events[idx].record(stream=input_stream)

    mru = mr_upper_buffer[idx]
    mrl = mr_lower_buffer[idx]

    llu = ll_upper_buffer[idx]
    lgu = lg_upper_buffer[idx]

    XR = XR_gpu[IB]
    WR = WR_gpu[IB]
    WL = WL_gpu[IB]
    WG = WG_gpu[IB]
    XRnn1 = XRnn1_gpu[0]
    XRn1n = XRn1n_gpu[idx]
    WLnn1 = WLnn1_gpu[0]
    WGnn1 = WGnn1_gpu[0]

    # NOTE: These were written directly to output in the last iteration of the backward pass
    xr = XR_gpu[IB]
    wl = WL_gpu[IB]
    wg = WG_gpu[IB]

    pxr = xR_gpu[nIB]
    # pwr = wR_gpu[nIB]
    pwl = wL_gpu[nIB]
    pwg = wG_gpu[nIB]

    dvh = dvh_dev[IB]

    vhd = vh_diag_buffer[idx]
    vhl = vh_lower_buffer[idx]

    computation_stream.wait_event(event=input_events[nidx])

    # _csr_to_dense(vhd.data, vhd.indices, vhd.indptr, vd, block_size)
    # _csr_to_dense(vhl.data, vhl.indices, vhl.indptr, vl, block_size)
    # vd_batch = cp.repeat(vd[cp.newaxis, :, :], batch_size, axis=0)
    # vl_batch = cp.repeat(vl[cp.newaxis, :, :], batch_size, axis=0)

    _get_coulomb_batch[num_thread_blocks, num_threads](
        vhd.data, vhd.indices, vhd.indptr, vd_batch, batch_size, block_size
    )
    computation_stream.synchronize()
    if dvh is not None:
        vd_batch -= dvh

    _get_coulomb_batch[num_thread_blocks, num_threads](
        vhl.data, vhl.indices, vhl.indptr, vl_batch, batch_size, block_size
    )

    xr_h = cp.conjugate(pxr[:, 0:NP, 0:NP].transpose((0, 2, 1)))
    mrl_h = cp.conjugate(mrl[:, 0:NP, 0:NI].transpose((0, 2, 1)))
    xr_x_mru = xr[:, 0:NI, 0:NI] @ mru[:, 0:NI, 0:NP]
    mrl_h_x_gr_h = mrl_h @ xr_h[:, 0:NP, 0:NP]
    # NOTE: These were written in the last iteration of the backward pass
    # GR[:] = gr
    # GL[:] = gl
    # GG[:] = gg
    cp.negative(xr_x_mru @ pxr[:, 0:NP, 0:NP], out=XRnn1[:, 0:NI, 0:NP])
    cp.negative(
        pxr[:, :NP, :NP] @ mrl[:, :NP, 0:NI] @ xr[:, :NI, :NI], out=XRn1n[:, 0:NP, 0:NI]
    )

    cp.subtract(
        xr[:, 0:NI, 0:NI] @ llu[:, 0:NI, 0:NP] @ xr_h[:, 0:NP, 0:NP]
        - xr_x_mru @ pwl[:, 0:NP, 0:NP],
        wl[:, 0:NI, 0:NI] @ mrl_h_x_gr_h,
        out=WLnn1[:, 0:NI, 0:NP],
    )
    cp.subtract(
        xr[:, 0:NI, 0:NI] @ lgu[:, 0:NI, 0:NP] @ xr_h[:, 0:NP, 0:NP]
        - xr_x_mru @ pwg[:, 0:NP, 0:NP],
        wg[:, 0:NI, 0:NI] @ mrl_h_x_gr_h,
        out=WGnn1[:, 0:NI, 0:NP],
    )

    cp.add(
        xr[:, 0:NI, 0:NI] @ vd_batch[:, 0:NI, 0:NI],
        XRnn1[:, 0:NI, 0:NP] @ vl_batch[:, 0:NP, 0:NI],
        out=WR[:, 0:NI, 0:NI],
    )

    WR_h = cp.conjugate(WR[:, 0:NI, 0:NI].transpose((0, 2, 1)))
    DOS_gpu[:, 0] = 1j * cp.trace(
        WR[:, 0:NI, 0:NI] - WR_h[:, 0:NI, 0:NI], axis1=1, axis2=2
    )
    nE_gpu[:, 0] = -1j * cp.trace(wl[:, 0:NI, 0:NI], axis1=1, axis2=2)
    nP_gpu[:, 0] = 1j * cp.trace(wg[:, 0:NI, 0:NI], axis1=1, axis2=2)

    backward_events[nidx].record(stream=computation_stream)

    computation_stream.synchronize()
    _store_compressed(
        map_diag_mm_dev,
        map_upper_mm_dev,
        map_lower_mm_dev,
        WR,
        None,
        IB,
        WR_compressed,
    )
    _store_compressed(
        map_diag_mm_dev,
        map_upper_mm_dev,
        map_lower_mm_dev,
        WL,
        WLnn1,
        IB,
        WL_compressed,
    )
    _store_compressed(
        map_diag_mm_dev,
        map_upper_mm_dev,
        map_lower_mm_dev,
        WG,
        WGnn1,
        IB,
        WG_compressed,
    )

    # Rest iterations
    for IB in range(1, num_blocks):

        pIB = IB - 1
        nIB = IB + 1
        idx = IB % 2
        pidx = pIB % 2
        nidx = nIB % 2
        NI = bmax[IB] - bmin[IB] + 1
        NM = bmax[IB - 1] - bmin[IB - 1] + 1

        with input_stream:

            input_stream.wait_event(event=backward_events[idx])
            if nIB < num_blocks:

                npmru = mr_upper_buffer[idx]
                npmrl = mr_lower_buffer[idx]

                npmru[:] = mr_diag_buffer[idx]
                npmrl[:] = ll_diag_buffer[idx]

                nplll = ll_lower_buffer[idx]
                nplgl = lg_lower_buffer[idx]

                _get_dense_block_batch(ll_dev, map_lower_l_dev, IB, nplll)
                _get_dense_block_batch(lg_dev, map_lower_l_dev, IB, nplgl)

                _copy_csr_to_gpu(vh_upper_host[IB], vh_upper_buffer[nidx])

                _copy_csr_to_gpu(vh_diag_host[nIB], vh_diag_buffer[nidx])

                if nIB < num_blocks - 1:

                    nmru = mr_diag_buffer[nidx]
                    nmrl = ll_diag_buffer[nidx]
                    nllu = ll_upper_buffer[nidx]
                    nlgu = lg_upper_buffer[nidx]

                    _get_dense_block_batch(mr_dev, map_upper_m_dev, nIB, nmru)
                    _get_dense_block_batch(mr_dev, map_lower_m_dev, nIB, nmrl)
                    _get_dense_block_batch(ll_dev, map_upper_l_dev, nIB, nllu)
                    _get_dense_block_batch(lg_dev, map_upper_l_dev, nIB, nlgu)

                    _copy_csr_to_gpu(vh_lower_host[nIB], vh_lower_buffer[nidx])

            input_events[idx].record(stream=input_stream)

        pmru = mr_upper_buffer[pidx]
        pmrl = mr_lower_buffer[pidx]
        plll = ll_lower_buffer[pidx]
        plgl = lg_lower_buffer[pidx]

        XR = XR_gpu[idx]
        pXR = XR_gpu[pidx]
        pXRn1n = XRn1n_gpu[pidx]
        WR = WR_gpu[idx]
        WL = WL_gpu[idx]
        pWL = WL_gpu[pidx]
        WG = WG_gpu[idx]
        pWG = WG_gpu[pidx]

        xr = xR_gpu[IB]
        wl = wL_gpu[IB]
        wg = wG_gpu[IB]

        dvh = dvh_dev[IB]

        vhd = vh_diag_buffer[idx]
        vhu = vh_upper_buffer[idx]
        vhl = vh_lower_buffer[idx]

        computation_stream.wait_event(event=input_events[pidx])

        _get_coulomb_batch[num_thread_blocks, num_threads](
            vhd.data, vhd.indices, vhd.indptr, vd_batch, batch_size, block_size
        )

        if dvh is not None:
            vd_batch -= dvh

        _get_coulomb_batch[num_thread_blocks, num_threads](
            vhu.data, vhu.indices, vhu.indptr, vu_batch, batch_size, block_size
        )

        if nIB < num_blocks - 1:
            _get_coulomb_batch[num_thread_blocks, num_threads](
                vhl.data, vhl.indices, vhl.indptr, vl_batch, batch_size, block_size
            )

        computation_stream.synchronize()

        pXRh = cp.conjugate(pXR[:, 0:NM, 0:NM].transpose((0, 2, 1)))
        pmrlh = cp.conjugate(pmrl[:, 0:NI, 0:NM].transpose((0, 2, 1)))
        xrh = cp.conjugate(xr[:, 0:NI, 0:NI].transpose((0, 2, 1)))
        xrpmrl = xr[:, 0:NI, 0:NI] @ pmrl[:, 0:NI, 0:NM]
        xrpmrlpXRpmru = xrpmrl @ pXR[:, 0:NM, 0:NM] @ pmru[:, 0:NM, 0:NI]
        pmrlhxrh = pmrlh @ xrh
        pXRhpmrlhxrh = pXRh @ pmrlhxrh
        al = xr[:, 0:NI, 0:NI] @ plll[:, 0:NI, 0:NI] @ pXRhpmrlhxrh
        bl = xrpmrlpXRpmru @ wl[:, 0:NI, 0:NI]
        ag = xr[:, 0:NI, 0:NI] @ plgl[:, 0:NI, 0:NM] @ pXRhpmrlhxrh
        bg = xrpmrlpXRpmru @ wg[:, 0:NI, 0:NI]
        cp.add(
            xr[:, 0:NI, 0:NI], xrpmrlpXRpmru @ xr[:, 0:NI, 0:NI], out=XR[:, 0:NI, 0:NI]
        )

        cp.subtract(
            wl[:, 0:NI, 0:NI] + xrpmrl @ pWL[:, 0:NM, 0:NM] @ pmrlhxrh,
            al
            - cp.conjugate(al.transpose((0, 2, 1)))
            - bl
            + cp.conjugate(bl.transpose((0, 2, 1))),
            out=WL[:, 0:NI, 0:NI],
        )
        cp.subtract(
            wg[:, 0:NI, 0:NI] + xrpmrl @ pWG[:, 0:NM, 0:NM] @ pmrlhxrh,
            ag
            - cp.conjugate(ag.transpose((0, 2, 1)))
            - bg
            + cp.conjugate(bg.transpose((0, 2, 1))),
            out=WG[:, 0:NI, 0:NI],
        )

        if IB < num_blocks - 1:

            NP = bmax[IB + 1] - bmin[IB + 1] + 1

            XRnn1 = XRnn1_gpu[0]
            XRn1n = XRn1n_gpu[idx]
            WLnn1 = WLnn1_gpu[0]
            WGnn1 = WGnn1_gpu[0]

            nxr = xR_gpu[nIB]
            nwl = wL_gpu[nIB]
            nwg = wG_gpu[nIB]

            mru = mr_diag_buffer[idx]  # NOTE: This is not a mistake.
            mrl = ll_diag_buffer[idx]
            llu = ll_upper_buffer[idx]
            lgu = lg_upper_buffer[idx]

            nxrh = cp.conjugate(nxr[:, 0:NP, 0:NP].transpose((0, 2, 1)))
            mrlh = cp.conjugate(mrl[:, 0:NP, 0:NI].transpose((0, 2, 1)))
            XRhu = XR[:, 0:NI, 0:NI] @ mru[:, 0:NI, 0:NP]
            mrlhnxrh = mrlh @ nxrh
            cp.negative(
                XR[:, 0:NI, 0:NI] @ mru[:, 0:NI, 0:NP] @ nxr[:, 0:NP, 0:NP],
                out=XRnn1[:, 0:NI, 0:NP],
            )

            cp.negative(
                nxr[:, 0:NP, 0:NP] @ mrl[:, 0:NP, 0:NI] @ XR[:, 0:NI, 0:NI],
                out=XRn1n[:, 0:NP, 0:NI],
            )

            cp.subtract(
                XR[:, 0:NI, 0:NI] @ llu[:, 0:NI, 0:NP] @ nxrh,
                XRhu @ nwl[:, 0:NP, 0:NP] + WL[:, 0:NI, 0:NI] @ mrlhnxrh,
                out=WLnn1[:, 0:NI, 0:NP],
            )
            cp.subtract(
                XR[:, 0:NI, 0:NI] @ lgu[:, 0:NI, 0:NP] @ nxrh,
                XRhu @ nwg[:, 0:NP, 0:NP] + WG[:, 0:NI, 0:NI] @ mrlhnxrh,
                out=WGnn1[:, 0:NI, 0:NP],
            )

            cp.add(
                pXRn1n[:, 0:NI, 0:NP] @ vu_batch[:, 0:NP, 0:NI]
                + XRnn1[:, 0:NI, 0:NP] @ vl_batch[:, 0:NP, 0:NI],
                XR[:, 0:NI, 0:NI] @ vd_batch[:, 0:NI, 0:NI],
                out=WR[:, 0:NI, 0:NI],
            )

            backward_events[idx].record(stream=computation_stream)

            computation_stream.synchronize()
            _store_compressed(
                map_diag_mm_dev,
                map_upper_mm_dev,
                map_lower_mm_dev,
                WL,
                WLnn1,
                IB,
                WL_compressed,
            )
            _store_compressed(
                map_diag_mm_dev,
                map_upper_mm_dev,
                map_lower_mm_dev,
                WG,
                WGnn1,
                IB,
                WG_compressed,
            )

            _store_compressed(
                map_diag_mm_dev,
                map_upper_mm_dev,
                map_lower_mm_dev,
                WR,
                None,
                IB,
                WR_compressed,
            )

        else:

            cp.add(
                pXRn1n[:, 0:NI, 0:NP] @ vu_batch[:, 0:NP, 0:NI],
                XR[:, 0:NI, 0:NI] @ vd_batch[:, 0:NI, 0:NI],
                out=WR[:, 0:NI, 0:NI],
            )

            computation_stream.synchronize()
            backward_events[idx].record(stream=computation_stream)

        DOS_gpu[:, IB] = 1j * cp.trace(
            WR[:, 0:NI, 0:NI] - cp.conjugate(WR[:, 0:NI, 0:NI].transpose(0, 2, 1)),
            axis1=1,
            axis2=2,
        )
        nE_gpu[:, IB] = -1j * cp.trace(WL[:, 0:NI, 0:NI], axis1=1, axis2=2)
        nP_gpu[:, IB] = 1j * cp.trace(WG[:, 0:NI, 0:NI], axis1=1, axis2=2)

    computation_stream.synchronize()
    # return WG_compressed.get()
    with input_stream:
        _store_compressed(
            map_diag_mm_dev,
            map_upper_mm_dev,
            map_lower_mm_dev,
            WL,
            None,
            num_blocks - 1,
            WL_compressed,
        )
        _store_compressed(
            map_diag_mm_dev,
            map_upper_mm_dev,
            map_lower_mm_dev,
            WG,
            None,
            num_blocks - 1,
            WG_compressed,
        )
        WL_compressed.get(out=wl_host)
        WG_compressed.get(out=wg_host)
    _store_compressed(
        map_diag_mm_dev,
        map_upper_mm_dev,
        map_lower_mm_dev,
        WR,
        None,
        num_blocks - 1,
        WR_compressed,
    )
    WR_compressed.get(out=wr_host)

    computation_stream.synchronize()
    DOS_gpu.get(out=dosw)
    nE_gpu.get(out=nEw)
    nP_gpu.get(out=nPw)
    input_stream.synchronize()
    computation_stream.synchronize()

    # end = time.time()
    # print(f"Computation: {end - med}")
