import cupy as cp
import cupyx as cpx
import numpy as np
import scipy.sparse as sp

from collections import namedtuple
csr_matrix = namedtuple('csr_matrix', ['data', 'indices', 'indptr'])


def map_to_mapping(map, num_blocks):
    mapping = [None for _ in range(num_blocks)]
    for block_id in range(num_blocks):
        block_indices = np.nonzero(map[0] == block_id)[0]
        block_map = cpx.empty_pinned((3, len(block_indices)), dtype=np.int32)
        block_map[0] = map[3][block_indices]  # data indices
        block_map[1] = map[1][block_indices]  # rows
        block_map[2] = map[2][block_indices]  # cols
        mapping[block_id] = block_map
    return mapping


def canonicalize_csr(csr: sp.csr_matrix):
    result = csr
    if not isinstance(csr, sp.csr_matrix):
        result = csr.tocsr()
    result.eliminate_zeros()
    result.sum_duplicates()
    result.sort_indices()
    result.has_canonical_format = True
    return result


def _set_block_pinned(block) -> csr_matrix:
    data = cpx.empty_like_pinned(block.data)
    indices = cpx.empty_like_pinned(block.indices)
    indptr = cpx.empty_like_pinned(block.indptr)
    data[:] = block.data
    indices[:] = block.indices
    indptr[:] = block.indptr
    return csr_matrix(data, indices, indptr)


def csr_to_block_tridiagonal_csr(csr: sp.csr_matrix, bmin, bmax):
    num_blocks = len(bmin)
    block_diag = [None for _ in range(num_blocks)]
    block_upper = [None for _ in range(num_blocks - 1)]
    block_lower = [None for _ in range(num_blocks - 1)]

    for block_id in range(num_blocks):
        diag_slice = slice(bmin[block_id], bmax[block_id])
        block_diag[block_id] = _set_block_pinned(canonicalize_csr(csr[diag_slice, diag_slice]))
        if block_id < num_blocks - 1:
            upper_slice = lower_slice = slice(bmin[block_id + 1], bmax[block_id + 1])
            block_upper[block_id] = _set_block_pinned(canonicalize_csr(csr[diag_slice, upper_slice]))
            block_lower[block_id] = _set_block_pinned(canonicalize_csr(csr[lower_slice, diag_slice]))

    return block_diag, block_upper, block_lower


def self_energy_preprocess_2d(sl, sg, sr, sl_phn, sg_phn, sr_phn, rows, columns, ij2ji):
    sl[:] = (sl - sl[:, ij2ji].conj()) / 2
    sg[:] = (sg - sg[:, ij2ji].conj()) / 2
    sr[:] = np.real(sr) + (sg - sl) / 2

    sl[:, rows == columns] += sl_phn
    sg[:, rows == columns] += sg_phn
    sr[:, rows == columns] += sr_phn


def _copy_csr_to_gpu(csr, csr_matrix):
    nnz = csr.data.size
    csr_matrix.data[:nnz].set(csr.data)
    csr_matrix.indices[:nnz].set(csr.indices)
    csr_matrix.indptr.set(csr.indptr)


def _get_dense_block_batch(compressed_data,  # Input data, (NE, NNZ) format
                           mapping,  # Mapping (NE, NNZ) format to (NB, NE, BS, BS) format
                           block_idx, # Block index
                           uncompressed_data,  # Output data, (NE, BS, BS) format
                           add_block=None  # Additional block to add
                           ):
    block_data_indices = mapping[block_idx][0]
    block_rows = mapping[block_idx][1]
    block_cols = mapping[block_idx][2]
    uncompressed_data[:] = 0
    uncompressed_data[:, block_rows, block_cols] = compressed_data[:, block_data_indices]
    if add_block is not None:
        uncompressed_data += add_block


@cpx.jit.rawkernel()
def _store_block_compressed(mapping, uncompressed, compressed, batch_size, copy_size, nnz, bsize, lower):
    tid = cpx.jit.blockIdx.x * cpx.jit.blockDim.x + cpx.jit.threadIdx.x
    if tid < batch_size * copy_size:
        ie = tid // copy_size
        idx = tid % copy_size
        out = mapping[0][idx]
        row = mapping[1][idx]
        col = mapping[2][idx]
        out_idx = ie * nnz + out
        if lower:
            inp_idx = ie * bsize * bsize + col * bsize + row
            compressed[out_idx] = -uncompressed[inp_idx]
        else:
            inp_idx = ie * bsize * bsize + row * bsize + col
            compressed[out_idx] = uncompressed[inp_idx]


def _store_compressed(mapping_diag, mapping_upper, mapping_lower,
                      uncompressed_diag, uncompressed_upper,
                      block_idx,
                      compressed_data):

    batch_size = uncompressed_diag.shape[0]
    copy_size = mapping_diag[block_idx][0].size
    nnz = compressed_data.shape[1]
    bsize = uncompressed_diag.shape[1]
    num_threads = 1024
    num_blocks = (batch_size * copy_size + num_threads - 1) // num_threads
    _store_block_compressed[num_blocks, num_threads](mapping_diag[block_idx],
                                                     uncompressed_diag.reshape(-1),
                                                     compressed_data.reshape(-1),
                                                     batch_size, copy_size, nnz, bsize, False)
    

    if uncompressed_upper is not None:

        block_data_indices = mapping_upper[block_idx][0]
        copy_size = mapping_upper[block_idx][0].size
        num_blocks = (batch_size * copy_size + num_threads - 1) // num_threads
        _store_block_compressed[num_blocks, num_threads](mapping_upper[block_idx],
                                                         uncompressed_upper.reshape(-1),
                                                         compressed_data.reshape(-1),
                                                         batch_size, copy_size, nnz, bsize, False)
        
        block_data_indices = mapping_lower[block_idx][0]
        copy_size = mapping_lower[block_idx][0].size
        num_blocks = (batch_size * copy_size + num_threads - 1) // num_threads
        _store_block_compressed[num_blocks, num_threads](mapping_lower[block_idx],
                                                         uncompressed_upper.conj().reshape(-1),
                                                         compressed_data.reshape(-1),
                                                         batch_size, copy_size, nnz, bsize, True)


@cpx.jit.rawkernel()
def _get_system_matrix(energies,
                       H_data, H_indices, H_indptr,
                       S_data, S_indices, S_indptr,
                       SR,
                       out,
                       batch_size, block_size):
    
    tid = cpx.jit.threadIdx.x
    if tid < block_size:

        num_threads = cpx.jit.blockDim.x
        bid = cpx.jit.blockIdx.x
        ie = bid // block_size
        ir = bid % block_size

        energy = energies[ie]
         
        buf = cpx.jit.shared_memory(cp.complex128, 416)
        for i in range(tid, block_size, num_threads):
            buf[i] = 0
        cpx.jit.syncthreads()

        start = S_indptr[ir] 
        end = S_indptr[ir + 1]
        i = start + tid
        while i < end:
            j = S_indices[i]
            buf[j] += energy * S_data[i]
            i += num_threads
        cpx.jit.syncthreads()

        start = H_indptr[ir]
        end = H_indptr[ir + 1]
        i = start + tid
        while i < end:
            j = H_indices[i]
            buf[j] -= H_data[i]
            i += num_threads
        cpx.jit.syncthreads()

        for i in range(tid, block_size, num_threads):
            out[ie, ir, i] = buf[i] - SR[ie, ir, i]


def rgf_batched_GPU(energies,  # Energy vector, dense format
                    map_diag, map_upper, map_lower,  # Mapping (NE, NNZ) format to (NB, NE, BS, BS) format
                    H_diag_host, H_upper_host, H_lower_host,  # Hamiltonian matrix, CSR format
                    S_diag_host, S_upper_host, S_lower_host,  # Overlap matrix, CSR format
                    SR_host, SL_host, SG_host,  # Retarded, Lesser, Greater self-energy, (NE, NNZ) format
                    SigRB_left_host, SigRB_right_host,  # Retarded boundary conditions, dense format
                    SigLB_left_host, SigLB_right_host,  # Lesser boundary conditions, dense format
                    SigGB_left_host, SigGB_right_host,  # Greater boundary conditions, dense format
                    GR_host, GL_host, GG_host,  # Output Green's Functions, (NE, NNZ) format
                    DOS, nE, nP, idE,  # Output Observables
                    Bmin_fi, Bmax_fi,  # Indices
                    solve: bool = True,
                    input_stream: cp.cuda.Stream = None
                    ):
    
    # start = time.time()
    # print(f"Starting RGF: {start}")

    # Sizes
    # Why are subtracing by 1 every time? Fix 0-based indexing
    Bmax = Bmax_fi - 1
    Bmin = Bmin_fi - 1
    batch_size = len(energies)
    num_blocks = len(H_diag_host)
    block_size = max(Bmax - Bmin + 1)
    dtype = np.complex128
    #hdtype = np.float64
    hdtype = np.complex128

    num_threads = min(1024, block_size)
    num_thread_blocks = batch_size * block_size

    map_diag_dev = [cp.empty_like(m) if isinstance(m, np.ndarray) else None for m in map_diag]
    map_upper_dev = [cp.empty_like(m) if isinstance(m, np.ndarray) else None for m in map_upper]
    map_lower_dev = [cp.empty_like(m) if isinstance(m, np.ndarray) else None for m in map_lower]

    md = cp.empty((batch_size, block_size, block_size), dtype=dtype)
    mu = cp.empty((batch_size, block_size, block_size), dtype=dtype)
    ml = cp.empty((batch_size, block_size, block_size), dtype=dtype)


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
    

    prev_H_upper_buffer = [csr_matrix(cp.empty(block_size * block_size, hdtype),
                                      cp.empty(block_size * block_size, cp.int32),
                                      cp.empty(block_size + 1, cp.int32),) for _ in range(2)]
    prev_H_lower_buffer = [csr_matrix(cp.empty(block_size * block_size, hdtype),
                                      cp.empty(block_size * block_size, cp.int32),
                                      cp.empty(block_size + 1, cp.int32),) for _ in range(2)]
    prev_S_upper_buffer = [csr_matrix(cp.empty(block_size * block_size, hdtype),
                                      cp.empty(block_size * block_size, cp.int32),
                                      cp.empty(block_size + 1, cp.int32),) for _ in range(2)]
    prev_S_lower_buffer = [csr_matrix(cp.empty(block_size * block_size, hdtype),
                                      cp.empty(block_size * block_size, cp.int32),
                                      cp.empty(block_size + 1, cp.int32),) for _ in range(2)]

    SR_diag_buffer = cp.empty((2, batch_size, block_size, block_size), dtype=dtype)
    SR_upper_buffer = cp.empty((2, batch_size, block_size, block_size), dtype=dtype)
    SR_lower_buffer = cp.empty((2, batch_size, block_size, block_size), dtype=dtype)
    SL_diag_buffer = cp.empty((2, batch_size, block_size, block_size), dtype=dtype)
    SL_upper_buffer = cp.empty((2, batch_size, block_size, block_size), dtype=dtype)
    SL_lower_buffer = cp.empty((2, batch_size, block_size, block_size), dtype=dtype)
    SG_diag_buffer = cp.empty((2, batch_size, block_size, block_size), dtype=dtype)
    SG_upper_buffer = cp.empty((2, batch_size, block_size, block_size), dtype=dtype)
    SG_lower_buffer = cp.empty((2, batch_size, block_size, block_size), dtype=dtype)

    computation_stream = cp.cuda.Stream.null
    input_stream = input_stream or cp.cuda.Stream(non_blocking=True)
    input_events = [cp.cuda.Event() for _ in range(2)]
    backward_events = [cp.cuda.Event() for _ in range(2)]

    gR_gpu = cp.empty((num_blocks, batch_size, block_size, block_size), dtype=dtype)  # Retarded (right)
    gL_gpu = cp.empty((num_blocks, batch_size, block_size, block_size), dtype=dtype)  # Lesser (right)
    gG_gpu = cp.empty((num_blocks, batch_size, block_size, block_size), dtype=dtype)  # Greater (right)
    SigLB_gpu = cp.empty((num_blocks-1, batch_size, block_size, block_size), dtype=dtype)  # Lesser boundary self-energy
    SigGB_gpu = cp.empty((num_blocks-1, batch_size, block_size, block_size), dtype=dtype)  # Greater boundary self-energy
    DOS_gpu = cp.empty((batch_size, num_blocks), dtype=dtype) if isinstance(DOS, np.ndarray) else DOS
    nE_gpu = cp.empty((batch_size, num_blocks), dtype=dtype) if isinstance(nE, np.ndarray) else nE
    nP_gpu = cp.empty((batch_size, num_blocks), dtype=dtype) if isinstance(nP, np.ndarray) else nP
    idE_gpu = cp.empty((batch_size, num_blocks), dtype=idE.dtype) if isinstance(idE, np.ndarray) else idE

    GR_gpu = cp.empty((2, batch_size, block_size, block_size), dtype=dtype)  # Retarded (right)
    GRnn1_gpu = cp.empty((1, batch_size, block_size, block_size), dtype=dtype)  # Retarded (right)
    GL_gpu = cp.empty((2, batch_size, block_size, block_size), dtype=dtype)  # Lesser (right)
    GLnn1_gpu = cp.empty((1, batch_size, block_size, block_size), dtype=dtype)  # Lesser (right)
    GG_gpu = cp.empty((2, batch_size, block_size, block_size), dtype=dtype)  # Greater (right)
    GGnn1_gpu = cp.empty((1, batch_size, block_size, block_size), dtype=dtype)  # Greater (right)

    GR_compressed = cp.zeros_like(GR_host) if isinstance(GR_host, np.ndarray) else GR_host
    GL_compressed = cp.zeros_like(GL_host) if isinstance(GL_host, np.ndarray) else GL_host
    GG_compressed = cp.zeros_like(GG_host) if isinstance(GG_host, np.ndarray) else GG_host

    SR_dev = cp.empty_like(SR_host) if isinstance(SR_host, np.ndarray) else None
    SL_dev = cp.empty_like(SL_host) if isinstance(SL_host, np.ndarray) else None
    SG_dev = cp.empty_like(SG_host) if isinstance(SG_host, np.ndarray) else None
    SigRB_dev = [None for _ in range(num_blocks)]
    SigLB_dev = [None for _ in range(num_blocks)]
    SigGB_dev = [None for _ in range(num_blocks)]

    # med = time.time()
    # print(f"Memory allocation: {med - start}")
    
    # Backward pass IB \in {NB - 1, ..., 0}

    # First iteration IB = NB - 1
    IB = num_blocks - 1
    nIB = IB - 1
    idx = IB % 2
    nidx = nIB % 2
    NN = Bmax[-1] - Bmin[-1] + 1

    with input_stream:

        if map_diag_dev[0] is not None:
            for i in range(num_blocks):
                map_diag_dev[i].set(map_diag[i])
                if i < num_blocks - 1:
                    map_upper_dev[i].set(map_upper[i])
                    map_lower_dev[i].set(map_lower[i])
        else:
            for i in range(num_blocks - 1):
                map_diag_dev[i] = map_diag[i]
                map_upper_dev[i] = map_upper[i]
                map_lower_dev[i] = map_lower[i]
            map_diag_dev[-1] = map_diag[-1]
        _copy_csr_to_gpu(H_diag_host[IB], H_diag_buffer[idx])
        _copy_csr_to_gpu(S_diag_host[IB], S_diag_buffer[idx])
        if SR_dev is not None:
            SR_dev.set(SR_host)
            SL_dev.set(SL_host)
            SG_dev.set(SG_host)
        else:
            SR_dev = SR_host
            SL_dev = SL_host
            SG_dev = SG_host
        input_events[idx].record(stream=input_stream)

        if num_blocks > 1:

            nsrd = SR_diag_buffer[nidx]
            nsru = SR_upper_buffer[nidx]
            nsrl = SR_lower_buffer[nidx]
            nsld = SL_diag_buffer[nidx]
            nsll = SL_lower_buffer[nidx]
            nsgd = SG_diag_buffer[nidx]
            nsgl = SG_lower_buffer[nidx]

            _copy_csr_to_gpu(H_diag_host[nIB], H_diag_buffer[nidx])
            _copy_csr_to_gpu(H_upper_host[nIB], H_upper_buffer[nidx])
            _copy_csr_to_gpu(H_lower_host[nIB], H_lower_buffer[nidx])
            _copy_csr_to_gpu(S_diag_host[nIB], S_diag_buffer[nidx])
            _copy_csr_to_gpu(S_upper_host[nIB], S_upper_buffer[nidx])
            _copy_csr_to_gpu(S_lower_host[nIB], S_lower_buffer[nidx])

            # NOTE: We are assuming here that the next block is not in the boundary.
            _get_dense_block_batch(SR_dev, map_diag_dev, nIB, nsrd)
            _get_dense_block_batch(SR_dev, map_upper_dev, nIB, nsru)
            _get_dense_block_batch(SR_dev, map_lower_dev, nIB, nsrl)
            _get_dense_block_batch(SL_dev, map_diag_dev, nIB, nsld)
            _get_dense_block_batch(SL_dev, map_lower_dev, nIB, nsll)
            _get_dense_block_batch(SG_dev, map_diag_dev, nIB, nsgd)
            _get_dense_block_batch(SG_dev, map_lower_dev, nIB, nsgl)

            input_events[nidx].record(stream=input_stream)
    
    energies_dev = cp.asarray(energies)
    SigRB_dev[0] = cp.asarray(SigRB_left_host)
    SigRB_dev[-1] = cp.asarray(SigRB_right_host)
    SigLB_dev[0] = cp.asarray(SigLB_left_host)
    SigLB_dev[-1] = cp.asarray(SigLB_right_host)
    SigGB_dev[0] = cp.asarray(SigGB_left_host)
    SigGB_dev[-1] = cp.asarray(SigGB_right_host)

    computation_stream.wait_event(event=input_events[idx])

    hd = H_diag_buffer[idx]
    sd = S_diag_buffer[idx]
    srd = SR_diag_buffer[idx]
    sld = SL_diag_buffer[idx]
    sgd = SG_diag_buffer[idx]

    gr = gR_gpu[IB]
    gl = gL_gpu[IB]
    gg = gG_gpu[IB]

    srb = SigRB_dev[IB]
    slb = SigLB_dev[IB]
    sgb = SigGB_dev[IB]

    _get_dense_block_batch(SR_dev, map_diag_dev, IB, srd, srb)
    _get_dense_block_batch(SL_dev, map_diag_dev, IB, sld, slb)
    _get_dense_block_batch(SG_dev, map_diag_dev, IB, sgd, sgb)
    _get_system_matrix[num_thread_blocks, num_threads](energies_dev,
                                                       hd.data, hd.indices, hd.indptr,
                                                       sd.data, sd.indices, sd.indptr,
                                                       srd, md, batch_size, block_size)
    
    if solve:
        gpu_identity = cp.identity(NN, dtype=md.dtype)
        gpu_identity_batch = cp.repeat(gpu_identity[cp.newaxis, :, :], batch_size, axis=0)
        gr[:, 0:NN, 0:NN] = cp.linalg.solve(md[:, 0:NN, 0:NN], gpu_identity_batch)
    else:
        gr[:, 0:NN, 0:NN] = cp.linalg.inv(md[:, 0:NN, 0:NN])
    gr_h = cp.conjugate(gr[:, 0:NN, 0:NN].transpose((0,2,1)))
    cp.matmul(gr[:, 0:NN, 0:NN] @ sld[:, 0:NN, 0:NN], gr_h[:, 0:NN, 0:NN], out=gl[:, 0:NN, 0:NN])
    cp.matmul(gr[:, 0:NN, 0:NN] @ sgd[:, 0:NN, 0:NN], gr_h[:, 0:NN, 0:NN], out=gg[:, 0:NN, 0:NN])
    backward_events[idx].record(stream=computation_stream)

    # Rest iterations IB \in {NB - 2, ..., 0}
    for IB in range(num_blocks - 2, -1, -1):

        pIB = IB + 1
        nIB = IB - 1
        idx = IB % 2
        nidx = nIB % 2
        pidx = (IB + 1) % 2
        NI = Bmax[IB] - Bmin[IB] + 1
        NP = Bmax[IB + 1] - Bmin[IB + 1] + 1

        with input_stream:
                
            input_stream.wait_event(event=backward_events[pidx])
            if nIB >= 0:

                nsrd = SR_diag_buffer[nidx]
                nsru = SR_upper_buffer[nidx]
                nsrl = SR_lower_buffer[nidx]
                nsld = SL_diag_buffer[nidx]
                nsll = SL_lower_buffer[nidx]
                nsgd = SG_diag_buffer[nidx]
                nsgl = SG_lower_buffer[nidx]

                _copy_csr_to_gpu(H_diag_host[nIB], H_diag_buffer[nidx])
                _copy_csr_to_gpu(H_upper_host[nIB], H_upper_buffer[nidx])
                _copy_csr_to_gpu(H_lower_host[nIB], H_lower_buffer[nidx])
                _copy_csr_to_gpu(S_diag_host[nIB], S_diag_buffer[nidx])
                _copy_csr_to_gpu(S_upper_host[nIB], S_upper_buffer[nidx])
                _copy_csr_to_gpu(S_lower_host[nIB], S_lower_buffer[nidx])

                nsrb = SigRB_dev[nIB]
                nslb = SigLB_dev[nIB]
                nsgb = SigGB_dev[nIB]

                _get_dense_block_batch(SR_dev, map_diag_dev, nIB, nsrd, nsrb)
                _get_dense_block_batch(SR_dev, map_upper_dev, nIB, nsru)
                _get_dense_block_batch(SR_dev, map_lower_dev, nIB, nsrl)
                _get_dense_block_batch(SL_dev, map_diag_dev, nIB, nsld, nslb)
                _get_dense_block_batch(SL_dev, map_lower_dev, nIB, nsll)
                _get_dense_block_batch(SG_dev, map_diag_dev, nIB, nsgd, nsgb)
                _get_dense_block_batch(SG_dev, map_lower_dev, nIB, nsgl)
        
            else:  # nIB < 0

                nslu = SL_upper_buffer[idx]
                nsgu = SG_upper_buffer[idx]

                _get_dense_block_batch(SL_dev, map_upper_dev, IB, nslu)
                _get_dense_block_batch(SG_dev, map_upper_dev, IB, nsgu)

            input_events[nidx].record(stream=input_stream)
        
        hd = H_diag_buffer[idx]
        hu = H_upper_buffer[idx]
        hl = H_lower_buffer[idx]
        sd = S_diag_buffer[idx]
        su = S_upper_buffer[idx]
        sl = S_lower_buffer[idx]
        srd = SR_diag_buffer[idx]
        sru = SR_upper_buffer[idx]
        srl = SR_lower_buffer[idx]
        sld = SL_diag_buffer[idx]
        sll = SL_lower_buffer[idx]
        sgd = SG_diag_buffer[idx]
        sgl = SG_lower_buffer[idx]

        gr = gR_gpu[IB]
        pgr = gR_gpu[IB + 1]
        gl = gL_gpu[IB]
        pgl = gL_gpu[IB + 1]
        gg = gG_gpu[IB]
        pgg = gG_gpu[IB + 1]
        srb = SigRB_dev[IB]
        slb = SigLB_gpu[IB]
        sgb = SigGB_gpu[IB]

        if IB == 0:
            gr = GR_gpu[0]
            gl = GL_gpu[0]
            gg = GG_gpu[0]

        computation_stream.wait_event(event=input_events[idx])
        _get_system_matrix[num_thread_blocks, num_threads](energies_dev,
                                                           hd.data, hd.indices, hd.indptr,
                                                           sd.data, sd.indices, sd.indptr,
                                                           srd, md, batch_size, block_size)
        _get_system_matrix[num_thread_blocks, num_threads](energies_dev,
                                                           hu.data, hu.indices, hu.indptr,
                                                           su.data, su.indices, su.indptr,
                                                           sru, mu, batch_size, block_size)
        _get_system_matrix[num_thread_blocks, num_threads](energies_dev,
                                                           hl.data, hl.indices, hl.indptr,
                                                           sl.data, sl.indices, sl.indptr,
                                                           srl, ml, batch_size, block_size)

        mu_h = cp.conjugate(mu[:, 0:NI, 0:NP].transpose((0,2,1)))
        mu_x_pgr = mu[:, 0:NI, 0:NP] @ pgr[:, 0:NP, 0:NP]
        al = mu_x_pgr @ sll[:, 0:NP, 0:NI]
        ag = mu_x_pgr @ sgl[:, 0:NP, 0:NI]
        inv_arg = md[:, 0:NI, 0:NI] - mu_x_pgr @ ml[:, :NP, 0:NI]
        if solve:
            gpu_identity = cp.identity(NI, dtype=inv_arg.dtype)
            gpu_identity_batch = cp.repeat(gpu_identity[cp.newaxis, :, :], batch_size, axis=0)
            gr[:, 0:NI, 0:NI] = cp.linalg.solve(inv_arg, gpu_identity_batch)
        else:
            gr[:, 0:NI, 0:NI] = cp.linalg.inv(inv_arg)
        gr_h = cp.conjugate(gr[:, 0:NI, 0:NI].transpose((0,2,1)))
        cp.subtract(mu[:, 0:NI, 0:NP] @ pgl[:, 0:NP, 0:NP] @ mu_h[:, 0:NP, 0:NI],
                    al - cp.conjugate(al.transpose((0,2,1))), out=slb[:, 0:NI, 0:NI])  # SLB must change
        cp.subtract(mu[:, 0:NI, 0:NP] @ pgg[:, 0:NP, 0:NP] @ mu_h[:, 0:NP, 0:NI],
                    ag - cp.conjugate(ag.transpose((0,2,1))), out=sgb[:, 0:NI, 0:NI])  # SGB must change
        cp.matmul(gr[:, 0:NI, 0:NI] @ (sld[:, 0:NI, 0:NI] + slb[:, 0:NI, 0:NI]), gr_h[:, 0:NI, 0:NI], out=gl[:, 0:NI, 0:NI])
        cp.matmul(gr[:, 0:NI, 0:NI] @ (sgd[:, 0:NI, 0:NI] + sgb[:, 0:NI, 0:NI]), gr_h[:, 0:NI, 0:NI], out=gg[:, 0:NI, 0:NI])
        backward_events[idx].record(stream=computation_stream)

        if IB == 0:

            DOS_gpu[:, 0] = 1j * cp.trace(gr[:, 0:NI, 0:NI] - gr_h[:, 0:NI, 0:NI], axis1=1, axis2=2)
            nE_gpu[:, 0] = -1j * cp.trace(gl[:, 0:NI, 0:NI], axis1=1, axis2=2)
            nP_gpu[:, 0] = 1j * cp.trace(gg[:, 0:NI, 0:NI], axis1=1, axis2=2)
            idE_gpu[:, 0] = cp.real(cp.trace(sgb[:, 0:NI, 0:NI] @ gl[:, 0:NI, 0:NI] -
                                             gg[:, 0:NI, 0:NI] @ slb[:, 0:NI, 0:NI], axis1=1, axis2=2))
    
    # Forward pass

    # First iteration
    IB = 0
    nIB = IB + 1
    idx = IB % 2
    nidx = nIB % 2

    with input_stream:

        input_stream.wait_event(event=backward_events[idx])      
        if nIB < num_blocks:

            npsll = SL_lower_buffer[nidx]
            npsgl = SG_lower_buffer[nidx]

            _copy_csr_to_gpu(H_upper_host[IB], prev_H_upper_buffer[nidx])
            _copy_csr_to_gpu(H_lower_host[IB], prev_H_lower_buffer[nidx])
            _copy_csr_to_gpu(S_upper_host[IB], prev_S_upper_buffer[nidx])
            _copy_csr_to_gpu(S_lower_host[IB], prev_S_lower_buffer[nidx])

            _get_dense_block_batch(SL_dev, map_lower_dev, IB, npsll)
            _get_dense_block_batch(SG_dev, map_lower_dev, IB, npsgl)

            if nIB < num_blocks - 1:

                nsru = SR_upper_buffer[nidx]
                nsrl = SR_lower_buffer[nidx]
                nslu = SL_upper_buffer[nidx]
                nsgu = SG_upper_buffer[nidx]
                _copy_csr_to_gpu(H_upper_host[nIB], H_upper_buffer[nidx])
                _copy_csr_to_gpu(H_lower_host[nIB], H_lower_buffer[nidx])
                _copy_csr_to_gpu(S_upper_host[nIB], S_upper_buffer[nidx])
                _copy_csr_to_gpu(S_lower_host[nIB], S_lower_buffer[nidx])

                _get_dense_block_batch(SR_dev, map_upper_dev, nIB, nsru)
                _get_dense_block_batch(SR_dev, map_lower_dev, nIB, nsrl)
                _get_dense_block_batch(SL_dev, map_upper_dev, nIB, nslu)
                _get_dense_block_batch(SG_dev, map_upper_dev, nIB, nsgu)

            input_events[idx].record(stream=input_stream)
    
    hu = H_upper_buffer[idx]
    su = S_upper_buffer[idx]
    hl = H_lower_buffer[idx]
    sl = S_lower_buffer[idx]
    slu = SL_upper_buffer[idx]
    sgu = SG_upper_buffer[idx]

    GR = GR_gpu[IB]
    GL = GL_gpu[IB]
    GG = GG_gpu[IB]
    GRnn1 = GRnn1_gpu[0]
    GLnn1 = GLnn1_gpu[0]
    GGnn1 = GGnn1_gpu[0]

    # NOTE: These were written directly to output in the last iteration of the backward pass
    gr = GR_gpu[IB]
    gl = GL_gpu[IB]
    gg = GG_gpu[IB]

    pgr = gR_gpu[nIB]
    pgl = gL_gpu[nIB]
    pgg = gG_gpu[nIB]

    computation_stream.wait_event(event=input_events[nidx])
    _get_system_matrix[num_thread_blocks, num_threads](energies_dev,
                                                       hu.data, hu.indices, hu.indptr,
                                                       su.data, su.indices, su.indptr,
                                                       sru, mu, batch_size, block_size)
    _get_system_matrix[num_thread_blocks, num_threads](energies_dev,
                                                       hl.data, hl.indices, hl.indptr,
                                                       sl.data, sl.indices, sl.indptr,
                                                       srl, ml, batch_size, block_size)
    gr_h = cp.conjugate(pgr[:, 0:NP, 0:NP].transpose((0,2,1)))
    ml_h = cp.conjugate(ml[:, 0:NP, 0:NI].transpose((0,2,1)))
    gr_x_mu = gr[:, 0:NI, 0:NI] @ mu[:, 0:NI, 0:NP]
    ml_h_x_gr_h = ml_h @ gr_h[:, 0:NP, 0:NP]
    # NOTE: These were written in the last iteration of the backward pass
    # GR[:] = gr
    # GL[:] = gl
    # GG[:] = gg
    cp.negative(gr_x_mu @ pgr[:, 0:NP, 0:NP], out=GRnn1[:, 0:NI, 0:NP])
    cp.subtract(gr[:, 0:NI, 0:NI] @ slu[:, 0:NI, 0:NP] @ gr_h[:, 0:NP, 0:NP] - gr_x_mu @ pgl[:, 0:NP, 0:NP], gl[:, 0:NI, 0:NI] @ ml_h_x_gr_h, out=GLnn1[:, 0:NI, 0:NP])
    cp.subtract(gr[:, 0:NI, 0:NI] @ sgu[:, 0:NI, 0:NP] @ gr_h[:, 0:NP, 0:NP] - gr_x_mu @ pgg[:, 0:NP, 0:NP], gg[:, 0:NI, 0:NI] @ ml_h_x_gr_h, out=GGnn1[:, 0:NI, 0:NP])
    backward_events[nidx].record(stream=computation_stream)

    computation_stream.synchronize()
    _store_compressed(map_diag_dev, map_upper_dev, map_lower_dev, GR, GRnn1, IB, GR_compressed)
    _store_compressed(map_diag_dev, map_upper_dev, map_lower_dev, GL, GLnn1, IB, GL_compressed)
    _store_compressed(map_diag_dev, map_upper_dev, map_lower_dev, GG, GGnn1, IB, GG_compressed)

    # Rest iterations
    for IB in range(1, num_blocks):

        pIB = IB - 1
        nIB = IB + 1
        idx = IB % 2
        pidx = pIB % 2
        nidx = nIB % 2
        NI = Bmax[IB] - Bmin[IB] + 1
        NM = Bmax[IB - 1] - Bmin[IB - 1] + 1

        with input_stream:
                
            input_stream.wait_event(event=backward_events[idx])
            if nIB < num_blocks:

                npsll = SL_lower_buffer[idx]
                npsgl = SG_lower_buffer[idx]

                _get_dense_block_batch(SL_dev, map_lower_dev, IB, npsll)
                _get_dense_block_batch(SG_dev, map_lower_dev, IB, npsgl)


                if nIB < num_blocks - 1:

                    nsru = SR_upper_buffer[nidx]
                    nsrl = SR_lower_buffer[nidx]
                    nslu = SL_upper_buffer[nidx]
                    nsgu = SG_upper_buffer[nidx]

                    _copy_csr_to_gpu(H_upper_host[nIB], H_upper_buffer[nidx])
                    _copy_csr_to_gpu(H_lower_host[nIB], H_lower_buffer[nidx])
                    _copy_csr_to_gpu(S_upper_host[nIB], S_upper_buffer[nidx])
                    _copy_csr_to_gpu(S_lower_host[nIB], S_lower_buffer[nidx])

                    _get_dense_block_batch(SR_dev, map_upper_dev, nIB, nsru)
                    _get_dense_block_batch(SR_dev, map_lower_dev, nIB, nsrl)
                    _get_dense_block_batch(SL_dev, map_upper_dev, nIB, nslu)
                    _get_dense_block_batch(SG_dev, map_upper_dev, nIB, nsgu)

            input_events[idx].record(stream=input_stream)
        
        pmu = mu
        pml = ml
        psll = SL_lower_buffer[pidx]
        psgl = SG_lower_buffer[pidx]

        GR = GR_gpu[idx]
        pGR = GR_gpu[pidx]
        GL = GL_gpu[idx]
        pGL = GL_gpu[pidx]
        GG = GG_gpu[idx]
        pGG = GG_gpu[pidx]

        gr = gR_gpu[IB]
        gl = gL_gpu[IB]
        gg = gG_gpu[IB]

        computation_stream.wait_event(event=input_events[pidx])
        pGRh = cp.conjugate(pGR[:, 0:NM, 0:NM].transpose((0,2,1)))
        phlh = cp.conjugate(pml[:, 0:NI, 0:NM].transpose((0,2,1)))
        grh = cp.conjugate(gr[:, 0:NI, 0:NI].transpose((0,2,1)))
        grphl = gr[:, 0:NI, 0:NI] @ pml[:, 0:NI, 0:NM]
        grphlpGRphu = grphl @ pGR[:, 0:NM, 0:NM] @ pmu[:, 0:NM, 0:NI]
        phlhgrh = phlh @ grh
        pGRhphlhgrh = pGRh @ phlhgrh
        al = gr[:, 0:NI, 0:NI] @ psll[:, 0:NI, 0:NI] @ pGRhphlhgrh
        bl = grphlpGRphu @ gl[:, 0:NI, 0:NI]
        ag = gr[:, 0:NI, 0:NI] @ psgl[:, 0:NI, 0:NM] @ pGRhphlhgrh
        bg = grphlpGRphu @ gg[:, 0:NI, 0:NI]
        cp.add(gr[:, 0:NI, 0:NI], grphlpGRphu @ gr[:, 0:NI, 0:NI], out=GR[:, 0:NI, 0:NI])
        cp.subtract(gl[:, 0:NI, 0:NI] + grphl @ pGL[:, 0:NM, 0:NM] @ phlhgrh,
                    al - cp.conjugate(al.transpose((0,2,1))) - bl + cp.conjugate(bl.transpose((0,2,1))),
                    out=GL[:, 0:NI, 0:NI])
        cp.subtract(gg[:, 0:NI, 0:NI] + grphl @ pGG[:, 0:NM, 0:NM] @ phlhgrh,
                    ag - cp.conjugate(ag.transpose((0,2,1))) - bg + cp.conjugate(bg.transpose((0,2,1))),
                    out=GG[:, 0:NI, 0:NI])
    
        if IB < num_blocks - 1:

            NP = Bmax[IB + 1] - Bmin[IB + 1] + 1

            GRnn1 = GRnn1_gpu[0]
            GLnn1 = GLnn1_gpu[0]
            GGnn1 = GGnn1_gpu[0]

            ngr = gR_gpu[nIB]
            ngl = gL_gpu[nIB]
            ngg = gG_gpu[nIB]

            hu = H_upper_buffer[idx]
            hl = H_lower_buffer[idx]
            su = S_upper_buffer[idx]
            sl = S_lower_buffer[idx]
            sru = SR_upper_buffer[idx]
            srl = SR_lower_buffer[idx]
            slu = SL_upper_buffer[idx]
            sgu = SG_upper_buffer[idx]

            _get_system_matrix[num_thread_blocks, num_threads](energies_dev,
                                                               hu.data, hu.indices, hu.indptr,
                                                               su.data, su.indices, su.indptr,
                                                               sru, mu, batch_size, block_size)
            _get_system_matrix[num_thread_blocks, num_threads](energies_dev,
                                                               hl.data, hl.indices, hl.indptr,
                                                               sl.data, sl.indices, sl.indptr,
                                                               srl, ml, batch_size, block_size)

            ngrh = cp.conjugate(ngr[:, 0:NP, 0:NP].transpose((0,2,1)))
            hlh = cp.conjugate(ml[:, 0:NP, 0:NI].transpose((0,2,1)))
            GRhu = GR[:, 0:NI, 0:NI] @ mu[:, 0:NI, 0:NP]
            hlhngrh = hlh @ ngrh
            cp.negative(GR[:, 0:NI, 0:NI] @ mu[:, 0:NI, 0:NP] @ ngr[:, 0:NP, 0:NP], out=GRnn1[:, 0:NI, 0:NP])
            cp.subtract(GR[:, 0:NI, 0:NI] @ slu[:, 0:NI, 0:NP] @ ngrh,
                        GRhu @ ngl[:, 0:NP, 0:NP] + GL[:, 0:NI, 0:NI] @ hlhngrh, out=GLnn1[:, 0:NI, 0:NP])
            cp.subtract(GR[:, 0:NI, 0:NI] @ sgu[:, 0:NI, 0:NP] @ ngrh,
                        GRhu @ ngg[:, 0:NP, 0:NP] + GG[:, 0:NI, 0:NI] @ hlhngrh, out=GGnn1[:, 0:NI, 0:NP])
            backward_events[idx].record(stream=computation_stream)

            computation_stream.synchronize()
            _store_compressed(map_diag_dev, map_upper_dev, map_lower_dev, GR, GRnn1, IB, GR_compressed)
            _store_compressed(map_diag_dev, map_upper_dev, map_lower_dev, GL, GLnn1, IB, GL_compressed)
            _store_compressed(map_diag_dev, map_upper_dev, map_lower_dev, GG, GGnn1, IB, GG_compressed)

            slb = SigLB_gpu[IB]
            sgb = SigGB_gpu[IB]
            
            idE_gpu[:, IB] = cp.real(cp.trace(sgb[:, 0:NI, 0:NI] @ GL[:, 0:NI, 0:NI] -
                                              GG[:, 0:NI, 0:NI] @ slb[:, 0:NI, 0:NI], axis1=1, axis2=2))

        DOS_gpu[:, IB] = 1j * cp.trace(GR[:, 0:NI, 0:NI] - cp.conjugate(GR[:, 0:NI, 0:NI].transpose(0,2,1)), axis1=1, axis2=2)
        nE_gpu[:, IB] = -1j * cp.trace(GL[:, 0:NI, 0:NI], axis1=1, axis2=2)
        nP_gpu[:, IB] = 1j * cp.trace(GG[:, 0:NI, 0:NI], axis1=1, axis2=2)

    slb = SigLB_dev[-1]
    sgb = SigGB_dev[-1]
    idE_gpu[:, num_blocks - 1] = cp.real(cp.trace(sgb[:, 0:NI, 0:NI] @ GL[:, 0:NI, 0:NI] -
                                          GG[:, 0:NI, 0:NI] @ slb[:, 0:NI, 0:NI], axis1=1, axis2=2))
    

    computation_stream.synchronize()
    with input_stream:
        _store_compressed(map_diag_dev, map_upper_dev, map_lower_dev, GL, None, num_blocks - 1, GL_compressed)
        _store_compressed(map_diag_dev, map_upper_dev, map_lower_dev, GG, None, num_blocks - 1, GG_compressed)
        if isinstance(GL_host, np.ndarray):
            GL_compressed.get(out=GL_host)
        if isinstance(GG_host, np.ndarray):
            GG_compressed.get(out=GG_host)
    _store_compressed(map_diag_dev, map_upper_dev, map_lower_dev, GR, None, num_blocks - 1, GR_compressed)
    if isinstance(GR_host, np.ndarray):
        GR_compressed.get(out=GR_host)
    if isinstance(DOS, np.ndarray):
        DOS_gpu.get(out=DOS)
        nE_gpu.get(out=nE)
        nP_gpu.get(out=nP)
        idE_gpu.get(out=idE)
    input_stream.synchronize()
    computation_stream.synchronize()

    # end = time.time()
    # print(f"Computation: {end - med}")
