# Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.
""" Functions to calculate the polarization on the cpu. See README.md for more information. """

import numpy as np
import numpy.typing as npt
import typing
import cupy as cp
import sys
import os

main_path = os.path.abspath(os.path.dirname(__file__))
parent_path = os.path.abspath(os.path.join(main_path, "..", "..", ".."))
sys.path.append(parent_path)

from quatrex.utils import linalg_gpu

import cupyx as cpx
fft_gpu = cpx.scipy.fft.fft
ifft_gpu = cpx.scipy.fft.ifft


def g2p_fft_gpu(pre_factor: np.complex128, ij2ji: cp.ndarray, gg: cp.ndarray, gl: cp.ndarray,
                gr: cp.ndarray) -> typing.Tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
    """Calculate the polarization with fft on the gpu(see file description). 
        The inputs are the pre factor and the Green's Functions.
        Only the data and a mapping to the transposed indices are needed.

    Args:
        pre_factor       (np.complex128): pre_factor, multiplied at the end
        ij2ji               (cp.ndarray): mapping to transposed matrix, (#orbital)
        gg                  (cp.ndarray): Greater Green's Function,     (#orbital, #energy)
        gl                  (cp.ndarray): Lesser Green's Function,      (#orbital, #energy)
        gr                  (cp.ndarray): Retarded Green's Function,    (#orbital, #energy)

    Returns:
        typing.Tuple[cp.ndarray, Greater polarization  (#orbital, #energy)
                     cp.ndarray, Lesser polarization   (#orbital, #energy)
                     cp.ndarray  Retarded polarization (#orbital, #energy)
                    ]
    """

    # number of energy points
    ne = gg.shape[1]

    # fft
    gg_t = cp.fft.fft(gg, n=2 * ne, axis=1)
    gl_t = cp.fft.fft(gl, n=2 * ne, axis=1)
    gr_t = cp.fft.fft(gr, n=2 * ne, axis=1)

    # reverse and transpose
    # only once since identity is used for lesser polarization
    gl_t_mod = cp.roll(cp.flip(gl_t, axis=1), 1, axis=1)[ij2ji, :]

    # multiply elementwise
    pg_t = cp.multiply(gg_t, gl_t_mod)
    pr_t = linalg_gpu.pr_special_gpu(gr_t, gl_t_mod, gl_t)

    # ifft, cutoff and multiply with pre factor
    pr = cp.multiply(cp.fft.ifft(pr_t, axis=1)[:, :ne], pre_factor)
    pg = cp.multiply(cp.fft.ifft(pg_t, axis=1), pre_factor)

    # lesser polarization from identity
    pl = -cp.conjugate(cp.roll(cp.flip(pg, axis=1), 1, axis=1))

    # cutoff
    pg = pg[:, :ne]
    pl = pl[:, :ne]

    return (pg, pl, pr)


def g2p_conv_gpu(iteration: np.int32 = 1):
    """Creates function to calculate the polarization with conv on the gpu(see file description). 

    True Args: 
        iteration (np.int32): code iteration to test    
    
    The created function needs the following arguments and give following returns:

    Args:
        prefactor   (np.cdouble): 
        ij2ji       (cp.ndarray): mapping to transposed matrix, (#orbital)
        gg          (cp.ndarray): Greater Green's Function,     (#orbital, #energy)
        gl          (cp.ndarray): Lesser Green's Function,      (#orbital, #energy)
        gr          (cp.ndarray): Retarded Green's Function_,   (#orbital, #energy)
        pg          (cp.ndarray): Greater Green's polarization,     (#orbital, #energy)
        pl          (cp.ndarray): Lesser Green's polarization,      (#orbital, #energy)
        pr          (cp.ndarray): Retarded Green's polarization,   (#orbital, #energy)
        no                 (int): number of nnz/#orbital
        ne                 (int): number of energy points/#energy
    """
    if iteration == 1:
        code = r'''
        #include <cupy/complex.cuh>
        extern "C" __global__
        void g2p_conv(const complex<double> prefactor, 
                    const int* ij2ji, 
                    const complex<double>* gg, 
                    const complex<double>* gl, 
                    const complex<double>* gr, 
                    complex<double>* pg, 
                    complex<double>* pl, 
                    complex<double>* pr,
                    int no,
                    int ne) {
            for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
                 idx < no;
                 idx += gridDim.x * blockDim.x) {
                int ji = ij2ji[idx];
                for (int idy = blockIdx.y * blockDim.y + threadIdx.y;
                     idy < ne;
                     idy += gridDim.y * blockDim.y) {
                    complex<double> tmpg;
                    complex<double> tmpl;
                    complex<double> tmpr;
                    for (int ep = idy; ep < ne; ep++) {
                        complex<double> tmpgl1 = gl[ep + idx * ne];
                        complex<double> tmpgl2 = gl[ep - idy + ji * ne];
                        tmpg = tmpg + prefactor * gg[ep + idx * ne] * tmpgl2;     
                        tmpl = tmpl + prefactor * tmpgl1 * gg[ep - idy + ji * ne];     
                        tmpr = tmpr + prefactor * (gr[ep + idx * ne] * tmpgl2 + 
                                                   tmpgl1 * conj(gr[ep - idy + idx * ne]));     
                    }
                    pg[idy + idx * ne] = tmpg;
                    pl[idy + idx * ne] = tmpl;
                    pr[idy + idx * ne] = tmpr;
                }
            }
        }
        '''
    else:
        raise ValueError("iteration " + str(iteration) + " does not exist")

    kernel = cp.RawKernel(code, "g2p_conv")

    return kernel


def g2p_fft_mpi_gpu(
    pre_factor: np.complex128, gg: npt.NDArray[np.complex128], gl: npt.NDArray[np.complex128],
    gr: npt.NDArray[np.complex128], gl_transposed: npt.NDArray[np.complex128]
) -> typing.Tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128], npt.NDArray[np.complex128]]:
    """Calculate the polarization with fft and mpi on the gpu(see file description).
    In addition, already loads and unloads data to and from the gpu.

    Args:
        pre_factor            (np.complex128): pre_factor, multiplied at the end
        gg       (npt.NDArray[np.complex128]): Greater Green's Function,          (#orbital/#ranks, #energy)
        gl       (npt.NDArray[np.complex128]): Lesser Green's Function,           (#orbital/#ranks, #energy)
        gr       (npt.NDArray[np.complex128]): Retarded Green's Function_,        (#orbital/#ranks, #energy)
        gl_transposed (npt.NDArray[np.complex128]): Transposed Lesser Green's Function (#orbital/#ranks, #energy)

    Returns:
        typing.Tuple[npt.NDArray[np.complex128], Greater polarization  (#orbital, #energy)  
                     npt.NDArray[np.complex128], Lesser polarization   (#orbital, #energy)
                     npt.NDArray[np.complex128]  Retarded polarization (#orbital, #energy)
                    ]
    """
    # number of energy points
    ne = gg.shape[1]

    # load data to gpu----------------------------------------------------------
    gg_gpu = cp.asarray(gg)
    gl_gpu = cp.asarray(gl)
    gr_gpu = cp.asarray(gr)
    gl_transposed_gpu = cp.asarray(gl_transposed)

    # compute pg/pl/pr----------------------------------------------------------

    # fft
    gg_t_gpu = cp.fft.fft(gg_gpu, n=2 * ne, axis=1)
    gl_t_gpu = cp.fft.fft(gl_gpu, n=2 * ne, axis=1)
    gr_t_gpu = cp.fft.fft(gr_gpu, n=2 * ne, axis=1)
    gl_t_transposed_gpu = cp.fft.fft(gl_transposed_gpu, n=2 * ne, axis=1)

    # time reversed
    gl_t_mod_gpu = cp.roll(cp.flip(gl_t_transposed_gpu, axis=1), 1, axis=1)

    # multiply elementwise
    pg_t_gpu = cp.multiply(gg_t_gpu, gl_t_mod_gpu)
    pr_t_gpu = linalg_gpu.pr_special_gpu(gr_t_gpu, gl_t_mod_gpu, gl_t_gpu)

    # ifft, cutoff and multiply with pre factor
    pr_gpu = cp.multiply(cp.fft.ifft(pr_t_gpu, axis=1)[:, :ne], pre_factor)
    pg_gpu = cp.multiply(cp.fft.ifft(pg_t_gpu, axis=1), pre_factor)

    pl_gpu = -cp.conjugate(cp.roll(cp.flip(pg_gpu, axis=1), 1, axis=1))

    # cutoff
    pg_gpu = pg_gpu[:, :ne]
    pl_gpu = pl_gpu[:, :ne]

    # load data to cpu----------------------------------------------------------

    pg = cp.asnumpy(pg_gpu)
    pl = cp.asnumpy(pl_gpu)
    pr = cp.asnumpy(pr_gpu)

    return (pg, pl, pr)

def g2p_fft_mpi_gpu_batched_nopr(pg, pl,
    pre_factor: np.complex128, gg: npt.NDArray[np.complex128], gl: npt.NDArray[np.complex128],
    gl_transposed: npt.NDArray[np.complex128], batch_size: int = 1000
) -> typing.Tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128], npt.NDArray[np.complex128]]:
    """Calculate the polarization with fft and mpi on the gpu(see file description).
    In addition, already loads and unloads data to and from the gpu.

    Args:
        pre_factor            (np.complex128): pre_factor, multiplied at the end
        gg       (npt.NDArray[np.complex128]): Greater Green's Function,          (#orbital/#ranks, #energy)
        gl       (npt.NDArray[np.complex128]): Lesser Green's Function,           (#orbital/#ranks, #energy)
        gr       (npt.NDArray[np.complex128]): Retarded Green's Function_,        (#orbital/#ranks, #energy)
        gl_transposed (npt.NDArray[np.complex128]): Transposed Lesser Green's Function (#orbital/#ranks, #energy)

    Returns:
        typing.Tuple[npt.NDArray[np.complex128], Greater polarization  (#orbital, #energy)  
                     npt.NDArray[np.complex128], Lesser polarization   (#orbital, #energy)
                     npt.NDArray[np.complex128]  Retarded polarization (#orbital, #energy)
                    ]
    """
    # number of energy points
    ne = gg.shape[1]
    no = gg.shape[0]

    mempool = cp.get_default_memory_pool()
    max_bytes = 64 * 1024 ** 3
    used_bytes = mempool.used_bytes()
    spare_bytes = max_bytes - used_bytes
    # spare_bytes -= 2 * no * ne * 16  #  PG, PL, 16 bytes for complex128
    num_buffers = 12  # closer to 6 but overapproximating
    avail_buffer_size = spare_bytes // num_buffers
    batch_size = avail_buffer_size // (2 * ne * 16)  # 16 bytes for complex128
    batch_size = min(batch_size, no)
    batches = int(np.ceil(no / batch_size))
    batch_size = int(np.ceil(no / batches))  # Balance last batch
    # print(f"Used bytes: {mempool.used_bytes()}", flush=True)
    # print(f"Total bytes: {mempool.total_bytes()}", flush=True)
    # print(f"Spare bytes: {spare_bytes}, Batches: {batches}, Batch size: {batch_size}", flush=True)

    # # Assuming FFT space is complexity is O(NlogN), where N is ne (above)
    # total_space = 2 * ne * int(np.log2(ne)) * no
    # # The following is an assumption for LUMI
    # max_energies = 64 * 8 * 64
    # max_orbitals = 1000
    # max_space = 2 * max_energies * int(np.log2(2 * max_energies)) * max_orbitals
    # batches = int(np.ceil(total_space / max_space))
    # batch_size = int(np.ceil(no / batches))
    # print("Total space: ", total_space, ", Max space: ", max_space, ", Batches: ", batches, flush=True)


    # batches = no // batch_size
    # if batches == 0:
    #     batch_size = no
    #     batches = 1

    # pg = cp.empty((no, ne), dtype=np.complex128)
    # pl = cp.empty((no, ne), dtype=np.complex128)


    # load data to gpu and compute----------------------------------------------
    # allocate gpu memory
    # gg_gpu = cp.empty((batch_size + no % batch_size, ne), dtype=np.complex128)
    # gl_gpu = cp.empty((batch_size + no % batch_size, ne), dtype=np.complex128)
    # gl_transposed_gpu = cp.empty((batch_size + no % batch_size, ne), dtype=np.complex128)


    # compute pg/pl/pr----------------------------------------------------------
    for batch in range(batches):
        batch_start = batch * batch_size
        # last batch different, if not dividable
        batch_end = batch_size * (batch + 1) if batch != batches - 1 else batch_size * (batch + 1) + no % batch_size

        # fft
        # gg_gpu[0:batch_end - batch_start] = cp.asarray(gg[batch_start:batch_end, :])
        # gg_t_gpu = cp.fft.fft(gg_gpu[0:batch_end - batch_start], n=2 * ne, axis=1)
        gg_gpu = gg[batch_start:batch_end, :]
        gg_t_gpu = cp.fft.fft(gg_gpu, n=2 * ne, axis=1)

        # gl_transposed_gpu[0:batch_end - batch_start] = cp.asarray(gl_transposed[batch_start:batch_end, :])
        # gl_t_transposed_gpu = cp.fft.fft(gl_transposed_gpu[0:batch_end - batch_start], n=2 * ne, axis=1)
        gl_transposed_gpu = gl_transposed[batch_start:batch_end, :]
        gl_t_transposed_gpu = cp.fft.fft(gl_transposed_gpu, n=2 * ne, axis=1)

        # time reversed
        # gl_t_mod_gpu = cp.roll(cp.flip(gl_t_transposed_gpu, axis=1), 1, axis=1)
        gl_t_transposed_gpu = cp.roll(cp.flip(gl_t_transposed_gpu, axis=1), 1, axis=1)

        # multiply elementwise
        # pg_t_gpu = cp.multiply(gg_t_gpu, gl_t_mod_gpu)
        pg_t_gpu = cp.multiply(gg_t_gpu, gl_t_transposed_gpu)

        # ifft, cutoff and multiply with pre factor
        # pg_gpu = cp.multiply(cp.fft.ifft(pg_t_gpu, axis=1), pre_factor)
        pg_gpu = cp.multiply(ifft_gpu(pg_t_gpu, axis=1, overwrite_x=True), pre_factor)
        pl_gpu = -cp.conjugate(cp.roll(cp.flip(pg_gpu, axis=1), 1, axis=1))

        # cutoff
        pg_gpu = pg_gpu[:, :ne]
        pl_gpu = pl_gpu[:, :ne]

        # load data to cpu----------------------------------------------------------

        pg[batch_start:batch_end, :] = pg_gpu[0:batch_end - batch_start]
        pl[batch_start:batch_end, :] = pl_gpu[0:batch_end - batch_start]

        # print(f"Used bytes: {mempool.used_bytes()}", flush=True)
        # print(f"Total bytes: {mempool.total_bytes()}", flush=True)

    # return (pg, pl)


def g2p_fft_mpi_gpu_streams(pre_factor: np.complex128, gg: npt.NDArray[np.complex128], gl: npt.NDArray[np.complex128],
                            gr: npt.NDArray[np.complex128], gl_transposed: npt.NDArray[np.complex128],
                            pg: npt.NDArray[np.complex128], pl: npt.NDArray[np.complex128],
                            pr: npt.NDArray[np.complex128], streams: typing.List[cp.cuda.Stream]):
    """Calculate the polarization with fft and mpi on the gpu(see file description).
    In addition, already loads and unloads data to and from the gpu.
    Uses streams to overlap data transfer and computation.
    For highest performance, input pinned memory.

    Args:
        pre_factor            (np.complex128): pre_factor, multiplied at the end
        gg       (npt.NDArray[np.complex128]): Greater Green's Function,          (#orbital/#ranks, #energy)
        gl       (npt.NDArray[np.complex128]): Lesser Green's Function,           (#orbital/#ranks, #energy)
        gr       (npt.NDArray[np.complex128]): Retarded Green's Function_,        (#orbital/#ranks, #energy)
        gl_transposed (npt.NDArray[np.complex128]): Transposed Lesser Green's Function (#orbital/#ranks, #energy)
        pg       (npt.NDArray[np.complex128]): Greater polarization,              (#orbital/#ranks, #energy)
        pl       (npt.NDArray[np.complex128]): Lesser polarization,               (#orbital/#ranks, #energy)
        pr       (npt.NDArray[np.complex128]): Retarded polarization,             (#orbital/#ranks, #energy)
        streams (typing.List[cp.cuda.Stream]): List of streams to overlay memcpy and computations, at least four streams needed
    """

    # number of energy points
    ne = gg.shape[1]

    # load data to gpu and compute----------------------------------------------
    gg_gpu = cp.empty_like(gg)
    gl_gpu = cp.empty_like(gl)
    gr_gpu = cp.empty_like(gr)
    gl_transposed_gpu = cp.empty_like(gl_transposed)

    with streams[0]:
        gg_gpu.set(gg, stream=streams[0])
        gg_t_gpu = cp.fft.fft(gg_gpu, n=2 * ne, axis=1)
    with streams[1]:
        gl_gpu.set(gl, stream=streams[1])
        gl_t_gpu = cp.fft.fft(gl_gpu, n=2 * ne, axis=1)
    with streams[2]:
        gr_gpu.set(gr, stream=streams[2])
        gr_t_gpu = cp.fft.fft(gr_gpu, n=2 * ne, axis=1)
    with streams[3]:
        gl_transposed_gpu.set(gl_transposed, stream=streams[3])
        gl_t_transposed_gpu = cp.fft.fft(gl_transposed_gpu, n=2 * ne, axis=1)
        gl_t_mod_gpu = cp.roll(cp.flip(gl_t_transposed_gpu, axis=1), 1, axis=1)
    with streams[0]:
        pg_t_gpu = cp.multiply(gg_t_gpu, gl_t_mod_gpu)
        pg_gpu = cp.multiply(cp.fft.ifft(pg_t_gpu, axis=1), pre_factor)
        pl_gpu = -cp.conjugate(cp.roll(cp.flip(pg_gpu, axis=1), 1, axis=1))

    streams[1].synchronize()
    streams[3].synchronize()
    with streams[2]:
        pr_t_gpu = linalg_gpu.pr_special_gpu(gr_t_gpu, gl_t_mod_gpu, gl_t_gpu)
        pr_gpu = cp.multiply(cp.fft.ifft(pr_t_gpu, axis=1)[:, :ne], pre_factor)
        cp.asnumpy(pr_gpu, out=pr, stream=streams[2])

    streams[0].synchronize()
    with streams[1]:
        pl_gpu = pl_gpu[:, :ne]
        cp.asnumpy(pl_gpu, out=pl, stream=streams[1])

    with streams[0]:
        pg_gpu = pg_gpu[:, :ne]
        cp.asnumpy(pg_gpu, out=pg, stream=streams[0])

    for stream in streams:
        stream.synchronize()


def g2p_fft_mpi_gpu_batched_streams(pre_factor: np.complex128, gg: npt.NDArray[np.complex128], gl: npt.NDArray[np.complex128],
                            gr: npt.NDArray[np.complex128], gl_transposed: npt.NDArray[np.complex128],
                            pg: npt.NDArray[np.complex128], pl: npt.NDArray[np.complex128],
                            pr: npt.NDArray[np.complex128], streams: typing.List[cp.cuda.Stream], batch_size: int):
    """Calculate the polarization with fft and mpi on the gpu(see file description).
    In addition, already loads and unloads data to and from the gpu.
    Uses streams to overlap data transfer and computation.
    For highest performance, input pinned memory.
    Batches the data, since the total matrix will not fit on the gpu.

    Args:
        pre_factor            (np.complex128): pre_factor, multiplied at the end
        gg       (npt.NDArray[np.complex128]): Greater Green's Function,          (#orbital/#ranks, #energy)
        gl       (npt.NDArray[np.complex128]): Lesser Green's Function,           (#orbital/#ranks, #energy)
        gr       (npt.NDArray[np.complex128]): Retarded Green's Function_,        (#orbital/#ranks, #energy)
        gl_transposed (npt.NDArray[np.complex128]): Transposed Lesser Green's Function (#orbital/#ranks, #energy)
        pg       (npt.NDArray[np.complex128]): Greater polarization,              (#orbital/#ranks, #energy)
        pl       (npt.NDArray[np.complex128]): Lesser polarization,               (#orbital/#ranks, #energy)
        pr       (npt.NDArray[np.complex128]): Retarded polarization,             (#orbital/#ranks, #energy)
        streams (typing.List[cp.cuda.Stream]): List of streams to overlay memcpy and computations, at least four streams needed
        batch_size                      (int): batch size, should be set to fully utilize the gpu
    """

    # number of energy points and nonzero elements
    ne = gg.shape[1]
    no = gg.shape[0]

    # determine number of batches
    # batch over no
    batches = no // batch_size
    if batches == 0:
        # print("Too large batch size")
        batch_size = no
        batches = 1

    # load data to gpu and compute----------------------------------------------
    # allocate gpu memory
    gg_gpu = cp.empty((batch_size + no % batch_size, ne), dtype=np.complex128)
    gl_gpu = cp.empty((batch_size + no % batch_size, ne), dtype=np.complex128)
    gr_gpu = cp.empty((batch_size + no % batch_size, ne), dtype=np.complex128)
    gl_transposed_gpu = cp.empty((batch_size + no % batch_size, ne), dtype=np.complex128)

    for batch in range(batches):
        batch_start = batch * batch_size
        # last batch different, if not dividable
        batch_end = batch_size * (batch + 1) if batch != batches - 1 else batch_size * (batch + 1) + no % batch_size
        with streams[0]:
            gg_gpu[0:batch_end - batch_start].set(gg[batch_start:batch_end, :], stream=streams[0])
            gg_t_gpu = cp.fft.fft(gg_gpu[0:batch_end - batch_start], n=2 * ne, axis=1)
        with streams[1]:
            gl_gpu[0:batch_end - batch_start].set(gl[batch_start:batch_end, :], stream=streams[1])
            gl_t_gpu = cp.fft.fft(gl_gpu[0:batch_end - batch_start], n=2 * ne, axis=1)
        with streams[2]:
            gr_gpu[0:batch_end - batch_start].set(gr[batch_start:batch_end, :], stream=streams[2])
            gr_t_gpu = cp.fft.fft(gr_gpu[0:batch_end - batch_start], n=2 * ne, axis=1)
        with streams[3]:
            gl_transposed_gpu[0:batch_end - batch_start].set(gl_transposed[batch_start:batch_end, :], stream=streams[3])
            gl_t_transposed_gpu = cp.fft.fft(gl_transposed_gpu[0:batch_end - batch_start], n=2 * ne, axis=1)
            gl_t_mod_gpu = cp.roll(cp.flip(gl_t_transposed_gpu, axis=1), 1, axis=1)

        streams[1].synchronize()
        streams[3].synchronize()
        with streams[0]:
            pg_t_gpu = cp.multiply(gg_t_gpu, gl_t_mod_gpu)
            pg_gpu = cp.multiply(cp.fft.ifft(pg_t_gpu, axis=1), pre_factor)
            pl_gpu = -cp.conjugate(cp.roll(cp.flip(pg_gpu, axis=1), 1, axis=1))

        streams[0].synchronize()
        with streams[1]:
            pl_gpu = pl_gpu[:, :ne]
            cp.asnumpy(pl_gpu[0:batch_end - batch_start], out=pl[batch_start:batch_end, :], stream=streams[1])

        with streams[0]:
            pg_gpu = pg_gpu[:, :ne]
            cp.asnumpy(pg_gpu[0:batch_end - batch_start], out=pg[batch_start:batch_end, :], stream=streams[0])

        with streams[2]:
            pr_t_gpu = linalg_gpu.pr_special_gpu(gr_t_gpu, gl_t_mod_gpu, gl_t_gpu)
            pr_gpu = cp.multiply(cp.fft.ifft(pr_t_gpu, axis=1)[:, :ne], pre_factor)
            cp.asnumpy(pr_gpu[0:batch_end - batch_start], out=pr[batch_start:batch_end, :], stream=streams[2])

    for stream in streams:
        stream.synchronize()
