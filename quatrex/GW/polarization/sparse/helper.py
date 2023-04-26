"""Few line helper functions used in g2p_sparse.py
    Most compiled with numba to run parallel on the cpu"""
import numpy as np
import numpy.typing as npt
import typing
import numba
from scipy import fft
#import cupy as cp
#from cupyx import jit

@numba.njit("(c16[:,:], i4[:])",
            parallel=True,
            cache=True,
            nogil=True,
            error_model="numpy")
def reversal_transpose(g1: npt.NDArray[np.complex128], ij2ji: npt.NDArray[np.int32]) -> npt.NDArray[np.complex128]:
    """Reverses data in time and transposes in orbital space
        Scaling: Halves with 4 numba threads

        Single core alternative (Can not be compiled with numba):
        out = np.roll(np.flip(g1, axis=1), 1, axis=1)[ij2ji, :]

    Args:
        g1 (npt.NDArray[np.complex128]): 2D array: orbital * energy
        ij2ji (npt.NDArray[np.int32]): Mapping to transposed

    Returns:
        npt.NDArray[np.complex128]: np.roll(
            np.flip(g1, axis=1), 1, axis=1)[ij2ji, :]
    """
    out: npt.NDArray[np.complex128] = np.empty_like(g1)
    tmp: npt.NDArray[np.complex128] = np.empty_like(g1)
    no:  np.int32                   = np.shape(g1)[0]
    ne:  np.int32                   = np.shape(g1)[1]
    for i in numba.prange(no):
        for j in range(ne):
            tmp[i, j] = g1[i, -j]
    for i in numba.prange(no):
        out[ij2ji[i], :] = tmp[i, :]
    return out

@numba.njit("(c16[:,:], c16)",
            parallel=True,
            cache=True,
            nogil=True,
            error_model="numpy")
def scalarmul(g1: npt.NDArray[np.complex128],
              fac: complex) -> npt.NDArray[np.complex128]:
    """Multiplies elementwise g1 with fac
        Scales multicore with numba, but not ideal scaling
        Did not parallelize with openBLAS 
        Alternative multicore:
        out = np.empty_like(g1)
        for i in numba.prange(g1.shape[0]):
            for j in numba.prange(g1.shape[1]):
                out[i, j] = g1[i, j] * fac
    Args:
        g1 (npt.NDArray[np.complex128]): 2D matrix
        fac (np.complex128): complex double

    Returns:
        npt.NDArray[np.complex128]: fac*g1
    """
    out = np.multiply(g1, fac)
    return out

@numba.njit("(c16[:,:], c16[:,:])",
            parallel=True,
            cache=True,
            nogil=True,
            error_model="numpy")
def elementmul(g1: npt.NDArray[np.complex128],
               g2: npt.NDArray[np.complex128]) -> npt.NDArray[np.complex128]:
    """Multiplies elementwise g1 and g2, both with same size
        Scales multicore with numba, but not ideal scaling
        Did not parallelize with openBLAS 
        Alternative multicore:
        out = np.empty_like(g1)
        for i in numba.prange(g1.shape[0]):
            for j in numba.prange(g1.shape[1]):
                out[i, j] = g1[i, j] * g2[i, j]
    Args:
        g1 (npt.NDArray[np.complex128]): 2D Matrix
        g2 (npt.NDArray[np.complex128]): 2D Matrix

    Returns:
        npt.NDArray[np.complex128]: g1*g2,
    """
    out: npt.NDArray[np.complex128] = np.multiply(g1, g2)
    return out

@numba.njit("(c16[:,:], i8, i8)",
            parallel=True,
            cache=True,
            nogil=True,
            error_model="numpy")
def fft_numba(g1: npt.NDArray[np.complex128], ne2: np.int64, no: np.int64) -> npt.NDArray[np.complex128]:
    """Tries to compile scipy fft with numba

    Args:
        g1 (npt.NDArray[np.complex128]): 2D matrix
        ne (np.int32): number energy points
        workers (np.int32): #workers for ifft

    Returns:
        npt.NDArray[np.complex128]: fft(g1) with padding ne, ne2 = 2*ne
    """
    out: npt.NDArray[np.complex128] = np.empty((no,ne2), dtype=np.complex128)
    for i in numba.prange(no):
        out[i,:] = fft.fft(g1[i,:], n=ne2, workers=1)
    return out

@numba.njit("(c16[:,:], c16, i8, i8)",
            parallel=True,
            cache=True,
            nogil=True,
            error_model="numpy")
def scalarmul_ifft(g1: npt.NDArray[np.complex128], fac: np.complex128, ne: np.int64, no: np.int64) -> npt.NDArray[np.complex128]:
    """IFFT, crops, and multiplies elementwise g1 with fac
        Needs RocketFFT! https://github.com/styfenschaer/rocket-fft
        Numba scalarmul and fft separately takes more or less the same time
        Alternative multicore:
        g1 = fft.ifft(g1, axis=1, workers=workers)[:, :ne]
        out = np.multiply(g1, fac)
    Args:
        g1 (npt.NDArray[np.complex128]): 2D matrix
        fac (np.complex128): complex double
        ne (np.int32): #energy points
        workers(np.int32): #workers for ifft

    Returns:
        npt.NDArray[np.complex128]: fac*ifft(g1)[:,:ne]
    """
    out = np.empty((no,ne), dtype=np.complex128)
    for i in numba.prange(no):
        out[i,:] = fft.ifft(g1[i,:], workers=1)[:ne]
    for i in numba.prange(no):
        for j in numba.prange(ne):
            out[i, j] = out[i, j] * fac
    return out

@numba.njit("(i4, i4, c16[:,:], c16[:,:])",
            parallel=True,
            cache=True,
            nogil=True,
            error_model="numpy")
def elementmult_chunk(start: np.int32, end: np.int32, g1: npt.NDArray[np.complex128], g2: npt.NDArray[np.complex128]) -> npt.NDArray[np.complex128]:
    """Function to elementwise multiply chunks of g1 and g2
        Todo not tested if njit + parallel true works

    Args:
        g1 (npt.NDArray[np.complex128]): First array to multiply
        g2 (npt.NDArray[np.complex128]): Second array to multiply

    Returns:
        npt.NDArray[np.complex128]: g1[chunk,:] * g2[chunk,:]
    """
    out: npt.NDArray[np.complex128] = np.multiply(g1[start: end, :], g2[start: end, :])
    return out

def reversal_transpose_memory(g1: npt.NDArray[np.complex128], ij2ji_reversal: typing.Tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]) -> npt.NDArray[np.complex128]:
    """Reverses data in time and transposes in orbital space
        todo not working great, njit does not compile and np.ix_
        But not used yet
    Args:
        g1 (npt.NDArray[np.complex128]): 2D array: orbital * energy
        ij2ji_reversal (typing.Tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]): 2D Mapping to transposed and reversed

    Returns:
        npt.NDArray[np.complex128]: np.roll(
            np.flip(g1, axis=1), 1, axis=1)[ij2ji, :]
    """
    return g1[ij2ji_reversal]

# @cp.fuse()
# def retarded_special_gpu(gr: cp.ndarray, gl_mod: cp.ndarray, gl: cp.ndarray) -> cp.ndarray:
#     """Calculates elementwise gr*gl_mod + gl*gr^* on the gpu

#     Args:
#         gr (cp.ndarray): complex array size (#orbital,#energy)
#         gl_mod (cp.ndarray): complex array size (#orbital,#energy)
#         gl (cp.ndarray): complex array size (#orbital,#energy)

#     Returns:
#         cp.ndarray: gr*gl_mod + gl*gr^*
#     """
#     return gr * gl_mod + gl * cp.conjugate(gr)

# @jit.rawkernel()
# def reversal_gpu(g1: cp.ndarray, g1_trans: cp.ndarray, sizex: np.int32, sizey: np.int32) -> cp.ndarray:
#     """reverses the array g1 along the energy axis

#         Example:
#         pl = cp.empty_like(gg, dtype=cp.cdouble)
#         num_threads: np.int32 = 256
#         num_blocks: np.int32 = (no + num_threads - 1) // num_threads
#         helper.reversal_gpu[num_blocks, num_threads](pg,pl,no,ne)
        
#     Args:
#         g1 (cp.ndarray): input, (#orbital,#energy) 
#         g1_trans (cp.ndarray): output, (#orbital,#energy)
#         sizex (np.int32): #orbital
#         sizey (np.int32): #energy

#     Returns:
#         cp.ndarray: _description_
#     """
#     idx = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x
#     if idx < sizex:
#         g1_trans[idx, 0] = -cp.conjugate(g1[idx, 0])
#         for j in range(1,sizey):
#             g1_trans[idx, j] = -cp.conjugate(g1[idx, 2*sizey-j])
            