"""
Few line linalg functions used in g2p and gw2sigma.
Used in gpu kernels.
"""
import numpy as np
import cupy as cp
from cupyx import jit

def aloc_pinned_empty_like(
        array: np.ndarray
    ) -> np.ndarray:
    """
    Allocates pinned memory of the same size as array.

    Args:
        array (np.ndarray): Input array

    Returns:
        np.ndarray: Pinned memory of the same size as array
    """
    # constructing pinned memory
    mem = cp.cuda.alloc_pinned_memory(array.nbytes)
    src = np.frombuffer(
                mem, array.dtype, array.size).reshape(array.shape)
    return src

def aloc_pinned_filled(
        array: np.ndarray
    ) -> np.ndarray:
    """
    Allocates pinned memory of the same size as array
    and fills it with the array values.

    Args:
        array (np.ndarray): Input array

    Returns:
        np.ndarray: Pinned memory of the same size as array
    """
    # constructing pinned memory
    mem = cp.cuda.alloc_pinned_memory(array.nbytes)
    src = np.frombuffer(
                mem, array.dtype, array.size).reshape(array.shape)
    src[...] = array
    return src

@cp.fuse()
def pr_special_gpu(gr: cp.ndarray, gl_mod: cp.ndarray, gl: cp.ndarray) -> cp.ndarray:
    """Calculates elementwise gr*gl_mod + gl*gr^* on the gpu

    Args:
        gr (cp.ndarray): complex array size (#orbital,#energy)
        gl_mod (cp.ndarray): complex array size (#orbital,#energy)
        gl (cp.ndarray): complex array size (#orbital,#energy)

    Returns:
        cp.ndarray: gr*gl_mod + gl*gr^*
    """
    return gr * gl_mod + gl * cp.conjugate(gr)

@jit.rawkernel()
def reversal_gpu(g1: cp.ndarray, g1_transposed: cp.ndarray, sizex: np.int32, sizey: np.int32) -> cp.ndarray:
    """reverses the array g1 along the energy axis

        Example:
        pl = cp.empty_like(gg, dtype=cp.cdouble)
        num_threads: np.int32 = 256
        num_blocks: np.int32 = (no + num_threads - 1) // num_threads
        helper.reversal_gpu[num_blocks, num_threads](pg,pl,no,ne)
        
    Args:
        g1 (cp.ndarray): input, (#orbital,#energy) 
        g1_transposed (cp.ndarray): output, (#orbital,#energy)
        sizex (np.int32): #orbital
        sizey (np.int32): #energy

    Returns:
        cp.ndarray: _description_
    """
    idx = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x
    if idx < sizex:
        g1_transposed[idx, 0] = -cp.conjugate(g1[idx, 0])
        for j in range(1,sizey):
            g1_transposed[idx, j] = -cp.conjugate(g1[idx, 2*sizey-j])
            