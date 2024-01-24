import numpy as np
import numpy.typing as npt
import cupy as cp
import cupyx as cpx
import typing
from scipy import sparse

from quatrex.utils.bsr import bsr_matrix

def sparse2block_energy_gpu(S, out_diag: npt.NDArray[np.complex128],
                        out_upper: npt.NDArray[np.complex128], out_lower: npt.NDArray[np.complex128], bmax: np.ndarray,
                        bmin: np.ndarray, energy: np.ndarray):
    """
    Transforms a sparse matrix into the three dense block matrices

    Args:
        S (cp.sparse.spmatrix): Overlap on GPU
        out_diag (npt.NDArray[np.complex128]): Diagonal block tensor on GPU
        out_upper (npt.NDArray[np.complex128]): Upper block tensor on GPU
        out_lower (npt.NDArray[np.complex128]): Lower block tensor on GPU
        bmax (np.ndarray): End index of each block
        bmin (np.ndarray): Start index of each block
        energy (np.ndarray): Energy values
    """
    # number of blocks
    nb = bmin.size
    for k in range(energy.shape[0]):
        in_sparse = S[k]
        for i in range(nb):
            out_diag[i, k, :, :] = in_sparse[bmin[i]:bmax[i] + 1, bmin[i]:bmax[i] + 1].toarray()
        for i in range(nb - 1):
            out_upper[i, k, :, :] = in_sparse[bmin[i]:bmax[i] + 1, bmin[i + 1]:bmax[i + 1] + 1].toarray()
            out_lower[i, k, :, :] = in_sparse[bmin[i + 1]:bmax[i + 1] + 1, bmin[i]:bmax[i] + 1].toarray()

def sparse2block_gpu(S, out_diag: npt.NDArray[np.complex128],
                        out_upper: npt.NDArray[np.complex128], out_lower: npt.NDArray[np.complex128], bmax: np.ndarray,
                        bmin: np.ndarray):
    """
    Transforms a sparse matrix into the three dense block matrices

    Args:
        S (cp.sparse.spmatrix): Overlap on GPU
        out_diag (npt.NDArray[np.complex128]): Diagonal block tensor on GPU
        out_upper (npt.NDArray[np.complex128]): Upper block tensor on GPU
        out_lower (npt.NDArray[np.complex128]): Lower block tensor on GPU
        bmax (np.ndarray): End index of each block
        bmin (np.ndarray): Start index of each block
        energy (np.ndarray): Energy values
    """
    # number of blocks
    nb = bmin.size
    for i in range(nb):
        out_diag[i, :, :] = S[bmin[i]:bmax[i] + 1, bmin[i]:bmax[i] + 1].toarray()
    for i in range(nb - 1):
        out_upper[i, :, :] = S[bmin[i]:bmax[i] + 1, bmin[i + 1]:bmax[i + 1] + 1].toarray()
        out_lower[i, :, :] = S[bmin[i + 1]:bmax[i + 1] + 1, bmin[i]:bmax[i] + 1].toarray()