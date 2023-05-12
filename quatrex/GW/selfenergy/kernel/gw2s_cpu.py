"""
Functions to calculate the self-energies on the cpu.
See README for more information.
"""
import numpy as np
import numpy.typing as npt
import typing
import sys
import os

main_path = os.path.abspath(os.path.dirname(__file__))
parent_path = os.path.abspath(os.path.join(main_path, "..", "..", ".."))
sys.path.append(parent_path)

from utils import linalg_cpu

def gw2s_fft_cpu(
    pre_factor: np.complex128,
    ij2ji: npt.NDArray[np.int32],
    gg: npt.NDArray[np.complex128],
    gl: npt.NDArray[np.complex128],
    gr: npt.NDArray[np.complex128],
    wg: npt.NDArray[np.complex128],
    wl: npt.NDArray[np.complex128],
    wr: npt.NDArray[np.complex128],
) -> typing.Tuple[npt.NDArray[np.complex128], 
                  npt.NDArray[np.complex128], 
                  npt.NDArray[np.complex128]]:
    """Calculate the self energy with fft on the cpu(see file description todo). 
        The inputs are the pre factor, the Green's Functions
        and the screened interactions.
        Takes into account the energy grid cutoff

    Args:
        pre_factor      (np.complex128): pre_factor, multiplied at the end
        ij2ji   (npt.NDArray[np.int32]): mapping to transposed matrix, (#orbital)
        gg (npt.NDArray[np.complex128]): Greater Green's Function,     (#orbital, #energy)
        gl (npt.NDArray[np.complex128]): Lesser Green's Function,      (#orbital, #energy)
        gr (npt.NDArray[np.complex128]): Retarded Green's Function,    (#orbital, #energy)
        wg (npt.NDArray[np.complex128]): Greater screened interaction, (#orbital, #energy)
        wl (npt.NDArray[np.complex128]): _description_
        wr (npt.NDArray[np.complex128]): Retarded screened interaction,(#orbital, #energy)

    Returns:
        typing.Tuple[npt.NDArray[np.complex128], Greater self energy  (#orbital, #energy)
                     npt.NDArray[np.complex128], Lesser self energy   (#orbital, #energy)
                     npt.NDArray[np.complex128]  Retarded self energy (#orbital, #energy)
                    ]
    """
    # number of energy points and nnz
    ne = gg.shape[1]
    no = gg.shape[0]
    ne2 = 2 * ne

    # todo possibility to avoid fft in global chain
    # fft
    gg_t = linalg_cpu.fft_numba(gg, ne2, no)
    gl_t = linalg_cpu.fft_numba(gl, ne2, no)
    gr_t = linalg_cpu.fft_numba(gr, ne2, no)
    wg_t = linalg_cpu.fft_numba(wg, ne2, no)
    wl_t = linalg_cpu.fft_numba(wl, ne2, no)
    wr_t = linalg_cpu.fft_numba(wr, ne2, no)

    # fft of energy reversed
    rgg_t =  linalg_cpu.fft_numba(np.flip(gg, axis=1), ne2, no)
    rgl_t =  linalg_cpu.fft_numba(np.flip(gl, axis=1), ne2, no)
    rgr_t =  linalg_cpu.fft_numba(np.flip(gr, axis=1), ne2, no)

    # multiply elementwise for sigma_1 the normal term
    sg_t_1 = linalg_cpu.elementmul(gg_t, wg_t)
    sl_t_1 = linalg_cpu.elementmul(gl_t, wl_t)
    sr_t_1 = linalg_cpu.elementmul(gr_t, wl_t) +  linalg_cpu.elementmul(gg_t, wr_t)

    # time reverse 
    wr_t_mod = np.roll(np.flip(wr_t, axis=1), 1, axis=1)

    # multiply elementwise the energy reversed with difference of transposed and energy zero
    # see the README for derivation
    sg_t_2 = linalg_cpu.elementmul(rgg_t, wl_t[ij2ji,:] - np.repeat(wl[ij2ji,0].reshape(-1,1), 2*ne, axis=1))
    sl_t_2 = linalg_cpu.elementmul(rgl_t, wg_t[ij2ji,:] - np.repeat(wg[ij2ji,0].reshape(-1,1), 2*ne, axis=1))
    sr_t_2 = (linalg_cpu.elementmul(rgg_t, np.conjugate(wr_t_mod - np.repeat(wr[:,0].reshape(-1,1), 2*ne, axis=1))) +
              linalg_cpu.elementmul(rgr_t, wg_t[ij2ji,:] - np.repeat(wg[:,0].reshape(-1,1), 2*ne, axis=1)))


    # ifft, cutoff and multiply with pre factor
    sg_1 = linalg_cpu.scalarmul_ifft_cutoff(sg_t_1, pre_factor, ne, no)
    sl_1 = linalg_cpu.scalarmul_ifft_cutoff(sl_t_1, pre_factor, ne, no)
    sr_1 = linalg_cpu.scalarmul_ifft_cutoff(sr_t_1, pre_factor, ne, no)

    sg_2 = np.flip(linalg_cpu.scalarmul_ifft_cutoff(sg_t_2, pre_factor, ne, no), axis=1)
    sl_2 = np.flip(linalg_cpu.scalarmul_ifft_cutoff(sl_t_2, pre_factor, ne, no), axis=1)
    sr_2 = np.flip(linalg_cpu.scalarmul_ifft_cutoff(sr_t_2, pre_factor, ne, no), axis=1)


    sg = sg_1 + sg_2
    sl = sl_1 + sl_2
    sr = sr_1 + sr_2

    return (sg, sl, sr)

def gw2s_fft_mpi_cpu(
    pre_factor: np.complex128,
    gg: npt.NDArray[np.complex128],
    gl: npt.NDArray[np.complex128],
    gr: npt.NDArray[np.complex128],
    wg: npt.NDArray[np.complex128],
    wl: npt.NDArray[np.complex128],
    wr: npt.NDArray[np.complex128],
    wg_transposed: npt.NDArray[np.complex128],
    wl_transposed: npt.NDArray[np.complex128]
) -> typing.Tuple[npt.NDArray[np.complex128], 
                  npt.NDArray[np.complex128], 
                  npt.NDArray[np.complex128]]:
    """Calculate the self energy with fft on the cpu(see file description todo). 
        The inputs are the pre factor, the Green's Functions
        and the screened interactions.
        Takes into account the energy grid cutoff
        MPI version, needs additionally the transposed as a input

    Args:
        pre_factor      (np.complex128): pre_factor, multiplied at the end
        ij2ji   (npt.NDArray[np.int32]): mapping to transposed matrix, (#orbital)
        gg (npt.NDArray[np.complex128]): Greater Green's Function,     (#orbital, #energy)
        gl (npt.NDArray[np.complex128]): Lesser Green's Function,      (#orbital, #energy)
        gr (npt.NDArray[np.complex128]): Retarded Green's Function,    (#orbital, #energy)
        wg (npt.NDArray[np.complex128]): Greater screened interaction, (#orbital, #energy)
        wl (npt.NDArray[np.complex128]): Lesser screened interaction,  (#orbital, #energy)
        wr (npt.NDArray[np.complex128]): Retarded screened interaction,(#orbital, #energy)
        wg_transposed (npt.NDArray[np.complex128]): Greater screened interaction, transposed in orbitals, (#orbital, #energy)
        wl_transposed (npt.NDArray[np.complex128]): Lesser screened interaction, transposed in orbitals,  (#orbital, #energy)

    Returns:
        typing.Tuple[npt.NDArray[np.complex128], Greater self energy  (#orbital, #energy)
                     npt.NDArray[np.complex128], Lesser self energy   (#orbital, #energy)
                     npt.NDArray[np.complex128]  Retarded self energy (#orbital, #energy)
                    ]
    """
    # number of energy points and nnz
    ne = gg.shape[1]
    no = gg.shape[0]
    ne2 = 2 * ne

    # todo possibility to avoid fft in global chain
    # fft
    gg_t = linalg_cpu.fft_numba(gg, ne2, no)
    gl_t = linalg_cpu.fft_numba(gl, ne2, no)
    gr_t = linalg_cpu.fft_numba(gr, ne2, no)
    wg_t = linalg_cpu.fft_numba(wg, ne2, no)
    wl_t = linalg_cpu.fft_numba(wl, ne2, no)
    wr_t = linalg_cpu.fft_numba(wr, ne2, no)
    wg_transposed_t = linalg_cpu.fft_numba(wg_transposed, ne2, no)
    wl_transposed_t = linalg_cpu.fft_numba(wl_transposed, ne2, no)

    # fft of energy reversed
    rgg_t =  linalg_cpu.fft_numba(np.flip(gg, axis=1), ne2, no)
    rgl_t =  linalg_cpu.fft_numba(np.flip(gl, axis=1), ne2, no)
    rgr_t =  linalg_cpu.fft_numba(np.flip(gr, axis=1), ne2, no)

    # multiply elementwise for sigma_1 the normal term
    sg_t_1 = linalg_cpu.elementmul(gg_t, wg_t)
    sl_t_1 = linalg_cpu.elementmul(gl_t, wl_t)
    sr_t_1 = linalg_cpu.elementmul(gr_t, wl_t) +  linalg_cpu.elementmul(gg_t, wr_t)

    # time reverse 
    wr_t_mod = np.roll(np.flip(wr_t, axis=1), 1, axis=1)

    # multiply elementwise the energy reversed with difference of transposed and energy zero
    # see the README for derivation
    sg_t_2 = linalg_cpu.elementmul(rgg_t, wl_transposed_t - np.repeat(wl_transposed[:,0].reshape(-1,1), 2*ne, axis=1))
    sl_t_2 = linalg_cpu.elementmul(rgl_t, wg_transposed_t - np.repeat(wg_transposed[:,0].reshape(-1,1), 2*ne, axis=1))
    sr_t_2 = (linalg_cpu.elementmul(rgg_t, np.conjugate(wr_t_mod - np.repeat(wr[:,0].reshape(-1,1), 2*ne, axis=1))) +
              linalg_cpu.elementmul(rgr_t, wg_transposed_t - np.repeat(wg[:,0].reshape(-1,1), 2*ne, axis=1)))


    # ifft, cutoff and multiply with pre factor
    sg_1 = linalg_cpu.scalarmul_ifft_cutoff(sg_t_1, pre_factor, ne, no)
    sl_1 = linalg_cpu.scalarmul_ifft_cutoff(sl_t_1, pre_factor, ne, no)
    sr_1 = linalg_cpu.scalarmul_ifft_cutoff(sr_t_1, pre_factor, ne, no)

    sg_2 = np.flip(linalg_cpu.scalarmul_ifft_cutoff(sg_t_2, pre_factor, ne, no), axis=1)
    sl_2 = np.flip(linalg_cpu.scalarmul_ifft_cutoff(sl_t_2, pre_factor, ne, no), axis=1)
    sr_2 = np.flip(linalg_cpu.scalarmul_ifft_cutoff(sr_t_2, pre_factor, ne, no), axis=1)


    sg = sg_1 + sg_2
    sl = sl_1 + sl_2
    sr = sr_1 + sr_2

    return (sg, sl, sr)

