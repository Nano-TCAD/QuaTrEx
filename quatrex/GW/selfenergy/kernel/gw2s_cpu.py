# Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.
""" Functions to calculate the self-energies on the cpu. See README for more information. """

import numpy as np
import numpy.typing as npt
import typing
import sys
import os
import numba

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
) -> typing.Tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128], npt.NDArray[np.complex128]]:
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
    rgg_t = linalg_cpu.fft_numba(np.flip(gg, axis=1), ne2, no)
    rgl_t = linalg_cpu.fft_numba(np.flip(gl, axis=1), ne2, no)
    rgr_t = linalg_cpu.fft_numba(np.flip(gr, axis=1), ne2, no)

    # multiply elementwise for sigma_1 the normal term
    sg_t_1 = linalg_cpu.elementmul(gg_t, wg_t)
    sl_t_1 = linalg_cpu.elementmul(gl_t, wl_t)
    sr_t_1 = linalg_cpu.elementmul(gr_t, wl_t) + linalg_cpu.elementmul(gg_t, wr_t)

    # time reverse
    wr_t_mod = np.roll(np.flip(wr_t, axis=1), 1, axis=1)

    # multiply elementwise the energy reversed with difference of transposed and energy zero
    # see the README for derivation
    sg_t_2 = linalg_cpu.elementmul(rgg_t, wl_t[ij2ji, :] - np.repeat(wl[ij2ji, 0].reshape(-1, 1), 2 * ne, axis=1))
    sl_t_2 = linalg_cpu.elementmul(rgl_t, wg_t[ij2ji, :] - np.repeat(wg[ij2ji, 0].reshape(-1, 1), 2 * ne, axis=1))
    sr_t_2 = (
        linalg_cpu.elementmul(rgg_t, np.conjugate(wr_t_mod - np.repeat(wr[:, 0].reshape(-1, 1), 2 * ne, axis=1))) +
        linalg_cpu.elementmul(rgr_t, wg_t[ij2ji, :] - np.repeat(wg[:, 0].reshape(-1, 1), 2 * ne, axis=1)))

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


def gw2s_fft_cpu_3part_sr(
    pre_factor: np.complex128,
    ij2ji: npt.NDArray[np.int32],
    gg: npt.NDArray[np.complex128],
    gl: npt.NDArray[np.complex128],
    gr: npt.NDArray[np.complex128],
    wg: npt.NDArray[np.complex128],
    wl: npt.NDArray[np.complex128],
    wr: npt.NDArray[np.complex128],
) -> typing.Tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128], npt.NDArray[np.complex128]]:
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
    rgg_t = linalg_cpu.fft_numba(np.flip(gg, axis=1), ne2, no)
    rgl_t = linalg_cpu.fft_numba(np.flip(gl, axis=1), ne2, no)
    rgr_t = linalg_cpu.fft_numba(np.flip(gr, axis=1), ne2, no)

    # multiply elementwise for sigma_1 the normal term
    sg_t_1 = linalg_cpu.elementmul(gg_t, wg_t)
    sl_t_1 = linalg_cpu.elementmul(gl_t, wl_t)
    sr_t_1 = linalg_cpu.elementmul(gr_t, wl_t) + linalg_cpu.elementmul(gl_t, wr_t) + linalg_cpu.elementmul(gr_t, wr_t)

    # time reverse
    wr_t_mod = np.roll(np.flip(wr_t, axis=1), 1, axis=1)

    # multiply elementwise the energy reversed with difference of transposed and energy zero
    # see the README for derivation
    sg_t_2 = linalg_cpu.elementmul(rgg_t, wl_t[ij2ji, :] - np.repeat(wl[ij2ji, 0].reshape(-1, 1), 2 * ne, axis=1))
    sl_t_2 = linalg_cpu.elementmul(rgl_t, wg_t[ij2ji, :] - np.repeat(wg[ij2ji, 0].reshape(-1, 1), 2 * ne, axis=1))
    #sr_t_2 = (linalg_cpu.elementmul(rgg_t, np.conjugate(wr_t_mod - np.repeat(wr[:,0].reshape(-1,1), 2*ne, axis=1))) +
    #          linalg_cpu.elementmul(rgr_t, wg_t[ij2ji,:] - np.repeat(wg[:,0].reshape(-1,1), 2*ne, axis=1)))
    sr_t_2 = (
        linalg_cpu.elementmul(rgl_t, np.conjugate(wr_t_mod - np.repeat(wr[:, 0].reshape(-1, 1), 2 * ne, axis=1))) +
        linalg_cpu.elementmul(rgr_t, wg_t[ij2ji, :] - np.repeat(wg[ij2ji, 0].reshape(-1, 1), 2 * ne, axis=1)) +
        linalg_cpu.elementmul(rgr_t, np.conjugate(wr_t_mod - np.repeat(wr[:, 0].reshape(-1, 1), 2 * ne, axis=1))))

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
    pre_factor: np.complex128, gg: npt.NDArray[np.complex128], gl: npt.NDArray[np.complex128],
    gr: npt.NDArray[np.complex128], wg: npt.NDArray[np.complex128], wl: npt.NDArray[np.complex128],
    wr: npt.NDArray[np.complex128], wg_transposed: npt.NDArray[np.complex128], wl_transposed: npt.NDArray[np.complex128]
) -> typing.Tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128], npt.NDArray[np.complex128]]:
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
    rgg_t = linalg_cpu.fft_numba(np.flip(gg, axis=1), ne2, no)
    rgl_t = linalg_cpu.fft_numba(np.flip(gl, axis=1), ne2, no)
    rgr_t = linalg_cpu.fft_numba(np.flip(gr, axis=1), ne2, no)

    # multiply elementwise for sigma_1 the normal term
    sg_t_1 = linalg_cpu.elementmul(gg_t, wg_t)
    sl_t_1 = linalg_cpu.elementmul(gl_t, wl_t)
    sr_t_1 = linalg_cpu.elementmul(gr_t, wl_t) + linalg_cpu.elementmul(gg_t, wr_t)

    # time reverse
    wr_t_mod = np.roll(np.flip(wr_t, axis=1), 1, axis=1)

    # multiply elementwise the energy reversed with difference of transposed and energy zero
    # see the README for derivation
    sg_t_2 = linalg_cpu.elementmul(rgg_t,
                                   wl_transposed_t - np.repeat(wl_transposed[:, 0].reshape(-1, 1), 2 * ne, axis=1))
    sl_t_2 = linalg_cpu.elementmul(rgl_t,
                                   wg_transposed_t - np.repeat(wg_transposed[:, 0].reshape(-1, 1), 2 * ne, axis=1))
    sr_t_2 = (
        linalg_cpu.elementmul(rgg_t, np.conjugate(wr_t_mod - np.repeat(wr[:, 0].reshape(-1, 1), 2 * ne, axis=1))) +
        linalg_cpu.elementmul(rgr_t, wg_transposed_t - np.repeat(wg_transposed[:, 0].reshape(-1, 1), 2 * ne, axis=1)))

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


@numba.njit("(c16, c16[:,:], c16[:,:], c16[:,:], c16[:,:], c16[:,:], c16[:,:], c16[:,:], c16[:,:])",
            parallel=True,
            cache=True,
            nogil=True,
            error_model="numpy")
def gw2s_fft_mpi_cpu_3part_sr(
    pre_factor: np.complex128, gg: npt.NDArray[np.complex128], gl: npt.NDArray[np.complex128],
    gr: npt.NDArray[np.complex128], wg: npt.NDArray[np.complex128], wl: npt.NDArray[np.complex128],
    wr: npt.NDArray[np.complex128], wg_transposed: npt.NDArray[np.complex128], wl_transposed: npt.NDArray[np.complex128]
) -> typing.Tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128], npt.NDArray[np.complex128]]:
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
    rgg_t = linalg_cpu.fft_numba(linalg_cpu.flip(gg), ne2, no)
    rgl_t = linalg_cpu.fft_numba(linalg_cpu.flip(gl), ne2, no)
    rgr_t = linalg_cpu.fft_numba(linalg_cpu.flip(gr), ne2, no)

    # multiply elementwise for sigma_1 the normal term
    sg_t_1 = linalg_cpu.elementmul(gg_t, wg_t)
    sl_t_1 = linalg_cpu.elementmul(gl_t, wl_t)
    #sr_t_1 = linalg_cpu.elementmul(gr_t, wl_t) +  linalg_cpu.elementmul(gg_t, wr_t)
    sr_t_1 = linalg_cpu.elementmul(gr_t, wl_t) + linalg_cpu.elementmul(gl_t, wr_t) + linalg_cpu.elementmul(gr_t, wr_t)
    # time reverse
    wr_t_mod = linalg_cpu.reversal(wr_t)

    # multiply elementwise the energy reversed with difference of transposed and energy zero
    # see the README for derivation
    sg_t_2 = linalg_cpu.elementmul(rgg_t, linalg_cpu.substract_special(wl_transposed_t, wl_transposed[:, 0]))
    sl_t_2 = linalg_cpu.elementmul(rgl_t, linalg_cpu.substract_special(wg_transposed_t, wg_transposed[:, 0]))
    #sr_t_2 = (linalg_cpu.elementmul(rgg_t, np.conjugate(wr_t_mod - np.repeat(wr[:,0].reshape(-1,1), 2*ne, axis=1))) +
    #          linalg_cpu.elementmul(rgr_t, wg_transposed_t - np.repeat(wg[:,0].reshape(-1,1), 2*ne, axis=1)))
    sr_t_2 = (linalg_cpu.elementmul(rgl_t, np.conjugate(linalg_cpu.substract_special(wr_t_mod, wr[:, 0]))) +
              linalg_cpu.elementmul(rgr_t, linalg_cpu.substract_special(wg_transposed_t, wg_transposed[:, 0])) +
              linalg_cpu.elementmul(rgr_t, np.conjugate(linalg_cpu.substract_special(wr_t_mod, wr[:, 0]))))

    # ifft, cutoff and multiply with pre factor
    sg_1 = linalg_cpu.scalarmul_ifft_cutoff(sg_t_1, pre_factor, ne, no)
    sl_1 = linalg_cpu.scalarmul_ifft_cutoff(sl_t_1, pre_factor, ne, no)
    sr_1 = linalg_cpu.scalarmul_ifft_cutoff(sr_t_1, pre_factor, ne, no)

    sg_2 = linalg_cpu.flip(linalg_cpu.scalarmul_ifft_cutoff(sg_t_2, pre_factor, ne, no))
    sl_2 = linalg_cpu.flip(linalg_cpu.scalarmul_ifft_cutoff(sl_t_2, pre_factor, ne, no))
    sr_2 = linalg_cpu.flip(linalg_cpu.scalarmul_ifft_cutoff(sr_t_2, pre_factor, ne, no))

    sg = sg_1 + sg_2
    sl = sl_1 + sl_2
    sr = sr_1 + sr_2

    return (sg, sl, sr)

def gw2s_fft_mpi_cpu_3part_sr_bare(
    pre_factor: np.complex128, gg: npt.NDArray[np.complex128], gl: npt.NDArray[np.complex128],
    gr: npt.NDArray[np.complex128], wg: npt.NDArray[np.complex128], wl: npt.NDArray[np.complex128],
    wr: npt.NDArray[np.complex128], wg_transposed: npt.NDArray[np.complex128], wl_transposed: npt.NDArray[np.complex128], vh: npt.NDArray[np.complex128],
      rows: npt.NDArray[np.int32], cols: npt.NDArray[np.int32]
) -> typing.Tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128], npt.NDArray[np.complex128]]:
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

    wl_full = np.zeros((wl.shape[0], wl.shape[1] + (wl.shape[1] -1)), dtype = np.complex128)
    wg_full = np.zeros((wg.shape[0], wg.shape[1] + (wg.shape[1] -1)), dtype = np.complex128)

    ne_full = wl_full.shape[1]
    ne2_full = 2 * ne_full

    wl_full[:, (wl.shape[1]-1):] = wl
    wl_full[:, 0:wl.shape[1]-1] = linalg_cpu.flip(wg_transposed)[:,0:(wl.shape[1]-1)]

    wg_full[:, (wg.shape[1]-1):] = wg
    wg_full[:, 0:wg.shape[1]-1] = linalg_cpu.flip(wl_transposed)[:,0:(wg.shape[1]-1)]

    # todo possibility to avoid fft in global chain
    # fft
    gg_t = linalg_cpu.fft_numba(gg, ne2, no)
    gg_full_t = linalg_cpu.fft_numba(gg, ne2_full, no)
    gl_t = linalg_cpu.fft_numba(gl, ne2, no)
    gl_full_t = linalg_cpu.fft_numba(gl, ne2_full, no)
    gr_t = linalg_cpu.fft_numba(gr, ne2, no)
    wg_t = linalg_cpu.fft_numba(wg, ne2, no)
    wg_full_t = linalg_cpu.fft_numba(wg_full, ne2_full, no)
    wl_t = linalg_cpu.fft_numba(wl, ne2, no)
    wl_full_t = linalg_cpu.fft_numba(wl_full, ne2_full, no)
    wr_t = linalg_cpu.fft_numba(wr, ne2, no)
    wg_transposed_t = linalg_cpu.fft_numba(wg_transposed, ne2, no)
    wl_transposed_t = linalg_cpu.fft_numba(wl_transposed, ne2, no)

    # fft of energy reversed
    rgg_t = linalg_cpu.fft_numba(linalg_cpu.flip(gg), ne2, no)
    rgl_t = linalg_cpu.fft_numba(linalg_cpu.flip(gl), ne2, no)
    rgr_t = linalg_cpu.fft_numba(linalg_cpu.flip(gr), ne2, no)

    # multiply elementwise for sigma_1 the normal term
    sg_t_1 = linalg_cpu.elementmul(gg_t, wg_t)
    sl_t_1 = linalg_cpu.elementmul(gl_t, wl_t)
    #sr_t_1 = linalg_cpu.elementmul(gr_t, wl_t) +  linalg_cpu.elementmul(gg_t, wr_t)
    sr_t_1 = linalg_cpu.elementmul(gr_t, wl_t) + linalg_cpu.elementmul(gl_t, wr_t) + linalg_cpu.elementmul(gr_t, wr_t)
    sr_2part_a_t_1 = linalg_cpu.elementmul(gr_t, wl_t) + linalg_cpu.elementmul(gg_t, wr_t)
    sr_2part_b_t_1 = linalg_cpu.elementmul(gr_t, wg_t) + linalg_cpu.elementmul(gl_t, wr_t)
    # time reverse
    wr_t_mod = linalg_cpu.reversal(wr_t)

    # multiply elementwise the energy reversed with difference of transposed and energy zero
    # see the README for derivation
    sg_t_2 = linalg_cpu.elementmul(rgg_t, linalg_cpu.substract_special(wl_transposed_t, wl_transposed[:, 0]))
    sl_t_2 = linalg_cpu.elementmul(rgl_t, linalg_cpu.substract_special(wg_transposed_t, wg_transposed[:, 0]))
    #sr_t_2 = (linalg_cpu.elementmul(rgg_t, np.conjugate(wr_t_mod - np.repeat(wr[:,0].reshape(-1,1), 2*ne, axis=1))) +
    #          linalg_cpu.elementmul(rgr_t, wg_transposed_t - np.repeat(wg[:,0].reshape(-1,1), 2*ne, axis=1)))
    sr_t_2 = (linalg_cpu.elementmul(rgl_t, np.conjugate(linalg_cpu.substract_special(wr_t_mod, wr[:, 0]))) +
              linalg_cpu.elementmul(rgr_t, linalg_cpu.substract_special(wg_transposed_t, wg_transposed[:, 0])) +
              linalg_cpu.elementmul(rgr_t, np.conjugate(linalg_cpu.substract_special(wr_t_mod, wr[:, 0]))))
    
    sr_2part_a_t_2 = (linalg_cpu.elementmul(rgg_t, np.conjugate(linalg_cpu.substract_special(wr_t_mod, wr[:, 0]))) +
              linalg_cpu.elementmul(rgr_t, linalg_cpu.substract_special(wg_transposed_t, wg_transposed[:, 0])))
    
    sr_2part_b_t_2 = (linalg_cpu.elementmul(rgl_t, np.conjugate(linalg_cpu.substract_special(wr_t_mod, wr[:, 0]))) +
              linalg_cpu.elementmul(rgr_t, linalg_cpu.substract_special(wl_transposed_t, wl_transposed[:, 0])))
    

    #The full (aka W negative + positive energy range) sl and sg are calculated here
    # multiply elementwise for sigma_1 the normal term
    sg_t_full = linalg_cpu.elementmul(gg_full_t, wg_full_t)
    sl_t_full = linalg_cpu.elementmul(gl_full_t, wl_full_t)
    sr_t_full = np.zeros_like(sg_t_full, dtype = np.complex128)
    sr_t_full[:, :ne_full] = (sg_t_full-sl_t_full)[:, :ne_full]

    # ifft, cutoff and multiply with pre factor
    sg_1 = linalg_cpu.scalarmul_ifft_cutoff(sg_t_1, pre_factor, ne, no)
    sl_1 = linalg_cpu.scalarmul_ifft_cutoff(sl_t_1, pre_factor, ne, no)
    sr_1 = linalg_cpu.scalarmul_ifft_cutoff(sr_t_1, pre_factor, ne, no)
    sr_2_part_a_1 = linalg_cpu.scalarmul_ifft_cutoff(sr_2part_a_t_1, pre_factor, ne, no)
    sr_2_part_b_1 = linalg_cpu.scalarmul_ifft_cutoff(sr_2part_b_t_1, pre_factor, ne, no)

    sg_2 = linalg_cpu.flip(linalg_cpu.scalarmul_ifft_cutoff(sg_t_2, pre_factor, ne, no))
    sl_2 = linalg_cpu.flip(linalg_cpu.scalarmul_ifft_cutoff(sl_t_2, pre_factor, ne, no))
    sr_2 = linalg_cpu.flip(linalg_cpu.scalarmul_ifft_cutoff(sr_t_2, pre_factor, ne, no))
    sr_2_part_a_2 = linalg_cpu.flip(linalg_cpu.scalarmul_ifft_cutoff(sr_2part_a_t_2, pre_factor, ne, no))
    sr_2_part_b_2 = linalg_cpu.flip(linalg_cpu.scalarmul_ifft_cutoff(sr_2part_b_t_2, pre_factor, ne, no))

    sg_full = linalg_cpu.scalarmul_ifft_cutoff(sg_t_full, pre_factor, ne_full, no)
    sl_full = linalg_cpu.scalarmul_ifft_cutoff(sl_t_full, pre_factor, ne_full, no)
    sr_full = linalg_cpu.scalarmul_ifft_cutoff(sr_t_full, pre_factor, ne_full, no)

    #Calculating the truncated fock part
    vh1d = np.asarray(vh[rows, cols].reshape(-1))
    gl_density = np.imag(np.sum(gl, axis = 1))
    rSigmaRF = -np.multiply(gl_density, vh1d).reshape((gl_density.shape[0],1)) * np.abs(pre_factor/2)
    rSigmaRF = np.tile(rSigmaRF, (1,ne))

    sg = sg_1 + sg_2
    sl = sl_1 + sl_2
    sr = sr_1 + sr_2
    sr_time = sr_full[:, ne-1:] + rSigmaRF
    sr_2part_a = sr_2_part_a_1 + sr_2_part_a_2
    sr_2part_b = sr_2_part_b_1 + sr_2_part_b_2

    return (sg, sl, sr)
