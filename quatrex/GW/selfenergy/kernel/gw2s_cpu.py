# Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.
""" Functions to calculate the self-energies on the cpu. See README for more information. """

from quatrex.utils import linalg_cpu
import numpy as np
import numpy.typing as npt
import typing
import sys
import os
import numba

main_path = os.path.abspath(os.path.dirname(__file__))
parent_path = os.path.abspath(os.path.join(main_path, "..", "..", ".."))
sys.path.append(parent_path)


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
    sr_t_1 = linalg_cpu.elementmul(
        gr_t, wl_t) + linalg_cpu.elementmul(gg_t, wr_t)

    # time reverse
    wr_t_mod = np.roll(np.flip(wr_t, axis=1), 1, axis=1)

    # multiply elementwise the energy reversed with difference of transposed and energy zero
    # see the README for derivation
    sg_t_2 = linalg_cpu.elementmul(
        rgg_t, wl_t[ij2ji, :] - np.repeat(wl[ij2ji, 0].reshape(-1, 1), 2 * ne, axis=1))
    sl_t_2 = linalg_cpu.elementmul(
        rgl_t, wg_t[ij2ji, :] - np.repeat(wg[ij2ji, 0].reshape(-1, 1), 2 * ne, axis=1))
    sr_t_2 = (
        linalg_cpu.elementmul(rgg_t, np.conjugate(wr_t_mod - np.repeat(wr[:, 0].reshape(-1, 1), 2 * ne, axis=1))) +
        linalg_cpu.elementmul(rgr_t, wg_t[ij2ji, :] - np.repeat(wg[:, 0].reshape(-1, 1), 2 * ne, axis=1)))

    # ifft, cutoff and multiply with pre factor
    sg_1 = linalg_cpu.scalarmul_ifft_cutoff(sg_t_1, pre_factor, ne, no)
    sl_1 = linalg_cpu.scalarmul_ifft_cutoff(sl_t_1, pre_factor, ne, no)
    sr_1 = linalg_cpu.scalarmul_ifft_cutoff(sr_t_1, pre_factor, ne, no)

    sg_2 = np.flip(linalg_cpu.scalarmul_ifft_cutoff(
        sg_t_2, pre_factor, ne, no), axis=1)
    sl_2 = np.flip(linalg_cpu.scalarmul_ifft_cutoff(
        sl_t_2, pre_factor, ne, no), axis=1)
    sr_2 = np.flip(linalg_cpu.scalarmul_ifft_cutoff(
        sr_t_2, pre_factor, ne, no), axis=1)

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
    sr_t_1 = linalg_cpu.elementmul(
        gr_t, wl_t) + linalg_cpu.elementmul(gl_t, wr_t) + linalg_cpu.elementmul(gr_t, wr_t)

    # time reverse
    wr_t_mod = np.roll(np.flip(wr_t, axis=1), 1, axis=1)

    # multiply elementwise the energy reversed with difference of transposed and energy zero
    # see the README for derivation
    sg_t_2 = linalg_cpu.elementmul(
        rgg_t, wl_t[ij2ji, :] - np.repeat(wl[ij2ji, 0].reshape(-1, 1), 2 * ne, axis=1))
    sl_t_2 = linalg_cpu.elementmul(
        rgl_t, wg_t[ij2ji, :] - np.repeat(wg[ij2ji, 0].reshape(-1, 1), 2 * ne, axis=1))
    # sr_t_2 = (linalg_cpu.elementmul(rgg_t, np.conjugate(wr_t_mod - np.repeat(wr[:,0].reshape(-1,1), 2*ne, axis=1))) +
    #          linalg_cpu.elementmul(rgr_t, wg_t[ij2ji,:] - np.repeat(wg[:,0].reshape(-1,1), 2*ne, axis=1)))
    sr_t_2 = (
        linalg_cpu.elementmul(rgl_t, np.conjugate(wr_t_mod - np.repeat(wr[:, 0].reshape(-1, 1), 2 * ne, axis=1))) +
        linalg_cpu.elementmul(rgr_t, wg_t[ij2ji, :] - np.repeat(wg[ij2ji, 0].reshape(-1, 1), 2 * ne, axis=1)) +
        linalg_cpu.elementmul(rgr_t, np.conjugate(wr_t_mod - np.repeat(wr[:, 0].reshape(-1, 1), 2 * ne, axis=1))))

    # ifft, cutoff and multiply with pre factor
    sg_1 = linalg_cpu.scalarmul_ifft_cutoff(sg_t_1, pre_factor, ne, no)
    sl_1 = linalg_cpu.scalarmul_ifft_cutoff(sl_t_1, pre_factor, ne, no)
    sr_1 = linalg_cpu.scalarmul_ifft_cutoff(sr_t_1, pre_factor, ne, no)

    sg_2 = np.flip(linalg_cpu.scalarmul_ifft_cutoff(
        sg_t_2, pre_factor, ne, no), axis=1)
    sl_2 = np.flip(linalg_cpu.scalarmul_ifft_cutoff(
        sl_t_2, pre_factor, ne, no), axis=1)
    sr_2 = np.flip(linalg_cpu.scalarmul_ifft_cutoff(
        sr_t_2, pre_factor, ne, no), axis=1)

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
    sr_t_1 = linalg_cpu.elementmul(
        gr_t, wg_t) + linalg_cpu.elementmul(gl_t, wr_t)

    # time reverse
    wr_t_mod = np.roll(np.flip(wr_t, axis=1), 1, axis=1)

    # multiply elementwise the energy reversed with difference of transposed and energy zero
    # see the README for derivation
    sg_t_2 = linalg_cpu.elementmul(rgg_t,
                                   wl_transposed_t - np.repeat(wl_transposed[:, 0].reshape(-1, 1), 2 * ne, axis=1))
    sl_t_2 = linalg_cpu.elementmul(rgl_t,
                                   wg_transposed_t - np.repeat(wg_transposed[:, 0].reshape(-1, 1), 2 * ne, axis=1))
    sr_t_2 = (
        linalg_cpu.elementmul(rgl_t, np.conjugate(wr_t_mod - np.repeat(wr[:, 0].reshape(-1, 1), 2 * ne, axis=1))) +
        linalg_cpu.elementmul(rgr_t, wl_transposed_t - np.repeat(wl_transposed[:, 0].reshape(-1, 1), 2 * ne, axis=1)))

    # ifft, cutoff and multiply with pre factor
    sg_1 = linalg_cpu.scalarmul_ifft_cutoff(sg_t_1, pre_factor, ne, no)
    sl_1 = linalg_cpu.scalarmul_ifft_cutoff(sl_t_1, pre_factor, ne, no)
    sr_1 = linalg_cpu.scalarmul_ifft_cutoff(sr_t_1, pre_factor, ne, no)

    sg_2 = np.flip(linalg_cpu.scalarmul_ifft_cutoff(
        sg_t_2, pre_factor, ne, no), axis=1)
    sl_2 = np.flip(linalg_cpu.scalarmul_ifft_cutoff(
        sl_t_2, pre_factor, ne, no), axis=1)
    sr_2 = np.flip(linalg_cpu.scalarmul_ifft_cutoff(
        sr_t_2, pre_factor, ne, no), axis=1)

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
    # sr_t_1 = linalg_cpu.elementmul(gr_t, wl_t) +  linalg_cpu.elementmul(gg_t, wr_t)
    sr_t_1 = linalg_cpu.elementmul(
        gr_t, wl_t) + linalg_cpu.elementmul(gl_t, wr_t) + linalg_cpu.elementmul(gr_t, wr_t)
    # time reverse
    wr_t_mod = linalg_cpu.reversal(wr_t)

    # multiply elementwise the energy reversed with difference of transposed and energy zero
    # see the README for derivation
    sg_t_2 = linalg_cpu.elementmul(rgg_t, linalg_cpu.substract_special(
        wl_transposed_t, wl_transposed[:, 0]))
    sl_t_2 = linalg_cpu.elementmul(rgl_t, linalg_cpu.substract_special(
        wg_transposed_t, wg_transposed[:, 0]))
    # sr_t_2 = (linalg_cpu.elementmul(rgg_t, np.conjugate(wr_t_mod - np.repeat(wr[:,0].reshape(-1,1), 2*ne, axis=1))) +
    #          linalg_cpu.elementmul(rgr_t, wg_transposed_t - np.repeat(wg[:,0].reshape(-1,1), 2*ne, axis=1)))
    sr_t_2 = (linalg_cpu.elementmul(rgl_t, np.conjugate(linalg_cpu.substract_special(wr_t_mod, wr[:, 0]))) +
              linalg_cpu.elementmul(rgr_t, linalg_cpu.substract_special(wg_transposed_t, wg_transposed[:, 0])) +
              linalg_cpu.elementmul(rgr_t, np.conjugate(linalg_cpu.substract_special(wr_t_mod, wr[:, 0]))))

    # ifft, cutoff and multiply with pre factor
    sg_1 = linalg_cpu.scalarmul_ifft_cutoff(sg_t_1, pre_factor, ne, no)
    sl_1 = linalg_cpu.scalarmul_ifft_cutoff(sl_t_1, pre_factor, ne, no)
    sr_1 = linalg_cpu.scalarmul_ifft_cutoff(sr_t_1, pre_factor, ne, no)

    sg_2 = linalg_cpu.flip(
        linalg_cpu.scalarmul_ifft_cutoff(sg_t_2, pre_factor, ne, no))
    sl_2 = linalg_cpu.flip(
        linalg_cpu.scalarmul_ifft_cutoff(sl_t_2, pre_factor, ne, no))
    sr_2 = linalg_cpu.flip(
        linalg_cpu.scalarmul_ifft_cutoff(sr_t_2, pre_factor, ne, no))

    sg = sg_1 + sg_2
    sl = sl_1 + sl_2
    sr = sr_1 + sr_2

    return (sg, sl, sr)

@numba.njit("(c16, c16[:,:], c16[:,:], c16[:,:], c16[:,:], c16[:,:], c16[:,:], c16[:], f8[:], i8, i8[:,:], i4[:,:])",
            parallel=True,
            cache=True,
            nogil=True,
            error_model="numpy")
def gw2s_fft_mpi_cpu_PI_sr(
    pre_factor: np.complex128, gg: npt.NDArray[np.complex128], gl: npt.NDArray[np.complex128],
    wg: npt.NDArray[np.complex128], wl: npt.NDArray[np.complex128],
    wg_transposed: npt.NDArray[np.complex128], wl_transposed: npt.NDArray[np.complex128], vh1D: npt.NDArray[np.complex128],
    energy: npt.NDArray[np.float64], rank: np.int32, disp: npt.NDArray[np.int32], count: npt.NDArray[np.int32]
) -> typing.Tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128], npt.NDArray[np.complex128]]:
    """Calculate the self energy with fft on the cpu(see file description todo). 
        The inputs are the pre factor, the Green's Functions
        and the screened interactions.
        Takes into account the energy grid cutoff
        MPI version, needs additionally the transposed as a input

    Args:
        pre_factor      (np.complex128): pre_factor, multiplied at the end (here it is 1j * dE/(2*pi))
        ij2ji   (npt.NDArray[np.int32]): mapping to transposed matrix, (#orbital)
        gg (npt.NDArray[np.complex128]): Greater Green's Function,     (#orbital, #energy)
        gl (npt.NDArray[np.complex128]): Lesser Green's Function,      (#orbital, #energy)
        wg (npt.NDArray[np.complex128]): Greater screened interaction, (#orbital, #energy)
        wl (npt.NDArray[np.complex128]): Lesser screened interaction,  (#orbital, #energy)
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
    # gg_full_t = linalg_cpu.fft_numba(gg, ne2_full, no)
    # gg_periodic = np.concatenate((gg, np.flip(gg, axis=1)[:,1:]), axis = 1)
    # gg_periodic_t = linalg_cpu.fft_numba(gg_periodic, ne2-1, no)
    gl_t = linalg_cpu.fft_numba(gl, ne2, no)
    # gl_full_t = linalg_cpu.fft_numba(gl, ne2_full, no)
    # gl_periodic = np.concatenate((gl, np.flip(gl, axis=1)[:,1:]), axis = 1)
    # gl_periodic_t = linalg_cpu.fft_numba(gl_periodic, ne2-1, no)
    # gr_t = linalg_cpu.fft_numba(gr, ne2, no)
    wg_t = linalg_cpu.fft_numba(wg, ne2, no)
    wl_t = linalg_cpu.fft_numba(wl, ne2, no)
    #wr_t = linalg_cpu.fft_numba(wr, ne2, no)
    wg_transposed_t = linalg_cpu.fft_numba(wg_transposed, ne2, no)
    wl_transposed_t = linalg_cpu.fft_numba(wl_transposed, ne2, no)

    # fft of energy reversed
    rgg_t = linalg_cpu.fft_numba(linalg_cpu.flip(gg), ne2, no)
    rgl_t = linalg_cpu.fft_numba(linalg_cpu.flip(gl), ne2, no)
    # rgr_t = linalg_cpu.fft_numba(linalg_cpu.flip(gr), ne2, no)

    # multiply elementwise for sigma_1 the normal term
    sg_t_1 = linalg_cpu.elementmul(gg_t, wg_t)
    sl_t_1 = linalg_cpu.elementmul(gl_t, wl_t)
    # sr_t_1 = linalg_cpu.elementmul(gr_t, wl_t) +  linalg_cpu.elementmul(gg_t, wr_t)
    # sr_t_1 = linalg_cpu.elementmul(gr_t, wl_t) + linalg_cpu.elementmul(gl_t, wr_t) + linalg_cpu.elementmul(gr_t, wr_t)
    # sr_2part_a_t_1 = linalg_cpu.elementmul(gr_t, wl_t) + linalg_cpu.elementmul(gg_t, wr_t)
    # sr_2part_b_t_1 = linalg_cpu.elementmul(gr_t, wg_t) + linalg_cpu.elementmul(gl_t, wr_t)
    # time reverse
    #wr_t_mod = linalg_cpu.reversal(wr_t)

    # multiply elementwise the energy reversed with difference of transposed and energy zero
    # see the README for derivation
    sg_t_2 = linalg_cpu.elementmul(rgg_t, linalg_cpu.substract_special(
        wl_transposed_t, wl_transposed[:, 0]))
    sl_t_2 = linalg_cpu.elementmul(rgl_t, linalg_cpu.substract_special(
        wg_transposed_t, wg_transposed[:, 0]))
    # sr_t_2 = (linalg_cpu.elementmul(rgg_t, np.conjugate(wr_t_mod - np.repeat(wr[:,0].reshape(-1,1), 2*ne, axis=1))) +
    #          linalg_cpu.elementmul(rgr_t, wg_transposed_t - np.repeat(wg[:,0].reshape(-1,1), 2*ne, axis=1)))
    # sr_t_2 = (linalg_cpu.elementmul(rgl_t, np.conjugate(linalg_cpu.substract_special(wr_t_mod, wr[:, 0]))) +
    #           linalg_cpu.elementmul(rgr_t, linalg_cpu.substract_special(wg_transposed_t, wg_transposed[:, 0])) +
    #           linalg_cpu.elementmul(rgr_t, np.conjugate(linalg_cpu.substract_special(wr_t_mod, wr[:, 0]))))

    # sr_2part_a_t_2 = (linalg_cpu.elementmul(rgg_t, np.conjugate(linalg_cpu.substract_special(wr_t_mod, wr[:, 0]))) +
    #           linalg_cpu.elementmul(rgr_t, linalg_cpu.substract_special(wg_transposed_t, wg_transposed[:, 0])))

    # sr_2part_b_t_2 = (linalg_cpu.elementmul(rgl_t, np.conjugate(linalg_cpu.substract_special(wr_t_mod, wr[:, 0]))) +
    #           linalg_cpu.elementmul(rgr_t, linalg_cpu.substract_special(wl_transposed_t, wl_transposed[:, 0])))

    # The full (aka W negative + positive energy range) sl and sg are calculated here
    # multiply elementwise for sigma_1 the normal term
    # sg_t_full = linalg_cpu.elementmul(gg_full_t, wg_full_t)
    # sl_t_full = linalg_cpu.elementmul(gl_full_t, wl_full_t)
    # sr_t_full = np.zeros_like(sg_t_full, dtype = np.complex128)
    # sr_t_full[:, :ne_full] = (sg_t_full-sl_t_full)[:, :ne_full]

    # ifft, cutoff and multiply with pre factor
    sg_1 = linalg_cpu.scalarmul_ifft_cutoff(sg_t_1, pre_factor, ne, no)
    sl_1 = linalg_cpu.scalarmul_ifft_cutoff(sl_t_1, pre_factor, ne, no)
    # sr_1 = linalg_cpu.scalarmul_ifft_cutoff(sr_t_1, pre_factor, ne, no)
    # sr_2_part_a_1 = linalg_cpu.scalarmul_ifft_cutoff(sr_2part_a_t_1, pre_factor, ne, no)
    # sr_2_part_b_1 = linalg_cpu.scalarmul_ifft_cutoff(sr_2part_b_t_1, pre_factor, ne, no)

    sg_2 = linalg_cpu.flip(
        linalg_cpu.scalarmul_ifft_cutoff(sg_t_2, pre_factor, ne, no))
    sl_2 = linalg_cpu.flip(
        linalg_cpu.scalarmul_ifft_cutoff(sl_t_2, pre_factor, ne, no))
    # sr_2 = linalg_cpu.flip(linalg_cpu.scalarmul_ifft_cutoff(sr_t_2, pre_factor, ne, no))
    # sr_2_part_a_2 = linalg_cpu.flip(linalg_cpu.scalarmul_ifft_cutoff(sr_2part_a_t_2, pre_factor, ne, no))
    # sr_2_part_b_2 = linalg_cpu.flip(linalg_cpu.scalarmul_ifft_cutoff(sr_2part_b_t_2, pre_factor, ne, no))

    # sg_full = linalg_cpu.scalarmul_ifft_cutoff(sg_t_full, pre_factor, ne_full, no)
    # sl_full = linalg_cpu.scalarmul_ifft_cutoff(sl_t_full, pre_factor, ne_full, no)
    # sr_full = linalg_cpu.scalarmul_ifft_cutoff(sr_t_full, pre_factor, ne_full, no)

    # Calculating the truncated fock part
    # vh1d = np.asarray(vh[rows, cols].reshape(-1))
    gl_density = np.imag(np.sum(gl, axis=1))
    rSigmaRF = -np.multiply(gl_density, vh1D[disp[0, rank]:disp[0, rank] + count[0, rank]]).reshape(
        (gl_density.shape[0], 1)) * np.abs(pre_factor)
    # rSigmaRF = np.tile(rSigmaRF, (1,ne))
    rSigmaRF = rSigmaRF.repeat(ne).reshape((-1, ne)).astype(np.complex128)

    # Using the principal value integral method for yet another sigma_r
    NE = len(energy)
    dE = energy[1] - energy[0]
    Evec = np.linspace(0, (NE-1)*dE, NE)

    # one_div_by_E = 1.0 / Evec
    # one_div_by_E[NE-1] = 0
    # one_div_by_E_t = np.fft.fft(one_div_by_E)
    one_div_by_E = np.concatenate((-1.0/(Evec[-1:0:-1]), np.array(
        [0.0], dtype=np.float64), 1/(Evec[1:]), np.array([1/(Evec[-1] + dE)], dtype=np.float64)))
    one_div_by_E_t = np.fft.fft(one_div_by_E)

    # Another experiment: comparing the two methods: principal value vs time domain manipulation to get gr from gl and gg
    # gr_t_new = np.zeros_like(gr_t, dtype = np.complex128)
    # gr_t_new[:,:ne] = (gg_t - gl_t)[:, :ne]
    # gr_new = linalg_cpu.scalarmul_ifft_cutoff(gr_t_new, 1.0, ne, no)
    # gr_t_new_periodic = np.zeros_like(gr_t_new, dtype = np.complex128)
    # gr_t_new_periodic[:, :ne] = (gg_periodic_t - gl_periodic_t)[:, :ne]
    # gr_new_periodic = linalg_cpu.scalarmul_ifft_cutoff(gr_t_new_periodic, 1.0, ne, no)

    # D1 = linalg_cpu.fft_numba(1j*np.imag(gg-gl), ne2, no)
    # gr_t_principale = np.multiply(D1, one_div_by_E_t)
    # gr_principale = linalg_cpu.scalarmul_ifft(gr_t_principale, pre_factor, ne, no)[:, ne:]

    sg = sg_1 + sg_2
    sl = sl_1 + sl_2
    # sr = sr_1 + sr_2

    SGmSL_t = linalg_cpu.fft_numba(1j*np.imag(sg-sl), ne2, no)
    rSigmaR_t = np.multiply(SGmSL_t, one_div_by_E_t)
    # rSigmaR_t = linalg_cpu.elementmul(SGmSL_t, one_div_by_E_t)
    rSigmaR = linalg_cpu.scalarmul_ifft_cutoff(
        rSigmaR_t, pre_factor*2, ne2, no)[:, ne-1:-1].astype(np.complex128)

    sr_principale = rSigmaR/2 + \
        (1j*np.imag(sg-sl)/2).astype(np.complex128) + rSigmaRF
    # sr_time = np.real(sr_full[:, ne-1:]) + 1j*np.imag(sg-sl)/2 + rSigmaRF
    # sr_2part_a = sr_2_part_a_1 + sr_2_part_a_2
    # sr_2part_b = sr_2_part_b_1 + sr_2_part_b_2

    return (sg, sl, sr_principale)


# @numba.njit(#"(c16, c16[:,:], c16[:,:], c16[:,:], c16[:,:], c16[:,:], c16[:,:], c16[:,:], c16[:,:], i4[:], f8[:,:], f8[:], i4, i4[:,:], i4[:,:])",
#             parallel=True,
#             cache=True,
#             nogil=True,
#             error_model="numpy")
def gw2s_fft_mpi_cpu_PI_sr_kpoint(
    pre_factor: np.complex128,
    gg: npt.NDArray[np.complex128],
    gl: npt.NDArray[np.complex128],
    gr: npt.NDArray[np.complex128],
    wg: npt.NDArray[np.complex128],
    wl: npt.NDArray[np.complex128],
    wr: npt.NDArray[np.complex128],
    wg_transposed: npt.NDArray[np.complex128],
    wl_transposed: npt.NDArray[np.complex128],
    num_kpoints: tuple[np.int32],  # has to be tuple because of numba and ind_mat
    vh1D_k: npt.NDArray[np.float64],
    energy: npt.NDArray[np.float64],
    rank: np.int32,
    disp: npt.NDArray[np.int32],
    count: npt.NDArray[np.int32]
) -> typing.Tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128], npt.NDArray[np.complex128]]:
    """Calculate the self energy with fft on the cpu(see file description todo).
        The inputs are the pre factor, the Green's Functions
        and the screened interactions.
        Takes into account the energy grid cutoff
        MPI version, needs additionally the transposed as a input.
        Includes k-points

    Args:
        pre_factor      (np.complex128): pre_factor, multiplied at the end (here it is 1j * dE/(2*pi))
        ij2ji   (npt.NDArray[np.int32]): mapping to transposed matrix, (#orbital)
        gg (npt.NDArray[np.complex128]): Greater Green's Function,     (#orbital, #energy)
        gl (npt.NDArray[np.complex128]): Lesser Green's Function,      (#orbital, #energy)
        gr (npt.NDArray[np.complex128]): Retarded Green's Function,    (#orbital, #energy)
        wg (npt.NDArray[np.complex128]): Greater screened interaction, (#orbital, #energy)
        wl (npt.NDArray[np.complex128]): Lesser screened interaction,  (#orbital, #energy)
        wr (npt.NDArray[np.complex128]): Retarded screened interaction,(#orbital, #energy)
        wg_transposed (npt.NDArray[np.complex128]): Greater screened interaction, transposed in orbitals, (#orbital, #energy)
        wl_transposed (npt.NDArray[np.complex128]): Lesser screened interaction, transposed in orbitals,  (#orbital, #energy)
        kpoints (npt.NDArray[np.int32]): kpoints indices
        vh1D_k (npt.NDArray[np.float64]): Flattened coulomb matrix, (#kpoints, #orbitals)
        energy (npt.NDArray[np.float64]): energy grid, (#energy)

    Returns:
        typing.Tuple[npt.NDArray[np.complex128], Greater self energy  (#orbital, #energy)
                     npt.NDArray[np.complex128], Lesser self energy   (#orbital, #energy)
                     npt.NDArray[np.complex128]  Retarded self energy (#orbital, #energy)
                    ]
    """
    # nnz
    no = gg.shape[0]
    # tot number of kpoints
    nkpts = np.prod(np.asarray(num_kpoints))
    # number of energy points and
    ne = int(gg.shape[1]/nkpts)
    ne2 = 2 * ne
    # Index matrix for kpoints
    ind_mat = np.arange(nkpts, dtype=np.int32).reshape(num_kpoints)

    # Create self-energy arrays.
    sg: npt.NDArray[np.complex128] = np.empty_like(gg, dtype=np.complex128)
    sl: npt.NDArray[np.complex128] = np.empty_like(gg, dtype=np.complex128)
    sr_principale: npt.NDArray[np.complex128] = np.empty_like(
        gg, dtype=np.complex128)

    for ki in range(nkpts):
        for kip in range(nkpts):

            # find correct energy range that correspond to the right k-point.
            # energy kpoint indices.
            ek = ki * ne
            # Below does not compile with numba
            mip = np.array(np.where(ind_mat == kip))
            mi = np.array(np.where(ind_mat == ki))
            md = tuple(mi-mip)
            ekd = int(ind_mat[md] * ne)

            # create the flattened Coulomb matrix
            vh1D = np.squeeze(vh1D_k[ki])

            # todo possibility to avoid fft in global chain
            # fft
            gg_t = linalg_cpu.fft_numba(gg[:, ek:ek+ne], ne2, no)
            gl_t = linalg_cpu.fft_numba(gl[:, ek:ek+ne], ne2, no)
            wg_t = linalg_cpu.fft_numba(wg[:, ekd:ekd+ne], ne2, no)
            wl_t = linalg_cpu.fft_numba(wl[:, ekd:ekd+ne], ne2, no)
            wg_transposed_t = linalg_cpu.fft_numba(
                wg_transposed[:, ekd:ekd+ne], ne2, no)
            wl_transposed_t = linalg_cpu.fft_numba(
                wl_transposed[:, ekd:ekd+ne], ne2, no)

            # fft of energy reversed
            rgg_t = linalg_cpu.fft_numba(
                linalg_cpu.flip(gg[:, ek:ek+ne]), ne2, no)
            rgl_t = linalg_cpu.fft_numba(
                linalg_cpu.flip(gl[:, ek:ek+ne]), ne2, no)

            # multiply elementwise for sigma_1 the normal term
            sg_t_1 = linalg_cpu.elementmul(gg_t, wg_t)
            sl_t_1 = linalg_cpu.elementmul(gl_t, wl_t)

            # multiply elementwise the energy reversed with difference of transposed and energy zero
            # see the README for derivation
            sg_t_2 = linalg_cpu.elementmul(rgg_t, linalg_cpu.substract_special(
                wl_transposed_t, wl_transposed[:, 0]))
            sl_t_2 = linalg_cpu.elementmul(rgl_t, linalg_cpu.substract_special(
                wg_transposed_t, wg_transposed[:, 0]))

            # ifft, cutoff and multiply with pre factor
            sg_1 = linalg_cpu.scalarmul_ifft_cutoff(sg_t_1, pre_factor, ne, no)
            sl_1 = linalg_cpu.scalarmul_ifft_cutoff(sl_t_1, pre_factor, ne, no)

            sg_2 = linalg_cpu.flip(
                linalg_cpu.scalarmul_ifft_cutoff(sg_t_2, pre_factor, ne, no))
            sl_2 = linalg_cpu.flip(
                linalg_cpu.scalarmul_ifft_cutoff(sl_t_2, pre_factor, ne, no))

            # Calculating the truncated fock part
            # vh1d = np.asarray(vh[rows, cols].reshape(-1))
            gl_density = np.imag(np.sum(gl[:, ek:ek+ne], axis=1))
            assert gl_density.shape[0] == vh1D[disp[0, rank]:disp[0, rank]+count[0, rank]].shape[0], f"gl_density.shape={gl_density.shape} vh1D.shape={vh1D[disp[0, rank]:disp[0, rank]+count[0, rank]].shape} rank={rank} disp={disp[0, rank]} count={count[0, rank]}"
            rSigmaRF = -np.multiply(gl_density, vh1D[disp[0, rank]:disp[0, rank] + count[0, rank]]).reshape(
                (gl_density.shape[0], 1)) * np.abs(pre_factor)
            # rSigmaRF = np.tile(rSigmaRF, (1,ne))
            rSigmaRF = rSigmaRF.repeat(ne).reshape((-1, ne)).astype(np.complex128)

            # Using the principal value integral method for yet another sigma_r
            NE = len(energy)
            dE = energy[1] - energy[0]
            Evec = np.linspace(0, (NE-1)*dE, NE)

            one_div_by_E = np.concatenate((-1.0/(Evec[-1:0:-1]), np.array(
                [0.0], dtype=np.float64), 1/(Evec[1:]), np.array([1/(Evec[-1] + dE)], dtype=np.float64)))
            one_div_by_E_t = np.fft.fft(one_div_by_E)

            sgk = sg_1 + sg_2
            slk = sl_1 + sl_2

            SGmSL_t = linalg_cpu.fft_numba(1j*np.imag(sgk-slk), ne2, no)
            rSigmaR_t = np.multiply(SGmSL_t, one_div_by_E_t)
            rSigmaR = linalg_cpu.scalarmul_ifft_cutoff(
                rSigmaR_t, pre_factor, ne2, no)[:, ne-1:-1].astype(np.complex128)

            srk_principale = rSigmaR/2 + (1j*np.imag(sgk-slk)/2).astype(np.complex128) + rSigmaRF

            sg[:, ek:ek+ne] += sgk
            sl[:, ek:ek+ne] += slk
            sr_principale[:, ek:ek+ne] += srk_principale
    return (sg, sl, sr_principale)


def gw2s_fft_mpi_cpu_3part_sr_bare(
    pre_factor: np.complex128, gg: npt.NDArray[np.complex128], gl: npt.NDArray[np.complex128],
    gr: npt.NDArray[np.complex128], wg: npt.NDArray[np.complex128], wl: npt.NDArray[np.complex128],
    wr: npt.NDArray[np.complex128], wg_transposed: npt.NDArray[np.complex128], wl_transposed: npt.NDArray[np.complex128], vh: npt.NDArray[np.complex128],
    energy: npt.NDArray[np.float64],
    rows: npt.NDArray[np.int32], cols: npt.NDArray[np.int32], rank: np.int32, disp: np.int32, count: np.int32
) -> typing.Tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128], npt.NDArray[np.complex128]]:
    """Calculate the self energy with fft on the cpu(see file description todo). 
        The inputs are the pre factor, the Green's Functions
        and the screened interactions.
        Takes into account the energy grid cutoff
        MPI version, needs additionally the transposed as a input

    Args:
        pre_factor      (np.complex128): pre_factor, multiplied at the end (here it is 1j * dE/(2*pi))
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

    wl_full = np.zeros(
        (wl.shape[0], wl.shape[1] + (wl.shape[1] - 1)), dtype=np.complex128)
    wg_full = np.zeros(
        (wg.shape[0], wg.shape[1] + (wg.shape[1] - 1)), dtype=np.complex128)

    ne_full = wl_full.shape[1]
    ne2_full = 2 * ne_full

    wl_full[:, (wl.shape[1]-1):] = wl
    wl_full[:, 0:wl.shape[1] -
            1] = linalg_cpu.flip(wg_transposed)[:, 0:(wl.shape[1]-1)]

    wg_full[:, (wg.shape[1]-1):] = wg
    wg_full[:, 0:wg.shape[1] -
            1] = linalg_cpu.flip(wl_transposed)[:, 0:(wg.shape[1]-1)]

    # todo possibility to avoid fft in global chain
    # fft
    gg_t = linalg_cpu.fft_numba(gg, ne2, no)
    gg_full_t = linalg_cpu.fft_numba(gg, ne2_full, no)
    gg_periodic = np.concatenate((gg, np.flip(gg, axis=1)[:, 1:]), axis=1)
    gg_periodic_t = linalg_cpu.fft_numba(gg_periodic, ne2-1, no)
    gl_t = linalg_cpu.fft_numba(gl, ne2, no)
    gl_full_t = linalg_cpu.fft_numba(gl, ne2_full, no)
    gl_periodic = np.concatenate((gl, np.flip(gl, axis=1)[:, 1:]), axis=1)
    gl_periodic_t = linalg_cpu.fft_numba(gl_periodic, ne2-1, no)
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
    # sr_t_1 = linalg_cpu.elementmul(gr_t, wl_t) +  linalg_cpu.elementmul(gg_t, wr_t)
    sr_t_1 = linalg_cpu.elementmul(
        gr_t, wl_t) + linalg_cpu.elementmul(gl_t, wr_t) + linalg_cpu.elementmul(gr_t, wr_t)
    sr_2part_a_t_1 = linalg_cpu.elementmul(
        gr_t, wl_t) + linalg_cpu.elementmul(gg_t, wr_t)
    sr_2part_b_t_1 = linalg_cpu.elementmul(
        gr_t, wg_t) + linalg_cpu.elementmul(gl_t, wr_t)
    # time reverse
    wr_t_mod = linalg_cpu.reversal(wr_t)

    # multiply elementwise the energy reversed with difference of transposed and energy zero
    # see the README for derivation
    sg_t_2 = linalg_cpu.elementmul(rgg_t, linalg_cpu.substract_special(
        wl_transposed_t, wl_transposed[:, 0]))
    sl_t_2 = linalg_cpu.elementmul(rgl_t, linalg_cpu.substract_special(
        wg_transposed_t, wg_transposed[:, 0]))
    # sr_t_2 = (linalg_cpu.elementmul(rgg_t, np.conjugate(wr_t_mod - np.repeat(wr[:,0].reshape(-1,1), 2*ne, axis=1))) +
    #          linalg_cpu.elementmul(rgr_t, wg_transposed_t - np.repeat(wg[:,0].reshape(-1,1), 2*ne, axis=1)))
    sr_t_2 = (linalg_cpu.elementmul(rgl_t, np.conjugate(linalg_cpu.substract_special(wr_t_mod, wr[:, 0]))) +
              linalg_cpu.elementmul(rgr_t, linalg_cpu.substract_special(wg_transposed_t, wg_transposed[:, 0])) +
              linalg_cpu.elementmul(rgr_t, np.conjugate(linalg_cpu.substract_special(wr_t_mod, wr[:, 0]))))

    sr_2part_a_t_2 = (linalg_cpu.elementmul(rgg_t, np.conjugate(linalg_cpu.substract_special(wr_t_mod, wr[:, 0]))) +
                      linalg_cpu.elementmul(rgr_t, linalg_cpu.substract_special(wg_transposed_t, wg_transposed[:, 0])))

    sr_2part_b_t_2 = (linalg_cpu.elementmul(rgl_t, np.conjugate(linalg_cpu.substract_special(wr_t_mod, wr[:, 0]))) +
                      linalg_cpu.elementmul(rgr_t, linalg_cpu.substract_special(wl_transposed_t, wl_transposed[:, 0])))

    # The full (aka W negative + positive energy range) sl and sg are calculated here
    # multiply elementwise for sigma_1 the normal term
    sg_t_full = linalg_cpu.elementmul(gg_full_t, wg_full_t)
    sl_t_full = linalg_cpu.elementmul(gl_full_t, wl_full_t)
    sr_t_full = np.zeros_like(sg_t_full, dtype=np.complex128)
    sr_t_full[:, :ne_full] = (sg_t_full-sl_t_full)[:, :ne_full]

    # ifft, cutoff and multiply with pre factor
    sg_1 = linalg_cpu.scalarmul_ifft_cutoff(sg_t_1, pre_factor, ne, no)
    sl_1 = linalg_cpu.scalarmul_ifft_cutoff(sl_t_1, pre_factor, ne, no)
    sr_1 = linalg_cpu.scalarmul_ifft_cutoff(sr_t_1, pre_factor, ne, no)
    sr_2_part_a_1 = linalg_cpu.scalarmul_ifft_cutoff(
        sr_2part_a_t_1, pre_factor, ne, no)
    sr_2_part_b_1 = linalg_cpu.scalarmul_ifft_cutoff(
        sr_2part_b_t_1, pre_factor, ne, no)

    sg_2 = linalg_cpu.flip(
        linalg_cpu.scalarmul_ifft_cutoff(sg_t_2, pre_factor, ne, no))
    sl_2 = linalg_cpu.flip(
        linalg_cpu.scalarmul_ifft_cutoff(sl_t_2, pre_factor, ne, no))
    sr_2 = linalg_cpu.flip(
        linalg_cpu.scalarmul_ifft_cutoff(sr_t_2, pre_factor, ne, no))
    sr_2_part_a_2 = linalg_cpu.flip(
        linalg_cpu.scalarmul_ifft_cutoff(sr_2part_a_t_2, pre_factor, ne, no))
    sr_2_part_b_2 = linalg_cpu.flip(
        linalg_cpu.scalarmul_ifft_cutoff(sr_2part_b_t_2, pre_factor, ne, no))

    sg_full = linalg_cpu.scalarmul_ifft_cutoff(
        sg_t_full, pre_factor, ne_full, no)
    sl_full = linalg_cpu.scalarmul_ifft_cutoff(
        sl_t_full, pre_factor, ne_full, no)
    sr_full = linalg_cpu.scalarmul_ifft_cutoff(
        sr_t_full, pre_factor, ne_full, no)

    # Calculating the truncated fock part
    vh1d = np.asarray(vh[rows, cols].reshape(-1))
    gl_density = np.imag(np.sum(gl, axis=1))
    rSigmaRF = -np.multiply(gl_density, vh1d[:, disp[0, rank]:disp[0, rank] +
                            count[0, rank]]).reshape((gl_density.shape[0], 1)) * np.abs(pre_factor/2)
    rSigmaRF = np.tile(rSigmaRF, (1, ne))

    # Using the principal value integral method for yet another sigma_r
    NE = len(energy)
    dE = energy[1] - energy[0]
    Evec = np.linspace(0, (NE-1)*dE, NE, endpoint=True, dtype=float)

    # one_div_by_E = 1.0 / Evec
    # one_div_by_E[NE-1] = 0
    # one_div_by_E_t = np.fft.fft(one_div_by_E)
    one_div_by_E = np.concatenate((-1.0/(Evec[-1:0:-1]), np.array(
        [0.0], dtype=float), 1/(Evec[1:]), np.array([1/(Evec[-1] + dE)], dtype=float)))
    one_div_by_E_t = np.fft.fft(one_div_by_E)

    # Another experiment: comparing the two methods: principal value vs time domain manipulation to get gr from gl and gg
    gr_t_new = np.zeros_like(gr_t, dtype=np.complex128)
    gr_t_new[:, :ne] = (gg_t - gl_t)[:, :ne]
    gr_new = linalg_cpu.scalarmul_ifft_cutoff(gr_t_new, 1.0, ne, no)
    gr_t_new_periodic = np.zeros_like(gr_t_new, dtype=np.complex128)
    gr_t_new_periodic[:, :ne] = (gg_periodic_t - gl_periodic_t)[:, :ne]
    gr_new_periodic = linalg_cpu.scalarmul_ifft_cutoff(
        gr_t_new_periodic, 1.0, ne, no)

    D1 = linalg_cpu.fft_numba(1j*np.imag(gg-gl), ne2, no)
    gr_t_principale = np.multiply(D1, one_div_by_E_t)
    gr_principale = linalg_cpu.scalarmul_ifft(
        gr_t_principale, pre_factor, ne, no)[:, ne:]

    sg = sg_1 + sg_2
    sl = sl_1 + sl_2
    sr = sr_1 + sr_2

    SGmSL_t = linalg_cpu.fft_numba(1j*np.imag(sg-sl), ne2, no)
    rSigmaR_t = np.multiply(SGmSL_t, one_div_by_E_t)
    # rSigmaR_t = linalg_cpu.elementmul(SGmSL_t, one_div_by_E_t)
    rSigmaR = linalg_cpu.scalarmul_ifft_cutoff(
        rSigmaR_t, pre_factor, ne2, no)[:, ne:]

    sr_principale = rSigmaR + 1j*np.imag(sg-sl)/2 + rSigmaRF
    sr_time = np.real(sr_full[:, ne-1:]) + 1j*np.imag(sg-sl)/2 + rSigmaRF
    sr_2part_a = sr_2_part_a_1 + sr_2_part_a_2
    sr_2part_b = sr_2_part_b_1 + sr_2_part_b_2

    if (not rank):
        import matplotlib.pyplot as plt
        import matplotlib
        font = {'weight': 'normal',
                'size': 35}
        matplotlib.rc('font', **font)

        plt.figure(0, figsize=(20, 10))
        plt.plot(np.real(gr_new[0, :]), label='time domain manip real gr')
        plt.plot(np.real(gr_new_periodic[0, :]),
                 label='time domain impr. real gr')
        plt.plot(np.real(gr[0, :]), label='reference real gr',
                 linewidth=4, linestyle='--')
        plt.plot(np.real(gr_principale[0, :]),
                 label='principal value real gr', linestyle=':')
        # plt.xlim((5000, 12500))
        plt.ylim((-1, 1))
        plt.legend()
        plt.savefig('gr_comparison.png')

        plt.figure(1, figsize=(20, 10))
        plt.plot(np.real(sr[0, :]), label='3 terms real sr')
        plt.plot(np.real(sr_time[0, :]), label='time domain manip real sr')
        plt.plot(np.real(sr_2part_a[0, :]), label='2 terms real sr 2part_a')
        plt.plot(np.real(sr_2part_b[0, :]), label='2 terms real sr 2part_b')
        plt.plot(np.real(sr_principale[0, :]), label='principal value real sr')
        plt.legend()
        plt.savefig('sr_comparison.png')

        plt.figure(2, figsize=(20, 10))
        plt.plot(np.imag(gg[0, :]), label='reference imag gg')
        plt.plot(np.imag(gl[0, :]), label='reference imag gl')
        plt.legend()
        plt.savefig('gg_gl_comparison.png')

        plt.figure(3, figsize=(20, 10))
        plt.plot(np.real(sr[0, :]), label='3 terms real sr')
        plt.plot(
            np.real((sr_2part_a[0, :]+sr_2part_b[0, :])/2), label='average 2 terms real sr')
        plt.legend()
        plt.savefig('sr_comparison_2.png')

        plt.figure(4, figsize=(20, 10))
        plt.plot(np.imag(sr[0, :]), label='3 terms imag sr')
        plt.plot(
            np.imag((sr_2part_a[0, :]+sr_2part_b[0, :])/2), label='average 2 terms imag sr')
        plt.legend()
        plt.savefig('sr_comparison_3.png')

        np.savetxt('gg.dat', gg[0, :].view(float).reshape(-1, 2))
        np.savetxt('gl.dat', gl[0, :].view(float).reshape(-1, 2))
        np.savetxt('gr.dat', gr[0, :].view(float).reshape(-1, 2))
        np.savetxt('energy.dat', energy)

    return (sg, sl, sr)
