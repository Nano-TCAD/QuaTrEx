# Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.
""" Functions to calculate the self-energies on the gpu. See README for more information. """

import numpy as np
import numpy.typing as npt
import typing
import cupy as cp
import sys
import os

main_path = os.path.abspath(os.path.dirname(__file__))
parent_path = os.path.abspath(os.path.join(main_path, "..", "..", ".."))
sys.path.append(parent_path)


def gw2s_fft_gpu_fullgrid(pre_factor: np.complex128, gg: cp.ndarray, gl: cp.ndarray, gr: cp.ndarray, wg: cp.ndarray,
                          wl: cp.ndarray, wr: cp.ndarray) -> typing.Tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
    """Calculate the self energy with fft on the gpu(see file description todo). 
        The inputs are the pre factor, the Green's Functions
        and the screened interactions.
        This function only gives the correct result if in the previous steps 
        the energy grid is not cutoff.

    Args:
        pre_factor   (np.complex128): pre_factor, multiplied at the end
        gg              (cp.ndarray): Greater Green's Function,     (#orbital, #energy)
        gl              (cp.ndarray): Lesser Green's Function,      (#orbital, #energy)
        gr              (cp.ndarray): Retarded Green's Function,    (#orbital, #energy)
        wg              (cp.ndarray): Greater screened interaction, (#orbital, #energy)
        wl              (cp.ndarray): Lesser screened interaction,  (#orbital, #energy)
        wr              (cp.ndarray): Retarded screened interaction,(#orbital, #energy)

    Returns:
        typing.Tuple[cp.ndarray, Greater self energy  (#orbital, #energy)
                     cp.ndarray, Lesser self energy   (#orbital, #energy)
                     cp.ndarray  Retarded self energy (#orbital, #energy)
                    ]
    """
    # number of energy points
    ne: int = gg.shape[1]

    # todo possiblity to avoid fft in global chain
    # fft
    gg_t = cp.fft.fft(gg, n=2 * ne, axis=1)
    gl_t = cp.fft.fft(gl, n=2 * ne, axis=1)
    gr_t = cp.fft.fft(gr, n=2 * ne, axis=1)
    wg_t = cp.fft.fft(wg, n=2 * ne, axis=1)
    wl_t = cp.fft.fft(wl, n=2 * ne, axis=1)
    wr_t = cp.fft.fft(wr, n=2 * ne, axis=1)

    # multiply elementwise
    sg_t = cp.multiply(gg_t, wg_t)
    sl_t = cp.multiply(gl_t, wl_t)
    sr_t = cp.multiply(gr_t, wl_t) + cp.multiply(gg_t, wr_t)

    # ifft, cutoff and multiply with pre factor
    sg = cp.multiply(cp.fft.ifft(sg_t, axis=1)[:, :ne], pre_factor)
    sl = cp.multiply(cp.fft.ifft(sl_t, axis=1)[:, :ne], pre_factor)
    sr = cp.multiply(cp.fft.ifft(sr_t, axis=1)[:, :ne], pre_factor)

    return (sg, sl, sr)


def gw2s_fft_gpu(pre_factor: np.complex128, ij2ji: cp.ndarray, gg: cp.ndarray, gl: cp.ndarray, gr: cp.ndarray,
                 wg: cp.ndarray, wl: cp.ndarray, wr: cp.ndarray) -> typing.Tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
    """Calculate the self energy with fft on the gpu(see file description todo). 
        The inputs are the pre factor, the Green's Functions
        and the screened interactions.
        Takes into account the energy grid cutoff

    Args:
        pre_factor   (np.complex128): pre_factor, multiplied at the end
        ij2ji           (cp.ndarray): mapping to transposed matrix, (#orbital)
        gg              (cp.ndarray): Greater Green's Function,     (#orbital, #energy)
        gl              (cp.ndarray): Lesser Green's Function,      (#orbital, #energy)
        gr              (cp.ndarray): Retarded Green's Function,    (#orbital, #energy)
        wg              (cp.ndarray): Greater screened interaction, (#orbital, #energy)
        wl              (cp.ndarray): Lesser screened interaction,  (#orbital, #energy)
        wr              (cp.ndarray): Retarded screened interaction,(#orbital, #energy)

    Returns:
        typing.Tuple[cp.ndarray, Greater self energy  (#orbital, #energy)
                     cp.ndarray, Lesser self energy   (#orbital, #energy)
                     cp.ndarray  Retarded self energy (#orbital, #energy)
                    ]
    """
    # number of energy points
    ne = gg.shape[1]

    # todo possibility to avoid fft in global chain
    # fft
    gg_t = cp.fft.fft(gg, n=2 * ne, axis=1)
    gl_t = cp.fft.fft(gl, n=2 * ne, axis=1)
    gr_t = cp.fft.fft(gr, n=2 * ne, axis=1)
    wg_t = cp.fft.fft(wg, n=2 * ne, axis=1)
    wl_t = cp.fft.fft(wl, n=2 * ne, axis=1)
    wr_t = cp.fft.fft(wr, n=2 * ne, axis=1)

    # fft of energy reversed
    rgg_t = cp.fft.fft(cp.flip(gg, axis=1), n=2 * ne, axis=1)
    rgl_t = cp.fft.fft(cp.flip(gl, axis=1), n=2 * ne, axis=1)
    rgr_t = cp.fft.fft(cp.flip(gr, axis=1), n=2 * ne, axis=1)

    # multiply elementwise for sigma_1 the normal term
    sg_t_1 = cp.multiply(gg_t, wg_t)
    sl_t_1 = cp.multiply(gl_t, wl_t)
    sr_t_1 = cp.multiply(gr_t, wl_t) + cp.multiply(gg_t, wr_t)

    # time reverse
    wr_t_mod = cp.roll(cp.flip(wr_t, axis=1), 1, axis=1)

    # multiply elementwise the energy reversed with difference of transposed and energy zero
    # see the README for derivation
    sg_t_2 = cp.multiply(rgg_t, wl_t[ij2ji, :] - cp.repeat(wl[ij2ji, 0].reshape(-1, 1), 2 * ne, axis=1))
    sl_t_2 = cp.multiply(rgl_t, wg_t[ij2ji, :] - cp.repeat(wg[ij2ji, 0].reshape(-1, 1), 2 * ne, axis=1))
    sr_t_2 = (cp.multiply(rgg_t, cp.conjugate(wr_t_mod - cp.repeat(wr[:, 0].reshape(-1, 1), 2 * ne, axis=1))) +
              cp.multiply(rgr_t, wg_t[ij2ji, :] - cp.repeat(wg[:, 0].reshape(-1, 1), 2 * ne, axis=1)))

    # ifft, cutoff and multiply with pre factor
    sg_1 = cp.fft.ifft(sg_t_1, axis=1)[:, :ne]
    sl_1 = cp.fft.ifft(sl_t_1, axis=1)[:, :ne]
    sr_1 = cp.fft.ifft(sr_t_1, axis=1)[:, :ne]

    sg_2 = cp.flip(cp.fft.ifft(sg_t_2, axis=1)[:, :ne], axis=1)
    sl_2 = cp.flip(cp.fft.ifft(sl_t_2, axis=1)[:, :ne], axis=1)
    sr_2 = cp.flip(cp.fft.ifft(sr_t_2, axis=1)[:, :ne], axis=1)

    sg = cp.multiply(sg_1 + sg_2, pre_factor)
    sl = cp.multiply(sl_1 + sl_2, pre_factor)
    sr = cp.multiply(sr_1 + sr_2, pre_factor)

    return (sg, sl, sr)


def gw2s_fft_gpu_3part_sr(pre_factor: np.complex128, ij2ji: cp.ndarray, gg: cp.ndarray, gl: cp.ndarray, gr: cp.ndarray,
                          wg: cp.ndarray, wl: cp.ndarray,
                          wr: cp.ndarray) -> typing.Tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
    """Calculate the self energy with fft on the gpu(see file description todo). 
        The inputs are the pre factor, the Green's Functions
        and the screened interactions.
        Takes into account the energy grid cutoff

    Args:
        pre_factor   (np.complex128): pre_factor, multiplied at the end
        ij2ji           (cp.ndarray): mapping to transposed matrix, (#orbital)
        gg              (cp.ndarray): Greater Green's Function,     (#orbital, #energy)
        gl              (cp.ndarray): Lesser Green's Function,      (#orbital, #energy)
        gr              (cp.ndarray): Retarded Green's Function,    (#orbital, #energy)
        wg              (cp.ndarray): Greater screened interaction, (#orbital, #energy)
        wl              (cp.ndarray): Lesser screened interaction,  (#orbital, #energy)
        wr              (cp.ndarray): Retarded screened interaction,(#orbital, #energy)

    Returns:
        typing.Tuple[cp.ndarray, Greater self energy  (#orbital, #energy)
                     cp.ndarray, Lesser self energy   (#orbital, #energy)
                     cp.ndarray  Retarded self energy (#orbital, #energy)
                    ]
    """
    # number of energy points
    ne = gg.shape[1]

    # todo possibility to avoid fft in global chain
    # fft
    gg_t = cp.fft.fft(gg, n=2 * ne, axis=1)
    gl_t = cp.fft.fft(gl, n=2 * ne, axis=1)
    gr_t = cp.fft.fft(gr, n=2 * ne, axis=1)
    wg_t = cp.fft.fft(wg, n=2 * ne, axis=1)
    wl_t = cp.fft.fft(wl, n=2 * ne, axis=1)
    wr_t = cp.fft.fft(wr, n=2 * ne, axis=1)

    # fft of energy reversed
    rgg_t = cp.fft.fft(cp.flip(gg, axis=1), n=2 * ne, axis=1)
    rgl_t = cp.fft.fft(cp.flip(gl, axis=1), n=2 * ne, axis=1)
    rgr_t = cp.fft.fft(cp.flip(gr, axis=1), n=2 * ne, axis=1)

    # multiply elementwise for sigma_1 the normal term
    sg_t_1 = cp.multiply(gg_t, wg_t)
    sl_t_1 = cp.multiply(gl_t, wl_t)
    sr_t_1 = cp.multiply(gr_t, wl_t) + cp.multiply(gl_t, wr_t) + cp.multiply(gr_t, wr_t)

    # time reverse
    wr_t_mod = cp.roll(cp.flip(wr_t, axis=1), 1, axis=1)

    # multiply elementwise the energy reversed with difference of transposed and energy zero
    # see the README for derivation
    sg_t_2 = cp.multiply(rgg_t, wl_t[ij2ji, :] - cp.repeat(wl[ij2ji, 0].reshape(-1, 1), 2 * ne, axis=1))
    sl_t_2 = cp.multiply(rgl_t, wg_t[ij2ji, :] - cp.repeat(wg[ij2ji, 0].reshape(-1, 1), 2 * ne, axis=1))
    sr_t_2 = (cp.multiply(rgl_t, cp.conjugate(wr_t_mod - cp.repeat(wr[:, 0].reshape(-1, 1), 2 * ne, axis=1))) +
              cp.multiply(rgr_t, wg_t[ij2ji, :] - cp.repeat(wg[ij2ji, 0].reshape(-1, 1), 2 * ne, axis=1)) +
              cp.multiply(rgr_t, cp.conjugate(wr_t_mod - cp.repeat(wr[:, 0].reshape(-1, 1), 2 * ne, axis=1))))

    # ifft, cutoff and multiply with pre factor
    sg_1 = cp.fft.ifft(sg_t_1, axis=1)[:, :ne]
    sl_1 = cp.fft.ifft(sl_t_1, axis=1)[:, :ne]
    sr_1 = cp.fft.ifft(sr_t_1, axis=1)[:, :ne]

    sg_2 = cp.flip(cp.fft.ifft(sg_t_2, axis=1)[:, :ne], axis=1)
    sl_2 = cp.flip(cp.fft.ifft(sl_t_2, axis=1)[:, :ne], axis=1)
    sr_2 = cp.flip(cp.fft.ifft(sr_t_2, axis=1)[:, :ne], axis=1)

    sg = cp.multiply(sg_1 + sg_2, pre_factor)
    sl = cp.multiply(sl_1 + sl_2, pre_factor)
    sr = cp.multiply(sr_1 + sr_2, pre_factor)

    return (sg, sl, sr)


def gw2s_fft_mpi_gpu(
    pre_factor: np.complex128, gg: npt.NDArray[np.complex128], gl: npt.NDArray[np.complex128],
    gr: npt.NDArray[np.complex128], wg: npt.NDArray[np.complex128], wl: npt.NDArray[np.complex128],
    wr: npt.NDArray[np.complex128], wg_transposed: npt.NDArray[np.complex128], wl_transposed: npt.NDArray[np.complex128]
) -> typing.Tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128], npt.NDArray[np.complex128]]:
    """Calculate the self energy with fft on the gpu(see file description todo). 
        The inputs are the pre factor, the Green's Functions
        and the screened interactions.
        Takes into account the energy grid cutoff.
        In addition, loads/unloads the data to/from the gpu.

    Args:
        pre_factor                   (np.complex128): pre_factor, multiplied at the end
        gg              (npt.NDArray[np.complex128]): Greater Green's Function,                 (#orbital, #energy)
        gl              (npt.NDArray[np.complex128]): Lesser Green's Function,                  (#orbital, #energy)
        gr              (npt.NDArray[np.complex128]): Retarded Green's Function,                (#orbital, #energy)
        wg              (npt.NDArray[np.complex128]): Greater screened interaction,             (#orbital, #energy)
        wl              (npt.NDArray[np.complex128]): Lesser screened interaction,              (#orbital, #energy)
        wr              (npt.NDArray[np.complex128]): Retarded screened interaction,            (#orbital, #energy)
        wg_transposed   (npt.NDArray[np.complex128]): Greater screened interaction transposed,  (#orbital, #energy)
        wl_transposed   (npt.NDArray[np.complex128]): Lesser screened interaction transposed,   (#orbital, #energy)

    Returns:
        typing.Tuple[npt.NDArray[np.complex128], Greater self energy  (#orbital, #energy)
                     npt.NDArray[np.complex128], Lesser self energy   (#orbital, #energy)
                     npt.NDArray[np.complex128]  Retarded self energy (#orbital, #energy)
                    ]
    """
    # number of energy points
    ne: int = gg.shape[1]

    # load data to gpu----------------------------------------------------------
    gg_gpu = cp.asarray(gg)
    gl_gpu = cp.asarray(gl)
    gr_gpu = cp.asarray(gr)
    wg_gpu = cp.asarray(wg)
    wl_gpu = cp.asarray(wl)
    wr_gpu = cp.asarray(wr)
    wg_transposed_gpu = cp.asarray(wg_transposed)
    wl_transposed_gpu = cp.asarray(wl_transposed)
    # compute sg/sl/sr----------------------------------------------------------

    # todo possibility to avoid fft in global chain
    # fft
    gg_t = cp.fft.fft(gg_gpu, n=2 * ne, axis=1)
    gl_t = cp.fft.fft(gl_gpu, n=2 * ne, axis=1)
    gr_t = cp.fft.fft(gr_gpu, n=2 * ne, axis=1)
    wg_t = cp.fft.fft(wg_gpu, n=2 * ne, axis=1)
    wl_t = cp.fft.fft(wl_gpu, n=2 * ne, axis=1)
    wr_t = cp.fft.fft(wr_gpu, n=2 * ne, axis=1)
    wg_transposed_t = cp.fft.fft(wg_transposed_gpu, n=2 * ne, axis=1)
    wl_transposed_t = cp.fft.fft(wl_transposed_gpu, n=2 * ne, axis=1)

    # fft of energy reversed
    rgg_t = cp.fft.fft(cp.flip(gg_gpu, axis=1), n=2 * ne, axis=1)
    rgl_t = cp.fft.fft(cp.flip(gl_gpu, axis=1), n=2 * ne, axis=1)
    rgr_t = cp.fft.fft(cp.flip(gr_gpu, axis=1), n=2 * ne, axis=1)

    # multiply elementwise for sigma_1 the normal term
    sg_t_1 = cp.multiply(gg_t, wg_t)
    sl_t_1 = cp.multiply(gl_t, wl_t)
    sr_t_1 = cp.multiply(gr_t, wl_t) + cp.multiply(gg_t, wr_t)

    # time reverse
    wr_t_mod = cp.roll(cp.flip(wr_t, axis=1), 1, axis=1)

    # multiply elementwise the energy reversed with difference of transposed and energy zero
    # see the README for derivation
    sg_t_2 = cp.multiply(rgg_t, wl_transposed_t - cp.repeat(wl_transposed_gpu[:, 0].reshape(-1, 1), 2 * ne, axis=1))
    sl_t_2 = cp.multiply(rgl_t, wg_transposed_t - cp.repeat(wg_transposed_gpu[:, 0].reshape(-1, 1), 2 * ne, axis=1))
    sr_t_2 = (cp.multiply(rgg_t, cp.conjugate(wr_t_mod - cp.repeat(wr_gpu[:, 0].reshape(-1, 1), 2 * ne, axis=1))) +
              cp.multiply(rgr_t, wg_transposed_t - cp.repeat(wg_gpu[:, 0].reshape(-1, 1), 2 * ne, axis=1)))

    # ifft, cutoff and multiply with pre factor
    sg_1 = cp.fft.ifft(sg_t_1, axis=1)[:, :ne]
    sl_1 = cp.fft.ifft(sl_t_1, axis=1)[:, :ne]
    sr_1 = cp.fft.ifft(sr_t_1, axis=1)[:, :ne]

    sg_2 = cp.flip(cp.fft.ifft(sg_t_2, axis=1)[:, :ne], axis=1)
    sl_2 = cp.flip(cp.fft.ifft(sl_t_2, axis=1)[:, :ne], axis=1)
    sr_2 = cp.flip(cp.fft.ifft(sr_t_2, axis=1)[:, :ne], axis=1)

    sg_gpu = cp.multiply(sg_1 + sg_2, pre_factor)
    sl_gpu = cp.multiply(sl_1 + sl_2, pre_factor)
    sr_gpu = cp.multiply(sr_1 + sr_2, pre_factor)

    # load data to cpu----------------------------------------------------------

    sg = cp.asnumpy(sg_gpu)
    sl = cp.asnumpy(sl_gpu)
    sr = cp.asnumpy(sr_gpu)

    return (sg, sl, sr)


def gw2s_fft_mpi_gpu_3part_sr(
    pre_factor: np.complex128, gg: npt.NDArray[np.complex128], gl: npt.NDArray[np.complex128],
    gr: npt.NDArray[np.complex128], wg: npt.NDArray[np.complex128], wl: npt.NDArray[np.complex128],
    wr: npt.NDArray[np.complex128], wg_transposed: npt.NDArray[np.complex128], wl_transposed: npt.NDArray[np.complex128]
) -> typing.Tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128], npt.NDArray[np.complex128]]:
    """Calculate the self energy with fft on the gpu(see file description todo). 
        The inputs are the pre factor, the Green's Functions
        and the screened interactions.
        Takes into account the energy grid cutoff.
        In addition, loads/unloads the data to/from the gpu.

    Args:
        pre_factor                   (np.complex128): pre_factor, multiplied at the end
        gg              (npt.NDArray[np.complex128]): Greater Green's Function,                 (#orbital, #energy)
        gl              (npt.NDArray[np.complex128]): Lesser Green's Function,                  (#orbital, #energy)
        gr              (npt.NDArray[np.complex128]): Retarded Green's Function,                (#orbital, #energy)
        wg              (npt.NDArray[np.complex128]): Greater screened interaction,             (#orbital, #energy)
        wl              (npt.NDArray[np.complex128]): Lesser screened interaction,              (#orbital, #energy)
        wr              (npt.NDArray[np.complex128]): Retarded screened interaction,            (#orbital, #energy)
        wg_transposed   (npt.NDArray[np.complex128]): Greater screened interaction transposed,  (#orbital, #energy)
        wl_transposed   (npt.NDArray[np.complex128]): Lesser screened interaction transposed,   (#orbital, #energy)

    Returns:
        typing.Tuple[npt.NDArray[np.complex128], Greater self energy  (#orbital, #energy)
                     npt.NDArray[np.complex128], Lesser self energy   (#orbital, #energy)
                     npt.NDArray[np.complex128]  Retarded self energy (#orbital, #energy)
                    ]
    """
    # number of energy points
    ne: int = gg.shape[1]

    # load data to gpu----------------------------------------------------------
    gg_gpu = cp.asarray(gg)
    gl_gpu = cp.asarray(gl)
    gr_gpu = cp.asarray(gr)
    wg_gpu = cp.asarray(wg)
    wl_gpu = cp.asarray(wl)
    wr_gpu = cp.asarray(wr)
    wg_transposed_gpu = cp.asarray(wg_transposed)
    wl_transposed_gpu = cp.asarray(wl_transposed)
    # compute sg/sl/sr----------------------------------------------------------

    # todo possibility to avoid fft in global chain
    # fft
    gg_t = cp.fft.fft(gg_gpu, n=2 * ne, axis=1)
    gl_t = cp.fft.fft(gl_gpu, n=2 * ne, axis=1)
    gr_t = cp.fft.fft(gr_gpu, n=2 * ne, axis=1)
    wg_t = cp.fft.fft(wg_gpu, n=2 * ne, axis=1)
    wl_t = cp.fft.fft(wl_gpu, n=2 * ne, axis=1)
    wr_t = cp.fft.fft(wr_gpu, n=2 * ne, axis=1)
    wg_transposed_t = cp.fft.fft(wg_transposed_gpu, n=2 * ne, axis=1)
    wl_transposed_t = cp.fft.fft(wl_transposed_gpu, n=2 * ne, axis=1)

    # fft of energy reversed
    rgg_t = cp.fft.fft(cp.flip(gg_gpu, axis=1), n=2 * ne, axis=1)
    rgl_t = cp.fft.fft(cp.flip(gl_gpu, axis=1), n=2 * ne, axis=1)
    rgr_t = cp.fft.fft(cp.flip(gr_gpu, axis=1), n=2 * ne, axis=1)

    # multiply elementwise for sigma_1 the normal term
    sg_t_1 = cp.multiply(gg_t, wg_t)
    sl_t_1 = cp.multiply(gl_t, wl_t)
    sr_t_1 = cp.multiply(gr_t, wl_t) + cp.multiply(gl_t, wr_t) + cp.multiply(gr_t, wr_t)

    # time reverse
    wr_t_mod = cp.roll(cp.flip(wr_t, axis=1), 1, axis=1)

    # multiply elementwise the energy reversed with difference of transposed and energy zero
    # see the README for derivation
    sg_t_2 = cp.multiply(rgg_t, wl_transposed_t - cp.repeat(wl_transposed_gpu[:, 0].reshape(-1, 1), 2 * ne, axis=1))
    sl_t_2 = cp.multiply(rgl_t, wg_transposed_t - cp.repeat(wg_transposed_gpu[:, 0].reshape(-1, 1), 2 * ne, axis=1))
    #sr_t_2 = (cp.multiply(rgg_t, cp.conjugate(wr_t_mod - cp.repeat(wr_gpu[:,0].reshape(-1,1), 2*ne, axis=1))) +
    #          cp.multiply(rgr_t, wg_transposed_t - cp.repeat(wg_gpu[:,0].reshape(-1,1), 2*ne, axis=1)))

    sr_t_2 = (cp.multiply(rgl_t, cp.conjugate(wr_t_mod - cp.repeat(wr_gpu[:, 0].reshape(-1, 1), 2 * ne, axis=1))) +
              cp.multiply(rgr_t, wg_transposed_t - cp.repeat(wg_transposed_gpu[:, 0].reshape(-1, 1), 2 * ne, axis=1)) +
              cp.multiply(rgr_t, cp.conjugate(wr_t_mod - cp.repeat(wr_gpu[:, 0].reshape(-1, 1), 2 * ne, axis=1))))

    # ifft, cutoff and multiply with pre factor
    sg_1 = cp.fft.ifft(sg_t_1, axis=1)[:, :ne]
    sl_1 = cp.fft.ifft(sl_t_1, axis=1)[:, :ne]
    sr_1 = cp.fft.ifft(sr_t_1, axis=1)[:, :ne]

    sg_2 = cp.flip(cp.fft.ifft(sg_t_2, axis=1)[:, :ne], axis=1)
    sl_2 = cp.flip(cp.fft.ifft(sl_t_2, axis=1)[:, :ne], axis=1)
    sr_2 = cp.flip(cp.fft.ifft(sr_t_2, axis=1)[:, :ne], axis=1)

    sg_gpu = cp.multiply(sg_1 + sg_2, pre_factor)
    sl_gpu = cp.multiply(sl_1 + sl_2, pre_factor)
    sr_gpu = cp.multiply(sr_1 + sr_2, pre_factor)

    # load data to cpu----------------------------------------------------------

    sg = cp.asnumpy(sg_gpu)
    sl = cp.asnumpy(sl_gpu)
    sr = cp.asnumpy(sr_gpu)

    return (sg, sl, sr)


def gw2s_fft_mpi_gpu_PI_sr(
    pre_factor: np.complex128, gg: npt.NDArray[np.complex128], gl: npt.NDArray[np.complex128],
    wg: npt.NDArray[np.complex128], wl: npt.NDArray[np.complex128],
    wg_transposed: npt.NDArray[np.complex128], wl_transposed: npt.NDArray[np.complex128],
    vh1D: npt.NDArray[np.complex128], energy: npt.NDArray[np.float64], rank: np.int32, disp: npt.NDArray[np.int32], count: npt.NDArray[np.int32]
) -> typing.Tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128], npt.NDArray[np.complex128]]:
    """Calculate the self energy with fft on the gpu(see file description todo). 
        The inputs are the pre factor, the Green's Functions
        and the screened interactions.
        Takes into account the energy grid cutoff.
        In addition, loads/unloads the data to/from the gpu.

    Args:
        pre_factor                   (np.complex128): pre_factor, multiplied at the end
        gg              (npt.NDArray[np.complex128]): Greater Green's Function,                 (#orbital, #energy)
        gl              (npt.NDArray[np.complex128]): Lesser Green's Function,                  (#orbital, #energy)
        gr              (npt.NDArray[np.complex128]): Retarded Green's Function,                (#orbital, #energy)
        wg              (npt.NDArray[np.complex128]): Greater screened interaction,             (#orbital, #energy)
        wl              (npt.NDArray[np.complex128]): Lesser screened interaction,              (#orbital, #energy)
        wr              (npt.NDArray[np.complex128]): Retarded screened interaction,            (#orbital, #energy)
        wg_transposed   (npt.NDArray[np.complex128]): Greater screened interaction transposed,  (#orbital, #energy)
        wl_transposed   (npt.NDArray[np.complex128]): Lesser screened interaction transposed,   (#orbital, #energy)

    Returns:
        typing.Tuple[npt.NDArray[np.complex128], Greater self energy  (#orbital, #energy)
                     npt.NDArray[np.complex128], Lesser self energy   (#orbital, #energy)
                     npt.NDArray[np.complex128]  Retarded self energy (#orbital, #energy)
                    ]
    """
    # number of energy points
    ne: int = gg.shape[1]
    no: int = gg.shape[0]

    # load data to gpu----------------------------------------------------------
    gg_gpu = cp.asarray(gg)
    gl_gpu = cp.asarray(gl)
    wg_gpu = cp.asarray(wg)
    wl_gpu = cp.asarray(wl)
    wg_transposed_gpu = cp.asarray(wg_transposed)
    wl_transposed_gpu = cp.asarray(wl_transposed)
    # compute sg/sl/sr----------------------------------------------------------

    # todo possibility to avoid fft in global chain
    # fft
    gg_t = cp.fft.fft(gg_gpu, n=2 * ne, axis=1)
    gl_t = cp.fft.fft(gl_gpu, n=2 * ne, axis=1)
    wg_t = cp.fft.fft(wg_gpu, n=2 * ne, axis=1)
    wl_t = cp.fft.fft(wl_gpu, n=2 * ne, axis=1)
    wg_transposed_t = cp.fft.fft(wg_transposed_gpu, n=2 * ne, axis=1)
    wl_transposed_t = cp.fft.fft(wl_transposed_gpu, n=2 * ne, axis=1)

    # fft of energy reversed
    rgg_t = cp.fft.fft(cp.flip(gg_gpu, axis=1), n=2 * ne, axis=1)
    rgl_t = cp.fft.fft(cp.flip(gl_gpu, axis=1), n=2 * ne, axis=1)

    # multiply elementwise for sigma_1 the normal term
    sg_t_1 = cp.multiply(gg_t, wg_t)
    sl_t_1 = cp.multiply(gl_t, wl_t)


    # multiply elementwise the energy reversed with difference of transposed and energy zero
    # see the README for derivation
    sg_t_2 = cp.multiply(rgg_t, wl_transposed_t - cp.repeat(wl_transposed_gpu[:, 0].reshape(-1, 1), 2 * ne, axis=1))
    sl_t_2 = cp.multiply(rgl_t, wg_transposed_t - cp.repeat(wg_transposed_gpu[:, 0].reshape(-1, 1), 2 * ne, axis=1))
    #sr_t_2 = (cp.multiply(rgg_t, cp.conjugate(wr_t_mod - cp.repeat(wr_gpu[:,0].reshape(-1,1), 2*ne, axis=1))) +
    #          cp.multiply(rgr_t, wg_transposed_t - cp.repeat(wg_gpu[:,0].reshape(-1,1), 2*ne, axis=1)))


    # ifft, cutoff and multiply with pre factor
    sg_1 = cp.fft.ifft(sg_t_1, axis=1)[:, :ne]
    sl_1 = cp.fft.ifft(sl_t_1, axis=1)[:, :ne]

    sg_2 = cp.flip(cp.fft.ifft(sg_t_2, axis=1)[:, :ne], axis=1)
    sl_2 = cp.flip(cp.fft.ifft(sl_t_2, axis=1)[:, :ne], axis=1)

    sg_gpu = cp.multiply(sg_1 + sg_2, pre_factor)
    sl_gpu = cp.multiply(sl_1 + sl_2, pre_factor)

    #Calculating the truncated fock part
    #vh1d = np.asarray(vh[rows, cols].reshape(-1))
    gl_density = np.imag(np.sum(gl, axis = 1))
    rSigmaRF = -np.multiply(gl_density, vh1D[disp[0, rank]:disp[0, rank] + count[0, rank]]).reshape((gl_density.shape[0],1)) * np.abs(pre_factor)
    #rSigmaRF = np.tile(rSigmaRF, (1,ne))
    rSigmaRF = rSigmaRF.repeat(ne).reshape((-1, ne)).astype(np.complex128)

    # Using the principal value integral method for yet another sigma_r
    NE = len(energy)
    dE = energy[1] - energy[0]
    Evec = np.linspace(0, (NE-1)*dE, NE)

    # one_div_by_E = 1.0 / Evec
    # one_div_by_E[NE-1] = 0
    # one_div_by_E_t = np.fft.fft(one_div_by_E)
    one_div_by_E = np.concatenate((-1.0/(Evec[-1:0:-1]), np.array([0.0], dtype = np.float64), 1/(Evec[1:]), np.array([1/(Evec[-1] + dE)], dtype = np.float64)))
    one_div_by_E_t = np.fft.fft(one_div_by_E)
    one_div_by_E_t_gpu = cp.asarray(one_div_by_E_t)

    SGmSL_t_gpu = cp.fft.fft(1j*cp.imag(sg_gpu-sl_gpu),  n=2 * ne, axis=1)
    rSigmaR_t_gpu = cp.multiply(SGmSL_t_gpu, one_div_by_E_t_gpu)
    #rSigmaR_t = linalg_cpu.elementmul(SGmSL_t, one_div_by_E_t)
    rSigmaR_gpu = cp.fft.ifft(rSigmaR_t_gpu, axis = 1)[:, ne-1:-1].astype(np.complex128)
    cp.multiply(rSigmaR_gpu, 2*pre_factor, out=rSigmaR_gpu)
    sr_principale = (rSigmaR_gpu/2 + (1j*cp.imag(sg_gpu-sl_gpu)/2).astype(np.complex128)).get() + rSigmaRF

    # load data to cpu----------------------------------------------------------

    sg = cp.asnumpy(sg_gpu)
    sl = cp.asnumpy(sl_gpu)

    return (sg, sl, sr_principale)


def gw2s_fft_mpi_gpu_PI_sr_batched(
    pre_factor: np.complex128, gg: npt.NDArray[np.complex128], gl: npt.NDArray[np.complex128],
    wg: npt.NDArray[np.complex128], wl: npt.NDArray[np.complex128],
    wg_transposed: npt.NDArray[np.complex128], wl_transposed: npt.NDArray[np.complex128],
    vh1D: npt.NDArray[np.complex128], energy: npt.NDArray[np.float64], rank: np.int32, disp: npt.NDArray[np.int32], count: npt.NDArray[np.int32],
    batch_size: int = 1000
) -> typing.Tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128], npt.NDArray[np.complex128]]:
    """Calculate the self energy with fft on the gpu(see file description todo). 
        The inputs are the pre factor, the Green's Functions
        and the screened interactions.
        Takes into account the energy grid cutoff.
        In addition, loads/unloads the data to/from the gpu.

    Args:
        pre_factor                   (np.complex128): pre_factor, multiplied at the end
        gg              (npt.NDArray[np.complex128]): Greater Green's Function,                 (#orbital, #energy)
        gl              (npt.NDArray[np.complex128]): Lesser Green's Function,                  (#orbital, #energy)
        gr              (npt.NDArray[np.complex128]): Retarded Green's Function,                (#orbital, #energy)
        wg              (npt.NDArray[np.complex128]): Greater screened interaction,             (#orbital, #energy)
        wl              (npt.NDArray[np.complex128]): Lesser screened interaction,              (#orbital, #energy)
        wr              (npt.NDArray[np.complex128]): Retarded screened interaction,            (#orbital, #energy)
        wg_transposed   (npt.NDArray[np.complex128]): Greater screened interaction transposed,  (#orbital, #energy)
        wl_transposed   (npt.NDArray[np.complex128]): Lesser screened interaction transposed,   (#orbital, #energy)

    Returns:
        typing.Tuple[npt.NDArray[np.complex128], Greater self energy  (#orbital, #energy)
                     npt.NDArray[np.complex128], Lesser self energy   (#orbital, #energy)
                     npt.NDArray[np.complex128]  Retarded self energy (#orbital, #energy)
                    ]
    """
    # number of energy points
    ne: int = gg.shape[1]
    no: int = gg.shape[0]


    # determine number of batches
    # batch over no
    batches = no // batch_size
    if batches == 0:
        # print("Too large batch size")
        batch_size = no
        batches = 1

    sg = cp.empty((no, ne), dtype=np.complex128)
    sl = cp.empty((no, ne), dtype=np.complex128)
    sr_principale = cp.empty((no, ne), dtype=np.complex128)

    
    # load data to gpu and compute----------------------------------------------------
    # gg_gpu = cp.empty((batch_size + no % batch_size, ne), dtype=np.complex128)
    # gl_gpu = cp.empty((batch_size + no % batch_size, ne), dtype=np.complex128)
    # wg_gpu = cp.empty((batch_size + no % batch_size, ne), dtype=np.complex128)
    # wl_gpu = cp.empty((batch_size + no % batch_size, ne), dtype=np.complex128)
    # wg_transposed_gpu = cp.empty((batch_size + no % batch_size, ne), dtype=np.complex128)
    # wl_transposed_gpu = cp.empty((batch_size + no % batch_size, ne), dtype=np.complex128)
    # compute sg/sl/sr----------------------------------------------------------

    #Calculating the truncated fock part
    #vh1d = np.asarray(vh[rows, cols].reshape(-1))
    gl_density = cp.imag(cp.sum(gl, axis = 1))
    rSigmaRF = -cp.multiply(gl_density, vh1D[disp[0, rank]:disp[0, rank] + count[0, rank]]).reshape((gl_density.shape[0],1)) * np.abs(pre_factor)
    #rSigmaRF = np.tile(rSigmaRF, (1,ne))
    rSigmaRF = rSigmaRF.repeat(ne).reshape((-1, ne)).astype(np.complex128)

    # Using the principal value integral method for yet another sigma_r
    NE = len(energy)
    dE = energy[1] - energy[0]
    Evec = np.linspace(0, (NE-1)*dE, NE)

    # one_div_by_E = 1.0 / Evec
    # one_div_by_E[NE-1] = 0
    # one_div_by_E_t = np.fft.fft(one_div_by_E)
    one_div_by_E = np.concatenate((-1.0/(Evec[-1:0:-1]), np.array([0.0], dtype = np.float64), 1/(Evec[1:]), np.array([1/(Evec[-1] + dE)], dtype = np.float64)))
    one_div_by_E_t = cp.fft.fft(cp.asarray(one_div_by_E))
    one_div_by_E_t_gpu = one_div_by_E_t

    for batch in range(batches):
        batch_start = batch * batch_size
        # last batch different, if not dividable
        batch_end = batch_size * (batch + 1) if batch != batches - 1 else batch_size * (batch + 1) + no % batch_size

        # todo possibility to avoid fft in global chain
        # fft
        # gg_gpu[0:batch_end - batch_start] = cp.asarray(gg[batch_start:batch_end, :])
        # gg_t = cp.fft.fft(gg_gpu[0:batch_end - batch_start], n=2 * ne, axis=1)
        gg_gpu = gg[batch_start:batch_end, :]
        gg_t = cp.fft.fft(gg_gpu, n=2 * ne, axis=1)

        # gl_gpu[0:batch_end - batch_start] = cp.asarray(gl[batch_start:batch_end, :])
        # gl_t = cp.fft.fft(gl_gpu[0:batch_end - batch_start], n=2 * ne, axis=1)
        gl_gpu = gl[batch_start:batch_end, :]
        gl_t = cp.fft.fft(gl_gpu, n=2 * ne, axis=1)

        # wg_gpu[0:batch_end - batch_start] = cp.asarray(wg[batch_start:batch_end, :])
        # wg_t = cp.fft.fft(wg_gpu[0:batch_end - batch_start], n=2 * ne, axis=1)
        wg_gpu = wg[batch_start:batch_end, :]
        wg_t = cp.fft.fft(wg_gpu, n=2 * ne, axis=1)

        # wl_gpu[0:batch_end - batch_start] = cp.asarray(wl[batch_start:batch_end, :])
        # wl_t = cp.fft.fft(wl_gpu[0:batch_end - batch_start], n=2 * ne, axis=1)
        wl_gpu = wl[batch_start:batch_end, :]
        wl_t = cp.fft.fft(wl_gpu, n=2 * ne, axis=1)

        # wg_transposed_gpu[0:batch_end - batch_start] = cp.asarray(wg_transposed[batch_start:batch_end, :])
        # wg_transposed_t = cp.fft.fft(wg_transposed_gpu[0:batch_end - batch_start], n=2 * ne, axis=1)
        wg_transposed_gpu = wg_transposed[batch_start:batch_end, :]
        wg_transposed_t = cp.fft.fft(wg_transposed_gpu, n=2 * ne, axis=1)

        # wl_transposed_gpu[0:batch_end - batch_start] = cp.asarray(wl_transposed[batch_start:batch_end, :])
        # wl_transposed_t = cp.fft.fft(wl_transposed_gpu[0:batch_end - batch_start], n=2 * ne, axis=1)
        wl_transposed_gpu = wl_transposed[batch_start:batch_end, :]
        wl_transposed_t = cp.fft.fft(wl_transposed_gpu, n=2 * ne, axis=1)
        
        # fft of energy reversed
        rgg_t = cp.fft.fft(cp.flip(gg_gpu, axis=1), n=2 * ne, axis=1)
        rgl_t = cp.fft.fft(cp.flip(gl_gpu, axis=1), n=2 * ne, axis=1)

        # multiply elementwise for sigma_1 the normal term
        sg_t_1 = cp.multiply(gg_t, wg_t)
        sl_t_1 = cp.multiply(gl_t, wl_t)


        # multiply elementwise the energy reversed with difference of transposed and energy zero
        # see the README for derivation
        # sg_t_2 = cp.multiply(rgg_t, wl_transposed_t - cp.repeat(wl_transposed_gpu[0:batch_end - batch_start, 0].reshape(-1, 1), 2 * ne, axis=1))
        # sl_t_2 = cp.multiply(rgl_t, wg_transposed_t - cp.repeat(wg_transposed_gpu[0:batch_end - batch_start, 0].reshape(-1, 1), 2 * ne, axis=1))
        sg_t_2 = cp.multiply(rgg_t, wl_transposed_t - cp.repeat(wl_transposed_gpu[:, 0].reshape(-1, 1), 2 * ne, axis=1))
        sl_t_2 = cp.multiply(rgl_t, wg_transposed_t - cp.repeat(wg_transposed_gpu[:, 0].reshape(-1, 1), 2 * ne, axis=1))
        #sr_t_2 = (cp.multiply(rgg_t, cp.conjugate(wr_t_mod - cp.repeat(wr_gpu[:,0].reshape(-1,1), 2*ne, axis=1))) +
        #          cp.multiply(rgr_t, wg_transposed_t - cp.repeat(wg_gpu[:,0].reshape(-1,1), 2*ne, axis=1)))


        # ifft, cutoff and multiply with pre factor
        sg_1 = cp.fft.ifft(sg_t_1, axis=1)[:, :ne]
        sl_1 = cp.fft.ifft(sl_t_1, axis=1)[:, :ne]

        sg_2 = cp.flip(cp.fft.ifft(sg_t_2, axis=1)[:, :ne], axis=1)
        sl_2 = cp.flip(cp.fft.ifft(sl_t_2, axis=1)[:, :ne], axis=1)

        sg_gpu = cp.multiply(sg_1 + sg_2, pre_factor)
        sl_gpu = cp.multiply(sl_1 + sl_2, pre_factor)

        # SGmSL_t_gpu = cp.fft.fft(1j*cp.imag(sg_gpu-sl_gpu)[0:batch_end - batch_start,:],  n=2 * ne, axis=1)
        SGmSL_t_gpu = cp.fft.fft(1j*cp.imag(sg_gpu-sl_gpu),  n=2 * ne, axis=1)
        rSigmaR_t_gpu = cp.multiply(SGmSL_t_gpu, one_div_by_E_t_gpu)
        #rSigmaR_t = linalg_cpu.elementmul(SGmSL_t, one_div_by_E_t)
        rSigmaR_gpu = cp.fft.ifft(rSigmaR_t_gpu, axis = 1)[:, ne-1:-1].astype(np.complex128)
        cp.multiply(rSigmaR_gpu, 2*pre_factor, out=rSigmaR_gpu)
        # sr_principale[batch_start:batch_end] = (rSigmaR_gpu/2 + (1j*cp.imag(sg_gpu-sl_gpu)/2)[0:batch_end - batch_start,:].astype(np.complex128)).get() + rSigmaRF[batch_start:batch_end, :].get()
        sr_principale[batch_start:batch_end] = (rSigmaR_gpu/2 + (1j*cp.imag(sg_gpu-sl_gpu)/2).astype(np.complex128)) + rSigmaRF[batch_start:batch_end, :]

        # load data to cpu----------------------------------------------------------

        sg[batch_start:batch_end] = sg_gpu[0:batch_end - batch_start]
        sl[batch_start:batch_end] = sl_gpu[0:batch_end - batch_start]
        

    return (sg, sl, sr_principale)



def gw2s_fft_mpi_gpu_streams(
    pre_factor: np.complex128, gg: npt.NDArray[np.complex128], gl: npt.NDArray[np.complex128],
    gr: npt.NDArray[np.complex128], wg: npt.NDArray[np.complex128], wl: npt.NDArray[np.complex128],
    wr: npt.NDArray[np.complex128], wg_transposed: npt.NDArray[np.complex128],
    wl_transposed: npt.NDArray[np.complex128], sg: npt.NDArray[np.complex128], sl: npt.NDArray[np.complex128],
    sr: npt.NDArray[np.complex128], streams: typing.List[cp.cuda.Stream]
) -> typing.Tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128], npt.NDArray[np.complex128]]:
    """Calculate the self energy with fft on the gpu(see file description todo). 
        The inputs are the pre factor, the Green's Functions
        and the screened interactions.
        Takes into account the energy grid cutoff.
        In addition, loads/unloads the data to/from the gpu.
        The input should be pinned memory for highest performance.
        See above kernel and README for more information.

    Args:
        pre_factor                   (np.complex128): pre_factor, multiplied at the end
        gg              (npt.NDArray[np.complex128]): Greater Green's Function,                 (#orbital, #energy)
        gl              (npt.NDArray[np.complex128]): Lesser Green's Function,                  (#orbital, #energy)
        gr              (npt.NDArray[np.complex128]): Retarded Green's Function,                (#orbital, #energy)
        wg              (npt.NDArray[np.complex128]): Greater screened interaction,             (#orbital, #energy)
        wl              (npt.NDArray[np.complex128]): Lesser screened interaction,              (#orbital, #energy)
        wr              (npt.NDArray[np.complex128]): Retarded screened interaction,            (#orbital, #energy)
        wg_transposed   (npt.NDArray[np.complex128]): Greater screened interaction transposed,  (#orbital, #energy)
        wl_transposed   (npt.NDArray[np.complex128]): Lesser screened interaction transposed,   (#orbital, #energy)
        sg              (npt.NDArray[np.complex128]): Output Greater self energy                (#orbital, #energy)
        sl              (npt.NDArray[np.complex128]): Output Lesser self energy                 (#orbital, #energy)
        sr              (npt.NDArray[np.complex128]): Output Retarded self energy               (#orbital, #energy)
        streams        (typing.List[cp.cuda.Stream]): List of streams to overlay memcpy and computations, at least eight streams needed

    """
    # number of energy points
    ne: int = gg.shape[1]

    # load data to gpu and compute----------------------------------------------------
    gg_gpu = cp.empty_like(gg)
    gl_gpu = cp.empty_like(gl)
    gr_gpu = cp.empty_like(gr)
    wg_gpu = cp.empty_like(wg)
    wl_gpu = cp.empty_like(wl)
    wr_gpu = cp.empty_like(wr)
    wg_transposed_gpu = cp.empty_like(wg_transposed)
    wl_transposed_gpu = cp.empty_like(wl_transposed)

    # todo possibility to avoid fft of green's function in global chain

    # load to gpu and fft
    with streams[0]:
        gg_gpu.set(gg, stream=streams[0])
        gg_t = cp.fft.fft(gg_gpu, n=2 * ne, axis=1)
    with streams[1]:
        gl_gpu.set(gl, stream=streams[1])
        gl_t = cp.fft.fft(gl_gpu, n=2 * ne, axis=1)
    with streams[2]:
        gr_gpu.set(gr, stream=streams[2])
        gr_t = cp.fft.fft(gr_gpu, n=2 * ne, axis=1)
    with streams[3]:
        wg_gpu.set(wg, stream=streams[3])
        wg_t = cp.fft.fft(wg_gpu, n=2 * ne, axis=1)
    with streams[4]:
        wl_gpu.set(wl, stream=streams[4])
        wl_t = cp.fft.fft(wl_gpu, n=2 * ne, axis=1)
    with streams[5]:
        wr_gpu.set(wr, stream=streams[5])
        wr_t = cp.fft.fft(wr_gpu, n=2 * ne, axis=1)
    with streams[6]:
        wg_transposed_gpu.set(wg_transposed, stream=streams[6])
        wg_transposed_t = cp.fft.fft(wg_transposed_gpu, n=2 * ne, axis=1)
    with streams[7]:
        wl_transposed_gpu.set(wl_transposed, stream=streams[7])
        wl_transposed_t = cp.fft.fft(wl_transposed_gpu, n=2 * ne, axis=1)

    # element wise multiplication of the screened interaction with the Green's function
    streams[3].synchronize()
    streams[0].synchronize()
    with streams[0]:
        rgg_t = cp.fft.fft(cp.flip(gg_gpu, axis=1), n=2 * ne, axis=1)
        sg_t_1 = cp.multiply(gg_t, wg_t)
    streams[4].synchronize()
    with streams[1]:
        rgl_t = cp.fft.fft(cp.flip(gl_gpu, axis=1), n=2 * ne, axis=1)
        sl_t_1 = cp.multiply(gl_t, wl_t)
    streams[5].synchronize()
    with streams[2]:
        rgr_t = cp.fft.fft(cp.flip(gr_gpu, axis=1), n=2 * ne, axis=1)
        sr_t_1 = cp.multiply(gr_t, wl_t) + cp.multiply(gg_t, wr_t)
        # time reverse
        wr_t_mod = cp.roll(cp.flip(wr_t, axis=1), 1, axis=1)

    # multiply elementwise the energy reversed with difference of transposed and energy zero
    # see the README for derivation
    streams[0].synchronize()
    streams[7].synchronize()
    with streams[0]:
        sg_t_2 = cp.multiply(rgg_t, wl_transposed_t - cp.repeat(wl_transposed_gpu[:, 0].reshape(-1, 1), 2 * ne, axis=1))
    streams[6].synchronize()
    with streams[1]:
        sl_t_2 = cp.multiply(rgl_t, wg_transposed_t - cp.repeat(wg_transposed_gpu[:, 0].reshape(-1, 1), 2 * ne, axis=1))
    with streams[2]:
        sr_t_2 = (cp.multiply(rgg_t, cp.conjugate(wr_t_mod - cp.repeat(wr_gpu[:, 0].reshape(-1, 1), 2 * ne, axis=1))) +
                  cp.multiply(rgr_t, wg_transposed_t - cp.repeat(wg_gpu[:, 0].reshape(-1, 1), 2 * ne, axis=1)))

    # load data to cpu----------------------------------------------------------
    # ifft, cutoff and multiply with pre factor
    with streams[0]:
        sg_1 = cp.fft.ifft(sg_t_1, axis=1)[:, :ne]
        sg_2 = cp.flip(cp.fft.ifft(sg_t_2, axis=1)[:, :ne], axis=1)
        sg_gpu = cp.multiply(sg_1 + sg_2, pre_factor)
        cp.asnumpy(sg_gpu, out=sg, stream=streams[0])
    with streams[1]:
        sl_1 = cp.fft.ifft(sl_t_1, axis=1)[:, :ne]
        sl_2 = cp.flip(cp.fft.ifft(sl_t_2, axis=1)[:, :ne], axis=1)
        sl_gpu = cp.multiply(sl_1 + sl_2, pre_factor)
        cp.asnumpy(sl_gpu, out=sl, stream=streams[1])
    with streams[2]:
        sr_1 = cp.fft.ifft(sr_t_1, axis=1)[:, :ne]
        sr_2 = cp.flip(cp.fft.ifft(sr_t_2, axis=1)[:, :ne], axis=1)
        sr_gpu = cp.multiply(sr_1 + sr_2, pre_factor)
        cp.asnumpy(sr_gpu, out=sr, stream=streams[2])

    for stream in streams:
        stream.synchronize()


def gw2s_fft_mpi_gpu_batched_streams(
    pre_factor: np.complex128, gg: npt.NDArray[np.complex128], gl: npt.NDArray[np.complex128],
    gr: npt.NDArray[np.complex128], wg: npt.NDArray[np.complex128], wl: npt.NDArray[np.complex128],
    wr: npt.NDArray[np.complex128], wg_transposed: npt.NDArray[np.complex128],
    wl_transposed: npt.NDArray[np.complex128], sg: npt.NDArray[np.complex128], sl: npt.NDArray[np.complex128],
    sr: npt.NDArray[np.complex128], streams: typing.List[cp.cuda.Stream], batch_size: int
) -> typing.Tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128], npt.NDArray[np.complex128]]:
    """Calculate the self energy with fft on the gpu(see file description todo). 
        The inputs are the pre factor, the Green's Functions
        and the screened interactions.
        Takes into account the energy grid cutoff.
        In addition, loads/unloads the data to/from the gpu.
        The input should be pinned memory for highest performance.
        See above kernel and README for more information.

    Args:
        pre_factor                   (np.complex128): pre_factor, multiplied at the end
        gg              (npt.NDArray[np.complex128]): Greater Green's Function,                 (#orbital, #energy)
        gl              (npt.NDArray[np.complex128]): Lesser Green's Function,                  (#orbital, #energy)
        gr              (npt.NDArray[np.complex128]): Retarded Green's Function,                (#orbital, #energy)
        wg              (npt.NDArray[np.complex128]): Greater screened interaction,             (#orbital, #energy)
        wl              (npt.NDArray[np.complex128]): Lesser screened interaction,              (#orbital, #energy)
        wr              (npt.NDArray[np.complex128]): Retarded screened interaction,            (#orbital, #energy)
        wg_transposed   (npt.NDArray[np.complex128]): Greater screened interaction transposed,  (#orbital, #energy)
        wl_transposed   (npt.NDArray[np.complex128]): Lesser screened interaction transposed,   (#orbital, #energy)
        sg              (npt.NDArray[np.complex128]): Output Greater self energy                (#orbital, #energy)
        sl              (npt.NDArray[np.complex128]): Output Lesser self energy                 (#orbital, #energy)
        sr              (npt.NDArray[np.complex128]): Output Retarded self energy               (#orbital, #energy)
        streams        (typing.List[cp.cuda.Stream]): List of streams to overlay memcpy and computations, at least eight streams needed
        batch_size                             (int): batch size, should be set to fully utilize the gpu
    
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

    # load data to gpu and compute----------------------------------------------------
    gg_gpu = cp.empty((batch_size + no % batch_size, ne), dtype=np.complex128)
    gl_gpu = cp.empty((batch_size + no % batch_size, ne), dtype=np.complex128)
    gr_gpu = cp.empty((batch_size + no % batch_size, ne), dtype=np.complex128)
    wg_gpu = cp.empty((batch_size + no % batch_size, ne), dtype=np.complex128)
    wl_gpu = cp.empty((batch_size + no % batch_size, ne), dtype=np.complex128)
    wr_gpu = cp.empty((batch_size + no % batch_size, ne), dtype=np.complex128)
    wg_transposed_gpu = cp.empty((batch_size + no % batch_size, ne), dtype=np.complex128)
    wl_transposed_gpu = cp.empty((batch_size + no % batch_size, ne), dtype=np.complex128)

    # todo possibility to avoid fft of green's function in global chain

    for batch in range(batches):
        batch_start = batch * batch_size
        # last batch different, if not dividable
        batch_end = batch_size * (batch + 1) if batch != batches - 1 else batch_size * (batch + 1) + no % batch_size

        # load to gpu and fft
        with streams[0]:
            gg_gpu[0:batch_end - batch_start].set(gg[batch_start:batch_end, :], stream=streams[0])
            gg_t = cp.fft.fft(gg_gpu, n=2 * ne, axis=1)
        with streams[1]:
            gl_gpu[0:batch_end - batch_start].set(gl[batch_start:batch_end, :], stream=streams[1])
            gl_t = cp.fft.fft(gl_gpu, n=2 * ne, axis=1)
        with streams[2]:
            gr_gpu[0:batch_end - batch_start].set(gr[batch_start:batch_end, :], stream=streams[2])
            gr_t = cp.fft.fft(gr_gpu, n=2 * ne, axis=1)
        with streams[3]:
            wg_gpu[0:batch_end - batch_start].set(wg[batch_start:batch_end, :], stream=streams[3])
            wg_t = cp.fft.fft(wg_gpu, n=2 * ne, axis=1)
        with streams[4]:
            wl_gpu[0:batch_end - batch_start].set(wl[batch_start:batch_end, :], stream=streams[4])
            wl_t = cp.fft.fft(wl_gpu, n=2 * ne, axis=1)
        with streams[5]:
            wr_gpu[0:batch_end - batch_start].set(wr[batch_start:batch_end, :], stream=streams[5])
            wr_t = cp.fft.fft(wr_gpu, n=2 * ne, axis=1)
        with streams[6]:
            wg_transposed_gpu[0:batch_end - batch_start].set(wg_transposed[batch_start:batch_end, :], stream=streams[6])
            wg_transposed_t = cp.fft.fft(wg_transposed_gpu, n=2 * ne, axis=1)
        with streams[7]:
            wl_transposed_gpu[0:batch_end - batch_start].set(wl_transposed[batch_start:batch_end, :], stream=streams[7])
            wl_transposed_t = cp.fft.fft(wl_transposed_gpu, n=2 * ne, axis=1)

        # element wise multiplication of the screened interaction with the Green's function
        streams[3].synchronize()
        streams[0].synchronize()
        with streams[0]:
            rgg_t = cp.fft.fft(cp.flip(gg_gpu, axis=1), n=2 * ne, axis=1)
            sg_t_1 = cp.multiply(gg_t, wg_t)
        streams[4].synchronize()
        with streams[1]:
            rgl_t = cp.fft.fft(cp.flip(gl_gpu, axis=1), n=2 * ne, axis=1)
            sl_t_1 = cp.multiply(gl_t, wl_t)
        streams[5].synchronize()
        with streams[2]:
            rgr_t = cp.fft.fft(cp.flip(gr_gpu, axis=1), n=2 * ne, axis=1)
            sr_t_1 = cp.multiply(gr_t, wl_t) + cp.multiply(gg_t, wr_t)
            # time reverse
            wr_t_mod = cp.roll(cp.flip(wr_t, axis=1), 1, axis=1)

        # multiply elementwise the energy reversed with difference of transposed and energy zero
        # see the README for derivation
        streams[0].synchronize()
        streams[7].synchronize()
        with streams[0]:
            sg_t_2 = cp.multiply(rgg_t,
                                 wl_transposed_t - cp.repeat(wl_transposed_gpu[:, 0].reshape(-1, 1), 2 * ne, axis=1))
        streams[6].synchronize()
        with streams[1]:
            sl_t_2 = cp.multiply(rgl_t,
                                 wg_transposed_t - cp.repeat(wg_transposed_gpu[:, 0].reshape(-1, 1), 2 * ne, axis=1))
        with streams[2]:
            sr_t_2 = (
                cp.multiply(rgg_t, cp.conjugate(wr_t_mod - cp.repeat(wr_gpu[:, 0].reshape(-1, 1), 2 * ne, axis=1))) +
                cp.multiply(rgr_t, wg_transposed_t - cp.repeat(wg_gpu[:, 0].reshape(-1, 1), 2 * ne, axis=1)))

        # load data to cpu----------------------------------------------------------
        # ifft, cutoff and multiply with pre factor
        with streams[0]:
            sg_1 = cp.fft.ifft(sg_t_1, axis=1)[:, :ne]
            sg_2 = cp.flip(cp.fft.ifft(sg_t_2, axis=1)[:, :ne], axis=1)
            sg_gpu = cp.multiply(sg_1 + sg_2, pre_factor)
            cp.asnumpy(sg_gpu[0:batch_end - batch_start], out=sg[batch_start:batch_end, :], stream=streams[0])
        with streams[1]:
            sl_1 = cp.fft.ifft(sl_t_1, axis=1)[:, :ne]
            sl_2 = cp.flip(cp.fft.ifft(sl_t_2, axis=1)[:, :ne], axis=1)
            sl_gpu = cp.multiply(sl_1 + sl_2, pre_factor)
            cp.asnumpy(sl_gpu[0:batch_end - batch_start], out=sl[batch_start:batch_end, :], stream=streams[1])
        with streams[2]:
            sr_1 = cp.fft.ifft(sr_t_1, axis=1)[:, :ne]
            sr_2 = cp.flip(cp.fft.ifft(sr_t_2, axis=1)[:, :ne], axis=1)
            sr_gpu = cp.multiply(sr_1 + sr_2, pre_factor)
            cp.asnumpy(sr_gpu[0:batch_end - batch_start], out=sr[batch_start:batch_end, :], stream=streams[2])

    for stream in streams:
        stream.synchronize()
