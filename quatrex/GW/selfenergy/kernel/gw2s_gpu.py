"""
Functions to calculate the self-energies on the gpu.
See README for more information.
"""
import numpy as np
import numpy.typing as npt
import typing
import cupy as cp
import sys
import os

main_path = os.path.abspath(os.path.dirname(__file__))
parent_path = os.path.abspath(os.path.join(main_path, "..", "..", ".."))
sys.path.append(parent_path)


def gw2s_fft_gpu_fullgrid(
    pre_factor: np.complex128,
    gg: cp.ndarray,
    gl: cp.ndarray,
    gr: cp.ndarray,
    wg: cp.ndarray,
    wl: cp.ndarray,
    wr: cp.ndarray
) -> typing.Tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
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
    sr_t = cp.multiply(gr_t, wl_t) +  cp.multiply(gg_t, wr_t)

    # ifft, cutoff and multiply with pre factor
    sg = cp.multiply(cp.fft.ifft(sg_t, axis=1)[:, :ne], pre_factor)
    sl = cp.multiply(cp.fft.ifft(sl_t, axis=1)[:, :ne], pre_factor)
    sr = cp.multiply(cp.fft.ifft(sr_t, axis=1)[:, :ne], pre_factor)

    return (sg, sl, sr)

def gw2s_fft_gpu(
    pre_factor: np.complex128,
    ij2ji: cp.ndarray,
    gg: cp.ndarray,
    gl: cp.ndarray,
    gr: cp.ndarray,
    wg: cp.ndarray,
    wl: cp.ndarray,
    wr: cp.ndarray
) -> typing.Tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
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
    sr_t_1 = cp.multiply(gr_t, wl_t) +  cp.multiply(gg_t, wr_t)

    # time reverse 
    wr_t_mod = cp.roll(cp.flip(wr_t, axis=1), 1, axis=1)    

    # multiply elementwise the energy reversed with difference of transposed and energy zero
    # see the document "derivation_selfenergy.pdf" for an explanation
    sg_t_2 = cp.multiply(rgg_t, wl_t[ij2ji,:] - cp.repeat(wl[ij2ji,0].reshape(-1,1), 2*ne, axis=1))
    sl_t_2 = cp.multiply(rgl_t, wg_t[ij2ji,:] - cp.repeat(wg[ij2ji,0].reshape(-1,1), 2*ne, axis=1))
    sr_t_2 = (cp.multiply(rgg_t, cp.conjugate(wr_t_mod - cp.repeat(wr[:,0].reshape(-1,1), 2*ne, axis=1))) +
              cp.multiply(rgr_t, wg_t[ij2ji,:] - cp.repeat(wg[:,0].reshape(-1,1), 2*ne, axis=1)))


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
    pre_factor: np.complex128,
    gg: npt.NDArray[np.complex128],
    gl: npt.NDArray[np.complex128],
    gr: npt.NDArray[np.complex128],
    wg: npt.NDArray[np.complex128],
    wl: npt.NDArray[np.complex128],
    wr: npt.NDArray[np.complex128],
    wg_transposed: npt.NDArray[np.complex128],
    wl_transposed: npt.NDArray[np.complex128]
) -> typing.Tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128],
                  npt.NDArray[np.complex128]]:
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
    gg_gpu       = cp.asarray(gg)
    gl_gpu       = cp.asarray(gl)
    gr_gpu       = cp.asarray(gr)
    wg_gpu       = cp.asarray(wg)
    wl_gpu       = cp.asarray(wl)
    wr_gpu       = cp.asarray(wr)
    wg_transposed_gpu = cp.asarray(wg_transposed)
    wl_transposed_gpu = cp.asarray(wl_transposed)
    # compute sg/sl/sr----------------------------------------------------------

    # number of energy points
    ne = gg.shape[1]

    # todo possibility to avoid fft in global chain
    # fft
    gg_t        = cp.fft.fft(gg_gpu, n=2 * ne, axis=1)
    gl_t        = cp.fft.fft(gl_gpu, n=2 * ne, axis=1)
    gr_t        = cp.fft.fft(gr_gpu, n=2 * ne, axis=1)
    wg_t        = cp.fft.fft(wg_gpu, n=2 * ne, axis=1)
    wl_t        = cp.fft.fft(wl_gpu, n=2 * ne, axis=1)
    wr_t        = cp.fft.fft(wr_gpu, n=2 * ne, axis=1)
    wg_transposed_t  = cp.fft.fft(wg_transposed_gpu, n=2 * ne, axis=1)
    wl_transposed_t  = cp.fft.fft(wl_transposed_gpu, n=2 * ne, axis=1)

    # fft of energy reversed
    rgg_t = cp.fft.fft(cp.flip(gg_gpu, axis=1), n=2 * ne, axis=1)
    rgl_t = cp.fft.fft(cp.flip(gl_gpu, axis=1), n=2 * ne, axis=1)
    rgr_t = cp.fft.fft(cp.flip(gr_gpu, axis=1), n=2 * ne, axis=1)

    # multiply elementwise for sigma_1 the normal term
    sg_t_1 = cp.multiply(gg_t, wg_t)
    sl_t_1 = cp.multiply(gl_t, wl_t)
    sr_t_1 = cp.multiply(gr_t, wl_t) +  cp.multiply(gg_t, wr_t)

    # time reverse
    wr_t_mod = cp.roll(cp.flip(wr_t, axis=1), 1, axis=1)

    # multiply elementwise the energy reversed with difference of transposed and energy zero
    # see the document "derivation_selfenergy.pdf" for an explanation
    sg_t_2 = cp.multiply(rgg_t, wl_transposed_t - cp.repeat(wl_transposed_gpu[:,0].reshape(-1,1), 2*ne, axis=1))
    sl_t_2 = cp.multiply(rgl_t, wg_transposed_t - cp.repeat(wg_transposed_gpu[:,0].reshape(-1,1), 2*ne, axis=1))
    sr_t_2 = (cp.multiply(rgg_t, cp.conjugate(wr_t_mod - cp.repeat(wr_gpu[:,0].reshape(-1,1), 2*ne, axis=1))) +
              cp.multiply(rgr_t, wg_transposed_t - cp.repeat(wg_gpu[:,0].reshape(-1,1), 2*ne, axis=1)))


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
