"""Functions to calculate the self-energies.
Where the inputs Green's Functions and the Screening is in the sparse format.

The self-energies are calculated in the following way on a discrete energy grid:

\Sigma^{\lessgtr}_{ij} \left(E\right) = 
i*\frac{dE}{2*\pi} \sum \limits_{E^{\prime}} 
G^{\lessgtr}_{ij}\left(E^{\prime}\right) W^{\lessgtr}_{ij}\left(E-E^{\prime}\right)


\Sigma^{r}_{ij} \left(E\right) = 
i*\frac{dE}{2*\pi} \sum \limits_{E^{\prime}} 
G^{r}_{ij}\left(E^{\prime}\right) W^{<}_{ij}\left(E-E^{\prime}\right) +
G^{>}_{ij}\left(E^{\prime}\right) W^{r}_{ij}\left(E-E^{\prime}\right)

As all of the formulas are convolutions, the convolution theorem is used.
The convolution is the same as the inverse fourier transform of the product
of both fourier transformation with the right padding.

In addition, in the previous step the energy grid for W/P 
got cutoff to the same as G. Such we need to adapt the formula above
to account for this error with using identities for W. 
The derivation can be found in the document selfenergy/docs/derivation_selfenergy.pdf.

To be noted:
- The document does not derive why the double counting of energy zero happens
- All the derivations are for the continuous case
- Only fft implementation exist, since direct convolution calculations are not competitive,
but nice for testing against.

"""
import numpy as np
import numpy.typing as npt
import typing
# import cupy as cp
import sys
import os

main_path = os.path.abspath(os.path.dirname(__file__))
parent_path = os.path.abspath(os.path.join(main_path, "..", "..", ".."))

from GW.polarization.sparse import helper


# def gw2s_fft_gpu_fullgrid(
#     pre_factor: np.complex128,
#     gg: cp.ndarray,
#     gl: cp.ndarray,
#     gr: cp.ndarray,
#     wg: cp.ndarray,
#     wl: cp.ndarray,
#     wr: cp.ndarray
# ) -> typing.Tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
#     """Calculate the self energy with fft on the gpu(see file description todo). 
#         The inputs are the pre factor, the Green's Functions
#         and the screened interactions.
#         This function only gives the correct result if in the previous steps 
#         the energy grid is not cutoff.

#     Args:
#         pre_factor   (np.complex128): pre_factor, multiplied at the end
#         gg              (cp.ndarray): Greater Green's Function,     (#orbital, #energy)
#         gl              (cp.ndarray): Lesser Green's Function,      (#orbital, #energy)
#         gr              (cp.ndarray): Retarded Green's Function,    (#orbital, #energy)
#         wg              (cp.ndarray): Greater screened interaction, (#orbital, #energy)
#         wl              (cp.ndarray): Lesser screened interaction,  (#orbital, #energy)
#         wr              (cp.ndarray): Retarded screened interaction,(#orbital, #energy)

#     Returns:
#         typing.Tuple[cp.ndarray, Greater self energy  (#orbital, #energy)
#                      cp.ndarray, Lesser self energy   (#orbital, #energy)
#                      cp.ndarray  Retarded self energy (#orbital, #energy)
#                     ]
#     """
#     # number of energy points
#     ne: int = gg.shape[1]

#     # todo possiblity to avoid fft in global chain
#     # fft
#     gg_t = cp.fft.fft(gg, n=2 * ne, axis=1)
#     gl_t = cp.fft.fft(gl, n=2 * ne, axis=1)
#     gr_t = cp.fft.fft(gr, n=2 * ne, axis=1)
#     wg_t = cp.fft.fft(wg, n=2 * ne, axis=1)
#     wl_t = cp.fft.fft(wl, n=2 * ne, axis=1)
#     wr_t = cp.fft.fft(wr, n=2 * ne, axis=1)

#     # multiply elementwise
#     sg_t = cp.multiply(gg_t, wg_t)
#     sl_t = cp.multiply(gl_t, wl_t)
#     sr_t = cp.multiply(gr_t, wl_t) +  cp.multiply(gg_t, wr_t)

#     # ifft, cutoff and multiply with pre factor
#     sg = cp.multiply(cp.fft.ifft(sg_t, axis=1)[:, :ne], pre_factor)
#     sl = cp.multiply(cp.fft.ifft(sl_t, axis=1)[:, :ne], pre_factor)
#     sr = cp.multiply(cp.fft.ifft(sr_t, axis=1)[:, :ne], pre_factor)

#     return (sg, sl, sr)


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
    gg_t = helper.fft_numba(gg, ne2, no)
    gl_t = helper.fft_numba(gl, ne2, no)
    gr_t = helper.fft_numba(gr, ne2, no)
    wg_t = helper.fft_numba(wg, ne2, no)
    wl_t = helper.fft_numba(wl, ne2, no)
    wr_t = helper.fft_numba(wr, ne2, no)

    # fft of energy reversed
    rgg_t =  helper.fft_numba(np.flip(gg, axis=1), ne2, no)
    rgl_t =  helper.fft_numba(np.flip(gl, axis=1), ne2, no)
    rgr_t =  helper.fft_numba(np.flip(gr, axis=1), ne2, no)

    # multiply elementwise for sigma_1 the normal term
    sg_t_1 = helper.elementmul(gg_t, wg_t)
    sl_t_1 = helper.elementmul(gl_t, wl_t)
    sr_t_1 = helper.elementmul(gr_t, wl_t) +  helper.elementmul(gg_t, wr_t)

    # time reverse 
    wr_t_mod = np.roll(np.flip(wr_t, axis=1), 1, axis=1)

    # multiply elementwise the energy reversed with difference of transposed and energy zero
    # see the document "derivation_selfenergy.pdf" for an explanation
    sg_t_2 = helper.elementmul(rgg_t, wl_t[ij2ji,:] - np.repeat(wl[ij2ji,0].reshape(-1,1), 2*ne, axis=1))
    sl_t_2 = helper.elementmul(rgl_t, wg_t[ij2ji,:] - np.repeat(wg[ij2ji,0].reshape(-1,1), 2*ne, axis=1))
    sr_t_2 = (helper.elementmul(rgg_t, np.conjugate(wr_t_mod - np.repeat(wr[:,0].reshape(-1,1), 2*ne, axis=1))) +
              helper.elementmul(rgr_t, wg_t[ij2ji,:] - np.repeat(wg[:,0].reshape(-1,1), 2*ne, axis=1)))


    # ifft, cutoff and multiply with pre factor
    sg_1 = helper.scalarmul_ifft(sg_t_1, pre_factor, ne, no)
    sl_1 = helper.scalarmul_ifft(sl_t_1, pre_factor, ne, no)
    sr_1 = helper.scalarmul_ifft(sr_t_1, pre_factor, ne, no)

    sg_2 = np.flip(helper.scalarmul_ifft(sg_t_2, pre_factor, ne, no), axis=1)
    sl_2 = np.flip(helper.scalarmul_ifft(sl_t_2, pre_factor, ne, no), axis=1)
    sr_2 = np.flip(helper.scalarmul_ifft(sr_t_2, pre_factor, ne, no), axis=1)


    sg = sg_1 + sg_2
    sl = sl_1 + sl_2
    sr = sr_1 + sr_2

    return (sg, sl, sr)

# def gw2s_fft_gpu(
#     pre_factor: np.complex128,
#     ij2ji: cp.ndarray,
#     gg: cp.ndarray,
#     gl: cp.ndarray,
#     gr: cp.ndarray,
#     wg: cp.ndarray,
#     wl: cp.ndarray,
#     wr: cp.ndarray
# ) -> typing.Tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
#     """Calculate the self energy with fft on the gpu(see file description todo). 
#         The inputs are the pre factor, the Green's Functions
#         and the screened interactions.
#         Takes into account the energy grid cutoff

#     Args:
#         pre_factor   (np.complex128): pre_factor, multiplied at the end
#         ij2ji           (cp.ndarray): mapping to transposed matrix, (#orbital)
#         gg              (cp.ndarray): Greater Green's Function,     (#orbital, #energy)
#         gl              (cp.ndarray): Lesser Green's Function,      (#orbital, #energy)
#         gr              (cp.ndarray): Retarded Green's Function,    (#orbital, #energy)
#         wg              (cp.ndarray): Greater screened interaction, (#orbital, #energy)
#         wl              (cp.ndarray): Lesser screened interaction,  (#orbital, #energy)
#         wr              (cp.ndarray): Retarded screened interaction,(#orbital, #energy)

#     Returns:
#         typing.Tuple[cp.ndarray, Greater self energy  (#orbital, #energy)
#                      cp.ndarray, Lesser self energy   (#orbital, #energy)
#                      cp.ndarray  Retarded self energy (#orbital, #energy)
#                     ]
#     """
#     # number of energy points
#     ne = gg.shape[1]

#     # todo possibility to avoid fft in global chain
#     # fft
#     gg_t = cp.fft.fft(gg, n=2 * ne, axis=1)
#     gl_t = cp.fft.fft(gl, n=2 * ne, axis=1)
#     gr_t = cp.fft.fft(gr, n=2 * ne, axis=1)
#     wg_t = cp.fft.fft(wg, n=2 * ne, axis=1)
#     wl_t = cp.fft.fft(wl, n=2 * ne, axis=1)
#     wr_t = cp.fft.fft(wr, n=2 * ne, axis=1)

#     # fft of energy reversed
#     rgg_t = cp.fft.fft(cp.flip(gg, axis=1), n=2 * ne, axis=1)
#     rgl_t = cp.fft.fft(cp.flip(gl, axis=1), n=2 * ne, axis=1)
#     rgr_t = cp.fft.fft(cp.flip(gr, axis=1), n=2 * ne, axis=1)

#     # multiply elementwise for sigma_1 the normal term
#     sg_t_1 = cp.multiply(gg_t, wg_t)
#     sl_t_1 = cp.multiply(gl_t, wl_t)
#     sr_t_1 = cp.multiply(gr_t, wl_t) +  cp.multiply(gg_t, wr_t)

#     # time reverse 
#     wr_t_mod = cp.roll(cp.flip(wr_t, axis=1), 1, axis=1)    

#     # multiply elementwise the energy reversed with difference of transposed and energy zero
#     # see the document "derivation_selfenergy.pdf" for an explanation
#     sg_t_2 = cp.multiply(rgg_t, wl_t[ij2ji,:] - cp.repeat(wl[ij2ji,0].reshape(-1,1), 2*ne, axis=1))
#     sl_t_2 = cp.multiply(rgl_t, wg_t[ij2ji,:] - cp.repeat(wg[ij2ji,0].reshape(-1,1), 2*ne, axis=1))
#     sr_t_2 = (cp.multiply(rgg_t, cp.conjugate(wr_t_mod - cp.repeat(wr[:,0].reshape(-1,1), 2*ne, axis=1))) +
#               cp.multiply(rgr_t, wg_t[ij2ji,:] - cp.repeat(wg[:,0].reshape(-1,1), 2*ne, axis=1)))


#     # ifft, cutoff and multiply with pre factor
#     sg_1 = cp.fft.ifft(sg_t_1, axis=1)[:, :ne]
#     sl_1 = cp.fft.ifft(sl_t_1, axis=1)[:, :ne]
#     sr_1 = cp.fft.ifft(sr_t_1, axis=1)[:, :ne]

#     sg_2 = cp.flip(cp.fft.ifft(sg_t_2, axis=1)[:, :ne], axis=1)
#     sl_2 = cp.flip(cp.fft.ifft(sl_t_2, axis=1)[:, :ne], axis=1)
#     sr_2 = cp.flip(cp.fft.ifft(sr_t_2, axis=1)[:, :ne], axis=1)


#     sg = cp.multiply(sg_1 + sg_2, pre_factor)
#     sl = cp.multiply(sl_1 + sl_2, pre_factor)
#     sr = cp.multiply(sr_1 + sr_2, pre_factor)

#     return (sg, sl, sr)
