"""Functions to calculate the polarization. 
Where the Green's Functions are in the sparse format.

The polarization is calculated in the following way on a discrete energy grid:

P^{\lessgtr}_{ij}\left(E^{\prime}\right) = 
-2*i*\frac{dE}{2 \pi} \sum \limits_{E} G^{\lessgtr}_{ij}\left(E\right) 
G^{\gtrless}_{ji}\left(E-E^{\prime}\right)
P^{r}_{ij}\left(E^{\prime}\right) = 
-2*i*\frac{dE}{2 \pi} \sum \limits_{E} G^{<}_{ij}\left(E\right) 
G^{a}_{ji}\left(E-E^{\prime}\right)+
G^{r}_{ij}\left(E\right) G^{<}_{ji}\left(E-E^{\prime}\right)

There are two main ways to evaluate the above functions:
Either directly evaluating the sum or fft-transforming with the convolution/correlation theorem

--------------------------------------------------------------------------------
The idea for the solution with fft is:

P^{r} = fac*\left( G^{r}_{ij}(t) G^{<}_{ji}(-t) +
G^{<}_{ij}(t) \left(G^{r}_{ij}(t)\right)^{\prime} \right)

-fft with a zero padding of #energy due comply with the convolution theorem

-Reverse and transpose the second input 
due to F(f^*(E)) = g^*(-t), where F(f(E)) = g(t)

-Then elementwise multiply the two inputs and elementwise multiply with a pre-factor in time domain
the pre-factor is given in the following way: -2*i*dE/(2*pi)
with factor two due to spin

-Then ifft

-Cut off the the last #energy points to get the polarization on the same energy grid as the Green's Function


As side notes:

-In the time domain the following identity should hold:
P^{>}_{ij}\left(E\right) = -P^{<}_{ij}\left(-E\right)^{*}
Lesser or Greater polarization can be calculated from this identity.
Saving of a reversal/transpose operation is gained

-The elementwise multiplication with the prefactor can be switched with 
applying the iff/cutting of elements

-The transposing and reversal could be switched with the fft, 
but less efficient since more fft calls would be needed

-Gpu  implementation mirrors cpu one

-Mpi implementation needes the transposed arrays as arguments 
since it is not local per rank possible
as transposing is a global operation
--------------------------------------------------------------------------------


Calculating the convolution directly is straightforward and needs no comments
-was measured to be slower than fft/ifft




"""
import numpy as np
import numpy.typing as npt
import typing
from scipy import fft
import numba
# import cupy as cp
import dace
import sys
import os

main_path = os.path.abspath(os.path.dirname(__file__))
parent_path = os.path.abspath(os.path.join(main_path, "..", ".."))
sys.path.append(parent_path)

from GW.polarization.sparse import helper
# create symbol for dace matrix sizes-------------------------------------------

# number of energy points
NE = dace.symbol("NE")
# number of energy points
NO = dace.symbol("NO")


# define various functions for cpu/gpu with mpi/dace----------------------------
@numba.njit("(c16, i4[:], c16[:,:], c16[:,:], c16[:,:])",
            parallel=True,
            cache=True,
            nogil=True,
            error_model="numpy")
def g2p_fft_cpu(
    pre_factor: np.complex128,
    ij2ji: npt.NDArray[np.int32],
    gg: npt.NDArray[np.complex128],
    gl: npt.NDArray[np.complex128],
    gr: npt.NDArray[np.complex128]
) -> typing.Tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128],
                  npt.NDArray[np.complex128]]:
    """Calculates the polarization with fft on the cpu(see file description). 
        The Green's function and a mapping to the transposed indices are needed.

    Args:
        pre_factor      (np.complex128): pre_factor, multiplied at the end
        ij2ji   (npt.NDArray[np.int32]): mapping to transposed matrix, (#orbital)
        gg (npt.NDArray[np.complex128]): Greater Green's Function,     (#orbital, #energy)
        gl (npt.NDArray[np.complex128]): Lesser Green's Function,      (#orbital, #energy)
        gr (npt.NDArray[np.complex128]): Retarded Green's Function_,   (#orbital, #energy)

    Returns:
        typing.Tuple[npt.NDArray[np.complex128], Greater polarization  (#orbital, #energy)
                     npt.NDArray[np.complex128], Lesser polarization   (#orbital, #energy)
                     npt.NDArray[np.complex128]  Retarded polarization (#orbital, #energy)
                    ] 
    """

    # number of energy points and nnz
    ne = gg.shape[1]
    no = gg.shape[0]
    ne2 = 2 * ne

    # fft
    gg_t: npt.NDArray[np.complex128] = helper.fft_numba(gg,ne2,no)
    gl_t: npt.NDArray[np.complex128] = helper.fft_numba(gl,ne2,no)
    gr_t: npt.NDArray[np.complex128] = helper.fft_numba(gr,ne2,no)

    # reverse and transpose
    gl_t_mod: npt.NDArray[np.complex128] = helper.reversal_transpose(gl_t, ij2ji)
    gg_t_mod: npt.NDArray[np.complex128] = helper.reversal_transpose(gg_t, ij2ji)

    # multiply elementwise
    pg_t: npt.NDArray[np.complex128] = helper.elementmul(gg_t, gl_t_mod)
    pl_t: npt.NDArray[np.complex128] = helper.elementmul(gl_t, gg_t_mod)
    pr_t: npt.NDArray[np.complex128] = helper.elementmul(
        gr_t, gl_t_mod) + helper.elementmul(gl_t, gr_t.conjugate())

    # test identity
    # assert np.allclose(pre_factor * pg_t, -np.conjugate(pre_factor * pl_t))

    # ifft, cutoff and multiply with pre factor
    pg: npt.NDArray[np.complex128] = helper.scalarmul_ifft(
        pg_t, pre_factor, ne, no)
    pl: npt.NDArray[np.complex128] = helper.scalarmul_ifft(
        pl_t, pre_factor, ne, no)
    pr: npt.NDArray[np.complex128] = helper.scalarmul_ifft(
        pr_t, pre_factor, ne, no)

    return (pg, pl, pr)


@numba.njit("(c16, i4[:], c16[:,:], c16[:,:], c16[:,:])",
            parallel=True,
            cache=True,
            nogil=True,
            error_model="numpy")
def g2p_fft_cpu_inlined(
    pre_factor: np.complex128,
    ij2ji: npt.NDArray[np.int32],
    gg: npt.NDArray[np.complex128],
    gl: npt.NDArray[np.complex128],
    gr: npt.NDArray[np.complex128]
) -> typing.Tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128],
                  npt.NDArray[np.complex128]]:
    """Calculates the polarization with fft on the cpu(see file description). 
        The Green's function and a mapping to the transposed indices are needed.

    Args:
        pre_factor      (np.complex128): pre_factor, multiplied at the end
        ij2ji   (npt.NDArray[np.int32]): mapping to transposed matrix, (#orbital)
        gg (npt.NDArray[np.complex128]): Greater Green's Function,     (#orbital, #energy)
        gl (npt.NDArray[np.complex128]): Lesser Green's Function,      (#orbital, #energy)
        gr (npt.NDArray[np.complex128]): Retarded Green's Function_,   (#orbital, #energy)

    Returns:
        typing.Tuple[npt.NDArray[np.complex128], Greater polarization  (#orbital, #energy)
                     npt.NDArray[np.complex128], Lesser polarization   (#orbital, #energy)
                     npt.NDArray[np.complex128]  Retarded polarization (#orbital, #energy)
                    ] 
    """

    # number of energy points and nnz
    ne = gg.shape[1]
    no = gg.shape[0]
    ne2 = 2 * ne

    # fft
    gg_t: npt.NDArray[np.complex128] = np.empty((no,ne2), dtype=np.complex128)
    gl_t: npt.NDArray[np.complex128] = np.empty((no,ne2), dtype=np.complex128)
    gr_t: npt.NDArray[np.complex128] = np.empty((no,ne2), dtype=np.complex128)
    for i in numba.prange(no):
        gg_t[i,:] = fft.fft(gg[i,:], n=ne2, workers=1)
        gl_t[i,:] = fft.fft(gl[i,:], n=ne2, workers=1)
        gr_t[i,:] = fft.fft(gr[i,:], n=ne2, workers=1)


    # reverse and transpose
    gl_t_mod: npt.NDArray[np.complex128] = np.empty_like(gl_t, dtype=np.complex128)
    gg_t_mod: npt.NDArray[np.complex128] = np.empty_like(gl_t, dtype=np.complex128)

    tmpl: npt.NDArray[np.complex128] = np.empty_like(gl_t, dtype=np.complex128)
    tmpg: npt.NDArray[np.complex128] = np.empty_like(gl_t, dtype=np.complex128)

    for i in numba.prange(no):
        for j in range(ne2):
            tmpl[i, j] = gl_t[i, -j]
            tmpg[i, j] = gg_t[i, -j]
    for i in numba.prange(no):
        gl_t_mod[ij2ji[i], :] = tmpl[i, :]
        gg_t_mod[ij2ji[i], :] = tmpg[i, :]

    # multiply elementwise
    pg_t: npt.NDArray[np.complex128] = np.empty_like(gl_t, dtype=np.complex128)
    pl_t: npt.NDArray[np.complex128] = np.empty_like(gl_t, dtype=np.complex128)
    pr_t: npt.NDArray[np.complex128] = np.empty_like(gl_t, dtype=np.complex128)

    for i in numba.prange(no):
        for j in numba.prange(ne2):
            pg_t[i,j] = gg_t[i,j] * gl_t_mod[i,j]
            pl_t[i,j] = gl_t[i,j] * gg_t_mod[i,j]
            pr_t[i,j] = gr_t[i,j] * gl_t_mod[i,j] + gl_t[i,j] * np.conjugate(gr_t[i,j])


    # ifft, cutoff and multiply with pre factor
    pg: npt.NDArray[np.complex128] = np.empty_like(gg, dtype=np.complex128)
    pl: npt.NDArray[np.complex128] = np.empty_like(gg, dtype=np.complex128)
    pr: npt.NDArray[np.complex128] = np.empty_like(gg, dtype=np.complex128)

    for i in numba.prange(no):
        pg[i,:] = fft.ifft(pg_t[i,:], workers=1)[:ne]
        pl[i,:] = fft.ifft(pl_t[i,:], workers=1)[:ne]
        pr[i,:] = fft.ifft(pr_t[i,:], workers=1)[:ne]
    for i in numba.prange(no):
        for j in numba.prange(ne):
            pg[i, j] = pg[i, j] * pre_factor
            pl[i, j] = pl[i, j] * pre_factor
            pr[i, j] = pr[i, j] * pre_factor

    return (pg, pl, pr)


@numba.njit("(c16, i4[:], c16[:,:], c16[:,:], c16[:,:])",
            parallel=True,
            cache=True,
            nogil=True,
            error_model="numpy")
def g2p_conv_cpu(
    pre_factor: np.complex128,
    ij2ji: npt.NDArray[np.int32],
    gg: npt.NDArray[np.complex128],
    gl: npt.NDArray[np.complex128],
    gr: npt.NDArray[np.complex128]
) -> typing.Tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128],
                  npt.NDArray[np.complex128]]:
    """Calculates the polarization with convolution on the cpu(see file description). 
        The Green's function and a mapping to the transposed indices are needed.
        Is njit compiled

    Args:
        pre_factor      (np.complex128): pre_factor, multiplied at the end
        ij2ji   (npt.NDArray[np.int32]): mapping to transposed matrix, (#orbital)
        gg (npt.NDArray[np.complex128]): Greater Green's Function,     (#orbital, #energy)
        gl (npt.NDArray[np.complex128]): Lesser Green's Function,      (#orbital, #energy)
        gr (npt.NDArray[np.complex128]): Retarded Green's Function,   (#orbital, #energy)

    Returns:
        typing.Tuple[npt.NDArray[np.complex128], Greater polarization  (#orbital, #energy)
                     npt.NDArray[np.complex128], Lesser polarization   (#orbital, #energy)
                     npt.NDArray[np.complex128]  Retarded polarization (#orbital, #energy)
                    ] 
    """

    # #energy
    no: np.int32 = gg.shape[0]
    # nnz
    ne: np.int32 = gg.shape[1]

    # create polarization arrays
    pg: npt.NDArray[np.complex128] = np.empty_like(gg, dtype=np.complex128)
    pl: npt.NDArray[np.complex128] = np.empty_like(gg, dtype=np.complex128)
    pr: npt.NDArray[np.complex128] = np.empty_like(gg, dtype=np.complex128)

    # evaluate convolution
    for ij in numba.prange(no):
        for e in numba.prange(ne):
            ji = ij2ji[ij]
            tmpg = 0
            tmpl = 0
            tmpr = 0
            for ep in range(e, ne):
                tmpg += pre_factor * gg[ij, ep] * gl[ji, ep-e]
                tmpl += pre_factor * gl[ij, ep] * gg[ji, ep-e]
                tmpr += pre_factor * (gr[ij, ep] * gl[ji, ep-e] +
                gl[ij, ep] * np.conjugate(gr[ij, ep-e]))
            pg[ij, e] = tmpg
            pl[ij, e] = tmpl
            pr[ij, e] = tmpr

    return (pg, pl, pr)


# def g2p_fft_gpu(
#     pre_factor: np.complex128,
#     ij2ji: cp.ndarray,
#     gg: cp.ndarray,
#     gl: cp.ndarray,
#     gr: cp.ndarray
# ) -> typing.Tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
#     """Calculate the polarization with fft on the gpu(see file description). 
#         The inputs are the pre factor and the Green's Functions.
#         Only the data and a mapping to the transposed indices are needed.

#     Args:
#         pre_factor       (np.complex128): pre_factor, multiplied at the end
#         ij2ji               (cp.ndarray): mapping to transposed matrix, (#orbital)
#         gg                  (cp.ndarray): Greater Green's Function,     (#orbital, #energy)
#         gl                  (cp.ndarray): Lesser Green's Function,      (#orbital, #energy)
#         gr                  (cp.ndarray): Retarded Green's Function,    (#orbital, #energy)

#     Returns:
#         typing.Tuple[cp.ndarray, Greater polarization  (#orbital, #energy)
#                      cp.ndarray, Lesser polarization   (#orbital, #energy)
#                      cp.ndarray  Retarded polarization (#orbital, #energy)
#                     ]
#     """

#     # number of energy points
#     ne: int = gg.shape[1]

#     # fft
#     gg_t = cp.fft.fft(gg, n=2 * ne, axis=1)
#     gl_t = cp.fft.fft(gl, n=2 * ne, axis=1)
#     gr_t = cp.fft.fft(gr, n=2 * ne, axis=1)

#     # reverse and transpose
#     # only once since identity is used for lesser polarization
#     gl_t_mod = cp.roll(cp.flip(gl_t, axis=1), 1, axis=1)[ij2ji, :]

#     # multiply elementwise
#     pg_t = cp.multiply(gg_t, gl_t_mod)
#     pr_t = helper.retarded_special_gpu(gr_t, gl_t_mod, gl_t)

#     # ifft, cutoff and multiply with pre factor
#     pr = cp.multiply(cp.fft.ifft(pr_t, axis=1)[:, :ne], pre_factor)
#     pg = cp.multiply(cp.fft.ifft(pg_t, axis=1), pre_factor)

#     # lesser polarization from identity
#     pl = -cp.conjugate(cp.roll(cp.flip(pg, axis=1), 1, axis=1))

#     # cutoff
#     pg = pg[:, :ne]
#     pl = pl[:, :ne]

#     return (pg, pl, pr)

# def g2p_conv_gpu(iteration: np.int32 = 1):
#     """Creates function to calculate the polarization with conv on the gpu(see file description). 

#     True Args: 
#         iteration (np.int32): code iteration to test    
    
#     The created function needs the following arguments and give following returns:

#     Args:
#         prefactor   (np.cdouble): 
#         ij2ji       (cp.ndarray): mapping to transposed matrix, (#orbital)
#         gg          (cp.ndarray): Greater Green's Function,     (#orbital, #energy)
#         gl          (cp.ndarray): Lesser Green's Function,      (#orbital, #energy)
#         gr          (cp.ndarray): Retarded Green's Function_,   (#orbital, #energy)
#         pg          (cp.ndarray): Greater Green's polarization,     (#orbital, #energy)
#         pl          (cp.ndarray): Lesser Green's polarization,      (#orbital, #energy)
#         pr          (cp.ndarray): Retarded Green's polarization,   (#orbital, #energy)
#         no                 (int): number of nnz/#orbital
#         ne                 (int): number of energy points/#energy
#     """
#     if iteration == 1:
#         code = r'''
#         #include <cupy/complex.cuh>
#         extern "C" __global__
#         void g2p_conv(const complex<double> prefactor, 
#                     const int* ij2ji, 
#                     const complex<double>* gg, 
#                     const complex<double>* gl, 
#                     const complex<double>* gr, 
#                     complex<double>* pg, 
#                     complex<double>* pl, 
#                     complex<double>* pr,
#                     int no,
#                     int ne) {
#             for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
#                  idx < no;
#                  idx += gridDim.x * blockDim.x) {
#                 int ji = ij2ji[idx];
#                 for (int idy = blockIdx.y * blockDim.y + threadIdx.y;
#                      idy < ne;
#                      idy += gridDim.y * blockDim.y) {
#                     complex<double> tmpg;
#                     complex<double> tmpl;
#                     complex<double> tmpr;
#                     for (int ep = idy; ep < ne; ep++) {
#                         complex<double> tmpgl1 = gl[ep + idx * ne];
#                         complex<double> tmpgl2 = gl[ep - idy + ji * ne];
#                         tmpg = tmpg + prefactor * gg[ep + idx * ne] * tmpgl2;     
#                         tmpl = tmpl + prefactor * tmpgl1 * gg[ep - idy + ji * ne];     
#                         tmpr = tmpr + prefactor * (gr[ep + idx * ne] * tmpgl2 + 
#                                                    tmpgl1 * conj(gr[ep - idy + idx * ne]));     
#                     }
#                     pg[idy + idx * ne] = tmpg;
#                     pl[idy + idx * ne] = tmpl;
#                     pr[idy + idx * ne] = tmpr;
#                 }
#             }
#         }
#         '''
#     else:
#         raise ValueError(
#                 "iteration " + str(iteration) + " does not exist")

#     kernel = cp.RawKernel(code, "g2p_conv")

#     return kernel

# def g2p_fft_mpi_gpu(
#     pre_factor: np.complex128,
#     gg: npt.NDArray[np.complex128],
#     gl: npt.NDArray[np.complex128],
#     gr: npt.NDArray[np.complex128],
#     gl_trans: npt.NDArray[np.complex128]
# ) -> typing.Tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128],
#                   npt.NDArray[np.complex128]]:
#     """Calculate the polarization with fft and mpi on the gpu(see file description). 

#     Args:
#         pre_factor            (np.complex128): pre_factor, multiplied at the end
#         gg       (npt.NDArray[np.complex128]): Greater Green's Function,          (#orbital/#ranks, #energy)
#         gl       (npt.NDArray[np.complex128]): Lesser Green's Function,           (#orbital/#ranks, #energy)
#         gr       (npt.NDArray[np.complex128]): Retarded Green's Function_,        (#orbital/#ranks, #energy)
#         gl_trans (npt.NDArray[np.complex128]): Transposed Lesser Green's Function (#orbital/#ranks, #energy)

#     Returns:
#         typing.Tuple[npt.NDArray[np.complex128], Greater polarization  (#orbital, #energy)  
#                      npt.NDArray[np.complex128], Lesser polarization   (#orbital, #energy)
#                      npt.NDArray[np.complex128]  Retarded polarization (#orbital, #energy)
#                     ]
#     """
#     # number of energy points
#     ne: int = gg.shape[1]
    
#     # load data to gpu----------------------------------------------------------
#     gg_gpu = cp.asarray(gg)
#     gl_gpu = cp.asarray(gl)
#     gr_gpu = cp.asarray(gr)
#     gl_trans_gpu = cp.asarray(gl_trans)

#     # compute pg/pl/pr----------------------------------------------------------

#     # fft
#     gg_t_gpu = cp.fft.fft(gg_gpu, n=2 * ne, axis=1)
#     gl_t_gpu = cp.fft.fft(gl_gpu, n=2 * ne, axis=1)
#     gr_t_gpu = cp.fft.fft(gr_gpu, n=2 * ne, axis=1)
#     gl_t_trans_gpu = cp.fft.fft(gl_trans_gpu, n=2 * ne, axis=1)

#     # time reversed
#     gl_t_mod_gpu = cp.roll(cp.flip(gl_t_trans_gpu, axis=1), 1, axis=1)

#     # multiply elementwise
#     pg_t_gpu = cp.multiply(gg_t_gpu, gl_t_mod_gpu)
#     pr_t_gpu = helper.retarded_special_gpu(gr_t_gpu, gl_t_mod_gpu, gl_t_gpu)

#     # ifft, cutoff and multiply with pre factor
#     pr_gpu = cp.multiply(cp.fft.ifft(pr_t_gpu, axis=1)[:, :ne], pre_factor)
#     pg_gpu = cp.multiply(cp.fft.ifft(pg_t_gpu, axis=1), pre_factor)

#     # lesser polarization from identity
#     pl_gpu = -cp.conjugate(cp.roll(cp.flip(pg_gpu, axis=1), 1, axis=1))

#     # cutoff
#     pg_gpu = pg_gpu[:, :ne]
#     pl_gpu = pl_gpu[:, :ne]


#     # load data to cpu----------------------------------------------------------

#     pg = cp.asnumpy(pg_gpu)
#     pl = cp.asnumpy(pl_gpu)
#     pr = cp.asnumpy(pr_gpu)

#     return (pg, pl, pr)

@dace.program(auto_optimize=True)
def g2p_conv_dace(pre_factor: dace.complex128[1],
                 ij2ji: dace.int32[NO],
                 gg: dace.complex128[NO,NE],
                 gl: dace.complex128[NO,NE],
                 gr: dace.complex128[NO,NE],
                 pg: dace.complex128[NO,NE],
                 pl: dace.complex128[NO,NE],
                 pr: dace.complex128[NO,NE]
                ):
    for ij in range(NO):
        for e in range(NE):
            for ep in range(e, NE):
                ji = ij2ji[ij]
                pg[ij, e] += pre_factor[0] * gg[ij, ep] * gl[ji, ep-e]
                pl[ij, e] += pre_factor[0] * gl[ij, ep] * gg[ji, ep-e]
                pr[ij, e] += pre_factor[0] * (gr[ij, ep] * gl[ji, ep-e] + gl[ij, ep] * np.conjugate(gr[ij, ep-e]))

@dace.program(auto_optimize=True)
def g2p_fft_dace(
                 pre_factor: dace.complex128[1],
                 ij2ji: dace.int32[NO],
                 gg: dace.complex128[NO,NE],
                 gl: dace.complex128[NO,NE],
                 gr: dace.complex128[NO,NE],
                 pg: dace.complex128[NO,NE],
                 pl: dace.complex128[NO,NE],
                 pr: dace.complex128[NO,NE]
                ):

    # fft
    gg_t = dace.ndarray((NO,2*NE), dtype=dace.complex128)
    gl_t = dace.ndarray((NO,2*NE), dtype=dace.complex128)
    gr_t = dace.ndarray((NO,2*NE), dtype=dace.complex128)
    for i in dace.map[0:NO]:
        gg_t[i,:] = fft.fft(gg[i,:], n=2*NE, workers=1)
        gl_t[i,:] = fft.fft(gl[i,:], n=2*NE, workers=1)
        gr_t[i,:] = fft.fft(gr[i,:], n=2*NE, workers=1)


    # reverse and transpose
    gl_t_mod: npt.NDArray[np.complex128] = dace.ndarray((NO,2*NE), dtype=dace.complex128)
    gg_t_mod: npt.NDArray[np.complex128] = dace.ndarray((NO,2*NE), dtype=dace.complex128)

    tmpl: npt.NDArray[np.complex128] = dace.ndarray((NO,2*NE), dtype=dace.complex128)
    tmpg: npt.NDArray[np.complex128] = dace.ndarray((NO,2*NE), dtype=dace.complex128)

    for i in dace.map[0:NO]:
        for j in dace.map[0:2*NE]:
            tmpl[i, j] = gl_t[i, -j]
            tmpg[i, j] = gg_t[i, -j]
    for i in dace.map[0:NO]:
        gl_t_mod[ij2ji[i], :] = tmpl[i, :]
        gg_t_mod[ij2ji[i], :] = tmpg[i, :]

    # multiply elementwise
    pg_t: npt.NDArray[np.complex128] = dace.ndarray((NO,2*NE), dtype=dace.complex128)
    pl_t: npt.NDArray[np.complex128] = dace.ndarray((NO,2*NE), dtype=dace.complex128)
    pr_t: npt.NDArray[np.complex128] = dace.ndarray((NO,2*NE), dtype=dace.complex128)

    for i in dace.map[0:NO]:
        for j in dace.map[0:2*NE]:
            pg_t[i,j] = gg_t[i,j] * gl_t_mod[i,j]
            pl_t[i,j] = gl_t[i,j] * gg_t_mod[i,j]
            pr_t[i,j] = gr_t[i,j] * gl_t_mod[i,j] + gl_t[i,j] * np.conjugate(gr_t[i,j])

    # ifft, cutoff and multiply with pre factor
    for i in dace.map[0:NO]:
        pg[i,:] = fft.ifft(pg_t[i,:], workers=1)[:NE]
        pl[i,:] = fft.ifft(pl_t[i,:], workers=1)[:NE]
        pr[i,:] = fft.ifft(pr_t[i,:], workers=1)[:NE]
    for i in dace.map[0:NO]:
        for j in dace.map[0:NE]:
            pg[i, j] = pg[i, j] * pre_factor[0]
            pl[i, j] = pl[i, j] * pre_factor[0]
            pr[i, j] = pr[i, j] * pre_factor[0]


