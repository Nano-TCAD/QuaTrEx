# Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.
""" Functions to calculate the polarization on the cpu. See README.md for more information. """

import numpy as np
import numpy.typing as npt
import typing
from numpy import fft
import numba

import dace
import sys
import os

main_path = os.path.abspath(os.path.dirname(__file__))
parent_path = os.path.abspath(os.path.join(main_path, "..", "..", ".."))
sys.path.append(parent_path)

from quatrex.utils import linalg_cpu

# create symbol for dace matrix sizes-------------------------------------------

# number of energy points
NE = dace.symbol("NE")
# number of energy points
NO = dace.symbol("NO")


# define various functions for cpu/gpu with mpi/dace----------------------------
@numba.njit("(c16, i4[:], c16[:,:], c16[:,:], c16[:,:])", parallel=True, cache=True, nogil=True, error_model="numpy")
def g2p_fft_cpu(
    pre_factor: np.complex128, ij2ji: npt.NDArray[np.int32], gg: npt.NDArray[np.complex128],
    gl: npt.NDArray[np.complex128], gr: npt.NDArray[np.complex128]
) -> typing.Tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128], npt.NDArray[np.complex128]]:
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
    gg_t: npt.NDArray[np.complex128] = linalg_cpu.fft_numba(gg, ne2, no)
    gl_t: npt.NDArray[np.complex128] = linalg_cpu.fft_numba(gl, ne2, no)
    gr_t: npt.NDArray[np.complex128] = linalg_cpu.fft_numba(gr, ne2, no)

    # reverse and transpose
    gl_t_mod: npt.NDArray[np.complex128] = linalg_cpu.reversal_transpose(gl_t, ij2ji)
    gg_t_mod: npt.NDArray[np.complex128] = linalg_cpu.reversal_transpose(gg_t, ij2ji)

    # multiply elementwise
    pg_t: npt.NDArray[np.complex128] = linalg_cpu.elementmul(gg_t, gl_t_mod)
    pl_t: npt.NDArray[np.complex128] = linalg_cpu.elementmul(gl_t, gg_t_mod)
    pr_t: npt.NDArray[np.complex128] = linalg_cpu.elementmul(gr_t, gl_t_mod) + linalg_cpu.elementmul(
        gl_t, gr_t.conjugate())

    # test identity
    # assert np.allclose(pre_factor * pg_t, -np.conjugate(pre_factor * pl_t))

    # ifft, cutoff and multiply with pre factor
    pg: npt.NDArray[np.complex128] = linalg_cpu.scalarmul_ifft_cutoff(pg_t, pre_factor, ne, no)
    pl: npt.NDArray[np.complex128] = linalg_cpu.scalarmul_ifft_cutoff(pl_t, pre_factor, ne, no)
    pr: npt.NDArray[np.complex128] = linalg_cpu.scalarmul_ifft_cutoff(pr_t, pre_factor, ne, no)

    return (pg, pl, pr)


@numba.njit("(c16, c16[:,:], c16[:,:], c16[:,:], c16[:,:])", parallel=True, cache=True, nogil=True, error_model="numpy")
def g2p_fft_mpi_cpu(
    pre_factor: np.complex128, gg: npt.NDArray[np.complex128], gl: npt.NDArray[np.complex128],
    gr: npt.NDArray[np.complex128], gl_transposed: npt.NDArray[np.complex128]
) -> typing.Tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128], npt.NDArray[np.complex128]]:
    """Calculates the polarization with fft on the cpu(see file description). 
        The Green's function and a mapping to the transposed indices are needed.

    Args:
        pre_factor      (np.complex128): pre_factor, multiplied at the end
        gg (npt.NDArray[np.complex128]): Greater Green's Function,     (#orbital, #energy)
        gl (npt.NDArray[np.complex128]): Lesser Green's Function,      (#orbital, #energy)
        gr (npt.NDArray[np.complex128]): Retarded Green's Function_,   (#orbital, #energy)
        gl_transposed (npt.NDArray[np.complex128]): Lesser Green's Function, in orbital transposed (#orbital, #energy)

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
    gg_t: npt.NDArray[np.complex128] = linalg_cpu.fft_numba(gg, ne2, no)
    gl_t: npt.NDArray[np.complex128] = linalg_cpu.fft_numba(gl, ne2, no)
    gr_t: npt.NDArray[np.complex128] = linalg_cpu.fft_numba(gr, ne2, no)
    gl_transposed_t: npt.NDArray[np.complex128] = linalg_cpu.fft_numba(gl_transposed, ne2, no)

    # time reversed
    gl_t_mod: npt.NDArray[np.complex128] = linalg_cpu.reversal(gl_transposed_t)

    # multiply elementwise
    pg_t: npt.NDArray[np.complex128] = linalg_cpu.elementmul(gg_t, gl_t_mod)
    pr_t: npt.NDArray[np.complex128] = linalg_cpu.elementmul(gr_t, gl_t_mod) + linalg_cpu.elementmul(
        gl_t, gr_t.conjugate())

    # test identity
    # assert np.allclose(pre_factor * pg_t, -np.conjugate(pre_factor * pl_t))

    # ifft, and multiply with pre factor
    pg: npt.NDArray[np.complex128] = linalg_cpu.scalarmul_ifft(pg_t, pre_factor, ne, no)
    pr: npt.NDArray[np.complex128] = linalg_cpu.scalarmul_ifft(pr_t, pre_factor, ne, no)
    # lesser polarization from identity
    pl = -np.conjugate(linalg_cpu.reversal(pg))

    return (pg[:, :ne], pl[:, :ne], pr[:, :ne])


@numba.njit("(c16, i4[:], c16[:,:], c16[:,:], c16[:,:])", parallel=True, cache=True, nogil=True, error_model="numpy")
def g2p_fft_cpu_inlined(
    pre_factor: np.complex128, ij2ji: npt.NDArray[np.int32], gg: npt.NDArray[np.complex128],
    gl: npt.NDArray[np.complex128], gr: npt.NDArray[np.complex128]
) -> typing.Tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128], npt.NDArray[np.complex128]]:
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
    gg_t: npt.NDArray[np.complex128] = np.empty((no, ne2), dtype=np.complex128)
    gl_t: npt.NDArray[np.complex128] = np.empty((no, ne2), dtype=np.complex128)
    gr_t: npt.NDArray[np.complex128] = np.empty((no, ne2), dtype=np.complex128)
    for i in numba.prange(no):
        gg_t[i, :] = fft.fft(gg[i, :], n=ne2)
        gl_t[i, :] = fft.fft(gl[i, :], n=ne2)
        gr_t[i, :] = fft.fft(gr[i, :], n=ne2)

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
            pg_t[i, j] = gg_t[i, j] * gl_t_mod[i, j]
            pl_t[i, j] = gl_t[i, j] * gg_t_mod[i, j]
            pr_t[i, j] = gr_t[i, j] * gl_t_mod[i, j] + gl_t[i, j] * np.conjugate(gr_t[i, j])

    # ifft, cutoff and multiply with pre factor
    pg: npt.NDArray[np.complex128] = np.empty_like(gg, dtype=np.complex128)
    pl: npt.NDArray[np.complex128] = np.empty_like(gg, dtype=np.complex128)
    pr: npt.NDArray[np.complex128] = np.empty_like(gg, dtype=np.complex128)

    for i in numba.prange(no):
        pg[i, :] = fft.ifft(pg_t[i, :])[:ne]
        pl[i, :] = fft.ifft(pl_t[i, :])[:ne]
        pr[i, :] = fft.ifft(pr_t[i, :])[:ne]
    for i in numba.prange(no):
        for j in numba.prange(ne):
            pg[i, j] = pg[i, j] * pre_factor
            pl[i, j] = pl[i, j] * pre_factor
            pr[i, j] = pr[i, j] * pre_factor

    return (pg, pl, pr)


@numba.njit("(c16, c16[:,:], c16[:,:], c16[:,:], c16[:,:])", parallel=True, cache=True, nogil=True, error_model="numpy")
def g2p_fft_mpi_cpu_inlined(
    pre_factor: np.complex128, gg: npt.NDArray[np.complex128], gl: npt.NDArray[np.complex128],
    gr: npt.NDArray[np.complex128], gl_transposed: npt.NDArray[np.complex128]
) -> typing.Tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128], npt.NDArray[np.complex128]]:
    """Calculates the polarization with fft on the cpu(see file description). 
        The Green's function and a the lesser transposed are needed.


    Args:
        pre_factor      (np.complex128): pre_factor, multiplied at the end
        gg (npt.NDArray[np.complex128]): Greater Green's Function,     (#orbital, #energy)
        gl (npt.NDArray[np.complex128]): Lesser Green's Function,      (#orbital, #energy)
        gr (npt.NDArray[np.complex128]): Retarded Green's Function,    (#orbital, #energy)
        gl_tranposed (npt.NDArray[np.complex128]): Transposed in orbital lesser Green's Function,    (#orbital, #energy)

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
    gg_t: npt.NDArray[np.complex128] = np.empty((no, ne2), dtype=np.complex128)
    gl_t: npt.NDArray[np.complex128] = np.empty((no, ne2), dtype=np.complex128)
    gr_t: npt.NDArray[np.complex128] = np.empty((no, ne2), dtype=np.complex128)
    gl_transposed_t: npt.NDArray[np.complex128] = np.empty((no, ne2), dtype=np.complex128)
    for i in numba.prange(no):
        gg_t[i, :] = fft.fft(gg[i, :], n=ne2)
        gl_t[i, :] = fft.fft(gl[i, :], n=ne2)
        gr_t[i, :] = fft.fft(gr[i, :], n=ne2)
        gl_transposed_t[i, :] = fft.fft(gl_transposed[i, :], n=ne2)

    # reverse and transpose
    gl_t_mod: npt.NDArray[np.complex128] = np.empty_like(gl_t, dtype=np.complex128)

    for i in numba.prange(no):
        for j in range(ne2):
            gl_t_mod[i, j] = gl_transposed_t[i, -j]

    # multiply elementwise
    pg_t: npt.NDArray[np.complex128] = np.empty_like(gl_t, dtype=np.complex128)
    pr_t: npt.NDArray[np.complex128] = np.empty_like(gl_t, dtype=np.complex128)

    for i in numba.prange(no):
        for j in numba.prange(ne2):
            pg_t[i, j] = gg_t[i, j] * gl_t_mod[i, j]
            pr_t[i, j] = gr_t[i, j] * gl_t_mod[i, j] + gl_t[i, j] * np.conjugate(gr_t[i, j])

    # ifft, cutoff and multiply with pre factor
    pg: npt.NDArray[np.complex128] = np.empty_like(gg_t, dtype=np.complex128)
    pr: npt.NDArray[np.complex128] = np.empty_like(gg_t, dtype=np.complex128)
    pl: npt.NDArray[np.complex128] = np.empty_like(gg_t, dtype=np.complex128)
    for i in numba.prange(no):
        pg[i, :] = fft.ifft(pg_t[i, :])
        pr[i, :] = fft.ifft(pr_t[i, :])
    for i in numba.prange(no):
        for j in numba.prange(ne2):
            pg[i, j] = pg[i, j] * pre_factor
            pr[i, j] = pr[i, j] * pre_factor

    # lesser polarization from identity
    for i in numba.prange(no):
        for j in range(ne2):
            pl[i, j] = -np.conjugate(pg[i, -j])

    return (pg[:, :ne], pl[:, :ne], pr[:, :ne])


@numba.njit("(c16, i4[:], c16[:,:], c16[:,:], c16[:,:])", parallel=True, cache=True, nogil=True, error_model="numpy")
def g2p_conv_cpu(
    pre_factor: np.complex128, ij2ji: npt.NDArray[np.int32], gg: npt.NDArray[np.complex128],
    gl: npt.NDArray[np.complex128], gr: npt.NDArray[np.complex128]
) -> typing.Tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128], npt.NDArray[np.complex128]]:
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
                tmpg += pre_factor * gg[ij, ep] * gl[ji, ep - e]
                tmpl += pre_factor * gl[ij, ep] * gg[ji, ep - e]
                tmpr += pre_factor * (gr[ij, ep] * gl[ji, ep - e] + gl[ij, ep] * np.conjugate(gr[ij, ep - e]))
            pg[ij, e] = tmpg
            pl[ij, e] = tmpl
            pr[ij, e] = tmpr

    return (pg, pl, pr)


def g2p_dense(gg: np.ndarray, gl: np.ndarray, gr: np.ndarray, energy: np.ndarray, workers=64):
    '''
    See readme, implementation of runsheng.

    Args
        gg, gl, gr : np.ndarray
            greater, lesser, and retarded green's function
            shape = (ne, n_orb, n_orb)
        energy : np.ndarray
            energy = np.linspace(E_min, E_max, ne) has to be evenly spaced!
            shape = (ne,)

    Returns
        pg, pl, pr : np.ndarray
            greater, lesser, and retarded polarization
            shape = (nep, n_orb, n_orb)
        ep : np.ndarray
            energy list for P and W
            should be symmetric
            E'~[-(ne-1)*denergy, ..., 0, ..., +(ne-1)*denergy], denergy=(E_max-E_min)/(ne-1)=E[1]-E[0]
            shape = (nep,)
    '''

    assert len(gr.shape) == 3
    assert len(energy.shape) == 1
    assert energy.shape[0] == gr.shape[0]
    assert gr.shape[1] == gr.shape[2]
    assert gr.shape == gl.shape
    assert gr.shape == gg.shape
    assert np.allclose(np.diff(energy), np.diff(energy)[0])  # evenly spaced

    ne = energy.shape[0]
    denergy = energy[1] - energy[0]

    nep = 2 * (ne - 1) + 1  # mode="full"
    ep = np.linspace(-(ne - 1) * denergy, +(ne - 1) * denergy, nep)

    # P^<>_ij(E') = -i*denergy/2pi \sum_{E} (G^<>_ij(E) G^><_ji(E-E'))
    pl = linalg_cpu.correlate_3D(gl, gg, mode="full", b_index="ji", method="fft", n_worker=workers)
    assert pl.shape[0] == nep

    pg = linalg_cpu.correlate_3D(gg, gl, mode="full", b_index="ji", method="fft", n_worker=workers)

    # P^r_ij(E') = -i*denergy/2pi \sum_{E} (G^<_ij(E) G^a_ji(E-E')+G^r_ij(E) G^<_ji(E-E'))
    # G^a=(G^r)^dagger
    ga = np.conjugate(np.einsum("ijk->ikj", gr, optimize="optimal"))
    pr = linalg_cpu.correlate_3D(gl, ga, mode="full", b_index="ji", method="fft", n_worker=workers)
    del ga
    pr += linalg_cpu.correlate_3D(gr, gl, mode="full", b_index="ji", method="fft", n_worker=workers)
    # times factor 2 to match gold solution, changed minus to plus
    pre_factor = -1.0j * denergy / (np.pi)
    pr = pre_factor * pr
    pl = pre_factor * pl
    pg = pre_factor * pg

    return pg, pl, pr, ep


@dace.program(auto_optimize=True)
def g2p_conv_dace(pre_factor: dace.complex128[1], ij2ji: dace.int32[NO], gg: dace.complex128[NO, NE],
                  gl: dace.complex128[NO, NE], gr: dace.complex128[NO, NE], pg: dace.complex128[NO, NE],
                  pl: dace.complex128[NO, NE], pr: dace.complex128[NO, NE]):
    """Todo finalize and test with new dace version

    Args:
        pre_factor (dace.complex128[1]): _description_
        ij2ji (dace.int32[NO]): _description_
        gg (dace.complex128[NO,NE]): _description_
        gl (dace.complex128[NO,NE]): _description_
        gr (dace.complex128[NO,NE]): _description_
        pg (dace.complex128[NO,NE]): _description_
        pl (dace.complex128[NO,NE]): _description_
        pr (dace.complex128[NO,NE]): _description_
    """
    for ij in range(NO):
        for e in range(NE):
            for ep in range(e, NE):
                ji = ij2ji[ij]
                pg[ij, e] += pre_factor[0] * gg[ij, ep] * gl[ji, ep - e]
                pl[ij, e] += pre_factor[0] * gl[ij, ep] * gg[ji, ep - e]
                pr[ij, e] += pre_factor[0] * (gr[ij, ep] * gl[ji, ep - e] + gl[ij, ep] * np.conjugate(gr[ij, ep - e]))


@dace.program(auto_optimize=True)
def g2p_fft_dace(pre_factor: dace.complex128[1], ij2ji: dace.int32[NO], gg: dace.complex128[NO, NE],
                 gl: dace.complex128[NO, NE], gr: dace.complex128[NO, NE], pg: dace.complex128[NO, NE],
                 pl: dace.complex128[NO, NE], pr: dace.complex128[NO, NE]):
    """Todo finalize, not working state

    Args:
        pre_factor (dace.complex128[1]): _description_fft(
        ij2ji (dace.int32[NO]): _description_
        gg (dace.complex128[NO,NE]): _description_
        gl (dace.complex128[NO,NE]): _description_
        gr (dace.complex128[NO,NE]): _description_
        pg (dace.complex128[NO,NE]): _description_
        pl (dace.complex128[NO,NE]): _description_
        pr (dace.complex128[NO,NE]): _description_
    """
    # fft
    gg_t = dace.ndarray((NO, 2 * NE), dtype=dace.complex128)
    gl_t = dace.ndarray((NO, 2 * NE), dtype=dace.complex128)
    gr_t = dace.ndarray((NO, 2 * NE), dtype=dace.complex128)
    for i in dace.map[0:NO]:
        gg_t[i, :] = fft.fft(gg[i, :], n=2 * NE)
        gl_t[i, :] = fft.fft(gl[i, :], n=2 * NE)
        gr_t[i, :] = fft.fft(gr[i, :], n=2 * NE)

    # reverse and transpose
    gl_t_mod: npt.NDArray[np.complex128] = dace.ndarray((NO, 2 * NE), dtype=dace.complex128)
    gg_t_mod: npt.NDArray[np.complex128] = dace.ndarray((NO, 2 * NE), dtype=dace.complex128)

    tmpl: npt.NDArray[np.complex128] = dace.ndarray((NO, 2 * NE), dtype=dace.complex128)
    tmpg: npt.NDArray[np.complex128] = dace.ndarray((NO, 2 * NE), dtype=dace.complex128)

    for i in dace.map[0:NO]:
        for j in dace.map[0:2 * NE]:
            tmpl[i, j] = gl_t[i, -j]
            tmpg[i, j] = gg_t[i, -j]
    for i in dace.map[0:NO]:
        gl_t_mod[ij2ji[i], :] = tmpl[i, :]
        gg_t_mod[ij2ji[i], :] = tmpg[i, :]

    # multiply elementwise
    pg_t: npt.NDArray[np.complex128] = dace.ndarray((NO, 2 * NE), dtype=dace.complex128)
    pl_t: npt.NDArray[np.complex128] = dace.ndarray((NO, 2 * NE), dtype=dace.complex128)
    pr_t: npt.NDArray[np.complex128] = dace.ndarray((NO, 2 * NE), dtype=dace.complex128)

    for i in dace.map[0:NO]:
        for j in dace.map[0:2 * NE]:
            pg_t[i, j] = gg_t[i, j] * gl_t_mod[i, j]
            pl_t[i, j] = gl_t[i, j] * gg_t_mod[i, j]
            pr_t[i, j] = gr_t[i, j] * gl_t_mod[i, j] + gl_t[i, j] * np.conjugate(gr_t[i, j])

    # ifft, cutoff and multiply with pre factor
    for i in dace.map[0:NO]:
        pg[i, :] = fft.ifft(pg_t[i, :])[:NE]
        pl[i, :] = fft.ifft(pl_t[i, :])[:NE]
        pr[i, :] = fft.ifft(pr_t[i, :])[:NE]
    for i in dace.map[0:NO]:
        for j in dace.map[0:NE]:
            pg[i, j] = pg[i, j] * pre_factor[0]
            pl[i, j] = pl[i, j] * pre_factor[0]
            pr[i, j] = pr[i, j] * pre_factor[0]


def g2p_fft_mpi_cpu_bare(
    pre_factor: np.complex128,
    gg: npt.NDArray[np.complex128],
    gl: npt.NDArray[np.complex128],
    gr: npt.NDArray[np.complex128],
    gl_transposed: npt.NDArray[np.complex128]
) -> typing.Tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128],
                  npt.NDArray[np.complex128]]:
    """Calculates the polarization with fft on the cpu(see file description). 
        The Green's function and a the lesser transposed are needed. This is a test function without numba compilation for debugging reasons.


    Args:
        pre_factor      (np.complex128): pre_factor, multiplied at the end
        gg (npt.NDArray[np.complex128]): Greater Green's Function,     (#orbital, #energy)
        gl (npt.NDArray[np.complex128]): Lesser Green's Function,      (#orbital, #energy)
        gr (npt.NDArray[np.complex128]): Retarded Green's Function,    (#orbital, #energy)
        gl_tranposed (npt.NDArray[np.complex128]): Transposed in orbital lesser Green's Function,    (#orbital, #energy)

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
    gl_transposed_t: npt.NDArray[np.complex128] = np.empty((no,ne2), dtype=np.complex128)
    for i in range(no):
        gg_t[i,:] = fft.fft(gg[i,:], n=ne2)
        gl_t[i,:] = fft.fft(gl[i,:], n=ne2)
        gr_t[i,:] = fft.fft(gr[i,:], n=ne2)
        gl_transposed_t[i,:] = fft.fft(gl_transposed[i,:], n=ne2)

    # reverse and transpose
    gl_t_mod: npt.NDArray[np.complex128] = np.empty_like(gl_t, dtype=np.complex128)

    for i in range(no):
        for j in range(ne2):
            gl_t_mod[i, j] = gl_transposed_t[i, -j]

    # multiply elementwise
    pg_t: npt.NDArray[np.complex128] = np.empty_like(gl_t, dtype=np.complex128)
    pr_t: npt.NDArray[np.complex128] = np.empty_like(gl_t, dtype=np.complex128)
    pr_t_new: npt.NDArray[np.complex128] = np.zeros_like(gl_t, dtype=np.complex128)
    pl_t: npt.NDArray[np.complex128] = np.empty_like(gl_t, dtype=np.complex128)

    for i in range(no):
        for j in range(ne2):
            pg_t[i,j] = gg_t[i,j] * gl_t_mod[i,j]
            pr_t[i,j] = gr_t[i,j] * gl_t_mod[i,j] + gl_t[i,j] * np.conjugate(gr_t[i,j])


    # ifft, cutoff and multiply with pre factor
    pg: npt.NDArray[np.complex128] = np.empty_like(gg_t, dtype=np.complex128)
    pr: npt.NDArray[np.complex128] = np.empty_like(gg_t, dtype=np.complex128)
    pl: npt.NDArray[np.complex128] = np.empty_like(gg_t, dtype=np.complex128)
    pr_new: npt.NDArray[np.complex128] = np.empty_like(gg_t, dtype=np.complex128)
    pr2: npt.NDArray[np.complex128] = np.empty_like(gg_t, dtype=np.complex128)
    for i in range(no):
        pg[i,:] = fft.ifft(pg_t[i,:])
        pr[i,:] = fft.ifft(pr_t[i,:])
    for i in range(no):
        for j in range(ne2):
            pg[i, j] = pg[i, j] * pre_factor
            pr[i, j] = pr[i, j] * pre_factor

    # lesser polarization from identity
    for i in range(no):
        for j in range(ne2):
            pl[i, j] = -np.conjugate(pg[i, -j])

    # computing pr_new with identity pr(t) = sigma(t) * (pg(t) - pl(t))
    for i in range(no):
        pl_t[i, :] = fft.ifft(pl[i,:] / pre_factor)

    for i in range(no):
        pr_t_new[i, :ne] = (pg_t[i,:] - pl_t[i,:])[:ne]

    for i in range(no):
        pr_new[i,:] = fft.ifft(pr_t_new[i,:])

    for i in range(no):
        for j in range(ne2):
            pr_new[i, j] = pr_new[i, j] * pre_factor
    
    # computing pr2 with identity pr(E) = 1j * np.imag(pg(E) - pl(E))/2
    for i in range(no):
        pr2[i,:] = 1j * np.imag(pg[i,:] - pl[i,:])/2
    

    return (pg[:, :ne], pl[:, :ne], pr[:, :ne])
