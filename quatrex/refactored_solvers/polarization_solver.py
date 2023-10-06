# Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.


import numpy as np
from numpy import fft


def compute_polarization(
    G_retarded,
    G_lesser,
    G_greater,
    delta_energy,
):  
    pass

    """ Polarisation : np.ndarray

    scaling_factor = -1.0j * delta_energy / (np.pi)

    numer_of_energies = G_greater.shape[1]
    number_of_orbitals = G_greater.shape[0]


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
    
            
    pg = scaling_factor * pg 
    pl = scaling_factor * pl
    pr = scaling_factor * pr

    return (pg, pl, pr) """



