""" Script for calculating the Electron-Phonon self-energy. Based on Mathieu's calc_SE_GF_EPHN function
from the matlab GW code."""
import cupy as cp
import numpy as np


def calc_SE_GF_EPHN(energy, gl_diag, gg_diag, sg_phn_old, sl_phn_old, sr_phn_old, EPHN, DPHN, temp, memory_factor):

    kB = 1.38e-23
    q = 1.6022e-19

    dE = energy[1]-energy[0]

    UT = kB*temp/q

    SigmaL = cp.zeros(sl_phn_old.shape, dtype=np.complex128)
    SigmaG = cp.zeros(sg_phn_old.shape, dtype=np.complex128)

    for IPH in range(len(EPHN)):
        NPH = 1 / (np.exp(max(EPHN[IPH], 5e-3) / UT) - 1)
        SigmaL += 1j * cp.imag(calc_Sigma_el_phon(gl_diag, NPH, NPH+1, EPHN[IPH], DPHN[IPH], dE))
        SigmaG += 1j * cp.imag(calc_Sigma_el_phon(gg_diag, NPH+1, NPH, EPHN[IPH], DPHN[IPH], dE))

    SigmaR = (SigmaG - SigmaL) / 2.0

    sl_phn_old[:] = (1 - memory_factor) * SigmaL + memory_factor * sl_phn_old
    sg_phn_old[:] = (1 - memory_factor) * SigmaG + memory_factor * sg_phn_old
    sr_phn_old[:] = (1 - memory_factor) * SigmaR + memory_factor * sr_phn_old

    # return SigmaG, SigmaL, SigmaR


def calc_Sigma_el_phon(g, fup, fdown, homega_ph, D, dE):

    ne, nao = g.shape

    Sigma = cp.zeros((ne, nao), dtype=np.complex128)

    neph = round(homega_ph/dE)

    Sigma[neph:ne, :] = D * fup * g[:ne-neph, :]
    Sigma[:ne-neph, :] = Sigma[:ne-neph, :] + D * fdown * g[neph:ne, :]

    return Sigma
