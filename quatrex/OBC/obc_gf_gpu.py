# Copyright 2024 ETH Zurich and the QuaTrEx authors. All rights reserved.
"""
Containing functions to apply the OBC for the Green's Function


- _ct is for the original matrix complex conjugated
- _mm how certain sizes after matrix multiplication, because a block tri diagonal gets two new non zero off diagonals
- _s stands for values related to the left/start/top contact block
- _e stands for values related to the right/end/bottom contact block
- _d stands for diagonal block (X00/NN in matlab)
- _u stands for upper diagonal block (X01/NN1 in matlab)
- _l stands for lower diagonal block (X10/N1N in matlab) 
- exception _l/_r can stand for left/right in context of condition of OBC
"""

import cupy as cp
import cupyx as cpx

import numpy as np
import numpy.typing as npt
from scipy import sparse
from quatrex.OBC.beyn_new_gpu import beyn_gpu
from quatrex.OBC.sancho import open_boundary_conditions
import typing
from functools import partial

def obc_GF_gpu(M,
           SigR,
           fL,
           fR,
           SigRBL, SigRBR, SigLBL, SigLBR, SigGBL, SigGBR,
           Bmin_fi,
           Bmax_fi,
           NCpSC=1,
           use_dace=False,
           validate_dace=False,
           sancho=False):

    beyn_func = beyn_gpu
    if use_dace:
        from quatrex.OBC import beyn_dace
        beyn_func = partial(beyn_dace.beyn, validate=validate_dace)

    imag_lim = 5e-4
    R = 1000

    min_dEkL = 1e8
    min_dEkR = 1e8
    Bmax = Bmax_fi - 1
    Bmin = Bmin_fi - 1
    Bsize = max(Bmax - Bmin + 1)  # Used for declaration of variables
    NB = len(Bmin)
    NT = Bmax[NB - 1] + 1  # Not used in this fcn

    LBsize = Bmax[0] - Bmin[0] + 1
    RBsize = Bmax[NB - 1] - Bmin[NB - 1] + 1

    condL = 0.0
    condR = 0.0

    SigRBL_gpu = cp.empty_like(SigRBL)
    SigRBR_gpu = cp.empty_like(SigRBR)

    #GR/GL/GG OBC Left
    #_, SigRBL, _, condL = open_boundary_conditions(M[:LBsize, :LBsize].toarray(), M[LBsize:2*LBsize, :LBsize].toarray(),
    #                                                    M[:LBsize, LBsize:2*LBsize].toarray(), np.eye(LBsize, LBsize))
    if not sancho:
        SigRBL_gpu, _, condL, min_dEkL = beyn_func(NCpSC, cp.asarray(M[:LBsize, :LBsize].toarray()-SigR[:LBsize, :LBsize].toarray()),
                                                  cp.asarray(M[:LBsize, LBsize:2 * LBsize].toarray() - SigR[:LBsize, LBsize:2 * LBsize].toarray()),
                                                  cp.asarray(M[LBsize:2 * LBsize, :LBsize].toarray() - SigR[LBsize:2 * LBsize, :LBsize].toarray()),
                                                  imag_lim,
                                                  R,
                                                  'L')

    if np.isnan(condL) or sancho:
        _, SigRBL_gpu[:, :], _, condL = open_boundary_conditions(M[:LBsize, :LBsize].toarray() - SigR[:LBsize, :LBsize].toarray(),
                                                       M[LBsize:2 * LBsize, :LBsize].toarray() - SigR[LBsize:2 * LBsize, :LBsize].toarray(),
                                                       M[:LBsize, LBsize:2 * LBsize].toarray() - SigR[:LBsize, LBsize:2 * LBsize].toarray(), np.eye(LBsize, LBsize))

    #condL = np.nan
    if not np.isnan(condL):
        GammaL = 1j * (SigRBL_gpu - SigRBL_gpu.conj().T)
        SigLBL[:, :] = (1j * fL * GammaL).get()
        SigGBL[:, :] = (1j * (fL - 1) * GammaL).get()
        SigRBL[:, :] = SigRBL_gpu.get()

    #GR/GL/GG OBC right
    if not sancho:
        SigRBR_gpu, _, condR, min_dEkR = beyn_func(NCpSC, cp.asarray(M[NT - RBsize:NT, NT - RBsize:NT].toarray() - SigR[NT - RBsize:NT, NT - RBsize:NT].toarray()),
                                                  cp.asarray(M[NT - 2 * RBsize:NT - RBsize, NT - RBsize:NT].toarray() - SigR[NT - 2 * RBsize:NT - RBsize, NT - RBsize:NT].toarray()),
                                                  cp.asarray(M[NT - RBsize:NT, NT - 2 * RBsize:NT - RBsize].toarray() - SigR[NT - RBsize:NT, NT - 2 * RBsize:NT - RBsize].toarray()),
                                                  imag_lim,
                                                  R,
                                                  'R')
                                                  
    if np.isnan(condR) or sancho:
        _, SigRBR_gpu[:, :], _, condR = open_boundary_conditions(M[NT - RBsize:NT, NT - RBsize:NT].toarray() - SigR[NT - RBsize:NT, NT - RBsize:NT].toarray(),
                                                       M[NT - 2 * RBsize:NT - RBsize, NT - RBsize:NT].toarray() - SigR[NT - 2 * RBsize:NT - RBsize, NT - RBsize:NT].toarray(),
                                                       M[NT - RBsize:NT, NT - 2 * RBsize:NT - RBsize].toarray() - SigR[NT - RBsize:NT, NT - 2 * RBsize:NT - RBsize].toarray(),
                                                       np.eye(RBsize, RBsize))
    #condR = np.nan
    if not np.isnan(condR):
        GammaR = 1j * (SigRBR_gpu - SigRBR_gpu.conj().T)
        SigLBR[:, :] = (1j * fR * GammaR).get()
        SigGBR[:, :] = (1j * (fR - 1) * GammaR).get()
        SigRBR[:, :] = SigRBR_gpu.get()
    return condL, condR