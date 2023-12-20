import concurrent.futures
from itertools import repeat

import time

import numpy as np
import numpy.typing as npt
from scipy import sparse
import mkl

from quatrex.utils.matrix_creation import initialize_block_G, mat_assembly_fullG, homogenize_matrix, \
                                            homogenize_matrix_Rnosym, extract_small_matrix_blocks

def self_energy_preprocess(SigL: npt.ArrayLike, SigG: npt.ArrayLike, SigR: npt.ArrayLike, SigL_ephn: npt.ArrayLike,
                            SigG_ephn: npt.ArrayLike, SigR_ephn: npt.ArrayLike,
                             NCpSC: int, bmin: npt.ArrayLike, bmax: npt.ArrayLike, homogenize: bool):
    #SigL[ie] = 1j * np.imag(SigL[ie])
    #SigG[ie] = 1j * np.imag(SigG[ie])

    SigL[:, :] = (SigL - SigL.T.conj()) / 2
    SigG[:, :] = (SigG - SigG.T.conj()) / 2
    #SigR[ie] = np.real(SigR[ie]) + 1j * np.imag(SigG[ie] - SigL[ie]) / 2
    SigR[:, :] = np.real(SigR) + (SigG - SigL) / 2
    #SigR[ie] = (SigR[ie] + SigR[ie].T) / 2

    SigL[:, :] += SigL_ephn
    SigG[:, :] += SigG_ephn
    SigR[:, :] += SigR_ephn

    if homogenize:
        (SigR00, SigR01, SigR10, _) = extract_small_matrix_blocks(SigR[bmin[0] - 1:bmax[0], bmin[0] - 1:bmax[0]],\
                                                                    SigR[bmin[0] - 1:bmax[0], bmin[1] - 1:bmax[1]], \
                                                                    SigR[bmin[1] - 1:bmax[1], bmin[0] - 1:bmax[0]], NCpSC, 'L')
        SigR[:, :] = homogenize_matrix_Rnosym(SigR00,
                                            SigR01, 
                                            SigR10, 
                                            len(bmax))
        (SigL00, SigL01, SigL10, _) = extract_small_matrix_blocks(SigL[bmin[0] - 1:bmax[0], bmin[0] - 1:bmax[0]],\
                                                                    SigL[bmin[0] - 1:bmax[0], bmin[1] - 1:bmax[1]], \
                                                                    SigL[bmin[1] - 1:bmax[1], bmin[0] - 1:bmax[0]], NCpSC, 'L')
        SigL[:, :] = homogenize_matrix_Rnosym(SigL00,
                                        SigL01,
                                        SigL10,
                                        len(bmax))
        (SigG00, SigG01, SigG10, _) = extract_small_matrix_blocks(SigG[bmin[0] - 1:bmax[0], bmin[0] - 1:bmax[0]],\
                                                                    SigG[bmin[0] - 1:bmax[0], bmin[1] - 1:bmax[1]], \
                                                                    SigG[bmin[1] - 1:bmax[1], bmin[0] - 1:bmax[0]], NCpSC, 'L')
        SigG[:, :] = homogenize_matrix_Rnosym(SigG00,
                                        SigG01, SigG10, len(bmax))
        
def self_energy_preprocess_2d(sl, sg, sr, sl_phn, sg_phn, sr_phn, rows, columns, ij2ji,  NCpSC, bmin, bmax, homogenize):
    sl_rgf = (sl - sl[:, ij2ji].conj()) / 2
    sg_rgf = (sg - sg[:, ij2ji].conj()) / 2
    sr_rgf = np.real(sr) + (sg - sl) / 2

    sl_rgf[:, rows == columns] += sl_phn
    sg_rgf[:, rows == columns] += sg_phn
    sr_rgf[:, rows == columns] += sr_phn

    #To_DO homogenize
    # This can be done using change_format.sparse2vecspase_v2 and after extracting the matrices do
    # change_format.sparse2block_no_map

    return sl_rgf, sg_rgf, sr_rgf
