import concurrent.futures
from itertools import repeat

import time

import numpy as np
import numpy.typing as npt
from scipy import sparse
class dummy:
    def __init__(self):
        pass
    def set_num_threads(self, n):
        pass

try:
    import mkl
except (ImportError, ModuleNotFoundError):
    mkl = dummy()

from quatrex.utils.matrix_creation import initialize_block_G, mat_assembly_fullG, homogenize_matrix, \
                                            homogenize_matrix_Rnosym, extract_small_matrix_blocks

def polarization_preprocess(PL: npt.ArrayLike, PG: npt.ArrayLike, PR: npt.ArrayLike,
                             NCpSC: int, bmin: npt.ArrayLike, bmax: npt.ArrayLike, homogenize: bool):
    #PL[ie] = 1j * np.imag(PL[ie])
    #PG[ie] = 1j * np.imag(PG[ie])

    PL[:, :] = (PL - PL.T.conj()) / 2
    PG[:, :] = (PG - PG.T.conj()) / 2
    #PR[ie] = np.real(PR[ie]) + 1j * np.imag(PG[ie] - PL[ie]) / 2
    PR[:, :] =  + (PG - PL) / 2
    #PR[ie] = (PR[ie] + PR[ie].T) / 2



    if homogenize:
        (PR00, PR01, PR10, _) = extract_small_matrix_blocks(PR[bmin[0]:bmax[0] + 1, bmin[0]:bmax[0] + 1],\
                                                                    PR[bmin[0]:bmax[0] + 1, bmin[1] - 1:bmax[1]], \
                                                                    PR[bmin[1] - 1:bmax[1], bmin[0]:bmax[0] + 1], NCpSC, 'L')
        PR[:, :] = homogenize_matrix_Rnosym(PR00,
                                            PR01, 
                                            PR10, 
                                            len(bmax))
        (PL00, PL01, PL10, _) = extract_small_matrix_blocks(PL[bmin[0]:bmax[0] + 1, bmin[0]:bmax[0] + 1],\
                                                                    PL[bmin[0]:bmax[0] + 1, bmin[1] - 1:bmax[1]], \
                                                                    PL[bmin[1] - 1:bmax[1], bmin[0]:bmax[0] + 1], NCpSC, 'L')
        PL[:, :] = homogenize_matrix_Rnosym(PL00,
                                        PL01,
                                        PL10,
                                        len(bmax))
        (PG00, PG01, PG10, _) = extract_small_matrix_blocks(PG[bmin[0]:bmax[0] + 1, bmin[0]:bmax[0] + 1],\
                                                                    PG[bmin[0]:bmax[0] + 1, bmin[1] - 1:bmax[1]], \
                                                                    PG[bmin[1] - 1:bmax[1], bmin[0]:bmax[0] + 1], NCpSC, 'L')
        PG[:, :] = homogenize_matrix_Rnosym(PG00,
                                        PG01, PG10, len(bmax))
        
def polarization_preprocess_2d(pl, pg, pr, rows, columns, ij2ji,  NCpSC, bmin, bmax, homogenize):
    pl_rgf = (pl - pl[:, ij2ji].conj()) / 2
    pg_rgf = (pg - pg[:, ij2ji].conj()) / 2
    pr_rgf = (pg - pl) / 2

    #To_DO homogenize
    # This can be done using change_format.sparse2vecspase_v2 and after extracting the matrices do
    # change_format.sparse2block_no_map

    return pl_rgf, pg_rgf, pr_rgf
