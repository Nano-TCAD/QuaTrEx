# Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.
""" Contains functions for calculating the different correction terms for the calculation of the screened interaction.
    Naming is correction_ + from where the correction term is read from. . """

import numpy as np
import numpy.typing as npt
from scipy import sparse
import typing
import numba


def correction_system_matrix(
        MR: sparse.csr_array,
        Coulomb_matrix: sparse.csr_array,
        PR: sparse.csr_array,
        PL: sparse.csr_array,
        PG: sparse.csr_array,
        LL: sparse.csr_array,
        LG: sparse.csr_array,
        bmin: npt.NDArray[np.int32],
        bmax: npt.NDArray[np.int32],
        bmin_mm: npt.NDArray[np.int32],
        bmax_mm: npt.NDArray[np.int32]) -> sparse.csr_array:
    
    """ Calculates the correction term from the system matrix.
    This will only work for nbc > 1 at the moment. It assumes that the changes in the system matrix are only in the 
    new diagonal blocks."""

    # bigger block size
    lb_mm = bmax_mm[0] - bmin_mm[0] + 1

    lb_mm_1 = bmax_mm[0] - bmin_mm[0] + 1
    lb_mm_N = bmax_mm[-1] - bmin_mm[-1] + 1

    #slices from smaller block (left side)
    # vector of block lengths
    lb_vec = bmax - bmin + 1

    lb_1 = bmax[0] - bmin[0] + 1
    lb_N = bmax[-1] - bmin[-1] + 1

    CM00 = Coulomb_matrix[bmin[0]:bmax[0] + 1, bmin[0]:bmax[0] + 1].toarray()
    CM01 = Coulomb_matrix[bmin[0]:bmax[0] + 1, bmin[1]:bmax[1] + 1].toarray()
    CM10 = Coulomb_matrix[bmin[1]:bmax[1] + 1, bmin[0]:bmax[0] + 1].toarray()

    CMNN = Coulomb_matrix[bmin[-1]:bmax[-1] + 1, bmin[-1]:bmax[-1] + 1].toarray()
    CMNN_1 = Coulomb_matrix[bmin[-1]:bmax[-1] + 1, bmin[-2]:bmax[-2] + 1].toarray()
    CMN_1N = Coulomb_matrix[bmin[-2]:bmax[-2] + 1, bmin[-1]:bmax[-1] + 1].toarray()

    PR00 = PR[bmin[0]:bmax[0] + 1, bmin[0]:bmax[0] + 1].toarray()
    PR01 = PR[bmin[0]:bmax[0] + 1, bmin[1]:bmax[1] + 1].toarray()
    PR10 = PR[bmin[1]:bmax[1] + 1, bmin[0]:bmax[0] + 1].toarray()

    PRNN = PR[bmin[-1]:bmax[-1] + 1, bmin[-1]:bmax[-1] + 1].toarray()
    PRNN_1 = PR[bmin[-1]:bmax[-1] + 1, bmin[-2]:bmax[-2] + 1].toarray()
    PRN_1N = PR[bmin[-2]:bmax[-2] + 1, bmin[-1]:bmax[-1] + 1].toarray()
    
    PL00 = PL[bmin[0]:bmax[0] + 1, bmin[0]:bmax[0] + 1].toarray()
    PL01 = PL[bmin[0]:bmax[0] + 1, bmin[1]:bmax[1] + 1].toarray()
    PL10 = PL[bmin[1]:bmax[1] + 1, bmin[0]:bmax[0] + 1].toarray()

    PLNN = PL[bmin[-1]:bmax[-1] + 1, bmin[-1]:bmax[-1] + 1].toarray()
    PLNN_1 = PL[bmin[-1]:bmax[-1] + 1, bmin[-2]:bmax[-2] + 1].toarray()
    PLN_1N = PL[bmin[-2]:bmax[-2] + 1, bmin[-1]:bmax[-1] + 1].toarray()

    PG00 = PG[bmin[0]:bmax[0] + 1, bmin[0]:bmax[0] + 1].toarray()
    PG01 = PG[bmin[0]:bmax[0] + 1, bmin[1]:bmax[1] + 1].toarray()
    PG10 = PG[bmin[1]:bmax[1] + 1, bmin[0]:bmax[0] + 1].toarray()

    PGNN = PG[bmin[-1]:bmax[-1] + 1, bmin[-1]:bmax[-1] + 1].toarray()
    PGNN_1 = PG[bmin[-1]:bmax[-1] + 1, bmin[-2]:bmax[-2] + 1].toarray()
    PGN_1N = PG[bmin[-2]:bmax[-2] + 1, bmin[-1]:bmax[-1] + 1].toarray()

    M1 = -CM10 @ PR01
    MN = -CMN_1N @ PRNN_1

    C1 = CM10 @ PG00 \
          @ CM10.conj().T
    C2 = CM00 @ PG10 \
          @ CM10.conj().T
    C3 = CM10 @ PG01 \
            @ CM00.conj().T
    
    LG1_d = C1 + C2 + C3
    LG1_l = CM10 @ PG10 @ CM10.conj().T
    LG1_u = CM10 @ PG01 @ CM10.conj().T

    C1 = CMN_1N @ PGNN_1 @ CMNN.conj().T
    C2 = CMNN @ PGN_1N @ CMN_1N.conj().T
    C3 = CMN_1N @ PGNN @ CMN_1N.conj().T

    LGN_d = C1 + C2 + C3
    LGN_u = CMN_1N @ PGNN_1 @ CMN_1N.conj().T
    LGN_l = CMN_1N @ PGN_1N @ CMN_1N.conj().T

    C1 = CM10 @ PL00 \
          @ CM10.conj().T
    C2 = CM00 @ PL10 \
          @ CM10.conj().T
    C3 = CM10 @ PL01 \
            @ CM00.conj().T
    
    LL1_d = C1 + C2 + C3
    LL1_l = CM10 @ PL10 @ CM10.conj().T
    LL1_u = CM10 @ PL01 @ CM10.conj().T

    C1 = CMN_1N @ PLNN_1 @ CMNN.conj().T
    C2 = CMNN @ PLN_1N @ CMN_1N.conj().T
    C3 = CMN_1N @ PLNN @ CMN_1N.conj().T

    LLN_d = C1 + C2 + C3
    LLN_u = CMN_1N @ PLNN_1 @ CMN_1N.conj().T
    LLN_l = CMN_1N @ PLN_1N @ CMN_1N.conj().T

    # output matrices
    mr_d1 = np.zeros((lb_mm, lb_mm), dtype=np.complex128)
    mr_u1 = np.zeros((lb_mm, lb_mm), dtype=np.complex128)
    mr_l1 = np.zeros((lb_mm, lb_mm), dtype=np.complex128)

    mr_d2 = np.zeros((lb_mm, lb_mm), dtype=np.complex128)
    mr_u2 = np.zeros((lb_mm, lb_mm), dtype=np.complex128)
    mr_l2 = np.zeros((lb_mm, lb_mm), dtype=np.complex128)

    lg_d1 = np.zeros((lb_mm, lb_mm), dtype=np.complex128)
    lg_u1 = np.zeros((lb_mm, lb_mm), dtype=np.complex128)
    lg_l1 = np.zeros((lb_mm, lb_mm), dtype=np.complex128)

    lg_d2 = np.zeros((lb_mm, lb_mm), dtype=np.complex128)
    lg_u2 = np.zeros((lb_mm, lb_mm), dtype=np.complex128)
    lg_l2 = np.zeros((lb_mm, lb_mm), dtype=np.complex128)

    ll_d1 = np.zeros((lb_mm, lb_mm), dtype=np.complex128)
    ll_u1 = np.zeros((lb_mm, lb_mm), dtype=np.complex128)
    ll_l1 = np.zeros((lb_mm, lb_mm), dtype=np.complex128)

    ll_d2 = np.zeros((lb_mm, lb_mm), dtype=np.complex128)
    ll_u2 = np.zeros((lb_mm, lb_mm), dtype=np.complex128)
    ll_l2 = np.zeros((lb_mm, lb_mm), dtype=np.complex128)

    dmr_lu1 = np.zeros((lb_mm, lb_mm), dtype=np.complex128)
    dmr_ul1 = np.zeros((lb_mm, lb_mm), dtype=np.complex128)

    dmr_lu2 = np.zeros((lb_mm, lb_mm), dtype=np.complex128)
    dmr_ul2 = np.zeros((lb_mm, lb_mm), dtype=np.complex128)

    dlg_lu1 = np.zeros((lb_mm, lb_mm), dtype=np.complex128)
    dlg_ul1 = np.zeros((lb_mm, lb_mm), dtype=np.complex128)

    dlg_lu2 = np.zeros((lb_mm, lb_mm), dtype=np.complex128)
    dlg_ul2 = np.zeros((lb_mm, lb_mm), dtype=np.complex128)
    
    dll_lu1 = np.zeros((lb_mm, lb_mm), dtype=np.complex128)
    dll_ul1 = np.zeros((lb_mm, lb_mm), dtype=np.complex128)

    dll_lu2 = np.zeros((lb_mm, lb_mm), dtype=np.complex128)
    dll_ul2 = np.zeros((lb_mm, lb_mm), dtype=np.complex128)


    vh_u1 = np.zeros((lb_mm, lb_mm), dtype=np.complex128)
    vh_l1 = np.zeros((lb_mm, lb_mm), dtype=np.complex128)

    vh_u2 = np.zeros((lb_mm, lb_mm), dtype=np.complex128)
    vh_l2 = np.zeros((lb_mm, lb_mm), dtype=np.complex128)

    #fill output matrices
    # Output 1/7
    mr_d1[0:lb_1, 0:lb_1] = M1
    mr_d1 += MR[0:lb_mm_1, 0:lb_mm_1].toarray()
    mr_u1 = MR[0:lb_mm_1, lb_mm_1:2*lb_mm_1].toarray()
    mr_l1 = MR[lb_mm_1:2*lb_mm_1, 0:lb_mm_1].toarray()

    mr_d2[-lb_N:, -lb_N:] = MN
    mr_d2 += MR[-lb_mm_N:, -lb_mm_N:].toarray()
    mr_u2 = MR[-lb_mm_N:, -2*lb_mm_N:-lb_mm_N].toarray()
    mr_l2 = MR[-2*lb_mm_N:-lb_mm_N, -lb_mm_N:].toarray()
    
    # Output 2/7
    lg_d1[0:lb_1, 0:lb_1] = LG1_d
    lg_d1[0:lb_1, lb_1:2*lb_1] = LG1_u
    lg_d1[lb_1:2*lb_1, 0:lb_1] = LG1_l
    lg_d1 += LG[0:lb_mm_1, 0:lb_mm_1].toarray()
    lg_u1 = LG[0:lb_mm_1, lb_mm_1:2*lb_mm_1].toarray()
    lg_l1 = LG[lb_mm_1:2*lb_mm_1, 0:lb_mm_1].toarray()

    lg_d2[-lb_N:, -lb_N:] = LGN_d
    lg_d2[-lb_N:, -2*lb_N:-lb_N] = LGN_u
    lg_d2[-2*lb_N:-lb_N, -lb_N:] = LGN_l
    lg_d2 += LG[-lb_mm_N:, -lb_mm_N:].toarray()
    lg_u2 = LG[-lb_mm_N:, -2*lb_mm_N:-lb_mm_N].toarray()
    lg_l2 = LG[-2*lb_mm_N:-lb_mm_N, -lb_mm_N:].toarray()

    # Output 3/7
    ll_d1[0:lb_1, 0:lb_1] = LL1_d
    ll_d1[0:lb_1, lb_1:2*lb_1] = LL1_u
    ll_d1[lb_1:2*lb_1, 0:lb_1] = LL1_l
    ll_d1 += LL[0:lb_mm_1, 0:lb_mm_1].toarray()
    ll_u1 = LL[0:lb_mm_1, lb_mm_1:2*lb_mm_1].toarray()
    ll_l1 = LL[lb_mm_1:2*lb_mm_1, 0:lb_mm_1].toarray()

    ll_d2[-lb_N:, -lb_N:] = LLN_d
    ll_d2[-lb_N:, -2*lb_N:-lb_N] = LLN_u
    ll_d2[-2*lb_N:-lb_N, -lb_N:] = LLN_l
    ll_d2 += LL[-lb_mm_N:, -lb_mm_N:].toarray()
    ll_u2 = LL[-lb_mm_N:, -2*lb_mm_N:-lb_mm_N].toarray()
    ll_l2 = LL[-2*lb_mm_N:-lb_mm_N, -lb_mm_N:].toarray()

    # Output 4/7
    dmr_lu1[0:lb_1, 0:lb_1] = M1
    dmr_ul2[-lb_N:, -lb_N:] = MN

    # Output 5/7
    dlg_lu1[0:lb_1, 0:lb_1] = LG1_d
    dlg_lu1[0:lb_1, lb_1:2*lb_1] = LG1_u
    dlg_lu1[lb_1:2*lb_1, 0:lb_1] = LG1_l

    dlg_ul2[-lb_N:, -lb_N:] = LGN_d
    dlg_ul2[-lb_N:, -2*lb_N:-lb_N] = LGN_u
    dlg_ul2[-2*lb_N:-lb_N, -lb_N:] = LGN_l

    # Output 6/7
    dll_lu1[0:lb_1, 0:lb_1] = LL1_d
    dll_lu1[0:lb_1, lb_1:2*lb_1] = LL1_u
    dll_lu1[lb_1:2*lb_1, 0:lb_1] = LL1_l

    dll_ul2[-lb_N:, -lb_N:] = LLN_d
    dll_ul2[-lb_N:, -2*lb_N:-lb_N] = LLN_u
    dll_ul2[-2*lb_N:-lb_N, -lb_N:] = LLN_l

    # Output 7/7
    
    vh_u1[-lb_1:, 0:lb_1] = CM01
    vh_l1[0:lb_1, -lb_1:] = CM10

    vh_u2[-lb_N:, 0:lb_N] = CMN_1N
    vh_l2[0:lb_N, -lb_N:] = CMNN_1



    return ((mr_d1, mr_u1, mr_l1), (mr_d2, mr_u2, mr_l2), \
            (lg_d1, lg_u1, lg_l1), (lg_d2, lg_u2, lg_l2), \
            (ll_d1, ll_u1, ll_l1), (ll_d2, ll_u2, ll_l2), \
            (dmr_lu1, dmr_ul1),  (dmr_lu2, dmr_ul2),\
            (dlg_lu1, dlg_ul1), (dlg_lu2, dlg_ul2), \
            (dll_lu1, dll_ul1), (dll_lu2, dll_ul2),\
            (vh_u1, vh_l1), (vh_u2, vh_l2))

