# Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.
""" Contains functions for calculating the different correction terms for the calculation of the screened interaction.
    Naming is correction_ + from where the correction term is read from. Has the same function as get_OBC_blocks.m,
    but split up for readability. """

import cupy as cp
import numpy as np
import typing

from quatrex.OBC.beyn_new_gpu import extract_small_matrix_blocks_gpu
from quatrex.OBC.beyn_batched import extract_small_matrix_blocks_batched_gpu

def get_mm_obc_dense(
    vh_1: cp.ndarray, vh_2: cp.ndarray, pg_1: cp.ndarray, pg_2: cp.ndarray, pl_1: cp.ndarray, pl_2: cp.ndarray,
    pr_1: cp.ndarray, pr_2: cp.ndarray, pr_3: cp.ndarray, nbc: int, NCpSC: int, side: str = "L"
) -> typing.Tuple[typing.Tuple[cp.ndarray, cp.ndarray, cp.ndarray], typing.Tuple[cp.ndarray, cp.ndarray, cp.ndarray],
                  typing.Tuple[cp.ndarray, cp.ndarray, cp.ndarray], typing.Tuple[cp.ndarray, cp.ndarray], typing.Tuple[
                      cp.ndarray, cp.ndarray], typing.Tuple[cp.ndarray, cp.ndarray], typing.Tuple[cp.ndarray,
                                                                                                  cp.ndarray]]:
    """
    Calculates the different correction terms for the scattering OBCs for the screened interaction calculations.
    In this version the blocks are not stacked, but the matrices are multiplied directly and inserted afterwards.

    Args:
        vh_1 (cp.ndarray): Diagonal block of effective interaction
        vh_2 (cp.ndarray): Off diagonal block of effective interaction
        pg_1 (cp.ndarray): Diagonal block of greater polarization
        pg_2 (cp.ndarray): Off diagonal block of greater polarization
        pl_1 (cp.ndarray): Diagonal block of lesser polarization
        pl_2 (cp.ndarray): Off diagonal block of lesser polarization
        pr_1 (cp.ndarray): Diagonal block of retarded polarization
        pr_2 (cp.ndarray): Off diagonal block of retarded polarization
        nbc (int): How block size changes after matrix multiplication

    Returns:
        typing.Tuple[
            typing.Tuple[cp.ndarray, cp.ndarray, cp.ndarray],
            typing.Tuple[cp.ndarray, cp.ndarray, cp.ndarray],
            typing.Tuple[cp.ndarray, cp.ndarray, cp.ndarray],
            typing.Tuple[cp.ndarray, cp.ndarray],
            typing.Tuple[cp.ndarray, cp.ndarray],
            typing.Tuple[cp.ndarray, cp.ndarray],
            typing.Tuple[cp.ndarray, cp.ndarray]
            ]: mr_d/u/l, lg_d/u/l, ll_d/u/l, dg_lu/ul, dl_lu/ul, vh_u/l
    """
    # block size
    lb = vh_1.shape[0]
    # block size after mm
    lb_mm = nbc * lb
    # define the right blocks
    (vh_d1, vh_u1, vh_l1, _) = extract_small_matrix_blocks_gpu(vh_1, vh_2,
                                                            vh_2.conjugate().transpose(),
                                                            NCpSC, side, densify = True)  
    # else:
    #     vh_d1 = vh_1
    #     vh_u1 = vh_2
    #     vh_l1 = vh_u1.conjugate().transpose()
    # pg_d1 = pg_1
    # pg_u1 = pg_2
    # pg_l1 = -pg_u1.conjugate().transpose()
    (pg_d1, pg_u1, pg_l1, _) = extract_small_matrix_blocks_gpu(pg_1, pg_2,
                                                            -pg_2.conjugate().transpose(),
                                                            NCpSC, side, densify = True)


    # if side == "L":
    (pl_d1, pl_u1, pl_l1, _) = extract_small_matrix_blocks_gpu(pl_1, pl_2,
                                                            -pl_2.conjugate().transpose(),
                                                            NCpSC, side, densify = True)
    # else: 
    #     pl_d1 = pl_1
    #     pl_u1 = pl_2
    #     pl_l1 = -pl_u1.conjugate().transpose()

    # if side == "L":
    (pr_d1, pr_u1, pr_l1, _) = extract_small_matrix_blocks_gpu(pr_1, pr_2,
                                                            pr_3,
                                                            NCpSC, side, densify = True)
    # else:

    # output matrices
    mr_d2 = cp.empty((lb_mm, lb_mm), dtype=cp.complex128)
    mr_u2 = cp.zeros((lb_mm, lb_mm), dtype=cp.complex128)
    mr_l2 = cp.zeros((lb_mm, lb_mm), dtype=cp.complex128)
    lg_d2 = cp.empty((lb_mm, lb_mm), dtype=cp.complex128)
    lg_u2 = cp.zeros((lb_mm, lb_mm), dtype=cp.complex128)
    lg_l2 = cp.empty((lb_mm, lb_mm), dtype=cp.complex128)
    ll_d2 = cp.empty((lb_mm, lb_mm), dtype=cp.complex128)
    ll_u2 = cp.zeros((lb_mm, lb_mm), dtype=cp.complex128)
    ll_l2 = cp.empty((lb_mm, lb_mm), dtype=cp.complex128)
    dmr_lu = cp.zeros((lb_mm, lb_mm), dtype=cp.complex128)
    dmr_ul = cp.zeros((lb_mm, lb_mm), dtype=cp.complex128)
    dlg_lu = cp.zeros((lb_mm, lb_mm), dtype=cp.complex128)
    dlg_ul = cp.zeros((lb_mm, lb_mm), dtype=cp.complex128)
    dll_lu = cp.zeros((lb_mm, lb_mm), dtype=cp.complex128)
    dll_ul = cp.zeros((lb_mm, lb_mm), dtype=cp.complex128)
    vh_u = cp.zeros((lb_mm, lb_mm), dtype=cp.complex128)
    vh_l = cp.zeros((lb_mm, lb_mm), dtype=cp.complex128)
    if nbc == 1:
        # fill output matrices
        # M^{r}\left(E\right)
        mr_d2[:, :] = -vh_d1 @ pr_d1 - vh_u1 @ pr_l1 - vh_l1 @ pr_u1
        mr_u2[:, :] = -vh_d1 @ pr_u1 - vh_u1 @ pr_d1
        mr_l2[:, :] = -vh_d1 @ pr_l1 - vh_l1 @ pr_d1

        # from L^{\lessgtr}\left(E\right)
        lg_d2[:, :] = (vh_l1 @ pg_d1 @ vh_u1 + vh_l1 @ pg_u1 @ vh_d1 + vh_d1 @ pg_l1 @ vh_u1 + vh_d1 @ pg_d1 @ vh_d1 +
                       vh_d1 @ pg_u1 @ vh_l1 + vh_u1 @ pg_l1 @ vh_d1 + vh_u1 @ pg_d1 @ vh_l1)
        lg_u2[:, :] = (vh_l1 @ pg_u1 @ vh_u1 + vh_d1 @ pg_d1 @ vh_u1 + vh_d1 @ pg_u1 @ vh_d1 + vh_u1 @ pg_l1 @ vh_u1 +
                       vh_u1 @ pg_d1 @ vh_d1)
        ll_d2[:, :] = (vh_l1 @ pl_d1 @ vh_u1 + vh_l1 @ pl_u1 @ vh_d1 + vh_d1 @ pl_l1 @ vh_u1 + vh_d1 @ pl_d1 @ vh_d1 +
                       vh_d1 @ pl_u1 @ vh_l1 + vh_u1 @ pl_l1 @ vh_d1 + vh_u1 @ pl_d1 @ vh_l1)
        ll_u2[:, :] = (vh_l1 @ pl_u1 @ vh_u1 + vh_d1 @ pl_d1 @ vh_u1 + vh_d1 @ pl_u1 @ vh_d1 + vh_u1 @ pl_l1 @ vh_u1 +
                       vh_u1 @ pl_d1 @ vh_d1)
        dmr_lu[:, :] = -vh_l1 @ pr_u1
        dmr_ul[:, :] = -vh_u1 @ pr_l1
        dlg_lu[:, :] = vh_l1 @ pg_d1 @ vh_u1 + vh_l1 @ pg_u1 @ vh_d1 + vh_d1 @ pg_l1 @ vh_u1
        dll_lu[:, :] = vh_l1 @ pl_d1 @ vh_u1 + vh_l1 @ pl_u1 @ vh_d1 + vh_d1 @ pl_l1 @ vh_u1
        dlg_ul[:, :] = vh_u1 @ pg_d1 @ vh_l1 + vh_u1 @ pg_l1 @ vh_d1 + vh_d1 @ pg_u1 @ vh_l1
        dll_ul[:, :] = vh_u1 @ pl_d1 @ vh_l1 + vh_u1 @ pl_l1 @ vh_d1 + vh_d1 @ pl_u1 @ vh_l1
        vh_u[:, :] = vh_u1
        vh_l[:, :] = vh_l1
    elif nbc == 2:
        # compute multiplications
        vhpr_d1d1 = -vh_d1 @ pr_d1
        vhpr_d1u1 = -vh_d1 @ pr_u1
        vhpr_d1l1 = -vh_d1 @ pr_l1
        vhpr_u1d1 = -vh_u1 @ pr_d1
        vhpr_u1u1 = -vh_u1 @ pr_u1
        vhpr_u1l1 = -vh_u1 @ pr_l1
        vhpr_l1d1 = -vh_l1 @ pr_d1
        vhpr_l1u1 = -vh_l1 @ pr_u1
        vhpr_l1l1 = -vh_l1 @ pr_l1

        vhpg_d1d1 = vh_d1 @ pg_d1
        vhpg_d1u1 = vh_d1 @ pg_u1
        vhpg_d1l1 = vh_d1 @ pg_l1
        vhpg_u1d1 = vh_u1 @ pg_d1
        vhpg_u1u1 = vh_u1 @ pg_u1
        vhpg_u1l1 = vh_u1 @ pg_l1
        vhpg_l1d1 = vh_l1 @ pg_d1
        vhpg_l1u1 = vh_l1 @ pg_u1
        vhpg_l1l1 = vh_l1 @ pg_l1
        vhpgvh_d1d1d1 = vhpg_d1d1 @ vh_d1
        vhpgvh_d1d1u1 = vhpg_d1d1 @ vh_u1
        vhpgvh_d1d1l1 = vhpg_d1d1 @ vh_l1
        vhpgvh_d1u1d1 = vhpg_d1u1 @ vh_d1
        vhpgvh_d1u1u1 = vhpg_d1u1 @ vh_u1
        vhpgvh_d1u1l1 = vhpg_d1u1 @ vh_l1
        vhpgvh_d1l1d1 = vhpg_d1l1 @ vh_d1
        vhpgvh_d1l1u1 = vhpg_d1l1 @ vh_u1
        vhpgvh_u1d1d1 = vhpg_u1d1 @ vh_d1
        vhpgvh_u1d1u1 = vhpg_u1d1 @ vh_u1
        vhpgvh_u1d1l1 = vhpg_u1d1 @ vh_l1
        vhpgvh_u1u1d1 = vhpg_u1u1 @ vh_d1
        vhpgvh_u1u1u1 = vhpg_u1u1 @ vh_u1
        vhpgvh_u1u1l1 = vhpg_u1u1 @ vh_l1
        vhpgvh_u1l1d1 = vhpg_u1l1 @ vh_d1
        vhpgvh_u1l1u1 = vhpg_u1l1 @ vh_u1
        vhpgvh_u1l1l1 = vhpg_u1l1 @ vh_l1
        vhpgvh_l1d1d1 = vhpg_l1d1 @ vh_d1
        vhpgvh_l1d1u1 = vhpg_l1d1 @ vh_u1
        vhpgvh_l1u1d1 = vhpg_l1u1 @ vh_d1
        vhpgvh_l1u1u1 = vhpg_l1u1 @ vh_u1
        vhpgvh_l1u1l1 = vhpg_l1u1 @ vh_l1
        vhpgvh_l1l1u1 = vhpg_l1l1 @ vh_u1
        vhpl_d1d1 = vh_d1 @ pl_d1
        vhpl_d1u1 = vh_d1 @ pl_u1
        vhpl_d1l1 = vh_d1 @ pl_l1
        vhpl_u1d1 = vh_u1 @ pl_d1
        vhpl_u1u1 = vh_u1 @ pl_u1
        vhpl_u1l1 = vh_u1 @ pl_l1
        vhpl_l1d1 = vh_l1 @ pl_d1
        vhpl_l1u1 = vh_l1 @ pl_u1
        vhpl_l1l1 = vh_l1 @ pl_l1
        vhplvh_d1d1d1 = vhpl_d1d1 @ vh_d1
        vhplvh_d1d1u1 = vhpl_d1d1 @ vh_u1
        vhplvh_d1d1l1 = vhpl_d1d1 @ vh_l1
        vhplvh_d1u1d1 = vhpl_d1u1 @ vh_d1
        vhplvh_d1u1u1 = vhpl_d1u1 @ vh_u1
        vhplvh_d1u1l1 = vhpl_d1u1 @ vh_l1
        vhplvh_d1l1d1 = vhpl_d1l1 @ vh_d1
        vhplvh_d1l1u1 = vhpl_d1l1 @ vh_u1
        vhplvh_u1d1d1 = vhpl_u1d1 @ vh_d1
        vhplvh_u1d1u1 = vhpl_u1d1 @ vh_u1
        vhplvh_u1d1l1 = vhpl_u1d1 @ vh_l1
        vhplvh_u1u1d1 = vhpl_u1u1 @ vh_d1
        vhplvh_u1u1u1 = vhpl_u1u1 @ vh_u1
        vhplvh_u1u1l1 = vhpl_u1u1 @ vh_l1
        vhplvh_u1l1d1 = vhpl_u1l1 @ vh_d1
        vhplvh_u1l1u1 = vhpl_u1l1 @ vh_u1
        vhplvh_u1l1l1 = vhpl_u1l1 @ vh_l1
        vhplvh_l1d1d1 = vhpl_l1d1 @ vh_d1
        vhplvh_l1d1u1 = vhpl_l1d1 @ vh_u1
        vhplvh_l1u1d1 = vhpl_l1u1 @ vh_d1
        vhplvh_l1u1u1 = vhpl_l1u1 @ vh_u1
        vhplvh_l1u1l1 = vhpl_l1u1 @ vh_l1
        vhplvh_l1l1u1 = vhpl_l1l1 @ vh_u1

        # fill output matrices
        # M^{r}\left(E\right)
        mr_d2[:lb, :lb] = vhpr_d1d1 + vhpr_u1l1 + vhpr_l1u1
        mr_d2[:lb, lb:] = vhpr_d1u1 + vhpr_u1d1
        mr_d2[lb:, :lb] = vhpr_d1l1 + vhpr_l1d1
        mr_d2[lb:, lb:] = vhpr_d1d1 + vhpr_l1u1 + vhpr_u1l1

        mr_u2[:lb, :lb] = vhpr_u1u1
        mr_u2[lb:, :lb] = vhpr_d1u1 + vhpr_u1d1
        mr_u2[lb:, lb:] = vhpr_u1u1

        mr_l2[:lb, :lb] = vhpr_l1l1
        mr_l2[:lb, lb:] = vhpr_d1l1 + vhpr_l1d1
        mr_l2[lb:, lb:] = vhpr_l1l1

        # L^{\lessgtr}\left(E\right)
        lg_d2[:lb, :
              lb] = vhpgvh_d1l1u1 + vhpgvh_l1d1u1 + vhpgvh_l1u1d1 + vhpgvh_d1d1d1 + vhpgvh_u1l1d1 + vhpgvh_d1u1l1 + vhpgvh_u1d1l1
        lg_d2[:lb, lb:] = vhpgvh_l1u1u1 + vhpgvh_u1u1l1 + vhpgvh_d1d1u1 + vhpgvh_u1l1u1 + vhpgvh_d1u1d1 + vhpgvh_u1d1d1
        lg_d2[lb:, :lb] = vhpgvh_l1l1u1 + vhpgvh_u1l1l1 + vhpgvh_d1d1l1 + vhpgvh_l1u1l1 + vhpgvh_d1l1d1 + vhpgvh_l1d1d1
        lg_d2[
            lb:,
            lb:] = vhpgvh_d1u1l1 + vhpgvh_u1d1l1 + vhpgvh_u1l1d1 + vhpgvh_d1d1d1 + vhpgvh_l1u1d1 + vhpgvh_d1l1u1 + vhpgvh_l1d1u1

        lg_u2[:lb, :lb] = vhpgvh_u1u1d1 + vhpgvh_d1u1u1 + vhpgvh_u1d1u1
        lg_u2[:lb, lb:] = vhpgvh_u1u1u1
        lg_u2[lb:, :lb] = vhpgvh_d1u1d1 + vhpgvh_u1d1d1 + vhpgvh_u1l1u1 + vhpgvh_u1u1l1 + vhpgvh_d1d1u1 + vhpgvh_l1u1u1
        lg_u2[lb:, lb:] = vhpgvh_d1u1u1 + vhpgvh_u1d1u1 + vhpgvh_u1u1d1

        ll_d2[:lb, :
              lb] = vhplvh_d1l1u1 + vhplvh_l1d1u1 + vhplvh_l1u1d1 + vhplvh_d1d1d1 + vhplvh_u1l1d1 + vhplvh_d1u1l1 + vhplvh_u1d1l1
        ll_d2[:lb, lb:] = vhplvh_l1u1u1 + vhplvh_u1u1l1 + vhplvh_d1d1u1 + vhplvh_u1l1u1 + vhplvh_d1u1d1 + vhplvh_u1d1d1
        ll_d2[lb:, :lb] = vhplvh_l1l1u1 + vhplvh_u1l1l1 + vhplvh_d1d1l1 + vhplvh_l1u1l1 + vhplvh_d1l1d1 + vhplvh_l1d1d1
        ll_d2[
            lb:,
            lb:] = vhplvh_d1u1l1 + vhplvh_u1d1l1 + vhplvh_u1l1d1 + vhplvh_d1d1d1 + vhplvh_l1u1d1 + vhplvh_d1l1u1 + vhplvh_l1d1u1

        ll_u2[:lb, :lb] = vhplvh_u1u1d1 + vhplvh_d1u1u1 + vhplvh_u1d1u1
        ll_u2[:lb, lb:] = vhplvh_u1u1u1
        ll_u2[lb:, :lb] = vhplvh_d1u1d1 + vhplvh_u1d1d1 + vhplvh_u1l1u1 + vhplvh_u1u1l1 + vhplvh_d1d1u1 + vhplvh_l1u1u1
        ll_u2[lb:, lb:] = vhplvh_d1u1u1 + vhplvh_u1d1u1 + vhplvh_u1u1d1

        dmr_lu[:lb, :lb] = vhpr_l1u1
        dmr_ul[lb:, lb:] = vhpr_u1l1

        dlg_lu[:lb, :lb] = vhpgvh_d1l1u1 + vhpgvh_l1d1u1 + vhpgvh_l1u1d1
        dlg_lu[:lb, lb:] = vhpgvh_l1u1u1
        dlg_lu[lb:, :lb] = vhpgvh_l1l1u1

        dlg_ul[:lb, lb:] = vhpgvh_u1u1l1
        dlg_ul[lb:, :lb] = vhpgvh_u1l1l1
        dlg_ul[lb:, lb:] = vhpgvh_d1u1l1 + vhpgvh_u1d1l1 + vhpgvh_u1l1d1

        dll_lu[:lb, :lb] = vhplvh_d1l1u1 + vhplvh_l1d1u1 + vhplvh_l1u1d1
        dll_lu[:lb, lb:] = vhplvh_l1u1u1
        dll_lu[lb:, :lb] = vhplvh_l1l1u1

        dll_ul[:lb, lb:] = vhplvh_u1u1l1
        dll_ul[lb:, :lb] = vhplvh_u1l1l1
        dll_ul[lb:, lb:] = vhplvh_d1u1l1 + vhplvh_u1d1l1 + vhplvh_u1l1d1

        vh_u[lb:, :lb] = vh_u1
        vh_l[:lb, lb:] = vh_l1
    elif nbc == 3:
        # compute multiplications
        vhpr_d1d1 = -vh_d1 @ pr_d1
        vhpr_d1u1 = -vh_d1 @ pr_u1
        vhpr_d1l1 = -vh_d1 @ pr_l1
        vhpr_u1d1 = -vh_u1 @ pr_d1
        vhpr_u1u1 = -vh_u1 @ pr_u1
        vhpr_u1l1 = -vh_u1 @ pr_l1
        vhpr_l1d1 = -vh_l1 @ pr_d1
        vhpr_l1u1 = -vh_l1 @ pr_u1
        vhpr_l1l1 = -vh_l1 @ pr_l1

        vhpg_d1d1 = vh_d1 @ pg_d1
        vhpg_d1u1 = vh_d1 @ pg_u1
        vhpg_d1l1 = vh_d1 @ pg_l1
        vhpg_u1d1 = vh_u1 @ pg_d1
        vhpg_u1u1 = vh_u1 @ pg_u1
        vhpg_u1l1 = vh_u1 @ pg_l1
        vhpg_l1d1 = vh_l1 @ pg_d1
        vhpg_l1u1 = vh_l1 @ pg_u1
        vhpg_l1l1 = vh_l1 @ pg_l1
        # vhpgvh_d1d1d1 = vhpl_d1d1 @ vh_d1
        # vhpgvh_d1d1u1 = vhpl_d1d1 @ vh_u1
        # vhpgvh_d1d1l1 = vhpl_d1d1 @ vh_l1
        # vhpgvh_d1u1d1 = vhpl_d1u1 @ vh_d1
        # vhpgvh_d1u1u1 = vhpl_d1u1 @ vh_u1
        # vhpgvh_d1u1l1 = vhpl_d1u1 @ vh_l1
        # vhpgvh_d1l1d1 = vhpl_d1l1 @ vh_d1
        # vhpgvh_d1l1u1 = vhpl_d1l1 @ vh_u1
        # vhpgvh_d1l1l1 = vhpl_d1l1 @ vh_l1
        # vhpgvh_u1d1d1 = vhpl_u1d1 @ vh_d1
        # vhpgvh_u1d1u1 = vhpl_u1d1 @ vh_u1
        # vhpgvh_u1d1l1 = vhpl_u1d1 @ vh_l1
        # vhpgvh_u1u1d1 = vhpl_u1u1 @ vh_d1
        # vhpgvh_u1u1u1 = vhpl_u1u1 @ vh_u1
        # vhpgvh_u1u1l1 = vhpl_u1u1 @ vh_l1
        # vhpgvh_u1l1d1 = vhpl_u1l1 @ vh_d1
        # vhpgvh_u1l1u1 = vhpl_u1l1 @ vh_u1
        # vhpgvh_u1l1l1 = vhpl_u1l1 @ vh_l1
        # vhpgvh_l1d1d1 = vhpl_l1d1 @ vh_d1
        # vhpgvh_l1d1u1 = vhpl_l1d1 @ vh_u1
        # vhpgvh_l1d1l1 = vhpl_l1d1 @ vh_l1
        # vhpgvh_l1u1d1 = vhpl_l1u1 @ vh_d1
        # vhpgvh_l1u1u1 = vhpl_l1u1 @ vh_u1
        # vhpgvh_l1u1l1 = vhpl_l1u1 @ vh_l1
        # vhpgvh_l1l1d1 = vhpl_l1l1 @ vh_d1
        # vhpgvh_l1l1u1 = vhpl_l1l1 @ vh_u1
        # Replacing above block with below block
        vhpgvh_d1d1d1 = vhpg_d1d1 @ vh_d1
        vhpgvh_d1d1u1 = vhpg_d1d1 @ vh_u1
        vhpgvh_d1d1l1 = vhpg_d1d1 @ vh_l1
        vhpgvh_d1u1d1 = vhpg_d1u1 @ vh_d1
        vhpgvh_d1u1u1 = vhpg_d1u1 @ vh_u1
        vhpgvh_d1u1l1 = vhpg_d1u1 @ vh_l1
        vhpgvh_d1l1d1 = vhpg_d1l1 @ vh_d1
        vhpgvh_d1l1u1 = vhpg_d1l1 @ vh_u1
        vhpgvh_u1d1d1 = vhpg_u1d1 @ vh_d1
        vhpgvh_u1d1u1 = vhpg_u1d1 @ vh_u1
        vhpgvh_u1d1l1 = vhpg_u1d1 @ vh_l1
        vhpgvh_u1u1d1 = vhpg_u1u1 @ vh_d1
        vhpgvh_u1u1u1 = vhpg_u1u1 @ vh_u1
        vhpgvh_u1u1l1 = vhpg_u1u1 @ vh_l1
        vhpgvh_u1l1d1 = vhpg_u1l1 @ vh_d1
        vhpgvh_u1l1u1 = vhpg_u1l1 @ vh_u1
        vhpgvh_u1l1l1 = vhpg_u1l1 @ vh_l1
        vhpgvh_l1d1d1 = vhpg_l1d1 @ vh_d1
        vhpgvh_l1d1u1 = vhpg_l1d1 @ vh_u1
        vhpgvh_l1u1d1 = vhpg_l1u1 @ vh_d1
        vhpgvh_l1u1u1 = vhpg_l1u1 @ vh_u1
        vhpgvh_l1u1l1 = vhpg_l1u1 @ vh_l1
        vhpgvh_l1l1u1 = vhpg_l1l1 @ vh_u1
        vhpgvh_l1l1d1 = vhpg_l1l1 @ vh_d1
        vhpgvh_d1l1l1 = vhpg_d1l1 @ vh_l1
        vhpgvh_l1d1l1 = vhpg_l1d1 @ vh_l1
        vhpl_d1d1 = vh_d1 @ pl_d1
        vhpl_d1u1 = vh_d1 @ pl_u1
        vhpl_d1l1 = vh_d1 @ pl_l1
        vhpl_u1d1 = vh_u1 @ pl_d1
        vhpl_u1u1 = vh_u1 @ pl_u1
        vhpl_u1l1 = vh_u1 @ pl_l1
        vhpl_l1d1 = vh_l1 @ pl_d1
        vhpl_l1u1 = vh_l1 @ pl_u1
        vhpl_l1l1 = vh_l1 @ pl_l1
        vhplvh_d1d1d1 = vhpl_d1d1 @ vh_d1
        vhplvh_d1d1u1 = vhpl_d1d1 @ vh_u1
        vhplvh_d1d1l1 = vhpl_d1d1 @ vh_l1
        vhplvh_d1u1d1 = vhpl_d1u1 @ vh_d1
        vhplvh_d1u1u1 = vhpl_d1u1 @ vh_u1
        vhplvh_d1u1l1 = vhpl_d1u1 @ vh_l1
        vhplvh_d1l1d1 = vhpl_d1l1 @ vh_d1
        vhplvh_d1l1u1 = vhpl_d1l1 @ vh_u1
        vhplvh_d1l1l1 = vhpl_d1l1 @ vh_l1
        vhplvh_u1d1d1 = vhpl_u1d1 @ vh_d1
        vhplvh_u1d1u1 = vhpl_u1d1 @ vh_u1
        vhplvh_u1d1l1 = vhpl_u1d1 @ vh_l1
        vhplvh_u1u1d1 = vhpl_u1u1 @ vh_d1
        vhplvh_u1u1u1 = vhpl_u1u1 @ vh_u1
        vhplvh_u1u1l1 = vhpl_u1u1 @ vh_l1
        vhplvh_u1l1d1 = vhpl_u1l1 @ vh_d1
        vhplvh_u1l1u1 = vhpl_u1l1 @ vh_u1
        vhplvh_u1l1l1 = vhpl_u1l1 @ vh_l1
        vhplvh_l1d1d1 = vhpl_l1d1 @ vh_d1
        vhplvh_l1d1u1 = vhpl_l1d1 @ vh_u1
        vhplvh_l1d1l1 = vhpl_l1d1 @ vh_l1
        vhplvh_l1u1d1 = vhpl_l1u1 @ vh_d1
        vhplvh_l1u1u1 = vhpl_l1u1 @ vh_u1
        vhplvh_l1u1l1 = vhpl_l1u1 @ vh_l1
        vhplvh_l1l1d1 = vhpl_l1l1 @ vh_d1
        vhplvh_l1l1u1 = vhpl_l1l1 @ vh_u1

        # fill output matrices
        # M^{r}\left(E\right)
        mr_d2[:lb, :lb] = vhpr_d1d1 + vhpr_l1u1 + vhpr_u1l1
        mr_d2[:lb, lb:2 * lb] = vhpr_d1u1 + vhpr_u1d1
        mr_d2[:lb, 2 * lb:] = vhpr_u1u1
        mr_d2[lb:2 * lb, :lb] = vhpr_d1l1 + vhpr_l1d1
        mr_d2[lb:2 * lb, lb:2 * lb] = vhpr_d1d1 + vhpr_l1u1 + vhpr_u1l1
        mr_d2[lb:2 * lb, 2 * lb:] = vhpr_d1u1 + vhpr_u1d1
        mr_d2[2 * lb:, :lb] = vhpr_l1l1
        mr_d2[2 * lb:, lb:2 * lb] = vhpr_d1l1 + vhpr_l1d1
        mr_d2[2 * lb:, 2 * lb:] = vhpr_d1d1 + vhpr_l1u1 + vhpr_u1l1

        mr_u2[lb:2 * lb, :lb] = vhpr_u1u1
        mr_u2[2 * lb:, :lb] = vhpr_d1u1 + vhpr_u1d1
        mr_u2[2 * lb:, lb:2 * lb] = vhpr_u1u1

        mr_l2[:lb, lb:2 * lb] = vhpr_l1l1
        mr_l2[:lb, 2 * lb:] = vhpr_d1l1 + vhpr_l1d1
        mr_l2[lb:2 * lb, 2 * lb:] = vhpr_l1l1

        lg_d2[:lb, :
              lb] = vhpgvh_d1l1u1 + vhpgvh_l1d1u1 + vhpgvh_l1u1d1 + vhpgvh_d1d1d1 + vhpgvh_u1l1d1 + vhpgvh_d1u1l1 + vhpgvh_u1d1l1
        lg_d2[:lb,
              lb:2 * lb] = vhpgvh_l1u1u1 + vhpgvh_u1u1l1 + vhpgvh_d1d1u1 + vhpgvh_u1l1u1 + vhpgvh_d1u1d1 + vhpgvh_u1d1d1
        lg_d2[:lb, 2 * lb:] = vhpgvh_u1u1d1 + vhpgvh_d1u1u1 + vhpgvh_u1d1u1
        lg_d2[lb:2 *
              lb, :lb] = vhpgvh_l1l1u1 + vhpgvh_d1l1d1 + vhpgvh_l1d1d1 + vhpgvh_d1d1l1 + vhpgvh_l1u1l1 + vhpgvh_u1l1l1
        lg_d2[
            lb:2 * lb, lb:2 *
            lb] = vhpgvh_d1l1u1 + vhpgvh_l1d1u1 + vhpgvh_d1u1l1 + vhpgvh_u1d1l1 + vhpgvh_d1d1d1 + vhpgvh_l1u1d1 + vhpgvh_u1l1d1
        lg_d2[lb:2 * lb,
              2 * lb:] = vhpgvh_u1u1l1 + vhpgvh_d1u1d1 + vhpgvh_u1d1d1 + vhpgvh_d1d1u1 + vhpgvh_l1u1u1 + vhpgvh_u1l1u1
        lg_d2[2 * lb:, :lb] = vhpgvh_l1l1d1 + vhpgvh_d1l1l1 + vhpgvh_l1d1l1
        lg_d2[2 * lb:,
              lb:2 * lb] = vhpgvh_l1l1u1 + vhpgvh_u1l1l1 + vhpgvh_d1d1l1 + vhpgvh_l1u1l1 + vhpgvh_d1l1d1 + vhpgvh_l1d1d1
        lg_d2[
            2 * lb:, 2 *
            lb:] = vhpgvh_d1u1l1 + vhpgvh_u1d1l1 + vhpgvh_u1l1d1 + vhpgvh_d1d1d1 + vhpgvh_l1u1d1 + vhpgvh_d1l1u1 + vhpgvh_l1d1u1

        lg_u2[:lb, :lb] = vhpgvh_u1u1u1
        lg_u2[lb:2 * lb, :lb] = vhpgvh_u1u1d1 + vhpgvh_d1u1u1 + vhpgvh_u1d1u1
        lg_u2[lb:2 * lb, lb:2 * lb] = vhpgvh_u1u1u1
        lg_u2[2 *
              lb:, :lb] = vhpgvh_d1u1d1 + vhpgvh_u1d1d1 + vhpgvh_u1l1u1 + vhpgvh_u1u1l1 + vhpgvh_d1d1u1 + vhpgvh_l1u1u1
        lg_u2[2 * lb:, lb:2 * lb] = vhpgvh_d1u1u1 + vhpgvh_u1d1u1 + vhpgvh_u1u1d1
        lg_u2[2 * lb:, 2 * lb:] = vhpgvh_u1u1u1

        ll_d2[:lb, :
              lb] = vhplvh_d1l1u1 + vhplvh_l1d1u1 + vhplvh_l1u1d1 + vhplvh_d1d1d1 + vhplvh_u1l1d1 + vhplvh_d1u1l1 + vhplvh_u1d1l1
        ll_d2[:lb,
              lb:2 * lb] = vhplvh_l1u1u1 + vhplvh_u1u1l1 + vhplvh_d1d1u1 + vhplvh_u1l1u1 + vhplvh_d1u1d1 + vhplvh_u1d1d1
        ll_d2[:lb, 2 * lb:] = vhplvh_u1u1d1 + vhplvh_d1u1u1 + vhplvh_u1d1u1
        ll_d2[lb:2 *
              lb, :lb] = vhplvh_l1l1u1 + vhplvh_d1l1d1 + vhplvh_l1d1d1 + vhplvh_d1d1l1 + vhplvh_l1u1l1 + vhplvh_u1l1l1
        ll_d2[
            lb:2 * lb, lb:2 *
            lb] = vhplvh_d1l1u1 + vhplvh_l1d1u1 + vhplvh_d1u1l1 + vhplvh_u1d1l1 + vhplvh_d1d1d1 + vhplvh_l1u1d1 + vhplvh_u1l1d1
        ll_d2[lb:2 * lb,
              2 * lb:] = vhplvh_u1u1l1 + vhplvh_d1u1d1 + vhplvh_u1d1d1 + vhplvh_d1d1u1 + vhplvh_l1u1u1 + vhplvh_u1l1u1
        ll_d2[2 * lb:, :lb] = vhplvh_l1l1d1 + vhplvh_d1l1l1 + vhplvh_l1d1l1
        ll_d2[2 * lb:,
              lb:2 * lb] = vhplvh_l1l1u1 + vhplvh_u1l1l1 + vhplvh_d1d1l1 + vhplvh_l1u1l1 + vhplvh_d1l1d1 + vhplvh_l1d1d1
        ll_d2[
            2 * lb:, 2 *
            lb:] = vhplvh_d1u1l1 + vhplvh_u1d1l1 + vhplvh_u1l1d1 + vhplvh_d1d1d1 + vhplvh_l1u1d1 + vhplvh_d1l1u1 + vhplvh_l1d1u1

        ll_u2[:lb, :lb] = vhplvh_u1u1u1
        ll_u2[lb:2 * lb, :lb] = vhplvh_u1u1d1 + vhplvh_d1u1u1 + vhplvh_u1d1u1
        ll_u2[lb:2 * lb, lb:2 * lb] = vhplvh_u1u1u1
        ll_u2[2 *
              lb:, :lb] = vhplvh_d1u1d1 + vhplvh_u1d1d1 + vhplvh_u1l1u1 + vhplvh_u1u1l1 + vhplvh_d1d1u1 + vhplvh_l1u1u1
        ll_u2[2 * lb:, lb:2 * lb] = vhplvh_d1u1u1 + vhplvh_u1d1u1 + vhplvh_u1u1d1
        ll_u2[2 * lb:, 2 * lb:] = vhplvh_u1u1u1

        dmr_lu[:lb, :lb] = vhpr_l1u1

        dmr_ul[2 * lb:, 2 * lb:] = vhpr_u1l1

        dlg_lu[:lb, :lb] = vhpgvh_d1l1u1 + vhpgvh_l1d1u1 + vhpgvh_l1u1d1
        dlg_lu[:lb, lb:2 * lb] = vhpgvh_l1u1u1
        dlg_lu[lb:2 * lb, :lb] = vhpgvh_l1l1u1

        dlg_ul[lb:2 * lb, 2 * lb:] = vhpgvh_u1u1l1
        dlg_ul[2 * lb:, lb:2 * lb] = vhpgvh_u1l1l1
        dlg_ul[2 * lb:, 2 * lb:] = vhpgvh_d1u1l1 + vhpgvh_u1d1l1 + vhpgvh_u1l1d1

        dll_lu[:lb, :lb] = vhplvh_d1l1u1 + vhplvh_l1d1u1 + vhplvh_l1u1d1
        dll_lu[:lb, lb:2 * lb] = vhplvh_l1u1u1
        dll_lu[lb:2 * lb, :lb] = vhplvh_l1l1u1

        dll_ul[lb:2 * lb, 2 * lb:] = vhplvh_u1u1l1
        dll_ul[2 * lb:, lb:2 * lb] = vhplvh_u1l1l1
        dll_ul[2 * lb:, 2 * lb:] = vhplvh_d1u1l1 + vhplvh_u1d1l1 + vhplvh_u1l1d1

        vh_u[2 * lb:, :lb] = vh_u1 # change

        vh_l[:lb, 2 * lb:] = vh_l1 # change


    lg_l2[:, :] = -lg_u2.conjugate().T
    ll_l2[:, :] = -ll_u2.conjugate().T
    mr_d2 = mr_d2 + cp.identity(lb_mm, dtype=cp.complex128) * (1 + 1j * 1e-10)
    (mr_d2, mr_u2, mr_l2, matrix_blocks ) = extract_small_matrix_blocks_gpu(mr_d2, mr_u2, mr_l2, NCpSC*nbc, side, densify=True)

    return ((mr_d2, mr_u2, mr_l2), (lg_d2, lg_u2, lg_l2), (ll_d2, ll_u2, ll_l2), (dmr_lu, dmr_ul), (dlg_lu, dlg_ul),
            (dll_lu, dll_ul), (vh_u, vh_l), matrix_blocks)

def get_dl_obc_alt(xr_d: cp.ndarray, lg_d: cp.ndarray, lg_o: cp.ndarray, ll_d: cp.ndarray, ll_o: cp.ndarray,
                   mr_x: cp.ndarray, blk: str)-> typing.Tuple[np.ndarray, np.ndarray]:
    """
    Calculates open boundary corrections for lg and ll.
    Assumes that input blocks are dense.

    Args:
        xr_d (cp.ndarray):
        ll_d (cp.ndarray):
        ll_o (cp.ndarray):
        lg_d (cp.ndarray):
        lg_o (cp.ndarray):
        mr_x (cp.ndarray):
        blk (str): either "R" or "L" depending on which side to correct

    Raises:
        ValueError: if blk is not "R" or "L"

    Returns:

    """
    # Number of iterations for the refinement
    ref_iteration = 1

    # length of block
    lb = mr_x.shape[0]

    # non zero indexes of mr_x
    #rows, cols = mr_x.nonzero()

    # non zero indexes of mr_x
    mr_x_max = np.max(np.abs(mr_x.get()))
    rows, cols = np.where(np.abs(mr_x.get()) > mr_x_max / 1e8)

    if (not rows.size):
        return np.nan, np.nan

    # conjugate transpose of mr/xr
    mr_x_ct = mr_x.conjugate().T
    xr_d_ct = xr_d.conjugate().T

    mrxr_xd = mr_x @ xr_d
    ag = mrxr_xd @ lg_o
    al = mrxr_xd @ ll_o

    # only difference between ax and ax^H is needed
    ag_diff = ag - ag.conjugate().T
    al_diff = al - al.conjugate().T

    fg = xr_d @ (lg_d - ag_diff) @ xr_d_ct
    fl = xr_d @ (ll_d - al_diff) @ xr_d_ct

    # case for the left/right (start/end) block
    # differentiates between which block to look at
    if blk == "L":
        #idx_max = np.max([np.max(rows), lb - np.min(cols)])
        idx_max = np.max([np.max(rows) + 1, lb - np.min(cols)])
        ip = lb - idx_max
        sl_x = slice(ip, lb)
        #sl_y = slice(0, idx_max + 1)
        sl_y = slice(0, idx_max)
    elif blk == "R":
        #idx_max = np.max([np.max(cols), lb - np.min(rows)])
        idx_max = np.max([np.max(cols) + 1, lb - np.min(rows)])
        ip = lb - idx_max
        #sl_x = slice(0, idx_max + 1)
        sl_x = slice(0, idx_max)
        sl_y = slice(ip, lb)
    else:
        raise ValueError("Argument error, type input not possible")

    ar = xr_d[sl_x, sl_y] @ mr_x[sl_y, sl_x]
    # add imaginary part to stabilize
    #ar = ar + np.identity(ar.shape[0])*1j*1e-4

    ar_host = ar.get()
    eival_host, eivec_host = np.linalg.eig(ar_host)
    eival = cp.array(eival_host)
    eivec = cp.array(eivec_host)
    # eigen values and eigen vectors
    #eival, eivec = cp.linalg.eig(ar)

    # compute the reduced systems of eigenvalues and eigenvectors
    ieivec = cp.linalg.inv(eivec)

    Emax = cp.max(cp.abs(eival))
    ind = cp.where(cp.abs(eival) > Emax / 1e8)[0]
    eival_red = eival[ind]
    # eivec = eivec[:, ind]
    # ieivec = ieivec[ind, :]


    # conjugate/transpose/abs square
    eivec_ct = eivec.conjugate().T
    #ieivec = np.linalg.inv(eivec)
    ieivec_ct = ieivec.conjugate().T
    eival_sq = cp.diag(eival) @ cp.diag(eival).conjugate()
    eival_sq_red = cp.outer(eival_red,eival_red.conjugate())

    # greater component
    yg_d = cp.divide(ieivec[ind,:] @ fg[sl_x, sl_x] @ ieivec[ind, :].conjugate().T, 1 - eival_sq_red) - ieivec[ind,:] @ fg[sl_x, sl_x] @ ieivec[ind, :].conjugate().T
    qg = cp.zeros((sl_x.stop - sl_x.start, sl_x.stop - sl_x.start), dtype=np.complex128)
    qg[:ind.shape[0], :ind.shape[0]] = yg_d
    wg_d = fg[sl_x, sl_x] +  eivec @ qg @ eivec_ct
    xrmr_dx_s = xr_d[sl_x, :] @ mr_x[:, sl_x]
    mrxr_ct_xd_s = mr_x_ct[sl_x, :] @ xr_d_ct[:, sl_x]

    for i in range(ref_iteration):
        wg_d = fg[sl_x, sl_x] + xrmr_dx_s @ wg_d @ mrxr_ct_xd_s

    dlg_d = (mr_x[:, sl_x] @ wg_d @ mr_x_ct[sl_x, :] - ag_diff).get()
    # dlg_d = mr_x[:, sl_x] @ wg_d @ mr_x_ct[sl_x, :] - ag_diff

    # lesser component
    yl_d = cp.divide(ieivec[ind,:] @ fl[sl_x, sl_x] @ ieivec[ind, :].conjugate().T, 1 - eival_sq_red) - ieivec[ind,:] @ fl[sl_x, sl_x] @ ieivec[ind, :].conjugate().T
    ql = cp.zeros((sl_x.stop - sl_x.start, sl_x.stop - sl_x.start), dtype=np.complex128)
    ql[:ind.shape[0], :ind.shape[0]] = yl_d
    wl_d = fl[sl_x, sl_x] +  eivec @ ql @ eivec_ct
    for i in range(ref_iteration):
        wl_d = fl[sl_x, sl_x] + xrmr_dx_s @ wl_d @ mrxr_ct_xd_s

    dll_d = (mr_x[:, sl_x] @ wl_d @ mr_x_ct[sl_x, :] - al_diff).get()
    # dll_d = mr_x[:, sl_x] @ wl_d @ mr_x_ct[sl_x, :] - al_diff

    return dlg_d, dll_d


def get_mm_obc_dense_batched(
    vh_1: cp.ndarray, vh_2: cp.ndarray, pg_1: cp.ndarray, pg_2: cp.ndarray, pl_1: cp.ndarray, pl_2: cp.ndarray,
    pr_1: cp.ndarray, pr_2: cp.ndarray, pr_3: cp.ndarray, nbc: int, NCpSC: int, side: str = "L"
) -> typing.Tuple[typing.Tuple[cp.ndarray, cp.ndarray, cp.ndarray], typing.Tuple[cp.ndarray, cp.ndarray, cp.ndarray],
                  typing.Tuple[cp.ndarray, cp.ndarray, cp.ndarray], typing.Tuple[cp.ndarray, cp.ndarray], typing.Tuple[
                      cp.ndarray, cp.ndarray], typing.Tuple[cp.ndarray, cp.ndarray], typing.Tuple[cp.ndarray,
                                                                                                  cp.ndarray]]:
    """
    Calculates the different correction terms for the scattering OBCs for the screened interaction calculations.
    In this version the blocks are not stacked, but the matrices are multiplied directly and inserted afterwards.

    Args:
        vh_1 (cp.ndarray): Diagonal block of effective interaction
        vh_2 (cp.ndarray): Off diagonal block of effective interaction
        pg_1 (cp.ndarray): Diagonal block of greater polarization
        pg_2 (cp.ndarray): Off diagonal block of greater polarization
        pl_1 (cp.ndarray): Diagonal block of lesser polarization
        pl_2 (cp.ndarray): Off diagonal block of lesser polarization
        pr_1 (cp.ndarray): Diagonal block of retarded polarization
        pr_2 (cp.ndarray): Off diagonal block of retarded polarization
        nbc (int): How block size changes after matrix multiplication

    Returns:
        typing.Tuple[
            typing.Tuple[cp.ndarray, cp.ndarray, cp.ndarray],
            typing.Tuple[cp.ndarray, cp.ndarray, cp.ndarray],
            typing.Tuple[cp.ndarray, cp.ndarray, cp.ndarray],
            typing.Tuple[cp.ndarray, cp.ndarray],
            typing.Tuple[cp.ndarray, cp.ndarray],
            typing.Tuple[cp.ndarray, cp.ndarray],
            typing.Tuple[cp.ndarray, cp.ndarray]
            ]: mr_d/u/l, lg_d/u/l, ll_d/u/l, dg_lu/ul, dl_lu/ul, vh_u/l
    """
    # block size
    lb = vh_1.shape[1]
    # block size after mm
    lb_mm = nbc * lb
    # define the right blocks
    (vh_d1, vh_u1, vh_l1, _) = extract_small_matrix_blocks_batched_gpu(vh_1, vh_2,
                                                            vh_2.conjugate().transpose(0, 2, 1),
                                                            NCpSC, side)  
    # else:
    #     vh_d1 = vh_1
    #     vh_u1 = vh_2
    #     vh_l1 = vh_u1.conjugate().transpose()
    # pg_d1 = pg_1
    # pg_u1 = pg_2
    # pg_l1 = -pg_u1.conjugate().transpose()
    (pg_d1, pg_u1, pg_l1, _) = extract_small_matrix_blocks_batched_gpu(pg_1, pg_2,
                                                            -pg_2.conjugate().transpose(0, 2, 1),
                                                            NCpSC, side)


    # if side == "L":
    (pl_d1, pl_u1, pl_l1, _) = extract_small_matrix_blocks_batched_gpu(pl_1, pl_2,
                                                            -pl_2.conjugate().transpose(0, 2, 1),
                                                            NCpSC, side)
    # else: 
    #     pl_d1 = pl_1
    #     pl_u1 = pl_2
    #     pl_l1 = -pl_u1.conjugate().transpose()

    # if side == "L":
    (pr_d1, pr_u1, pr_l1, _) = extract_small_matrix_blocks_batched_gpu(pr_1, pr_2, pr_3, NCpSC, side)
    # else:

    # output matrices
    batch_size = vh_1.shape[0]
    if nbc > 1:
        mr_d2 = cp.empty((batch_size, lb_mm, lb_mm), dtype=cp.complex128)
        mr_u2 = cp.zeros((batch_size, lb_mm, lb_mm), dtype=cp.complex128)
        mr_l2 = cp.zeros((batch_size, lb_mm, lb_mm), dtype=cp.complex128)
        lg_d2 = cp.empty((batch_size, lb_mm, lb_mm), dtype=cp.complex128)
        lg_u2 = cp.zeros((batch_size, lb_mm, lb_mm), dtype=cp.complex128)
        lg_l2 = cp.empty((batch_size, lb_mm, lb_mm), dtype=cp.complex128)
        ll_d2 = cp.empty((batch_size, lb_mm, lb_mm), dtype=cp.complex128)
        ll_u2 = cp.zeros((batch_size, lb_mm, lb_mm), dtype=cp.complex128)
        ll_l2 = cp.empty((batch_size, lb_mm, lb_mm), dtype=cp.complex128)
        dmr_lu = cp.zeros((batch_size, lb_mm, lb_mm), dtype=cp.complex128)
        dmr_ul = cp.zeros((batch_size, lb_mm, lb_mm), dtype=cp.complex128)
        dlg_lu = cp.zeros((batch_size, lb_mm, lb_mm), dtype=cp.complex128)
        dlg_ul = cp.zeros((batch_size, lb_mm, lb_mm), dtype=cp.complex128)
        dll_lu = cp.zeros((batch_size, lb_mm, lb_mm), dtype=cp.complex128)
        dll_ul = cp.zeros((batch_size, lb_mm, lb_mm), dtype=cp.complex128)
        vh_u = cp.zeros((batch_size, lb_mm, lb_mm), dtype=cp.complex128)
        vh_l = cp.zeros((batch_size, lb_mm, lb_mm), dtype=cp.complex128)
    if nbc == 1:
        # fill output matrices
        # M^{r}\left(E\right)
        # mr_d2[:] = -vh_d1 @ pr_d1 - vh_u1 @ pr_l1 - vh_l1 @ pr_u1
        # mr_u2[:] = -vh_d1 @ pr_u1 - vh_u1 @ pr_d1
        # mr_l2[:] = -vh_d1 @ pr_l1 - vh_l1 @ pr_d1

        # # from L^{\lessgtr}\left(E\right)
        # lg_d2[:, :] = (vh_l1 @ pg_d1 @ vh_u1 + vh_l1 @ pg_u1 @ vh_d1 + vh_d1 @ pg_l1 @ vh_u1 + vh_d1 @ pg_d1 @ vh_d1 +
        #                vh_d1 @ pg_u1 @ vh_l1 + vh_u1 @ pg_l1 @ vh_d1 + vh_u1 @ pg_d1 @ vh_l1)
        # lg_u2[:, :] = (vh_l1 @ pg_u1 @ vh_u1 + vh_d1 @ pg_d1 @ vh_u1 + vh_d1 @ pg_u1 @ vh_d1 + vh_u1 @ pg_l1 @ vh_u1 +
        #                vh_u1 @ pg_d1 @ vh_d1)
        # ll_d2[:, :] = (vh_l1 @ pl_d1 @ vh_u1 + vh_l1 @ pl_u1 @ vh_d1 + vh_d1 @ pl_l1 @ vh_u1 + vh_d1 @ pl_d1 @ vh_d1 +
        #                vh_d1 @ pl_u1 @ vh_l1 + vh_u1 @ pl_l1 @ vh_d1 + vh_u1 @ pl_d1 @ vh_l1)
        # ll_u2[:, :] = (vh_l1 @ pl_u1 @ vh_u1 + vh_d1 @ pl_d1 @ vh_u1 + vh_d1 @ pl_u1 @ vh_d1 + vh_u1 @ pl_l1 @ vh_u1 +
        #                vh_u1 @ pl_d1 @ vh_d1)
        # dmr_lu[:, :] = -vh_l1 @ pr_u1
        # dmr_ul[:, :] = -vh_u1 @ pr_l1
        # dlg_lu[:, :] = vh_l1 @ pg_d1 @ vh_u1 + vh_l1 @ pg_u1 @ vh_d1 + vh_d1 @ pg_l1 @ vh_u1
        # dll_lu[:, :] = vh_l1 @ pl_d1 @ vh_u1 + vh_l1 @ pl_u1 @ vh_d1 + vh_d1 @ pl_l1 @ vh_u1
        # dlg_ul[:, :] = vh_u1 @ pg_d1 @ vh_l1 + vh_u1 @ pg_l1 @ vh_d1 + vh_d1 @ pg_u1 @ vh_l1
        # dll_ul[:, :] = vh_u1 @ pl_d1 @ vh_l1 + vh_u1 @ pl_l1 @ vh_d1 + vh_d1 @ pl_u1 @ vh_l1
        # vh_u[:, :] = vh_u1
        # vh_l[:, :] = vh_l1

        mr_d2 = -vh_d1 @ pr_d1 - vh_u1 @ pr_l1 - vh_l1 @ pr_u1
        mr_u2 = -vh_d1 @ pr_u1 - vh_u1 @ pr_d1
        mr_l2 = -vh_d1 @ pr_l1 - vh_l1 @ pr_d1

        # from L^{\lessgtr}\left(E\right)
        lg_d2 = (vh_l1 @ pg_d1 @ vh_u1 + vh_l1 @ pg_u1 @ vh_d1 + vh_d1 @ pg_l1 @ vh_u1 + vh_d1 @ pg_d1 @ vh_d1 +
                       vh_d1 @ pg_u1 @ vh_l1 + vh_u1 @ pg_l1 @ vh_d1 + vh_u1 @ pg_d1 @ vh_l1)
        lg_u2 = (vh_l1 @ pg_u1 @ vh_u1 + vh_d1 @ pg_d1 @ vh_u1 + vh_d1 @ pg_u1 @ vh_d1 + vh_u1 @ pg_l1 @ vh_u1 +
                       vh_u1 @ pg_d1 @ vh_d1)
        ll_d2 = (vh_l1 @ pl_d1 @ vh_u1 + vh_l1 @ pl_u1 @ vh_d1 + vh_d1 @ pl_l1 @ vh_u1 + vh_d1 @ pl_d1 @ vh_d1 +
                       vh_d1 @ pl_u1 @ vh_l1 + vh_u1 @ pl_l1 @ vh_d1 + vh_u1 @ pl_d1 @ vh_l1)
        ll_u2 = (vh_l1 @ pl_u1 @ vh_u1 + vh_d1 @ pl_d1 @ vh_u1 + vh_d1 @ pl_u1 @ vh_d1 + vh_u1 @ pl_l1 @ vh_u1 +
                       vh_u1 @ pl_d1 @ vh_d1)
        dmr_lu = -vh_l1 @ pr_u1
        dmr_ul = -vh_u1 @ pr_l1
        dlg_lu = vh_l1 @ pg_d1 @ vh_u1 + vh_l1 @ pg_u1 @ vh_d1 + vh_d1 @ pg_l1 @ vh_u1
        dll_lu = vh_l1 @ pl_d1 @ vh_u1 + vh_l1 @ pl_u1 @ vh_d1 + vh_d1 @ pl_l1 @ vh_u1
        dlg_ul = vh_u1 @ pg_d1 @ vh_l1 + vh_u1 @ pg_l1 @ vh_d1 + vh_d1 @ pg_u1 @ vh_l1
        dll_ul = vh_u1 @ pl_d1 @ vh_l1 + vh_u1 @ pl_l1 @ vh_d1 + vh_d1 @ pl_u1 @ vh_l1
        vh_u = vh_u1
        vh_l = vh_l1
    elif nbc == 2:
        vhpx_d1d1 = -vh_d1 @ pr_d1
        vhpx_d1u1 = -vh_d1 @ pr_u1
        vhpx_d1l1 = -vh_d1 @ pr_l1
        vhpx_u1d1 = -vh_u1 @ pr_d1
        vhpx_u1u1 = -vh_u1 @ pr_u1
        vhpx_u1l1 = -vh_u1 @ pr_l1
        vhpx_l1d1 = -vh_l1 @ pr_d1
        vhpx_l1u1 = -vh_l1 @ pr_u1
        vhpx_l1l1 = -vh_l1 @ pr_l1

        # M^{r}\left(E\right)
        mr_d2[:, :lb, :lb] = vhpx_d1d1 + vhpx_u1l1 + vhpx_l1u1
        mr_d2[:, :lb, lb:] = vhpx_d1u1 + vhpx_u1d1
        mr_d2[:, lb:, :lb] = vhpx_d1l1 + vhpx_l1d1
        mr_d2[:, lb:, lb:] = vhpx_d1d1 + vhpx_l1u1 + vhpx_u1l1

        mr_u2[:, :lb, :lb] = vhpx_u1u1
        mr_u2[:, lb:, :lb] = vhpx_d1u1 + vhpx_u1d1
        mr_u2[:, lb:, lb:] = vhpx_u1u1

        mr_l2[:, :lb, :lb] = vhpx_l1l1
        mr_l2[:, :lb, lb:] = vhpx_d1l1 + vhpx_l1d1
        mr_l2[:, lb:, lb:] = vhpx_l1l1

        dmr_lu[:, :lb, :lb] = vhpx_l1u1
        dmr_ul[:, lb:, lb:] = vhpx_u1l1

        # vhpx_d1d1[:] = vh_d1 @ pg_d1
        # vhpx_d1u1[:] = vh_d1 @ pg_u1
        # vhpx_d1l1[:] = vh_d1 @ pg_l1
        # vhpx_u1d1[:] = vh_u1 @ pg_d1
        # vhpx_u1u1[:] = vh_u1 @ pg_u1
        # vhpx_u1l1[:] = vh_u1 @ pg_l1
        # vhpx_l1d1[:] = vh_l1 @ pg_d1
        # vhpx_l1u1[:] = vh_l1 @ pg_u1
        # vhpx_l1l1[:] = vh_l1 @ pg_l1
        cp.matmul(vh_d1, pg_d1, out=vhpx_d1d1)
        cp.matmul(vh_d1, pg_u1, out=vhpx_d1u1)
        cp.matmul(vh_d1, pg_l1, out=vhpx_d1l1)
        cp.matmul(vh_u1, pg_d1, out=vhpx_u1d1)
        cp.matmul(vh_u1, pg_u1, out=vhpx_u1u1)
        cp.matmul(vh_u1, pg_l1, out=vhpx_u1l1)
        cp.matmul(vh_l1, pg_d1, out=vhpx_l1d1)
        cp.matmul(vh_l1, pg_u1, out=vhpx_l1u1)
        cp.matmul(vh_l1, pg_l1, out=vhpx_l1l1)
        vhpxvh_d1d1d1 = vhpx_d1d1 @ vh_d1
        vhpxvh_d1d1u1 = vhpx_d1d1 @ vh_u1
        vhpxvh_d1d1l1 = vhpx_d1d1 @ vh_l1
        vhpxvh_d1u1d1 = vhpx_d1u1 @ vh_d1
        vhpxvh_d1u1u1 = vhpx_d1u1 @ vh_u1
        vhpxvh_d1u1l1 = vhpx_d1u1 @ vh_l1
        vhpxvh_d1l1d1 = vhpx_d1l1 @ vh_d1
        vhpxvh_d1l1u1 = vhpx_d1l1 @ vh_u1
        vhpxvh_u1d1d1 = vhpx_u1d1 @ vh_d1
        vhpxvh_u1d1u1 = vhpx_u1d1 @ vh_u1
        vhpxvh_u1d1l1 = vhpx_u1d1 @ vh_l1
        vhpxvh_u1u1d1 = vhpx_u1u1 @ vh_d1
        vhpxvh_u1u1u1 = vhpx_u1u1 @ vh_u1
        vhpxvh_u1u1l1 = vhpx_u1u1 @ vh_l1
        vhpxvh_u1l1d1 = vhpx_u1l1 @ vh_d1
        vhpxvh_u1l1u1 = vhpx_u1l1 @ vh_u1
        vhpxvh_u1l1l1 = vhpx_u1l1 @ vh_l1
        vhpxvh_l1d1d1 = vhpx_l1d1 @ vh_d1
        vhpxvh_l1d1u1 = vhpx_l1d1 @ vh_u1
        vhpxvh_l1u1d1 = vhpx_l1u1 @ vh_d1
        vhpxvh_l1u1u1 = vhpx_l1u1 @ vh_u1
        vhpxvh_l1u1l1 = vhpx_l1u1 @ vh_l1
        vhpxvh_l1l1u1 = vhpx_l1l1 @ vh_u1

        # L^{\gtr}\left(E\right)
        lg_d2[:, :lb, :
              lb] = vhpxvh_d1l1u1 + vhpxvh_l1d1u1 + vhpxvh_l1u1d1 + vhpxvh_d1d1d1 + vhpxvh_u1l1d1 + vhpxvh_d1u1l1 + vhpxvh_u1d1l1
        lg_d2[:, :lb, lb:] = vhpxvh_l1u1u1 + vhpxvh_u1u1l1 + vhpxvh_d1d1u1 + vhpxvh_u1l1u1 + vhpxvh_d1u1d1 + vhpxvh_u1d1d1
        lg_d2[:, lb:, :lb] = vhpxvh_l1l1u1 + vhpxvh_u1l1l1 + vhpxvh_d1d1l1 + vhpxvh_l1u1l1 + vhpxvh_d1l1d1 + vhpxvh_l1d1d1
        lg_d2[:,
            lb:,
            lb:] = vhpxvh_d1u1l1 + vhpxvh_u1d1l1 + vhpxvh_u1l1d1 + vhpxvh_d1d1d1 + vhpxvh_l1u1d1 + vhpxvh_d1l1u1 + vhpxvh_l1d1u1

        lg_u2[:, :lb, :lb] = vhpxvh_u1u1d1 + vhpxvh_d1u1u1 + vhpxvh_u1d1u1
        lg_u2[:, :lb, lb:] = vhpxvh_u1u1u1
        lg_u2[:, lb:, :lb] = vhpxvh_d1u1d1 + vhpxvh_u1d1d1 + vhpxvh_u1l1u1 + vhpxvh_u1u1l1 + vhpxvh_d1d1u1 + vhpxvh_l1u1u1
        lg_u2[:, lb:, lb:] = vhpxvh_d1u1u1 + vhpxvh_u1d1u1 + vhpxvh_u1u1d1

        dlg_lu[:, :lb, :lb] = vhpxvh_d1l1u1 + vhpxvh_l1d1u1 + vhpxvh_l1u1d1
        dlg_lu[:, :lb, lb:] = vhpxvh_l1u1u1
        dlg_lu[:, lb:, :lb] = vhpxvh_l1l1u1

        dlg_ul[:, :lb, lb:] = vhpxvh_u1u1l1
        dlg_ul[:, lb:, :lb] = vhpxvh_u1l1l1
        dlg_ul[:, lb:, lb:] = vhpxvh_d1u1l1 + vhpxvh_u1d1l1 + vhpxvh_u1l1d1

        # vhpx_d1d1[:] = vh_d1 @ pl_d1
        # vhpx_d1u1[:] = vh_d1 @ pl_u1
        # vhpx_d1l1[:] = vh_d1 @ pl_l1
        # vhpx_u1d1[:] = vh_u1 @ pl_d1
        # vhpx_u1u1[:] = vh_u1 @ pl_u1
        # vhpx_u1l1[:] = vh_u1 @ pl_l1
        # vhpx_l1d1[:] = vh_l1 @ pl_d1
        # vhpx_l1u1[:] = vh_l1 @ pl_u1
        # vhpx_l1l1[:] = vh_l1 @ pl_l1
        # vhpxvh_d1d1d1[:] = vhpx_d1d1 @ vh_d1
        # vhpxvh_d1d1u1[:] = vhpx_d1d1 @ vh_u1
        # vhpxvh_d1d1l1[:] = vhpx_d1d1 @ vh_l1
        # vhpxvh_d1u1d1[:] = vhpx_d1u1 @ vh_d1
        # vhpxvh_d1u1u1[:] = vhpx_d1u1 @ vh_u1
        # vhpxvh_d1u1l1[:] = vhpx_d1u1 @ vh_l1
        # vhpxvh_d1l1d1[:] = vhpx_d1l1 @ vh_d1
        # vhpxvh_d1l1u1[:] = vhpx_d1l1 @ vh_u1
        # vhpxvh_u1d1d1[:] = vhpx_u1d1 @ vh_d1
        # vhpxvh_u1d1u1[:] = vhpx_u1d1 @ vh_u1
        # vhpxvh_u1d1l1[:] = vhpx_u1d1 @ vh_l1
        # vhpxvh_u1u1d1[:] = vhpx_u1u1 @ vh_d1
        # vhpxvh_u1u1u1[:] = vhpx_u1u1 @ vh_u1
        # vhpxvh_u1u1l1[:] = vhpx_u1u1 @ vh_l1
        # vhpxvh_u1l1d1[:] = vhpx_u1l1 @ vh_d1
        # vhpxvh_u1l1u1[:] = vhpx_u1l1 @ vh_u1
        # vhpxvh_u1l1l1[:] = vhpx_u1l1 @ vh_l1
        # vhpxvh_l1d1d1[:] = vhpx_l1d1 @ vh_d1
        # vhpxvh_l1d1u1[:] = vhpx_l1d1 @ vh_u1
        # vhpxvh_l1u1d1[:] = vhpx_l1u1 @ vh_d1
        # vhpxvh_l1u1u1[:] = vhpx_l1u1 @ vh_u1
        # vhpxvh_l1u1l1[:] = vhpx_l1u1 @ vh_l1
        # vhpxvh_l1l1u1[:] = vhpx_l1l1 @ vh_u1
        cp.matmul(vh_d1, pl_d1, out=vhpx_d1d1)
        cp.matmul(vh_d1, pl_u1, out=vhpx_d1u1)
        cp.matmul(vh_d1, pl_l1, out=vhpx_d1l1)
        cp.matmul(vh_u1, pl_d1, out=vhpx_u1d1)
        cp.matmul(vh_u1, pl_u1, out=vhpx_u1u1)
        cp.matmul(vh_u1, pl_l1, out=vhpx_u1l1)
        cp.matmul(vh_l1, pl_d1, out=vhpx_l1d1)
        cp.matmul(vh_l1, pl_u1, out=vhpx_l1u1)
        cp.matmul(vh_l1, pl_l1, out=vhpx_l1l1)
        cp.matmul(vhpx_d1d1, vh_d1, out=vhpxvh_d1d1d1)
        cp.matmul(vhpx_d1d1, vh_u1, out=vhpxvh_d1d1u1)
        cp.matmul(vhpx_d1d1, vh_l1, out=vhpxvh_d1d1l1)
        cp.matmul(vhpx_d1u1, vh_d1, out=vhpxvh_d1u1d1)
        cp.matmul(vhpx_d1u1, vh_u1, out=vhpxvh_d1u1u1)
        cp.matmul(vhpx_d1u1, vh_l1, out=vhpxvh_d1u1l1)
        cp.matmul(vhpx_d1l1, vh_d1, out=vhpxvh_d1l1d1)
        cp.matmul(vhpx_d1l1, vh_u1, out=vhpxvh_d1l1u1)
        cp.matmul(vhpx_u1d1, vh_d1, out=vhpxvh_u1d1d1)
        cp.matmul(vhpx_u1d1, vh_u1, out=vhpxvh_u1d1u1)
        cp.matmul(vhpx_u1d1, vh_l1, out=vhpxvh_u1d1l1)
        cp.matmul(vhpx_u1u1, vh_d1, out=vhpxvh_u1u1d1)
        cp.matmul(vhpx_u1u1, vh_u1, out=vhpxvh_u1u1u1)
        cp.matmul(vhpx_u1u1, vh_l1, out=vhpxvh_u1u1l1)
        cp.matmul(vhpx_u1l1, vh_d1, out=vhpxvh_u1l1d1)
        cp.matmul(vhpx_u1l1, vh_u1, out=vhpxvh_u1l1u1)
        cp.matmul(vhpx_u1l1, vh_l1, out=vhpxvh_u1l1l1)
        cp.matmul(vhpx_l1d1, vh_d1, out=vhpxvh_l1d1d1)
        cp.matmul(vhpx_l1d1, vh_u1, out=vhpxvh_l1d1u1)
        cp.matmul(vhpx_l1u1, vh_d1, out=vhpxvh_l1u1d1)
        cp.matmul(vhpx_l1u1, vh_u1, out=vhpxvh_l1u1u1)
        cp.matmul(vhpx_l1u1, vh_l1, out=vhpxvh_l1u1l1)
        cp.matmul(vhpx_l1l1, vh_u1, out=vhpxvh_l1l1u1)

        # L^{\less}\left(E\right)
        ll_d2[:, :lb, :
              lb] = vhpxvh_d1l1u1 + vhpxvh_l1d1u1 + vhpxvh_l1u1d1 + vhpxvh_d1d1d1 + vhpxvh_u1l1d1 + vhpxvh_d1u1l1 + vhpxvh_u1d1l1
        ll_d2[:, :lb, lb:] = vhpxvh_l1u1u1 + vhpxvh_u1u1l1 + vhpxvh_d1d1u1 + vhpxvh_u1l1u1 + vhpxvh_d1u1d1 + vhpxvh_u1d1d1
        ll_d2[:, lb:, :lb] = vhpxvh_l1l1u1 + vhpxvh_u1l1l1 + vhpxvh_d1d1l1 + vhpxvh_l1u1l1 + vhpxvh_d1l1d1 + vhpxvh_l1d1d1
        ll_d2[:,
            lb:,
            lb:] = vhpxvh_d1u1l1 + vhpxvh_u1d1l1 + vhpxvh_u1l1d1 + vhpxvh_d1d1d1 + vhpxvh_l1u1d1 + vhpxvh_d1l1u1 + vhpxvh_l1d1u1

        ll_u2[:, :lb, :lb] = vhpxvh_u1u1d1 + vhpxvh_d1u1u1 + vhpxvh_u1d1u1
        ll_u2[:, :lb, lb:] = vhpxvh_u1u1u1
        ll_u2[:, lb:, :lb] = vhpxvh_d1u1d1 + vhpxvh_u1d1d1 + vhpxvh_u1l1u1 + vhpxvh_u1u1l1 + vhpxvh_d1d1u1 + vhpxvh_l1u1u1
        ll_u2[:, lb:, lb:] = vhpxvh_d1u1u1 + vhpxvh_u1d1u1 + vhpxvh_u1u1d1

        dll_lu[:, :lb, :lb] = vhpxvh_d1l1u1 + vhpxvh_l1d1u1 + vhpxvh_l1u1d1
        dll_lu[:, :lb, lb:] = vhpxvh_l1u1u1
        dll_lu[:, lb:, :lb] = vhpxvh_l1l1u1

        dll_ul[:, :lb, lb:] = vhpxvh_u1u1l1
        dll_ul[:, lb:, :lb] = vhpxvh_u1l1l1
        dll_ul[:, lb:, lb:] = vhpxvh_d1u1l1 + vhpxvh_u1d1l1 + vhpxvh_u1l1d1

        vh_u[:, lb:, :lb] = vh_u1
        vh_l[:, :lb, lb:] = vh_l1
    elif nbc == 3:
        # compute multiplications
        vhpr_d1d1 = -vh_d1 @ pr_d1
        vhpr_d1u1 = -vh_d1 @ pr_u1
        vhpr_d1l1 = -vh_d1 @ pr_l1
        vhpr_u1d1 = -vh_u1 @ pr_d1
        vhpr_u1u1 = -vh_u1 @ pr_u1
        vhpr_u1l1 = -vh_u1 @ pr_l1
        vhpr_l1d1 = -vh_l1 @ pr_d1
        vhpr_l1u1 = -vh_l1 @ pr_u1
        vhpr_l1l1 = -vh_l1 @ pr_l1

        vhpg_d1d1 = vh_d1 @ pg_d1
        vhpg_d1u1 = vh_d1 @ pg_u1
        vhpg_d1l1 = vh_d1 @ pg_l1
        vhpg_u1d1 = vh_u1 @ pg_d1
        vhpg_u1u1 = vh_u1 @ pg_u1
        vhpg_u1l1 = vh_u1 @ pg_l1
        vhpg_l1d1 = vh_l1 @ pg_d1
        vhpg_l1u1 = vh_l1 @ pg_u1
        vhpg_l1l1 = vh_l1 @ pg_l1
        # vhpgvh_d1d1d1 = vhpl_d1d1 @ vh_d1
        # vhpgvh_d1d1u1 = vhpl_d1d1 @ vh_u1
        # vhpgvh_d1d1l1 = vhpl_d1d1 @ vh_l1
        # vhpgvh_d1u1d1 = vhpl_d1u1 @ vh_d1
        # vhpgvh_d1u1u1 = vhpl_d1u1 @ vh_u1
        # vhpgvh_d1u1l1 = vhpl_d1u1 @ vh_l1
        # vhpgvh_d1l1d1 = vhpl_d1l1 @ vh_d1
        # vhpgvh_d1l1u1 = vhpl_d1l1 @ vh_u1
        # vhpgvh_d1l1l1 = vhpl_d1l1 @ vh_l1
        # vhpgvh_u1d1d1 = vhpl_u1d1 @ vh_d1
        # vhpgvh_u1d1u1 = vhpl_u1d1 @ vh_u1
        # vhpgvh_u1d1l1 = vhpl_u1d1 @ vh_l1
        # vhpgvh_u1u1d1 = vhpl_u1u1 @ vh_d1
        # vhpgvh_u1u1u1 = vhpl_u1u1 @ vh_u1
        # vhpgvh_u1u1l1 = vhpl_u1u1 @ vh_l1
        # vhpgvh_u1l1d1 = vhpl_u1l1 @ vh_d1
        # vhpgvh_u1l1u1 = vhpl_u1l1 @ vh_u1
        # vhpgvh_u1l1l1 = vhpl_u1l1 @ vh_l1
        # vhpgvh_l1d1d1 = vhpl_l1d1 @ vh_d1
        # vhpgvh_l1d1u1 = vhpl_l1d1 @ vh_u1
        # vhpgvh_l1d1l1 = vhpl_l1d1 @ vh_l1
        # vhpgvh_l1u1d1 = vhpl_l1u1 @ vh_d1
        # vhpgvh_l1u1u1 = vhpl_l1u1 @ vh_u1
        # vhpgvh_l1u1l1 = vhpl_l1u1 @ vh_l1
        # vhpgvh_l1l1d1 = vhpl_l1l1 @ vh_d1
        # vhpgvh_l1l1u1 = vhpl_l1l1 @ vh_u1
        # Replacing above block with below block
        vhpgvh_d1d1d1 = vhpg_d1d1 @ vh_d1
        vhpgvh_d1d1u1 = vhpg_d1d1 @ vh_u1
        vhpgvh_d1d1l1 = vhpg_d1d1 @ vh_l1
        vhpgvh_d1u1d1 = vhpg_d1u1 @ vh_d1
        vhpgvh_d1u1u1 = vhpg_d1u1 @ vh_u1
        vhpgvh_d1u1l1 = vhpg_d1u1 @ vh_l1
        vhpgvh_d1l1d1 = vhpg_d1l1 @ vh_d1
        vhpgvh_d1l1u1 = vhpg_d1l1 @ vh_u1
        vhpgvh_u1d1d1 = vhpg_u1d1 @ vh_d1
        vhpgvh_u1d1u1 = vhpg_u1d1 @ vh_u1
        vhpgvh_u1d1l1 = vhpg_u1d1 @ vh_l1
        vhpgvh_u1u1d1 = vhpg_u1u1 @ vh_d1
        vhpgvh_u1u1u1 = vhpg_u1u1 @ vh_u1
        vhpgvh_u1u1l1 = vhpg_u1u1 @ vh_l1
        vhpgvh_u1l1d1 = vhpg_u1l1 @ vh_d1
        vhpgvh_u1l1u1 = vhpg_u1l1 @ vh_u1
        vhpgvh_u1l1l1 = vhpg_u1l1 @ vh_l1
        vhpgvh_l1d1d1 = vhpg_l1d1 @ vh_d1
        vhpgvh_l1d1u1 = vhpg_l1d1 @ vh_u1
        vhpgvh_l1u1d1 = vhpg_l1u1 @ vh_d1
        vhpgvh_l1u1u1 = vhpg_l1u1 @ vh_u1
        vhpgvh_l1u1l1 = vhpg_l1u1 @ vh_l1
        vhpgvh_l1l1u1 = vhpg_l1l1 @ vh_u1
        vhpgvh_l1l1d1 = vhpg_l1l1 @ vh_d1
        vhpgvh_d1l1l1 = vhpg_d1l1 @ vh_l1
        vhpgvh_l1d1l1 = vhpg_l1d1 @ vh_l1
        vhpl_d1d1 = vh_d1 @ pl_d1
        vhpl_d1u1 = vh_d1 @ pl_u1
        vhpl_d1l1 = vh_d1 @ pl_l1
        vhpl_u1d1 = vh_u1 @ pl_d1
        vhpl_u1u1 = vh_u1 @ pl_u1
        vhpl_u1l1 = vh_u1 @ pl_l1
        vhpl_l1d1 = vh_l1 @ pl_d1
        vhpl_l1u1 = vh_l1 @ pl_u1
        vhpl_l1l1 = vh_l1 @ pl_l1
        vhplvh_d1d1d1 = vhpl_d1d1 @ vh_d1
        vhplvh_d1d1u1 = vhpl_d1d1 @ vh_u1
        vhplvh_d1d1l1 = vhpl_d1d1 @ vh_l1
        vhplvh_d1u1d1 = vhpl_d1u1 @ vh_d1
        vhplvh_d1u1u1 = vhpl_d1u1 @ vh_u1
        vhplvh_d1u1l1 = vhpl_d1u1 @ vh_l1
        vhplvh_d1l1d1 = vhpl_d1l1 @ vh_d1
        vhplvh_d1l1u1 = vhpl_d1l1 @ vh_u1
        vhplvh_d1l1l1 = vhpl_d1l1 @ vh_l1
        vhplvh_u1d1d1 = vhpl_u1d1 @ vh_d1
        vhplvh_u1d1u1 = vhpl_u1d1 @ vh_u1
        vhplvh_u1d1l1 = vhpl_u1d1 @ vh_l1
        vhplvh_u1u1d1 = vhpl_u1u1 @ vh_d1
        vhplvh_u1u1u1 = vhpl_u1u1 @ vh_u1
        vhplvh_u1u1l1 = vhpl_u1u1 @ vh_l1
        vhplvh_u1l1d1 = vhpl_u1l1 @ vh_d1
        vhplvh_u1l1u1 = vhpl_u1l1 @ vh_u1
        vhplvh_u1l1l1 = vhpl_u1l1 @ vh_l1
        vhplvh_l1d1d1 = vhpl_l1d1 @ vh_d1
        vhplvh_l1d1u1 = vhpl_l1d1 @ vh_u1
        vhplvh_l1d1l1 = vhpl_l1d1 @ vh_l1
        vhplvh_l1u1d1 = vhpl_l1u1 @ vh_d1
        vhplvh_l1u1u1 = vhpl_l1u1 @ vh_u1
        vhplvh_l1u1l1 = vhpl_l1u1 @ vh_l1
        vhplvh_l1l1d1 = vhpl_l1l1 @ vh_d1
        vhplvh_l1l1u1 = vhpl_l1l1 @ vh_u1

        # fill output matrices
        # M^{r}\left(E\right)
        mr_d2[:, :lb, :lb] = vhpr_d1d1 + vhpr_l1u1 + vhpr_u1l1
        mr_d2[:, :lb, lb:2 * lb] = vhpr_d1u1 + vhpr_u1d1
        mr_d2[:, :lb, 2 * lb:] = vhpr_u1u1
        mr_d2[:, lb:2 * lb, :lb] = vhpr_d1l1 + vhpr_l1d1
        mr_d2[:, lb:2 * lb, lb:2 * lb] = vhpr_d1d1 + vhpr_l1u1 + vhpr_u1l1
        mr_d2[:, lb:2 * lb, 2 * lb:] = vhpr_d1u1 + vhpr_u1d1
        mr_d2[:, 2 * lb:, :lb] = vhpr_l1l1
        mr_d2[:, 2 * lb:, lb:2 * lb] = vhpr_d1l1 + vhpr_l1d1
        mr_d2[:, 2 * lb:, 2 * lb:] = vhpr_d1d1 + vhpr_l1u1 + vhpr_u1l1

        mr_u2[:, lb:2 * lb, :lb] = vhpr_u1u1
        mr_u2[:, 2 * lb:, :lb] = vhpr_d1u1 + vhpr_u1d1
        mr_u2[:, 2 * lb:, lb:2 * lb] = vhpr_u1u1

        mr_l2[:, :lb, lb:2 * lb] = vhpr_l1l1
        mr_l2[:, :lb, 2 * lb:] = vhpr_d1l1 + vhpr_l1d1
        mr_l2[:, lb:2 * lb, 2 * lb:] = vhpr_l1l1

        lg_d2[:, :lb, :
              lb] = vhpgvh_d1l1u1 + vhpgvh_l1d1u1 + vhpgvh_l1u1d1 + vhpgvh_d1d1d1 + vhpgvh_u1l1d1 + vhpgvh_d1u1l1 + vhpgvh_u1d1l1
        lg_d2[:, :lb,
              lb:2 * lb] = vhpgvh_l1u1u1 + vhpgvh_u1u1l1 + vhpgvh_d1d1u1 + vhpgvh_u1l1u1 + vhpgvh_d1u1d1 + vhpgvh_u1d1d1
        lg_d2[:, :lb, 2 * lb:] = vhpgvh_u1u1d1 + vhpgvh_d1u1u1 + vhpgvh_u1d1u1
        lg_d2[:, lb:2 *
              lb, :lb] = vhpgvh_l1l1u1 + vhpgvh_d1l1d1 + vhpgvh_l1d1d1 + vhpgvh_d1d1l1 + vhpgvh_l1u1l1 + vhpgvh_u1l1l1
        lg_d2[:,
            lb:2 * lb, lb:2 *
            lb] = vhpgvh_d1l1u1 + vhpgvh_l1d1u1 + vhpgvh_d1u1l1 + vhpgvh_u1d1l1 + vhpgvh_d1d1d1 + vhpgvh_l1u1d1 + vhpgvh_u1l1d1
        lg_d2[:, lb:2 * lb,
              2 * lb:] = vhpgvh_u1u1l1 + vhpgvh_d1u1d1 + vhpgvh_u1d1d1 + vhpgvh_d1d1u1 + vhpgvh_l1u1u1 + vhpgvh_u1l1u1
        lg_d2[:, 2 * lb:, :lb] = vhpgvh_l1l1d1 + vhpgvh_d1l1l1 + vhpgvh_l1d1l1
        lg_d2[:, 2 * lb:,
              lb:2 * lb] = vhpgvh_l1l1u1 + vhpgvh_u1l1l1 + vhpgvh_d1d1l1 + vhpgvh_l1u1l1 + vhpgvh_d1l1d1 + vhpgvh_l1d1d1
        lg_d2[:,
            2 * lb:, 2 *
            lb:] = vhpgvh_d1u1l1 + vhpgvh_u1d1l1 + vhpgvh_u1l1d1 + vhpgvh_d1d1d1 + vhpgvh_l1u1d1 + vhpgvh_d1l1u1 + vhpgvh_l1d1u1

        lg_u2[:, :lb, :lb] = vhpgvh_u1u1u1
        lg_u2[:, lb:2 * lb, :lb] = vhpgvh_u1u1d1 + vhpgvh_d1u1u1 + vhpgvh_u1d1u1
        lg_u2[:, lb:2 * lb, lb:2 * lb] = vhpgvh_u1u1u1
        lg_u2[:, 2 *
              lb:, :lb] = vhpgvh_d1u1d1 + vhpgvh_u1d1d1 + vhpgvh_u1l1u1 + vhpgvh_u1u1l1 + vhpgvh_d1d1u1 + vhpgvh_l1u1u1
        lg_u2[:, 2 * lb:, lb:2 * lb] = vhpgvh_d1u1u1 + vhpgvh_u1d1u1 + vhpgvh_u1u1d1
        lg_u2[:, 2 * lb:, 2 * lb:] = vhpgvh_u1u1u1

        ll_d2[:, :lb, :
              lb] = vhplvh_d1l1u1 + vhplvh_l1d1u1 + vhplvh_l1u1d1 + vhplvh_d1d1d1 + vhplvh_u1l1d1 + vhplvh_d1u1l1 + vhplvh_u1d1l1
        ll_d2[:, :lb,
              lb:2 * lb] = vhplvh_l1u1u1 + vhplvh_u1u1l1 + vhplvh_d1d1u1 + vhplvh_u1l1u1 + vhplvh_d1u1d1 + vhplvh_u1d1d1
        ll_d2[:, :lb, 2 * lb:] = vhplvh_u1u1d1 + vhplvh_d1u1u1 + vhplvh_u1d1u1
        ll_d2[:, lb:2 *
              lb, :lb] = vhplvh_l1l1u1 + vhplvh_d1l1d1 + vhplvh_l1d1d1 + vhplvh_d1d1l1 + vhplvh_l1u1l1 + vhplvh_u1l1l1
        ll_d2[:,
            lb:2 * lb, lb:2 *
            lb] = vhplvh_d1l1u1 + vhplvh_l1d1u1 + vhplvh_d1u1l1 + vhplvh_u1d1l1 + vhplvh_d1d1d1 + vhplvh_l1u1d1 + vhplvh_u1l1d1
        ll_d2[:, lb:2 * lb,
              2 * lb:] = vhplvh_u1u1l1 + vhplvh_d1u1d1 + vhplvh_u1d1d1 + vhplvh_d1d1u1 + vhplvh_l1u1u1 + vhplvh_u1l1u1
        ll_d2[:, 2 * lb:, :lb] = vhplvh_l1l1d1 + vhplvh_d1l1l1 + vhplvh_l1d1l1
        ll_d2[:, 2 * lb:,
              lb:2 * lb] = vhplvh_l1l1u1 + vhplvh_u1l1l1 + vhplvh_d1d1l1 + vhplvh_l1u1l1 + vhplvh_d1l1d1 + vhplvh_l1d1d1
        ll_d2[:,
            2 * lb:, 2 *
            lb:] = vhplvh_d1u1l1 + vhplvh_u1d1l1 + vhplvh_u1l1d1 + vhplvh_d1d1d1 + vhplvh_l1u1d1 + vhplvh_d1l1u1 + vhplvh_l1d1u1

        ll_u2[:, :lb, :lb] = vhplvh_u1u1u1
        ll_u2[:, lb:2 * lb, :lb] = vhplvh_u1u1d1 + vhplvh_d1u1u1 + vhplvh_u1d1u1
        ll_u2[:, lb:2 * lb, lb:2 * lb] = vhplvh_u1u1u1
        ll_u2[:, 2 *
              lb:, :lb] = vhplvh_d1u1d1 + vhplvh_u1d1d1 + vhplvh_u1l1u1 + vhplvh_u1u1l1 + vhplvh_d1d1u1 + vhplvh_l1u1u1
        ll_u2[:, 2 * lb:, lb:2 * lb] = vhplvh_d1u1u1 + vhplvh_u1d1u1 + vhplvh_u1u1d1
        ll_u2[:, 2 * lb:, 2 * lb:] = vhplvh_u1u1u1

        dmr_lu[:, :lb, :lb] = vhpr_l1u1

        dmr_ul[:, 2 * lb:, 2 * lb:] = vhpr_u1l1

        dlg_lu[:, :lb, :lb] = vhpgvh_d1l1u1 + vhpgvh_l1d1u1 + vhpgvh_l1u1d1
        dlg_lu[:, :lb, lb:2 * lb] = vhpgvh_l1u1u1
        dlg_lu[:, lb:2 * lb, :lb] = vhpgvh_l1l1u1

        dlg_ul[:, lb:2 * lb, 2 * lb:] = vhpgvh_u1u1l1
        dlg_ul[:, 2 * lb:, lb:2 * lb] = vhpgvh_u1l1l1
        dlg_ul[:, 2 * lb:, 2 * lb:] = vhpgvh_d1u1l1 + vhpgvh_u1d1l1 + vhpgvh_u1l1d1

        dll_lu[:, :lb, :lb] = vhplvh_d1l1u1 + vhplvh_l1d1u1 + vhplvh_l1u1d1
        dll_lu[:, :lb, lb:2 * lb] = vhplvh_l1u1u1
        dll_lu[:, lb:2 * lb, :lb] = vhplvh_l1l1u1

        dll_ul[:, lb:2 * lb, 2 * lb:] = vhplvh_u1u1l1
        dll_ul[:, 2 * lb:, lb:2 * lb] = vhplvh_u1l1l1
        dll_ul[:, 2 * lb:, 2 * lb:] = vhplvh_d1u1l1 + vhplvh_u1d1l1 + vhplvh_u1l1d1

        vh_u[:, 2 * lb:, :lb] = vh_u1 # change

        vh_l[:, :lb, 2 * lb:] = vh_l1 # change


    # lg_l2[:] = -lg_u2.transpose(0, 2, 1).conjugate()
    # ll_l2[:] = -ll_u2.transpose(0, 2, 1).conjugate()
    lg_l2 = -lg_u2.transpose(0, 2, 1).conjugate()
    ll_l2 = -ll_u2.transpose(0, 2, 1).conjugate()
    mr_d2 = mr_d2 + cp.repeat(cp.identity(lb_mm, dtype=cp.complex128)[cp.newaxis, :, :] * (1 + 1j * 1e-10), batch_size, axis=0)
    (mr_d2, mr_u2, mr_l2, matrix_blocks ) = extract_small_matrix_blocks_batched_gpu(mr_d2, mr_u2, mr_l2, NCpSC*nbc, side)

    return ((mr_d2, mr_u2, mr_l2), (lg_d2, lg_u2, lg_l2), (ll_d2, ll_u2, ll_l2), (dmr_lu, dmr_ul), (dlg_lu, dlg_ul),
            (dll_lu, dll_ul), (vh_u, vh_l), matrix_blocks)
