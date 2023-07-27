# Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.
""" Contains functions for calculating the different correction terms for the calculation of the screened interaction.
    Naming is correction_ + from where the correction term is read from. Has the same function as get_OBC_blocks.m,
    but split up for readability. """

import cupy as cp
import typing


def get_mm_obc_dense(
    vh_1: cp.ndarray, vh_2: cp.ndarray, pg_1: cp.ndarray, pg_2: cp.ndarray, pl_1: cp.ndarray, pl_2: cp.ndarray,
    pr_1: cp.ndarray, pr_2: cp.ndarray, nbc: int
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
    vh_d1 = vh_1
    vh_u1 = vh_2
    vh_l1 = vh_u1.conjugate().transpose()
    pg_d1 = pg_1
    pg_u1 = pg_2
    pg_l1 = -pg_u1.conjugate().transpose()
    pl_d1 = pl_1
    pl_u1 = pl_2
    pl_l1 = -pl_u1.conjugate().transpose()
    pr_d1 = pr_1
    pr_u1 = pr_2
    pr_l1 = pr_u1.transpose()

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
        vhpgvh_d1d1d1 = vhpl_d1d1 @ vh_d1
        vhpgvh_d1d1u1 = vhpl_d1d1 @ vh_u1
        vhpgvh_d1d1l1 = vhpl_d1d1 @ vh_l1
        vhpgvh_d1u1d1 = vhpl_d1u1 @ vh_d1
        vhpgvh_d1u1u1 = vhpl_d1u1 @ vh_u1
        vhpgvh_d1u1l1 = vhpl_d1u1 @ vh_l1
        vhpgvh_d1l1d1 = vhpl_d1l1 @ vh_d1
        vhpgvh_d1l1u1 = vhpl_d1l1 @ vh_u1
        vhpgvh_d1l1l1 = vhpl_d1l1 @ vh_l1
        vhpgvh_u1d1d1 = vhpl_u1d1 @ vh_d1
        vhpgvh_u1d1u1 = vhpl_u1d1 @ vh_u1
        vhpgvh_u1d1l1 = vhpl_u1d1 @ vh_l1
        vhpgvh_u1u1d1 = vhpl_u1u1 @ vh_d1
        vhpgvh_u1u1u1 = vhpl_u1u1 @ vh_u1
        vhpgvh_u1u1l1 = vhpl_u1u1 @ vh_l1
        vhpgvh_u1l1d1 = vhpl_u1l1 @ vh_d1
        vhpgvh_u1l1u1 = vhpl_u1l1 @ vh_u1
        vhpgvh_u1l1l1 = vhpl_u1l1 @ vh_l1
        vhpgvh_l1d1d1 = vhpl_l1d1 @ vh_d1
        vhpgvh_l1d1u1 = vhpl_l1d1 @ vh_u1
        vhpgvh_l1d1l1 = vhpl_l1d1 @ vh_l1
        vhpgvh_l1u1d1 = vhpl_l1u1 @ vh_d1
        vhpgvh_l1u1u1 = vhpl_l1u1 @ vh_u1
        vhpgvh_l1u1l1 = vhpl_l1u1 @ vh_l1
        vhpgvh_l1l1d1 = vhpl_l1l1 @ vh_d1
        vhpgvh_l1l1u1 = vhpl_l1l1 @ vh_u1
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
        lg_d2[:lb, 2 * lb:] = vhpgvh_u1u1d1 + vhpgvh_d1u1d1 + vhpgvh_u1d1u1
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
        lg_u2[lb:2 * lb, :lb] = vhpgvh_u1u1d1 + vhpgvh_d1u1d1 + vhpgvh_u1d1u1
        lg_u2[lb:2 * lb, lb:2 * lb] = vhpgvh_u1u1u1
        lg_u2[2 *
              lb:, :lb] = vhpgvh_d1u1d1 + vhpgvh_u1d1d1 + vhpgvh_u1l1u1 + vhpgvh_u1u1l1 + vhpgvh_d1d1u1 + vhpgvh_l1u1u1
        lg_u2[2 * lb:, lb:2 * lb] = vhpgvh_d1u1d1 + vhpgvh_u1d1u1 + vhpgvh_u1u1d1
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

        vh_u[2 * lb:, :lb] = vh_u

        vh_l[:lb, 2 * lb:] = vh_l

    lg_l2[:, :] = -lg_u2.conjugate().T
    ll_l2[:, :] = -ll_u2.conjugate().T
    mr_d2 = mr_d2 + cp.identity(lb_mm, dtype=cp.complex128) * (1 + 1j * 1e-10)

    return ((mr_d2, mr_u2, mr_l2), (lg_d2, lg_u2, lg_l2), (ll_d2, ll_u2, ll_l2), (dmr_lu, dmr_ul), (dlg_lu, dlg_ul),
            (dll_lu, dll_ul), (vh_u, vh_l))
