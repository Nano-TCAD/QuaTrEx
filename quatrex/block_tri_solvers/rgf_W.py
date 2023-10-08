# Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.
"""
The rgf solver for the step from the polarization OP to the screened interaction W is
implemented.

The formula for the screened interaction is given as:

W^{r}\left(E\right) = \left[\mathbb{I} - \hat(V) P^{r}\left(E\right)\right]^{-1} \hat(V)

W^{\lessgtr}\left(E\right) = W^{r}\left(E\right) P^{\lessgtr}\left(E\right) W^{r}\left(E\right)^{H} 

All of the above formulas need corrections in practice
since we cutoff the semi infinite boundary blocks.
The corrections are needed for both matrix inverse and matrix multiplication.

The beyn algorithm is used for correction of the inverse.
If beyn fails, it falls back to sancho rubio.

Only diagonal and upper diagonal blocks are calculated of the inverse.
The fast RGF algorithm can be used for the inverse 
(Certain assumptions are made as the true inverse would be full, see the original RGF paper).

Additional quantities will be defined:

- S^{r}\left(E\right) = \hat(V) P^{r}\left(E\right) (helper function to split up calculation of retarded screening)

- M^{r}\left(E\right) = \mathbb{I} - S^{r}\left(E\right) (helper function to split up calculation of retarded screening)

- W^{r}\left(E\right) = \left[M^{r}\left(E\right)\right]^{-1} (helper function to split up calculation of retarded screening)

- W^{>}_{rgf}\left(E\right), W^{>}_{rgf}\left(E\right), W^{r}_{rgf}\left(E\right) (see RGF paper, partial inverses)
  (Standard naming is to call them just w^{>}, w^{<}, w^{r} as small characters)

- L^{\lessgtr}\left(E\right) = \hat(V) P^{\lessgtr}\left(E\right) \hat(V)^{H} (helper function to split up calculation of lesser/greater screening)


Notes about variable naming:

- _ct is for the original matrix complex conjugated
- _mm how certain sizes after matrix multiplication, because a block tri diagonal gets two new non zero off diagonals
- _s stands for values related to the left/start/top contact block
- _e stands for values related to the right/end/bottom contact block
- _d stands for diagonal block (X00/NN in matlab)
- _u stands for upper diagonal block (X01/NN1 in matlab)
- _l stands for lower diagonal block (X10/N1N in matlab) 
- exception _l/_r can stand for left/right in context of condition of OBC
- _rgf are the not true inverse tensor (partial inverse) 
more standard notation would be small characters for partial and large for true
but this goes against python naming style guide
"""
import numpy as np
import numpy.typing as npt
from scipy import sparse
import typing

from quatrex.OBC import beyn_cpu
from quatrex.OBC import sancho
from quatrex.OBC import dL_OBC_eigenmode_cpu


def rgf_w(
    vh: sparse.csr_matrix,
    pg: sparse.csr_matrix,
    pl: sparse.csr_matrix,
    pr: sparse.csr_matrix,
    bmax: np.ndarray[np.int32],
    bmin: np.ndarray[np.int32],
    wg_diag: np.ndarray[np.complex128],
    wg_upper: np.ndarray[np.complex128],
    wl_diag: np.ndarray[np.complex128],
    wl_upper: np.ndarray[np.complex128],
    wr_diag: np.ndarray[np.complex128],
    wr_upper: np.ndarray[np.complex128],
    xr_diag: np.ndarray[np.complex128],
    dosw: np.ndarray[np.complex128],
    nEw: np.ndarray[np.complex128],
    nPw: np.ndarray[np.complex128],
    nbc: np.int64,
    ie: np.int32,
    factor: np.float64 = 1.0,
    sancho_flag: bool = False
):
    """Calculates the step from the polarization to the screened interaction.
    Beyn open boundary conditions are used by default.
    The outputs (w and x) are inputs which are changed inplace.
    See the start of this file for more informations.

    Args:
        vh (sparse.csr_matrix): sparse matrix of size (#orbitals,#orbitals)
        pg (sparse.csr_matrix): sparse matrix of size (#orbitals,#orbitals)
        pl (sparse.csr_matrix): sparse matrix of size (#orbitals,#orbitals)
        pr (sparse.csr_matrix): sparse matrix of size (#orbitals,#orbitals)
        bmax (np.ndarray[np.int32]): end idx of the blocks, vector of size number of blocks
        bmin (np.ndarray[np.int32]): start idx of the blocks, vector of size number of blocks
        wg_diag (np.ndarray[np.complex128]): dense matrix of size (#blocks_mm, maxblocklength_mm, maxblocklength_mm)
        wg_upper (np.ndarray[np.complex128]): dense matrix of size (#blocks_mm-1, maxblocklength_mm, maxblocklength_mm)
        wl_diag (np.ndarray[np.complex128]): dense matrix of size (#blocks_mm, maxblocklength_mm, maxblocklength_mm)
        wl_upper (np.ndarray[np.complex128]): dense matrix of size (#blocks_mm-1, maxblocklength_mm, maxblocklength_mm)
        wr_diag (np.ndarray[np.complex128]): dense matrix of size (#blocks_mm, maxblocklength_mm, maxblocklength_mm)
        wr_upper (np.ndarray[np.complex128]): dense matrix of size (#blocks_mm-1, maxblocklength_mm, maxblocklength_mm)
        xr_diag (np.ndarray[np.complex128]): dense matrix of size (#blocks_mm, maxblocklength_mm, maxblocklength_mm)
        dosw (np.ndarray[np.complex128]): density of states, vector of size number of blocks
        nbc (np.int64): how block size changes after matrix multiplication
        ie (np.int32): energy index (not used)
        factor (np.float64, optional): factor to multiply the result with. Defaults to 1.0.
        ref_flag (bool, optional): If reference solution to rgf made by np.linalg.inv should be returned
        sancho_flag (bool, optional): If sancho or beyn should be used. Defaults to False.
    """

    # limit for beyn
    imag_lim = 1e-4
    # todo find out what rr/R is
    # (R only is against the style guide)
    # todo, compare with matlab
    rr = 1e6
    # copy vh to overwrite it
    vh_cp = vh.copy()

    # number of blocks
    nb = bmin.size
    # number of total orbitals (number of atomic orbitals)
    nao = pr.shape[0]

    bmax_mm = bmax[nbc - 1:nb:nbc]
    bmin_mm = bmin[0:nb:nbc]
    blocksize = bmax[0] - bmin[0] + 1
    blocksize_after_mm = bmax_mm[0] - bmin_mm[0] + 1

    number_of_blocks_after_mm = bmax_mm.size

    lb_vec_mm = bmax_mm - bmin_mm + 1


    # calculate L^{\lessgtr}\left(E\right)
    vh_cp_ct = vh_cp.conjugate().transpose()
    lg = vh_cp @ pg @ vh_cp_ct
    ll = vh_cp @ pl @ vh_cp_ct

    # calculate M^{r}\left(E\right)
    mr = sparse.identity(nao, format="csr") - vh_cp @ pr



    # from G^{\lessgtr}\left(E\right) / G^{r}\left(E\right)
    # (diagonal, upper, lower block of the start block at the left)
    pg_sd, pg_su, pg_sl = dL_OBC_eigenmode_cpu.stack_px(pg[:blocksize, :blocksize], pg[:blocksize, blocksize:2*blocksize], nbc)
    pl_sd, pl_su, pl_sl = dL_OBC_eigenmode_cpu.stack_px(pl[:blocksize, :blocksize], pl[:blocksize, blocksize:2*blocksize], nbc)
    pr_sd, pr_su, pr_sl = dL_OBC_eigenmode_cpu.stack_pr(pr[:blocksize, :blocksize], pr[:blocksize, blocksize:2*blocksize], nbc)
    # (diagonal, upper, lower block of the end block at the right)
    pg_ed, pg_eu, pg_el = dL_OBC_eigenmode_cpu.stack_px(pg[-blocksize:, -blocksize:], -pg[-blocksize:, -2*blocksize:-blocksize].conjugate().transpose(),
                                                        nbc)
    pl_ed, pl_eu, pl_el = dL_OBC_eigenmode_cpu.stack_px(pl[-blocksize:, -blocksize:], -pl[-blocksize:, -2*blocksize:-blocksize].conjugate().transpose(),
                                                        nbc)
    pr_ed, pr_eu, pr_el = dL_OBC_eigenmode_cpu.stack_pr(pr[-blocksize:, -blocksize:], pr[-blocksize:, -2*blocksize:-blocksize].transpose(), nbc)

    # from \hat(V)\left(E\right) / G^{r}\left(E\right)
    # (diagonal, upper, lower block of the start block at the left)
    vh_sd, vh_su, vh_sl = dL_OBC_eigenmode_cpu.stack_vh(vh[:blocksize, :blocksize], vh[:blocksize, blocksize:2*blocksize], nbc)
    # (diagonal, upper, lower block of the end block at the right)
    vh_ed, vh_eu, vh_el = dL_OBC_eigenmode_cpu.stack_vh(vh[-blocksize:, -blocksize:], vh[-blocksize:, -2*blocksize:-blocksize].conjugate().transpose(),
                                                        nbc)

    # from L^{\lessgtr}\left(E\right)
    # (diagonal, upper, lower block of the start block at the left)
    lg_sd, lg_su, _ = dL_OBC_eigenmode_cpu.stack_lx(vh_sd, vh_su, vh_sl, pg_sd, pg_su, pg_sl)
    ll_sd, ll_su, _ = dL_OBC_eigenmode_cpu.stack_lx(vh_sd, vh_su, vh_sl, pl_sd, pl_su, pl_sl)

    # (diagonal, upper, lower block of the end block at the right)
    lg_ed, _, lg_el = dL_OBC_eigenmode_cpu.stack_lx(vh_ed, vh_eu, vh_el, pg_ed, pg_eu, pg_el)
    ll_ed, _, ll_el = dL_OBC_eigenmode_cpu.stack_lx(vh_ed, vh_eu, vh_el, pl_ed, pl_eu, pl_el)

    # from M^{r}\left(E\right)
    # (diagonal, upper, lower block of the start block at the left)
    mr_sd, mr_su, mr_sl = dL_OBC_eigenmode_cpu.stack_mr(vh_sd, vh_su, vh_sl, pr_sd, pr_su, pr_sl)
    # (diagonal, upper, lower block of the end block at the right)
    mr_ed, mr_eu, mr_el = dL_OBC_eigenmode_cpu.stack_mr(vh_ed, vh_eu, vh_el, pr_ed, pr_eu, pr_el)


    # correct first and last block to account for the contacts in multiplication
    dmr_sd = -vh_sl @ pr_su
    dmr_ed = -vh_eu @ pr_el

    # correct first and last block to account for the contacts in multiplication
    # first block
    # L^{\lessgtr}_E_00 = V_10*P^{\lessgtr}_E_00*V_01 + V_10*P^{\lessgtr}_E_01*V_00 + V_00*P^{\lessgtr}_E_10*V_01
    dlg_sd = (vh_sl @ pg_sd @ vh_su + vh_sl @ pg_su @ vh_sd + vh_sd @ pg_sl @ vh_su).toarray()
    dll_sd = (vh_sl @ pl_sd @ vh_su + vh_sl @ pl_su @ vh_sd + vh_sd @ pl_sl @ vh_su).toarray()

    # last block
    # L^{\lessgtr}_E_nn = V_nn*PL_E_n-1n*V_nn-1 + V_n-1n*PL_E_nn-1*V_nn + V_n-1n*PL_E_nn*V_nn-1
    dlg_ed = (vh_ed @ pg_eu @ vh_el + vh_eu @ pg_el @ vh_ed + vh_eu @ pg_ed @ vh_el).toarray()
    dll_ed = (vh_ed @ pl_eu @ vh_el + vh_eu @ pl_el @ vh_ed + vh_eu @ pl_ed @ vh_el).toarray()


    # correction for the matrix inverse calculations----------------------------

    # conditions about convergence or meaningful results from
    # boundary correction calculations
    cond_l = 0.0
    cond_r = 0.0

    # correction for first block
    if not sancho_flag:
        _, cond_l, dxr_sd, dmr, _ = beyn_cpu.beyn(
                                                mr_sd.toarray(),
                                                mr_su.toarray(),
                                                mr_sl.toarray(),
                                                imag_lim, rr, "L", block=False)

        if not np.isnan(cond_l):
            dmr_sd -= dmr
            dvh_sd = mr_sl @ dxr_sd @ vh_su

    if np.isnan(cond_l) or sancho_flag:
        dxr_sd, dmr, dvh_sd, cond_l = sancho.open_boundary_conditions(
                                                mr_sd.toarray(),
                                                mr_sl.toarray(),
                                                mr_su.toarray(),
                                                vh_su.toarray()
                                                )
        dmr_sd -= dmr
    # correction for last block
    if not sancho_flag:
        _, cond_r, dxr_ed, dmr, _ = beyn_cpu.beyn(
                                                mr_ed.toarray(),
                                                mr_eu.toarray(),
                                                mr_el.toarray(),
                                                imag_lim, rr, "R", block=False)
        if not np.isnan(cond_r):
            dmr_ed -= dmr
            dvh_ed = mr_eu @ dxr_ed @ vh_el


    if np.isnan(cond_r) or sancho_flag:
        dxr_ed, dmr, dvh_ed, cond_r = sancho.open_boundary_conditions(
                                                mr_ed.toarray(),
                                                mr_eu.toarray(),
                                                mr_el.toarray(),
                                                vh_el.toarray()
                                                )
        dmr_ed -= dmr


    # beyn gave a meaningful result
    if not np.isnan(cond_l) and not np.isnan(cond_l):
        dlg, dll = dL_OBC_eigenmode_cpu.get_dl_obc_alt(
                                                dxr_sd,
                                                lg_sd.toarray(),
                                                lg_su.toarray(),
                                                ll_sd.toarray(),
                                                ll_su.toarray(),
                                                mr_sl.toarray(),
                                                blk="L")

        if np.isnan(dll).any():
            cond_l = np.nan
        else:
            dlg_sd += dlg
            dll_sd += dll

        dlg, dll = dL_OBC_eigenmode_cpu.get_dl_obc_alt(
                                                dxr_ed,
                                                lg_ed.toarray(),
                                                lg_el.toarray(),
                                                ll_ed.toarray(),
                                                ll_el.toarray(),
                                                mr_eu.toarray(),
                                                blk="R")

        if np.isnan(dll_ed).any():
            cond_r = np.nan
        else:
            dlg_ed += dlg
            dll_ed += dll


    # start of rgf_W------------------------------------------------------------

    # check if OBC did not fail
    if not np.isnan(cond_r) and not np.isnan(cond_l) and ie:

        # add OBC corrections to the start and end block
        mr[-blocksize_after_mm:, -blocksize_after_mm:] += dmr_ed
        lg[-blocksize_after_mm:, -blocksize_after_mm:] += dlg_ed
        ll[-blocksize_after_mm:, -blocksize_after_mm:] += dll_ed
        vh_cp[-blocksize_after_mm:, -blocksize_after_mm:] -= dvh_ed

        lg[:blocksize_after_mm, :blocksize_after_mm] += dlg_sd
        ll[:blocksize_after_mm, :blocksize_after_mm] += dll_sd
        vh_cp[:blocksize_after_mm, :blocksize_after_mm] -= dvh_sd
        mr[:blocksize_after_mm, :blocksize_after_mm] += dmr_sd



        # System_matrix_invert = np.linalg.inv(mr.toarray())
        # Screened_interaction_retarded = System_matrix_invert @ vh_cp
        # Screened_interaction_lesser = Screened_interaction_retarded @ ll @ Screened_interaction_retarded.conjugate().transpose()
        # Screened_interaction_greater = Screened_interaction_retarded @ lg @ Screened_interaction_retarded.conjugate().transpose()

        # # extract the diagonal and upper diagonal blocks from the screened interaction retarded, lesser and greater
        # # and store them in the corresponding arrays
        # for j in range(number_of_blocks_after_mm):
        #     wr_diag[j] = Screened_interaction_retarded[j * blocksize_after_mm : (j + 1) * blocksize_after_mm, j * blocksize_after_mm : (j + 1) * blocksize_after_mm]
        #     wl_diag[j] = Screened_interaction_lesser[j * blocksize_after_mm : (j + 1) * blocksize_after_mm, j * blocksize_after_mm : (j + 1) * blocksize_after_mm]
        #     wg_diag[j] = Screened_interaction_greater[j * blocksize_after_mm : (j + 1) * blocksize_after_mm, j * blocksize_after_mm : (j + 1) * blocksize_after_mm]

        # for j in range(number_of_blocks_after_mm-1):
        #     wr_upper[j] = Screened_interaction_retarded[j * blocksize_after_mm : (j + 1) * blocksize_after_mm, (j + 1) * blocksize_after_mm : (j + 2) * blocksize_after_mm]
        #     wl_upper[j] = Screened_interaction_lesser[j * blocksize_after_mm : (j + 1) * blocksize_after_mm, (j + 1) * blocksize_after_mm : (j + 2) * blocksize_after_mm]
        #     wg_upper[j] = Screened_interaction_greater[j * blocksize_after_mm : (j + 1) * blocksize_after_mm, (j + 1) * blocksize_after_mm : (j + 2) * blocksize_after_mm]


        # create buffer for results
        # not true inverse, but build up inverses from either corner
        xr_diag_rgf = np.zeros((number_of_blocks_after_mm, blocksize_after_mm, blocksize_after_mm), dtype=np.complex128)
        wg_diag_rgf = np.zeros((number_of_blocks_after_mm, blocksize_after_mm, blocksize_after_mm), dtype=np.complex128)
        wl_diag_rgf = np.zeros((number_of_blocks_after_mm, blocksize_after_mm, blocksize_after_mm), dtype=np.complex128)
        wr_diag_rgf = np.zeros((number_of_blocks_after_mm, blocksize_after_mm, blocksize_after_mm), dtype=np.complex128)



        desc = """
        Meaning of variables:
        _lb: last block
        _c:  current or center
        _p:  previous
        _n:  next
        _r:  right
        _l:  left
        _d:  down
        _u:  up
        """

        # first step of iteration

        # first iteration starts at last block
        # then goes upward


        # x^{r}_E_nn = M_E_nn^{-1}
        xr_lb = np.linalg.inv(mr[-blocksize_after_mm:, -blocksize_after_mm:].toarray())
        xr_lb_ct = xr_lb.conjugate().transpose()
        xr_diag_rgf[number_of_blocks_after_mm - 1] = xr_lb

        # w^{>}_E_nn = x^{r}_E_nn * L^{>}_E_nn * (x^{r}_E_nn).H
        wg_lb = xr_lb @ lg[-blocksize_after_mm:, -blocksize_after_mm:] @ xr_lb_ct
        wg_diag_rgf[number_of_blocks_after_mm - 1] = wg_lb

        # w^{<}_E_nn = x^{r}_E_nn * L^{<}_E_nn * (x^{r}_E_nn).H
        wl_lb = xr_lb @ ll[-blocksize_after_mm:, -blocksize_after_mm:] @ xr_lb_ct
        wl_diag_rgf[number_of_blocks_after_mm - 1] = wl_lb

        # wR_E_nn = xR_E_nn * V_nn
        wr_lb = xr_lb @ vh_cp[-blocksize_after_mm:, -blocksize_after_mm:]
        wr_diag_rgf[number_of_blocks_after_mm - 1] = wr_lb

        # save the diagonal blocks from the previous step
        xr_c = xr_lb
        wg_c = wg_lb
        wl_c = wl_lb

        # loop over all blocks from last to first
        for idx_ib in reversed(range(0, number_of_blocks_after_mm - 1)):
            # block length i
            lb_i = lb_vec_mm[idx_ib]
            # diagonal blocks from previous step
            # avoids a read out operation
            xr_p = xr_c
            wg_p = wg_c
            wl_p = wl_c

            # slice of current and previous block
            slb_c = slice(bmin_mm[idx_ib], bmax_mm[idx_ib] + 1)
            slb_p = slice(bmin_mm[idx_ib + 1], bmax_mm[idx_ib + 1] + 1)

            # read out blocks needed
            mr_c = mr[slb_c, slb_c].toarray()
            mr_r = mr[slb_c, slb_p].toarray()
            mr_d = mr[slb_p, slb_c].toarray()
            vh_c = vh_cp[slb_c, slb_c].toarray()
            vh_d = vh_cp[slb_p, slb_c].toarray()
            lg_c = lg[slb_c, slb_c].toarray()
            lg_d = lg[slb_p, slb_c].toarray()
            ll_c = ll[slb_c, slb_c].toarray()
            ll_d = ll[slb_p, slb_c].toarray()


            # MxR = M_E_kk+1 * xR_E_k+1k+1
            mr_xr = mr_r @ xr_p

            # xR_E_kk = (M_E_kk - M_E_kk+1*xR_E_k+1k+1*M_E_k+1k)^{-1}
            xr_c = np.linalg.inv(mr_c - mr_xr @ mr_d)
            xr_diag_rgf[idx_ib, :lb_i, :lb_i] = xr_c

            # conjugate and transpose
            mr_r_ct = mr_r.conjugate().transpose()
            xr_c_ct = xr_c.conjugate().transpose()

            # A^{\lessgtr} = M_E_kk+1 * xR_E_k+1k+1 * L^{\lessgtr}_E_k+1k
            ag = mr_xr @ lg_d
            al = mr_xr @ ll_d
            ag_diff = ag - ag.conjugate().transpose()
            al_diff = al - al.conjugate().transpose()

            # w^{\lessgtr}_E_kk = xR_E_kk * (L^{\lessgtr}_E_kk + M_E_kk+1*w^{\lessgtr}_E_k+1k+1*M_E_kk+1.H - (A^{\lessgtr} - A^{\lessgtr}.H)) * xR_E_kk.H
            wg_c = xr_c @ (lg_c + mr_r @ wg_p @ mr_r_ct - ag_diff) @ xr_c_ct
            wg_diag_rgf[idx_ib, :lb_i, :lb_i] = wg_c

            wl_c = xr_c @ (ll_c + mr_r @ wl_p @ mr_r_ct - al_diff) @ xr_c_ct
            wl_diag_rgf[idx_ib, :lb_i, :lb_i] = wl_c

            # wR_E_kk = xR_E_kk * (V_kk - M_E_kk+1 * xR_E_k+1k+1 * V_k+1k)
            wr_c = xr_c @ (vh_c - mr_xr @ vh_d)
            wr_diag_rgf[idx_ib, :lb_i, :lb_i] = wr_c

        # block length 0
        lb_f = lb_vec_mm[0]
        lb_p = lb_vec_mm[1]

        # slice of current and previous block
        slb_c = slice(bmin_mm[0], bmax_mm[0] + 1)
        slb_p = slice(bmin_mm[1], bmax_mm[1] + 1)

        # WARNING the last read blocks from the above for loop are used

        # second step of iteration
        vh_r = vh_cp[slb_c, slb_p].toarray()
        lg_r = lg[slb_c, slb_p].toarray()
        ll_r = ll[slb_c, slb_p].toarray()

        xr_mr = xr_p @ mr_d
        xr_mr_ct = xr_mr.conjugate().transpose()

        # WR_E_00 = wR_E_00
        wr_diag[0, :lb_f, :lb_f] = wr_diag_rgf[0, :lb_f, :lb_f]

        # WR_E_01 = (V_01 - WR_E_00*M_E_10) * xR_E_11
        # todo if vh_r can be used instead of vh_cp[slb_c,slb_p]
        wr_upper[0, :lb_f, :lb_p] = (vh_r - wr_diag[0, :lb_f, :lb_f] @ mr_d.transpose()) @ xr_p.transpose()

        # XR_E_00 = xR_E_00
        xr_diag[0, :lb_f, :lb_f] = xr_diag_rgf[0, :lb_f, :lb_f]

        # W^{\lessgtr}_E_00 = w^{\lessgtr}_E_00
        wg_diag[0, :lb_f, :lb_f] = wg_diag_rgf[0, :lb_f, :lb_f]
        wl_diag[0, :lb_f, :lb_f] = wl_diag_rgf[0, :lb_f, :lb_f]

        # W^{\lessgtr}_E_01 = xR_E_00*L^{\lessgtr}_01*_xR_E_11.H - xR_E_00*M_E_01*wL_E_11 - w^{\lessgtr}_E_00*M_E_10.H*xR_E_11.H
        xr_p_ct = xr_p.conjugate().transpose()
        wg_upper[0, :lb_f, :lb_p] = xr_c @ lg_r @ xr_p_ct - xr_c @ mr_r @ wg_p - wg_c @ xr_mr_ct
        wl_upper[0, :lb_f, :lb_p] = xr_c @ ll_r @ xr_p_ct - xr_c @ mr_r @ wl_p - wl_c @ xr_mr_ct

        # loop from left to right corner
        xr_diag_rgf_n = xr_diag_rgf[1, :lb_p, :lb_p]
        xr_diag_c = xr_diag_rgf[0, :lb_f, :lb_f]

        wg_diag_rgf_n = wg_diag_rgf[1, :lb_p, :lb_p]
        wg_diag_c = wg_diag_rgf[0, :lb_f, :lb_f]

        wl_diag_rgf_n = wl_diag_rgf[1, :lb_p, :lb_p]
        wl_diag_c = wl_diag_rgf[0, :lb_f, :lb_f]

        wr_upper_c = wr_upper[0, :lb_f, :lb_p]

        desc = """
        Meaning of variables:
        _lb: last block
        _c:  current or center
        _p:  previous
        _n:  next
        _r:  right
        _l:  left
        _d:  down
        _u:  up
        """

        for idx_ib in range(1, number_of_blocks_after_mm):
            # block length i
            lb_i = lb_vec_mm[idx_ib]
            # slice of current and previous block
            slb_c = slice(bmin_mm[idx_ib], bmax_mm[idx_ib] + 1)
            slb_p = slice(bmin_mm[idx_ib - 1], bmax_mm[idx_ib - 1] + 1)

            # blocks from previous step
            # avoids a read out operation
            xr_diag_rgf_c = xr_diag_rgf_n
            xr_diag_p = xr_diag_c
            wg_diag_rgf_c = wg_diag_rgf_n
            wg_diag_p = wg_diag_c
            wl_diag_rgf_c = wl_diag_rgf_n
            wl_diag_p = wl_diag_c
            wr_upper_p = wr_upper_c

            # read out blocks needed
            mr_l = mr[slb_c, slb_p].toarray()
            mr_u = mr[slb_p, slb_c].toarray()
            lg_l = lg[slb_c, slb_p].toarray()
            ll_l = ll[slb_c, slb_p].toarray()

            # xRM = xR_E_kk * M_E_kk-1
            xr_mr_xr = xr_mr @ xr_diag_p
            xr_mr_xr_ct = xr_mr_xr.conjugate().transpose()
            xr_mr_xr_mr = xr_mr_xr @ mr_u

            # WR_E_kk = wR_E_kk - xR_E_kk*M_E_kk-1*WR_E_k-1k
            wr_diag_rgf_c = wr_diag_rgf[idx_ib, :lb_i, :lb_i]
            # todo if wr_upper_p can be used wr_upper[idx_ib-1,:lb_vec_mm[idx_ib-1],:lb_i]
            wr_diag_c = wr_diag_rgf_c - xr_mr @ wr_upper[idx_ib - 1, :lb_vec_mm[idx_ib - 1], :lb_i]
            wr_diag[idx_ib, :lb_i, :lb_i] = wr_diag_c

            # XR_E_kk = xR_E_kk + (xR_E_kk * M_E_kk-1 * XR_E_k-1k-1 * M_E_k-1k * xR_E_kk)
            xr_diag_c = xr_diag_rgf_c + xr_mr_xr_mr @ xr_diag_rgf_c
            xr_diag[idx_ib, :lb_i, :lb_i] = xr_diag_c

            # A^{\lessgtr} = xR_E_kk * L^{\lessgtr}_E_kk-1 * XR_E_k-1k-1.H * (xR_E_kk * M_E_kk-1).H
            ag = xr_diag_rgf_c @ lg_l @ xr_mr_xr_ct
            al = xr_diag_rgf_c @ ll_l @ xr_mr_xr_ct
            ag_diff = ag - ag.conjugate().transpose()
            al_diff = al - al.conjugate().transpose()

            # B^{\lessgtr} = xR_E_kk * M_E_kk-1 * XR_E_k-1k-1 * M_E_k-1k * w^{\lessgtr}_E_kk
            bg = xr_mr_xr_mr @ wg_diag_rgf_c
            bl = xr_mr_xr_mr @ wl_diag_rgf_c
            bg_diff = bg - bg.conjugate().transpose()
            bl_diff = bl - bl.conjugate().transpose()

            # W^{\lessgtr}_E_kk = w^{\lessgtr}_E_kk + xR_E_kk*M_E_kk-1*W^{\lessgtr}_E_k-1k-1*(xR_E_kk*M_E_kk-1).H - (A^{\lessgtr}-A^{\lessgtr}.H) + (B^{\lessgtr}-B^{\lessgtr}.H)
            wg_diag_c = wg_diag_rgf_c + xr_mr @ wg_diag_p @ xr_mr_ct - ag_diff + bg_diff
            wl_diag_c = wl_diag_rgf_c + xr_mr @ wl_diag_p @ xr_mr_ct - al_diff + bl_diff
            wg_diag[idx_ib, :lb_i, :lb_i] = wg_diag_c
            wl_diag[idx_ib, :lb_i, :lb_i] = wl_diag_c

            # following code block has problems
            if idx_ib < number_of_blocks_after_mm - 1:
                # block length i
                lb_n = lb_vec_mm[idx_ib + 1]
                # slice of current and previous block
                slb_n = slice(bmin_mm[idx_ib + 1], bmax_mm[idx_ib + 1] + 1)

                # read out blocks needed
                vh_d = vh_cp[slb_n, slb_c].toarray()
                mr_d = mr[slb_n, slb_c].toarray()
                mr_r = mr[slb_c, slb_n].toarray()
                lg_r = lg[slb_c, slb_n].toarray()
                ll_r = ll[slb_c, slb_n].toarray()

                xr_diag_rgf_n = xr_diag_rgf[idx_ib + 1, :lb_n, :lb_n]
                wg_diag_rgf_n = wg_diag_rgf[idx_ib + 1, :lb_n, :lb_n]
                wl_diag_rgf_n = wl_diag_rgf[idx_ib + 1, :lb_n, :lb_n]
                xr_diag_rgf_n_ct = xr_diag_rgf_n.conjugate().transpose()

                # xRM_next = M_E_k+1k * xR_E_k+1k+1
                xr_mr = xr_diag_rgf_n @ mr_d
                xr_mr_ct = xr_mr.conjugate().transpose()

                # WR_E_kk+1 = (V_k+1k.T - WR_E_kk*M_E_k+1k.T) * xR_E_k+1k+1.T
                # difference between matlab and python silvio todo
                # this line is wrong todo in the second part
                wr_upper_c = vh_d.transpose() @ xr_diag_rgf_n.transpose() - wr_diag_c @ xr_mr.transpose()
                wr_upper[idx_ib, :lb_i, :lb_n] = wr_upper_c

                # W^{\lessgtr}_E_kk+1 = XR_E_kk*(L^{\lessgtr}_E_kk+1*xR_E_k+1k+1.H - M_E_kk+1*w^{\lessgtr}_E_k+1k+1) - W^{\lessgtr}_E_kk*M_E_k+1k.H*xxR_E_k+1k+1.H
                wg_upper_c = xr_diag_c @ (lg_r @ xr_diag_rgf_n_ct - mr_r @ wg_diag_rgf_n) - wg_diag_c @ xr_mr_ct
                wg_upper[idx_ib, :lb_i, :lb_n] = wg_upper_c
                wl_upper_c = xr_diag_c @ (ll_r @ xr_diag_rgf_n_ct - mr_r @ wl_diag_rgf_n) - wl_diag_c @ xr_mr_ct
                wl_upper[idx_ib, :lb_i, :lb_n] = wl_upper_c

        for idx_ib in range(0, number_of_blocks_after_mm):
            xr_diag[idx_ib, :, :] *= factor
            wg_diag[idx_ib, :, :] *= factor
            wl_diag[idx_ib, :, :] *= factor
            wr_diag[idx_ib, :, :] *= factor
            dosw[idx_ib] = 1j * np.trace(wr_diag[idx_ib, :, :] - wr_diag[idx_ib, :, :].conjugate().transpose())
            nEw[idx_ib] = -1j * np.trace(wl_diag[idx_ib, :, :])
            nPw[idx_ib] = 1j * np.trace(wg_diag[idx_ib, :, :])
            if idx_ib < number_of_blocks_after_mm - 1:
                wr_upper[idx_ib, :, :] *= factor
                wg_upper[idx_ib, :, :] *= factor
                wl_upper[idx_ib, :, :] *= factor

