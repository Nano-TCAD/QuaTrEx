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
import time
import typing

from functools import partial
# from quatrex.OBC import beyn_cpu
from quatrex.OBC import beyn_new
from quatrex.OBC import sancho
from quatrex.OBC import dL_OBC_eigenmode_cpu
from quatrex.OBC import periodic_matmul_correction


def rgf_W(
    vh: sparse.csr_matrix,
    pg: sparse.csr_matrix,
    pl: sparse.csr_matrix,
    pr: sparse.csr_matrix,
    bmax: npt.NDArray[np.int32],
    bmin: npt.NDArray[np.int32],
    wg_diag: npt.NDArray[np.complex128],
    wg_upper: npt.NDArray[np.complex128],
    wl_diag: npt.NDArray[np.complex128],
    wl_upper: npt.NDArray[np.complex128],
    wr_diag: npt.NDArray[np.complex128],
    wr_upper: npt.NDArray[np.complex128],
    xr_diag: npt.NDArray[np.complex128],
    DOSW: npt.NDArray[np.complex128],
    nEW: npt.NDArray[np.complex128],
    nPW: npt.NDArray[np.complex128],
    nbc: np.int64,
    ie: np.int32,
    factor: np.float64 = 1.0,
    ref_flag: bool = False,
    sancho_flag: bool = False
) -> typing.Tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128], npt.NDArray[np.complex128],
                  npt.NDArray[np.complex128]]:
    """Calculates the step from the polarization to the screened interaction.
    Beyn open boundary conditions are used by default.
    The outputs (w and x) are inputs which are changed inplace.
    See the start of this file for more informations.

    Args:
        vh (sparse.csr_matrix): sparse matrix of size (#orbitals,#orbitals)
        pg (sparse.csr_matrix): sparse matrix of size (#orbitals,#orbitals)
        pl (sparse.csr_matrix): sparse matrix of size (#orbitals,#orbitals)
        pr (sparse.csr_matrix): sparse matrix of size (#orbitals,#orbitals)
        bmax (npt.NDArray[np.int32]): end idx of the blocks, vector of size number of blocks
        bmin (npt.NDArray[np.int32]): start idx of the blocks, vector of size number of blocks
        wg_diag (npt.NDArray[np.complex128]): dense matrix of size (#blocks_mm, maxblocklength_mm, maxblocklength_mm)
        wg_upper (npt.NDArray[np.complex128]): dense matrix of size (#blocks_mm-1, maxblocklength_mm, maxblocklength_mm)
        wl_diag (npt.NDArray[np.complex128]): dense matrix of size (#blocks_mm, maxblocklength_mm, maxblocklength_mm)
        wl_upper (npt.NDArray[np.complex128]): dense matrix of size (#blocks_mm-1, maxblocklength_mm, maxblocklength_mm)
        wr_diag (npt.NDArray[np.complex128]): dense matrix of size (#blocks_mm, maxblocklength_mm, maxblocklength_mm)
        wr_upper (npt.NDArray[np.complex128]): dense matrix of size (#blocks_mm-1, maxblocklength_mm, maxblocklength_mm)
        xr_diag (npt.NDArray[np.complex128]): dense matrix of size (#blocks_mm, maxblocklength_mm, maxblocklength_mm)
        ie (np.int32): energy index
        factor (np.float64, optional): factor to multiply the result with. Defaults to 1.0.
        ref_flag (bool, optional): If reference solution to rgf made by np.linalg.inv should be returned
        sancho_flag (bool, optional): If sancho or beyn should be used. Defaults to False.
    
    Returns:
        typing.Tuple[npt.NDArray[np.complex128], xr from inv
                  npt.NDArray[np.complex128],    wg from inv
                  npt.NDArray[np.complex128],    wl from inv
                  npt.NDArray[np.complex128]     wr from inv
                ] warning all dense arrays, only returned if ref_flag is True
    """

    times = np.zeros((10))

    times[0] = -time.perf_counter()

    # Anti-Hermitian symmetrizing of PL and PG
    pl = 1j * np.imag(pl)
    pl = (pl - pl.conj().T) / 2

    pg = 1j * np.imag(pg)
    pg = (pg - pg.conj().T) / 2

    # PR has to be derived from PL and PG and then has to be symmetrized
    pr = 1j * np.imag(pg - pl) / 2
    pr = (pr + pr.T) / 2

    times[0] += time.perf_counter()

    times[1] = -time.perf_counter()

    # limit for beyn
    imag_lim = 1e-4
    # todo find out what rr/R is
    # (R only is against the style guide)
    rr = 1e12
    # copy vh to overwrite it
    vh_cp = vh.copy()

    # number of blocks
    nb = bmin.size
    # number of total orbitals (number of atomic orbitals)
    nao = pr.shape[0]

    # vector of block lengths
    lb_vec = bmax - bmin + 1
    # length of first and last block
    lb_start = lb_vec[0]
    lb_end = lb_vec[nb - 1]

    # block sizes after matrix multiplication
    bmax_mm = bmax[nbc - 1:nb:nbc]
    bmin_mm = bmin[0:nb:nbc]
    # number of blocks after matrix multiplication
    nb_mm = bmin_mm.size
    # vector of block lengths after matrix multiplication
    lb_vec_mm = bmax_mm - bmin_mm + 1
    # max block size after matrix multiplication
    lb_max_mm = np.max(lb_vec_mm)

    # slice block start diagonal and slice block start off diagonal
    slb_sd = slice(0, lb_start)
    slb_so = slice(lb_start, 2 * lb_start)
    # slice block end diagonal and slice block end off diagonal
    slb_ed = slice(nao - lb_end, nao)
    slb_eo = slice(nao - 2 * lb_end, nao - lb_end)

    # slice block start diagonal
    # for block after matrix multiplication
    slb_sd_mm = slice(bmin[0], bmax[nbc - 1] + 1)
    # slice block end diagonal
    # for block after matrix multiplication
    slb_ed_mm = slice(bmin[nb - nbc], nao)

    # calculate S^{r}\left(E\right)
    sr = vh_cp @ pr

    # calculate L^{\lessgtr}\left(E\right)
    vh_cp_ct = vh_cp.conjugate().transpose()
    lg = vh_cp @ pg @ vh_cp_ct
    ll = vh_cp @ pl @ vh_cp_ct

    times[1] += time.perf_counter()

    times[2] = -time.perf_counter()

    # from G^{\lessgtr}\left(E\right) / G^{r}\left(E\right)
    # (diagonal, upper, lower block of the start block at the left)
    pg_sd, pg_su, pg_sl = dL_OBC_eigenmode_cpu.stack_px(pg[slb_sd, slb_sd], pg[slb_sd, slb_so], nbc)
    pl_sd, pl_su, pl_sl = dL_OBC_eigenmode_cpu.stack_px(pl[slb_sd, slb_sd], pl[slb_sd, slb_so], nbc)
    pr_sd, pr_su, pr_sl = dL_OBC_eigenmode_cpu.stack_pr(pr[slb_sd, slb_sd], pr[slb_sd, slb_so], nbc)
    # (diagonal, upper, lower block of the end block at the right)
    pg_ed, pg_eu, pg_el = dL_OBC_eigenmode_cpu.stack_px(pg[slb_ed, slb_ed], -pg[slb_ed, slb_eo].conjugate().transpose(),
                                                        nbc)
    pl_ed, pl_eu, pl_el = dL_OBC_eigenmode_cpu.stack_px(pl[slb_ed, slb_ed], -pl[slb_ed, slb_eo].conjugate().transpose(),
                                                        nbc)
    pr_ed, pr_eu, pr_el = dL_OBC_eigenmode_cpu.stack_pr(pr[slb_ed, slb_ed], pr[slb_ed, slb_eo].transpose(), nbc)

    # from \hat(V)\left(E\right) / G^{r}\left(E\right)
    # (diagonal, upper, lower block of the start block at the left)
    vh_sd, vh_su, vh_sl = dL_OBC_eigenmode_cpu.stack_vh(vh[slb_sd, slb_sd], vh[slb_sd, slb_so], nbc)
    # (diagonal, upper, lower block of the end block at the right)
    vh_ed, vh_eu, vh_el = dL_OBC_eigenmode_cpu.stack_vh(vh[slb_ed, slb_ed], vh[slb_ed, slb_eo].conjugate().transpose(),
                                                        nbc)

    # from L^{\lessgtr}\left(E\right)
    # (diagonal, upper, lower block of the start block at the left)
    lg_sd, lg_su, lg_sl = dL_OBC_eigenmode_cpu.stack_lx(vh_sd, vh_su, vh_sl, pg_sd, pg_su, pg_sl)
    ll_sd, ll_su, ll_sl = dL_OBC_eigenmode_cpu.stack_lx(vh_sd, vh_su, vh_sl, pl_sd, pl_su, pl_sl)

    # (diagonal, upper, lower block of the end block at the right)
    lg_ed, lg_eu, lg_el = dL_OBC_eigenmode_cpu.stack_lx(vh_ed, vh_eu, vh_el, pg_ed, pg_eu, pg_el)
    ll_ed, ll_eu, ll_el = dL_OBC_eigenmode_cpu.stack_lx(vh_ed, vh_eu, vh_el, pl_ed, pl_eu, pl_el)

    # from M^{r}\left(E\right)
    # (diagonal, upper, lower block of the start block at the left)
    mr_sd, mr_su, mr_sl = dL_OBC_eigenmode_cpu.stack_mr(vh_sd, vh_su, vh_sl, pr_sd, pr_su, pr_sl)
    # (diagonal, upper, lower block of the end block at the right)
    mr_ed, mr_eu, mr_el = dL_OBC_eigenmode_cpu.stack_mr(vh_ed, vh_eu, vh_el, pr_ed, pr_eu, pr_el)

    # correction for matrix multiplication--------------------------------------

    # correct first and last block to account for the contacts in multiplication
    # first block
    sr[slb_sd_mm, slb_sd_mm] = sr[slb_sd_mm, slb_sd_mm] + vh_sl @ pr_su
    # last block
    sr[slb_ed_mm, slb_ed_mm] = sr[slb_ed_mm, slb_ed_mm] + vh_eu @ pr_el

    # calculate M^{r}\left(E\right) (only after above correction)
    mr = sparse.identity(nao, format="csr") - sr

    # correct first and last block to account for the contacts in multiplication
    # first block
    # L^{\lessgtr}_E_00 = V_10*P^{\lessgtr}_E_00*V_01 + V_10*P^{\lessgtr}_E_01*V_00 + V_00*P^{\lessgtr}_E_10*V_01
    lg[slb_sd_mm, slb_sd_mm] = lg[slb_sd_mm, slb_sd_mm] + vh_sl @ pg_sd @ vh_su
    lg[slb_sd_mm, slb_sd_mm] = lg[slb_sd_mm, slb_sd_mm] + vh_sl @ pg_su @ vh_sd
    lg[slb_sd_mm, slb_sd_mm] = lg[slb_sd_mm, slb_sd_mm] + vh_sd @ pg_sl @ vh_su

    ll[slb_sd_mm, slb_sd_mm] = ll[slb_sd_mm, slb_sd_mm] + vh_sl @ pl_sd @ vh_su
    ll[slb_sd_mm, slb_sd_mm] = ll[slb_sd_mm, slb_sd_mm] + vh_sl @ pl_su @ vh_sd
    ll[slb_sd_mm, slb_sd_mm] = ll[slb_sd_mm, slb_sd_mm] + vh_sd @ pl_sl @ vh_su

    # last block
    # L^{\lessgtr}_E_nn = V_nn*PL_E_n-1n*V_nn-1 + V_n-1n*PL_E_nn-1*V_nn + V_n-1n*PL_E_nn*V_nn-1
    lg[slb_ed_mm, slb_ed_mm] = lg[slb_ed_mm, slb_ed_mm] + vh_ed @ pg_eu @ vh_el
    lg[slb_ed_mm, slb_ed_mm] = lg[slb_ed_mm, slb_ed_mm] + vh_eu @ pg_el @ vh_ed
    lg[slb_ed_mm, slb_ed_mm] = lg[slb_ed_mm, slb_ed_mm] + vh_eu @ pg_ed @ vh_el

    ll[slb_ed_mm, slb_ed_mm] = ll[slb_ed_mm, slb_ed_mm] + vh_ed @ pl_eu @ vh_el
    ll[slb_ed_mm, slb_ed_mm] = ll[slb_ed_mm, slb_ed_mm] + vh_eu @ pl_el @ vh_ed
    ll[slb_ed_mm, slb_ed_mm] = ll[slb_ed_mm, slb_ed_mm] + vh_eu @ pl_ed @ vh_el

    times[2] += time.perf_counter()

    # correction for the matrix inverse calculations----------------------------

    times[3] = -time.perf_counter()

    # conditions about convergence or meaningful results from
    # boundary correction calculations
    cond_l = 0.0
    cond_r = 0.0

    # correction for first block
    # if not sancho_flag:
    #     _, cond_l, dxr_sd, dmr_sd, min_dEkL = beyn_cpu.beyn(
    #                                             mr_sd.toarray(),
    #                                             mr_su.toarray(),
    #                                             mr_sl.toarray(),
    #                                             imag_lim, rr, "L")
    #     if not np.isnan(cond_l):
    #         dvh_sd = mr_sl @ dxr_sd @ vh_su
    #old wrong version of beyn
    if not sancho_flag:
        _, cond_l, dxr_sd, dmr_sd, min_dEkL = beyn_new.beyn_old(mr_sd.toarray(), mr_su.toarray(), mr_sl.toarray(),
                                                                imag_lim, rr, "L")
        if not np.isnan(cond_l):
            dvh_sd = mr_sl @ dxr_sd @ vh_su

    if np.isnan(cond_l) or sancho_flag:
        dxr_sd, dmr_sd, dvh_sd, cond_l = sancho.open_boundary_conditions(mr_sd.toarray(), mr_sl.toarray(),
                                                                         mr_su.toarray(), vh_su.toarray())
    # correction for last block
    # if not sancho_flag:
    #     _, cond_r, dxr_ed, dmr_ed, min_dEkR = beyn_cpu.beyn(
    #                                             mr_ed.toarray(),
    #                                             mr_eu.toarray(),
    #                                             mr_el.toarray(),
    #                                             imag_lim, rr, "R")
    #     if not np.isnan(cond_r):
    #         dvh_ed = mr_eu @ dxr_ed @ vh_el
    # old wrong version of beyn
    if not sancho_flag:
        _, cond_r, dxr_ed, dmr_ed, min_dEkR = beyn_new.beyn_old(mr_ed.toarray(), mr_eu.toarray(), mr_el.toarray(),
                                                                imag_lim, rr, "R")
        if not np.isnan(cond_r):
            dvh_ed = mr_eu @ dxr_ed @ vh_el

    if np.isnan(cond_r) or sancho_flag:
        dxr_ed, dmr_ed, dvh_ed, cond_r = sancho.open_boundary_conditions(mr_ed.toarray(), mr_eu.toarray(),
                                                                         mr_el.toarray(), vh_el.toarray())

    times[3] += time.perf_counter()

    times[4] = -time.perf_counter()

    # beyn gave a meaningful result
    if not np.isnan(cond_l):
        dlg_sd, dll_sd = dL_OBC_eigenmode_cpu.get_dl_obc(dxr_sd,
                                                         lg_sd.toarray(),
                                                         lg_su.toarray(),
                                                         ll_sd.toarray(),
                                                         ll_su.toarray(),
                                                         mr_sl.toarray(),
                                                         blk="L")

        mr[slb_sd_mm, slb_sd_mm] = mr[slb_sd_mm, slb_sd_mm] - dmr_sd
        vh_cp[slb_sd_mm, slb_sd_mm] = vh_cp[slb_sd_mm, slb_sd_mm] - dvh_sd

        if not np.isnan(dll_sd).any():
            lg[slb_sd_mm, slb_sd_mm] = lg[slb_sd_mm, slb_sd_mm] + dlg_sd
            ll[slb_sd_mm, slb_sd_mm] = ll[slb_sd_mm, slb_sd_mm] + dll_sd
        else:
            cond_l = np.nan

    # condR = np.nan
    if not np.isnan(cond_r):
        dlg_ed, dll_ed = dL_OBC_eigenmode_cpu.get_dl_obc(dxr_ed,
                                                         lg_ed.toarray(),
                                                         lg_el.toarray(),
                                                         ll_ed.toarray(),
                                                         ll_el.toarray(),
                                                         mr_eu.toarray(),
                                                         blk="R")

        mr[slb_ed_mm, slb_ed_mm] = mr[slb_ed_mm, slb_ed_mm] - dmr_ed
        vh_cp[slb_ed_mm, slb_ed_mm] = vh_cp[slb_ed_mm, slb_ed_mm] - dvh_ed

        if not np.isnan(dll_ed).any():
            lg[slb_ed_mm, slb_ed_mm] = lg[slb_ed_mm, slb_ed_mm] + dlg_ed
            ll[slb_ed_mm, slb_ed_mm] = ll[slb_ed_mm, slb_ed_mm] + dll_ed
        else:
            cond_r = np.nan

    min_dEk = np.min((min_dEkL, min_dEkR))

    times[4] += time.perf_counter()
    # start of rgf_W------------------------------------------------------------

    times[5] = -time.perf_counter()

    # check if OBC did not fail
    if not np.isnan(cond_r) and not np.isnan(cond_l) and ie:

        # create buffer for results

        # not true inverse, but build up inverses from either corner
        xr_diag_rgf = np.empty((nb_mm, lb_max_mm, lb_max_mm), dtype=np.complex128)
        wg_diag_rgf = np.empty((nb_mm, lb_max_mm, lb_max_mm), dtype=np.complex128)
        wl_diag_rgf = np.empty((nb_mm, lb_max_mm, lb_max_mm), dtype=np.complex128)
        wr_diag_rgf = np.empty((nb_mm, lb_max_mm, lb_max_mm), dtype=np.complex128)

        # todo
        # find out if IdE, DOS, dWL, dWG are needed

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

        # last block index
        idx_lb = nb_mm - 1
        # slice for last block
        slb_lb = slice(bmin_mm[idx_lb], bmax_mm[idx_lb] + 1)
        # length last block
        lb_lb = lb_vec_mm[idx_lb]

        # x^{r}_E_nn = M_E_nn^{-1}
        xr_lb = np.linalg.inv(mr[slb_lb, slb_lb].toarray())
        xr_lb_ct = xr_lb.conjugate().transpose()
        xr_diag_rgf[idx_lb, :lb_lb, :lb_lb] = xr_lb

        # w^{>}_E_nn = x^{r}_E_nn * L^{>}_E_nn * (x^{r}_E_nn).H
        wg_lb = xr_lb @ lg[slb_lb, slb_lb] @ xr_lb_ct
        wg_diag_rgf[idx_lb, :lb_lb, :lb_lb] = wg_lb

        # w^{<}_E_nn = x^{r}_E_nn * L^{<}_E_nn * (x^{r}_E_nn).H
        wl_lb = xr_lb @ ll[slb_lb, slb_lb] @ xr_lb_ct
        wl_diag_rgf[idx_lb, :lb_lb, :lb_lb] = wl_lb

        # wR_E_nn = xR_E_nn * V_nn
        wr_lb = xr_lb @ vh_cp[slb_lb, slb_lb]
        wr_diag_rgf[idx_lb, :lb_lb, :lb_lb] = wr_lb

        # save the diagonal blocks from the previous step
        xr_c = xr_lb
        wg_c = wg_lb
        wl_c = wl_lb

        # loop over all blocks from last to first
        for idx_ib in reversed(range(0, nb_mm - 1)):
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

        for idx_ib in range(1, nb_mm):
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

            #W^{\lessgtr}_E_kk = w^{\lessgtr}_E_kk + xR_E_kk*M_E_kk-1*W^{\lessgtr}_E_k-1k-1*(xR_E_kk*M_E_kk-1).H - (A^{\lessgtr}-A^{\lessgtr}.H) + (B^{\lessgtr}-B^{\lessgtr}.H)
            wg_diag_c = wg_diag_rgf_c + xr_mr @ wg_diag_p @ xr_mr_ct - ag_diff + bg_diff
            wl_diag_c = wl_diag_rgf_c + xr_mr @ wl_diag_p @ xr_mr_ct - al_diff + bl_diff
            wg_diag[idx_ib, :lb_i, :lb_i] = wg_diag_c
            wl_diag[idx_ib, :lb_i, :lb_i] = wl_diag_c

            # following code block has problems
            if idx_ib < nb_mm - 1:
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

        for idx_ib in range(0, nb_mm):
            xr_diag[idx_ib, :, :] *= factor
            wg_diag[idx_ib, :, :] *= factor
            wl_diag[idx_ib, :, :] *= factor
            wr_diag[idx_ib, :, :] *= factor
            DOSW[idx_ib] = 1j * np.trace(wr_diag[idx_ib, :, :] - wr_diag[idx_ib, :, :].conjugate().transpose())
            nEW[idx_ib] = -1j * np.trace(wl_diag[idx_ib, :, :])
            nPW[idx_ib] = 1j * np.trace(wg_diag[idx_ib, :, :])

            if idx_ib < nb_mm - 1:
                wr_upper[idx_ib, :, :] *= factor
                wg_upper[idx_ib, :, :] *= factor
                wl_upper[idx_ib, :, :] *= factor

        times[5] += time.perf_counter()

        if ref_flag:
            # reference solution
            # invert m
            mr_dense = mr.toarray()
            xr_ref = np.linalg.inv(mr_dense)
            wr_ref = xr_ref @ vh_cp.todense()
            # todo change L to differentiate between greater/lesser
            xr_ref_ct = xr_ref.conjugate().transpose()
            wg_ref = xr_ref @ lg @ xr_ref_ct
            wl_ref = xr_ref @ ll @ xr_ref_ct

            return xr_ref * factor, wg_ref * factor, wl_ref * factor, wr_ref * factor

        return times


def rgf_w_opt(
    vh: sparse.csr_matrix,
    pg: sparse.csr_matrix,
    pl: sparse.csr_matrix,
    pr: sparse.csr_matrix,
    bmax: npt.NDArray[np.int32],
    bmin: npt.NDArray[np.int32],
    wg_diag: npt.NDArray[np.complex128],
    wg_upper: npt.NDArray[np.complex128],
    wl_diag: npt.NDArray[np.complex128],
    wl_upper: npt.NDArray[np.complex128],
    wr_diag: npt.NDArray[np.complex128],
    wr_upper: npt.NDArray[np.complex128],
    xr_diag: npt.NDArray[np.complex128],
    dosw: npt.NDArray[np.complex128],
    nEw: npt.NDArray[np.complex128],
    nPw: npt.NDArray[np.complex128],
    nbc: np.int64,
    ie: np.int32,
    factor: np.float64 = 1.0,
    NCpSC: np.int32 = 1,
    block_inv: bool = False,
    use_dace: bool = False,
    validate_dace: bool = False,
    ref_flag: bool = False,
    sancho_flag: bool = False,
) -> typing.Tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128], npt.NDArray[np.complex128],
                  npt.NDArray[np.complex128]]:
    """Calculates the step from the polarization to the screened interaction.
    Beyn open boundary conditions are used by default.
    The outputs (w and x) are inputs which are changed inplace.
    See the start of this file for more informations.

    Args:
        vh (sparse.csr_matrix): sparse matrix of size (#orbitals,#orbitals)
        pg (sparse.csr_matrix): sparse matrix of size (#orbitals,#orbitals)
        pl (sparse.csr_matrix): sparse matrix of size (#orbitals,#orbitals)
        pr (sparse.csr_matrix): sparse matrix of size (#orbitals,#orbitals)
        bmax (npt.NDArray[np.int32]): end idx of the blocks, vector of size number of blocks
        bmin (npt.NDArray[np.int32]): start idx of the blocks, vector of size number of blocks
        wg_diag (npt.NDArray[np.complex128]): dense matrix of size (#blocks_mm, maxblocklength_mm, maxblocklength_mm)
        wg_upper (npt.NDArray[np.complex128]): dense matrix of size (#blocks_mm-1, maxblocklength_mm, maxblocklength_mm)
        wl_diag (npt.NDArray[np.complex128]): dense matrix of size (#blocks_mm, maxblocklength_mm, maxblocklength_mm)
        wl_upper (npt.NDArray[np.complex128]): dense matrix of size (#blocks_mm-1, maxblocklength_mm, maxblocklength_mm)
        wr_diag (npt.NDArray[np.complex128]): dense matrix of size (#blocks_mm, maxblocklength_mm, maxblocklength_mm)
        wr_upper (npt.NDArray[np.complex128]): dense matrix of size (#blocks_mm-1, maxblocklength_mm, maxblocklength_mm)
        xr_diag (npt.NDArray[np.complex128]): dense matrix of size (#blocks_mm, maxblocklength_mm, maxblocklength_mm)
        dosw (npt.NDArray[np.complex128]): density of states, vector of size number of blocks
        nbc (np.int64): how block size changes after matrix multiplication
        ie (np.int32): energy index (not used)
        factor (np.float64, optional): factor to multiply the result with. Defaults to 1.0.
        ref_flag (bool, optional): If reference solution to rgf made by np.linalg.inv should be returned
        sancho_flag (bool, optional): If sancho or beyn should be used. Defaults to False.
    
    Returns:
        typing.Tuple[npt.NDArray[np.complex128], xr from inv
                  npt.NDArray[np.complex128],    wg from inv
                  npt.NDArray[np.complex128],    wl from inv
                  npt.NDArray[np.complex128]     wr from inv
                ] warning all dense arrays, only returned if ref_flag is True
    """
    beyn = beyn_new.beyn_new
    if use_dace:
        # from OBC.beyn_dace import beyn, contour_integral_dace, sort_k_dace
        # contour_integral, _ = contour_integral_dace.load_precompiled_sdfg(f'.dacecache/{contour_integral_dace.name}')
        # sortk, _ = sort_k_dace.load_precompiled_sdfg(f'.dacecache/{sort_k_dace.name}')
        # beyn = partial(beyn, contour_integral=contour_integral, sortk=sortk)
        from OBC.beyn_dace import beyn as beyn_dace
        beyn = partial(beyn_dace, validate=validate_dace)

    times = np.zeros((10), dtype=np.float64)

    times[0] = -time.perf_counter()

    # TODO: Fix module to import BSR matrix
    bsr = False
    if not (isinstance(vh, sparse.csr_matrix) or isinstance(vh, sparse.csc_matrix)):
        bsr = True

    # is now done outside of function.
    # # Anti-Hermitian symmetrizing of PL and PG
    # pl = 1j * np.imag(pl)
    # pl = (pl - pl.conj().T) / 2

    # pg = 1j * np.imag(pg)
    # pg = (pg - pg.conj().T) / 2

    # # PR has to be derived from PL and PG and then has to be symmetrized
    # pr = 1j * np.imag(pg - pl) / 2
    # pr = (pr + pr.T) / 2

    times[0] += time.perf_counter()

    times[1] = -time.perf_counter()

    # limit for beyn
    imag_lim = 1e-4
    # todo find out what rr/R is
    # (R only is against the style guide)
    #todo, compare with matlab
    rr = 1e6
    # copy vh to overwrite it
    vh_cp = vh.copy()

    # number of blocks
    nb = bmin.size
    # number of total orbitals (number of atomic orbitals)
    nao = pr.shape[0]

    # vector of block lengths
    lb_vec = bmax - bmin + 1
    # length of first and last block
    lb_start = lb_vec[0]
    lb_end = lb_vec[nb - 1]

    # block sizes after matrix multiplication
    bmax_mm = bmax[nbc - 1:nb:nbc]
    bmin_mm = bmin[0:nb:nbc]
    # number of blocks after matrix multiplication
    nb_mm = bmin_mm.size
    # vector of block lengths after matrix multiplication
    lb_vec_mm = bmax_mm - bmin_mm + 1
    # max block size after matrix multiplication
    lb_max_mm = np.max(lb_vec_mm)

    # slice block start diagonal and slice block start off diagonal
    slb_sd = slice(0, lb_start)
    slb_so = slice(lb_start, 2 * lb_start)
    # slice block end diagonal and slice block end off diagonal
    slb_ed = slice(nao - lb_end, nao)
    slb_eo = slice(nao - 2 * lb_end, nao - lb_end)

    times[1] = -time.perf_counter()

    # calculate L^{\lessgtr}\left(E\right)
    vh_cp_ct = vh_cp.conjugate().transpose()
    lg = vh_cp @ pg @ vh_cp_ct
    ll = vh_cp @ pl @ vh_cp_ct
    # calculate M^{r}\left(E\right)
    if bsr:
        mr = sparse.identity(nao, format="csr").tobsr(blocksize=vh_cp.blocksize) - vh_cp @ pr
    else:
        mr = sparse.identity(nao, format="csr") - vh_cp @ pr

    times[1] += time.perf_counter()

    times[2] = -time.perf_counter()

    # from M^{r}\left(E\right)/L^{\lessgtr}\left(E\right)\hat(V)\left(E\right)
    # (diagonal, upper, lower block of the start block at the left)
    # mr_s, lg_s, ll_s, dmr_s, dlg_s, dll_s, vh_s = dL_OBC_eigenmode_cpu.get_mm_obc(vh[slb_sd,slb_sd], vh[slb_sd,slb_so],
    #                                        pg[slb_sd,slb_sd], pg[slb_sd,slb_so],
    #                                        pl[slb_sd,slb_sd], pl[slb_sd,slb_so],
    #                                        pr[slb_sd,slb_sd], pr[slb_sd,slb_so],
    #                                        nbc)
    # # (diagonal, upper, lower block of the end block at the right)
    # mr_e, lg_e, ll_e, dmr_e, dlg_e, dll_e, vh_e = dL_OBC_eigenmode_cpu.get_mm_obc(vh[slb_ed,slb_ed], vh[slb_ed,slb_eo].conjugate().transpose(),
    #                                        pg[slb_ed,slb_ed], -pg[slb_ed,slb_eo].conjugate().transpose(),
    #                                        pl[slb_ed,slb_ed], -pl[slb_ed,slb_eo].conjugate().transpose(),
    #                                        pr[slb_ed,slb_ed], pr[slb_ed,slb_eo].transpose(),
    #                                        nbc)

    # todo ask Alex, why numba still has strided inputs after these operations
    if bsr:
        vh_s1 = vh[slb_sd, slb_sd]
        vh_s2 = vh[slb_sd, slb_so]
        pg_s1 = pg[slb_sd, slb_sd]
        pg_s2 = pg[slb_sd, slb_so]
        pl_s1 = pl[slb_sd, slb_sd]
        pl_s2 = pl[slb_sd, slb_so]
        pr_s1 = pr[slb_sd, slb_sd]
        pr_s2 = pr[slb_sd, slb_so]
        pr_s3 = pr[slb_so, slb_sd]
    else:
        vh_s1 = np.ascontiguousarray(vh[slb_sd, slb_sd].toarray(order="C"))
        vh_s2 = np.ascontiguousarray(vh[slb_sd, slb_so].toarray(order="C"))
        pg_s1 = np.ascontiguousarray(pg[slb_sd, slb_sd].toarray(order="C"))
        pg_s2 = np.ascontiguousarray(pg[slb_sd, slb_so].toarray(order="C"))
        pl_s1 = np.ascontiguousarray(pl[slb_sd, slb_sd].toarray(order="C"))
        pl_s2 = np.ascontiguousarray(pl[slb_sd, slb_so].toarray(order="C"))
        pr_s1 = np.ascontiguousarray(pr[slb_sd, slb_sd].toarray(order="C"))
        pr_s2 = np.ascontiguousarray(pr[slb_sd, slb_so].toarray(order="C"))
        pr_s3 = np.ascontiguousarray(pr[slb_so, slb_sd].toarray(order="C"))
    mr_s, lg_s, ll_s, dmr_s, dlg_s, dll_s, vh_s, mb00 = dL_OBC_eigenmode_cpu.get_mm_obc_dense(
        vh_s1, vh_s2, pg_s1, pg_s2, pl_s1, pl_s2, pr_s1, pr_s2, pr_s3, nbc, NCpSC, 'L')
    # (diagonal, upper, lower block of the end block at the right)
    if bsr:
        vh_e1 = vh[slb_ed, slb_ed]
        vh_e2 = vh[slb_ed, slb_eo].conjugate().transpose()
        pg_e1 = pg[slb_ed, slb_ed]
        pg_e2 = -pg[slb_ed, slb_eo].conjugate().transpose()
        pl_e1 = pl[slb_ed, slb_ed]
        pl_e2 = -pl[slb_ed, slb_eo].conjugate().transpose()
        pr_e1 = pr[slb_ed, slb_ed]
        pr_e2 = pr[slb_ed, slb_eo].transpose()
    else:
        vh_e1 = np.ascontiguousarray(vh[slb_ed, slb_ed].toarray(order="C"))
        vh_e2 = np.ascontiguousarray(vh[slb_ed, slb_eo].conjugate().transpose().toarray(order="C"))
        pg_e1 = np.ascontiguousarray(pg[slb_ed, slb_ed].toarray(order="C"))
        pg_e2 = np.ascontiguousarray(-pg[slb_ed, slb_eo].conjugate().transpose().toarray(order="C"))
        pl_e1 = np.ascontiguousarray(pl[slb_ed, slb_ed].toarray(order="C"))
        pl_e2 = np.ascontiguousarray(-pl[slb_ed, slb_eo].conjugate().transpose().toarray(order="C"))
        pr_e1 = np.ascontiguousarray(pr[slb_ed, slb_ed].toarray(order="C"))
        pr_e2 = np.ascontiguousarray(pr[slb_eo, slb_ed].toarray(order="C"))
        pr_e3 = np.ascontiguousarray(pr[slb_ed, slb_eo].toarray(order="C"))
    mr_e, lg_e, ll_e, dmr_e, dlg_e, dll_e, vh_e, mbNN = dL_OBC_eigenmode_cpu.get_mm_obc_dense(
        vh_e1, vh_e2, pg_e1, pg_e2, pl_e1, pl_e2, pr_e1, pr_e2, pr_e3, nbc, NCpSC, 'R')
    # dmr_s[0] += np.identity(lb * nbc) * (1+1j*1e-10)
    # dmr_e[0] += np.identity(lb * nbc) * (1+1j*1e-10)
    # correction for matrix multiplication--------------------------------------

    # # new correction of matrix multiplication
    # mr_st, mr_et, lg_st, lg_et, ll_st, ll_et, dmr_st, dmr_et, dlg_st, dlg_et, dll_st, dll_et, vh_st, vh_et = \
    #     periodic_matmul_correction.correction_system_matrix(mr, vh, pr, pl, pg, ll, lg, bmin, bmax, bmin_mm, bmax_mm)
    

    # correct first and last block to account for the contacts in multiplication
    dmr_sd = dmr_s[0]
    dmr_ed = dmr_e[1]

    # correct first and last block to account for the contacts in multiplication
    # first block
    # L^{\lessgtr}_E_00 = V_10*P^{\lessgtr}_E_00*V_01 + V_10*P^{\lessgtr}_E_01*V_00 + V_00*P^{\lessgtr}_E_10*V_01
    dlg_sd = dlg_s[0]
    dll_sd = dll_s[0]

    # last block
    # L^{\lessgtr}_E_nn = V_nn*PL_E_n-1n*V_nn-1 + V_n-1n*PL_E_nn-1*V_nn + V_n-1n*PL_E_nn*V_nn-1
    dlg_ed = dlg_e[1]
    dll_ed = dll_e[1]

    times[2] += time.perf_counter()

    # correction for the matrix inverse calculations----------------------------

    times[3] = -time.perf_counter()

    # conditions about convergence or meaningful results from
    # boundary correction calculations
    cond_l = 0.0
    cond_r = 0.0

    # correction for first block
    if not sancho_flag:
        dmr, dxr_sd, cond_l, min_dEkL = beyn(nbc * NCpSC, mb00, mr_s[0], mr_s[1], mr_s[2], imag_lim, rr, "L")

        if not np.isnan(cond_l):
            dmr_sd -= dmr
            dvh_sd = mr_s[2] @ dxr_sd @ vh_s[0]

    # old wrong version
    # if not sancho_flag:
    #     _, cond_l, dxr_sd, dmr, min_dEkL = beyn_cpu.beyn_old(
    #                                             mr_s[0],
    #                                             mr_s[1],
    #                                             mr_s[2],
    #                                             imag_lim, rr, "L")

    #     if not np.isnan(cond_l):
    #         dmr_sd -= dmr
    #         dvh_sd = mr_s[2] @ dxr_sd @ vh_s[0]

    if np.isnan(cond_l) or sancho_flag:
        dxr_sd, dmr, dvh_sd, cond_l = sancho.open_boundary_conditions(mr_s[0], mr_s[2], mr_s[1], vh_s[0])
        dmr_sd -= dmr
    # correction for last block
    if not sancho_flag:
        dmr, dxr_ed, cond_r, min_dEkR = beyn(nbc * NCpSC, mbNN, mr_e[0], mr_e[1], mr_e[2], imag_lim, rr, "R")
        if not np.isnan(cond_r):
            dmr_ed -= dmr
            dvh_ed = mr_e[1] @ dxr_ed @ vh_e[1]
    # old wrong version
    # if not sancho_flag:
    #     _, cond_r, dxr_ed, dmr, min_dEkR = beyn_cpu.beyn_old(
    #                                             mr_e[0],
    #                                             mr_e[1],
    #                                             mr_e[2],
    #                                             imag_lim, rr, "R")
    #     if not np.isnan(cond_r):
    #         dmr_ed -= dmr
    #         dvh_ed = mr_e[1] @ dxr_ed @ vh_e[1]
    if np.isnan(cond_r) or sancho_flag:
        dxr_ed, dmr, dvh_ed, cond_r = sancho.open_boundary_conditions(mr_e[0], mr_e[1], mr_e[2], vh_e[1])
        dmr_ed -= dmr

    times[3] += time.perf_counter()

    times[4] = -time.perf_counter()

    # beyn gave a meaningful result
    if not np.isnan(cond_l) and not np.isnan(cond_l):
        dlg, dll = dL_OBC_eigenmode_cpu.get_dl_obc_alt(dxr_sd, lg_s[0], lg_s[1], ll_s[0], ll_s[1], mr_s[2], blk="L")

        if np.isnan(dll).any():
            cond_l = np.nan
        else:
            dlg_sd += dlg
            dll_sd += dll

        dlg, dll = dL_OBC_eigenmode_cpu.get_dl_obc_alt(dxr_ed, lg_e[0], lg_e[2], ll_e[0], ll_e[2], mr_e[1], blk="R")

        if np.isnan(dll_ed).any():
            cond_r = np.nan
        else:
            dlg_ed += dlg
            dll_ed += dll

    min_dEk = np.min((min_dEkL, min_dEkR))

    times[4] += time.perf_counter()
    # start of rgf_W------------------------------------------------------------

    times[5] = -time.perf_counter()

    # check if OBC did not fail
    if not np.isnan(cond_r) and not np.isnan(cond_l) and ie:

        # create buffer for results

        # not true inverse, but build up inverses from either corner
        xr_diag_rgf = np.zeros((nb_mm, lb_max_mm, lb_max_mm), dtype=np.complex128)
        wg_diag_rgf = np.zeros((nb_mm, lb_max_mm, lb_max_mm), dtype=np.complex128)
        wl_diag_rgf = np.zeros((nb_mm, lb_max_mm, lb_max_mm), dtype=np.complex128)
        wr_diag_rgf = np.zeros((nb_mm, lb_max_mm, lb_max_mm), dtype=np.complex128)

        # True inverse M^-1 off-diagonal blocks
        xr_upper = np.zeros((nb_mm - 1, lb_max_mm, lb_max_mm), dtype=np.complex128)
        xr_lower = np.zeros((nb_mm - 1, lb_max_mm, lb_max_mm), dtype=np.complex128)

        # todo
        # find out if IdE, DOS, dWL, dWG are needed

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

        # last block index
        idx_lb = nb_mm - 1
        # slice for last block
        slb_lb = slice(bmin_mm[idx_lb], bmax_mm[idx_lb] + 1)
        # length last block
        lb_lb = lb_vec_mm[idx_lb]

        # x^{r}_E_nn = M_E_nn^{-1}
        if bsr:
            xr_lb = np.linalg.inv(mr[slb_lb, slb_lb] + dmr_ed)
            xr_lb_ct = xr_lb.conjugate().transpose()
            xr_diag_rgf[idx_lb, :lb_lb, :lb_lb] = xr_lb

            # w^{>}_E_nn = x^{r}_E_nn * L^{>}_E_nn * (x^{r}_E_nn).H
            wg_lb = xr_lb @ (lg[slb_lb, slb_lb] + dlg_ed) @ xr_lb_ct
            wg_diag_rgf[idx_lb, :lb_lb, :lb_lb] = wg_lb

            # w^{<}_E_nn = x^{r}_E_nn * L^{<}_E_nn * (x^{r}_E_nn).H
            wl_lb = xr_lb @ (ll[slb_lb, slb_lb] + dll_ed) @ xr_lb_ct
            wl_diag_rgf[idx_lb, :lb_lb, :lb_lb] = wl_lb

            # wR_E_nn = xR_E_nn * V_nn
            wr_lb = xr_lb @ (vh_cp[slb_lb, slb_lb] - dvh_ed)
            wr_diag_rgf[idx_lb, :lb_lb, :lb_lb] = wr_lb
        else:
            xr_lb = np.linalg.inv(mr[slb_lb, slb_lb].toarray() + dmr_ed)
            xr_lb_ct = xr_lb.conjugate().transpose()
            xr_diag_rgf[idx_lb, :lb_lb, :lb_lb] = xr_lb

            # w^{>}_E_nn = x^{r}_E_nn * L^{>}_E_nn * (x^{r}_E_nn).H
            wg_lb = xr_lb @ (lg[slb_lb, slb_lb].toarray() + dlg_ed) @ xr_lb_ct
            wg_diag_rgf[idx_lb, :lb_lb, :lb_lb] = wg_lb

            # w^{<}_E_nn = x^{r}_E_nn * L^{<}_E_nn * (x^{r}_E_nn).H
            wl_lb = xr_lb @ (ll[slb_lb, slb_lb].toarray() + dll_ed) @ xr_lb_ct
            wl_diag_rgf[idx_lb, :lb_lb, :lb_lb] = wl_lb

            # wR_E_nn = xR_E_nn * V_nn
            wr_lb = xr_lb @ (vh_cp[slb_lb, slb_lb].toarray() - dvh_ed)
            wr_diag_rgf[idx_lb, :lb_lb, :lb_lb] = wr_lb

        # save the diagonal blocks from the previous step
        xr_c = xr_lb
        wg_c = wg_lb
        wl_c = wl_lb

        # loop over all blocks from last to first
        for idx_ib in reversed(range(0, nb_mm - 1)):
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
            if bsr:
                mr_c = mr[slb_c, slb_c]
                mr_r = mr[slb_c, slb_p]
                mr_d = mr[slb_p, slb_c]
                vh_c = vh_cp[slb_c, slb_c]
                vh_d = vh_cp[slb_p, slb_c]
                lg_c = lg[slb_c, slb_c]
                lg_d = lg[slb_p, slb_c]
                ll_c = ll[slb_c, slb_c]
                ll_d = ll[slb_p, slb_c]
            else:
                mr_c = mr[slb_c, slb_c].toarray()
                mr_r = mr[slb_c, slb_p].toarray()
                mr_d = mr[slb_p, slb_c].toarray()
                vh_c = vh_cp[slb_c, slb_c].toarray()
                vh_d = vh_cp[slb_p, slb_c].toarray()
                lg_c = lg[slb_c, slb_c].toarray()
                lg_d = lg[slb_p, slb_c].toarray()
                ll_c = ll[slb_c, slb_c].toarray()
                ll_d = ll[slb_p, slb_c].toarray()

            if idx_ib == 0:
                lg_c += dlg_sd
                ll_c += dll_sd
                vh_c -= dvh_sd
                mr_c += dmr_sd
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
        if bsr:
            vh_r = vh_cp[slb_c, slb_p]
            lg_r = lg[slb_c, slb_p]
            ll_r = ll[slb_c, slb_p]
        else:
            vh_r = vh_cp[slb_c, slb_p].toarray()
            vh_d = vh_cp[slb_p, slb_c].toarray()
            lg_r = lg[slb_c, slb_p].toarray()
            ll_r = ll[slb_c, slb_p].toarray()

        xr_mr = xr_p @ mr_d
        xr_mr_ct = xr_mr.conjugate().transpose()

        # XR_E_00 = xR_E_00
        xr_diag[0, :lb_f, :lb_f] = xr_diag_rgf[0, :lb_f, :lb_f]
        xr_upper[0, :lb_f, :lb_p] = -xr_diag_rgf[0, :lb_f, :lb_f] @ mr_r @ xr_p
        xr_lower[0, :lb_p, :lb_f] = -xr_p @ mr_d @ xr_diag_rgf[0, :lb_f, :lb_f]

        # WR_E_00 = wR_E_00
        #wr_diag[0, :lb_f, :lb_f] = wr_diag_rgf[0, :lb_f, :lb_f]
        wr_diag[0, :lb_f, :lb_f] = xr_diag[0, :lb_f, :lb_f] @ vh_c + xr_upper[0, :lb_f, :lb_p] @ vh_d

        # WR_E_01 = (V_01 - WR_E_00*M_E_10) * xR_E_11
        # todo if vh_r can be used instead of vh_cp[slb_c,slb_p]
        wr_upper[0, :lb_f, :lb_p] = (vh_r - wr_diag[0, :lb_f, :lb_f] @ mr_d.transpose()) @ xr_p.transpose()

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


        for idx_ib in range(1, nb_mm):
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
            if bsr:
                mr_l = mr[slb_c, slb_p]
                mr_u = mr[slb_p, slb_c]
                lg_l = lg[slb_c, slb_p]
                ll_l = ll[slb_c, slb_p]
            else:
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
            #wr_diag_c = wr_diag_rgf_c - xr_mr @ wr_upper[idx_ib - 1, :lb_vec_mm[idx_ib - 1], :lb_i]
            #wr_diag[idx_ib, :lb_i, :lb_i] = wr_diag_c

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

            #W^{\lessgtr}_E_kk = w^{\lessgtr}_E_kk + xR_E_kk*M_E_kk-1*W^{\lessgtr}_E_k-1k-1*(xR_E_kk*M_E_kk-1).H - (A^{\lessgtr}-A^{\lessgtr}.H) + (B^{\lessgtr}-B^{\lessgtr}.H)
            wg_diag_c = wg_diag_rgf_c + xr_mr @ wg_diag_p @ xr_mr_ct - ag_diff + bg_diff
            wl_diag_c = wl_diag_rgf_c + xr_mr @ wl_diag_p @ xr_mr_ct - al_diff + bl_diff
            wg_diag[idx_ib, :lb_i, :lb_i] = wg_diag_c
            wl_diag[idx_ib, :lb_i, :lb_i] = wl_diag_c

            # following code block has problems
            if idx_ib < nb_mm - 1:
                # block length i
                lb_n = lb_vec_mm[idx_ib + 1]
                # slice of current and previous block
                slb_n = slice(bmin_mm[idx_ib + 1], bmax_mm[idx_ib + 1] + 1)
                #slb_p = slice(bmin_mm[idx_ib-1], bmax_mm[idx_ib] + 1)

                # read out blocks needed
                if bsr:
                    vh_d = vh_cp[slb_n, slb_c]
                    vh_r = vh_cp[slb_p, slb_c]
                    mr_d = mr[slb_n, slb_c]
                    mr_r = mr[slb_c, slb_n]
                    lg_r = lg[slb_c, slb_n]
                    ll_r = ll[slb_c, slb_n]
                else:
                    vh_d = vh_cp[slb_n, slb_c].toarray()
                    vh_r = vh_cp[slb_p, slb_c].toarray()
                    vh_c = vh_cp[slb_c, slb_c].toarray()
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

                #xr_upper and xr_lower
                xr_upper[idx_ib, :lb_i, :lb_n] = -xr_diag_c @ mr_r @ xr_diag_rgf_n
                xr_lower[idx_ib, :lb_n, :lb_i] = -xr_diag_rgf_n @ mr_d @ xr_diag_c
                
                wr_diag_c = xr_lower[idx_ib - 1, :lb_i, :lb_n] @ vh_r + xr_diag[idx_ib, :lb_i, :lb_i] @ vh_c + \
                    xr_upper[idx_ib, :lb_i, :lb_n] @ vh_d
                wr_diag[idx_ib, :lb_i, :lb_i] = wr_diag_c

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
            else:
                vh_r = vh_cp[slb_p, slb_c].toarray()
                vh_c = vh_cp[slb_c, slb_c].toarray() - dvh_ed
                wr_diag[idx_ib, :lb_i, :lb_i] = xr_lower[idx_ib - 1, :lb_i, :lb_n] @ vh_r + xr_diag[idx_ib, :lb_i, :lb_i] @ vh_c

        for idx_ib in range(0, nb_mm):
            xr_diag[idx_ib, :, :] *= factor
            wg_diag[idx_ib, :, :] *= factor
            wl_diag[idx_ib, :, :] *= factor
            wr_diag[idx_ib, :, :] *= factor
            dosw[idx_ib] = 1j * np.trace(wr_diag[idx_ib, :, :] - wr_diag[idx_ib, :, :].conjugate().transpose())
            nEw[idx_ib] = -1j * np.trace(wl_diag[idx_ib, :, :])
            nPw[idx_ib] = 1j * np.trace(wg_diag[idx_ib, :, :])
            if idx_ib < nb_mm - 1:
                wr_upper[idx_ib, :, :] *= factor
                wg_upper[idx_ib, :, :] *= factor
                wl_upper[idx_ib, :, :] *= factor

        if ref_flag:
            # reference solution
            # Apply boundary conditions.

            # last block index
            idx_lb = nb_mm - 1
            #   slice for last block
            slb_lb = slice(bmin_mm[idx_lb],bmax_mm[idx_lb]+1)

            # first block index
            idx_fb = 0
            #   slice for first block
            slb_fb = slice(bmin_mm[idx_fb],bmax_mm[idx_fb]+1)

            vh_cp[slb_lb, slb_lb] -= dvh_ed
            vh_cp[slb_fb, slb_fb] -= dvh_sd
            mr[slb_lb, slb_lb] += dmr_ed
            mr[slb_fb, slb_fb] += dmr_sd
            lg[slb_lb, slb_lb] += dlg_ed
            lg[slb_fb, slb_fb] += dlg_sd
            ll[slb_lb, slb_lb] += dll_ed
            ll[slb_fb, slb_fb] += dll_sd

            # invert m
            
            mr_dense = mr.toarray()
            xr_ref = np.linalg.inv(mr_dense)
            wr_ref = xr_ref @ vh_cp.todense()
            # todo change L to differentiate between greater/lesser
            xr_ref_ct = xr_ref.conjugate().transpose()
            wg_ref = xr_ref @ lg @ xr_ref_ct
            wl_ref = xr_ref @ ll @ xr_ref_ct

            return xr_ref * factor, wg_ref * factor, wl_ref * factor, wr_ref * factor

    times[5] += time.perf_counter()
    return times


def rgf_w_opt_standalone(
    vh: sparse.csr_matrix,
    pg: sparse.csr_matrix,
    pl: sparse.csr_matrix,
    pr: sparse.csr_matrix,
    bmax: npt.NDArray[np.int32],
    bmin: npt.NDArray[np.int32],
    wg_diag: npt.NDArray[np.complex128],
    wg_upper: npt.NDArray[np.complex128],
    wl_diag: npt.NDArray[np.complex128],
    wl_upper: npt.NDArray[np.complex128],
    wr_diag: npt.NDArray[np.complex128],
    wr_upper: npt.NDArray[np.complex128],
    xr_diag: npt.NDArray[np.complex128],
    dvh_sd: npt.NDArray[np.complex128],
    dvh_ed: npt.NDArray[np.complex128],
    dmr_sd: npt.NDArray[np.complex128],
    dmr_ed: npt.NDArray[np.complex128],
    dlg_sd: npt.NDArray[np.complex128],
    dlg_ed: npt.NDArray[np.complex128],
    dll_sd: npt.NDArray[np.complex128],
    dll_ed: npt.NDArray[np.complex128],
    dosw: npt.NDArray[np.complex128],
    nEw: npt.NDArray[np.complex128],
    nPw: npt.NDArray[np.complex128],
    nbc: np.int64,
    ie: np.int32,
    factor: np.float64 = 1.0,
    NCpSC: np.int32 = 1,
    block_inv: bool = False,
    use_dace: bool = False,
    validate_dace: bool = False,
    ref_flag: bool = False,
    sancho_flag: bool = False,
) -> typing.Tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128], npt.NDArray[np.complex128],
                  npt.NDArray[np.complex128]]:
    """Calculates the step from the polarization to the screened interaction.
    Beyn open boundary conditions are used by default.
    The outputs (w and x) are inputs which are changed inplace.
    See the start of this file for more informations.

    Args:
        vh (sparse.csr_matrix): sparse matrix of size (#orbitals,#orbitals)
        pg (sparse.csr_matrix): sparse matrix of size (#orbitals,#orbitals)
        pl (sparse.csr_matrix): sparse matrix of size (#orbitals,#orbitals)
        pr (sparse.csr_matrix): sparse matrix of size (#orbitals,#orbitals)
        bmax (npt.NDArray[np.int32]): end idx of the blocks, vector of size number of blocks
        bmin (npt.NDArray[np.int32]): start idx of the blocks, vector of size number of blocks
        wg_diag (npt.NDArray[np.complex128]): dense matrix of size (#blocks_mm, maxblocklength_mm, maxblocklength_mm)
        wg_upper (npt.NDArray[np.complex128]): dense matrix of size (#blocks_mm-1, maxblocklength_mm, maxblocklength_mm)
        wl_diag (npt.NDArray[np.complex128]): dense matrix of size (#blocks_mm, maxblocklength_mm, maxblocklength_mm)
        wl_upper (npt.NDArray[np.complex128]): dense matrix of size (#blocks_mm-1, maxblocklength_mm, maxblocklength_mm)
        wr_diag (npt.NDArray[np.complex128]): dense matrix of size (#blocks_mm, maxblocklength_mm, maxblocklength_mm)
        wr_upper (npt.NDArray[np.complex128]): dense matrix of size (#blocks_mm-1, maxblocklength_mm, maxblocklength_mm)
        xr_diag (npt.NDArray[np.complex128]): dense matrix of size (#blocks_mm, maxblocklength_mm, maxblocklength_mm)
        dosw (npt.NDArray[np.complex128]): density of states, vector of size number of blocks
        nbc (np.int64): how block size changes after matrix multiplication
        ie (np.int32): energy index (not used)
        factor (np.float64, optional): factor to multiply the result with. Defaults to 1.0.
        ref_flag (bool, optional): If reference solution to rgf made by np.linalg.inv should be returned
        sancho_flag (bool, optional): If sancho or beyn should be used. Defaults to False.
    
    Returns:
        typing.Tuple[npt.NDArray[np.complex128], xr from inv
                  npt.NDArray[np.complex128],    wg from inv
                  npt.NDArray[np.complex128],    wl from inv
                  npt.NDArray[np.complex128]     wr from inv
                ] warning all dense arrays, only returned if ref_flag is True
    """
    # number of blocks
    nb = bmin.size
    # number of total orbitals (number of atomic orbitals)
    nao = pr.shape[0]

    bsr = False

    # vector of block lengths
    lb_vec = bmax - bmin + 1
    # length of first and last block
    lb_start = lb_vec[0]
    lb_end = lb_vec[nb - 1]

    # block sizes after matrix multiplication
    bmax_mm = bmax[nbc - 1:nb:nbc]
    bmin_mm = bmin[0:nb:nbc]
    # number of blocks after matrix multiplication
    nb_mm = bmin_mm.size
    # vector of block lengths after matrix multiplication
    lb_vec_mm = bmax_mm - bmin_mm + 1
    # max block size after matrix multiplication
    lb_max_mm = np.max(lb_vec_mm)

    # slice block start diagonal and slice block start off diagonal
    slb_sd = slice(0, lb_start)
    slb_so = slice(lb_start, 2 * lb_start)
    # slice block end diagonal and slice block end off diagonal
    slb_ed = slice(nao - lb_end, nao)
    slb_eo = slice(nao - 2 * lb_end, nao - lb_end)

    vh_cp = vh.copy()


    # calculate L^{\lessgtr}\left(E\right)
    vh_cp_ct = vh_cp.conjugate().transpose()
    lg = vh_cp @ pg @ vh_cp_ct
    ll = vh_cp @ pl @ vh_cp_ct
    # calculate M^{r}\left(E\right)

    mr = sparse.identity(nao, format="csr") - vh_cp @ pr

    # create buffer for results

    # not true inverse, but build up inverses from either corner
    xr_diag_rgf = np.zeros((nb_mm, lb_max_mm, lb_max_mm), dtype=np.complex128)
    wg_diag_rgf = np.zeros((nb_mm, lb_max_mm, lb_max_mm), dtype=np.complex128)
    wl_diag_rgf = np.zeros((nb_mm, lb_max_mm, lb_max_mm), dtype=np.complex128)
    wr_diag_rgf = np.zeros((nb_mm, lb_max_mm, lb_max_mm), dtype=np.complex128)

    # True inverse M^-1 off-diagonal blocks
    xr_upper = np.zeros((nb_mm - 1, lb_max_mm, lb_max_mm), dtype=np.complex128)
    xr_lower = np.zeros((nb_mm - 1, lb_max_mm, lb_max_mm), dtype=np.complex128)

    # todo
    # find out if IdE, DOS, dWL, dWG are needed

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

    # last block index
    idx_lb = nb_mm - 1
    # slice for last block
    slb_lb = slice(bmin_mm[idx_lb], bmax_mm[idx_lb] + 1)
    # length last block
    lb_lb = lb_vec_mm[idx_lb]

    # x^{r}_E_nn = M_E_nn^{-1}
    if bsr:
        xr_lb = np.linalg.inv(mr[slb_lb, slb_lb] + dmr_ed)
        xr_lb_ct = xr_lb.conjugate().transpose()
        xr_diag_rgf[idx_lb, :lb_lb, :lb_lb] = xr_lb

        # w^{>}_E_nn = x^{r}_E_nn * L^{>}_E_nn * (x^{r}_E_nn).H
        wg_lb = xr_lb @ (lg[slb_lb, slb_lb] + dlg_ed) @ xr_lb_ct
        wg_diag_rgf[idx_lb, :lb_lb, :lb_lb] = wg_lb

        # w^{<}_E_nn = x^{r}_E_nn * L^{<}_E_nn * (x^{r}_E_nn).H
        wl_lb = xr_lb @ (ll[slb_lb, slb_lb] + dll_ed) @ xr_lb_ct
        wl_diag_rgf[idx_lb, :lb_lb, :lb_lb] = wl_lb

        # wR_E_nn = xR_E_nn * V_nn
        wr_lb = xr_lb @ (vh_cp[slb_lb, slb_lb] - dvh_ed)
        wr_diag_rgf[idx_lb, :lb_lb, :lb_lb] = wr_lb
    else:
        xr_lb = np.linalg.inv(mr[slb_lb, slb_lb].toarray() + dmr_ed)
        xr_lb_ct = xr_lb.conjugate().transpose()
        xr_diag_rgf[idx_lb, :lb_lb, :lb_lb] = xr_lb

        # w^{>}_E_nn = x^{r}_E_nn * L^{>}_E_nn * (x^{r}_E_nn).H
        wg_lb = xr_lb @ (lg[slb_lb, slb_lb].toarray() + dlg_ed) @ xr_lb_ct
        wg_diag_rgf[idx_lb, :lb_lb, :lb_lb] = wg_lb

        # w^{<}_E_nn = x^{r}_E_nn * L^{<}_E_nn * (x^{r}_E_nn).H
        wl_lb = xr_lb @ (ll[slb_lb, slb_lb].toarray() + dll_ed) @ xr_lb_ct
        wl_diag_rgf[idx_lb, :lb_lb, :lb_lb] = wl_lb

        # wR_E_nn = xR_E_nn * V_nn
        wr_lb = xr_lb @ (vh_cp[slb_lb, slb_lb].toarray() - dvh_ed)
        wr_diag_rgf[idx_lb, :lb_lb, :lb_lb] = wr_lb

    # save the diagonal blocks from the previous step
    xr_c = xr_lb
    wg_c = wg_lb
    wl_c = wl_lb

    # loop over all blocks from last to first
    for idx_ib in reversed(range(0, nb_mm - 1)):
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
        if bsr:
            mr_c = mr[slb_c, slb_c]
            mr_r = mr[slb_c, slb_p]
            mr_d = mr[slb_p, slb_c]
            vh_c = vh_cp[slb_c, slb_c]
            vh_d = vh_cp[slb_p, slb_c]
            lg_c = lg[slb_c, slb_c]
            lg_d = lg[slb_p, slb_c]
            ll_c = ll[slb_c, slb_c]
            ll_d = ll[slb_p, slb_c]
        else:
            mr_c = mr[slb_c, slb_c].toarray()
            mr_r = mr[slb_c, slb_p].toarray()
            mr_d = mr[slb_p, slb_c].toarray()
            vh_c = vh_cp[slb_c, slb_c].toarray()
            vh_d = vh_cp[slb_p, slb_c].toarray()
            lg_c = lg[slb_c, slb_c].toarray()
            lg_d = lg[slb_p, slb_c].toarray()
            ll_c = ll[slb_c, slb_c].toarray()
            ll_d = ll[slb_p, slb_c].toarray()

        if idx_ib == 0:
            lg_c += dlg_sd
            ll_c += dll_sd
            vh_c -= dvh_sd
            mr_c += dmr_sd
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
    if bsr:
        vh_r = vh_cp[slb_c, slb_p]
        lg_r = lg[slb_c, slb_p]
        ll_r = ll[slb_c, slb_p]
    else:
        vh_r = vh_cp[slb_c, slb_p].toarray()
        vh_d = vh_cp[slb_p, slb_c].toarray()
        lg_r = lg[slb_c, slb_p].toarray()
        ll_r = ll[slb_c, slb_p].toarray()

    xr_mr = xr_p @ mr_d
    xr_mr_ct = xr_mr.conjugate().transpose()

    # XR_E_00 = xR_E_00
    xr_diag[0, :lb_f, :lb_f] = xr_diag_rgf[0, :lb_f, :lb_f]
    xr_upper[0, :lb_f, :lb_p] = -xr_diag_rgf[0, :lb_f, :lb_f] @ mr_r @ xr_p
    xr_lower[0, :lb_p, :lb_f] = -xr_p @ mr_d @ xr_diag_rgf[0, :lb_f, :lb_f]

    # WR_E_00 = wR_E_00
    #wr_diag[0, :lb_f, :lb_f] = wr_diag_rgf[0, :lb_f, :lb_f]
    wr_diag[0, :lb_f, :lb_f] = xr_diag[0, :lb_f, :lb_f] @ vh_c + xr_upper[0, :lb_f, :lb_p] @ vh_d

    # WR_E_01 = (V_01 - WR_E_00*M_E_10) * xR_E_11
    # todo if vh_r can be used instead of vh_cp[slb_c,slb_p]
    wr_upper[0, :lb_f, :lb_p] = (vh_r - wr_diag[0, :lb_f, :lb_f] @ mr_d.transpose()) @ xr_p.transpose()

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


    for idx_ib in range(1, nb_mm):
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
        if bsr:
            mr_l = mr[slb_c, slb_p]
            mr_u = mr[slb_p, slb_c]
            lg_l = lg[slb_c, slb_p]
            ll_l = ll[slb_c, slb_p]
        else:
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
        #wr_diag_c = wr_diag_rgf_c - xr_mr @ wr_upper[idx_ib - 1, :lb_vec_mm[idx_ib - 1], :lb_i]
        #wr_diag[idx_ib, :lb_i, :lb_i] = wr_diag_c

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

        #W^{\lessgtr}_E_kk = w^{\lessgtr}_E_kk + xR_E_kk*M_E_kk-1*W^{\lessgtr}_E_k-1k-1*(xR_E_kk*M_E_kk-1).H - (A^{\lessgtr}-A^{\lessgtr}.H) + (B^{\lessgtr}-B^{\lessgtr}.H)
        wg_diag_c = wg_diag_rgf_c + xr_mr @ wg_diag_p @ xr_mr_ct - ag_diff + bg_diff
        wl_diag_c = wl_diag_rgf_c + xr_mr @ wl_diag_p @ xr_mr_ct - al_diff + bl_diff
        wg_diag[idx_ib, :lb_i, :lb_i] = wg_diag_c
        wl_diag[idx_ib, :lb_i, :lb_i] = wl_diag_c

        # following code block has problems
        if idx_ib < nb_mm - 1:
            # block length i
            lb_n = lb_vec_mm[idx_ib + 1]
            # slice of current and previous block
            slb_n = slice(bmin_mm[idx_ib + 1], bmax_mm[idx_ib + 1] + 1)
            #slb_p = slice(bmin_mm[idx_ib-1], bmax_mm[idx_ib] + 1)

            # read out blocks needed
            if bsr:
                vh_d = vh_cp[slb_n, slb_c]
                vh_r = vh_cp[slb_p, slb_c]
                mr_d = mr[slb_n, slb_c]
                mr_r = mr[slb_c, slb_n]
                lg_r = lg[slb_c, slb_n]
                ll_r = ll[slb_c, slb_n]
            else:
                vh_d = vh_cp[slb_n, slb_c].toarray()
                vh_r = vh_cp[slb_p, slb_c].toarray()
                vh_c = vh_cp[slb_c, slb_c].toarray()
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

            #xr_upper and xr_lower
            xr_upper[idx_ib, :lb_i, :lb_n] = -xr_diag_c @ mr_r @ xr_diag_rgf_n
            xr_lower[idx_ib, :lb_n, :lb_i] = -xr_diag_rgf_n @ mr_d @ xr_diag_c
            
            wr_diag_c = xr_lower[idx_ib - 1, :lb_i, :lb_n] @ vh_r + xr_diag[idx_ib, :lb_i, :lb_i] @ vh_c + \
                xr_upper[idx_ib, :lb_i, :lb_n] @ vh_d
            wr_diag[idx_ib, :lb_i, :lb_i] = wr_diag_c

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
        else:
            vh_r = vh_cp[slb_p, slb_c].toarray()
            vh_c = vh_cp[slb_c, slb_c].toarray() - dvh_ed
            wr_diag[idx_ib, :lb_i, :lb_i] = xr_lower[idx_ib - 1, :lb_i, :lb_n] @ vh_r + xr_diag[idx_ib, :lb_i, :lb_i] @ vh_c

    for idx_ib in range(0, nb_mm):
        xr_diag[idx_ib, :, :] *= factor
        wg_diag[idx_ib, :, :] *= factor
        wl_diag[idx_ib, :, :] *= factor
        wr_diag[idx_ib, :, :] *= factor
        dosw[idx_ib] = 1j * np.trace(wr_diag[idx_ib, :, :] - wr_diag[idx_ib, :, :].conjugate().transpose())
        nEw[idx_ib] = -1j * np.trace(wl_diag[idx_ib, :, :])
        nPw[idx_ib] = 1j * np.trace(wg_diag[idx_ib, :, :])
        if idx_ib < nb_mm - 1:
            wr_upper[idx_ib, :, :] *= factor
            wg_upper[idx_ib, :, :] *= factor
            wl_upper[idx_ib, :, :] *= factor
