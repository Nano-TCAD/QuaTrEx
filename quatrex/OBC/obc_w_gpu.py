"""
Containing functions to apply the OBC for the screened interaction


- _ct is for the original matrix complex conjugated
- _mm how certain sizes after matrix multiplication, because a block tri diagonal gets two new non zero off diagonals
- _s stands for values related to the left/start/top contact block
- _e stands for values related to the right/end/bottom contact block
- _d stands for diagonal block (X00/NN in matlab)
- _u stands for upper diagonal block (X01/NN1 in matlab)
- _l stands for lower diagonal block (X10/N1N in matlab) 
- exception _l/_r can stand for left/right in context of condition of OBC
"""
import numpy as np
from cupyx.scipy import sparse
from OBC import beyn_gpu
from OBC import sancho
from OBC import dL_OBC_eigenmode_gpu
import typing

def obc_w(
        pg: sparse.csr_matrix,
        pl: sparse.csr_matrix,
        pr: sparse.csr_matrix,
        vh: sparse.csr_matrix,
        sr: sparse.csr_matrix,
        lg: sparse.csr_matrix,
        ll: sparse.csr_matrix,
        slicing_obj: object,
        sancho_flag: bool = False
) -> typing.Tuple[bool, sparse.csr_matrix]:
    """
    Apply the OBC fro the screened interaction calculation.
    For a single energy point.

    Args:
        pg (sparse.csr_matrix): Greater polarization, vector of sparse matrices 
        pl (sparse.csr_matrix): Lesser polarization, vector of sparse matrices
        pr (sparse.csr_matrix): Retarded polarization, vector of sparse matrices
        vh (sparse.csr_matrix): Effective interaction, vector of sparse matrices
        sr (sparse.csr_matrix): Helper variable for inversion, vector of sparse matrices
        lg (sparse.csr_matrix): Helper variable for inversion, vector of sparse matrices
        ll (sparse.csr_matrix): Helper variable for inversion, vector of sparse matrices
        slicing_obj   (object): Contains 
        sancho_flag (bool, optional): If sancho OBC should be used, else beyn. Default false
    Returns:
        typing.Tuple[bool, sparse.csr_matrix]: If the inversion should be computed afterwards and mr
    """

    # from G^{\lessgtr}\left(E\right) / G^{r}\left(E\right)
    # (diagonal, upper, lower block of the start block at the left)
    pg_sd, pg_su, pg_sl = dL_OBC_eigenmode_gpu.stack_px(pg[slicing_obj.slb_sd,slicing_obj.slb_sd], pg[slicing_obj.slb_sd,slicing_obj.slb_so], slicing_obj.nbc)
    pl_sd, pl_su, pl_sl = dL_OBC_eigenmode_gpu.stack_px(pl[slicing_obj.slb_sd,slicing_obj.slb_sd], pl[slicing_obj.slb_sd,slicing_obj.slb_so], slicing_obj.nbc)
    pr_sd, pr_su, pr_sl = dL_OBC_eigenmode_gpu.stack_pr(pr[slicing_obj.slb_sd,slicing_obj.slb_sd], pr[slicing_obj.slb_sd,slicing_obj.slb_so], slicing_obj.nbc)
    # (diagonal, upper, lower block of the end block at the right)
    pg_ed, pg_eu, pg_el = dL_OBC_eigenmode_gpu.stack_px(pg[slicing_obj.slb_ed,slicing_obj.slb_ed], -pg[slicing_obj.slb_ed,slicing_obj.slb_eo].conjugate().transpose(), slicing_obj.nbc)
    pl_ed, pl_eu, pl_el = dL_OBC_eigenmode_gpu.stack_px(pl[slicing_obj.slb_ed,slicing_obj.slb_ed], -pl[slicing_obj.slb_ed,slicing_obj.slb_eo].conjugate().transpose(), slicing_obj.nbc)
    pr_ed, pr_eu, pr_el = dL_OBC_eigenmode_gpu.stack_pr(pr[slicing_obj.slb_ed,slicing_obj.slb_ed], pr[slicing_obj.slb_ed,slicing_obj.slb_eo].transpose(), slicing_obj.nbc)

    # from \hat(V)\left(E\right) / G^{r}\left(E\right)
    # (diagonal, upper, lower block of the start block at the left)
    vh_sd, vh_su, vh_sl = dL_OBC_eigenmode_gpu.stack_vh(vh[slicing_obj.slb_sd,slicing_obj.slb_sd], vh[slicing_obj.slb_sd,slicing_obj.slb_so], slicing_obj.nbc)
    # (diagonal, upper, lower block of the end block at the right)
    vh_ed, vh_eu, vh_el = dL_OBC_eigenmode_gpu.stack_vh(vh[slicing_obj.slb_ed,slicing_obj.slb_ed], vh[slicing_obj.slb_ed,slicing_obj.slb_eo].conjugate().transpose(), slicing_obj.nbc)

    # from L^{\lessgtr}\left(E\right)
    # (diagonal, upper, lower block of the start block at the left)
    lg_sd, lg_su, _ = dL_OBC_eigenmode_gpu.stack_lx(vh_sd, vh_su, vh_sl, pg_sd, pg_su, pg_sl)
    ll_sd, ll_su, _ = dL_OBC_eigenmode_gpu.stack_lx(vh_sd, vh_su, vh_sl, pl_sd, pl_su, pl_sl)

    # (diagonal, upper, lower block of the end block at the right)
    lg_ed, _, lg_el = dL_OBC_eigenmode_gpu.stack_lx(vh_ed, vh_eu, vh_el, pg_ed, pg_eu, pg_el)
    ll_ed, _, ll_el = dL_OBC_eigenmode_gpu.stack_lx(vh_ed, vh_eu, vh_el, pl_ed, pl_eu, pl_el)

    # from M^{r}\left(E\right)
    # (diagonal, upper, lower block of the start block at the left)
    mr_sd, mr_su, mr_sl = dL_OBC_eigenmode_gpu.stack_mr(vh_sd, vh_su, vh_sl, pr_sd, pr_su, pr_sl)
    # (diagonal, upper, lower block of the end block at the right)
    mr_ed, mr_eu, mr_el = dL_OBC_eigenmode_gpu.stack_mr(vh_ed, vh_eu, vh_el, pr_ed, pr_eu, pr_el)

    # correction for matrix multiplication--------------------------------------

    # correct first and last block to account for the contacts in multiplication
    # first block
    sr[slicing_obj.slb_sd_mm, slicing_obj.slb_sd_mm] = sr[slicing_obj.slb_sd_mm, slicing_obj.slb_sd_mm] + vh_sl @ pr_su
    # last block
    sr[slicing_obj.slb_ed_mm, slicing_obj.slb_ed_mm] = sr[slicing_obj.slb_ed_mm, slicing_obj.slb_ed_mm] + vh_eu @ pr_el

    # calculate M^{r}\left(E\right) (only after above correction)
    mr = sparse.identity(slicing_obj.nao, format="csr") - sr


    # correct first and last block to account for the contacts in multiplication
    # first block
    # L^{\lessgtr}_E_00 = V_10*P^{\lessgtr}_E_00*V_01 + V_10*P^{\lessgtr}_E_01*V_00 + V_00*P^{\lessgtr}_E_10*V_01
    lg[slicing_obj.slb_sd_mm, slicing_obj.slb_sd_mm] = lg[slicing_obj.slb_sd_mm, slicing_obj.slb_sd_mm] + vh_sl @ pg_sd @ vh_su
    lg[slicing_obj.slb_sd_mm, slicing_obj.slb_sd_mm] = lg[slicing_obj.slb_sd_mm, slicing_obj.slb_sd_mm] + vh_sl @ pg_su @ vh_sd
    lg[slicing_obj.slb_sd_mm, slicing_obj.slb_sd_mm] = lg[slicing_obj.slb_sd_mm, slicing_obj.slb_sd_mm] + vh_sd @ pg_sl @ vh_su

    ll[slicing_obj.slb_sd_mm, slicing_obj.slb_sd_mm] = ll[slicing_obj.slb_sd_mm, slicing_obj.slb_sd_mm] + vh_sl @ pl_sd @ vh_su
    ll[slicing_obj.slb_sd_mm, slicing_obj.slb_sd_mm] = ll[slicing_obj.slb_sd_mm, slicing_obj.slb_sd_mm] + vh_sl @ pl_su @ vh_sd
    ll[slicing_obj.slb_sd_mm, slicing_obj.slb_sd_mm] = ll[slicing_obj.slb_sd_mm, slicing_obj.slb_sd_mm] + vh_sd @ pl_sl @ vh_su

    # last block
    # L^{\lessgtr}_E_nn = V_nn*PL_E_n-1n*V_nn-1 + V_n-1n*PL_E_nn-1*V_nn + V_n-1n*PL_E_nn*V_nn-1
    lg[slicing_obj.slb_ed_mm, slicing_obj.slb_ed_mm] = lg[slicing_obj.slb_ed_mm, slicing_obj.slb_ed_mm] + vh_ed @ pg_eu @ vh_el
    lg[slicing_obj.slb_ed_mm, slicing_obj.slb_ed_mm] = lg[slicing_obj.slb_ed_mm, slicing_obj.slb_ed_mm] + vh_eu @ pg_el @ vh_ed
    lg[slicing_obj.slb_ed_mm, slicing_obj.slb_ed_mm] = lg[slicing_obj.slb_ed_mm, slicing_obj.slb_ed_mm] + vh_eu @ pg_ed @ vh_el

    ll[slicing_obj.slb_ed_mm, slicing_obj.slb_ed_mm] = ll[slicing_obj.slb_ed_mm, slicing_obj.slb_ed_mm] + vh_ed @ pl_eu @ vh_el
    ll[slicing_obj.slb_ed_mm, slicing_obj.slb_ed_mm] = ll[slicing_obj.slb_ed_mm, slicing_obj.slb_ed_mm] + vh_eu @ pl_el @ vh_ed
    ll[slicing_obj.slb_ed_mm, slicing_obj.slb_ed_mm] = ll[slicing_obj.slb_ed_mm, slicing_obj.slb_ed_mm] + vh_eu @ pl_ed @ vh_el


    # correction for the matrix inverse calculations----------------------------

    # conditions about convergence or meaningful results from
    # boundary correction calculations
    cond_l = 0.0
    cond_r = 0.0

    # correction for first block
    if not sancho_flag:
        _, cond_l, dxr_sd, dmr_sd, _ = beyn_gpu.beyn(
                                                mr_sd.toarray(),
                                                mr_su.toarray(),
                                                mr_sl.toarray(),
                                                slicing_obj.imag_lim, slicing_obj.rr, "L")
        if not np.isnan(cond_l):
            dvh_sd = mr_sl @ dxr_sd @ vh_su

    if np.isnan(cond_l) or sancho_flag:
        dxr_sd, dmr_sd, dvh_sd, cond_l = sancho.open_boundary_conditions(
                                                mr_sd.toarray(),
                                                mr_sl.toarray(),
                                                mr_su.toarray(),
                                                vh_su.toarray())
    # beyn gave a meaningful result
    if not np.isnan(cond_l):
        dlg_sd, dll_sd = dL_OBC_eigenmode_gpu.get_dl_obc(
                                                dxr_sd,
                                                lg_sd.toarray(),
                                                lg_su.toarray(),
                                                ll_sd.toarray(),
                                                ll_su.toarray(),
                                                mr_sl.toarray(),
                                                blk="L")

        # problem efficiency warning
        mr[slicing_obj.slb_sd_mm,slicing_obj.slb_sd_mm]  = mr[slicing_obj.slb_sd_mm,slicing_obj.slb_sd_mm] - dmr_sd
        vh[slicing_obj.slb_sd_mm,slicing_obj.slb_sd_mm]  = vh[slicing_obj.slb_sd_mm,slicing_obj.slb_sd_mm] - dvh_sd

        if not np.isnan(dll_sd).any():
            lg[slicing_obj.slb_sd_mm,slicing_obj.slb_sd_mm] = lg[slicing_obj.slb_sd_mm,slicing_obj.slb_sd_mm] + dlg_sd
            ll[slicing_obj.slb_sd_mm,slicing_obj.slb_sd_mm] = ll[slicing_obj.slb_sd_mm,slicing_obj.slb_sd_mm] + dll_sd
        else:
            cond_l = np.nan


    # correction for last block
    if not sancho_flag:
        _, cond_r, dxr_ed, dmr_ed, _ = beyn_gpu.beyn(
                                                mr_ed.toarray(),
                                                mr_eu.toarray(),
                                                mr_el.toarray(),
                                                slicing_obj.imag_lim, slicing_obj.rr, "R")
        if not np.isnan(cond_r):
            dvh_ed = mr_eu @ dxr_ed @ vh_el

    if np.isnan(cond_r) or sancho_flag:
        dxr_ed, dmr_ed, dvh_ed, cond_r = sancho.open_boundary_conditions(
                                                mr_ed.toarray(),
                                                mr_eu.toarray(),
                                                mr_el.toarray(),
                                                vh_el.toarray())
    # condR = np.nan
    if not np.isnan(cond_r):
        dlg_ed, dll_ed = dL_OBC_eigenmode_gpu.get_dl_obc(
                                                dxr_ed,
                                                lg_ed.toarray(),
                                                lg_el.toarray(),
                                                ll_ed.toarray(),
                                                ll_el.toarray(),
                                                mr_eu.toarray(),
                                                blk="R")

        mr[slicing_obj.slb_ed_mm,slicing_obj.slb_ed_mm]  = mr[slicing_obj.slb_ed_mm,slicing_obj.slb_ed_mm] - dmr_ed
        vh[slicing_obj.slb_ed_mm,slicing_obj.slb_ed_mm]  = vh[slicing_obj.slb_ed_mm,slicing_obj.slb_ed_mm] - dvh_ed

        if not np.isnan(dll_ed).any():
            lg[slicing_obj.slb_ed_mm,slicing_obj.slb_ed_mm] = lg[slicing_obj.slb_ed_mm,slicing_obj.slb_ed_mm] + dlg_ed
            ll[slicing_obj.slb_ed_mm,slicing_obj.slb_ed_mm] = ll[slicing_obj.slb_ed_mm,slicing_obj.slb_ed_mm] + dll_ed
        else:
            cond_r = np.nan

    return not np.isnan(cond_r) and not np.isnan(cond_l), mr
