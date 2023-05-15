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
import numpy.typing as npt
from scipy import sparse
from OBC import beyn_cpu
from OBC import sancho
from OBC import dL_OBC_eigenmode_cpu
import typing

def obc_w_sl(
    vh: sparse.csr_matrix,
    pg: sparse.csr_matrix,
    pl: sparse.csr_matrix,
    pr: sparse.csr_matrix
) -> typing.Tuple[sparse.csr_matrix,
                  sparse.csr_matrix,
                  sparse.csr_matrix]:
    """
    Calculates for the additional helper variables.

    Args:
        vh (npt.NDArray[np.complex128]): Effective interaction
        pg_vec (npt.NDArray[np.complex128]): Greater Polarization, vector of sparse matrices
        pl_vec (npt.NDArray[np.complex128]): Lesser Polarization, vector of sparse matrices
        pr_vec (npt.NDArray[np.complex128]): Retarded Polarization, vector of sparse matrices

    Returns:
        typing.Tuple[sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix]: S^{r}\left(E\right), L^{>}\left(E\right), L^{<}\left(E\right)
    """

    # create output vector

    vh_ct = vh.conjugate().transpose()

    # calculate S^{r}\left(E\right)
    sr = vh @ pr

    # calculate L^{\lessgtr}\left(E\right)
    lg = vh @ pg @ vh_ct
    ll = vh @ pl @ vh_ct

    return sr, lg, ll


def obc_w_sc(
    pg: sparse.csr_matrix,
    pl: sparse.csr_matrix,
    pr: sparse.csr_matrix,
    vh: sparse.csr_matrix,
    sr: sparse.csr_matrix,
    lg: sparse.csr_matrix,
    ll: sparse.csr_matrix,
    bmax: npt.NDArray[np.int32],
    bmin: npt.NDArray[np.int32],
    nbc: int,
) -> sparse.csr_matrix:
    # number of blocks
    nb = bmin.size
    # number of total orbitals (number of atomic orbitals)
    nao = vh.shape[0]

    # vector of block lengths
    lb_vec = bmax - bmin + 1
    # length of first and last block
    lb_start = lb_vec[0]
    lb_end = lb_vec[nb-1]

    # slice block start diagonal and slice block start off diagonal
    slb_sd = slice(0,lb_start)
    slb_so = slice(lb_start,2*lb_start)
    # slice block end diagonal and slice block end off diagonal
    slb_ed = slice(nao-lb_end,nao)
    slb_eo = slice(nao-2*lb_end,nao-lb_end)

    # slice block start diagonal
    # for block after matrix multiplication
    slb_sd_mm = slice(bmin[0], bmax[nbc-1] + 1)
    # slice block end diagonal
    # for block after matrix multiplication
    slb_ed_mm = slice(bmin[nb - nbc], nao)


    # from G^{\lessgtr}\left(E\right) / G^{r}\left(E\right)
    # (diagonal, upper, lower block of the start block at the left)
    pg_sd, pg_su, pg_sl = dL_OBC_eigenmode_cpu.stack_px(pg[slb_sd,slb_sd], pg[slb_sd,slb_so], nbc)
    pl_sd, pl_su, pl_sl = dL_OBC_eigenmode_cpu.stack_px(pl[slb_sd,slb_sd], pl[slb_sd,slb_so], nbc)
    _, pr_su, _ = dL_OBC_eigenmode_cpu.stack_pr(pr[slb_sd,slb_sd], pr[slb_sd,slb_so], nbc)
    # (diagonal, upper, lower block of the end block at the right)
    pg_ed, pg_eu, pg_el = dL_OBC_eigenmode_cpu.stack_px(pg[slb_ed,slb_ed], -pg[slb_ed,slb_eo].conjugate().transpose(), nbc)
    pl_ed, pl_eu, pl_el = dL_OBC_eigenmode_cpu.stack_px(pl[slb_ed,slb_ed], -pl[slb_ed,slb_eo].conjugate().transpose(), nbc)
    _, _, pr_el = dL_OBC_eigenmode_cpu.stack_pr(pr[slb_ed,slb_ed], pr[slb_ed,slb_eo].transpose(), nbc)

    # from \hat(V)\left(E\right) / G^{r}\left(E\right)
    # (diagonal, upper, lower block of the start block at the left)
    vh_sd, vh_su, vh_sl = dL_OBC_eigenmode_cpu.stack_vh(vh[slb_sd,slb_sd], vh[slb_sd,slb_so], nbc)
    # (diagonal, upper, lower block of the end block at the right)
    vh_ed, vh_eu, vh_el = dL_OBC_eigenmode_cpu.stack_vh(vh[slb_ed,slb_ed], vh[slb_ed,slb_eo].conjugate().transpose(), nbc)


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

    return mr

def obc_w_beyn(
        pr: sparse.csr_matrix,
        vh: sparse.csr_matrix,
        bmax: npt.NDArray[np.int32],
        bmin: npt.NDArray[np.int32],
        nbc: int,
        sancho_flag: bool = False
):
    # limit for beyn
    imag_lim = 1e-4
    # todo find out what rr/R is
    # (R only is against the style guide)
    rr = 1e12

    # number of blocks
    nb = bmin.size
    # number of total orbitals (number of atomic orbitals)
    nao = vh.shape[0]

    # vector of block lengths
    lb_vec = bmax - bmin + 1
    # length of first and last block
    lb_start = lb_vec[0]
    lb_end = lb_vec[nb-1]

    # slice block start diagonal and slice block start off diagonal
    slb_sd = slice(0,lb_start)
    slb_so = slice(lb_start,2*lb_start)
    # slice block end diagonal and slice block end off diagonal
    slb_ed = slice(nao-lb_end,nao)
    slb_eo = slice(nao-2*lb_end,nao-lb_end)


    # from G^{\lessgtr}\left(E\right) / G^{r}\left(E\right)
    # (diagonal, upper, lower block of the start block at the left)
    pr_sd, pr_su, pr_sl = dL_OBC_eigenmode_cpu.stack_pr(pr[slb_sd,slb_sd], pr[slb_sd,slb_so], nbc)
    # (diagonal, upper, lower block of the end block at the right)
    pr_ed, pr_eu, pr_el = dL_OBC_eigenmode_cpu.stack_pr(pr[slb_ed,slb_ed], pr[slb_ed,slb_eo].transpose(), nbc)

    # from \hat(V)\left(E\right) / G^{r}\left(E\right)
    # (diagonal, upper, lower block of the start block at the left)
    vh_sd, vh_su, vh_sl = dL_OBC_eigenmode_cpu.stack_vh(vh[slb_sd,slb_sd], vh[slb_sd,slb_so], nbc)
    # (diagonal, upper, lower block of the end block at the right)
    vh_ed, vh_eu, vh_el = dL_OBC_eigenmode_cpu.stack_vh(vh[slb_ed,slb_ed], vh[slb_ed,slb_eo].conjugate().transpose(), nbc)

    # from M^{r}\left(E\right)
    # (diagonal, upper, lower block of the start block at the left)
    mr_sd, mr_su, mr_sl = dL_OBC_eigenmode_cpu.stack_mr(vh_sd, vh_su, vh_sl, pr_sd, pr_su, pr_sl)
    # (diagonal, upper, lower block of the end block at the right)
    mr_ed, mr_eu, mr_el = dL_OBC_eigenmode_cpu.stack_mr(vh_ed, vh_eu, vh_el, pr_ed, pr_eu, pr_el)

    # correction for the matrix inverse calculations----------------------------

    # conditions about convergence or meaningful results from
    # boundary correction calculations
    cond_l = 0.0
    cond_r = 0.0

    # correction for first block
    if not sancho_flag:
        _, cond_l, dxr_sd, dmr_sd, _ = beyn_cpu.beyn(
                                                mr_sd.toarray(),
                                                mr_su.toarray(),
                                                mr_sl.toarray(),
                                                imag_lim, rr, "L")
        if not np.isnan(cond_l):
            dvh_sd = mr_sl @ dxr_sd @ vh_su

    if np.isnan(cond_l) or sancho_flag:
        dxr_sd, dmr_sd, dvh_sd, cond_l = sancho.open_boundary_conditions(
                                                mr_sd.toarray(),
                                                mr_sl.toarray(),
                                                mr_su.toarray(),
                                                vh_su.toarray())
    
    # correction for last block
    if not sancho_flag:
        _, cond_r, dxr_ed, dmr_ed, _ = beyn_cpu.beyn(
                                                mr_ed.toarray(),
                                                mr_eu.toarray(),
                                                mr_el.toarray(),
                                                imag_lim, rr, "R")
        if not np.isnan(cond_r):
            dvh_ed = mr_eu @ dxr_ed @ vh_el

    if np.isnan(cond_r) or sancho_flag:
        dxr_ed, dmr_ed, dvh_ed, cond_r = sancho.open_boundary_conditions(
                                                mr_ed.toarray(),
                                                mr_eu.toarray(),
                                                mr_el.toarray(),
                                                vh_el.toarray())

    return cond_r, cond_l, dxr_sd, dxr_ed, dmr_sd, dmr_ed, dvh_sd, dvh_ed

def obc_w_dl(
        pg: sparse.csr_matrix,
        pl: sparse.csr_matrix,
        pr: sparse.csr_matrix,
        vh: sparse.csr_matrix,
        mr: sparse.csr_matrix,
        lg: sparse.csr_matrix,
        ll: sparse.csr_matrix,
        dxr_sd: np.ndarray,
        dxr_ed: np.ndarray,
        dmr_sd: np.ndarray,
        dmr_ed: np.ndarray,
        dvh_sd: np.ndarray,
        dvh_ed: np.ndarray,
        bmax: np.ndarray,
        bmin: np.ndarray,
        nbc: int,
        cond_l: float,
        cond_r: float
):

    # number of blocks
    nb = bmin.size
    # number of total orbitals (number of atomic orbitals)
    nao = vh.shape[0]

    # vector of block lengths
    lb_vec = bmax - bmin + 1
    # length of first and last block
    lb_start = lb_vec[0]
    lb_end = lb_vec[nb-1]

    # slice block start diagonal and slice block start off diagonal
    slb_sd = slice(0,lb_start)
    slb_so = slice(lb_start,2*lb_start)
    # slice block end diagonal and slice block end off diagonal
    slb_ed = slice(nao-lb_end,nao)
    slb_eo = slice(nao-2*lb_end,nao-lb_end)

    # slice block start diagonal
    # for block after matrix multiplication
    slb_sd_mm = slice(bmin[0], bmax[nbc-1] + 1)
    # slice block end diagonal
    # for block after matrix multiplication
    slb_ed_mm = slice(bmin[nb - nbc], nao)


    # from G^{\lessgtr}\left(E\right) / G^{r}\left(E\right)
    # (diagonal, upper, lower block of the start block at the left)
    pg_sd, pg_su, pg_sl = dL_OBC_eigenmode_cpu.stack_px(pg[slb_sd,slb_sd], pg[slb_sd,slb_so], nbc)
    pl_sd, pl_su, pl_sl = dL_OBC_eigenmode_cpu.stack_px(pl[slb_sd,slb_sd], pl[slb_sd,slb_so], nbc)
    pr_sd, pr_su, pr_sl = dL_OBC_eigenmode_cpu.stack_pr(pr[slb_sd,slb_sd], pr[slb_sd,slb_so], nbc)
    # (diagonal, upper, lower block of the end block at the right)
    pg_ed, pg_eu, pg_el = dL_OBC_eigenmode_cpu.stack_px(pg[slb_ed,slb_ed], -pg[slb_ed,slb_eo].conjugate().transpose(), nbc)
    pl_ed, pl_eu, pl_el = dL_OBC_eigenmode_cpu.stack_px(pl[slb_ed,slb_ed], -pl[slb_ed,slb_eo].conjugate().transpose(), nbc)
    pr_ed, pr_eu, pr_el = dL_OBC_eigenmode_cpu.stack_pr(pr[slb_ed,slb_ed], pr[slb_ed,slb_eo].transpose(), nbc)

    # from \hat(V)\left(E\right) / G^{r}\left(E\right)
    # (diagonal, upper, lower block of the start block at the left)
    vh_sd, vh_su, vh_sl = dL_OBC_eigenmode_cpu.stack_vh(vh[slb_sd,slb_sd], vh[slb_sd,slb_so], nbc)
    # (diagonal, upper, lower block of the end block at the right)
    vh_ed, vh_eu, vh_el = dL_OBC_eigenmode_cpu.stack_vh(vh[slb_ed,slb_ed], vh[slb_ed,slb_eo].conjugate().transpose(), nbc)

    # from L^{\lessgtr}\left(E\right)
    # (diagonal, upper, lower block of the start block at the left)
    lg_sd, lg_su, _ = dL_OBC_eigenmode_cpu.stack_lx(vh_sd, vh_su, vh_sl, pg_sd, pg_su, pg_sl)
    ll_sd, ll_su, _ = dL_OBC_eigenmode_cpu.stack_lx(vh_sd, vh_su, vh_sl, pl_sd, pl_su, pl_sl)

    # (diagonal, upper, lower block of the end block at the right)
    lg_ed, _, lg_el = dL_OBC_eigenmode_cpu.stack_lx(vh_ed, vh_eu, vh_el, pg_ed, pg_eu, pg_el)
    ll_ed, _, ll_el = dL_OBC_eigenmode_cpu.stack_lx(vh_ed, vh_eu, vh_el, pl_ed, pl_eu, pl_el)

    # from M^{r}\left(E\right)
    # (diagonal, upper, lower block of the start block at the left)
    _, _, mr_sl = dL_OBC_eigenmode_cpu.stack_mr(vh_sd, vh_su, vh_sl, pr_sd, pr_su, pr_sl)
    # (diagonal, upper, lower block of the end block at the right)
    _, mr_eu, _ = dL_OBC_eigenmode_cpu.stack_mr(vh_ed, vh_eu, vh_el, pr_ed, pr_eu, pr_el)


    # beyn gave a meaningful result
    if not np.isnan(cond_l):
        dlg_sd, dll_sd = dL_OBC_eigenmode_cpu.get_dl_obc(
                                                dxr_sd,
                                                lg_sd.toarray(),
                                                lg_su.toarray(),
                                                ll_sd.toarray(),
                                                ll_su.toarray(),
                                                mr_sl.toarray(),
                                                blk="L")

        mr[slb_sd_mm,slb_sd_mm]  = mr[slb_sd_mm,slb_sd_mm] - dmr_sd
        vh[slb_sd_mm,slb_sd_mm]  = vh[slb_sd_mm,slb_sd_mm] - dvh_sd

        if not np.isnan(dll_sd).any():
            lg[slb_sd_mm,slb_sd_mm] = lg[slb_sd_mm,slb_sd_mm] + dlg_sd
            ll[slb_sd_mm,slb_sd_mm] = ll[slb_sd_mm,slb_sd_mm] + dll_sd
        else:
            cond_l = np.nan

    if not np.isnan(cond_r):
        dlg_ed, dll_ed = dL_OBC_eigenmode_cpu.get_dl_obc(
                                                dxr_ed,
                                                lg_ed.toarray(),
                                                lg_el.toarray(),
                                                ll_ed.toarray(),
                                                ll_el.toarray(),
                                                mr_eu.toarray(),
                                                blk="R")

        mr[slb_ed_mm,slb_ed_mm]  = mr[slb_ed_mm,slb_ed_mm] - dmr_ed
        vh[slb_ed_mm,slb_ed_mm]  = vh[slb_ed_mm,slb_ed_mm] - dvh_ed

        if not np.isnan(dll_ed).any():
            lg[slb_ed_mm,slb_ed_mm] = lg[slb_ed_mm,slb_ed_mm] + dlg_ed
            ll[slb_ed_mm,slb_ed_mm] = ll[slb_ed_mm,slb_ed_mm] + dll_ed
        else:
            cond_r = np.nan

    return cond_r, cond_l

def obc_w(
        pg: sparse.csr_matrix,
        pl: sparse.csr_matrix,
        pr: sparse.csr_matrix,
        vh: sparse.csr_matrix,
        sr: sparse.csr_matrix,
        lg: sparse.csr_matrix,
        ll: sparse.csr_matrix,
        bmax: npt.NDArray[np.int32],
        bmin: npt.NDArray[np.int32],
        nbc: int,
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
        bmax (npt.NDArray[np.int32]): End indexes of every block
        bmin (npt.NDArray[np.int32]): Start indexes of every block
        nbc (int): How block size changes after matrix multiplication
        sancho_flag (bool, optional): If sancho OBC should be used, else beyn. Default false
    Returns:
        typing.Tuple[bool, sparse.csr_matrix]: If the inversion should be computed afterwards and mr
    """


    # limit for beyn
    imag_lim = 1e-4
    # todo find out what rr/R is
    # (R only is against the style guide)
    rr = 1e12

    # number of blocks
    nb = bmin.size
    # number of total orbitals (number of atomic orbitals)
    nao = vh.shape[0]

    # vector of block lengths
    lb_vec = bmax - bmin + 1
    # length of first and last block
    lb_start = lb_vec[0]
    lb_end = lb_vec[nb-1]

    # slice block start diagonal and slice block start off diagonal
    slb_sd = slice(0,lb_start)
    slb_so = slice(lb_start,2*lb_start)
    # slice block end diagonal and slice block end off diagonal
    slb_ed = slice(nao-lb_end,nao)
    slb_eo = slice(nao-2*lb_end,nao-lb_end)

    # slice block start diagonal
    # for block after matrix multiplication
    slb_sd_mm = slice(bmin[0], bmax[nbc-1] + 1)
    # slice block end diagonal
    # for block after matrix multiplication
    slb_ed_mm = slice(bmin[nb - nbc], nao)


    # from G^{\lessgtr}\left(E\right) / G^{r}\left(E\right)
    # (diagonal, upper, lower block of the start block at the left)
    pg_sd, pg_su, pg_sl = dL_OBC_eigenmode_cpu.stack_px(pg[slb_sd,slb_sd], pg[slb_sd,slb_so], nbc)
    pl_sd, pl_su, pl_sl = dL_OBC_eigenmode_cpu.stack_px(pl[slb_sd,slb_sd], pl[slb_sd,slb_so], nbc)
    pr_sd, pr_su, pr_sl = dL_OBC_eigenmode_cpu.stack_pr(pr[slb_sd,slb_sd], pr[slb_sd,slb_so], nbc)
    # (diagonal, upper, lower block of the end block at the right)
    pg_ed, pg_eu, pg_el = dL_OBC_eigenmode_cpu.stack_px(pg[slb_ed,slb_ed], -pg[slb_ed,slb_eo].conjugate().transpose(), nbc)
    pl_ed, pl_eu, pl_el = dL_OBC_eigenmode_cpu.stack_px(pl[slb_ed,slb_ed], -pl[slb_ed,slb_eo].conjugate().transpose(), nbc)
    pr_ed, pr_eu, pr_el = dL_OBC_eigenmode_cpu.stack_pr(pr[slb_ed,slb_ed], pr[slb_ed,slb_eo].transpose(), nbc)

    # from \hat(V)\left(E\right) / G^{r}\left(E\right)
    # (diagonal, upper, lower block of the start block at the left)
    vh_sd, vh_su, vh_sl = dL_OBC_eigenmode_cpu.stack_vh(vh[slb_sd,slb_sd], vh[slb_sd,slb_so], nbc)
    # (diagonal, upper, lower block of the end block at the right)
    vh_ed, vh_eu, vh_el = dL_OBC_eigenmode_cpu.stack_vh(vh[slb_ed,slb_ed], vh[slb_ed,slb_eo].conjugate().transpose(), nbc)

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


    # correction for the matrix inverse calculations----------------------------

    # conditions about convergence or meaningful results from
    # boundary correction calculations
    cond_l = 0.0
    cond_r = 0.0

    # correction for first block
    if not sancho_flag:
        _, cond_l, dxr_sd, dmr_sd, _ = beyn_cpu.beyn(
                                                mr_sd.toarray(),
                                                mr_su.toarray(),
                                                mr_sl.toarray(),
                                                imag_lim, rr, "L")
        if not np.isnan(cond_l):
            dvh_sd = mr_sl @ dxr_sd @ vh_su

    if np.isnan(cond_l) or sancho_flag:
        dxr_sd, dmr_sd, dvh_sd, cond_l = sancho.open_boundary_conditions(
                                                mr_sd.toarray(),
                                                mr_sl.toarray(),
                                                mr_su.toarray(),
                                                vh_su.toarray())
    
    # correction for last block
    if not sancho_flag:
        _, cond_r, dxr_ed, dmr_ed, _ = beyn_cpu.beyn(
                                                mr_ed.toarray(),
                                                mr_eu.toarray(),
                                                mr_el.toarray(),
                                                imag_lim, rr, "R")
        if not np.isnan(cond_r):
            dvh_ed = mr_eu @ dxr_ed @ vh_el

    if np.isnan(cond_r) or sancho_flag:
        dxr_ed, dmr_ed, dvh_ed, cond_r = sancho.open_boundary_conditions(
                                                mr_ed.toarray(),
                                                mr_eu.toarray(),
                                                mr_el.toarray(),
                                                vh_el.toarray())
    
    # beyn gave a meaningful result
    if not np.isnan(cond_l):
        dlg_sd, dll_sd = dL_OBC_eigenmode_cpu.get_dl_obc(
                                                dxr_sd,
                                                lg_sd.toarray(),
                                                lg_su.toarray(),
                                                ll_sd.toarray(),
                                                ll_su.toarray(),
                                                mr_sl.toarray(),
                                                blk="L")

        mr[slb_sd_mm,slb_sd_mm]  = mr[slb_sd_mm,slb_sd_mm] - dmr_sd
        vh[slb_sd_mm,slb_sd_mm]  = vh[slb_sd_mm,slb_sd_mm] - dvh_sd

        if not np.isnan(dll_sd).any():
            lg[slb_sd_mm,slb_sd_mm] = lg[slb_sd_mm,slb_sd_mm] + dlg_sd
            ll[slb_sd_mm,slb_sd_mm] = ll[slb_sd_mm,slb_sd_mm] + dll_sd
        else:
            cond_l = np.nan


    # condR = np.nan
    if not np.isnan(cond_r):
        dlg_ed, dll_ed = dL_OBC_eigenmode_cpu.get_dl_obc(
                                                dxr_ed,
                                                lg_ed.toarray(),
                                                lg_el.toarray(),
                                                ll_ed.toarray(),
                                                ll_el.toarray(),
                                                mr_eu.toarray(),
                                                blk="R")

        mr[slb_ed_mm,slb_ed_mm]  = mr[slb_ed_mm,slb_ed_mm] - dmr_ed
        vh[slb_ed_mm,slb_ed_mm]  = vh[slb_ed_mm,slb_ed_mm] - dvh_ed

        if not np.isnan(dll_ed).any():
            lg[slb_ed_mm,slb_ed_mm] = lg[slb_ed_mm,slb_ed_mm] + dlg_ed
            ll[slb_ed_mm,slb_ed_mm] = ll[slb_ed_mm,slb_ed_mm] + dll_ed
        else:
            cond_r = np.nan

    return not np.isnan(cond_r) and not np.isnan(cond_l), mr
