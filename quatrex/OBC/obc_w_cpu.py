# Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.
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
from quatrex.OBC import beyn_cpu
from quatrex.OBC import beyn_new
from quatrex.OBC import sancho
from quatrex.OBC import dL_OBC_eigenmode_cpu
import typing
from functools import partial

def obc_w_cpu(vh: sparse.csr_matrix,
    pg: sparse.csr_matrix,
    pl: sparse.csr_matrix,
    pr: sparse.csr_matrix,
    bmax: npt.NDArray[np.int32],
    bmin: npt.NDArray[np.int32],
    dvh_sd: npt.NDArray[np.complex128],
    dvh_ed: npt.NDArray[np.complex128],
    dmr_sd: npt.NDArray[np.complex128],
    dmr_ed: npt.NDArray[np.complex128],
    dlg_sd: npt.NDArray[np.complex128],
    dlg_ed: npt.NDArray[np.complex128],
    dll_sd: npt.NDArray[np.complex128],
    dll_ed: npt.NDArray[np.complex128],    
    nbc: np.int32,
    NCpSC: np.int32 = 1,
    block_inv: bool = False,
    use_dace: bool = False,
    validate_dace: bool = False,
    ref_flag: bool = False,
    sancho_flag: bool = False):
    """
    Calculates the standalone boundary correction terms for the screened interaction calculation.

    Args:
        vh (sparse.csr_matrix): Effective interaction
        pg (sparse.csr_matrix): Greater Polarization
        pl (sparse.csr_matrix): Lesser Polarization
        pr (sparse.csr_matrix): Retarded Polarization
        bmax (npt.NDArray[np.int32]): Start indices of blocks
        bmin (npt.NDArray[np.int32]): End indices of blocks
        dvh_sd (npt.NDArray[np.complex128]): Output start/left block of dvh
        dvh_ed (npt.NDArray[np.complex128]): Output end/right block of dvh
        dmr_ed (npt.NDArray[np.complex128]): Output end/right block of dmr
        dmr_sd (npt.NDArray[np.complex128]): Output start/left block of dmr
        dlg_sd (npt.NDArray[np.complex128]): Output start/left block of dlg
        dlg_ed (npt.NDArray[np.complex128]): Output end/right block of dlg
        dll_sd (npt.NDArray[np.complex128]): Output start/left block of dll
        dll_ed (npt.NDArray[np.complex128]): Output end/right block of dll
        nbc (np.int64): How block size changes after matrix multiplication
        NCpSC (np.int32, optional): Number of unit cells per supercell. Defaults to 1.
    """
    beyn = beyn_new.beyn_new
    if use_dace:
        # from OBC.beyn_dace import beyn, contour_integral_dace, sort_k_dace
        # contour_integral, _ = contour_integral_dace.load_precompiled_sdfg(f'.dacecache/{contour_integral_dace.name}')
        # sortk, _ = sort_k_dace.load_precompiled_sdfg(f'.dacecache/{sort_k_dace.name}')
        # beyn = partial(beyn, contour_integral=contour_integral, sortk=sortk)
        from OBC.beyn_dace import beyn as beyn_dace
        beyn = partial(beyn_dace, validate=validate_dace)

    # limit for beyn
    imag_lim = 1e-4
    # todo find out what rr/R is
    # (R only is against the style guide)
    #todo, compare with matlab
    rr = 1e4
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


    # calculate L^{\lessgtr}\left(E\right)
    # vh_cp_ct = vh_cp.conjugate().transpose()
    # lg = vh_cp @ pg @ vh_cp_ct
    # ll = vh_cp @ pl @ vh_cp_ct

    vh_s1 = np.ascontiguousarray(vh[slb_sd, slb_sd].toarray(order="C"))
    vh_s2 = np.ascontiguousarray(vh[slb_sd, slb_so].toarray(order="C"))
    pg_s1 = np.ascontiguousarray(pg[slb_sd, slb_sd].toarray(order="C"))
    pg_s2 = np.ascontiguousarray(pg[slb_sd, slb_so].toarray(order="C"))
    pl_s1 = np.ascontiguousarray(pl[slb_sd, slb_sd].toarray(order="C"))
    pl_s2 = np.ascontiguousarray(pl[slb_sd, slb_so].toarray(order="C"))
    pr_s1 = np.ascontiguousarray(pr[slb_sd, slb_sd].toarray(order="C"))
    pr_s2 = np.ascontiguousarray(pr[slb_sd, slb_so].toarray(order="C"))
    pr_s3 = np.ascontiguousarray(pr[slb_so, slb_sd].toarray(order="C"))

    vh_e1 = np.ascontiguousarray(vh[slb_ed, slb_ed].toarray(order="C"))
    vh_e2 = np.ascontiguousarray(vh[slb_ed, slb_eo].conjugate().transpose().toarray(order="C"))
    pg_e1 = np.ascontiguousarray(pg[slb_ed, slb_ed].toarray(order="C"))
    pg_e2 = np.ascontiguousarray(-pg[slb_ed, slb_eo].conjugate().transpose().toarray(order="C"))
    pl_e1 = np.ascontiguousarray(pl[slb_ed, slb_ed].toarray(order="C"))
    pl_e2 = np.ascontiguousarray(-pl[slb_ed, slb_eo].conjugate().transpose().toarray(order="C"))
    pr_e1 = np.ascontiguousarray(pr[slb_ed, slb_ed].toarray(order="C"))
    pr_e2 = np.ascontiguousarray(pr[slb_eo, slb_ed].toarray(order="C"))
    pr_e3 = np.ascontiguousarray(pr[slb_ed, slb_eo].toarray(order="C"))

    mr_s, lg_s, ll_s, dmr_s, dlg_s, dll_s, vh_s, mb00 = dL_OBC_eigenmode_cpu.get_mm_obc_dense(
    vh_s1, vh_s2, pg_s1, pg_s2, pl_s1, pl_s2, pr_s1, pr_s2, pr_s3, nbc, NCpSC, 'L')

    mr_e, lg_e, ll_e, dmr_e, dlg_e, dll_e, vh_e, mbNN = dL_OBC_eigenmode_cpu.get_mm_obc_dense(
    vh_e1, vh_e2, pg_e1, pg_e2, pl_e1, pl_e2, pr_e1, pr_e2, pr_e3, nbc, NCpSC, 'R')

    # correct first and last block to account for the contacts in multiplication
    dmr_sd[:, :] = dmr_s[0]
    dmr_ed[:, :] = dmr_e[1]

    # correct first and last block to account for the contacts in multiplication
    # first block
    # L^{\lessgtr}_E_00 = V_10*P^{\lessgtr}_E_00*V_01 + V_10*P^{\lessgtr}_E_01*V_00 + V_00*P^{\lessgtr}_E_10*V_01
    dlg_sd[:, :] = dlg_s[0]
    dll_sd[:, :] = dll_s[0]

    # last block
    # L^{\lessgtr}_E_nn = V_nn*PL_E_n-1n*V_nn-1 + V_n-1n*PL_E_nn-1*V_nn + V_n-1n*PL_E_nn*V_nn-1
    dlg_ed[:, :] = dlg_e[1]
    dll_ed[:, :] = dll_e[1]

        # conditions about convergence or meaningful results from
    # boundary correction calculations
    cond_l = 0.0
    cond_r = 0.0

    # correction for first block
    if not sancho_flag:
        dmr, dxr_sd, cond_l, min_dEkL = beyn(nbc * NCpSC, mb00, mr_s[0], mr_s[1], mr_s[2], imag_lim, rr, "L")

        if not np.isnan(cond_l):
            dmr_sd -= dmr
            dvh_sd[:, :] = mr_s[2] @ dxr_sd @ vh_s[0]

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
        dxr_sd, dmr, dvh_sd[:, :], cond_l = sancho.open_boundary_conditions(mr_s[0], mr_s[2], mr_s[1], vh_s[0])
        dmr_sd -= dmr
    # correction for last block
    if not sancho_flag:
        dmr, dxr_ed, cond_r, min_dEkR = beyn(nbc * NCpSC, mbNN, mr_e[0], mr_e[1], mr_e[2], imag_lim, rr, "R")
        if not np.isnan(cond_r):
            dmr_ed -= dmr
            dvh_ed[:, :] = mr_e[1] @ dxr_ed @ vh_e[1]
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
        dxr_ed, dmr, dvh_ed[:, :], cond_r = sancho.open_boundary_conditions(mr_e[0], mr_e[1], mr_e[2], vh_e[1])
        dmr_ed -= dmr

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
    return cond_l, cond_r

def obc_w_sl(vh: sparse.spmatrix, pg: sparse.spmatrix, pl: sparse.spmatrix, pr: sparse.spmatrix,
             nao: int) -> typing.Tuple[sparse.spmatrix, sparse.spmatrix, sparse.spmatrix]:
    """
    Calculates for the additional helper variables.

    Args:
        vh (sparse.spmatrix): Effective interaction
        pg (sparse.spmatrix): Greater Polarization
        pl (sparse.spmatrix): Lesser Polarization
        pr (sparse.spmatrix): Retarded Polarization
        nao            (int): Number of atomic orbitals

    Returns:
        typing.Tuple[sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix]: M^{r}\left(E\right), L^{>}\left(E\right), L^{<}\left(E\right)
    """

    vh_ct = vh.conjugate().T

    # calculate L^{\lessgtr}\left(E\right)
    lg = vh @ pg @ vh_ct
    ll = vh @ pl @ vh_ct

    # calculate M^{r}\left(E\right)
    mr = sparse.identity(nao, format="csr") - vh @ pr

    return mr, lg, ll


def obc_w_sc(
    pg: sparse.csr_matrix,
    pl: sparse.csr_matrix,
    pr: sparse.csr_matrix,
    vh: sparse.csr_matrix,
    mr_sf: npt.NDArray[np.complex128],
    mr_ef: npt.NDArray[np.complex128],
    lg_sf: npt.NDArray[np.complex128],
    lg_ef: npt.NDArray[np.complex128],
    ll_sf: npt.NDArray[np.complex128],
    ll_ef: npt.NDArray[np.complex128],
    dmr_sf: npt.NDArray[np.complex128],
    dmr_ef: npt.NDArray[np.complex128],
    dlg_sf: npt.NDArray[np.complex128],
    dlg_ef: npt.NDArray[np.complex128],
    dll_sf: npt.NDArray[np.complex128],
    dll_ef: npt.NDArray[np.complex128],
    vh_sf: npt.NDArray[np.complex128],
    vh_ef: npt.NDArray[np.complex128],
    bmax: npt.NDArray[np.int32],
    bmin: npt.NDArray[np.int32],
    nbc: int,
):
    """
    Calculates the blocks needed for the boundary correction.

    Args:
        pg (sparse.csr_matrix): Greater Polarization
        pl (sparse.csr_matrix): Lesser Polarization
        pr (sparse.csr_matrix): Retarded Polarization
        vh (sparse.csr_matrix): Effective Screening
        mr_sf (npt.NDArray[np.complex128]): Output start/left blocks (diagonal, upper, lower) of mr
        mr_ef (npt.NDArray[np.complex128]): Output end/right blocks (diagonal, upper, lower) of mr
        lg_sf (npt.NDArray[np.complex128]): Output start/left blocks (diagonal, upper, lower) of lg
        lg_ef (npt.NDArray[np.complex128]): Output end/right blocks (diagonal, upper, lower) of lg
        ll_sf (npt.NDArray[np.complex128]): Output start/left blocks (diagonal, upper, lower) of ll
        ll_ef (npt.NDArray[np.complex128]): Output end/right blocks (diagonal, upper, lower) of ll
        dmr_sf (npt.NDArray[np.complex128]): Output start/left block of mr
        dmr_ef (npt.NDArray[np.complex128]): Output end/right block of mr
        dlg_sf (npt.NDArray[np.complex128]): Output start/left block of lg
        dlg_ef (npt.NDArray[np.complex128]): Output end/right block of lg
        dll_sf (npt.NDArray[np.complex128]): Output start/left block of ll
        dll_ef (npt.NDArray[np.complex128]): Output end/right block of ll
        vh_sf (npt.NDArray[np.complex128]): Output start/left block of vh
        vh_ef (npt.NDArray[np.complex128]): Output start/left block of vh
        bmax (npt.NDArray[np.int32]): Start indices of blocks
        bmin (npt.NDArray[np.int32]): End indices of blocks
        nbc (int): How block size changes after matrix multiplication
    """
    # number of blocks
    nb = bmin.size
    # number of total orbitals (number of atomic orbitals)
    nao = vh.shape[0]

    # vector of block lengths
    lb_vec = bmax - bmin + 1
    # length of first and last block
    lb_start = lb_vec[0]
    lb_end = lb_vec[nb - 1]

    # slice block start diagonal and slice block start off diagonal
    slb_sd = slice(0, lb_start)
    slb_so = slice(lb_start, 2 * lb_start)
    # slice block end diagonal and slice block end off diagonal
    slb_ed = slice(nao - lb_end, nao)
    slb_eo = slice(nao - 2 * lb_end, nao - lb_end)

    # from M^{r}\left(E\right)/L^{\lessgtr}\left(E\right)\hat(V)\left(E\right)
    mr_s, lg_s, ll_s, dmr_s, dlg_s, dll_s, vh_s = dL_OBC_eigenmode_cpu.get_mm_obc_dense(
        vh[slb_sd, slb_sd].toarray(order="C"), vh[slb_sd, slb_so].toarray(order="C"), pg[slb_sd,
                                                                                         slb_sd].toarray(order="C"),
        pg[slb_sd, slb_so].toarray(order="C"), pl[slb_sd, slb_sd].toarray(order="C"), pl[slb_sd,
                                                                                         slb_so].toarray(order="C"),
        pr[slb_sd, slb_sd].toarray(order="C"), pr[slb_sd, slb_so].toarray(order="C"), nbc)
    mr_e, lg_e, ll_e, dmr_e, dlg_e, dll_e, vh_e = dL_OBC_eigenmode_cpu.get_mm_obc_dense(
        vh[slb_ed, slb_ed].toarray(order="C"), vh[slb_ed, slb_eo].conjugate().transpose().toarray(order="C"),
        pg[slb_ed, slb_ed].toarray(order="C"), -pg[slb_ed, slb_eo].conjugate().transpose().toarray(order="C"),
        pl[slb_ed, slb_ed].toarray(order="C"), -pl[slb_ed, slb_eo].conjugate().transpose().toarray(order="C"),
        pr[slb_ed, slb_ed].toarray(order="C"), pr[slb_ed, slb_eo].transpose().toarray(order="C"), nbc)
    # write to output
    mr_sf[0] = mr_s[0]
    mr_sf[1] = mr_s[1]
    mr_sf[2] = mr_s[2]
    mr_ef[0] = mr_e[0]
    mr_ef[1] = mr_e[1]
    mr_ef[2] = mr_e[2]
    lg_sf[0] = lg_s[0]
    lg_sf[1] = lg_s[1]
    lg_sf[2] = lg_s[2]
    lg_ef[0] = lg_e[0]
    lg_ef[1] = lg_e[1]
    lg_ef[2] = lg_e[2]
    ll_sf[0] = ll_s[0]
    ll_sf[1] = ll_s[1]
    ll_sf[2] = ll_s[2]
    ll_ef[0] = ll_e[0]
    ll_ef[1] = ll_e[1]
    ll_ef[2] = ll_e[2]

    vh_sf[0] = vh_s[0]
    vh_ef[0] = vh_e[1]
    dmr_sf[0] = dmr_s[0]
    dmr_ef[0] = dmr_e[1]
    dlg_sf[0] = dlg_s[0]
    dlg_ef[0] = dlg_e[1]
    dll_sf[0] = dll_s[0]
    dll_ef[0] = dll_e[1]


def obc_w_beyn(dxr_sf: npt.NDArray[np.complex128],
               dxr_ef: npt.NDArray[np.complex128],
               mr_sf: npt.NDArray[np.complex128],
               mr_ef: npt.NDArray[np.complex128],
               vh_sf: npt.NDArray[np.complex128],
               vh_ef: npt.NDArray[np.complex128],
               dmr_sf: npt.NDArray[np.complex128],
               dmr_ef: npt.NDArray[np.complex128],
               dvh_sf: npt.NDArray[np.complex128],
               dvh_ef: npt.NDArray[np.complex128],
               sancho_flag: bool = False) -> typing.Tuple[float, float]:
    """
    Calculates the boundary correction with the beyn method.

    Args:
        dxr_sf (npt.NDArray[np.complex128]): Output start/left block of dxr
        dxr_ef (npt.NDArray[np.complex128]): Output end/right block of dxr
        mr_sf (npt.NDArray[np.complex128]): Input start/left blocks of mr
        mr_ef (npt.NDArray[np.complex128]): Input end/right blocks of mr
        vh_sf (npt.NDArray[np.complex128]): Input start/left block of vh
        vh_ef (npt.NDArray[np.complex128]): Input end/right block of vh
        dmr_sf (npt.NDArray[np.complex128]): Output start/left block of dmr
        dmr_ef (npt.NDArray[np.complex128]): Output end/right block of dmr
        dvh_sf (npt.NDArray[np.complex128]): Output start/left block of dvh
        dvh_ef (npt.NDArray[np.complex128]): Output end/right block of dvh
        sancho_flag (bool, optional): if sancho instead of beyn should be used. Defaults to False.

    Returns:
        typing.Tuple[float, float]: Right and left condition, meaning if the boundary correction is meaningful
    """
    # limit for beyn
    imag_lim = 1e-4
    # todo find out what rr/R is
    # (R only is against the style guide)
    rr = 1e12

    dxr_sd = np.zeros_like(dxr_sf[0])
    dxr_ed = np.zeros_like(dxr_ef[0])
    dmr_sd = np.zeros_like(dmr_sf[0])
    dmr_ed = np.zeros_like(dmr_ef[0])

    # old wrong version
    if not sancho_flag:
        cond_l, _ = beyn_cpu.beyn_old_opt(mr_sf[0], mr_sf[1], mr_sf[2], dmr_sd, dxr_sd, imag_lim, rr, "L")
        dxr_sf[0] = dxr_sd
        if cond_l:
            dmr_sf[0] -= dmr_sd
            dvh_sf[0] = mr_sf[2] @ dxr_sd @ vh_sf[0]
    # # old wrong version
    # if not sancho_flag:
    #     _, cond_l, dxr_sd, dmr, _ = beyn_cpu.beyn_old(
    #                                             mr_sf[0],
    #                                             mr_sf[1],
    #                                             mr_sf[2],
    #                                             imag_lim, rr, "L")

    #     dxr_sf[0] = dxr_sd
    #     if not np.isnan(cond_l):
    #         dmr_sf[0] -= dmr
    #         dvh_sf[0] = mr_sf[2] @ dxr_sd @ vh_sf[0]

    if (not cond_l) or sancho_flag:
        dxr_sd, dmr, dvh_sd, cond_l = sancho.open_boundary_conditions(mr_sf[0], mr_sf[2], mr_sf[1], vh_sf[0])
        dxr_sf[0] = dxr_sd
        dmr_sf[0] -= dmr
        dvh_sf[0] = dvh_sd
    if not sancho_flag:
        cond_r, _ = beyn_cpu.beyn_old_opt(mr_ef[0], mr_ef[1], mr_ef[2], dmr_ed, dxr_ed, imag_lim, rr, "R")
        dxr_ef[0] = dxr_ed
        if cond_r:
            dmr_ef[0] -= dmr_ed
            dvh_ef[0] = mr_ef[1] @ dxr_ed @ vh_ef[0]
    # if not sancho_flag:
    #     _, cond_r, dxr_ed, dmr, _ = beyn_cpu.beyn_old(
    #                                             mr_ef[0],
    #                                             mr_ef[1],
    #                                             mr_ef[2],
    #                                             imag_lim, rr, "R")
    #     dxr_ef[0] = dxr_ed
    #     if not np.isnan(cond_r):
    #         dmr_ef[0] -= dmr
    #         dvh_ef[0] = mr_ef[1] @ dxr_ed @ vh_ef[0]

    if (not cond_r) or sancho_flag:
        dxr_ed, dmr, dvh_ed, cond_r = sancho.open_boundary_conditions(mr_ef[0], mr_ef[1], mr_ef[2], vh_ef[0])
        dxr_ef[0] = dxr_ed
        dmr_ef[0] -= dmr
        dvh_ef[0] = dvh_ed

    return cond_r, cond_l


def obc_w_dl(dxr_sf: npt.NDArray[np.complex128], dxr_ef: npt.NDArray[np.complex128], lg_sf: npt.NDArray[np.complex128],
             lg_ef: npt.NDArray[np.complex128], ll_sf: npt.NDArray[np.complex128], ll_ef: npt.NDArray[np.complex128],
             mr_sf: npt.NDArray[np.complex128], mr_ef: npt.NDArray[np.complex128], dlg_sf: npt.NDArray[np.complex128],
             dll_sf: npt.NDArray[np.complex128], dlg_ef: npt.NDArray[np.complex128],
             dll_ef: npt.NDArray[np.complex128]):
    """
    Calculates the boundary correction with the dL method.

    Args:
        dxr_sf (npt.NDArray[np.complex128]): Input start/left block of dxr
        dxr_ef (npt.NDArray[np.complex128]): Input end/right block of dxr
        lg_sf (npt.NDArray[np.complex128]): Input start/left blocks of lg
        lg_ef (npt.NDArray[np.complex128]): Input end/right blocks of lg
        ll_sf (npt.NDArray[np.complex128]): Input start/left blocks of ll
        ll_ef (npt.NDArray[np.complex128]): Input end/right blocks of ll
        mr_sf (npt.NDArray[np.complex128]): Input start/left blocks of mr
        mr_ef (npt.NDArray[np.complex128]): Input end/right blocks of mr
        dlg_sf (npt.NDArray[np.complex128]): Output start/left block of dlg, will be incremented with result
        dll_sf (npt.NDArray[np.complex128]): Output start/left block of dll, will be incremented with result
        dlg_ef (npt.NDArray[np.complex128]): Output end/right block of dlg, will be incremented with result
        dll_ef (npt.NDArray[np.complex128]): Output end/right block of dll, will be incremented with result
    """
    dlg, dll = dL_OBC_eigenmode_cpu.get_dl_obc_alt(dxr_sf[0], lg_sf[0], lg_sf[1], ll_sf[0], ll_sf[1], mr_sf[2], blk="L")
    dlg_sf[0] += dlg
    dll_sf[0] += dll
    if np.isnan(dll).any():
        cond_l = np.nan
    if np.isnan(dlg).any():
        cond_l = np.nan
    # else:
    #     dlg_sd += dlg
    #     dll_sd += dll

    dlg, dll = dL_OBC_eigenmode_cpu.get_dl_obc_alt(dxr_ef[0], lg_ef[0], lg_ef[2], ll_ef[0], ll_ef[2], mr_ef[1], blk="R")
    dlg_ef[0] += dlg
    dll_ef[0] += dll
    if np.isnan(dll).any():
        cond_r = np.nan
    if np.isnan(dlg).any():
        cond_r = np.nan
    # else:
    #     dlg_ed += dlg
    #     dll_ed += dll
