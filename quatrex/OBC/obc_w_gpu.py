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
import cupy as cp
from cupyx.scipy import sparse as cusparse
from quatrex.OBC import beyn_new_gpu
from quatrex.OBC import sancho
from quatrex.OBC import dL_OBC_eigenmode_gpu
import typing
from functools import partial
from scipy import sparse


def obc_w_L_lg(dlg_sd: npt.NDArray[np.complex128],
    dlg_ed: npt.NDArray[np.complex128],
    dll_sd: npt.NDArray[np.complex128],
    dll_ed: npt.NDArray[np.complex128],
    mr_s: tuple,
    mr_e: tuple,
    lg_s: tuple,
    lg_e: tuple,
    ll_s: tuple,
    ll_e: tuple,
    dxr_sd: npt.NDArray[np.complex128],
    dxr_ed: npt.NDArray[np.complex128]):
     # beyn gave a meaningful result
    dlg, dll = dL_OBC_eigenmode_gpu.get_dl_obc_alt(cp.asarray(dxr_sd), cp.asarray(lg_s[0]), cp.asarray(lg_s[1]), cp.asarray(ll_s[0]), cp.asarray(ll_s[1]), cp.asarray(mr_s[2]), blk="L")

    if np.isnan(dll).any():
        cond_l = np.nan
    else:
        dlg_sd += dlg
        dll_sd += dll

    dlg, dll = dL_OBC_eigenmode_gpu.get_dl_obc_alt(cp.asarray(dxr_ed), cp.asarray(lg_e[0]), cp.asarray(lg_e[1]), cp.asarray(ll_e[0]), cp.asarray(ll_e[1]), cp.asarray(mr_e[1]), blk="R")

    if np.isnan(dll_ed).any():
        cond_r = np.nan
    else:
        dlg_ed += dlg
        dll_ed += dll


def obc_w_L_lg_2(dlg_sd: npt.NDArray[np.complex128],
    dlg_ed: npt.NDArray[np.complex128],
    dll_sd: npt.NDArray[np.complex128],
    dll_ed: npt.NDArray[np.complex128],
    mr_s0, mr_s1, mr_s2,
    mr_e0, mr_e1, mr_e2,
    lg_s0, lg_s1,
    lg_e0, lg_e1,
    ll_s0, ll_s1,
    ll_e0, ll_e1,
    dxr_sd: npt.NDArray[np.complex128],
    dxr_ed: npt.NDArray[np.complex128]):
     # beyn gave a meaningful result
    dlg, dll = dL_OBC_eigenmode_gpu.get_dl_obc_alt(cp.asarray(dxr_sd), cp.asarray(lg_s0), cp.asarray(lg_s1), cp.asarray(ll_s0), cp.asarray(ll_s1), cp.asarray(mr_s2), blk="L")

    if np.isnan(dll).any():
        cond_l = np.nan
    else:
        dlg_sd += dlg
        dll_sd += dll

    dlg, dll = dL_OBC_eigenmode_gpu.get_dl_obc_alt(cp.asarray(dxr_ed), cp.asarray(lg_e0), cp.asarray(lg_e1), cp.asarray(ll_e0), cp.asarray(ll_e1), cp.asarray(mr_e1), blk="R")

    if np.isnan(dll_ed).any():
        cond_r = np.nan
    else:
        dlg_ed += dlg
        dll_ed += dll


def obc_w_mm_gpu(vh: sparse.csr_matrix,
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
    mr_s: tuple,
    mr_e: tuple,
    lg_s: tuple,
    lg_e: tuple,
    ll_s: tuple,
    ll_e: tuple,
    vh_s: npt.NDArray[np.complex128],
    vh_e: npt.NDArray[np.complex128],
    mb00: npt.NDArray[np.complex128],
    mbNN: npt.NDArray[np.complex128],   
    nbc: np.int32,
    NCpSC: np.int32 = 1,
    block_inv: bool = False,
    use_dace: bool = False,
    validate_dace: bool = False,
    ref_flag: bool = False):
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

    mr_s_loc, lg_s_loc, ll_s_loc, dmr_s, dlg_s, dll_s, vh_s_loc, mb00_loc = dL_OBC_eigenmode_gpu.get_mm_obc_dense(
    cp.asarray(vh_s1), cp.asarray(vh_s2), cp.asarray(pg_s1), cp.asarray(pg_s2), cp.asarray(pl_s1), \
    cp.asarray(pl_s2), cp.asarray(pr_s1), cp.asarray(pr_s2), cp.asarray(pr_s3), nbc, NCpSC, 'L')

    mr_e_loc, lg_e_loc, ll_e_loc, dmr_e, dlg_e, dll_e, vh_e_loc, mbNN_loc = dL_OBC_eigenmode_gpu.get_mm_obc_dense(
    cp.asarray(vh_e1), cp.asarray(vh_e2), cp.asarray(pg_e1), cp.asarray(pg_e2), cp.asarray(pl_e1), \
    cp.asarray(pl_e2), cp.asarray(pr_e1), cp.asarray(pr_e2), cp.asarray(pr_e3), nbc, NCpSC, 'R')

    # correct first and last block to account for the contacts in multiplication
    dmr_sd[:, :] = dmr_s[0].get()
    dmr_ed[:, :] = dmr_e[1].get()

    # correct first and last block to account for the contacts in multiplication
    # first block
    # L^{\lessgtr}_E_00 = V_10*P^{\lessgtr}_E_00*V_01 + V_10*P^{\lessgtr}_E_01*V_00 + V_00*P^{\lessgtr}_E_10*V_01
    dlg_sd[:, :] = dlg_s[0].get()
    dll_sd[:, :] = dll_s[0].get()

    # last block
    # L^{\lessgtr}_E_nn = V_nn*PL_E_n-1n*V_nn-1 + V_n-1n*PL_E_nn-1*V_nn + V_n-1n*PL_E_nn*V_nn-1
    dlg_ed[:, :] = dlg_e[1].get()
    dll_ed[:, :] = dll_e[1].get()

    mr_s[0][:, :] = mr_s_loc[0].get()
    mr_s[1][:, :] = mr_s_loc[1].get()
    mr_s[2][:, :] = mr_s_loc[2].get()

    mr_e[0][:, :] = mr_e_loc[0].get()
    mr_e[1][:, :] = mr_e_loc[1].get()
    mr_e[2][:, :] = mr_e_loc[2].get()

    lg_s[0][:, :] = lg_s_loc[0].get()
    lg_s[1][:, :] = lg_s_loc[1].get()

    lg_e[0][:, :] = lg_e_loc[0].get()
    lg_e[1][:, :] = lg_e_loc[2].get()

    ll_s[0][:, :] = ll_s_loc[0].get()
    ll_s[1][:, :] = ll_s_loc[1].get()

    ll_e[0][:, :] = ll_e_loc[0].get()
    ll_e[1][:, :] = ll_e_loc[2].get()

    vh_s[:, :] = vh_s_loc[0].get()
    vh_e[:, :] = vh_e_loc[1].get()

    mb00[:, :, :] = mb00_loc.get()
    mbNN[:, :, :] = mbNN_loc.get()


def obc_w_mm_gpu_2(vh: sparse.csr_matrix,
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
    mr_s: tuple,
    mr_e: tuple,
    lg_s: tuple,
    lg_e: tuple,
    ll_s: tuple,
    ll_e: tuple,
    vh_s: npt.NDArray[np.complex128],
    vh_e: npt.NDArray[np.complex128],
    mb00: npt.NDArray[np.complex128],
    mbNN: npt.NDArray[np.complex128],
    rows, columns,   
    nbc: np.int32,
    NCpSC: np.int32 = 1,
    block_inv: bool = False,
    use_dace: bool = False,
    validate_dace: bool = False,
    ref_flag: bool = False):
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

    vh_s1 = cp.ascontiguousarray(vh[slb_sd, slb_sd].toarray(order="C"))
    vh_s2 = cp.ascontiguousarray(vh[slb_sd, slb_so].toarray(order="C"))
    pg_s1 = cp.ascontiguousarray(pg[slb_sd, slb_sd].toarray(order="C"))
    pg_s2 = cp.ascontiguousarray(pg[slb_sd, slb_so].toarray(order="C"))
    pl_s1 = cp.ascontiguousarray(pl[slb_sd, slb_sd].toarray(order="C"))
    pl_s2 = cp.ascontiguousarray(pl[slb_sd, slb_so].toarray(order="C"))
    pr_s1 = cp.ascontiguousarray(pr[slb_sd, slb_sd].toarray(order="C"))
    pr_s2 = cp.ascontiguousarray(pr[slb_sd, slb_so].toarray(order="C"))
    pr_s3 = cp.ascontiguousarray(pr[slb_so, slb_sd].toarray(order="C"))

    vh_e1 = cp.ascontiguousarray(vh[slb_ed, slb_ed].toarray(order="C"))
    vh_e2 = cp.ascontiguousarray(vh[slb_ed, slb_eo].conjugate().transpose().toarray(order="C"))
    pg_e1 = cp.ascontiguousarray(pg[slb_ed, slb_ed].toarray(order="C"))
    pg_e2 = cp.ascontiguousarray(-pg[slb_ed, slb_eo].conjugate().transpose().toarray(order="C"))
    pl_e1 = cp.ascontiguousarray(pl[slb_ed, slb_ed].toarray(order="C"))
    pl_e2 = cp.ascontiguousarray(-pl[slb_ed, slb_eo].conjugate().transpose().toarray(order="C"))
    pr_e1 = cp.ascontiguousarray(pr[slb_ed, slb_ed].toarray(order="C"))
    pr_e2 = cp.ascontiguousarray(pr[slb_eo, slb_ed].toarray(order="C"))
    pr_e3 = cp.ascontiguousarray(pr[slb_ed, slb_eo].toarray(order="C"))

    mr_s_loc, lg_s_loc, ll_s_loc, dmr_s, dlg_s, dll_s, vh_s_loc, mb00_loc = dL_OBC_eigenmode_gpu.get_mm_obc_dense(
    cp.asarray(vh_s1), cp.asarray(vh_s2), cp.asarray(pg_s1), cp.asarray(pg_s2), cp.asarray(pl_s1), \
    cp.asarray(pl_s2), cp.asarray(pr_s1), cp.asarray(pr_s2), cp.asarray(pr_s3), nbc, NCpSC, 'L')

    mr_e_loc, lg_e_loc, ll_e_loc, dmr_e, dlg_e, dll_e, vh_e_loc, mbNN_loc = dL_OBC_eigenmode_gpu.get_mm_obc_dense(
    cp.asarray(vh_e1), cp.asarray(vh_e2), cp.asarray(pg_e1), cp.asarray(pg_e2), cp.asarray(pl_e1), \
    cp.asarray(pl_e2), cp.asarray(pr_e1), cp.asarray(pr_e2), cp.asarray(pr_e3), nbc, NCpSC, 'R')

    # correct first and last block to account for the contacts in multiplication
    dmr_sd[:, :] = dmr_s[0].get()
    dmr_ed[:, :] = dmr_e[1].get()

    # correct first and last block to account for the contacts in multiplication
    # first block
    # L^{\lessgtr}_E_00 = V_10*P^{\lessgtr}_E_00*V_01 + V_10*P^{\lessgtr}_E_01*V_00 + V_00*P^{\lessgtr}_E_10*V_01
    dlg_sd[:, :] = dlg_s[0].get()
    dll_sd[:, :] = dll_s[0].get()

    # last block
    # L^{\lessgtr}_E_nn = V_nn*PL_E_n-1n*V_nn-1 + V_n-1n*PL_E_nn-1*V_nn + V_n-1n*PL_E_nn*V_nn-1
    dlg_ed[:, :] = dlg_e[1].get()
    dll_ed[:, :] = dll_e[1].get()

    mr_s[0][:, :] = mr_s_loc[0].get()
    mr_s[1][:, :] = mr_s_loc[1].get()
    mr_s[2][:, :] = mr_s_loc[2].get()

    mr_e[0][:, :] = mr_e_loc[0].get()
    mr_e[1][:, :] = mr_e_loc[1].get()
    mr_e[2][:, :] = mr_e_loc[2].get()

    lg_s[0][:, :] = lg_s_loc[0].get()
    lg_s[1][:, :] = lg_s_loc[1].get()

    lg_e[0][:, :] = lg_e_loc[0].get()
    lg_e[1][:, :] = lg_e_loc[2].get()

    ll_s[0][:, :] = ll_s_loc[0].get()
    ll_s[1][:, :] = ll_s_loc[1].get()

    ll_e[0][:, :] = ll_e_loc[0].get()
    ll_e[1][:, :] = ll_e_loc[2].get()

    vh_s[:, :] = vh_s_loc[0].get()
    vh_e[:, :] = vh_e_loc[1].get()

    mb00[:, :, :] = mb00_loc.get()
    mbNN[:, :, :] = mbNN_loc.get()


# def obc_w_mm_batched_gpu(
#     vh_s1, vh_s2, pg_s1, pg_s2, pl_s1, pl_s2, pr_s1, pr_s2, pr_s3,
#     vh_e1, vh_e2, pg_e1, pg_e2, pl_e1, pl_e2, pr_e1, pr_e2, pr_e3,
#     dmr_sd: npt.NDArray[np.complex128],
#     dmr_ed: npt.NDArray[np.complex128],
#     dlg_sd: npt.NDArray[np.complex128],
#     dlg_ed: npt.NDArray[np.complex128],
#     dll_sd: npt.NDArray[np.complex128],
#     dll_ed: npt.NDArray[np.complex128],
#     mr_s0, mr_s1, mr_s2, mr_e0, mr_e1, mr_e2, lg_s0, lg_s1, lg_e0, lg_e1, ll_s0, ll_s1, ll_e0, ll_e1,
#     vh_s: npt.NDArray[np.complex128],
#     vh_e: npt.NDArray[np.complex128],
#     mb00: npt.NDArray[np.complex128],
#     mbNN: npt.NDArray[np.complex128],
#     nbc,
#     NCpSC: np.int32 = 1):

def obc_w_mm_batched_gpu(
    vh_s1, vh_s2, pg_s1, pg_s2, pl_s1, pl_s2, pr_s1, pr_s2, pr_s3,
    vh_e1, vh_e2, pg_e1, pg_e2, pl_e1, pl_e2, pr_e1, pr_e2, pr_e3,
    nbc,
    NCpSC: np.int32 = 1):

    mr_s_loc, lg_s_loc, ll_s_loc, dmr_s, dlg_s, dll_s, vh_s_loc, mb00_loc = dL_OBC_eigenmode_gpu.get_mm_obc_dense_batched(
        vh_s1, vh_s2, pg_s1, pg_s2, pl_s1, pl_s2, pr_s1, pr_s2, pr_s3, nbc, NCpSC, 'L')

    mr_e_loc, lg_e_loc, ll_e_loc, dmr_e, dlg_e, dll_e, vh_e_loc, mbNN_loc = dL_OBC_eigenmode_gpu.get_mm_obc_dense_batched(
        vh_e1, vh_e2, pg_e1, pg_e2, pl_e1, pl_e2, pr_e1, pr_e2, pr_e3, nbc, NCpSC, 'R')
    
    return (
        dmr_s[0], dmr_e[1],
        dlg_s[0], dlg_e[1],
        dll_s[0], dll_e[1],
        mr_s_loc[0], mr_s_loc[1], mr_s_loc[2],
        mr_e_loc[0], mr_e_loc[1], mr_e_loc[2],
        lg_s_loc[0], lg_s_loc[1],
        lg_e_loc[0], lg_e_loc[2],
        ll_s_loc[0], ll_s_loc[1],
        ll_e_loc[0], ll_e_loc[2],
        vh_s_loc[0], vh_e_loc[1],
        mb00_loc, mbNN_loc
    )

    # correct first and last block to account for the contacts in multiplication
    dmr_s[0].get(out=dmr_sd)
    dmr_e[1].get(out=dmr_ed)

    # correct first and last block to account for the contacts in multiplication
    # first block
    # L^{\lessgtr}_E_00 = V_10*P^{\lessgtr}_E_00*V_01 + V_10*P^{\lessgtr}_E_01*V_00 + V_00*P^{\lessgtr}_E_10*V_01
    dlg_s[0].get(out=dlg_sd)
    dll_s[0].get(out=dll_sd)

    # last block
    # L^{\lessgtr}_E_nn = V_nn*PL_E_n-1n*V_nn-1 + V_n-1n*PL_E_nn-1*V_nn + V_n-1n*PL_E_nn*V_nn-1
    dlg_e[1].get(out=dlg_ed)
    dll_e[1].get(out=dll_ed)

    mr_s_loc[0].get(out=mr_s0)
    mr_s_loc[1].get(out=mr_s1)
    mr_s_loc[2].get(out=mr_s2)
    mr_e_loc[0].get(out=mr_e0)
    mr_e_loc[1].get(out=mr_e1)
    mr_e_loc[2].get(out=mr_e2)
    lg_s_loc[0].get(out=lg_s0)
    lg_s_loc[1].get(out=lg_s1)
    lg_e_loc[0].get(out=lg_e0)
    lg_e_loc[2].get(out=lg_e1)
    ll_s_loc[0].get(out=ll_s0)
    ll_s_loc[1].get(out=ll_s1)
    ll_e_loc[0].get(out=ll_e0)
    ll_e_loc[2].get(out=ll_e1)
    vh_s_loc[0].get(out=vh_s)
    vh_e_loc[1].get(out=vh_e)
    mb00_loc.get(out=mb00)
    mbNN_loc.get(out=mbNN)


def obc_w_gpu_beynonly(dxr_sd: npt.NDArray[np.complex128],
                          dxr_ed: npt.NDArray[np.complex128],
                          dvh_sd: npt.NDArray[np.complex128],
                          dvh_ed: npt.NDArray[np.complex128],
                          dmr_sd: npt.NDArray[np.complex128],
                          dmr_ed: npt.NDArray[np.complex128],
                          mr_s: tuple,
                          mr_e: tuple,
                          vh_s: npt.NDArray[np.complex128],
                          vh_e: npt.NDArray[np.complex128],
                          mb00: npt.NDArray[np.complex128],
                          mbNN: npt.NDArray[np.complex128],
                          nbc: np.int32,
                          NCpSC: np.int32 = 1,
                          block_inv: bool = False,
                          use_dace: bool = False,
                          validate_dace: bool = False,
                          ref_flag: bool = False,
                          sancho_flag: bool = False):
    
    beyn = beyn_new_gpu.beyn_new_mix
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
    # boundary correction calculations
    cond_l = 0.0
    cond_r = 0.0

    dxr_sd_gpu = cp.empty_like(dxr_sd)
    dxr_ed_gpu = cp.empty_like(dxr_ed)

    # correction for first block
    if not sancho_flag:
        dmr, dxr_sd_gpu, cond_l, min_dEkL = beyn(nbc * NCpSC, cp.asarray(mb00), cp.asarray(mr_s[0]), cp.asarray(mr_s[1]), cp.asarray(mr_s[2]), imag_lim, rr, "L")
        dxr_sd[:, :] = dxr_sd_gpu.get()
        if not np.isnan(cond_l):
            dmr_sd -= dmr.get()
            dvh_sd[:, :] = mr_s[2] @ dxr_sd @ vh_s

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
        dxr_sd[:, :], dmr, dvh_sd[:, :], cond_l = sancho.open_boundary_conditions(mr_s[0], mr_s[2], mr_s[1], vh_s)
        dmr_sd -= dmr
    # correction for last block
    if not sancho_flag:
        dmr, dxr_ed_gpu, cond_r, min_dEkR = beyn(nbc * NCpSC, cp.asarray(mbNN), cp.asarray(mr_e[0]), cp.asarray(mr_e[1]), cp.asarray(mr_e[2]), imag_lim, rr, "R")
        dxr_ed[:, :] = dxr_ed_gpu.get()
        if not np.isnan(cond_r):
            dmr_ed -= dmr.get()
            dvh_ed[:, :] = mr_e[1] @ dxr_ed @ vh_e
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
        dxr_ed[:, :], dmr, dvh_ed[:, :], cond_r = sancho.open_boundary_conditions(mr_e[0], mr_e[1], mr_e[2], vh_e)
        dmr_ed -= dmr
