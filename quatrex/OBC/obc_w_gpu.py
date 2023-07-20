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
from OBC import beyn_cpu
from OBC import sancho
from OBC import dL_OBC_eigenmode_gpu
import typing


def obc_w_sl(vh: cusparse.spmatrix, pg: cusparse.spmatrix, pl: cusparse.spmatrix, pr: cusparse.spmatrix,
             nao: int) -> typing.Tuple[cusparse.spmatrix, cusparse.spmatrix, cusparse.spmatrix]:
    """
    Calculates for the additional helper variables.

    Args:
        vh (cusparse.spmatrix): Effective interaction
        pg (cusparse.spmatrix): Greater Polarization
        pl (cusparse.spmatrix): Lesser Polarization
        pr (cusparse.spmatrix): Retarded Polarization
        nao            (int): Number of atomic orbitals

    Returns:
        typing.Tuple[cusparse.csr_matrix, cusparse.csr_matrix, cusparse.csr_matrix]: M^{r}\left(E\right), L^{>}\left(E\right), L^{<}\left(E\right)
    """

    vh_ct = vh.conjugate().T

    # calculate L^{\lessgtr}\left(E\right)
    lg = vh @ pg @ vh_ct
    ll = vh @ pl @ vh_ct

    # calculate M^{r}\left(E\right)
    mr = cusparse.identity(nao, format="csr") - vh @ pr

    return mr, lg, ll


def obc_w_sc(
    pg: cusparse.csr_matrix,
    pl: cusparse.csr_matrix,
    pr: cusparse.csr_matrix,
    vh: cusparse.csr_matrix,
    mr_sf: cp.ndarray,
    mr_ef: cp.ndarray,
    lg_sf: cp.ndarray,
    lg_ef: cp.ndarray,
    ll_sf: cp.ndarray,
    ll_ef: cp.ndarray,
    dmr_sf: cp.ndarray,
    dmr_ef: cp.ndarray,
    dlg_sf: cp.ndarray,
    dlg_ef: cp.ndarray,
    dll_sf: cp.ndarray,
    dll_ef: cp.ndarray,
    vh_sf: cp.ndarray,
    vh_ef: cp.ndarray,
    bmax: npt.NDArray[np.int32],
    bmin: npt.NDArray[np.int32],
    nbc: int,
):
    """
    Calculates the blocks needed for the boundary correction.

    Args:
        pg (cusparse.csr_matrix): Greater Polarization
        pl (cusparse.csr_matrix): Lesser Polarization
        pr (cusparse.csr_matrix): Retarded Polarization
        vh (cusparse.csr_matrix): Effective Screening
        mr_sf (cp.ndarray): Output start/left blocks (diagonal, upper, lower) of mr
        mr_ef (cp.ndarray): Output end/right blocks (diagonal, upper, lower) of mr
        lg_sf (cp.ndarray): Output start/left blocks (diagonal, upper, lower) of lg
        lg_ef (cp.ndarray): Output end/right blocks (diagonal, upper, lower) of lg
        ll_sf (cp.ndarray): Output start/left blocks (diagonal, upper, lower) of ll
        ll_ef (cp.ndarray): Output end/right blocks (diagonal, upper, lower) of ll
        dmr_sf (cp.ndarray): Output start/left block of mr
        dmr_ef (cp.ndarray): Output end/right block of mr
        dlg_sf (cp.ndarray): Output start/left block of lg
        dlg_ef (cp.ndarray): Output end/right block of lg
        dll_sf (cp.ndarray): Output start/left block of ll
        dll_ef (cp.ndarray): Output end/right block of ll
        vh_sf (cp.ndarray): Output start/left block of vh
        vh_ef (cp.ndarray): Output start/left block of vh
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
    mr_s, lg_s, ll_s, dmr_s, dlg_s, dll_s, vh_s = dL_OBC_eigenmode_gpu.get_mm_obc_dense(
        vh[slb_sd, slb_sd].toarray(order="C"), vh[slb_sd, slb_so].toarray(order="C"), pg[slb_sd,
                                                                                         slb_sd].toarray(order="C"),
        pg[slb_sd, slb_so].toarray(order="C"), pl[slb_sd, slb_sd].toarray(order="C"), pl[slb_sd,
                                                                                         slb_so].toarray(order="C"),
        pr[slb_sd, slb_sd].toarray(order="C"), pr[slb_sd, slb_so].toarray(order="C"), nbc)
    mr_e, lg_e, ll_e, dmr_e, dlg_e, dll_e, vh_e = dL_OBC_eigenmode_gpu.get_mm_obc_dense(
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


def obc_w_sc_alt(
    pg: cusparse.csr_matrix,
    pl: cusparse.csr_matrix,
    pr: cusparse.csr_matrix,
    vh: cusparse.csr_matrix,
    bmax: npt.NDArray[np.int32],
    bmin: npt.NDArray[np.int32],
    nbc: int,
) -> typing.Tuple[typing.Tuple[cp.ndarray, cp.ndarray, cp.ndarray], typing.Tuple[cp.ndarray, cp.ndarray, cp.ndarray],
                  typing.Tuple[cp.ndarray, cp.ndarray, cp.ndarray], typing.Tuple[cp.ndarray, cp.ndarray, cp.ndarray],
                  typing.Tuple[cp.ndarray, cp.ndarray, cp.ndarray], typing.Tuple[cp.ndarray, cp.ndarray, cp.ndarray],
                  cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray]:
    """
    Calculates the blocks needed for the boundary correction.

    Args:
        pg (cusparse.csr_matrix): Greater Polarization
        pl (cusparse.csr_matrix): Lesser Polarization
        pr (cusparse.csr_matrix): Retarded Polarization
        vh (cusparse.csr_matrix): Effective Screening
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
    mr_s, lg_s, ll_s, dmr_s, dlg_s, dll_s, vh_s = dL_OBC_eigenmode_gpu.get_mm_obc_dense(
        vh[slb_sd, slb_sd].toarray(order="C"), vh[slb_sd, slb_so].toarray(order="C"), pg[slb_sd,
                                                                                         slb_sd].toarray(order="C"),
        pg[slb_sd, slb_so].toarray(order="C"), pl[slb_sd, slb_sd].toarray(order="C"), pl[slb_sd,
                                                                                         slb_so].toarray(order="C"),
        pr[slb_sd, slb_sd].toarray(order="C"), pr[slb_sd, slb_so].toarray(order="C"), nbc)
    mr_e, lg_e, ll_e, dmr_e, dlg_e, dll_e, vh_e = dL_OBC_eigenmode_gpu.get_mm_obc_dense(
        vh[slb_ed, slb_ed].toarray(order="C"), vh[slb_ed, slb_eo].conjugate().transpose().toarray(order="C"),
        pg[slb_ed, slb_ed].toarray(order="C"), -pg[slb_ed, slb_eo].conjugate().transpose().toarray(order="C"),
        pl[slb_ed, slb_ed].toarray(order="C"), -pl[slb_ed, slb_eo].conjugate().transpose().toarray(order="C"),
        pr[slb_ed, slb_ed].toarray(order="C"), pr[slb_ed, slb_eo].transpose().toarray(order="C"), nbc)

    return ((mr_s, mr_e), (lg_s, lg_e), (ll_s, ll_e), (vh_s[0], vh_e[1]), (dmr_s[0], dmr_e[1]), (dlg_s[0], dlg_e[1]),
            (dll_s[0], dll_e[1]))
