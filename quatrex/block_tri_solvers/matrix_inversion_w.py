# Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.
"""
Inversion sovler for the screened interaction calculation are implemented.

The formula for the screened interaction is given as:

W^{r}\left(E\right) = \left[\mathbb{I} - \hat(V) P^{r}\left(E\right)\right]^{-1} \hat(V)

W^{\lessgtr}\left(E\right) = W^{r}\left(E\right) P^{\lessgtr}\left(E\right) W^{r}\left(E\right)^{H} 


All of the above formulas need corrections in practice
since we cutoff the semi infinite boundary blocks.
The corrections are needed for both matrix inverse and matrix multiplication.

The beyn algorithm can used for correction of the inverse.
If beyn fails, one can fall back to sancho rubio.

The OBC have to be applied before the inversion.
The function to apply the OBC is found under `OBC/obc_w.py`.


Only diagonal and upper diagonal blocks are calculated of the inverse.
For example, the fast RGF algorithm can be used for the inverse.
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
import cupy as cp
import numpy.typing as npt
from scipy import sparse
from cupyx.scipy import sparse as cusparse


def rgf(bmax_mm: npt.NDArray[np.int32], bmin_mm: npt.NDArray[np.int32], vh: sparse.csr_matrix, mr: sparse.csr_matrix,
        lg: sparse.csr_matrix, ll: sparse.csr_matrix, factor: np.float64, wg_diag: npt.NDArray[np.complex128],
        wg_upper: npt.NDArray[np.complex128], wl_diag: npt.NDArray[np.complex128], wl_upper: npt.NDArray[np.complex128],
        wr_diag: npt.NDArray[np.complex128], wr_upper: npt.NDArray[np.complex128], xr_diag: npt.NDArray[np.complex128],
        dmr_ed: npt.NDArray[np.complex128], dlg_ed: npt.NDArray[np.complex128], dll_ed: npt.NDArray[np.complex128],
        dvh_ed: npt.NDArray[np.complex128], dmr_sd: npt.NDArray[np.complex128], dlg_sd: npt.NDArray[np.complex128],
        dll_sd: npt.NDArray[np.complex128], dvh_sd: npt.NDArray[np.complex128]):
    """
    Inversion of the screened interaction using the RGF algorithm.

    Args:
        bmax_mm (npt.NDArray[np.int32]): Indexes of the end of the blocks after matrix multiplication
        bmin_mm (npt.NDArray[np.int32]): Indexes of the start of the blocks after matrix multiplication
        vh (sparse.csr_matrix): Effective interaction
        mr (sparse.csr_matrix): Helper variable for inversion
        lg (sparse.csr_matrix): Helper variable for inversion
        ll (sparse.csr_matrix): Helper variable for inversion
        factor (np.float64): Factor to smooth certain energies
        wg_diag (npt.NDArray[np.complex128]): Greater screened interaction diagonal blocks
        wg_upper (npt.NDArray[np.complex128]): Greater screened interaction upper diagonal blocks
        wl_diag (npt.NDArray[np.complex128]): Lesser screened interaction diagonal blocks
        wl_upper (npt.NDArray[np.complex128]): Lesser screened interaction upper diagonal blocks
        wr_diag (npt.NDArray[np.complex128]): Retarded screened interaction diagonal blocks
        wr_upper (npt.NDArray[np.complex128]): Retarded screened interaction upper diagonal blocks
        xr_diag (npt.NDArray[np.complex128]): Helper variable diagonal blocks
        dmr_ed (npt.NDArray[np.complex128]): Change of end block of mr through OBC
        dlg_ed (npt.NDArray[np.complex128]): Change of end block of lg through OBC
        dll_ed (npt.NDArray[np.complex128]): Change of end block of ll through OBC
        dvh_ed (npt.NDArray[np.complex128]): Change of end block of vh through OBC
        dmr_sd (npt.NDArray[np.complex128]): Change of start block of mr through OBC
        dlg_sd (npt.NDArray[np.complex128]): Change of start block of lg through OBC
        dll_sd (npt.NDArray[np.complex128]): Change of start block of ll through OBC
        dvh_sd (npt.NDArray[np.complex128]): Change of start block of vh through OBC
    """

    # number of blocks after matrix multiplication
    nb_mm = bmin_mm.size
    # vector of block lengths after matrix multiplication
    lb_vec_mm = bmax_mm - bmin_mm + 1
    # max block size after matrix multiplication
    lb_max_mm = np.max(lb_vec_mm)

    # create buffer for results
    # not true inverse, but build up inverses from either corner
    xr_diag_rgf = np.zeros((nb_mm, lb_max_mm, lb_max_mm), dtype=np.complex128)
    wg_diag_rgf = np.zeros((nb_mm, lb_max_mm, lb_max_mm), dtype=np.complex128)
    wl_diag_rgf = np.zeros((nb_mm, lb_max_mm, lb_max_mm), dtype=np.complex128)
    wr_diag_rgf = np.zeros((nb_mm, lb_max_mm, lb_max_mm), dtype=np.complex128)

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
    xr_lb = np.linalg.inv(mr[slb_lb, slb_lb].toarray() + dmr_ed)
    xr_lb_ct = xr_lb.conjugate().T
    xr_diag_rgf[idx_lb, :lb_lb, :lb_lb] = xr_lb

    # w^{>}_E_nn = x^{r}_E_nn * L^{>}_E_nn * (x^{r}_E_nn).H
    wg_lb = xr_lb @ (lg[slb_lb, slb_lb].toarray() + dlg_ed) @ xr_lb_ct
    wg_diag_rgf[idx_lb, :lb_lb, :lb_lb] = wg_lb

    # w^{<}_E_nn = x^{r}_E_nn * L^{<}_E_nn * (x^{r}_E_nn).H
    wl_lb = xr_lb @ (ll[slb_lb, slb_lb].toarray() + dll_ed) @ xr_lb_ct
    wl_diag_rgf[idx_lb, :lb_lb, :lb_lb] = wl_lb

    # wR_E_nn = xR_E_nn * V_nn
    wr_lb = xr_lb @ (vh[slb_lb, slb_lb].toarray() - dvh_ed)
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
        vh_c = vh[slb_c, slb_c].toarray()
        vh_d = vh[slb_p, slb_c].toarray()
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
        mr_r_ct = mr_r.conjugate().T
        xr_c_ct = xr_c.conjugate().T

        # A^{\lessgtr} = M_E_kk+1 * xR_E_k+1k+1 * L^{\lessgtr}_E_k+1k
        ag = mr_xr @ lg_d
        al = mr_xr @ ll_d
        ag_diff = ag - ag.conjugate().T
        al_diff = al - al.conjugate().T

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
    vh_r = vh[slb_c, slb_p].toarray()
    lg_r = lg[slb_c, slb_p].toarray()
    ll_r = ll[slb_c, slb_p].toarray()

    xr_mr = xr_p @ mr_d
    xr_mr_ct = xr_mr.conjugate().T

    # WR_E_00 = wR_E_00
    wr_diag[0, :lb_f, :lb_f] = wr_diag_rgf[0, :lb_f, :lb_f]

    # WR_E_01 = (V_01 - WR_E_00*M_E_10) * xR_E_11
    # todo if vh_r can be used instead of vh[slb_c,slb_p]
    wr_upper[0, :lb_f, :lb_p] = (vh_r - wr_diag[0, :lb_f, :lb_f] @ mr_d.T) @ xr_p.T

    # XR_E_00 = xR_E_00
    xr_diag[0, :lb_f, :lb_f] = xr_diag_rgf[0, :lb_f, :lb_f]

    # W^{\lessgtr}_E_00 = w^{\lessgtr}_E_00
    wg_diag[0, :lb_f, :lb_f] = wg_diag_rgf[0, :lb_f, :lb_f]
    wl_diag[0, :lb_f, :lb_f] = wl_diag_rgf[0, :lb_f, :lb_f]

    # W^{\lessgtr}_E_01 = xR_E_00*L^{\lessgtr}_01*_xR_E_11.H - xR_E_00*M_E_01*wL_E_11 - w^{\lessgtr}_E_00*M_E_10.H*xR_E_11.H
    xr_p_ct = xr_p.conjugate().T
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
        xr_mr_xr_ct = xr_mr_xr.conjugate().T
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
        ag_diff = ag - ag.conjugate().T
        al_diff = al - al.conjugate().T

        # B^{\lessgtr} = xR_E_kk * M_E_kk-1 * XR_E_k-1k-1 * M_E_k-1k * w^{\lessgtr}_E_kk
        bg = xr_mr_xr_mr @ wg_diag_rgf_c
        bl = xr_mr_xr_mr @ wl_diag_rgf_c
        bg_diff = bg - bg.conjugate().T
        bl_diff = bl - bl.conjugate().T

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
            vh_d = vh[slb_n, slb_c].toarray()
            mr_d = mr[slb_n, slb_c].toarray()
            mr_r = mr[slb_c, slb_n].toarray()
            lg_r = lg[slb_c, slb_n].toarray()
            ll_r = ll[slb_c, slb_n].toarray()

            xr_diag_rgf_n = xr_diag_rgf[idx_ib + 1, :lb_n, :lb_n]
            wg_diag_rgf_n = wg_diag_rgf[idx_ib + 1, :lb_n, :lb_n]
            wl_diag_rgf_n = wl_diag_rgf[idx_ib + 1, :lb_n, :lb_n]
            xr_diag_rgf_n_ct = xr_diag_rgf_n.conjugate().T

            # xRM_next = M_E_k+1k * xR_E_k+1k+1
            xr_mr = xr_diag_rgf_n @ mr_d
            xr_mr_ct = xr_mr.conjugate().T

            # WR_E_kk+1 = (V_k+1k.T - WR_E_kk*M_E_k+1k.T) * xR_E_k+1k+1.T
            # difference between matlab and python silvio todo
            # this line is wrong todo in the second part
            wr_upper_c = vh_d.T @ xr_diag_rgf_n.T - wr_diag_c @ xr_mr.T
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

        if idx_ib < nb_mm - 1:
            wr_upper[idx_ib, :, :] *= factor
            wg_upper[idx_ib, :, :] *= factor
            wl_upper[idx_ib, :, :] *= factor


def rgf_gpu(bmax_mm: npt.NDArray[np.int32], bmin_mm: npt.NDArray[np.int32], vh: cusparse.csr_matrix,
            mr: cusparse.csr_matrix, lg: cusparse.csr_matrix, ll: cusparse.csr_matrix, factor: cp.float64,
            wg_diag: cp.ndarray, wg_upper: cp.ndarray, wl_diag: cp.ndarray, wl_upper: cp.ndarray, wr_diag: cp.ndarray,
            wr_upper: cp.ndarray, xr_diag: cp.ndarray, dmr_ed: cp.ndarray, dlg_ed: cp.ndarray, dll_ed: cp.ndarray,
            dvh_ed: cp.ndarray, dmr_sd: cp.ndarray, dlg_sd: cp.ndarray, dll_sd: cp.ndarray, dvh_sd: cp.ndarray):
    """
    Inversion of the screened interaction using the RGF algorithm.

    Args:
        bmax_mm (npt.NDArray[np.int32]): Indexes of the end of the blocks after matrix multiplication
        bmin_mm (npt.NDArray[np.int32]): Indexes of the start of the blocks after matrix multiplication
        vh (cusparse.csr_matrix): Effective interaction
        mr (cusparse.csr_matrix): Helper variable for inversion
        lg (cusparse.csr_matrix): Helper variable for inversion
        ll (cusparse.csr_matrix): Helper variable for inversion
        factor (cp.float64): Factor to smooth certain energies
        wg_diag (cp.ndarray): Greater screened interaction diagonal blocks
        wg_upper (cp.ndarray): Greater screened interaction upper diagonal blocks
        wl_diag (cp.ndarray): Lesser screened interaction diagonal blocks
        wl_upper (cp.ndarray): Lesser screened interaction upper diagonal blocks
        wr_diag (cp.ndarray): Retarded screened interaction diagonal blocks
        wr_upper (cp.ndarray): Retarded screened interaction upper diagonal blocks
        xr_diag (cp.ndarray): Helper variable diagonal blocks
        dmr_ed (cp.ndarray): Change of end block of mr through OBC
        dlg_ed (cp.ndarray): Change of end block of lg through OBC
        dll_ed (cp.ndarray): Change of end block of ll through OBC
        dvh_ed (cp.ndarray): Change of end block of vh through OBC
        dmr_sd (cp.ndarray): Change of start block of mr through OBC
        dlg_sd (cp.ndarray): Change of start block of lg through OBC
        dll_sd (cp.ndarray): Change of start block of ll through OBC
        dvh_sd (cp.ndarray): Change of start block of vh through OBC
    """

    # number of blocks after matrix multiplication
    nb_mm = bmin_mm.size
    # vector of block lengths after matrix multiplication
    lb_vec_mm = bmax_mm - bmin_mm + 1
    # max block size after matrix multiplication
    lb_max_mm = np.max(lb_vec_mm)

    # create buffer for results
    # not true inverse, but build up inverses from either corner
    xr_diag_rgf = cp.zeros((nb_mm, lb_max_mm, lb_max_mm), dtype=cp.complex128)
    wg_diag_rgf = cp.zeros((nb_mm, lb_max_mm, lb_max_mm), dtype=cp.complex128)
    wl_diag_rgf = cp.zeros((nb_mm, lb_max_mm, lb_max_mm), dtype=cp.complex128)
    wr_diag_rgf = cp.zeros((nb_mm, lb_max_mm, lb_max_mm), dtype=cp.complex128)

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
    xr_lb = cp.linalg.inv(mr[slb_lb, slb_lb].toarray() + dmr_ed)
    xr_lb_ct = xr_lb.conjugate().T
    xr_diag_rgf[idx_lb, :lb_lb, :lb_lb] = xr_lb

    # w^{>}_E_nn = x^{r}_E_nn * L^{>}_E_nn * (x^{r}_E_nn).H
    wg_lb = xr_lb @ (lg[slb_lb, slb_lb].toarray() + dlg_ed) @ xr_lb_ct
    wg_diag_rgf[idx_lb, :lb_lb, :lb_lb] = wg_lb

    # w^{<}_E_nn = x^{r}_E_nn * L^{<}_E_nn * (x^{r}_E_nn).H
    wl_lb = xr_lb @ (ll[slb_lb, slb_lb].toarray() + dll_ed) @ xr_lb_ct
    wl_diag_rgf[idx_lb, :lb_lb, :lb_lb] = wl_lb

    # wR_E_nn = xR_E_nn * V_nn
    wr_lb = xr_lb @ (vh[slb_lb, slb_lb].toarray() - dvh_ed)
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
        vh_c = vh[slb_c, slb_c].toarray()
        vh_d = vh[slb_p, slb_c].toarray()
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
        xr_c = cp.linalg.inv(mr_c - mr_xr @ mr_d)
        xr_diag_rgf[idx_ib, :lb_i, :lb_i] = xr_c

        # conjugate and transpose
        mr_r_ct = mr_r.conjugate().T
        xr_c_ct = xr_c.conjugate().T

        # A^{\lessgtr} = M_E_kk+1 * xR_E_k+1k+1 * L^{\lessgtr}_E_k+1k
        ag = mr_xr @ lg_d
        al = mr_xr @ ll_d
        ag_diff = ag - ag.conjugate().T
        al_diff = al - al.conjugate().T

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
    vh_r = vh[slb_c, slb_p].toarray()
    lg_r = lg[slb_c, slb_p].toarray()
    ll_r = ll[slb_c, slb_p].toarray()

    xr_mr = xr_p @ mr_d
    xr_mr_ct = xr_mr.conjugate().T

    # WR_E_00 = wR_E_00
    wr_diag[0, :lb_f, :lb_f] = wr_diag_rgf[0, :lb_f, :lb_f]

    # WR_E_01 = (V_01 - WR_E_00*M_E_10) * xR_E_11
    # todo if vh_r can be used instead of vh[slb_c,slb_p]
    wr_upper[0, :lb_f, :lb_p] = (vh_r - wr_diag[0, :lb_f, :lb_f] @ mr_d.T) @ xr_p.T

    # XR_E_00 = xR_E_00
    xr_diag[0, :lb_f, :lb_f] = xr_diag_rgf[0, :lb_f, :lb_f]

    # W^{\lessgtr}_E_00 = w^{\lessgtr}_E_00
    wg_diag[0, :lb_f, :lb_f] = wg_diag_rgf[0, :lb_f, :lb_f]
    wl_diag[0, :lb_f, :lb_f] = wl_diag_rgf[0, :lb_f, :lb_f]

    # W^{\lessgtr}_E_01 = xR_E_00*L^{\lessgtr}_01*_xR_E_11.H - xR_E_00*M_E_01*wL_E_11 - w^{\lessgtr}_E_00*M_E_10.H*xR_E_11.H
    xr_p_ct = xr_p.conjugate().T
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
        xr_mr_xr_ct = xr_mr_xr.conjugate().T
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
        ag_diff = ag - ag.conjugate().T
        al_diff = al - al.conjugate().T

        # B^{\lessgtr} = xR_E_kk * M_E_kk-1 * XR_E_k-1k-1 * M_E_k-1k * w^{\lessgtr}_E_kk
        bg = xr_mr_xr_mr @ wg_diag_rgf_c
        bl = xr_mr_xr_mr @ wl_diag_rgf_c
        bg_diff = bg - bg.conjugate().T
        bl_diff = bl - bl.conjugate().T

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
            vh_d = vh[slb_n, slb_c].toarray()
            mr_d = mr[slb_n, slb_c].toarray()
            mr_r = mr[slb_c, slb_n].toarray()
            lg_r = lg[slb_c, slb_n].toarray()
            ll_r = ll[slb_c, slb_n].toarray()

            xr_diag_rgf_n = xr_diag_rgf[idx_ib + 1, :lb_n, :lb_n]
            wg_diag_rgf_n = wg_diag_rgf[idx_ib + 1, :lb_n, :lb_n]
            wl_diag_rgf_n = wl_diag_rgf[idx_ib + 1, :lb_n, :lb_n]
            xr_diag_rgf_n_ct = xr_diag_rgf_n.conjugate().T

            # xRM_next = M_E_k+1k * xR_E_k+1k+1
            xr_mr = xr_diag_rgf_n @ mr_d
            xr_mr_ct = xr_mr.conjugate().T

            # WR_E_kk+1 = (V_k+1k.T - WR_E_kk*M_E_k+1k.T) * xR_E_k+1k+1.T
            # difference between matlab and python silvio todo
            # this line is wrong todo in the second part
            wr_upper_c = vh_d.T @ xr_diag_rgf_n.T - wr_diag_c @ xr_mr.T
            wr_upper[idx_ib, :lb_i, :lb_n] = wr_upper_c

            # W^{\lessgtr}_E_kk+1 = XR_E_kk*(L^{\lessgtr}_E_kk+1*xR_E_k+1k+1.H - M_E_kk+1*w^{\lessgtr}_E_k+1k+1) - W^{\lessgtr}_E_kk*M_E_k+1k.H*xxR_E_k+1k+1.H
            wg_upper_c = xr_diag_c @ (lg_r @ xr_diag_rgf_n_ct - mr_r @ wg_diag_rgf_n) - wg_diag_c @ xr_mr_ct
            wg_upper[idx_ib, :lb_i, :lb_n] = wg_upper_c
            wl_upper_c = xr_diag_c @ (ll_r @ xr_diag_rgf_n_ct - mr_r @ wl_diag_rgf_n) - wl_diag_c @ xr_mr_ct
            wl_upper[idx_ib, :lb_i, :lb_n] = wl_upper_c

    xr_diag *= factor
    wg_diag *= factor
    wl_diag *= factor
    wr_diag *= factor
    wr_upper *= factor
    wg_upper *= factor
    wl_upper *= factor
