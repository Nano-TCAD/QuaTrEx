import numpy as np
import numpy.typing as npt
from scipy import sparse
import time
import typing
import cupy as cp

def rgf_w_opt_standalone(
    vh: sparse.csr_matrix,
    lg: sparse.csr_matrix,
    ll: sparse.csr_matrix,
    mr: sparse.csr_matrix,
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
    nao = lg.shape[0]

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
    #vh_cp_ct = vh_cp.conjugate().transpose()
    #lg = vh_cp @ pg @ vh_cp_ct
    #ll = vh_cp @ pl @ vh_cp_ct
    # calculate M^{r}\left(E\right)

    #mr = sparse.identity(nao, format="csr") - vh_cp @ pr

    # lg = sparse.csr_matrix(lg.get())
    # ll = sparse.csr_matrix(ll.get())
    # mr = sparse.csr_matrix(mr.get())

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

def rgf_w_opt_standalone_batched(
    vh_diag: npt.NDArray[np.complex128],
    vh_upper: npt.NDArray[np.complex128],
    vh_lower: npt.NDArray[np.complex128],
    lg_diag: npt.NDArray[np.complex128],
    lg_upper: npt.NDArray[np.complex128],
    lg_lower: npt.NDArray[np.complex128],
    ll_diag: npt.NDArray[np.complex128],
    ll_upper: npt.NDArray[np.complex128],
    ll_lower: npt.NDArray[np.complex128],
    mr_diag: npt.NDArray[np.complex128],
    mr_upper: npt.NDArray[np.complex128],
    mr_lower: npt.NDArray[np.complex128],
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
    # number of blocks
    nb = bmin.size

    energy_batchsize = lg_diag.shape[1]
    # number of total orbitals (number of atomic orbitals)

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

    # create buffer for results

    # not true inverse, but build up inverses from either corner
    xr_diag_rgf = np.zeros((nb_mm, energy_batchsize, lb_max_mm, lb_max_mm), dtype=np.complex128)
    wg_diag_rgf = np.zeros((nb_mm, energy_batchsize, lb_max_mm, lb_max_mm), dtype=np.complex128)
    wl_diag_rgf = np.zeros((nb_mm, energy_batchsize, lb_max_mm, lb_max_mm), dtype=np.complex128)
    wr_diag_rgf = np.zeros((nb_mm, energy_batchsize, lb_max_mm, lb_max_mm), dtype=np.complex128)

    # True inverse M^-1 off-diagonal blocks
    xr_upper = np.zeros((nb_mm - 1, energy_batchsize, lb_max_mm, lb_max_mm), dtype=np.complex128)
    xr_lower = np.zeros((nb_mm - 1, energy_batchsize, lb_max_mm, lb_max_mm), dtype=np.complex128)

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
    # length last block
    lb_lb = lb_vec_mm[idx_lb]


    
    xr_lb = np.linalg.inv(mr_diag[-1, :, :lb_lb, :lb_lb])
    xr_lb_ct = xr_lb.conjugate().transpose((0,2,1))
    xr_diag_rgf[idx_lb, :, :lb_lb, :lb_lb] = xr_lb

    # w^{>}_E_nn = x^{r}_E_nn * L^{>}_E_nn * (x^{r}_E_nn).H
    wg_lb = xr_lb @ (lg_diag[-1, :, :lb_lb, :lb_lb]) @ xr_lb_ct
    wg_diag_rgf[idx_lb, :, :lb_lb, :lb_lb] = wg_lb

    # w^{<}_E_nn = x^{r}_E_nn * L^{<}_E_nn * (x^{r}_E_nn).H
    wl_lb = xr_lb @ (ll_diag[-1, :, :lb_lb, :lb_lb]) @ xr_lb_ct
    wl_diag_rgf[idx_lb, :, :lb_lb, :lb_lb] = wl_lb

    # wR_E_nn = xR_E_nn * V_nn
    wr_lb = xr_lb @ (vh_diag[-1, :, :lb_lb, :lb_lb])
    wr_diag_rgf[idx_lb, :, :lb_lb, :lb_lb] = wr_lb

    # save the diagonal blocks from the previous step
    xr_c = xr_lb
    wg_c = wg_lb
    wl_c = wl_lb

    # loop over all blocks from last to first
    for idx_ib in reversed(range(0, nb_mm - 1)):
        # block length i
        lb_i = lb_vec_mm[idx_ib]
        lb_p = lb_vec_mm[idx_ib + 1]
        # diagonal blocks from previous step
        # avoids a read out operation
        xr_p = xr_c
        wg_p = wg_c
        wl_p = wl_c

        # # slice of current and previous block
        # slb_c = slice(bmin_mm[idx_ib], bmax_mm[idx_ib] + 1)
        # slb_p = slice(bmin_mm[idx_ib + 1], bmax_mm[idx_ib + 1] + 1)

        # mr_c = mr[slb_c, slb_c].toarray()
        # mr_r = mr[slb_c, slb_p].toarray()
        # mr_d = mr[slb_p, slb_c].toarray()
        # vh_c = vh_cp[slb_c, slb_c].toarray()
        # vh_d = vh_cp[slb_p, slb_c].toarray()
        # lg_c = lg[slb_c, slb_c].toarray()
        # lg_d = lg[slb_p, slb_c].toarray()
        # ll_c = ll[slb_c, slb_c].toarray()
        # ll_d = ll[slb_p, slb_c].toarray()

        # MxR = M_E_kk+1 * xR_E_k+1k+1
        mr_xr = mr_upper[idx_ib, :, :lb_i, :lb_p] @ xr_p

        # xR_E_kk = (M_E_kk - M_E_kk+1*xR_E_k+1k+1*M_E_k+1k)^{-1}
        xr_c = np.linalg.inv(mr_diag[idx_ib, :, :lb_i, :lb_i] - mr_xr @ mr_lower[idx_ib, :, :lb_p, :lb_i])
        xr_diag_rgf[idx_ib, :,  :lb_i, :lb_i] = xr_c

        # conjugate and transpose
        mr_r_ct = mr_upper[idx_ib, :, :lb_i, :lb_p].conjugate().transpose((0, 2, 1))
        xr_c_ct = xr_c.conjugate().transpose((0,2,1))

        # A^{\lessgtr} = M_E_kk+1 * xR_E_k+1k+1 * L^{\lessgtr}_E_k+1k
        ag = mr_xr @ lg_lower[idx_ib, :, :lb_p, :lb_i]
        al = mr_xr @ ll_lower[idx_ib, :, :lb_p, :lb_i]
        ag_diff = ag - ag.conjugate().transpose((0, 2, 1))
        al_diff = al - al.conjugate().transpose((0, 2, 1))

        # w^{\lessgtr}_E_kk = xR_E_kk * (L^{\lessgtr}_E_kk + M_E_kk+1*w^{\lessgtr}_E_k+1k+1*M_E_kk+1.H - (A^{\lessgtr} - A^{\lessgtr}.H)) * xR_E_kk.H
        wg_c = xr_c @ (lg_diag[idx_ib, :, :lb_i, :lb_i] + mr_upper[idx_ib, :, :lb_i, :lb_p] @ wg_p @ mr_r_ct - ag_diff) @ xr_c_ct
        wg_diag_rgf[idx_ib, :, :lb_i, :lb_i] = wg_c

        wl_c = xr_c @ (ll_diag[idx_ib, :, :lb_i, :lb_i] + mr_upper[idx_ib, :, :lb_i, :lb_p] @ wl_p @ mr_r_ct - al_diff) @ xr_c_ct
        wl_diag_rgf[idx_ib, :, :lb_i, :lb_i] = wl_c

        # wR_E_kk = xR_E_kk * (V_kk - M_E_kk+1 * xR_E_k+1k+1 * V_k+1k)
        wr_c = xr_c @ (vh_diag[idx_ib, :, :lb_i, :lb_i] - mr_xr @ np.repeat(vh_lower[idx_ib, np.newaxis, :lb_p, :lb_i], energy_batchsize, axis=0))
        wr_diag_rgf[idx_ib, :,  :lb_i, :lb_i] = wr_c

    # block length 0
    lb_f = lb_vec_mm[0]
    lb_p = lb_vec_mm[1]

    # slice of current and previous block
    slb_c = slice(bmin_mm[0], bmax_mm[0] + 1)
    slb_p = slice(bmin_mm[1], bmax_mm[1] + 1)

    # WARNING the last read blocks from the above for loop are used

    #vh_r = vh_cp[slb_c, slb_p].toarray()
    #vh_d = vh_cp[slb_p, slb_c].toarray()
    #lg_r = lg[slb_c, slb_p].toarray()
    #ll_r = ll[slb_c, slb_p].toarray()

    xr_mr = xr_p @ mr_lower[0, :, :lb_p, :lb_f]
    xr_mr_ct = xr_mr.conjugate().transpose((0,2,1))

    # XR_E_00 = xR_E_00
    xr_diag[0, :, :lb_f, :lb_f] = xr_diag_rgf[0, :, :lb_f, :lb_f]
    xr_upper[0, :, :lb_f, :lb_p] = -xr_diag_rgf[0, :,  :lb_f, :lb_f] @ mr_upper[0, :, :lb_f, :lb_p] @ xr_p
    xr_lower[0, :,  :lb_p, :lb_f] = -xr_p @ mr_lower[0, :, :lb_p, :lb_f] @ xr_diag_rgf[0, :lb_f, :lb_f]

    # WR_E_00 = wR_E_00
    #wr_diag[0, :lb_f, :lb_f] = wr_diag_rgf[0, :lb_f, :lb_f]
    wr_diag[0, :, :lb_f, :lb_f] = xr_diag[0, :, :lb_f, :lb_f] @ vh_diag[0, :, :lb_f, :lb_f] + xr_upper[0, :, :lb_f, :lb_p] @ np.repeat(vh_lower[0, np.newaxis,:lb_p, :lb_f], energy_batchsize, axis=0)

    # WR_E_01 = (V_01 - WR_E_00*M_E_10) * xR_E_11
    # todo if vh_r can be used instead of vh_cp[slb_c,slb_p]
    wr_upper[0, :lb_f, :lb_p] = (np.repeat(vh_upper[0, np.newaxis, :lb_f, :lb_p], energy_batchsize, axis=0) - wr_diag[0, :,  :lb_f, :lb_f] @ mr_lower[0, :, :lb_p, :lb_f].transpose((0,2,1))) @ xr_p.transpose((0,2,1))

    # W^{\lessgtr}_E_00 = w^{\lessgtr}_E_00
    wg_diag[0, :, :lb_f, :lb_f] = wg_diag_rgf[0, :, :lb_f, :lb_f]
    wl_diag[0, :, :lb_f, :lb_f] = wl_diag_rgf[0, :, :lb_f, :lb_f]

    # W^{\lessgtr}_E_01 = xR_E_00*L^{\lessgtr}_01*_xR_E_11.H - xR_E_00*M_E_01*wL_E_11 - w^{\lessgtr}_E_00*M_E_10.H*xR_E_11.H
    xr_p_ct = xr_p.conjugate().transpose((0,2,1))
    wg_upper[0, :, :lb_f, :lb_p] = xr_c @ lg_upper[0, :, :lb_f, :lb_p] @ xr_p_ct - xr_c @ mr_upper[0, :, :lb_f, :lb_p] @ wg_p - wg_c @ xr_mr_ct
    wl_upper[0, :, :lb_f, :lb_p] = xr_c @ ll_upper[0, :, :lb_f, :lb_p] @ xr_p_ct - xr_c @ mr_upper[0, :, :lb_f, :lb_p] @ wl_p - wl_c @ xr_mr_ct

    # loop from left to right corner
    xr_diag_rgf_n = xr_diag_rgf[1, :, :lb_p, :lb_p]
    xr_diag_c = xr_diag_rgf[0, :, :lb_f, :lb_f]

    wg_diag_rgf_n = wg_diag_rgf[1, :, :lb_p, :lb_p]
    wg_diag_c = wg_diag_rgf[0, :, :lb_f, :lb_f]

    wl_diag_rgf_n = wl_diag_rgf[1, :, :lb_p, :lb_p]
    wl_diag_c = wl_diag_rgf[0, :,  :lb_f, :lb_f]

    wr_upper_c = wr_upper[0, :,  :lb_f, :lb_p]

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
        lb_p = lb_vec_mm[idx_ib - 1]
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

        #mr_l = mr[slb_c, slb_p].toarray()
        #mr_u = mr[slb_p, slb_c].toarray()
        #lg_l = lg[slb_c, slb_p].toarray()
        #ll_l = ll[slb_c, slb_p].toarray()

        # xRM = xR_E_kk * M_E_kk-1
        xr_mr_xr = xr_mr @ xr_diag_p
        xr_mr_xr_ct = xr_mr_xr.conjugate().transpose((0,2,1))
        xr_mr_xr_mr = xr_mr_xr @ mr_upper[idx_ib - 1, :, :lb_p, :lb_i]

        # WR_E_kk = wR_E_kk - xR_E_kk*M_E_kk-1*WR_E_k-1k
        wr_diag_rgf_c = wr_diag_rgf[idx_ib, :,  :lb_i, :lb_i]
        # todo if wr_upper_p can be used wr_upper[idx_ib-1,:lb_vec_mm[idx_ib-1],:lb_i]
        #wr_diag_c = wr_diag_rgf_c - xr_mr @ wr_upper[idx_ib - 1, :lb_vec_mm[idx_ib - 1], :lb_i]
        #wr_diag[idx_ib, :lb_i, :lb_i] = wr_diag_c

        # XR_E_kk = xR_E_kk + (xR_E_kk * M_E_kk-1 * XR_E_k-1k-1 * M_E_k-1k * xR_E_kk)
        xr_diag_c = xr_diag_rgf_c + xr_mr_xr_mr @ xr_diag_rgf_c
        xr_diag[idx_ib, :, :lb_i, :lb_i] = xr_diag_c

        # A^{\lessgtr} = xR_E_kk * L^{\lessgtr}_E_kk-1 * XR_E_k-1k-1.H * (xR_E_kk * M_E_kk-1).H
        ag = xr_diag_rgf_c @ lg_lower[idx_ib -1, :, :lb_i, :lb_p] @ xr_mr_xr_ct
        al = xr_diag_rgf_c @ ll_lower[idx_ib -1, :, :lb_i, :lb_p] @ xr_mr_xr_ct
        ag_diff = ag - ag.conjugate().transpose((0,2,1))
        al_diff = al - al.conjugate().transpose((0,2,1))

        # B^{\lessgtr} = xR_E_kk * M_E_kk-1 * XR_E_k-1k-1 * M_E_k-1k * w^{\lessgtr}_E_kk
        bg = xr_mr_xr_mr @ wg_diag_rgf_c
        bl = xr_mr_xr_mr @ wl_diag_rgf_c
        bg_diff = bg - bg.conjugate().transpose((0,2,1))
        bl_diff = bl - bl.conjugate().transpose((0,2,1))

        #W^{\lessgtr}_E_kk = w^{\lessgtr}_E_kk + xR_E_kk*M_E_kk-1*W^{\lessgtr}_E_k-1k-1*(xR_E_kk*M_E_kk-1).H - (A^{\lessgtr}-A^{\lessgtr}.H) + (B^{\lessgtr}-B^{\lessgtr}.H)
        wg_diag_c = wg_diag_rgf_c + xr_mr @ wg_diag_p @ xr_mr_ct - ag_diff + bg_diff
        wl_diag_c = wl_diag_rgf_c + xr_mr @ wl_diag_p @ xr_mr_ct - al_diff + bl_diff
        wg_diag[idx_ib, :, :lb_i, :lb_i] = wg_diag_c
        wl_diag[idx_ib, :, :lb_i, :lb_i] = wl_diag_c

        # following code block has problems
        if idx_ib < nb_mm - 1:
            # block length i
            lb_n = lb_vec_mm[idx_ib + 1]
            # slice of current and previous block
            slb_n = slice(bmin_mm[idx_ib + 1], bmax_mm[idx_ib + 1] + 1)
            #slb_p = slice(bmin_mm[idx_ib-1], bmax_mm[idx_ib] + 1)

            # vh_d = vh_cp[slb_n, slb_c].toarray()
            # vh_r = vh_cp[slb_p, slb_c].toarray()
            # vh_c = vh_cp[slb_c, slb_c].toarray()
            # mr_d = mr[slb_n, slb_c].toarray()
            # mr_r = mr[slb_c, slb_n].toarray()
            # lg_r = lg[slb_c, slb_n].toarray()
            # ll_r = ll[slb_c, slb_n].toarray()

            xr_diag_rgf_n = xr_diag_rgf[idx_ib + 1, :, :lb_n, :lb_n]
            wg_diag_rgf_n = wg_diag_rgf[idx_ib + 1, :, :lb_n, :lb_n]
            wl_diag_rgf_n = wl_diag_rgf[idx_ib + 1, :, :lb_n, :lb_n]
            xr_diag_rgf_n_ct = xr_diag_rgf_n.conjugate().transpose((0,2,1))

            # xRM_next = M_E_k+1k * xR_E_k+1k+1
            xr_mr = xr_diag_rgf_n @ mr_lower[idx_ib, :, :lb_n, :lb_i]
            xr_mr_ct = xr_mr.conjugate().transpose((0,2,1))

            #xr_upper and xr_lower
            xr_upper[idx_ib, :,  :lb_i, :lb_n] = -xr_diag_c @ mr_upper[idx_ib, :, :lb_i, :lb_n] @ xr_diag_rgf_n
            xr_lower[idx_ib, :,  :lb_n, :lb_i] = -xr_diag_rgf_n @ mr_lower[idx_ib, :, :lb_n, :lb_i] @ xr_diag_c
            
            wr_diag_c = xr_lower[idx_ib - 1, :, :lb_i, :lb_n] @ np.repeat(vh_upper[idx_ib - 1, np.newaxis, :lb_p, :lb_i], energy_batchsize, axis=0) + xr_diag[idx_ib, :lb_i, :lb_i] @ vh_diag[idx_ib, :, :lb_i, :lb_i] + \
                xr_upper[idx_ib, :lb_i, :lb_n] @ np.repeat(vh_lower[idx_ib, np.newaxis, :lb_n, :lb_i], energy_batchsize, axis=0)
            wr_diag[idx_ib, :, :lb_i, :lb_i] = wr_diag_c

            # WR_E_kk+1 = (V_k+1k.T - WR_E_kk*M_E_k+1k.T) * xR_E_k+1k+1.T
            # difference between matlab and python silvio todo
            # this line is wrong todo in the second part
            wr_upper_c = np.repeat(vh_lower[idx_ib, np.newaxis, :lb_n, :lb_i], energy_batchsize, axis = 0).transpose((0,2,1)) @ xr_diag_rgf_n.transpose((0,2,1)) - wr_diag_c @ xr_mr.transpose((0,2,1))
            wr_upper[idx_ib, :, :lb_i, :lb_n] = wr_upper_c

            # W^{\lessgtr}_E_kk+1 = XR_E_kk*(L^{\lessgtr}_E_kk+1*xR_E_k+1k+1.H - M_E_kk+1*w^{\lessgtr}_E_k+1k+1) - W^{\lessgtr}_E_kk*M_E_k+1k.H*xxR_E_k+1k+1.H
            wg_upper_c = xr_diag_c @ (lg_upper[idx_ib, :, :lb_i, :lb_n] @ xr_diag_rgf_n_ct - mr_upper[idx_ib, :, :lb_i, :lb_n] @ wg_diag_rgf_n) - wg_diag_c @ xr_mr_ct
            wg_upper[idx_ib, :lb_i, :lb_n] = wg_upper_c
            wl_upper_c = xr_diag_c @ (ll_upper[idx_ib, :, :lb_i, :lb_n] @ xr_diag_rgf_n_ct - mr_upper[idx_ib, :, :lb_i, :lb_n] @ wl_diag_rgf_n) - wl_diag_c @ xr_mr_ct
            wl_upper[idx_ib, :lb_i, :lb_n] = wl_upper_c
        else:
            #vh_r = vh_cp[slb_p, slb_c].toarray()
            #vh_c = vh_cp[slb_c, slb_c].toarray() - dvh_ed
            wr_diag[idx_ib, :lb_i, :lb_i] = xr_lower[idx_ib - 1, :lb_i, :lb_n] @ np.repeat(vh_upper[idx_ib - 1, np.newaxis, :lb_p, :lb_i], energy_batchsize, axis=0) + xr_diag[idx_ib, :lb_i, :lb_i] @ vh_diag[idx_ib, :, :lb_i, :lb_i]

    for idx_ib in range(0, nb_mm):
        # xr_diag[idx_ib, :, :] *= factor
        # wg_diag[idx_ib, :, :] *= factor
        # wl_diag[idx_ib, :, :] *= factor
        # wr_diag[idx_ib, :, :] *= factor
        dosw[: , idx_ib] = 1j * np.trace(wr_diag[idx_ib,:, :, :] - wr_diag[idx_ib, :, :, :].conjugate().transpose((0,2,1)), axis1=1,axis2=2)
        nEw[:, idx_ib] = -1j * np.trace(wl_diag[idx_ib, :, :], axis1=1,axis2=2)
        nPw[:, idx_ib] = 1j * np.trace(wg_diag[idx_ib, :, :], axis1=1,axis2=2)
        # if idx_ib < nb_mm - 1:
        #     wr_upper[idx_ib, :, :] *= factor
        #     wg_upper[idx_ib, :, :] *= factor
        #     wl_upper[idx_ib, :, :] *= factor

def rgf_w_opt_standalone_batched_gpu(
    vh_diag: npt.NDArray[np.complex128],
    vh_upper: npt.NDArray[np.complex128],
    vh_lower: npt.NDArray[np.complex128],
    lg_diag: npt.NDArray[np.complex128],
    lg_upper: npt.NDArray[np.complex128],
    lg_lower: npt.NDArray[np.complex128],
    ll_diag: npt.NDArray[np.complex128],
    ll_upper: npt.NDArray[np.complex128],
    ll_lower: npt.NDArray[np.complex128],
    mr_diag: npt.NDArray[np.complex128],
    mr_upper: npt.NDArray[np.complex128],
    mr_lower: npt.NDArray[np.complex128],
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
    # number of blocks
    nb = bmin.size

    energy_batchsize = lg_diag.shape[1]
    # number of total orbitals (number of atomic orbitals)

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


    # Upload to GPU
    # vh_diag_gpu = cp.asarray(vh_diag)
    # lg_diag_gpu = cp.asarray(lg_diag)
    # lg_upper_gpu = cp.asarray(lg_upper)
    # lg_lower_gpu = cp.asarray(lg_lower)
    # ll_diag_gpu = cp.asarray(ll_diag)
    # ll_upper_gpu = cp.asarray(ll_upper)
    # ll_lower_gpu = cp.asarray(ll_lower)
    # mr_diag_gpu = cp.asarray(mr_diag)
    # mr_upper_gpu = cp.asarray(mr_upper)
    # mr_lower_gpu = cp.asarray(mr_lower)

    vh_diag_gpu = cp.empty_like(vh_diag)
    lg_diag_gpu = cp.empty_like(lg_diag)
    lg_upper_gpu = cp.empty_like(lg_upper)
    lg_lower_gpu = cp.empty_like(lg_lower)
    ll_diag_gpu = cp.empty_like(ll_diag)
    ll_upper_gpu = cp.empty_like(ll_upper)
    ll_lower_gpu = cp.empty_like(ll_lower)
    mr_diag_gpu = cp.empty_like(mr_diag)
    mr_upper_gpu = cp.empty_like(mr_upper)
    mr_lower_gpu = cp.empty_like(mr_lower)

    for i in range(len(vh_diag)):
        if i < len(vh_diag) - 1:
            lg_upper_gpu[i].set(lg_upper[i])
            lg_lower_gpu[i].set(lg_lower[i])
            ll_upper_gpu[i].set(ll_upper[i])
            ll_lower_gpu[i].set(ll_lower[i])
            mr_upper_gpu[i].set(mr_upper[i])
            mr_lower_gpu[i].set(mr_lower[i])

        vh_diag_gpu[i].set(vh_diag[i])
        lg_diag_gpu[i].set(lg_diag[i])
        ll_diag_gpu[i].set(ll_diag[i])
        mr_diag_gpu[i].set(mr_diag[i])


    vh_upper_gpu = cp.asarray(vh_upper)
    vh_lower_gpu = cp.asarray(vh_lower)


    # not true inverse, but build up inverses from either corner
    xr_diag_rgf = cp.zeros((nb_mm, energy_batchsize, lb_max_mm, lb_max_mm), dtype=cp.cfloat)
    wg_diag_rgf = cp.zeros((nb_mm, energy_batchsize, lb_max_mm, lb_max_mm), dtype=cp.cfloat)
    wl_diag_rgf = cp.zeros((nb_mm, energy_batchsize, lb_max_mm, lb_max_mm), dtype=cp.cfloat)
    wr_diag_rgf = cp.zeros((nb_mm, energy_batchsize, lb_max_mm, lb_max_mm), dtype=cp.cfloat)

    # True inverse M^-1 off-diagonal blocks
    xr_upper = cp.zeros((nb_mm - 1, energy_batchsize, lb_max_mm, lb_max_mm), dtype=cp.cfloat)
    xr_lower = cp.zeros((nb_mm - 1, energy_batchsize, lb_max_mm, lb_max_mm), dtype=cp.cfloat)

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
    # length last block
    lb_lb = lb_vec_mm[idx_lb]


    gpu_identity = cp.identity(lb_lb, dtype=cp.cfloat)
    gpu_identity_batch = cp.repeat(gpu_identity[cp.newaxis, :, :], energy_batchsize, axis=0)
    xr_lb = cp.linalg.solve(mr_diag_gpu[-1, :, :lb_lb, :lb_lb], gpu_identity_batch)
    #xr_lb = np.linalg.inv(mr_diag[-1, :, :lb_lb, :lb_lb])
    xr_lb_ct = xr_lb.conjugate().transpose((0,2,1))
    xr_diag_rgf[idx_lb, :, :lb_lb, :lb_lb] = xr_lb

    # w^{>}_E_nn = x^{r}_E_nn * L^{>}_E_nn * (x^{r}_E_nn).H
    wg_lb = xr_lb @ (lg_diag_gpu[-1, :, :lb_lb, :lb_lb]) @ xr_lb_ct
    wg_diag_rgf[idx_lb, :, :lb_lb, :lb_lb] = wg_lb

    # w^{<}_E_nn = x^{r}_E_nn * L^{<}_E_nn * (x^{r}_E_nn).H
    wl_lb = xr_lb @ (ll_diag_gpu[-1, :, :lb_lb, :lb_lb]) @ xr_lb_ct
    wl_diag_rgf[idx_lb, :, :lb_lb, :lb_lb] = wl_lb

    # wR_E_nn = xR_E_nn * V_nn
    wr_lb = xr_lb @ (vh_diag_gpu[-1, :, :lb_lb, :lb_lb])
    wr_diag_rgf[idx_lb, :, :lb_lb, :lb_lb] = wr_lb

    # save the diagonal blocks from the previous step
    xr_c = xr_lb
    wg_c = wg_lb
    wl_c = wl_lb

    # loop over all blocks from last to first
    for idx_ib in reversed(range(0, nb_mm - 1)):
        # block length i
        lb_i = lb_vec_mm[idx_ib]
        lb_p = lb_vec_mm[idx_ib + 1]
        # diagonal blocks from previous step
        # avoids a read out operation
        xr_p = xr_c
        wg_p = wg_c
        wl_p = wl_c

        # # slice of current and previous block
        # slb_c = slice(bmin_mm[idx_ib], bmax_mm[idx_ib] + 1)
        # slb_p = slice(bmin_mm[idx_ib + 1], bmax_mm[idx_ib + 1] + 1)

        # mr_c = mr[slb_c, slb_c].toarray()
        # mr_r = mr[slb_c, slb_p].toarray()
        # mr_d = mr[slb_p, slb_c].toarray()
        # vh_c = vh_cp[slb_c, slb_c].toarray()
        # vh_d = vh_cp[slb_p, slb_c].toarray()
        # lg_c = lg[slb_c, slb_c].toarray()
        # lg_d = lg[slb_p, slb_c].toarray()
        # ll_c = ll[slb_c, slb_c].toarray()
        # ll_d = ll[slb_p, slb_c].toarray()

        # MxR = M_E_kk+1 * xR_E_k+1k+1
        mr_xr = mr_upper_gpu[idx_ib, :, :lb_i, :lb_p] @ xr_p

        # xR_E_kk = (M_E_kk - M_E_kk+1*xR_E_k+1k+1*M_E_k+1k)^{-1}
        #xr_c = np.linalg.inv(mr_diag[idx_ib, :, :lb_i, :lb_i] - mr_xr @ mr_lower[idx_ib, :, :lb_p, :lb_i])
        xr_c = cp.linalg.solve(mr_diag_gpu[idx_ib, :, :lb_i, :lb_i] - mr_xr @ mr_lower_gpu[idx_ib, :, :lb_p, :lb_i], gpu_identity_batch)
        xr_diag_rgf[idx_ib, :,  :lb_i, :lb_i] = xr_c

        # conjugate and transpose
        mr_r_ct = mr_upper_gpu[idx_ib, :, :lb_i, :lb_p].conjugate().transpose((0, 2, 1))
        xr_c_ct = xr_c.conjugate().transpose((0,2,1))

        # A^{\lessgtr} = M_E_kk+1 * xR_E_k+1k+1 * L^{\lessgtr}_E_k+1k
        ag = mr_xr @ lg_lower_gpu[idx_ib, :, :lb_p, :lb_i]
        al = mr_xr @ ll_lower_gpu[idx_ib, :, :lb_p, :lb_i]
        ag_diff = ag - ag.conjugate().transpose((0, 2, 1))
        al_diff = al - al.conjugate().transpose((0, 2, 1))

        # w^{\lessgtr}_E_kk = xR_E_kk * (L^{\lessgtr}_E_kk + M_E_kk+1*w^{\lessgtr}_E_k+1k+1*M_E_kk+1.H - (A^{\lessgtr} - A^{\lessgtr}.H)) * xR_E_kk.H
        wg_c = xr_c @ (lg_diag_gpu[idx_ib, :, :lb_i, :lb_i] + mr_upper_gpu[idx_ib, :, :lb_i, :lb_p] @ wg_p @ mr_r_ct - ag_diff) @ xr_c_ct
        wg_diag_rgf[idx_ib, :, :lb_i, :lb_i] = wg_c

        wl_c = xr_c @ (ll_diag_gpu[idx_ib, :, :lb_i, :lb_i] + mr_upper_gpu[idx_ib, :, :lb_i, :lb_p] @ wl_p @ mr_r_ct - al_diff) @ xr_c_ct
        wl_diag_rgf[idx_ib, :, :lb_i, :lb_i] = wl_c

        # wR_E_kk = xR_E_kk * (V_kk - M_E_kk+1 * xR_E_k+1k+1 * V_k+1k)
        wr_c = xr_c @ (vh_diag_gpu[idx_ib, :, :lb_i, :lb_i] - mr_xr @ cp.repeat(vh_lower_gpu[idx_ib, cp.newaxis, :lb_p, :lb_i], energy_batchsize, axis=0))
        wr_diag_rgf[idx_ib, :,  :lb_i, :lb_i] = wr_c

    # block length 0
    lb_f = lb_vec_mm[0]
    lb_p = lb_vec_mm[1]

    # slice of current and previous block
    slb_c = slice(bmin_mm[0], bmax_mm[0] + 1)
    slb_p = slice(bmin_mm[1], bmax_mm[1] + 1)

    # WARNING the last read blocks from the above for loop are used

    #vh_r = vh_cp[slb_c, slb_p].toarray()
    #vh_d = vh_cp[slb_p, slb_c].toarray()
    #lg_r = lg[slb_c, slb_p].toarray()
    #ll_r = ll[slb_c, slb_p].toarray()


    xr_diag_gpu = cp.zeros_like(xr_diag_rgf)
    wg_diag_gpu = cp.zeros_like(wg_diag_rgf)
    wg_upper_gpu = cp.zeros_like(xr_upper)
    wl_diag_gpu = cp.zeros_like(wl_diag_rgf)
    wl_upper_gpu = cp.zeros_like(xr_upper)
    wr_diag_gpu = cp.zeros_like(wr_diag_rgf)
    wr_upper_gpu = cp.zeros_like(xr_upper)

    xr_mr = xr_p @ mr_lower_gpu[0, :, :lb_p, :lb_f]
    xr_mr_ct = xr_mr.conjugate().transpose((0,2,1))

    # XR_E_00 = xR_E_00
    xr_diag_gpu[0, :, :lb_f, :lb_f] = xr_diag_rgf[0, :, :lb_f, :lb_f]
    xr_upper[0, :, :lb_f, :lb_p] = -xr_diag_rgf[0, :,  :lb_f, :lb_f] @ mr_upper_gpu[0, :, :lb_f, :lb_p] @ xr_p
    xr_lower[0, :,  :lb_p, :lb_f] = -xr_p @ mr_lower_gpu[0, :, :lb_p, :lb_f] @ xr_diag_rgf[0, :lb_f, :lb_f]

    # WR_E_00 = wR_E_00
    #wr_diag[0, :lb_f, :lb_f] = wr_diag_rgf[0, :lb_f, :lb_f]
    wr_diag_gpu[0, :, :lb_f, :lb_f] = xr_diag_gpu[0, :, :lb_f, :lb_f] @ vh_diag_gpu[0, :, :lb_f, :lb_f] + xr_upper[0, :, :lb_f, :lb_p] @ cp.repeat(vh_lower_gpu[0, cp.newaxis,:lb_p, :lb_f], energy_batchsize, axis=0)

    # WR_E_01 = (V_01 - WR_E_00*M_E_10) * xR_E_11
    # todo if vh_r can be used instead of vh_cp[slb_c,slb_p]
    wr_upper_gpu[0, :lb_f, :lb_p] = (cp.repeat(vh_upper_gpu[0, cp.newaxis, :lb_f, :lb_p], energy_batchsize, axis=0) - wr_diag_gpu[0, :,  :lb_f, :lb_f] @ mr_lower_gpu[0, :, :lb_p, :lb_f].transpose((0,2,1))) @ xr_p.transpose((0,2,1))

    # W^{\lessgtr}_E_00 = w^{\lessgtr}_E_00
    wg_diag_gpu[0, :, :lb_f, :lb_f] = wg_diag_rgf[0, :, :lb_f, :lb_f]
    wl_diag_gpu[0, :, :lb_f, :lb_f] = wl_diag_rgf[0, :, :lb_f, :lb_f]

    # W^{\lessgtr}_E_01 = xR_E_00*L^{\lessgtr}_01*_xR_E_11.H - xR_E_00*M_E_01*wL_E_11 - w^{\lessgtr}_E_00*M_E_10.H*xR_E_11.H
    xr_p_ct = xr_p.conjugate().transpose((0,2,1))
    wg_upper_gpu[0, :, :lb_f, :lb_p] = xr_c @ lg_upper_gpu[0, :, :lb_f, :lb_p] @ xr_p_ct - xr_c @ mr_upper_gpu[0, :, :lb_f, :lb_p] @ wg_p - wg_c @ xr_mr_ct
    wl_upper_gpu[0, :, :lb_f, :lb_p] = xr_c @ ll_upper_gpu[0, :, :lb_f, :lb_p] @ xr_p_ct - xr_c @ mr_upper_gpu[0, :, :lb_f, :lb_p] @ wl_p - wl_c @ xr_mr_ct

    # loop from left to right corner
    xr_diag_rgf_n = xr_diag_rgf[1, :, :lb_p, :lb_p]
    xr_diag_c = xr_diag_rgf[0, :, :lb_f, :lb_f]

    wg_diag_rgf_n = wg_diag_rgf[1, :, :lb_p, :lb_p]
    wg_diag_c = wg_diag_rgf[0, :, :lb_f, :lb_f]

    wl_diag_rgf_n = wl_diag_rgf[1, :, :lb_p, :lb_p]
    wl_diag_c = wl_diag_rgf[0, :,  :lb_f, :lb_f]

    wr_upper_c = wr_upper_gpu[0, :,  :lb_f, :lb_p]

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
        lb_p = lb_vec_mm[idx_ib - 1]
        # slice of current and previous block
        slb_c = slice(bmin_mm[idx_ib], bmax_mm[idx_ib] + 1)
        slb_p = slice(bmin_mm[idx_ib - 1], bmax_mm[idx_ib - 1] + 1)

        # if(idx_ib == nb_mm -2):
        #     print("second last block")

        # blocks from previous step
        # avoids a read out operation
        xr_diag_rgf_c = xr_diag_rgf_n
        xr_diag_p = xr_diag_c
        wg_diag_rgf_c = wg_diag_rgf_n
        wg_diag_p = wg_diag_c
        wl_diag_rgf_c = wl_diag_rgf_n
        wl_diag_p = wl_diag_c
        wr_upper_p = wr_upper_c

        #mr_l = mr[slb_c, slb_p].toarray()
        #mr_u = mr[slb_p, slb_c].toarray()
        #lg_l = lg[slb_c, slb_p].toarray()
        #ll_l = ll[slb_c, slb_p].toarray()

        # xRM = xR_E_kk * M_E_kk-1
        xr_mr_xr = xr_mr @ xr_diag_p
        xr_mr_xr_ct = xr_mr_xr.conjugate().transpose((0,2,1))
        xr_mr_xr_mr = xr_mr_xr @ mr_upper_gpu[idx_ib - 1, :, :lb_p, :lb_i]

        # WR_E_kk = wR_E_kk - xR_E_kk*M_E_kk-1*WR_E_k-1k
        wr_diag_rgf_c = wr_diag_rgf[idx_ib, :,  :lb_i, :lb_i]
        # todo if wr_upper_p can be used wr_upper[idx_ib-1,:lb_vec_mm[idx_ib-1],:lb_i]
        #wr_diag_c = wr_diag_rgf_c - xr_mr @ wr_upper[idx_ib - 1, :lb_vec_mm[idx_ib - 1], :lb_i]
        #wr_diag[idx_ib, :lb_i, :lb_i] = wr_diag_c

        # XR_E_kk = xR_E_kk + (xR_E_kk * M_E_kk-1 * XR_E_k-1k-1 * M_E_k-1k * xR_E_kk)
        xr_diag_c = xr_diag_rgf_c + xr_mr_xr_mr @ xr_diag_rgf_c
        xr_diag_gpu[idx_ib, :, :lb_i, :lb_i] = xr_diag_c

        # A^{\lessgtr} = xR_E_kk * L^{\lessgtr}_E_kk-1 * XR_E_k-1k-1.H * (xR_E_kk * M_E_kk-1).H
        ag = xr_diag_rgf_c @ lg_lower_gpu[idx_ib -1, :, :lb_i, :lb_p] @ xr_mr_xr_ct
        al = xr_diag_rgf_c @ ll_lower_gpu[idx_ib -1, :, :lb_i, :lb_p] @ xr_mr_xr_ct
        ag_diff = ag - ag.conjugate().transpose((0,2,1))
        al_diff = al - al.conjugate().transpose((0,2,1))

        # B^{\lessgtr} = xR_E_kk * M_E_kk-1 * XR_E_k-1k-1 * M_E_k-1k * w^{\lessgtr}_E_kk
        bg = xr_mr_xr_mr @ wg_diag_rgf_c
        bl = xr_mr_xr_mr @ wl_diag_rgf_c
        bg_diff = bg - bg.conjugate().transpose((0,2,1))
        bl_diff = bl - bl.conjugate().transpose((0,2,1))

        #W^{\lessgtr}_E_kk = w^{\lessgtr}_E_kk + xR_E_kk*M_E_kk-1*W^{\lessgtr}_E_k-1k-1*(xR_E_kk*M_E_kk-1).H - (A^{\lessgtr}-A^{\lessgtr}.H) + (B^{\lessgtr}-B^{\lessgtr}.H)
        wg_diag_c = wg_diag_rgf_c + xr_mr @ wg_diag_p @ xr_mr_ct - ag_diff + bg_diff
        wl_diag_c = wl_diag_rgf_c + xr_mr @ wl_diag_p @ xr_mr_ct - al_diff + bl_diff
        wg_diag_gpu[idx_ib, :, :lb_i, :lb_i] = wg_diag_c
        wl_diag_gpu[idx_ib, :, :lb_i, :lb_i] = wl_diag_c

        # following code block has problems
        if idx_ib < nb_mm - 1:
            # block length i
            lb_n = lb_vec_mm[idx_ib + 1]
            # slice of current and previous block
            slb_n = slice(bmin_mm[idx_ib + 1], bmax_mm[idx_ib + 1] + 1)
            #slb_p = slice(bmin_mm[idx_ib-1], bmax_mm[idx_ib] + 1)

            # vh_d = vh_cp[slb_n, slb_c].toarray()
            # vh_r = vh_cp[slb_p, slb_c].toarray()
            # vh_c = vh_cp[slb_c, slb_c].toarray()
            # mr_d = mr[slb_n, slb_c].toarray()
            # mr_r = mr[slb_c, slb_n].toarray()
            # lg_r = lg[slb_c, slb_n].toarray()
            # ll_r = ll[slb_c, slb_n].toarray()

            xr_diag_rgf_n = xr_diag_rgf[idx_ib + 1, :, :lb_n, :lb_n]
            wg_diag_rgf_n = wg_diag_rgf[idx_ib + 1, :, :lb_n, :lb_n]
            wl_diag_rgf_n = wl_diag_rgf[idx_ib + 1, :, :lb_n, :lb_n]
            xr_diag_rgf_n_ct = xr_diag_rgf_n.conjugate().transpose((0,2,1))

            # xRM_next = M_E_k+1k * xR_E_k+1k+1
            xr_mr = xr_diag_rgf_n @ mr_lower_gpu[idx_ib, :, :lb_n, :lb_i]
            xr_mr_ct = xr_mr.conjugate().transpose((0,2,1))

            #xr_upper and xr_lower
            xr_upper[idx_ib, :,  :lb_i, :lb_n] = -xr_diag_c @ mr_upper_gpu[idx_ib, :, :lb_i, :lb_n] @ xr_diag_rgf_n
            xr_lower[idx_ib, :,  :lb_n, :lb_i] = -xr_diag_rgf_n @ mr_lower_gpu[idx_ib, :, :lb_n, :lb_i] @ xr_diag_c
            
            wr_diag_c = xr_lower[idx_ib - 1, :, :lb_i, :lb_n] @ cp.repeat(vh_upper_gpu[idx_ib - 1, cp.newaxis, :lb_p, :lb_i], energy_batchsize, axis=0) + xr_diag_gpu[idx_ib, :,  :lb_i, :lb_i] @ vh_diag_gpu[idx_ib, :, :lb_i, :lb_i] + \
                xr_upper[idx_ib, :lb_i, :lb_n] @ cp.repeat(vh_lower_gpu[idx_ib, cp.newaxis, :lb_n, :lb_i], energy_batchsize, axis=0)
            wr_diag_gpu[idx_ib, :, :lb_i, :lb_i] = wr_diag_c

            # WR_E_kk+1 = (V_k+1k.T - WR_E_kk*M_E_k+1k.T) * xR_E_k+1k+1.T
            # difference between matlab and python silvio todo
            # this line is wrong todo in the second part
            wr_upper_c = cp.repeat(vh_lower_gpu[idx_ib, cp.newaxis, :lb_n, :lb_i], energy_batchsize, axis = 0).transpose((0,2,1)) @ xr_diag_rgf_n.transpose((0,2,1)) - wr_diag_c @ xr_mr.transpose((0,2,1))
            wr_upper_gpu[idx_ib, :, :lb_i, :lb_n] = wr_upper_c

            # W^{\lessgtr}_E_kk+1 = XR_E_kk*(L^{\lessgtr}_E_kk+1*xR_E_k+1k+1.H - M_E_kk+1*w^{\lessgtr}_E_k+1k+1) - W^{\lessgtr}_E_kk*M_E_k+1k.H*xxR_E_k+1k+1.H
            wg_upper_c = xr_diag_c @ (lg_upper_gpu[idx_ib, :, :lb_i, :lb_n] @ xr_diag_rgf_n_ct - mr_upper_gpu[idx_ib, :, :lb_i, :lb_n] @ wg_diag_rgf_n) - wg_diag_c @ xr_mr_ct
            wg_upper_gpu[idx_ib, :lb_i, :lb_n] = wg_upper_c
            wl_upper_c = xr_diag_c @ (ll_upper_gpu[idx_ib, :, :lb_i, :lb_n] @ xr_diag_rgf_n_ct - mr_upper_gpu[idx_ib, :, :lb_i, :lb_n] @ wl_diag_rgf_n) - wl_diag_c @ xr_mr_ct
            wl_upper_gpu[idx_ib, :lb_i, :lb_n] = wl_upper_c
        else:
            #vh_r = vh_cp[slb_p, slb_c].toarray()
            #vh_c = vh_cp[slb_c, slb_c].toarray() - dvh_ed
            wr_diag_gpu[idx_ib, :, :lb_i, :lb_i] = xr_lower[idx_ib - 1, :, :lb_i, :lb_p] @ cp.repeat(vh_upper_gpu[idx_ib - 1, cp.newaxis, :lb_p, :lb_i], energy_batchsize, axis=0) + xr_diag_gpu[idx_ib, :, :lb_i, :lb_i] @ vh_diag_gpu[idx_ib, :, :lb_i, :lb_i]

    for idx_ib in range(0, nb_mm):
        # xr_diag[idx_ib, :, :] *= factor
        # wg_diag[idx_ib, :, :] *= factor
        # wl_diag[idx_ib, :, :] *= factor
        # wr_diag[idx_ib, :, :] *= factor
        dosw[: , idx_ib] = 1j * cp.trace(wr_diag_gpu[idx_ib,:, :, :] - wr_diag_gpu[idx_ib, :, :, :].conjugate().transpose((0,2,1)), axis1=1,axis2=2).get()
        nEw[:, idx_ib] = -1j * cp.trace(wl_diag_gpu[idx_ib, :, :], axis1=1,axis2=2).get()
        nPw[:, idx_ib] = 1j * cp.trace(wg_diag_gpu[idx_ib, :, :], axis1=1,axis2=2).get()
        # if idx_ib < nb_mm - 1:
        #     wr_upper[idx_ib, :, :] *= factor
        #     wg_upper[idx_ib, :, :] *= factor
        #     wl_upper[idx_ib, :, :] *= factor
    # wr_diag[:, :, :, :] = wr_diag_gpu.get()
    # wr_upper[:, :, :, :] = wr_upper_gpu.get()
    # wg_diag[:, :, :, :] = wg_diag_gpu.get()
    # wg_upper[:, :, :, :] = wg_upper_gpu.get()
    # wl_diag[:, :, :, :] = wl_diag_gpu.get()
    # wl_upper[:, :, :, :] = wl_upper_gpu.get()
    # xr_diag[:, :, :, :] = xr_diag_gpu.get()
    for i in range(len(wr_diag)):
        wr_diag_gpu[i].get(out = wr_diag[i])
        wg_diag_gpu[i].get(out = wg_diag[i])
        wl_diag_gpu[i].get(out = wl_diag[i])
        xr_diag_gpu[i].get(out = xr_diag[i])
        
    for i in range(len(wr_upper)):
        wr_upper_gpu[i].get(out = wr_upper[i])
        wg_upper_gpu[i].get(out = wg_upper[i])
        wl_upper_gpu[i].get(out = wl_upper[i])


def sp_mm_1_gpu(pr:sparse.csr_matrix,
          vh_gpu:cp.sparse.csr_matrix,
          vh_ct_gpu:cp.sparse.csr_matrix,
          mr,
          nao):
    """Matrix multiplication with sparse matrices """
    pr_gpu = cp.sparse.csr_matrix(pr)
    mr.append((cp.sparse.identity(nao) - vh_gpu @ pr_gpu).get())

def sp_mm_2_gpu(pg:sparse.csr_matrix,
          vh_gpu:cp.sparse.csr_matrix,
          vh_ct_gpu:cp.sparse.csr_matrix,
          lg,
          nao):
    """Matrix multiplication with sparse matrices """
    pg_gpu = cp.sparse.csr_matrix(pg)
    lg.append((vh_gpu @ pg_gpu @ vh_ct_gpu).get())

def sp_mm_3_gpu(pl:sparse.csr_matrix,
          vh_gpu:cp.sparse.csr_matrix,
          vh_ct_gpu:cp.sparse.csr_matrix,
          ll,
          nao):
    """Matrix multiplication with sparse matrices """
    pl_gpu = cp.sparse.csr_matrix(pl)
    ll.append((vh_gpu @ pl_gpu @ vh_ct_gpu).get())

def sp_mm_cpu(pr:sparse.csr_matrix,
              pg:sparse.csr_matrix,
                pl:sparse.csr_matrix,
          vh_cpu:sparse.csr_matrix,
          vh_ct_cpu:sparse.csr_matrix,
          nao):
    """Matrix multiplication with sparse matrices """
    mr = sparse.identity(nao) - vh_cpu @ pr
    lg = vh_cpu @ pg @ vh_ct_cpu
    ll = vh_cpu @ pl @ vh_ct_cpu
    return mr, lg, ll


