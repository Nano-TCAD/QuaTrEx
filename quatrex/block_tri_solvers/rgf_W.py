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
    Coulomb_matrix: sparse.csr_matrix,
    Polarization_greater: sparse.csr_matrix,
    Polarization_lesser: sparse.csr_matrix,
    Polarization_retarded: sparse.csr_matrix,
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
    factor: np.float64 = 1.0
):
    """Calculates the step from the polarization to the screened interaction.
    Beyn open boundary conditions are used by default.
    The outputs (w and x) are inputs which are changed inplace.
    See the start of this file for more informations.

    Args:
        Coulomb_matrix (sparse.csr_matrix): sparse matrix of size (#orbitals,#orbitals)
        Polarization_greater (sparse.csr_matrix): sparse matrix of size (#orbitals,#orbitals)
        Polarization_lesser (sparse.csr_matrix): sparse matrix of size (#orbitals,#orbitals)
        Polarization_retarded (sparse.csr_matrix): sparse matrix of size (#orbitals,#orbitals)
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


    # number of blocks
    nb = bmin.size
    # number of total orbitals (number of atomic orbitals)
    nao = Polarization_retarded.shape[0]

    bmax_mm = bmax[nbc - 1:nb:nbc]
    bmin_mm = bmin[0:nb:nbc]
    blocksize = bmax[0] - bmin[0] + 1
    blocksize_after_mm = bmax_mm[0] - bmin_mm[0] + 1
    number_of_blocks_after_mm = bmax_mm.size
    lb_vec_mm = bmax_mm - bmin_mm + 1


    # calculate M^{r}\left(E\right)
    M_retarded = sparse.identity(nao, format="csr") - Coulomb_matrix @ Polarization_retarded

    # calculate L^{\lessgtr}\left(E\right)
    L_greater = Coulomb_matrix @ Polarization_greater @ Coulomb_matrix.conjugate().transpose()
    L_lesser = Coulomb_matrix @ Polarization_lesser @ Coulomb_matrix.conjugate().transpose()

    # copy Coulomb_matrix to be able to change it inplace
    Coulomb_matrix_copy = Coulomb_matrix.copy()



    # from P^{\lessgtr}\left(E\right) / P^{r}\left(E\right)
    # (diagonal, upper, lower block of the start block at the left)
    (Polarization_greater_left_diag_block,
     Polarization_greater_left_upper_block,
     Polarization_greater_left_lower_block) = dL_OBC_eigenmode_cpu.stack_px(
                                                            Polarization_greater[:blocksize, :blocksize],
                                                            Polarization_greater[:blocksize, blocksize:2*blocksize],
                                                            nbc)
    (Polarization_lesser_left_diag_block,
     Polarization_lesser_left_upper_block,
     Polarization_lesser_left_lower_block) = dL_OBC_eigenmode_cpu.stack_px(
                                                            Polarization_lesser[:blocksize, :blocksize],
                                                            Polarization_lesser[:blocksize, blocksize:2*blocksize],
                                                            nbc)
    (Polarization_retarded_left_diag_block,
     Polarization_retarded_left_upper_block,
     Polarization_retarded_left_lower_block) = dL_OBC_eigenmode_cpu.stack_pr(
                                                            Polarization_retarded[:blocksize, :blocksize],
                                                            Polarization_retarded[:blocksize, blocksize:2*blocksize],
                                                            nbc)

    # (diagonal, upper, lower block of the end block at the right)
    (Polarization_greater_right_diag_block,
     Polarization_greater_right_upper_block,
     Polarization_greater_right_lower_block) = dL_OBC_eigenmode_cpu.stack_px(
                                                            Polarization_greater[-blocksize:, -blocksize:],
                                                            -Polarization_greater[-blocksize:, -2*blocksize:-blocksize].conjugate().transpose(),
                                                            nbc)
    (Polarization_lesser_right_diag_block,
     Polarization_lesser_right_upper_block,
     Polarization_lesser_right_lower_block) = dL_OBC_eigenmode_cpu.stack_px(
                                                            Polarization_lesser[-blocksize:, -blocksize:],
                                                            -Polarization_lesser[-blocksize:, -2*blocksize:-blocksize].conjugate().transpose(),
                                                            nbc)
    (Polarization_retarded_right_diag_block,
     Polarization_retarded_right_upper_block,
     Polarization_retarded_right_lower_block) = dL_OBC_eigenmode_cpu.stack_pr(
                                                            Polarization_retarded[-blocksize:, -blocksize:],
                                                            Polarization_retarded[-blocksize:, -2*blocksize:-blocksize].transpose(),
                                                            nbc)


    # from \hat(V)\left(E\right)
    # (diagonal, upper, lower block of the start block at the left)
    (Coulomb_matrix_left_diag_block,
     Coulomb_matrix_left_upper_block,
     Coulomb_matrix_left_lower_block) = dL_OBC_eigenmode_cpu.stack_vh(
                                                            Coulomb_matrix[:blocksize, :blocksize],
                                                            Coulomb_matrix[:blocksize, blocksize:2*blocksize],
                                                            nbc)

    # (diagonal, upper, lower block of the end block at the right)
    (Coulomb_matrix_right_diag_block,
     Coulomb_matrix_right_upper_block,
     Coulomb_matrix_right_lower_block) = dL_OBC_eigenmode_cpu.stack_vh(
                                                            Coulomb_matrix[-blocksize:, -blocksize:],
                                                            Coulomb_matrix[-blocksize:, -2*blocksize:-blocksize].conjugate().transpose(),
                                                            nbc)

    # from L^{\lessgtr}\left(E\right)
    # (diagonal, upper, lower block of the start block at the left)
    (L_greater_left_diag_block,
     L_greater_left_upper_block, _) = dL_OBC_eigenmode_cpu.stack_lx(
                                                            Coulomb_matrix_left_diag_block,
                                                            Coulomb_matrix_left_upper_block,
                                                            Coulomb_matrix_left_lower_block,
                                                            Polarization_greater_left_diag_block,
                                                            Polarization_greater_left_upper_block,
                                                            Polarization_greater_left_lower_block)
    (L_leser_left_diag_block,
     L_leser_left_upper_block, _) = dL_OBC_eigenmode_cpu.stack_lx(
                                                            Coulomb_matrix_left_diag_block,
                                                            Coulomb_matrix_left_upper_block,
                                                            Coulomb_matrix_left_lower_block,
                                                            Polarization_lesser_left_diag_block,
                                                            Polarization_lesser_left_upper_block,
                                                            Polarization_lesser_left_lower_block)


    # (diagonal, upper, lower block of the end block at the right)
    (L_greater_right_diag_block,_,
     L_greater_right_lower_block) = dL_OBC_eigenmode_cpu.stack_lx(
                                                            Coulomb_matrix_right_diag_block,
                                                            Coulomb_matrix_right_upper_block,
                                                            Coulomb_matrix_right_lower_block,
                                                            Polarization_greater_right_diag_block,
                                                            Polarization_greater_right_upper_block,
                                                            Polarization_greater_right_lower_block)
    (L_leser_right_diag_block, _,
     L_leser_right_lower_block) = dL_OBC_eigenmode_cpu.stack_lx(
                                                            Coulomb_matrix_right_diag_block,
                                                            Coulomb_matrix_right_upper_block,
                                                            Coulomb_matrix_right_lower_block,
                                                            Polarization_lesser_right_diag_block,
                                                            Polarization_lesser_right_upper_block,
                                                            Polarization_lesser_right_lower_block)

    # from M^{r}\left(E\right)
    # (diagonal, upper, lower block of the start block at the left)
    (M_retarded_left_diag_block,
     M_retarded_left_upper_block,
     M_retarded_left_lower_block) = dL_OBC_eigenmode_cpu.stack_mr(
                                                            Coulomb_matrix_left_diag_block,
                                                            Coulomb_matrix_left_upper_block,
                                                            Coulomb_matrix_left_lower_block,
                                                            Polarization_retarded_left_diag_block,
                                                            Polarization_retarded_left_upper_block,
                                                            Polarization_retarded_left_lower_block)
    # (diagonal, upper, lower block of the end block at the right)
    (M_retarded_right_diag_block,
     M_retarded_right_upper_block,
     M_retarded_right_lower_block) = dL_OBC_eigenmode_cpu.stack_mr(
                                                            Coulomb_matrix_right_diag_block,
                                                            Coulomb_matrix_right_upper_block,
                                                            Coulomb_matrix_right_lower_block,
                                                            Polarization_retarded_right_diag_block,
                                                            Polarization_retarded_right_upper_block,
                                                            Polarization_retarded_right_lower_block)


    # correct first and last block to account for the contacts in multiplication
    M_retarded_left_BC_block = -Coulomb_matrix_left_lower_block @ Polarization_retarded_left_upper_block
    M_retarded_right_BC_block = -Coulomb_matrix_right_upper_block @ Polarization_retarded_right_lower_block

    # L^{\lessgtr}_E_00 = V_10*P^{\lessgtr}_E_00*V_01 + V_10*P^{\lessgtr}_E_01*V_00 + V_00*P^{\lessgtr}_E_10*V_01
    L_greater_left_BC_block = (Coulomb_matrix_left_lower_block @ Polarization_greater_left_diag_block @ Coulomb_matrix_left_upper_block +
                               Coulomb_matrix_left_lower_block @ Polarization_greater_left_upper_block @ Coulomb_matrix_left_diag_block +
                               Coulomb_matrix_left_diag_block @ Polarization_greater_left_lower_block @ Coulomb_matrix_left_upper_block).toarray()

    L_lesser_left_BC_block = (Coulomb_matrix_left_lower_block @ Polarization_lesser_left_diag_block @ Coulomb_matrix_left_upper_block +
                              Coulomb_matrix_left_lower_block @ Polarization_lesser_left_upper_block @ Coulomb_matrix_left_diag_block +
                              Coulomb_matrix_left_diag_block @ Polarization_lesser_left_lower_block @ Coulomb_matrix_left_upper_block).toarray()

    # L^{\lessgtr}_E_nn = V_nn*Polarization_lesser_E_n-1n*V_nn-1 + V_n-1n*Polarization_lesser_E_nn-1*V_nn + V_n-1n*Polarization_lesser_E_nn*V_nn-1
    L_greater_right_BC_block = (Coulomb_matrix_right_diag_block @ Polarization_greater_right_upper_block @ Coulomb_matrix_right_lower_block +
                                Coulomb_matrix_right_upper_block @ Polarization_greater_right_lower_block @ Coulomb_matrix_right_diag_block +
                                Coulomb_matrix_right_upper_block @ Polarization_greater_right_diag_block @ Coulomb_matrix_right_lower_block).toarray()

    L_lesser_right_BC_block = (Coulomb_matrix_right_diag_block @ Polarization_lesser_right_upper_block @ Coulomb_matrix_right_lower_block +
                               Coulomb_matrix_right_upper_block @ Polarization_lesser_right_lower_block @ Coulomb_matrix_right_diag_block +
                               Coulomb_matrix_right_upper_block @ Polarization_lesser_right_diag_block @ Coulomb_matrix_right_lower_block).toarray()


    # correction for the matrix inverse calculations----------------------------

    # conditions about convergence or meaningful results from
    # boundary correction calculations
    cond_l = 0.0
    cond_r = 0.0

    # correction for first block
    _, cond_l, Chi_left_BC_block, M_retarded_BC_block, _ = beyn_cpu.beyn(
                                                            M_retarded_left_diag_block.toarray(),
                                                            M_retarded_left_upper_block.toarray(),
                                                            M_retarded_left_lower_block.toarray(),
                                                            imag_lim, rr, "L", block=False)

    if not np.isnan(cond_l):
        M_retarded_left_BC_block -= M_retarded_BC_block
        Coulomb_matrix_left_BC_block = M_retarded_left_lower_block @ Chi_left_BC_block @ Coulomb_matrix_left_upper_block

    # beyn failed
    if np.isnan(cond_l):
        (Chi_left_BC_block, M_retarded_BC_block, Coulomb_matrix_left_BC_block, cond_l) = sancho.open_boundary_conditions(
                                                            M_retarded_left_diag_block.toarray(),
                                                            M_retarded_left_lower_block.toarray(),
                                                            M_retarded_left_upper_block.toarray(),
                                                            Coulomb_matrix_left_upper_block.toarray()
                                                            )
        M_retarded_left_BC_block -= M_retarded_BC_block

    # correction for last block
    _, cond_r, Chi_right_BC_block, M_retarded_BC_block, _ = beyn_cpu.beyn(
                                                            M_retarded_right_diag_block.toarray(),
                                                            M_retarded_right_upper_block.toarray(),
                                                            M_retarded_right_lower_block.toarray(),
                                                            imag_lim, rr, "R", block=False)
    if not np.isnan(cond_r):
        M_retarded_right_BC_block -= M_retarded_BC_block
        Coulomb_matrix_right_BC_block = M_retarded_right_upper_block @ Chi_right_BC_block @ Coulomb_matrix_right_lower_block

    # beyn failed
    if np.isnan(cond_r):
        Chi_right_BC_block, M_retarded_BC_block, Coulomb_matrix_right_BC_block, cond_r = sancho.open_boundary_conditions(
                                                            M_retarded_right_diag_block.toarray(),
                                                            M_retarded_right_upper_block.toarray(),
                                                            M_retarded_right_lower_block.toarray(),
                                                            Coulomb_matrix_right_lower_block.toarray()
                                                            )
        M_retarded_right_BC_block -= M_retarded_BC_block


    # boundary conditions could be calculated
    if not np.isnan(cond_l) and not np.isnan(cond_l):
        L_greater_left_OBC_block, L_lesser_left_OBC_block = dL_OBC_eigenmode_cpu.get_dl_obc_alt(
                                                            Chi_left_BC_block,
                                                            L_greater_left_diag_block.toarray(),
                                                            L_greater_left_upper_block.toarray(),
                                                            L_leser_left_diag_block.toarray(),
                                                            L_leser_left_upper_block.toarray(),
                                                            M_retarded_left_lower_block.toarray(),
                                                            blk="L")

        if np.isnan(L_lesser_left_OBC_block).any():
            cond_l = np.nan
        else:
            L_greater_left_BC_block += L_greater_left_OBC_block
            L_lesser_left_BC_block += L_lesser_left_OBC_block

        L_greater_right_OBC_block, L_lesser_right_OBC_block = dL_OBC_eigenmode_cpu.get_dl_obc_alt(
                                                Chi_right_BC_block,
                                                L_greater_right_diag_block.toarray(),
                                                L_greater_right_lower_block.toarray(),
                                                L_leser_right_diag_block.toarray(),
                                                L_leser_right_lower_block.toarray(),
                                                M_retarded_right_upper_block.toarray(),
                                                blk="R")

        if np.isnan(L_lesser_right_OBC_block).any():
            cond_r = np.nan
        else:
            L_greater_right_BC_block += L_greater_right_OBC_block
            L_lesser_right_BC_block += L_lesser_right_OBC_block


    # start of rgf_W------------------------------------------------------------

    # check if OBC did not fail
    if not np.isnan(cond_r) and not np.isnan(cond_l) and ie:

        # add OBC corrections to the start and end block
        M_retarded[-blocksize_after_mm:, -blocksize_after_mm:] += M_retarded_right_BC_block
        M_retarded[:blocksize_after_mm, :blocksize_after_mm] += M_retarded_left_BC_block

        L_greater[:blocksize_after_mm, :blocksize_after_mm] += L_greater_left_BC_block
        L_greater[-blocksize_after_mm:, -blocksize_after_mm:] += L_greater_right_BC_block
        
        L_lesser[-blocksize_after_mm:, -blocksize_after_mm:] += L_lesser_right_BC_block
        L_lesser[:blocksize_after_mm, :blocksize_after_mm] += L_lesser_left_BC_block
        
        Coulomb_matrix_copy[-blocksize_after_mm:, -blocksize_after_mm:] -= Coulomb_matrix_right_BC_block
        Coulomb_matrix_copy[:blocksize_after_mm, :blocksize_after_mm] -= Coulomb_matrix_left_BC_block
        

        # System_matrix_invert = np.linalg.inv(M_retarded.toarray())
        # Screened_interaction_retarded = System_matrix_invert @ Coulomb_matrix_copy
        # Screened_interaction_lesser = System_matrix_invert @ L_lesser @ System_matrix_invert.conjugate().transpose()
        # Screened_interaction_greater = System_matrix_invert @ L_greater @ Screened_interaction_retarded.conjugate().transpose()

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
        xr_lb = np.linalg.inv(M_retarded[-blocksize_after_mm:, -blocksize_after_mm:].toarray())
        xr_lb_ct = xr_lb.conjugate().transpose()
        xr_diag_rgf[number_of_blocks_after_mm - 1] = xr_lb

        # w^{>}_E_nn = x^{r}_E_nn * L^{>}_E_nn * (x^{r}_E_nn).H
        wg_lb = xr_lb @ L_greater[-blocksize_after_mm:, -blocksize_after_mm:] @ xr_lb_ct
        wg_diag_rgf[number_of_blocks_after_mm - 1] = wg_lb

        # w^{<}_E_nn = x^{r}_E_nn * L^{<}_E_nn * (x^{r}_E_nn).H
        wl_lb = xr_lb @ L_lesser[-blocksize_after_mm:, -blocksize_after_mm:] @ xr_lb_ct
        wl_diag_rgf[number_of_blocks_after_mm - 1] = wl_lb

        # wR_E_nn = xR_E_nn * V_nn
        wr_lb = xr_lb @ Coulomb_matrix_copy[-blocksize_after_mm:, -blocksize_after_mm:]
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
            M_retarded_c = M_retarded[slb_c, slb_c].toarray()
            M_retarded_r = M_retarded[slb_c, slb_p].toarray()
            M_retarded_d = M_retarded[slb_p, slb_c].toarray()
            Coulomb_matrix_c = Coulomb_matrix_copy[slb_c, slb_c].toarray()
            Coulomb_matrix_d = Coulomb_matrix_copy[slb_p, slb_c].toarray()
            L_greater_c = L_greater[slb_c, slb_c].toarray()
            L_greater_d = L_greater[slb_p, slb_c].toarray()
            L_leser_c = L_lesser[slb_c, slb_c].toarray()
            L_leser_d = L_lesser[slb_p, slb_c].toarray()


            # MxR = M_E_kk+1 * xR_E_k+1k+1
            M_retarded_xr = M_retarded_r @ xr_p

            # xR_E_kk = (M_E_kk - M_E_kk+1*xR_E_k+1k+1*M_E_k+1k)^{-1}
            xr_c = np.linalg.inv(M_retarded_c - M_retarded_xr @ M_retarded_d)
            xr_diag_rgf[idx_ib, :lb_i, :lb_i] = xr_c

            # conjugate and transpose
            M_retarded_r_ct = M_retarded_r.conjugate().transpose()
            xr_c_ct = xr_c.conjugate().transpose()

            # A^{\lessgtr} = M_E_kk+1 * xR_E_k+1k+1 * L^{\lessgtr}_E_k+1k
            ag = M_retarded_xr @ L_greater_d
            al = M_retarded_xr @ L_leser_d
            ag_diff = ag - ag.conjugate().transpose()
            al_diff = al - al.conjugate().transpose()

            # w^{\lessgtr}_E_kk = xR_E_kk * (L^{\lessgtr}_E_kk + M_E_kk+1*w^{\lessgtr}_E_k+1k+1*M_E_kk+1.H - (A^{\lessgtr} - A^{\lessgtr}.H)) * xR_E_kk.H
            wg_c = xr_c @ (L_greater_c + M_retarded_r @ wg_p @ M_retarded_r_ct - ag_diff) @ xr_c_ct
            wg_diag_rgf[idx_ib, :lb_i, :lb_i] = wg_c

            wl_c = xr_c @ (L_leser_c + M_retarded_r @ wl_p @ M_retarded_r_ct - al_diff) @ xr_c_ct
            wl_diag_rgf[idx_ib, :lb_i, :lb_i] = wl_c

            # wR_E_kk = xR_E_kk * (V_kk - M_E_kk+1 * xR_E_k+1k+1 * V_k+1k)
            wr_c = xr_c @ (Coulomb_matrix_c - M_retarded_xr @ Coulomb_matrix_d)
            wr_diag_rgf[idx_ib, :lb_i, :lb_i] = wr_c

        # block length 0
        lb_f = lb_vec_mm[0]
        lb_p = lb_vec_mm[1]

        # slice of current and previous block
        slb_c = slice(bmin_mm[0], bmax_mm[0] + 1)
        slb_p = slice(bmin_mm[1], bmax_mm[1] + 1)

        # WARNING the last read blocks from the above for loop are used

        # second step of iteration
        Coulomb_matrix_r = Coulomb_matrix_copy[slb_c, slb_p].toarray()
        L_greater_r = L_greater[slb_c, slb_p].toarray()
        L_leser_r = L_lesser[slb_c, slb_p].toarray()

        xr_mr = xr_p @ M_retarded_d
        xr_M_retarded_ct = xr_mr.conjugate().transpose()

        # WR_E_00 = wR_E_00
        wr_diag[0, :lb_f, :lb_f] = wr_diag_rgf[0, :lb_f, :lb_f]

        # WR_E_01 = (V_01 - WR_E_00*M_E_10) * xR_E_11
        # todo if Coulomb_matrix_r can be used instead of Coulomb_matrix_copy[slb_c,slb_p]
        wr_upper[0, :lb_f, :lb_p] = (Coulomb_matrix_r - wr_diag[0, :lb_f, :lb_f] @ M_retarded_d.transpose()) @ xr_p.transpose()

        # XR_E_00 = xR_E_00
        xr_diag[0, :lb_f, :lb_f] = xr_diag_rgf[0, :lb_f, :lb_f]

        # W^{\lessgtr}_E_00 = w^{\lessgtr}_E_00
        wg_diag[0, :lb_f, :lb_f] = wg_diag_rgf[0, :lb_f, :lb_f]
        wl_diag[0, :lb_f, :lb_f] = wl_diag_rgf[0, :lb_f, :lb_f]

        # W^{\lessgtr}_E_01 = xR_E_00*L^{\lessgtr}_01*_xR_E_11.H - xR_E_00*M_E_01*wL_E_11 - w^{\lessgtr}_E_00*M_E_10.H*xR_E_11.H
        xr_p_ct = xr_p.conjugate().transpose()
        wg_upper[0, :lb_f, :lb_p] = xr_c @ L_greater_r @ xr_p_ct - xr_c @ M_retarded_r @ wg_p - wg_c @ xr_M_retarded_ct
        wl_upper[0, :lb_f, :lb_p] = xr_c @ L_leser_r @ xr_p_ct - xr_c @ M_retarded_r @ wl_p - wl_c @ xr_M_retarded_ct

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
            M_retarded_l = M_retarded[slb_c, slb_p].toarray()
            M_retarded_u = M_retarded[slb_p, slb_c].toarray()
            L_greater_l = L_greater[slb_c, slb_p].toarray()
            L_leser_l = L_lesser[slb_c, slb_p].toarray()

            # xRM = xR_E_kk * M_E_kk-1
            xr_M_retarded_xr = xr_mr @ xr_diag_p
            xr_M_retarded_xr_ct = xr_M_retarded_xr.conjugate().transpose()
            xr_M_retarded_xr_mr = xr_M_retarded_xr @ M_retarded_u

            # WR_E_kk = wR_E_kk - xR_E_kk*M_E_kk-1*WR_E_k-1k
            wr_diag_rgf_c = wr_diag_rgf[idx_ib, :lb_i, :lb_i]
            # todo if wr_upper_p can be used wr_upper[idx_ib-1,:lb_vec_mm[idx_ib-1],:lb_i]
            wr_diag_c = wr_diag_rgf_c - xr_mr @ wr_upper[idx_ib - 1, :lb_vec_mm[idx_ib - 1], :lb_i]
            wr_diag[idx_ib, :lb_i, :lb_i] = wr_diag_c

            # XR_E_kk = xR_E_kk + (xR_E_kk * M_E_kk-1 * XR_E_k-1k-1 * M_E_k-1k * xR_E_kk)
            xr_diag_c = xr_diag_rgf_c + xr_M_retarded_xr_mr @ xr_diag_rgf_c
            xr_diag[idx_ib, :lb_i, :lb_i] = xr_diag_c

            # A^{\lessgtr} = xR_E_kk * L^{\lessgtr}_E_kk-1 * XR_E_k-1k-1.H * (xR_E_kk * M_E_kk-1).H
            ag = xr_diag_rgf_c @ L_greater_l @ xr_M_retarded_xr_ct
            al = xr_diag_rgf_c @ L_leser_l @ xr_M_retarded_xr_ct
            ag_diff = ag - ag.conjugate().transpose()
            al_diff = al - al.conjugate().transpose()

            # B^{\lessgtr} = xR_E_kk * M_E_kk-1 * XR_E_k-1k-1 * M_E_k-1k * w^{\lessgtr}_E_kk
            bg = xr_M_retarded_xr_mr @ wg_diag_rgf_c
            bl = xr_M_retarded_xr_mr @ wl_diag_rgf_c
            bg_diff = bg - bg.conjugate().transpose()
            bl_diff = bl - bl.conjugate().transpose()

            # W^{\lessgtr}_E_kk = w^{\lessgtr}_E_kk + xR_E_kk*M_E_kk-1*W^{\lessgtr}_E_k-1k-1*(xR_E_kk*M_E_kk-1).H - (A^{\lessgtr}-A^{\lessgtr}.H) + (B^{\lessgtr}-B^{\lessgtr}.H)
            wg_diag_c = wg_diag_rgf_c + xr_mr @ wg_diag_p @ xr_M_retarded_ct - ag_diff + bg_diff
            wl_diag_c = wl_diag_rgf_c + xr_mr @ wl_diag_p @ xr_M_retarded_ct - al_diff + bl_diff
            wg_diag[idx_ib, :lb_i, :lb_i] = wg_diag_c
            wl_diag[idx_ib, :lb_i, :lb_i] = wl_diag_c

            # following code block has problems
            if idx_ib < number_of_blocks_after_mm - 1:
                # block length i
                lb_n = lb_vec_mm[idx_ib + 1]
                # slice of current and previous block
                slb_n = slice(bmin_mm[idx_ib + 1], bmax_mm[idx_ib + 1] + 1)

                # read out blocks needed
                Coulomb_matrix_d = Coulomb_matrix_copy[slb_n, slb_c].toarray()
                M_retarded_d = M_retarded[slb_n, slb_c].toarray()
                M_retarded_r = M_retarded[slb_c, slb_n].toarray()
                L_greater_r = L_greater[slb_c, slb_n].toarray()
                L_leser_r = L_lesser[slb_c, slb_n].toarray()

                xr_diag_rgf_n = xr_diag_rgf[idx_ib + 1, :lb_n, :lb_n]
                wg_diag_rgf_n = wg_diag_rgf[idx_ib + 1, :lb_n, :lb_n]
                wl_diag_rgf_n = wl_diag_rgf[idx_ib + 1, :lb_n, :lb_n]
                xr_diag_rgf_n_ct = xr_diag_rgf_n.conjugate().transpose()

                # xRM_next = M_E_k+1k * xR_E_k+1k+1
                xr_mr = xr_diag_rgf_n @ M_retarded_d
                xr_M_retarded_ct = xr_mr.conjugate().transpose()

                # WR_E_kk+1 = (V_k+1k.T - WR_E_kk*M_E_k+1k.T) * xR_E_k+1k+1.T
                # difference between matlab and python silvio todo
                # this line is wrong todo in the second part
                wr_upper_c = Coulomb_matrix_d.transpose() @ xr_diag_rgf_n.transpose() - wr_diag_c @ xr_mr.transpose()
                wr_upper[idx_ib, :lb_i, :lb_n] = wr_upper_c

                # W^{\lessgtr}_E_kk+1 = XR_E_kk*(L^{\lessgtr}_E_kk+1*xR_E_k+1k+1.H - M_E_kk+1*w^{\lessgtr}_E_k+1k+1) - W^{\lessgtr}_E_kk*M_E_k+1k.H*xxR_E_k+1k+1.H
                wg_upper_c = xr_diag_c @ (L_greater_r @ xr_diag_rgf_n_ct - M_retarded_r @ wg_diag_rgf_n) - wg_diag_c @ xr_M_retarded_ct
                wg_upper[idx_ib, :lb_i, :lb_n] = wg_upper_c
                wl_upper_c = xr_diag_c @ (L_leser_r @ xr_diag_rgf_n_ct - M_retarded_r @ wl_diag_rgf_n) - wl_diag_c @ xr_M_retarded_ct
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

