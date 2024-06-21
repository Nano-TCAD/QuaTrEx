# Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.

import numpy as np
from scipy.sparse import csc_matrix, diags
from scipy.linalg import eig


def calc_bandstructure(S, H, SigmaR_GW, SigmaR_PHN, indE, Bmin, Bmax, side):

    #SigR = SigR + np.diag(SigmaR_PHN[indE, :])
    H = H + SigmaR_GW[indE] + SigmaR_PHN[indE]

    if side == 'left':
        LBsize = Bmax[0] - Bmin[0] + 1
        H00 = H[:LBsize, :LBsize].toarray()
        H01 = H[:LBsize, LBsize:2 * LBsize].toarray()
        H10 = H[LBsize:2 * LBsize, :LBsize].toarray()

        S00 = S[:LBsize, :LBsize].toarray()
        S01 = S[:LBsize, LBsize:2 * LBsize].toarray()
        S10 = S[LBsize:2 * LBsize, :LBsize].toarray()

    elif side == 'right':
        RBsize = Bmax[-1] - Bmin[-1] + 1
        H00 = H[-RBsize:, -RBsize:].toarray()
        H01 = H[-RBsize:, -2 * RBsize:-RBsize].toarray()
        H10 = H[-2 * RBsize:-RBsize, -RBsize:].toarray()

        S00 = S[-RBsize:, -RBsize:].toarray()
        S01 = S[-RBsize:, -2 * RBsize:-RBsize].toarray()
        S10 = S[-2 * RBsize:-RBsize, -RBsize:].toarray()

    Ek = np.sort(np.real(eig(H00 + H01 + H10, b=S00 + S01 + S10, right=False)))

    return Ek

def calc_bandstructure_interpol(E, S, H, E_target, SigmaR_GW, SigmaL_GW, SigmaG_GW, SigmaR_PHN, indE, Bmin, Bmax, side):

    #SigR = SigR + np.diag(SigmaR_PHN[indE, :])
    #H = H + SigmaR_GW[indE] + SigmaR_PHN[indE]

    E1 = E[indE]
    E2 = E[indE + 1]

    SigL1 = 1j * np.imag(SigmaL_GW[indE])
    SigL2 = 1j * np.imag(SigmaL_GW[indE + 1])
    SigL = SigL1 + (SigL2 - SigL1) / (E2 - E1) * (E_target - E1)
    SigL = (SigL - SigL.conj().T) / 2

    SigG1 = 1j * np.imag(SigmaG_GW[indE])
    SigG2 = 1j * np.imag(SigmaG_GW[indE + 1])
    SigG = SigG1 + (SigG2 - SigG1) / (E2 - E1) * (E_target - E1)
    SigG = (SigG - SigG.conj().T) / 2

    SigR1 = SigmaR_GW[indE]
    SigR2 = SigmaR_GW[indE + 1]
    SigR = SigR1 + (SigR2 - SigR1) / (E2 - E1) * (E_target - E1)
    SigR = np.real(SigR) + 1j * np.imag(SigG - SigL) / 2
    SigR = (SigR + SigR.T) / 2
    
    SigR_PHN1 = diags((SigmaR_PHN[indE].diagonal()))
    SigR_PHN2 = diags((SigmaR_PHN[indE + 1].diagonal()))
    SigR_PHN = SigR_PHN1 + (SigR_PHN2 - SigR_PHN1) / (E2 - E1) * (E_target - E1)

    SigR = SigR + SigR_PHN
    H = H + SigR

    if side == 'left':
        LBsize = Bmax[0] - Bmin[0] + 1
        H00 = H[:LBsize, :LBsize].toarray()
        H01 = H[:LBsize, LBsize:2 * LBsize].toarray()
        H10 = H[LBsize:2 * LBsize, :LBsize].toarray()

        S00 = S[:LBsize, :LBsize].toarray()
        S01 = S[:LBsize, LBsize:2 * LBsize].toarray()
        S10 = S[LBsize:2 * LBsize, :LBsize].toarray()

    elif side == 'right':
        RBsize = Bmax[-1] - Bmin[-1] + 1
        H00 = H[-RBsize:, -RBsize:].toarray()
        H01 = H[-RBsize:, -2 * RBsize:-RBsize].toarray()
        H10 = H[-2 * RBsize:-RBsize, -RBsize:].toarray()

        S00 = S[-RBsize:, -RBsize:].toarray()
        S01 = S[-RBsize:, -2 * RBsize:-RBsize].toarray()
        S10 = S[-2 * RBsize:-RBsize, -RBsize:].toarray()

    Ek = np.sort(np.real(eig(H00 + H01 + H10, b=S00 + S01 + S10, right=False)))

    return Ek

def calc_bandstructure_mpi(S, H, SigmaR_GW, SigmaR_PHN, Bmin, Bmax, side):

    #SigR = SigR + np.diag(SigmaR_PHN[indE, :])
    H = H + SigmaR_GW + SigmaR_PHN

    if side == 'left':
        LBsize = Bmax[0] - Bmin[0] + 1
        H00 = H[:LBsize, :LBsize].toarray()
        H01 = H[:LBsize, LBsize:2 * LBsize].toarray()
        H10 = H[LBsize:2 * LBsize, :LBsize].toarray()

        S00 = S[:LBsize, :LBsize].toarray()
        S01 = S[:LBsize, LBsize:2 * LBsize].toarray()
        S10 = S[LBsize:2 * LBsize, :LBsize].toarray()

    elif side == 'right':
        RBsize = Bmax[-1] - Bmin[-1] + 1
        H00 = H[-RBsize:, -RBsize:].toarray()
        H01 = H[-RBsize:, -2 * RBsize:-RBsize].toarray()
        H10 = H[-2 * RBsize:-RBsize, -RBsize:].toarray()

        S00 = S[-RBsize:, -RBsize:].toarray()
        S01 = S[-RBsize:, -2 * RBsize:-RBsize].toarray()
        S10 = S[-2 * RBsize:-RBsize, -RBsize:].toarray()

    Ek = np.sort(np.real(eig(H00 + H01 + H10, b=S00 + S01 + S10, right=False)))

    return Ek


def calc_bandstructure_mpi_interpol(E, S, H, E_target, SigmaR_GW_vec, SigmaL_GW_vec, SigmaG_GW_vec, SigmaR_PHN_vec, indE, Bmin, Bmax, side):

    #SigR = SigR + np.diag(SigmaR_PHN[indE, :])
    #H = H + SigmaR_GW + SigmaR_PHN

    E1 = E[indE]
    E2 = E[indE + 1]

    # SigL1 = 1j * np.imag(SigmaL_GW_vec[0])
    # SigL2 = 1j * np.imag(SigmaL_GW_vec[1])
    SigL1 = SigmaL_GW_vec[0]
    SigL2 = SigmaL_GW_vec[1]
    SigL = SigL1 + (SigL2 - SigL1) / (E2 - E1) * (E_target - E1)
    SigL = (SigL - SigL.conj().T) / 2

    # SigG1 = 1j * np.imag(SigmaG_GW_vec[0])
    # SigG2 = 1j * np.imag(SigmaG_GW_vec[1])
    SigG1 = SigmaG_GW_vec[0]
    SigG2 = SigmaG_GW_vec[1]
    SigG = SigG1 + (SigG2 - SigG1) / (E2 - E1) * (E_target - E1)
    SigG = (SigG - SigG.conj().T) / 2

    SigR1 = SigmaR_GW_vec[0]
    SigR2 = SigmaR_GW_vec[1]
    SigR = SigR1 + (SigR2 - SigR1) / (E2 - E1) * (E_target - E1)
    #SigR = np.real(SigR) + 1j * np.imag(SigG - SigL) / 2
    SigR = np.real(SigR) + (SigG - SigL) / 2
    #Symmetrize here to guarantee real eigenvalues
    SigR = (SigR + SigR.T) / 2
    
    SigR_PHN1 = diags((SigmaR_PHN_vec[0].diagonal()))
    SigR_PHN2 = diags((SigmaR_PHN_vec[1].diagonal()))
    SigR_PHN = SigR_PHN1 + (SigR_PHN2 - SigR_PHN1) / (E2 - E1) * (E_target - E1)

    SigR = SigR + SigR_PHN
    H = H + SigR

    if side == 'left':
        LBsize = Bmax[0] - Bmin[0] + 1
        H00 = H[:LBsize, :LBsize].toarray()
        H01 = H[:LBsize, LBsize:2 * LBsize].toarray()
        H10 = H[LBsize:2 * LBsize, :LBsize].toarray()

        S00 = S[:LBsize, :LBsize].toarray()
        S01 = S[:LBsize, LBsize:2 * LBsize].toarray()
        S10 = S[LBsize:2 * LBsize, :LBsize].toarray()

    elif side == 'right':
        RBsize = Bmax[-1] - Bmin[-1] + 1
        H00 = H[-RBsize:, -RBsize:].toarray()
        H01 = H[-RBsize:, -2 * RBsize:-RBsize].toarray()
        H10 = H[-2 * RBsize:-RBsize, -RBsize:].toarray()

        S00 = S[-RBsize:, -RBsize:].toarray()
        S01 = S[-RBsize:, -2 * RBsize:-RBsize].toarray()
        S10 = S[-2 * RBsize:-RBsize, -RBsize:].toarray()

    Ek = np.sort(np.real(eig(H00 + H01 + H10, b=S00 + S01 + S10, right=False)))

    return Ek


def calc_bandstructure_mpi_interpol_2(E, S, H, E_target, SigmaR_GW_vec, indE, Bmin, Bmax, side, mapping_diag, mapping_upper, mapping_lower, block_idx = 0):

    #SigR = SigR + np.diag(SigmaR_PHN[indE, :])
    #H = H + SigmaR_GW + SigmaR_PHN

    E1 = E[indE]
    E2 = E[indE + 1]

    # SigL1 = 1j * np.imag(SigmaL_GW_vec[0])
    # SigL2 = 1j * np.imag(SigmaL_GW_vec[1])
    # SigL1 = SigmaL_GW_vec[0]
    # SigL2 = SigmaL_GW_vec[1]
    # SigL = SigL1 + (SigL2 - SigL1) / (E2 - E1) * (E_target - E1)
    # SigL = (SigL - SigL.conj().T) / 2

    # SigG1 = 1j * np.imag(SigmaG_GW_vec[0])
    # SigG2 = 1j * np.imag(SigmaG_GW_vec[1])
    # SigG1 = SigmaG_GW_vec[0]
    # SigG2 = SigmaG_GW_vec[1]
    # SigG = SigG1 + (SigG2 - SigG1) / (E2 - E1) * (E_target - E1)
    # SigG = (SigG - SigG.conj().T) / 2

    SigR1 = SigmaR_GW_vec[0]
    SigR2 = SigmaR_GW_vec[1]
    SigR = SigR1 + (SigR2 - SigR1) / (E2 - E1) * (E_target - E1)
    #SigR = np.real(SigR) + 1j * np.imag(SigG - SigL) / 2
    # SigR = np.real(SigR) + (SigG - SigL) / 2
    #Symmetrize here to guarantee real eigenvalues
    # SigR = (SigR + SigR.T) / 2
    
    # SigR_PHN1 = diags((SigmaR_PHN_vec[0].diagonal()))
    # SigR_PHN2 = diags((SigmaR_PHN_vec[1].diagonal()))
    # SigR_PHN = SigR_PHN1 + (SigR_PHN2 - SigR_PHN1) / (E2 - E1) * (E_target - E1)

    # SigR = SigR + SigR_PHN

    def _get_dense_block(compressed, mapping, block_idx, block_size):
        block_data_indices = mapping[block_idx][0]
        block_rows = mapping[block_idx][1]
        block_cols = mapping[block_idx][2]
        uncompressed = np.zeros((block_size, block_size), dtype=compressed.dtype)
        uncompressed[block_rows, block_cols] = compressed[block_data_indices]
        return uncompressed
    
    nb = len(mapping_diag)

    # H = H + SigR

    if side == 'left':
        LBsize = Bmax[0] - Bmin[0] + 1
        H00 = H[:LBsize, :LBsize].toarray()
        H01 = H[:LBsize, LBsize:2 * LBsize].toarray()
        H10 = H[LBsize:2 * LBsize, :LBsize].toarray()

        H00 += _get_dense_block(SigR, mapping_diag, 0, LBsize)
        H01 += _get_dense_block(SigR, mapping_upper, 0, LBsize)
        H10 += _get_dense_block(SigR, mapping_lower, 0, LBsize)

        S00 = S[:LBsize, :LBsize].toarray()
        S01 = S[:LBsize, LBsize:2 * LBsize].toarray()
        S10 = S[LBsize:2 * LBsize, :LBsize].toarray()

    elif side == 'right':
        RBsize = Bmax[-1] - Bmin[-1] + 1
        H00 = H[-RBsize:, -RBsize:].toarray()
        H01 = H[-RBsize:, -2 * RBsize:-RBsize].toarray()
        H10 = H[-2 * RBsize:-RBsize, -RBsize:].toarray()

        H00 += _get_dense_block(SigR, mapping_diag, nb-1, RBsize)
        H01 += _get_dense_block(SigR, mapping_upper, nb-2, RBsize)
        H10 += _get_dense_block(SigR, mapping_lower, nb-2, RBsize)

        S00 = S[-RBsize:, -RBsize:].toarray()
        S01 = S[-RBsize:, -2 * RBsize:-RBsize].toarray()
        S10 = S[-2 * RBsize:-RBsize, -RBsize:].toarray()

    else:
        if block_idx == 0:
            LBsize = Bmax[0] - Bmin[0] + 1
            H00 = H[:LBsize, :LBsize].toarray()
            H01 = H[:LBsize, LBsize:2 * LBsize].toarray()
            H10 = H[LBsize:2 * LBsize, :LBsize].toarray()

            H00 += _get_dense_block(SigR, mapping_diag, 0, LBsize)
            H01 += _get_dense_block(SigR, mapping_upper, 0, LBsize)
            H10 += _get_dense_block(SigR, mapping_lower, 0, LBsize)

            S00 = S[:LBsize, :LBsize].toarray()
            S01 = S[:LBsize, LBsize:2 * LBsize].toarray()
            S10 = S[LBsize:2 * LBsize, :LBsize].toarray()

        elif block_idx < nb - 1:
            Bsize = Bmax[block_idx] - Bmin[block_idx] + 1
            H00 = H[Bmin[block_idx] - 1:Bmax[block_idx], Bmin[block_idx] - 1:Bmax[block_idx]].toarray()
            H01 = H[Bmin[block_idx] - 1:Bmax[block_idx], Bmin[block_idx+1] - 1:Bmax[block_idx+1]].toarray()
            H10 = H[Bmin[block_idx+1] - 1:Bmax[block_idx+1], Bmin[block_idx] - 1:Bmax[block_idx]].toarray()

            H00 += _get_dense_block(SigR, mapping_diag, block_idx, Bsize)
            H01 += _get_dense_block(SigR, mapping_upper, block_idx, Bsize)
            H10 += _get_dense_block(SigR, mapping_lower, block_idx, Bsize)

            S00 = S[Bmin[block_idx] - 1:Bmax[block_idx], Bmin[block_idx] - 1:Bmax[block_idx]].toarray()
            S01 = S[Bmin[block_idx] - 1:Bmax[block_idx], Bmin[block_idx+1] - 1:Bmax[block_idx+1]].toarray()
            S10 = S[Bmin[block_idx+1] - 1:Bmax[block_idx+1], Bmin[block_idx] - 1:Bmax[block_idx]].toarray()
        else:
            RBsize = Bmax[-1] - Bmin[-1] + 1
            H00 = H[-RBsize:, -RBsize:].toarray()
            H01 = H[-RBsize:, -2 * RBsize:-RBsize].toarray()
            H10 = H[-2 * RBsize:-RBsize, -RBsize:].toarray()

            H00 += _get_dense_block(SigR, mapping_diag, nb-1, RBsize)
            H01 += _get_dense_block(SigR, mapping_upper, nb-2, RBsize)
            H10 += _get_dense_block(SigR, mapping_lower, nb-2, RBsize)

            S00 = S[-RBsize:, -RBsize:].toarray()
            S01 = S[-RBsize:, -2 * RBsize:-RBsize].toarray()
            S10 = S[-2 * RBsize:-RBsize, -RBsize:].toarray()


    Ek = np.sort(np.real(eig(H00 + H01 + H10, b=S00 + S01 + S10, right=False)))

    return Ek