# Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.
import cupy as cp
import cupyx as cpx
import numpy as np
import scipy.io  as io
import os

from scipy.linalg import svd
from numpy.linalg import eig
from scipy.sparse import csr_matrix


def extract_small_matrix_blocks(M00, M01, M10, factor, type):
    N = M00.shape[0] // factor
    num_blocks = 2 * factor + 1

    matrix_blocks = np.empty((num_blocks, N, N), dtype=M00.dtype)

    if type == 'L':

        for i, j in enumerate(range(factor, 1, -1)):
            I = j
            index = i + 1
            matrix_blocks[I - 1] = M00[index * N:(index + 1) * N, :N]
            matrix_blocks[2 * factor + 1 - I] = M00[:N, index * N:(index + 1) * N]

        matrix_blocks[factor] = M00[:N, :N]
        matrix_blocks[0] = M10[:N, :N]
        matrix_blocks[2 * factor] = M01[:N, :N]
    else:

        NM = N * factor

        for i, j in enumerate(range(factor, 1, -1)):
            I = j
            index = i + 1
            matrix_blocks[I - 1] = M00[NM - N:NM, NM - (index + 1) * N:NM - index * N]
            matrix_blocks[2 * factor + 1 - I] = M00[NM - (index + 1) * N:NM - index * N, NM - N:NM]
        
        matrix_blocks[factor] = M00[NM - N:NM, NM - N:NM]
        matrix_blocks[0] = M10[NM - N:NM, NM - N:NM]
        matrix_blocks[2 * factor] = M01[NM - N:NM, NM - N:NM]

    m00 = np.zeros((N * factor, N * factor), dtype=M00.dtype)
    m01 = np.zeros((N * factor, N * factor), dtype=M01.dtype)
    m10 = np.zeros((N * factor, N * factor), dtype=M10.dtype)

    for I in range(1, factor + 1):
        for J in range(1, factor + 1):
            m00[(I - 1) * N:I * N, (J - 1) * N:J * N] = matrix_blocks[factor - I + J]
            if I >= J:
                m01[(I - 1) * N:I * N, (J - 1) * N:J * N] = matrix_blocks[2 * factor - I + J]
            if I <= J:
                m10[(I - 1) * N:I * N, (J - 1) * N:J * N] = matrix_blocks[- I + J]

    m00 = csr_matrix(m00)
    m01 = csr_matrix(m01)
    m10 = csr_matrix(m10)

    return m00, m01, m10, matrix_blocks


def contour_integral_gpu(factor: int,
                         matrix_blocks: np.ndarray,
                         big_N: int,
                         R: float,
                         type: str,
                         YL=None,
                         YR=None):

    theta_min = 0
    theta_max = 2 * np.pi
    NT = 51

    N = big_N // factor

    theta = cp.linspace(theta_min, theta_max, NT)
    dtheta = cp.hstack((theta[1] - theta[0], theta[2:] - theta[:-2], theta[-1] - theta[-2])) / 2

    c = 0
    r1 = np.power(3.0, 1.0 / factor)
    r2 = np.power(1.0 / R, 1.0 / factor)

    zC1 = c + r1 * cp.exp(1j * theta)
    zC2 = c + r2 * cp.exp(1j * theta)
    z = cp.hstack((zC1, zC2))

    dzC1_dtheta = 1j * r1 * cp.exp(1j * theta)
    dzC2_dtheta = 1j * r2 * cp.exp(1j * theta)
    dz_dtheta = cp.hstack((dzC1_dtheta, -dzC2_dtheta))

    dtheta =  cp.hstack((dtheta, dtheta))

    if factor * N < 100:
        NM = round(3 * N / 4)
    else:
        NM = round(N / 2)
    NM = factor * NM

    if YL is None:
        YL = cp.random.rand(N, NM)
    if YR is None:
        YR = cp.random.rand(NM, N)
    P0 = cp.zeros((N, N), dtype=np.complex128)
    P1 = cp.zeros((N, N), dtype=np.complex128)

    for I in range(len(z)):

        T = cp.zeros((N, N), dtype=np.complex128)
        for J in range(2 * factor + 1):
            if type == 'L':
                T = T + matrix_blocks[J] * z[I] ** (factor - J)
            else:
                T = T + matrix_blocks[J] * z[I] ** (J - factor)

        iT = cp.linalg.inv(T)

        P0 += iT*dz_dtheta[I]*dtheta[I]/(2*np.pi*1j)
        P1 += iT*z[I]*dz_dtheta[I]*dtheta[I]/(2*np.pi*1j)

    LP0 = P0@YL
    LP1 = P1@YL

    RP0 = YR@P0
    RP1 = YR@P1

    return LP0, LP1, RP0, RP1


def contour_integral_batched_gpu(factor: int,
                                 matrix_blocks: np.ndarray,
                                 big_N: int,
                                 R: float,
                                 type: str,
                                 YL=None,
                                 YR=None):

    theta_min = 0
    theta_max = 2 * np.pi
    NT = 51

    N = big_N // factor

    theta = cp.linspace(theta_min, theta_max, NT)
    dtheta = cp.hstack((theta[1] - theta[0], theta[2:] - theta[:-2], theta[-1] - theta[-2])) / 2

    c = 0
    r1 = np.power(3.0, 1.0 / factor)
    r2 = np.power(1.0 / R, 1.0 / factor)

    zC1 = c + r1 * cp.exp(1j * theta)
    zC2 = c + r2 * cp.exp(1j * theta)
    z = np.hstack((zC1, zC2))

    dzC1_dtheta = 1j * r1 * cp.exp(1j * theta)
    dzC2_dtheta = 1j * r2 * cp.exp(1j * theta)
    dz_dtheta = cp.hstack((dzC1_dtheta, -dzC2_dtheta))

    dtheta = cp.hstack((dtheta, dtheta))

    if factor * N < 100:
        NM = round(3 * N / 4)
    else:
        NM = round(N / 2)
    NM = factor * NM

    if YL is None:
        YL = cp.random.rand(N, NM)
    if YR is None:
        YR = cp.random.rand(NM, N)
    P0 = cp.zeros((N, N), dtype=np.complex128)
    P1 = cp.zeros((N, N), dtype=np.complex128)

    T = cp.zeros((len(z), N, N), dtype=np.complex128)

    for I in range(len(z)):
        for J in range(2 * factor + 1):
            if type == 'L':
                T[I] = T[I] + matrix_blocks[J] * z[I] ** (factor - J)
            else:
                T[I] = T[I] + matrix_blocks[J] * z[I] ** (J - factor)

    iT = cp.linalg.inv(T)

    P0 = cp.sum(iT*(dz_dtheta*dtheta/(2*np.pi*1j)).reshape(len(z), 1, 1), axis=0)
    P1 = cp.sum(iT*(z*dz_dtheta*dtheta/(2*np.pi*1j)).reshape(len(z), 1, 1), axis=0)

    LP0 = P0@YL
    LP1 = P1@YL

    RP0 = YR@P0
    RP1 = YR@P1

    return LP0, LP1, RP0, RP1


def contour_integral(factor: int,
                     matrix_blocks: np.ndarray,
                     big_N: int,
                     R: float,
                     type: str,
                     YL=None,
                     YR=None):

    theta_min = 0
    theta_max = 2 * np.pi
    NT = 51

    N = big_N // factor

    theta = np.linspace(theta_min, theta_max, NT)
    dtheta = np.hstack((theta[1] - theta[0], theta[2:] - theta[:-2], theta[-1] - theta[-2])) / 2

    c = 0
    r1 = np.power(3.0, 1.0 / factor)
    r2 = np.power(1.0 / R, 1.0 / factor)

    zC1 = c + r1 * np.exp(1j * theta)
    zC2 = c + r2 * np.exp(1j * theta)
    z = np.hstack((zC1, zC2))

    dzC1_dtheta = 1j * r1 * np.exp(1j * theta)
    dzC2_dtheta = 1j * r2 * np.exp(1j * theta)
    dz_dtheta = np.hstack((dzC1_dtheta, -dzC2_dtheta))

    dtheta = np.hstack((dtheta, dtheta))

    if factor * N < 100:
        NM = round(3 * N / 4)
    else:
        NM = round(N / 2)
    NM = factor * NM

    if YL is None:
        YL = np.random.rand(N, NM)
    if YR is None:
        YR = np.random.rand(NM, N)
    P0 = np.zeros((N, N), dtype=np.complex128)
    P1 = np.zeros((N, N), dtype=np.complex128)

    for I in range(len(z)):

        T = np.zeros((N, N), dtype=np.complex128)
        for J in range(2 * factor + 1):
            if type == 'L':
                T = T + matrix_blocks[J] * z[I] ** (factor - J)
            else:
                T = T + matrix_blocks[J] * z[I] ** (J - factor)

        iT = np.linalg.inv(T)

        P0 += iT*dz_dtheta[I]*dtheta[I]/(2*np.pi*1j)
        P1 += iT*z[I]*dz_dtheta[I]*dtheta[I]/(2*np.pi*1j)

    LP0 = P0@YL
    LP1 = P1@YL

    RP0 = YR@P0
    RP1 = YR@P1

    return LP0, LP1, RP0, RP1


def contour_integral_batched(factor: int,
                             matrix_blocks: np.ndarray,
                             big_N: int,
                             R: float,
                             type: str,
                             YL=None,
                             YR=None):

    theta_min = 0
    theta_max = 2 * np.pi
    NT = 51

    N = big_N // factor

    theta = np.linspace(theta_min, theta_max, NT)
    dtheta = np.hstack((theta[1] - theta[0], theta[2:] - theta[:-2], theta[-1] - theta[-2])) / 2

    c = 0
    r1 = np.power(3.0, 1.0 / factor)
    r2 = np.power(1.0 / R, 1.0 / factor)

    zC1 = c + r1 * np.exp(1j * theta)
    zC2 = c + r2 * np.exp(1j * theta)
    z = np.hstack((zC1, zC2))

    dzC1_dtheta = 1j * r1 * np.exp(1j * theta)
    dzC2_dtheta = 1j * r2 * np.exp(1j * theta)
    dz_dtheta = np.hstack((dzC1_dtheta, -dzC2_dtheta))

    dtheta = np.hstack((dtheta, dtheta))

    if factor * N < 100:
        NM = round(3 * N / 4)
    else:
        NM = round(N / 2)
    NM = factor * NM

    if YL is None:
        YL = np.random.rand(N, NM)
    if YR is None:
        YR = np.random.rand(NM, N)
    P0 = np.zeros((N, N), dtype=np.complex128)
    P1 = np.zeros((N, N), dtype=np.complex128)

    T = np.zeros((len(z), N, N), dtype=np.complex128)

    for I in range(len(z)):
        for J in range(2 * factor + 1):
            if type == 'L':
                T[I] = T[I] + matrix_blocks[J] * z[I] ** (factor - J)
            else:
                T[I] = T[I] + matrix_blocks[J] * z[I] ** (J - factor)

    iT = np.linalg.inv(T)

    P0 = np.sum(iT*(dz_dtheta*dtheta/(2*np.pi*1j)).reshape(len(z), 1, 1), axis=0)
    P1 = np.sum(iT*(z*dz_dtheta*dtheta/(2*np.pi*1j)).reshape(len(z), 1, 1), axis=0)

    LP0 = P0@YL
    LP1 = P1@YL

    RP0 = YR@P0
    RP1 = YR@P1

    return LP0, LP1, RP0, RP1


def beyn_svd(LP0, RP0, eps_lim=1e-8):

    LV, LS, LW = svd(LP0, full_matrices=False)
    Lind = np.where(np.abs(LS) > eps_lim)[0]

    RV, RS, RW = svd(RP0, full_matrices=False)
    Rind = np.where(np.abs(RS) > eps_lim)[0]

    LV = LV[:, Lind]
    LS = np.diag(LS[Lind])
    LW = LW[Lind, :].conj().T

    RV = RV[:, Rind]
    RS = np.diag(RS[Rind])
    RW = RW[Rind, :].conj().T

    return LV, LS, LW, RV, RS, RW


def beyn_eig(LV, LS, LW, LP1, RV, RS, RW, RP1):

    Llambda, Lu = eig(LV.conj().T @ LP1 @ LW @ np.linalg.inv(LS))
    Rlambda, Ru = eig(np.linalg.inv(RS) @ RV.conj().T @ RP1 @ RW)

    return Lu, Llambda, Ru, Rlambda


def beyn_phi(LV, Lu, Llambda, RW, Ru, Rlambda, factor, type):

    N = LV.shape[0]

    phiL = LV @ Lu
    phiR = np.linalg.solve(Ru, RW.conj().T)

    if type == 'L':
        kL = 1j * np.log(Llambda)
        kR = 1j * np.log(Rlambda)
    else:
        kL = -1j * np.log(Llambda)
        kR = -1j * np.log(Rlambda)
    
    ind_sort_kL = np.argsort(abs(np.imag(kL)))
    kL = kL[ind_sort_kL]
    phiL = phiL[:, ind_sort_kL]

    ind_sort_kR = np.argsort(abs(np.imag(kR)))
    kR = kR[ind_sort_kR]
    phiR = phiR[ind_sort_kR, :]

    phiL_vec = np.zeros((factor * N, len(kL)), dtype=phiL.dtype)
    phiR_vec = np.zeros((len(kR), factor * N), dtype=phiR.dtype)

    for i in range(factor):
        phiL_vec[i * N:(i + 1) * N, :] = phiL @ np.diag(np.exp(i * 1j * kL))
        phiR_vec[:, i * N:(i + 1) * N] = np.diag(np.exp(-i * 1j * kR)) @ phiR
    
    phiL = phiL_vec / (np.ones((factor * N, 1)) @ np.sqrt(np.sum(np.abs(phiL_vec) ** 2, axis=0, keepdims=True)))
    phiR = phiR_vec / (np.sqrt(np.sum(np.abs(phiR_vec) ** 2, axis=1, keepdims=True)) @ np.ones((1, factor * N)))

    kL = factor * kL
    kR = factor * kR

    return kL, kR, phiL, phiR


def beyn_sigma(kL, kR, phiL, phiR, M00, M01, M10, imag_lim, ref_iteration, type):

    if type == 'L':

        ksurfL, VsurfL, inv_VsurfL, dEk_dk = prepare_input_data(kL, kR, phiL, phiR, M01, M10, imag_lim, 1.0)
        gR = np.linalg.inv(M00 + M10 @ VsurfL @ np.diag(np.exp(-1j * ksurfL)) @ inv_VsurfL)

        for _ in range(ref_iteration):
            gR = np.linalg.inv(M00 - M10 @ gR @ M01)
        
        Sigma = M10 @ gR @ M01
    
    else:

        ksurfR, VsurfR, inv_VsurfR, dEk_dk = prepare_input_data(kL, kR, phiL, phiR, M01, M10, imag_lim, -1.0)
        gR = np.linalg.inv(M00 + M01 @ VsurfR @ np.diag(np.exp(1j * ksurfR)) @ inv_VsurfR)

        for _ in range(ref_iteration):
            gR = np.linalg.inv(M00 - M01 @ gR @ M10)
        
        Sigma = M01 @ gR @ M10

    ind = np.where(abs(dEk_dk))
    if len(ind[0]) > 0:
        min_dEk = np.min(abs(dEk_dk[ind]))
    else:
        min_dEk = 1e8
    
    return Sigma, gR, min_dEk


def check_imag_cond(kref, kL, kR, phiL, phiR, M10, M01, max_imag, kside):
    Nk = len(kref)
    imag_cond = np.zeros(Nk, dtype=np.bool_)
    dEk_dk = np.zeros(Nk, dtype=phiL.dtype)

    ind = np.where(np.abs(np.imag(kref)) < np.max((0.5, max_imag)))[0]
    Ikmax = len(ind)

    if Ikmax % 2 == 1:
        Ikmax += 1

    for Ik in range(Ikmax):

        if kside == 'L':
            ind_kR = np.argmin(np.abs(np.ones(len(kR)) * kL[Ik] - kR))
            dEk_dk[Ik] = -(phiR[ind_kR, :] @ (-1j * M10 * np.exp(-1j * kL[Ik]) + 1j * M01 * np.exp(1j * kL[Ik])) @ phiL[:, Ik]) / \
                (phiR[ind_kR, :] @ phiL[:, Ik])
        
        else:
            ind_kL = np.argmin(np.abs(np.ones(len(kL)) * kR[Ik] - kL))
            dEk_dk[Ik] = -(phiR[Ik, :] @ (-1j * M10 * np.exp(-1j * kR[Ik]) + 1j * M01 * np.exp(1j * kR[Ik])) @ phiL[:, ind_kL]) / \
                (phiR[Ik, :] @ phiL[:, ind_kL])

    for Ik in range(Ikmax):
        if not imag_cond[Ik]:
            ind_neigh = np.argmin(np.abs(dEk_dk + dEk_dk[Ik] * np.ones(Nk)))

            if kside == 'L':
                k1 = kL[Ik]
                k2 = kL[ind_neigh]
            else:
                k1 = kR[Ik]
                k2 = kR[ind_neigh]
            dEk1 = dEk_dk[Ik]
            dEk2 = dEk_dk[ind_neigh]

            cond1 = np.abs(dEk1 + dEk2) / (np.abs(dEk1) + 1e-10) < 0.25
            cond2 = np.abs(k1 + k2) / (np.abs(k1) + 1e-10) < 0.25
            cond3 = (np.abs(np.imag(k1)) + np.abs(np.imag(k2))) / 2.0 < 1.5 * max_imag
            cond4 = np.sign(np.imag(k1)) == np.sign(np.imag(k2))

            if cond1 and cond2 and (cond3 or (not cond3 and cond4)):
                if not imag_cond[ind_neigh]:
                    imag_cond[Ik] = 1
                    imag_cond[ind_neigh] = 1

    return imag_cond, dEk_dk


def sort_k(kL, kR, phiL, phiR, M01, M10, imag_limit, factor, kside):
    Nk = max(len(kL), len(kR))
    ksurf = np.zeros(Nk, dtype=kL.dtype)
    dEk = np.zeros(Nk, dtype=phiL.dtype)

    if kside == 'L':
        NT = len(phiL[:, 0])
        Vsurf = np.zeros((NT, Nk), dtype=phiL.dtype)
        kref = kL
    else:
        NT = len(phiR[0, :])
        Vsurf = np.zeros((Nk, NT), dtype=phiR.dtype)
        kref = kR

    imag_cond, dEk_dk = check_imag_cond(kref, kL, kR, phiL, phiR, M10, M01, imag_limit, kside)

    Nref = 0

    for Ik in range(len(kref)):

        if not imag_cond[Ik] and abs(np.imag(kref[Ik])) > 1e-6:

            if factor * np.imag(kref[Ik]) < 0:

                Nref += 1
                ksurf[Nref - 1] = kref[Ik]

                if kside == 'L':
                    Vsurf[:, Nref - 1] = phiL[:, Ik]
                else:
                    Vsurf[Nref - 1, :] = phiR[Ik, :]

                dEk[Nref - 1] = 0

        else:

            cond1 = (abs(np.real(dEk_dk[Ik])) < abs(np.imag(dEk_dk[Ik])) / 100) and (factor * np.imag(kref[Ik]) < 0)
            cond2 = (abs(np.real(dEk_dk[Ik])) >= abs(np.imag(dEk_dk[Ik])) / 100) and (factor * np.real(dEk_dk[Ik]) < 0)

            if cond1 or cond2:

                Nref += 1
                ksurf[Nref - 1] = kref[Ik]

                if kside == 'L':
                    Vsurf[:, Nref - 1] = phiL[:, Ik]
                else:
                    Vsurf[Nref - 1, :] = phiR[Ik, :]
                
                dEk[Nref - 1] = dEk_dk[Ik]

    ksurf = ksurf[0:Nref]

    if kside == 'L':
        Vsurf = Vsurf[:, 0:Nref]
    else:
        Vsurf = Vsurf[0:Nref, :]
    
    dEk = dEk[0:Nref]

    return ksurf, Vsurf, dEk


def prepare_input_data(kL, kR, phiL, phiR, M01, M10, imag_limit, factor):
    ksurfL, VsurfL, dEk = sort_k(kL, kR, phiL, phiR, M01, M10, imag_limit, factor, 'L')
    ksurfR, VsurfR, _ = sort_k(kL, kR, phiL, phiR, M01, M10, imag_limit, factor, 'R')

    Nk = min(len(ksurfL), len(ksurfR))

    VsurfL = VsurfL[:, :Nk]
    ksurfL = ksurfL[:Nk]
    VsurfR = VsurfR[:Nk, :]

    # Compute the inverse of VsurfL with the help from the VsurfR eigenvectors
    # inv_VsurfL = np.linalg.pinv(VsurfR @ VsurfL) @ VsurfR
    inv_VsurfL = np.linalg.solve(VsurfR @ VsurfL, VsurfR)

    return ksurfL, VsurfL, inv_VsurfL, dEk


def beyn_new(factor: int,
             matrix_blocks,
             M00,
             M01,
             M10,
             imag_lim,
             R,
             type,
             YL=None,
             YR=None):
    

    # theta_min = 0
    # theta_max = 2 * np.pi
    # NT = 51
    # eps_lim = 1e-8
    # ref_iteration = 2
    # cond = 0
    # min_dEk = 1e8

    cond = 0
    min_dEk = 1e8

    try:
    
        LP0, LP1, RP0, RP1 = contour_integral(factor, matrix_blocks, M00.shape[0], R, type, YL=YL, YR=YR)
        LV, LS, LW, RV, RS, RW = beyn_svd(LP0, RP0, eps_lim=1e-8)

        if LS.size == 0 or RS.size == 0:
            raise Exception("No singular values above the threshold")

        Lu, Llambda, Ru, Rlambda = beyn_eig(LV, LS, LW, LP1, RV, RS, RW, RP1)
        kL, kR, phiL, phiR = beyn_phi(LV, Lu, Llambda, RW, Ru, Rlambda, factor, type)
        Sigma, gR, min_dEk = beyn_sigma(kL, kR, phiL, phiR, M00, M01, M10, imag_lim, 2, type)
    
    except Exception as e:

        print("Error in Beyn:")
        print(e)

        cond = np.nan
        Sigma = None
        gR = None

    return Sigma, gR, cond, min_dEk


def beyn(factor: int,
         M00,
         M01,
         M10,
         imag_lim,
         R,
         type,
         YL=None,
         YR=None):
    
    N00, N01, N10, matrix_blocks = extract_small_matrix_blocks(M00, M01, M10, factor, type)
    return beyn_new(factor, matrix_blocks, N00, N01, N10, imag_lim, R, type, YL=YL, YR=YR)
