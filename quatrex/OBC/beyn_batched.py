# Copyright 2023-2024 ETH Zurich and the QuaTrEx authors. All rights reserved.
import cupy as cp
import numpy as np

from .beyn_new_gpu import extract_small_matrix_blocks_gpu
from .beyn_new_gpu import ci_batched_gpu_internal
from .beyn_new_gpu import beyn_phi_gpu, beyn_sigma_gpu

from concurrent.futures import ThreadPoolExecutor


def extract_small_matrix_blocks_batched_gpu(M00, M01, M10, factor, type):

    batch_size = M00.shape[0]
    N = M00.shape[1]
    small_N = N // factor
    big_N = small_N * factor
 
    N00 = cp.empty((batch_size, big_N, big_N), dtype=M00.dtype)
    N01 = cp.empty((batch_size, big_N, big_N), dtype=M00.dtype)
    N10 = cp.empty((batch_size, big_N, big_N), dtype=M00.dtype)
    matrix_blocks = cp.empty((batch_size, 2 * factor + 1, small_N, small_N), dtype=M00.dtype)

    # TODO: Actual batched version
    for i in range(batch_size):
        N00[i], N01[i], N10[i], matrix_blocks[i] = extract_small_matrix_blocks_gpu(M00[i], M01[i], M10[i], factor, type, densify=True)
    
    return N00, N01, N10, matrix_blocks


def beyn_new_batched_gpu(factor: int,
                     matrix_blocks,
                     M00,
                     M01,
                     M10,
                     imag_lim,
                     R,
                     side,
                     YL=None,
                     YR=None):
    eps_lim = 1e-8

    batch_size = M00.shape[0]
    big_N = M00.shape[1]
    N = big_N // factor

    cond = [np.nan for _ in range(batch_size)]
    min_dEk = [1e8 for _ in range(batch_size)]
    Sigma = [None for _ in range(batch_size)]
    gR = [None for _ in range(batch_size)]

    if factor * N < 100:
        NM = round(3 * N / 4)
    else:
        NM = round(N / 2)
    NM = factor * NM

    if YL is None:
        YL = cp.random.rand(batch_size, N, NM)
    if YR is None:
        YR = cp.random.rand(batch_size, NM, N)
    
    def _svd(LP0, RP0, LP1, RP1, can_fail=True):
        LV, LS, LW = np.linalg.svd(LP0, full_matrices=False)
        Lind = np.where(np.abs(LS) > eps_lim)[0]
        RV, RS, RW = np.linalg.svd(RP0, full_matrices=False)
        Rind = np.where(np.abs(RS) > eps_lim)[0]

        if can_fail and (len(Lind) == N or len(Rind) == N):
            return False, (LV, LS, LW, Lind, RV, RS, RW, Rind)

        if len(Lind) == 0:
            Lind = [0]
        if len(Rind) == 0:
            Rind = [0]

        LV = LV[:, Lind]
        LS = np.diag(LS[Lind])
        LW = LW[Lind, :].T.conj()

        RV = RV[:, Rind]
        RS = np.diag(RS[Rind])
        RW = RW[Rind, :].T.conj()

        Llambda, Lu = np.linalg.eig(LV.T.conj() @ LP1 @ LW @ np.linalg.inv(LS))
        Rlambda, Ru = np.linalg.eig(np.linalg.inv(RS) @ RV.T.conj() @ RP1 @ RW)

        return True, (LV, Lu, Llambda, RW, Ru, Rlambda)

    
    futures = []
    idata = []
    executor = ThreadPoolExecutor(max_workers=1)
    for i in range(batch_size):

        P0C1, P1C1 = ci_batched_gpu_internal(N, factor, matrix_blocks[i], 3.0, 1.0, side)
        P0C2, P1C2 = ci_batched_gpu_internal(N, factor, matrix_blocks[i], 1.0 / R, -1.0, side)

        P0 = P0C1 + P0C2
        P1 = P1C1 + P1C2

        LP0 = P0@YL[i]
        LP1 = P1@YL[i]

        RP0 = YR[i]@P0
        RP1 = YR[i]@P1

        idata.append((P0C1, P0C2))
        futures.append(executor.submit(_svd, LP0.get(), RP0.get(), LP1.get(), RP1.get()))
    
    futures2 = [None for _ in range(batch_size)]
    for i, f in enumerate(futures):
        success, data = f.result()
        if success:
            LV, Lu, Llambda, RW, Ru, Rlambda = data
            LV = cp.asarray(LV)
            Lu = cp.asarray(Lu)
            Llambda = cp.asarray(Llambda)
            RW = cp.asarray(RW)
            Ru = cp.asarray(Ru)
            Rlambda = cp.asarray(Rlambda)
            kL, kR, phiL, phiR = beyn_phi_gpu(LV, Lu, Llambda, RW, Ru, Rlambda, factor, side)
            Sigma[i], gR[i], min_dEk[i] = beyn_sigma_gpu(kL, kR, phiL, phiR, M00[i], M01[i], M10[i], imag_lim, 2, side)
            cond[i] = 0
        else:
            P0C1, P0C2 = idata[i]
            P0C3, P1C3 = ci_batched_gpu_internal(N, factor, matrix_blocks[i], 10.0 / R, -1.0, side)

            P0 = P0C1 + P0C3
            P1 = P1C1 + P1C3

            LP0 = P0@YL[i]
            LP1 = P1@YL[i]

            RP0 = YR[i]@P0
            RP1 = YR[i]@P1

            futures2[i] = executor.submit(_svd, LP0.get(), RP0.get(), LP1.get(), RP1.get(), False)

    for i, f in enumerate(futures2):
        if f is not None:
            success, data = f.result()
            assert success
            LV, Lu, Llambda, RW, Ru, Rlambda = data
            LV = cp.asarray(LV)
            Lu = cp.asarray(Lu)
            Llambda = cp.asarray(Llambda)
            RW = cp.asarray(RW)
            Ru = cp.asarray(Ru)
            Rlambda = cp.asarray(Rlambda)
            kL, kR, phiL, phiR = beyn_phi_gpu(LV, Lu, Llambda, RW, Ru, Rlambda, factor, side)
            Sigma[i], gR[i], min_dEk[i] = beyn_sigma_gpu(kL, kR, phiL, phiR, M00[i], M01[i], M10[i], imag_lim, 2, side)
            cond[i] = 0
    
    return Sigma, gR, cond, min_dEk


def beyn_batched_gpu(factor: int,
                         M00, 
                         M01,
                         M10,
                         imag_lim: float,
                         R: float,
                         type: str,
                         YL = None,
                         YR = None):
    
    N00, N01, N10, matrix_blocks = extract_small_matrix_blocks_batched_gpu(M00, M01, M10, factor, type)
    return beyn_new_batched_gpu(factor, matrix_blocks, N00, N01, N10, imag_lim, R, type, YL, YR)