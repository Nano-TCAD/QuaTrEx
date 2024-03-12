# Copyright 2023-2024 ETH Zurich and the QuaTrEx authors. All rights reserved.
import cupy as cp
import numpy as np
import time

from .beyn_new_gpu import extract_small_matrix_blocks_gpu
from .beyn_new_gpu import ci_batched_gpu_internal
from .contour_integral import contour_integral_batched_squared_gpu as ci_batched_squared_gpu_internal
from .contour_integral import contour_integral_batched_combo_gpu as ci_combo_gpu_internal
from .beyn_new_gpu import beyn_phi_gpu, beyn_sigma_gpu, beyn_phi_batched_gpu, beyn_sigma_batched_gpu

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor


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
    
    copy_stream = cp.cuda.Stream(non_blocking=True)

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

        with copy_stream:
            LP0_h = cp.asnumpy(LP0)
            RP0_h = cp.asnumpy(RP0)
        copy_stream.synchronize()

        LV, LS, LW = np.linalg.svd(LP0_h, full_matrices=False)
        Lind = np.where(np.abs(LS) > eps_lim)[0]
        RV, RS, RW = np.linalg.svd(RP0_h, full_matrices=False)
        Rind = np.where(np.abs(RS) > eps_lim)[0]

        if can_fail and (len(Lind) == N or len(Rind) == N):
            return False, (LV, LS, LW, Lind, RV, RS, RW, Rind)

        if len(Lind) == 0:
            Lind = [0]
        if len(Rind) == 0:
            Rind = [0]

        LV = LV[:, Lind]
        LS = LS[Lind]
        LW = LW[Lind, :].T.conj()

        RV = RV[:, Rind]
        RS = RS[Rind]
        RW = RW[Rind, :].T.conj()

        with copy_stream:
            LP1_h = cp.asnumpy(LP1)
            RP1_h = cp.asnumpy(RP1)
        copy_stream.synchronize()

        Llambda, Lu = np.linalg.eig(LV.T.conj() @ LP1_h @ LW @ np.diag(1 / LS))
        Rlambda, Ru = np.linalg.eig(np.diag(1 / RS) @ RV.T.conj() @ RP1_h @ RW)

        return True, (LV, Lu, Llambda, RW, Ru, Rlambda)

    # start = time.time()

    futures = []
    idata = []
    executor = ThreadPoolExecutor(max_workers=batch_size)
    for i in range(batch_size):

        # start_i = time.time()

        P0C1, P1C1 = ci_batched_gpu_internal(N, factor, matrix_blocks[i], 3.0, 1.0, side)
        P0C2, P1C2 = ci_batched_gpu_internal(N, factor, matrix_blocks[i], 1.0 / R, -1.0, side)

        P0 = P0C1 + P0C2
        P1 = P1C1 + P1C2

        LP0 = P0@YL[i]
        LP1 = P1@YL[i]

        RP0 = YR[i]@P0
        RP1 = YR[i]@P1

        # finish_i = time.time()
        # print(f"Time for {i}th contour (1): {finish_i - start_i}", flush=True)

        idata.append((P0C1, P1C1))
        futures.append(executor.submit(_svd, LP0, RP0, LP1, RP1))
    
    # finish = time.time()
    # print(f"Time for all contours (1): {finish - start}", flush=True)
    
    # start = time.time()
    futures2 = [None for _ in range(batch_size)]
    for i, f in enumerate(futures):
        # start_i = time.time()
        success, data = f.result()
        # finish_i = time.time()
        # print(f"Time for {i}th svd-eig (1): {finish_i - start_i}", flush=True)
        if success:
            # start_i = time.time()
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
            # finish_i = time.time()
            # print(f"Time for {i}th phi/sigma: {finish_i - start_i}", flush=True)
        else:
            # start_i = time.time()
            P0C1, P1C1 = idata[i]
            P0C3, P1C3 = ci_batched_gpu_internal(N, factor, matrix_blocks[i], 10.0 / R, -1.0, side)

            P0 = P0C1 + P0C3
            P1 = P1C1 + P1C3

            LP0 = P0@YL[i]
            LP1 = P1@YL[i]

            RP0 = YR[i]@P0
            RP1 = YR[i]@P1

            # finish_i = time.time()
            # print(f"Time for {i}th contour (2): {finish_i - start_i}", flush=True)

            futures2[i] = executor.submit(_svd, LP0, RP0, LP1, RP1, False)

            # finish2_i = time.time()
            # print(f"Time for {i}th submit overhead (2): {finish2_i - finish_i}", flush=True)

    # finish = time.time()
    # print(f"Time for all svd-eig (1) contours (2) and partial phi/sigma: {finish - start}", flush=True)

    # start = time.time()
    for i, f in enumerate(futures2):
        if f is not None:
            # start_i = time.time()
            success, data = f.result()
            # finish_i = time.time()
            # print(f"Time for {i}th svd-eig (2): {finish_i - start_i}", flush=True)
            # start_i = time.time()
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
            # finish_i = time.time()
            # print(f"Time for {i}th phi/sigma: {finish_i - start_i}", flush=True)
    
    # finish = time.time()
    # print(f"Time for svd/eig (2) and last phi/sigma: {finish - start}", flush=True)
    
    return Sigma, gR, cond, min_dEk


def beyn_new_batched_gpu_2(factor: int,
                     matrix_blocks,
                     M00,
                     M01,
                     M10,
                     imag_lim,
                     R,
                     side,
                     YL=None,
                     YR=None):
    
    # copy_stream = cp.cuda.Stream(non_blocking=True)

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

        # with copy_stream:
        LP0_h = cp.asnumpy(LP0)
        RP0_h = cp.asnumpy(RP0)
        # copy_stream.synchronize()

        LV, LS, LW = np.linalg.svd(LP0_h, full_matrices=False)
        Lind = np.where(np.abs(LS) > eps_lim)[0]
        RV, RS, RW = np.linalg.svd(RP0_h, full_matrices=False)
        Rind = np.where(np.abs(RS) > eps_lim)[0]

        if can_fail and (len(Lind) == N or len(Rind) == N):
            return False, (LV, LS, LW, Lind, RV, RS, RW, Rind)

        if len(Lind) == 0:
            Lind = [0]
        if len(Rind) == 0:
            Rind = [0]

        LV = LV[:, Lind]
        LS = LS[Lind]
        LW = LW[Lind, :].T.conj()

        RV = RV[:, Rind]
        RS = RS[Rind]
        RW = RW[Rind, :].T.conj()
        # RW = RW[Rind, :]

        LP1_h = cp.asnumpy(LP1)
        RP1_h = cp.asnumpy(RP1)

        Llambda, Lu = np.linalg.eig(LV.T.conj() @ LP1_h @ LW @ np.diag(1 / LS))
        Rlambda, Ru = np.linalg.eig(np.diag(1 / RS) @ RV.T.conj() @ RP1_h @ RW)
        # Rlambda, Ru = np.linalg.eig(np.diag(1 / RS) @ RV.T.conj() @ RP1_h @ RW.T.conj())

        return True, (LV, Lu, Llambda, RW, Ru, Rlambda)

    start = time.time()

    # futures = []
    idata = []
    svd_data = []
    # for i in range(batch_size):

    #     # start_i = time.time()

    #     P0C1, P1C1 = ci_batched_gpu_internal(N, factor, matrix_blocks[i], 3.0, 1.0, side)
    #     P0C2, P1C2 = ci_batched_gpu_internal(N, factor, matrix_blocks[i], 1.0 / R, -1.0, side)

    #     P0 = P0C1 + P0C2
    #     P1 = P1C1 + P1C2

    #     LP0 = P0@YL[i]
    #     LP1 = P1@YL[i]

    #     RP0 = YR[i]@P0
    #     RP1 = YR[i]@P1

    #     # finish_i = time.time()
    #     # print(f"Time for {i}th contour (1): {finish_i - start_i}", flush=True)

    #     idata.append((P0C1, P0C2))
    #     svd_data.append((LP0, RP0, LP1, RP1))
    
    P0C1_all, P1C1_all = ci_batched_squared_gpu_internal(N, factor, matrix_blocks, 3.0, 1.0, side)
    P0C2_all, P1C2_all = ci_batched_squared_gpu_internal(N, factor, matrix_blocks, 1.0 / R, -1.0, side)

    P0_all = P0C1_all + P0C2_all
    P1_all = P1C1_all + P1C2_all

    LP0_all = P0_all@YL
    LP1_all = P1_all@YL

    RP0_all = YR@P0_all
    RP1_all = YR@P1_all

    
    finish = time.time()
    print(f"Time for all contours (1): {finish - start}", flush=True)
    
    executor = ThreadPoolExecutor(max_workers=batch_size)
    futures = []
    start = time.time()
    for i in range(batch_size):
        P0C1, P1C1 = P0C1_all[i], P1C1_all[i]
        LP0, RP0, LP1, RP1 = LP0_all[i], RP0_all[i], LP1_all[i], RP1_all[i]
        idata.append((P0C1, P1C1))
        svd_data.append((LP0, RP0, LP1, RP1))
        # LP0, RP0, LP1, RP1 = svd_data[i]
        futures.append(executor.submit(_svd, LP0, RP0, LP1, RP1))
    finish = time.time()
    print(f"Time for submitting (1): {finish - start}", flush=True)

    # futures2 = [None for _ in range(batch_size)]
    phi_data = [None for _ in range(batch_size)]
    start = time.time()
    for i, f in enumerate(futures):
        # start_i = time.time()
        success, data = f.result()
        # finish_i = time.time()
        # print(f"Time for {i}th svd-eig (1): {finish_i - start_i}", flush=True)
        if success:
            phi_data[i] = data
    finish = time.time()
    print(f"Time for total svd-eig (1): {finish - start}", flush=True)

    start = time.time()
    for i in range(batch_size):
        if phi_data[i] is None:
            # start_i = time.time()
            P0C1, P1C1 = idata[i]
            P0C3, P1C3 = ci_batched_gpu_internal(N, factor, matrix_blocks[i], 10.0 / R, -1.0, side)

            P0 = P0C1 + P0C3
            P1 = P1C1 + P1C3

            LP0 = P0@YL[i]
            LP1 = P1@YL[i]

            RP0 = YR[i]@P0
            RP1 = YR[i]@P1

            # finish_i = time.time()
            # print(f"Time for {i}th contour (2): {finish_i - start_i}", flush=True)

            svd_data[i] = (LP0, RP0, LP1, RP1)
    finish = time.time()
    print(f"Time for all contours (2): {finish - start}", flush=True)

    start = time.time()
    for i in range(batch_size):
        if phi_data[i] is None:
            LP0, RP0, LP1, RP1 = svd_data[i]
            futures[i] = executor.submit(_svd, LP0, RP0, LP1, RP1, False)
    finish = time.time()
    print(f"Time for submitting (2): {finish - start}", flush=True)

    start = time.time()
    for i in range(batch_size):
        if phi_data[i] is None:
            # start_i = time.time()
            success, data = futures[i].result()
            # finish_i = time.time()
            # print(f"Time for {i}th svd-eig (2): {finish_i - start_i}", flush=True)
            assert success
            phi_data[i] = data
    finish = time.time()
    print(f"Time for total svd-eig (2): {finish - start}", flush=True)

    # start = time.time()
    # for i in range(batch_size):
    #     # start_i = time.time()
    #     LV, Lu, Llambda, RW, Ru, Rlambda = phi_data[i]
    #     LV = cp.asarray(LV)
    #     Lu = cp.asarray(Lu)
    #     Llambda = cp.asarray(Llambda)
    #     RW = cp.asarray(RW)
    #     Ru = cp.asarray(Ru)
    #     Rlambda = cp.asarray(Rlambda)
    #     start_i = time.time()
    #     kL, kR, phiL, phiR = beyn_phi_gpu(LV, Lu, Llambda, RW, Ru, Rlambda, factor, side)
    #     mid_i = time.time()
    #     Sigma[i], gR[i], min_dEk[i] = beyn_sigma_gpu(kL, kR, phiL, phiR, M00[i], M01[i], M10[i], imag_lim, 2, side)
    #     finish_i = time.time()
    #     print(f"Time for {i}th phi: {mid_i - start_i}, sigma: {finish_i - mid_i}", flush=True)
    #     cond[i] = 0
    # #     finish_i = time.time()
    # #     print(f"Time for {i}th phi/sigma: {finish_i - start_i}", flush=True)
    # finish = time.time()
    # print(f"Time for all phi/sigma: {finish - start}", flush=True)

    def _phi_sigma(i):
        LV, Lu, Llambda, RW, Ru, Rlambda = phi_data[i]
        LV = cp.asarray(LV)
        Lu = cp.asarray(Lu)
        Llambda = cp.asarray(Llambda)
        RW = cp.asarray(RW)
        Ru = cp.asarray(Ru)
        Rlambda = cp.asarray(Rlambda)
        kL, kR, phiL, phiR = beyn_phi_gpu(LV, Lu, Llambda, RW, Ru, Rlambda, factor, side)
        Sigma[i], gR[i], min_dEk[i] = beyn_sigma_gpu(kL, kR, phiL, phiR, M00[i], M01[i], M10[i], imag_lim, 2, side)
        cond[i] = 0
        return True


    start = time.time()
    iter = executor.map(_phi_sigma, range(batch_size))
    for it in iter:
        assert it
    finish = time.time()
    print(f"Time for all phi/sigma: {finish - start}", flush=True)
    
    return Sigma, gR, cond, min_dEk


def beyn_new_batched_gpu_3(factor: int,
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

    cond = [0 for _ in range(batch_size)]
    # min_dEk = [1e8 for _ in range(batch_size)]
    # Sigma = [None for _ in range(batch_size)]
    # gR = [None for _ in range(batch_size)]

    if factor * N < 100:
        NM = round(3 * N / 4)
    else:
        NM = round(N / 2)
    NM = factor * NM

    if YL is None:
        YL = cp.random.rand(batch_size, N, NM)
    if YR is None:
        YR = cp.random.rand(batch_size, NM, N)
    
    def _svd(idx, P0C1, P0C2, P0C3, P1C1, P1C2, P1C3):

        P0 = P0C1 + P0C2
        LP0 = cp.asnumpy(P0@YL[idx])
        LV, LS, LW = np.linalg.svd(LP0,  full_matrices=False)
        Lind = np.where(np.abs(LS) > eps_lim)[0]
        if len(Lind) != N:
            RP0 = cp.asnumpy(YR[idx]@P0)
            RV, RS, RW = np.linalg.svd(RP0, full_matrices=False)
            Rind = np.where(np.abs(RS) > eps_lim)[0]
            len_Rind = len(Rind)
        else:
            len_Rind = N
        
        if len_Rind != N:
            P1 = P1C1 + P1C2
            LP1 = cp.asnumpy(P1@YL[idx])
            RP1 = cp.asnumpy(YR[idx]@P1)
        else:
            P0 = P0C1 + P0C3
            P1 = P1C1 + P1C3

            LP0 = cp.asnumpy(P0@YL[idx])
            LP1 = cp.asnumpy(P1@YL[idx])

            RP0 = cp.asnumpy(YR[idx]@P0)
            RP1 = cp.asnumpy(YR[idx]@P1)


            LV, LS, LW = np.linalg.svd(LP0, full_matrices=False)
            Lind = np.where(np.abs(LS) > eps_lim)[0]

            RV, RS, RW = np.linalg.svd(RP0, full_matrices=False)
            Rind = np.where(np.abs(RS) > eps_lim)[0]
        
        if len(Lind) == 0:
            Lind = [0]
        if len(Rind) == 0:
            Rind = [0]

        LV = LV[:, Lind]
        LS = LS[Lind]
        LW = LW[Lind, :].T.conj()

        RV = RV[:, Rind]
        RS = RS[Rind]
        RW = RW[Rind, :].T.conj()

        Llambda, Lu = np.linalg.eig(LV.T.conj() @ LP1 @ LW @ np.diag(1 / LS))
        Rlambda, Ru = np.linalg.eig(np.diag(1 / RS) @ RV.T.conj() @ RP1 @ RW)

        LV = cp.asarray(LV)
        Llambda = cp.asarray(Llambda)
        Lu = cp.asarray(Lu)
        RW = cp.asarray(RW)
        Rlambda = cp.asarray(Rlambda)
        Ru = cp.asarray(Ru)
    

        return LV, Lu, Llambda, RW, Ru, Rlambda

    futures = []
    executor = ThreadPoolExecutor(max_workers=batch_size)
    # start = time.time()
    for i in range(batch_size):
        # start_i = time.time()
        P0C1, P0C2, P0C3, P1C1, P1C2, P1C3 = ci_combo_gpu_internal(N, factor, matrix_blocks[i], [3.0, 1.0 / R, 10.0 / R], [1.0, -1.0, -1.0], side)
        # finish_i = time.time()
        # print(f"Time for {i}th contour: {finish_i - start_i}", flush=True)
        futures.append(executor.submit(_svd, i, P0C1, P0C2, P0C3, P1C1, P1C2, P1C3))
    # finish = time.time()
    # print(f"Time for all contours: {finish - start}", flush=True)

    
    # start = time.time()
    kL = [None for _ in range(batch_size)]
    kR = [None for _ in range(batch_size)]
    phiL = [None for _ in range(batch_size)]
    phiR = [None for _ in range(batch_size)]
    for i, f in enumerate(futures):
        # start_i = time.time()
        LV, Lu, Llambda, RW, Ru, Rlambda = f.result()
        # mid_i = time.time()
        kL[i], kR[i], phiL[i], phiR[i] = beyn_phi_gpu(LV, Lu, Llambda, RW, Ru, Rlambda, factor, side)
        # finish_i = time.time()
        # print(f"Time for {i}th svd-eig: {mid_i - start_i}, phi: {finish_i - mid_i}", flush=True)
    # finish = time.time()
    # print(f"Time for all svd-eig-phi: {finish - start}", flush=True)

    start = time.time()
    Sigma, gR, min_dEk = beyn_sigma_batched_gpu(kL, kR, phiL, phiR, M00, M01, M10, imag_lim, 2, side)
    # finish = time.time()
    # print(f"Time for all sigma: {finish - start}", flush=True)

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


def beyn_batched_gpu_2(factor: int,
                         M00, 
                         M01,
                         M10,
                         imag_lim: float,
                         R: float,
                         type: str,
                         YL = None,
                         YR = None):
    
    N00, N01, N10, matrix_blocks = extract_small_matrix_blocks_batched_gpu(M00, M01, M10, factor, type)
    return beyn_new_batched_gpu_2(factor, matrix_blocks, N00, N01, N10, imag_lim, R, type, YL, YR)





def beyn_batched_gpu_3(factor: int,
                         M00, 
                         M01,
                         M10,
                         imag_lim: float,
                         R: float,
                         type: str,
                         YL = None,
                         YR = None):
    
    N00, N01, N10, matrix_blocks = extract_small_matrix_blocks_batched_gpu(M00, M01, M10, factor, type)
    return beyn_new_batched_gpu_3(factor, matrix_blocks, N00, N01, N10, imag_lim, R, type, YL, YR)