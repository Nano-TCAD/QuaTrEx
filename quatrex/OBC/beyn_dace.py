# Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.

import dace
import numpy as np
import time

from dace.transformation.auto.auto_optimize import auto_optimize
from numpy.linalg import eig
from scipy.linalg import svd

theta_min = 0
theta_max = 2 * np.pi
NT = 51
eps_lim = 1e-8
ref_iteration = 2
# rng = np.random.default_rng(42)
np.random.seed(0)

N, NM = (dace.symbol(s) for s in ('N', 'NM'))


def check_imag_cond(k, kR, phiR, phiL, M10, M01, max_imag):
    imag_cond = np.zeros(len(k))
    dEk_dk = np.zeros(len(k), dtype=np.cfloat)

    ind = np.where(np.abs(np.imag(k)) < np.max((0.5, max_imag)))[0]
    Ikmax = len(ind)

    if Ikmax % 2 == 1:
        Ikmax += 1

    for Ik in range(Ikmax):
        ind_kR = np.argmin(np.abs(np.ones(len(kR)) * k[Ik] - kR))

        dEk_dk[Ik] = -(phiR[ind_kR, :] @ (-1j * M10 * np.exp(-1j * k[Ik]) + 1j * M01 * np.exp(1j * k[Ik])) @ phiL[:, Ik]) / \
            (phiR[ind_kR, :] @ phiL[:, Ik])

    for Ik in range(Ikmax):
        if not imag_cond[Ik]:
            ind_neigh = np.argmin(np.abs(dEk_dk + dEk_dk[Ik] * np.ones(len(k))))

            k1 = k[Ik]
            k2 = k[ind_neigh]
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


def sort_k(k, kR, phiL, phiR, M01, M10, imag_limit, factor):
    Nk = len(k)
    NT = len(phiL[:, 0])

    ksurf = np.zeros(Nk, dtype=np.cfloat)
    Vsurf = np.zeros((NT, Nk), dtype=np.cfloat)

    imag_cond, dEk_dk = check_imag_cond(k, kR, phiR, phiL, M10, M01, imag_limit)

    Nref = 0

    for Ik in range(Nk):

        if not imag_cond[Ik] and abs(np.imag(k[Ik])) > 1e-6:

            if factor * np.imag(k[Ik]) < 0:

                Nref += 1

                ksurf[Nref - 1] = k[Ik]
                Vsurf[:, Nref - 1] = phiL[:, Ik]

        else:

            cond1 = (abs(np.real(dEk_dk[Ik])) < abs(np.imag(dEk_dk[Ik])) / 100) and (factor * np.imag(k[Ik]) < 0)
            cond2 = (abs(np.real(dEk_dk[Ik])) >= abs(np.imag(dEk_dk[Ik])) / 100) and (factor * np.real(dEk_dk[Ik]) < 0)

            if cond1 or cond2:

                Nref += 1

                ksurf[Nref - 1] = k[Ik]
                Vsurf[:, Nref - 1] = phiL[:, Ik]
    ksurf = ksurf[0:Nref]
    Vsurf = Vsurf[:, 0:Nref]
    return ksurf, Vsurf, dEk_dk


@dace.program
def check_imag_cond_dace(imag_cond: dace.bool[NM], dEk_dk: dace.complex128[NM], k: dace.complex128[NM],
                         kR: dace.complex128[NM], phiR: dace.complex128[NM, N], phiL: dace.complex128[N, NM],
                         M10: dace.complex128[N, N], M01: dace.complex128[N, N], max_imag: dace.float64):

    # imag_cond = np.zeros((NM,), dtype=np.bool_)
    # dEk_dk = np.zeros((NM,), dtype =  np.complex128)
    imag_cond[:] = False
    dEk_dk[:] = 0

    # ind = np.where(np.abs(np.imag(k)) < np.max((0.5, max_imag)))[0]
    # Ikmax = len(ind)

    # if Ikmax % 2 == 1:
    #     Ikmax += 1

    Ikmax = np.add.reduce(
        np.int32(np.abs(np.imag(k)) < np.maximum(0.5, max_imag)))  # Casting because reduces to bool otherwise
    Ikmax += Ikmax % 2

    # ind_kRs = np.argmin(np.abs(np.ones((NM, 1)) * k - kR), axis=1)
    ind_kRs = np.argmin(np.abs(np.subtract.outer(k, kR)), axis=1)

    for Ik in range(Ikmax):
        # ind_kR = np.argmin(np.abs(np.ones(len(kR)) * k[Ik] - kR))
        ind_kR = ind_kRs[Ik]

        dEk_dk[Ik] = -(phiR[ind_kR, :] @ (-(1j * M10) * np.exp(-(1j * k[Ik])) + 1j * M01 * np.exp(1j * k[Ik]))
                       @ phiL[:, Ik]) / (phiR[ind_kR, :] @ phiL[:, Ik])

    # ind_neighs = np.argmin(np.abs(np.ones((NM,1)) * dEk_dk  + dEk_dk), axis=1)
    ind_neighs = np.argmin(np.abs(np.add.outer(dEk_dk, dEk_dk)), axis=1)

    for Ik in range(Ikmax):
        if not imag_cond[Ik]:
            # ind_neigh = np.argmin(np.abs(dEk_dk + dEk_dk[Ik] * np.ones(len(k))))
            ind_neigh = ind_neighs[Ik]

            k1 = k[Ik]
            k2 = k[ind_neigh]
            dEk1 = dEk_dk[Ik]
            dEk2 = dEk_dk[ind_neigh]

            cond1 = np.abs(dEk1 + dEk2) / (np.abs(dEk1) + 1e-10) < 0.25
            cond2 = np.abs(k1 + k2) / (np.abs(k1) + 1e-10) < 0.25
            cond3 = (np.abs(np.imag(k1)) + np.abs(np.imag(k2))) / 2.0 < 1.5 * max_imag
            cond4 = np.sign(np.imag(k1)) == np.sign(np.imag(k2))

            if cond1 and cond2 and (cond3 or (not cond3 and cond4)):
                if not imag_cond[ind_neigh]:
                    imag_cond[Ik] = True
                    imag_cond[ind_neigh] = True

    # return imag_cond, dEk_dk


@dace.program
def sort_k_dace(ksurf: dace.complex128[NM], Vsurf: dace.complex128[N, NM], dEk_dk: dace.complex128[NM],
                out_Nref: dace.int32[1], k: dace.complex128[NM], kR: dace.complex128[NM], phiL: dace.complex128[N, NM],
                phiR: dace.complex128[NM, N], M01: dace.complex128[N, N], M10: dace.complex128[N, N],
                imag_limit: dace.float64, factor: dace.float64):

    # ksurf = np.zeros((NM,), dtype=np.complex128)
    # Vsurf = np.zeros((N, NM), dtype=np.complex128)
    ksurf[:] = 0
    Vsurf[:] = 0

    # imag_cond, dEk_dk = check_imag_cond_dace(k, kR, phiR, phiL, M10, M01, imag_limit)
    imag_cond = np.empty((NM, ), dtype=np.bool_)
    check_imag_cond_dace(imag_cond, dEk_dk, k, kR, phiR, phiL, M10, M01, imag_limit)

    Nref = 0

    for Ik in range(NM):

        if not imag_cond[Ik] and abs(np.imag(k[Ik])) > 1e-6:

            if factor * np.imag(k[Ik]) < 0:

                ksurf[Nref] = k[Ik]
                Vsurf[:, Nref] = phiL[:, Ik]
                Nref += 1

        else:

            cond1 = (abs(np.real(dEk_dk[Ik])) < abs(np.imag(dEk_dk[Ik])) / 100) and (factor * np.imag(k[Ik]) < 0)
            cond2 = (abs(np.real(dEk_dk[Ik])) >= abs(np.imag(dEk_dk[Ik])) / 100) and (factor * np.real(dEk_dk[Ik]) < 0)

            if cond1 or cond2:

                Nref += 1

                ksurf[Nref - 1] = k[Ik]
                Vsurf[:, Nref - 1] = phiL[:, Ik]
    # ksurf = ksurf[0:Nref]
    # Vsurf = Vsurf[:,0:Nref]
    # return ksurf, Vsurf, dEk_dk, Nref
    out_Nref[0] = Nref


from . import beyn_globals as bg


def beyn(M00, M01, M10, imag_lim, R, type: str = 'L', function: str = 'W', block: bool = False, validate: bool = False):

    R = np.float64(R)

    if block:
        contour_integral = bg.contour_integral_block
    contour_integral = bg.contour_integral
    # sortk = bg.sort_k

    if contour_integral is None:
        contour_integral, _ = contour_integral_dace.load_precompiled_sdfg(f'.dacecache/{contour_integral_dace.name}')
    # if sortk is None:
    #     sortk, _ = sort_k_dace.load_precompiled_sdfg(f'.dacecache/{sort_k_dace.name}')

    # ctime = - time.perf_counter()

    N = M00.shape[0]
    min_dEk = 1e8
    cond = 0

    if N < 100:
        NM = round(3 * N / 4)
    else:
        NM = round(N / 2)

    Y = np.complex128(np.random.rand(N, NM))

    # LP0, LP1, RP0, RP1 = contour_integral(M00=M00, M01=M01, M10=M10, Y=Y, R=R, is_left=np.bool_(type=='L'), N=N, NM=NM)
    LP0 = np.empty((N, NM), dtype=np.complex128)
    LP1 = np.empty((N, NM), dtype=np.complex128)
    RP0 = np.empty((NM, N), dtype=np.complex128)
    RP1 = np.empty((NM, N), dtype=np.complex128)
    contour_integral(LP0=LP0,
                     LP1=LP1,
                     RP0=RP0,
                     RP1=RP1,
                     M00=M00,
                     M01=M01,
                     M10=M10,
                     Y=Y,
                     R=R,
                     is_left=np.bool_(type == 'L'),
                     N=N,
                     NM=NM)

    if validate:

        theta = np.linspace(theta_min, theta_max, NT)
        dtheta = np.hstack((theta[1] - theta[0], theta[2:] - theta[:-2], theta[-1] - theta[-2])) / 2

        c = 0
        r1 = 3.0
        r2 = 1 / R

        zC1 = c + r1 * np.exp(1j * theta)
        zC2 = c + r2 * np.exp(1j * theta)
        z = np.hstack((zC1, zC2))

        dzC1_dtheta = 1j * r1 * np.exp(1j * theta)
        dzC2_dtheta = 1j * r2 * np.exp(1j * theta)
        dz_dtheta = np.hstack((dzC1_dtheta, -dzC2_dtheta))

        dtheta = np.hstack((dtheta, dtheta))

        P0 = np.zeros((N, N), dtype=np.complex128)
        P1 = np.zeros((N, N), dtype=np.complex128)

        if block:

            A = M00[:N // 2, :N // 2]
            B = M00[:N // 2, N // 2:]
            C = M00[N // 2:, :N // 2]
            D = M00[N // 2:, N // 2:]

            iA = np.linalg.inv(A)
            iD = np.linalg.inv(D)

            iT = np.empty((N, N), dtype=np.complex128)

            for I in range(len(z)):

                if type == 'L':
                    Bi = B + M10[:N // 2, N // 2:] * z[I]
                    Ci = C + M01[N // 2:, :N // 2] / z[I]
                else:
                    Bi = B + M10[:N // 2, N // 2:] / z[I]
                    Ci = C + M01[N // 2:, :N // 2] * z[I]

                T0 = np.linalg.inv(A - Bi @ iD @ Ci)
                T1 = np.linalg.inv(D - Ci @ iA @ Bi)

                iT[:N // 2, :N // 2] = T0
                iT[:N // 2, N // 2:] = -iA @ Bi @ T1
                iT[N // 2:, :N // 2] = -iD @ Ci @ T0
                iT[N // 2:, N // 2:] = T1

                P0 += iT * dz_dtheta[I] * dtheta[I] / (2 * np.pi * 1j)
                P1 += iT * z[I] * dz_dtheta[I] * dtheta[I] / (2 * np.pi * 1j)

        else:

            for I in range(len(z)):

                if type == 'L':
                    T = M00 + M01 / z[I] + M10 * z[I]
                else:
                    T = M00 + M01 * z[I] + M10 / z[I]

                iT = np.linalg.inv(T)

                P0 += iT * dz_dtheta[I] * dtheta[I] / (2 * np.pi * 1j)
                P1 += iT * z[I] * dz_dtheta[I] * dtheta[I] / (2 * np.pi * 1j)

        LP0_ref = P0 @ Y
        LP1_ref = P1 @ Y

        RP0_ref = Y.T @ P0
        RP1_ref = Y.T @ P1

        error = False
        for n, val, ref in zip(['LP0', 'LP1', 'RP0', 'RP1'], [LP0, LP1, RP0, RP1],
                               [LP0_ref, LP1_ref, RP0_ref, RP1_ref]):
            if not np.allclose(val, ref):
                error = True
                print(f"Validation failed for {n} with relerror {np.linalg.norm(val-ref)/np.linalg.norm(ref)}!")
        # if not error:
        #     print("Validation successful!")

    # ctime += time.perf_counter()
    # print(f'Time for contour integral (DaCe): {ctime} s')

    # ctime = - time.perf_counter()

    LV, LS, LW = svd(LP0, full_matrices=False)
    Lind = np.where(abs(np.diag(LS)) > eps_lim)[0]

    RV, RS, RW = svd(RP0, full_matrices=True)
    Rind = np.where(abs(np.diag(RS)) > eps_lim)[0]

    # ctime += time.perf_counter()
    # print(f'Time for SVD (DaCe): {ctime} s')

    if len(Lind) == 0:

        cond = np.nan
        ksurf = None
        Sigma = None
        gR = None

    else:

        # ctime = - time.perf_counter()

        LV = LV[:, Lind]
        LS = LS[Lind]
        LW = np.conj(LW).T[:, Lind]

        Llambda, Lu = np.linalg.eig(np.conj(LV).T @ LP1 @ LW @ np.linalg.inv(np.diag(LS)))
        #Llambda = np.diag(Llambda)
        phiL = LV @ Lu

        RV = RV[:, Rind]
        RS = RS[Rind]
        RW = np.conj(RW).T[:, Rind]

        Rlambda, Ru = np.linalg.eig(np.linalg.inv(np.diag(RS)) @ np.conj(RV).T @ RP1 @ RW)
        #Rlambda = np.diag(Rlambda)
        phiR = np.linalg.solve(Ru, np.conj(RW).T)

        if type == 'L':
            kL = 1j * np.log(Llambda)
            kR = 1j * np.log(Rlambda)
        else:
            kL = -1j * np.log(Llambda)
            kR = -1j * np.log(Rlambda)

        ind_sort_kL = np.argsort(abs(np.imag(kL)))
        k = kL[ind_sort_kL]
        phiL = phiL[:, ind_sort_kL].copy()

        # ctime += time.perf_counter()
        # print(f'Time for eigenvalue problem (DaCe): {ctime} s')

        # ksurf, Vsurf, dEk_dk, Nref, gR, Sigma, Vs = calc_sigma(
        #     k=k, kR=kR, phiL=phiL, phiR=phiR, M00=M00, M01=M01, M10=M10, imag_lim=imag_lim,
        #     is_left=type=='L', is_W=function=='W', ref_iteration=ref_iteration,
        #     N=N, NM=NM)
        # ksurf = ksurf[:Nref[0]]
        # Vsurf = Vsurf[:,:Nref[0]]

        # assert np.allclose(Vs, Vsurf)

        # ctime = - time.perf_counter()

        if type == 'L':
            ksurf, Vsurf, dEk_dk = sort_k(k, kR, phiL, phiR, M01, M10, imag_lim, 1.0)
            gR = Vsurf @ np.linalg.inv(Vsurf.T @ M00 @ Vsurf +
                                       Vsurf.T @ M10 @ Vsurf @ np.diag(np.exp(-1j * ksurf))) @ Vsurf.T
            for IC in range(ref_iteration):
                gR = np.linalg.inv(M00 - M10 @ gR @ M01)
            if (np.imag(np.trace(gR)) > 0 and function == 'G'):
                ksurf, Vsurf, dEk_dk = sort_k(k, kR, phiL, phiR, M01, M10, 0.5, 1.0)
                gR = Vsurf @ np.linalg.inv(Vsurf.T @ M00 @ Vsurf +
                                           Vsurf.T @ M10 @ Vsurf @ np.diag(np.exp(-1j * ksurf))) @ Vsurf.T
                for IC in range(ref_iteration):
                    gR = np.linalg.inv(M00 - M10 @ gR @ M01)
            Sigma = M10 @ gR @ M01
        else:
            ksurf, Vsurf, dEk_dk = sort_k(k, kR, phiL, phiR, M01, M10, imag_lim, -1.0)
            gR = Vsurf @ np.linalg.inv(Vsurf.T @ M00 @ Vsurf +
                                       Vsurf.T @ M01 @ Vsurf @ np.diag(np.exp(1j * ksurf))) @ Vsurf.T
            for IC in range(ref_iteration):
                gR = np.linalg.inv(M00 - M01 @ gR @ M10)
            if (np.imag(np.trace(gR)) > 0 and function == 'G'):
                ksurf, Vsurf, dEk_dk = sort_k(k, kR, phiL, phiR, M01, M10, 0.5, -1.0)
                gR = Vsurf @ np.linalg.inv(Vsurf.T @ M00 @ Vsurf +
                                           Vsurf.T @ M01 @ Vsurf @ np.diag(np.exp(1j * ksurf))) @ Vsurf.T
                for IC in range(ref_iteration):
                    gR = np.linalg.inv(M00 - M01 @ gR @ M10)
            Sigma = M01 @ gR @ M10

        # ksurf_buf = np.empty((NM,), dtype=np.complex128)
        # Vsurf_buf = np.empty((N, NM), dtype=np.complex128)
        # dEk_dk = np.empty((NM,), dtype=np.complex128)
        # Nref = np.empty((1,), dtype=np.int32)

        # if type == 'L':

        #     # ksurf, Vsurf, dEk_dk, Nref = sortk(k=k, kR=kR, phiL=phiL, phiR=phiR, M01=M01, M10=M10, imag_limit=imag_lim, factor=1.0, N=N, NM=NM)
        #     sortk(ksurf=ksurf_buf, Vsurf=Vsurf_buf, dEk_dk=dEk_dk, out_Nref=Nref,
        #           k=k, kR=kR, phiL=phiL, phiR=phiR, M01=M01, M10=M10, imag_limit=imag_lim, factor=1.0, N=N, NM=NM)
        #     ksurf = ksurf_buf[0:Nref[0]]
        #     Vsurf = Vsurf_buf[:,0:Nref[0]]

        #     gR = Vsurf @ np.linalg.inv(Vsurf.T @ M00 @ Vsurf + Vsurf.T @ M10 @ Vsurf @ np.diag(np.exp(-1j * ksurf))) @ Vsurf.T
        #     for IC in range(ref_iteration):
        #         gR = np.linalg.inv(M00 - M10 @ gR @ M01)

        #     if(np.imag(np.trace(gR)) > 0 and function == 'G'):

        #         sortk(ksurf=ksurf_buf, Vsurf=Vsurf_buf, dEk_dk=dEk_dk, out_Nref=Nref,
        #               k=k, kR=kR, phiL=phiL, phiR=phiR, M01=M01, M10=M10, imag_limit=0.5, factor=1.0, N=N, NM=NM)
        #         ksurf = ksurf_buf[0:Nref[0]]
        #         Vsurf = Vsurf_buf[:,0:Nref[0]]

        #         gR = Vsurf @ np.linalg.inv(Vsurf.T @ M00 @ Vsurf + Vsurf.T @ M10 @ Vsurf @ np.diag(np.exp(-1j * ksurf))) @ Vsurf.T
        #         for IC in range(ref_iteration):
        #             gR = np.linalg.inv(M00 - M10 @ gR @ M01)

        #     Sigma = M10 @ gR @ M01
        # else:

        #     sortk(ksurf=ksurf_buf, Vsurf=Vsurf_buf, dEk_dk=dEk_dk, out_Nref=Nref,
        #           k=k, kR=kR, phiL=phiL, phiR=phiR, M01=M01, M10=M10, imag_limit=imag_lim, factor=-1.0, N=N, NM=NM)
        #     ksurf = ksurf_buf[0:Nref[0]]
        #     Vsurf = Vsurf_buf[:,0:Nref[0]]

        #     gR = Vsurf @ np.linalg.inv(Vsurf.T @ M00 @ Vsurf + Vsurf.T @ M01 @ Vsurf @ np.diag(np.exp(1j * ksurf))) @ Vsurf.T
        #     for IC in range(ref_iteration):
        #         gR = np.linalg.inv(M00 - M01 @ gR @ M10)

        #     if(np.imag(np.trace(gR)) > 0 and function == 'G'):

        #         sortk(ksurf=ksurf_buf, Vsurf=Vsurf_buf, dEk_dk=dEk_dk, out_Nref=Nref,
        #               k=k, kR=kR, phiL=phiL, phiR=phiR, M01=M01, M10=M10, imag_limit=0.5, factor=-1.0, N=N, NM=NM)
        #         ksurf = ksurf_buf[0:Nref[0]]
        #         Vsurf = Vsurf_buf[:,0:Nref[0]]

        #         gR = Vsurf @ np.linalg.inv(Vsurf.T @ M00 @ Vsurf + Vsurf.T @ M01 @ Vsurf @ np.diag(np.exp(1j * ksurf))) @ Vsurf.T
        #         for IC in range(ref_iteration):
        #             gR = np.linalg.inv(M00 - M01 @ gR @ M10)

        #     Sigma = M01 @ gR @ M10

        # ctime += time.perf_counter()
        # print(f'Time for Sigma (DaCe): {ctime} s')

        ind = np.where(abs(dEk_dk))
        if len(ind[0]) > 0:
            min_dEk = np.min(abs(dEk_dk[ind]))
    return ksurf, cond, gR, Sigma, min_dEk


@dace.program
def contour_integral_dace(LP0: dace.complex128[N, NM], LP1: dace.complex128[N, NM], RP0: dace.complex128[NM, N],
                          RP1: dace.complex128[NM, N], M00: dace.complex128[N, N], M01: dace.complex128[N, N],
                          M10: dace.complex128[N, N], Y: dace.complex128[N, NM], R: dace.float64, is_left: dace.bool):

    theta = np.linspace(theta_min, theta_max, NT)
    dtheta1 = np.hstack((theta[1] - theta[0], theta[2:] - theta[:-2], theta[-1] - theta[-2])) / 2

    c = 0
    r1 = 3.0
    r2 = 1 / R

    zC1 = c + r1 * np.exp(1j * theta)
    zC2 = c + r2 * np.exp(1j * theta)
    z = np.hstack((zC1, zC2))

    dzC1_dtheta = 1j * r1 * np.exp(1j * theta)
    dzC2_dtheta = 1j * r2 * np.exp(1j * theta)
    dz_dtheta = np.hstack((dzC1_dtheta, -dzC2_dtheta))

    dtheta = np.hstack((dtheta1, dtheta1))

    # preY : dace.float64[N, NM] = rng.random((N, NM))
    # Y = np.complex128(preY)
    # Y = np.complex128(rng.random((N, NM)))

    P0 = np.zeros((N, N), dtype=np.complex128)
    P1 = np.zeros((N, N), dtype=np.complex128)

    for I in range(len(z)):

        # if type == 'L':
        if is_left:
            T = M00 + M01 / z[I] + M10 * z[I]
        else:
            T = M00 + M01 * z[I] + M10 / z[I]

        iT = np.linalg.inv(T)

        P0 += iT * dz_dtheta[I] * dtheta[I] / (2 * np.pi * 1j)
        P1 += iT * z[I] * dz_dtheta[I] * dtheta[I] / (2 * np.pi * 1j)

    LP0[:] = P0 @ Y
    LP1[:] = P1 @ Y

    RP0[:] = Y.T @ P0
    RP1[:] = Y.T @ P1

    # return LP0, LP1, RP0, RP1


@dace.program
def contour_integral_block_dace(LP0: dace.complex128[N, NM], LP1: dace.complex128[N, NM], RP0: dace.complex128[NM, N],
                                RP1: dace.complex128[NM, N], M00: dace.complex128[N, N], M01: dace.complex128[N, N],
                                M10: dace.complex128[N,
                                                     N], Y: dace.complex128[N,
                                                                            NM], R: dace.float64, is_left: dace.bool):

    theta = np.linspace(theta_min, theta_max, NT)
    dtheta1 = np.hstack((theta[1] - theta[0], theta[2:] - theta[:-2], theta[-1] - theta[-2])) / 2

    c = 0
    r1 = 3.0
    r2 = 1 / R

    zC1 = c + r1 * np.exp(1j * theta)
    zC2 = c + r2 * np.exp(1j * theta)
    z = np.hstack((zC1, zC2))

    dzC1_dtheta = 1j * r1 * np.exp(1j * theta)
    dzC2_dtheta = 1j * r2 * np.exp(1j * theta)
    dz_dtheta = np.hstack((dzC1_dtheta, -dzC2_dtheta))

    dtheta = np.hstack((dtheta1, dtheta1))

    # preY : dace.float64[N, NM] = rng.random((N, NM))
    # Y = np.complex128(preY)
    # Y = np.complex128(rng.random((N, NM)))

    P0 = np.zeros((N, N), dtype=np.complex128)
    P1 = np.zeros((N, N), dtype=np.complex128)

    A = M00[:N // 2, :N // 2]
    B = M00[:N // 2, N // 2:]
    C = M00[N // 2:, :N // 2]
    D = M00[N // 2:, N // 2:]

    iA = np.linalg.inv(A)
    iD = np.linalg.inv(D)

    iT = np.empty((N, N), dtype=np.complex128)

    for I in range(len(z)):

        if is_left:
            Bi = B + M10[:N // 2, N // 2:] * z[I]
            Ci = C + M01[N // 2:, :N // 2] / z[I]
        else:
            Bi = B + M10[:N // 2, N // 2:] / z[I]
            Ci = C + M01[N // 2:, :N // 2] * z[I]

        T0 = np.linalg.inv(A - Bi @ iD @ Ci)
        T1 = np.linalg.inv(D - Ci @ iA @ Bi)

        iT[:N // 2, :N // 2] = T0
        iT[:N // 2, N // 2:] = -iA @ Bi @ T1
        iT[N // 2:, :N // 2] = -iD @ Ci @ T0
        iT[N // 2:, N // 2:] = T1

        P0 += iT * dz_dtheta[I] * dtheta[I] / (2 * np.pi * 1j)
        P1 += iT * z[I] * dz_dtheta[I] * dtheta[I] / (2 * np.pi * 1j)

    LP0[:] = P0 @ Y
    LP1[:] = P1 @ Y

    RP0[:] = Y.T @ P0
    RP1[:] = Y.T @ P1


S = dace.symbol('S')


@dace.program
def diag(d: dace.complex128[S]):
    out = np.zeros((S, S), dtype=d.dtype)
    for i in dace.map[0:S]:
        out[i, i] = d[i]
    return out


@dace.program
def trace(d: dace.complex128[S, S]):
    out = np.zeros((1, ), dtype=d.dtype)
    for i in dace.map[0:S]:
        out[0] += d[i, i]
    return out


@dace.program
def calc_sigma(k: dace.complex128[NM], kR: dace.complex128[NM], phiL: dace.complex128[N, NM], phiR: dace.complex128[NM,
                                                                                                                    N],
               M00: dace.complex128[N, N], M01: dace.complex128[N, N], M10: dace.complex128[N, N],
               imag_lim: dace.float64, is_left: dace.bool, is_W: dace.bool, ref_iteration: int):

    if is_left:
        factor = 1.0
    else:
        factor = -1.0

    ksurf, Vsurf, dEk_dk, Nref = sort_k_dace(k, kR, phiL, phiR, M01, M10, imag_lim, factor)
    # ks = ksurf[:Nref[0]]
    # Vs = Vsurf[:,:Nref[0]]
    # ksurf[Nref[0]:] = 0
    # Vsurf[:, Nref[0]:] = 0
    # ks = ksurf
    # Vs = Vsurf
    Vs = np.copy(Vsurf[:, :Nref[0]])
    # gR = Vs @ np.linalg.inv(Vs.T @ M00 @ Vs + Vs.T @ M10 @ Vs @ diag(np.exp(-(1j * ks)))) @ Vs.T
    gR = Vs @ Vs.T
    # for IC in range(ref_iteration):
    #     gR[:] = np.linalg.inv(M00 - M10 @ gR @ M01)

    # if (np.imag(trace(gR)) > 0 and not is_W):
    #     ksurf[:], Vsurf[:], dEk_dk[:], Nref[:] = sort_k_dace(k, kR, phiL, phiR, M01, M10, 0.5, factor)
    #     ks1 = ksurf[:Nref[0]]
    #     Vs1 = Vsurf[:,:Nref[0]]
    #     gR[:] = Vs1 @ np.linalg.inv(Vs1.T @ M00 @ Vs1 + Vs1.T @ M10 @ Vs1 @ diag(np.exp(-(1j * ks1)))) @ Vs1.T
    #     for IC in range(ref_iteration):
    #         gR[:] = np.linalg.inv(M00 - M10 @ gR @ M01)

    if is_left:
        Sigma = M10 @ gR @ M01
    else:
        Sigma = M01 @ gR @ M10

    return ksurf, Vsurf, dEk_dk, Nref, gR, Sigma, Vs


def random_complex(shape, rng, dtype=np.float64):
    return (rng.random(shape, dtype) - 0.5) + 1j * (rng.random(shape, dtype) - 0.5)


def random_float(shape, rng, dtype=np.float64):
    return rng.random(shape, dtype) - 0.5


if __name__ == "__main__":

    N = 512
    NM = 256

    M00 = random_complex((N, N), rng)
    M01 = random_complex((N, N), rng)
    M10 = random_complex((N, N), rng)
    Y = random_complex((N, NM), rng)
    imag_lim = 1e-8
    R = 1e8

    # ksurf_ref, Vsurf_ref, cond_ref, gR_ref, Sigma_ref, min_dEk_ref = beyn(M00, M01, M10, imag_lim, R, Y)

    ci_sdfg = contour_integral_dace.to_sdfg(simplify=True)
    auto_optimize(ci_sdfg, dace.DeviceType.CPU)
    ci_func = ci_sdfg.compile()

    # cs_sdfg = calc_sigma.to_sdfg(simplify=True)
    # auto_optimize(cs_sdfg, dace.DeviceType.CPU)
    # cs_func = cs_sdfg.compile()

    sk_sdfg = sort_k_dace.to_sdfg(simplify=True)
    auto_optimize(sk_sdfg, dace.DeviceType.CPU)
    sk_func = sk_sdfg.compile()

    ksurf_ref, Vsurf_ref, cond_ref, gR_ref, Sigma_ref, min_dEk_ref = beyn(M00, M01, M10, imag_lim, R, Y)
    runtimes = repeat("beyn(M00, M01, M10, imag_lim, R, Y)", globals={**globals(), **locals()}, number=1, repeat=10)
    print(f"Reference beyn: {np.median(runtimes) * 1000} ms")

    ksurf_val, Vsurf_val, cond_val, gR_val, Sigma_val, min_dEk_val = beyn_dace(M00,
                                                                               M01,
                                                                               M10,
                                                                               imag_lim,
                                                                               R,
                                                                               Y,
                                                                               contour_integral=ci_func,
                                                                               sortk=sk_func)
    runtimes = repeat("beyn_dace(M00, M01, M10, imag_lim, R, Y, contour_integral=ci_func, sortk=sk_func)",
                      globals={
                          **globals(),
                          **locals()
                      },
                      number=1,
                      repeat=10)
    print(f"DaCe beyn: {np.median(runtimes) * 1000} ms")

    assert np.allclose(ksurf_ref, ksurf_val)
    assert np.allclose(Vsurf_ref, Vsurf_val)
    assert np.allclose(cond_ref, cond_val)
    assert np.allclose(gR_ref, gR_val)
    assert np.allclose(Sigma_ref, Sigma_val)
    assert np.allclose(min_dEk_ref, min_dEk_val)

    # k = random_complex((NM, ), rng)
    # kR = random_complex((NM, ), rng)
    # phiR = random_complex((NM, N), rng)
    # phiL = random_complex((N, NM), rng)
    # factor = rng.random(dtype=np.float64)

    # # imag_cond_ref, dEk_dk_ref = check_imag_cond(k, kR, phiR, phiL, M10, M01, imag_lim)
    # ksurf_ref, Vsurf_ref, dEk_dk_ref = sort_k(k, kR, phiL, phiR, M01, M10, imag_lim, factor)
    # # runtimes = repeat("check_imag_cond(k, kR, phiR, phiL, M10, M01, imag_lim)", globals=globals(), number=1, repeat=10)
    # # print(f"Reference check_imag_cond: {np.median(runtimes) * 1000} ms")

    # # sdfg = check_imag_cond_dace.to_sdfg(simplify=True)
    # sdfg = sort_k_dace.to_sdfg(simplify=True)
    # auto_optimize(sdfg, dace.DeviceType.CPU)
    # func = sdfg.compile()
    # # imag_cond_val, dEk_dk_val = func(k=k, kR=kR, phiR=phiR, phiL=phiL, M01=M01, M10=M10, max_imag=imag_lim, N=N, NM=NM)
    # ksurf_val, Vsurf_val, dEk_dk_val, Nref = func(k=k, kR=kR, phiR=phiR, phiL=phiL, M01=M01, M10=M10, imag_limit=imag_lim, factor=factor, N=N, NM=NM, )

    # # print(imag_cond_ref)
    # # print(imag_cond_val)

    # # assert np.allclose(imag_cond_ref, imag_cond_val)
    # assert np.allclose(ksurf_ref, ksurf_val[:Nref[0]])
    # assert np.allclose(Vsurf_ref, Vsurf_val[:, :Nref[0]])
    # print(np.linalg.norm(dEk_dk_ref - dEk_dk_val) / np.linalg.norm(dEk_dk_ref))
    # assert np.allclose(dEk_dk_ref, dEk_dk_val)

    # # sdfg = check_imag_cond_dace.to_sdfg(simplify=True)
    # # sdfg.name = f"{sdfg.label}_gpu"
    # # auto_optimize(sdfg, dace.DeviceType.GPU)
    # # func = sdfg.compile()
    # # imag_cond_val, dEk_dk_val = func(k=k, kR=kR, phiR=phiR, phiL=phiL, M01=M01, M10=M10, max_imag=imag_lim, N=N, NM=NM)

    # # assert np.allclose(imag_cond_ref, imag_cond_val)
    # # print(np.linalg.norm(dEk_dk_ref - dEk_dk_val) / np.linalg.norm(dEk_dk_ref))
    # # assert np.allclose(dEk_dk_ref, dEk_dk_val)
