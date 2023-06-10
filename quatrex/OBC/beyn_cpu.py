import numpy as np
import time

from scipy.sparse import lil_matrix, csr_matrix, csc_matrix, linalg
from scipy.linalg import svd
from numpy.linalg import eig
from utils.read_utils import read_file_to_float_ndarray

np.random.seed(0)

def beyn(M00, M01, M10, imag_lim, R, type, function = 'W', block: bool = False):
    
    #np.seterr(divide='ignore', invalid='ignore')

    # ctime = - time.perf_counter()

    theta_min = 0
    theta_max = 2*np.pi
    NT = 51
    eps_lim = 1e-8
    ref_iteration = 2                                
    cond = 0
    min_dEk = 1e8

    N = M00.shape[0]

    theta = np.linspace(theta_min, theta_max, NT)
    dtheta = np.hstack((theta[1]-theta[0], theta[2:]-theta[:-2], theta[-1]-theta[-2]))/2

    c = 0
    r1 = 3.0
    r2 = 1/R

    zC1 = c + r1*np.exp(1j*theta)
    zC2 = c + r2*np.exp(1j*theta)
    z = np.hstack((zC1, zC2))

    dzC1_dtheta = 1j*r1*np.exp(1j*theta)
    dzC2_dtheta = 1j*r2*np.exp(1j*theta)
    dz_dtheta = np.hstack((dzC1_dtheta, -dzC2_dtheta))

    dtheta = np.hstack((dtheta, dtheta))

    if N < 100:
        NM = round(3*N/4)
    else:
        NM = round(N/2)
    
    # ctime += time.perf_counter()
    # print(f'Time for setting up the parameters: {ctime} s')

    # ctime = - time.perf_counter()

    Y = np.random.rand(N, NM)
    #Y = np.loadtxt('CNT_newwannier/' + 'ymatrix.dat')

    P0 = np.zeros((N, N), dtype=np.complex128)
    P1 = np.zeros((N, N), dtype=np.complex128)

    if block:

        A = M00[:N//2, :N//2]
        B = M00[:N//2, N//2:]
        C = M00[N//2:, :N//2]
        D = M00[N//2:, N//2:]

        iA = np.linalg.inv(A)
        iD = np.linalg.inv(D)

        iT = np.empty((N, N), dtype=np.complex128)

        for I in range(len(z)):

            if type == 'L':
                Bi = B + M10[:N//2, N//2:]*z[I]
                Ci = C + M01[N//2:, :N//2]/z[I]
            else:
                Bi = B + M10[:N//2, N//2:]/z[I]
                Ci = C + M01[N//2:, :N//2]*z[I]

            T0 = np.linalg.inv(A - Bi@iD@Ci)
            T1 = np.linalg.inv(D - Ci@iA@Bi)

            iT[:N//2, :N//2] = T0
            iT[:N//2, N//2:] = -iA@Bi@T1
            iT[N//2:, :N//2] = -iD@Ci@T0
            iT[N//2:, N//2:] = T1

            P0 += iT*dz_dtheta[I]*dtheta[I]/(2*np.pi*1j)
            P1 += iT*z[I]*dz_dtheta[I]*dtheta[I]/(2*np.pi*1j)
    
    else:

        for I in range(len(z)):

            if type == 'L':
                T = M00 + M01/z[I] + M10*z[I]
            else:
                T = M00 + M01*z[I] + M10/z[I]

            iT = np.linalg.inv(T)

            P0 += iT*dz_dtheta[I]*dtheta[I]/(2*np.pi*1j)
            P1 += iT*z[I]*dz_dtheta[I]*dtheta[I]/(2*np.pi*1j)

    LP0 = P0@Y
    LP1 = P1@Y

    RP0 = Y.T@P0
    RP1 = Y.T@P1

    # ctime += time.perf_counter()
    # print(f'Time for contour integral: {ctime} s')

    # ctime = - time.perf_counter()

    LV, LS, LW = svd(LP0, full_matrices=False)
    Lind = np.where(abs(np.diag(LS)) > eps_lim)[0]

    RV, RS, RW = svd(RP0, full_matrices=True)
    Rind = np.where(abs(np.diag(RS)) > eps_lim)[0]

    # ctime += time.perf_counter()
    # print(f'Time for SVD: {ctime} s')

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

        Llambda, Lu = eig(np.conj(LV).T@LP1@LW@np.linalg.inv(np.diag(LS)))
        #Llambda = np.diag(Llambda)
        phiL = LV@Lu

        RV = RV[:, Rind]
        RS = RS[Rind]
        RW = np.conj(RW).T[:, Rind]

        Rlambda, Ru = eig(np.linalg.inv(np.diag(RS))@np.conj(RV).T@RP1@RW)
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
        phiL = phiL[:, ind_sort_kL]

        # ctime += time.perf_counter()
        # print(f'Time for eigenvalue problem: {ctime} s')

        # ctime = - time.perf_counter()

        if type == 'L':
            ksurf, Vsurf, dEk_dk = sort_k(k, kR, phiL, phiR, M01, M10, imag_lim, 1.0)
            gR = Vsurf @ np.linalg.inv(Vsurf.T @ M00 @ Vsurf + Vsurf.T @ M10 @ Vsurf @ np.diag(np.exp(-1j * ksurf))) @ Vsurf.T
            for IC in range(ref_iteration):
                gR = np.linalg.inv(M00 - M10 @ gR @ M01)
            if(np.imag(np.trace(gR)) > 0 and function == 'G'):
                    ksurf, Vsurf, dEk_dk = sort_k(k, kR, phiL, phiR, M01, M10, 0.5, 1.0)
                    gR = Vsurf @ np.linalg.inv(Vsurf.T @ M00 @ Vsurf + Vsurf.T @ M10 @ Vsurf @ np.diag(np.exp(-1j * ksurf))) @ Vsurf.T
                    for IC in range(ref_iteration):
                        gR = np.linalg.inv(M00 - M10 @ gR @ M01)
            Sigma = M10 @ gR @ M01
        else:
            ksurf, Vsurf, dEk_dk = sort_k(k, kR, phiL, phiR, M01, M10, imag_lim, -1.0)
            gR = Vsurf @ np.linalg.inv(Vsurf.T @ M00 @ Vsurf + Vsurf.T @ M01 @ Vsurf @ np.diag(np.exp(1j * ksurf))) @ Vsurf.T
            for IC in range(ref_iteration):
                gR = np.linalg.inv(M00 - M01 @ gR @ M10)
            if(np.imag(np.trace(gR)) > 0 and function == 'G'):
                    ksurf, Vsurf, dEk_dk = sort_k(k, kR, phiL, phiR, M01, M10, 0.5, -1.0)
                    gR = Vsurf @ np.linalg.inv(Vsurf.T @ M00 @ Vsurf + Vsurf.T @ M01 @ Vsurf @ np.diag(np.exp(1j * ksurf))) @ Vsurf.T
                    for IC in range(ref_iteration):
                        gR = np.linalg.inv(M00 - M01 @ gR @ M10)
            Sigma = M01 @ gR @ M10
        
        # ctime += time.perf_counter()
        # print(f'Time for Sigma: {ctime} s')

        ind = np.where(abs(dEk_dk))
        if len(ind[0]) > 0:
            min_dEk = np.min(abs(dEk_dk[ind]))
    return ksurf, cond, gR, Sigma, min_dEk

def check_imag_cond(k, kR, phiR, phiL, M10, M01, max_imag):
    imag_cond = np.zeros(len(k))
    dEk_dk = np.zeros(len(k), dtype =  np.cfloat)

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
    NT = len(phiL[:,0])

    ksurf = np.zeros(Nk, dtype = np.cfloat)
    Vsurf = np.zeros((NT, Nk), dtype = np.cfloat)

    imag_cond, dEk_dk = check_imag_cond(k, kR, phiR, phiL, M10, M01, imag_limit)

    Nref = 0

    for Ik in range(Nk):

        if not imag_cond[Ik] and abs(np.imag(k[Ik])) > 1e-6:

            if factor*np.imag(k[Ik]) < 0:

                Nref += 1

                ksurf[Nref-1] = k[Ik]
                Vsurf[:,Nref-1] = phiL[:,Ik]

        else:
        
            cond1 = (abs(np.real(dEk_dk[Ik])) < abs(np.imag(dEk_dk[Ik]))/100) and (factor*np.imag(k[Ik]) < 0)
            cond2 = (abs(np.real(dEk_dk[Ik])) >= abs(np.imag(dEk_dk[Ik]))/100) and (factor*np.real(dEk_dk[Ik]) < 0)

            if cond1 or cond2:

                Nref += 1

                ksurf[Nref-1] = k[Ik]
                Vsurf[:,Nref-1] = phiL[:,Ik]
    ksurf = ksurf[0:Nref]
    Vsurf = Vsurf[:,0:Nref]
    return ksurf, Vsurf, dEk_dk

def beyn_old(M00, M01, M10, imag_lim, R, type):
    """
    Old wrong version of beyn function.
    Used for a wrong reference test.

    Args:
        M00 (_type_): _description_
        M01 (_type_): _description_
        M10 (_type_): _description_
        imag_lim (_type_): _description_
        R (_type_): _description_
        type (_type_): _description_

    Returns:
        _type_: _description_
    """
    
    #np.seterr(divide='ignore', invalid='ignore')

    theta_min = 0
    theta_max = 2*np.pi
    NT = 51
    eps_lim = 1e-8
    ref_iteration = 2                                   
    cond = 0
    min_dEk = 1e8

    N = M00.shape[0]

    theta = np.linspace(theta_min, theta_max, NT)
    dtheta = np.hstack((theta[1]-theta[0], theta[2:]-theta[:-2], theta[-1]-theta[-2]))/2

    c = 0
    r1 = 3.0
    r2 = 1/R

    zC1 = c + r1*np.exp(1j*theta)
    zC2 = c + r2*np.exp(1j*theta)
    z = np.hstack((zC1, zC2))

    dzC1_dtheta = 1j*r1*np.exp(1j*theta)
    dzC2_dtheta = 1j*r2*np.exp(1j*theta)
    dz_dtheta = np.hstack((dzC1_dtheta, -dzC2_dtheta))

    dtheta = np.hstack((dtheta, dtheta))

    if N < 100:
        NM = round(3*N/4)
    else:
        NM = round(N/2)

    Y = np.random.rand(N, NM)
    #Y = np.loadtxt('CNT_newwannier/' + 'ymatrix.dat')

    P0 = np.zeros((N, N), dtype=np.complex)
    P1 = np.zeros((N, N), dtype=np.complex)

    # ztmp = z.reshape(-1,1,1)
    # dz_dthetatmp = dz_dtheta.reshape(-1,1,1)
    # dthetatmp = dtheta.reshape(-1,1,1)
    # tmpx = dz_dthetatmp*dthetatmp
    # tmpy = ztmp*dz_dthetatmp*dthetatmp
    # if type == 'L':
    #     T = M00 + 1/ztmp*M01 + ztmp*M10
    # else:
    #     T = M00 + ztmp*M01 + 1/ztmp*M10
    # iT = np.linalg.inv(T)
    # P0 = np.sum(iT*tmpx, axis = 0)/(2*np.pi*1j)
    # P1 = np.sum(iT*tmpy, axis = 0)/(2*np.pi*1j)

    for I in range(len(z)):

        if type == 'L':
            T = M00 + M01/z[I] + M10*z[I]
        else:
            T = M00 + M01*z[I] + M10/z[I]

        iT = np.linalg.inv(T)

        P0 += iT*dz_dtheta[I]*dtheta[I]/(2*np.pi*1j)
        P1 += iT*z[I]*dz_dtheta[I]*dtheta[I]/(2*np.pi*1j)

    LP0 = P0@Y
    LP1 = P1@Y

    RP0 = Y.T@P0
    RP1 = Y.T@P1

    LV, LS, LW = svd(LP0, full_matrices=False)
    Lind = np.count_nonzero(abs(LS) > eps_lim)
    # Lind = np.where(abs(np.diag(LS)) > eps_lim)[0]

    RV, RS, RW = svd(RP0, full_matrices=True)
    Rind = np.count_nonzero(abs(RS) > eps_lim)
    # Rind = np.where(abs(np.diag(RS)) > eps_lim)[0]

    if Lind == 0:

        cond = np.nan
        ksurf = None
        Sigma = None
        gR = None

    else:

        LV = LV[:, :Lind]
        LS = LS[:Lind]
        LW = np.conj(LW).T[:, :Lind]

        Llambda, Lu = eig(np.conj(LV).T@LP1@LW@np.linalg.inv(np.diag(LS)))
        #Llambda = np.diag(Llambda)
        phiL = LV@Lu

        RV = RV[:, :Rind]
        RS = RS[:Rind]
        RW = np.conj(RW).T[:, :Rind]

        Rlambda, Ru = eig(np.linalg.inv(np.diag(RS))@np.conj(RV).T@RP1@RW)
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
        phiL = phiL[:, ind_sort_kL]

        if type == 'L':
            ksurf, Vsurf, dEk_dk = sort_k(k, kR, phiL, phiR, M01, M10, imag_lim, 1.0)
            gR = Vsurf @ np.linalg.inv(Vsurf.T @ M00 @ Vsurf + Vsurf.T @ M10 @ Vsurf @ np.diag(np.exp(-1j * ksurf))) @ Vsurf.T
            for IC in range(ref_iteration):
                gR = np.linalg.inv(M00 - M10 @ gR @ M01)
            Sigma = M10 @ gR @ M01
        else:
            ksurf, Vsurf, dEk_dk = sort_k(k, kR, phiL, phiR, M01, M10, imag_lim, -1.0)
            gR = Vsurf @ np.linalg.inv(Vsurf.T @ M00 @ Vsurf + Vsurf.T @ M01 @ Vsurf @ np.diag(np.exp(1j * ksurf))) @ Vsurf.T
            for IC in range(ref_iteration):
                gR = np.linalg.inv(M00 - M01 @ gR @ M10)
            Sigma = M01 @ gR @ M10

        ind = np.where(abs(dEk_dk))
        if len(ind[0]) > 0:
            min_dEk = np.min(abs(dEk_dk[ind]))
    return ksurf, cond, gR, Sigma, min_dEk
