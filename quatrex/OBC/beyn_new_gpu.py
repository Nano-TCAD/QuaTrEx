# Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.
import cupy as cp
import cupyx as cpx
import numpy as np
import time

from scipy.linalg import svd
from numpy.linalg import eig
from scipy.sparse import csr_matrix


from quatrex.OBC.contour_integral import contour_integral_gpu as ci_gpu_internal, contour_integral_batched_gpu as ci_batched_gpu_internal
from quatrex.OBC.contour_integral import contour_integral_batched_combo_gpu as ci_batched_combo_gpu_internal

compute_theta_kernel = cp.RawKernel(r'''
#include <cupy/complex.cuh>
extern "C" __global__
void compute_theta(double* dtheta, complex<double>* dz_dtheta, complex<double>* z, double theta_min, double theta_max, long long NT, long long factor, double R) {
                                                
    int idx = blockDim.x*blockIdx.x + threadIdx.x;
                                                
    if (idx <  2 * NT) {

        double step = (theta_max - theta_min) / (NT - 1);
        double theta = theta_min + (idx % NT) * step;
        double dth = step;
        if (idx == 0 || idx == NT - 1 || idx == NT || idx == 2 * NT - 1) {
            dth = dth / 2;
        }

        complex<double> exp_theta = exp(complex<double>(0.0, theta));
        complex<double> zC;
        complex<double> dzC_dth;
        double r;
        complex<double> i(0.0, 1.0);
        if (idx < NT) {
            r = pow(3.0, 1.0 / factor);
            zC = r * exp_theta;
            dzC_dth = i * r * exp_theta;
        } else {
            r = pow(1.0 / R, 1.0 / factor);
            zC = r * exp_theta;
            dzC_dth = - i * r * exp_theta;
        }
                     
        dtheta[idx] = dth;
        dz_dtheta[idx] = dzC_dth;
        z[idx] = zC;                     
    }
}''', "compute_theta")


# @cpx.jit.rawkernel()
# def compute_theta(dtheta, dz_dtheta, z, theta_min, theta_max, NT, factor, R):

#     idx = cpx.jit.blockIdx.x * cpx.jit.blockDim.x + cpx.jit.threadIdx.x
#     if idx < NT * 2:

#         step = (theta_max - theta_min) / (NT - 1)
#         theta = theta_min + (idx % NT) * step
#         dth = step
#         if idx == 0 or idx == NT - 1 or idx == NT or idx == 2 * NT - 1:
#             dth = dth / 2
        
#         c = 0
#         if idx < NT:
#             r = cp.power(3.0, 1.0 / factor)
#         else:
#             r = cp.power(1.0 / R, 1.0 / factor)
        
#         zC = c + r * cp.exp(1j * theta)
#         dzC_dth = 1j * r * cp.exp(1j * theta)

#         dtheta[idx] = dth
#         if idx < NT:
#             dz_dtheta[idx] = dzC_dth
#         else:
#             dz_dtheta[idx] = -dzC_dth
#         z[idx] = zC


@cpx.jit.rawkernel()
def contour(T, matrix_blocks, z, factor, z_size, b_size, isL):

    idx = cpx.jit.blockIdx.x * cpx.jit.blockDim.x + cpx.jit.threadIdx.x
    if idx < z_size * b_size * b_size:

        i = idx // (b_size * b_size)
        jk = idx % (b_size * b_size)
        j = jk // b_size
        k = jk % b_size
        t_idx = i * b_size * b_size + j * b_size + k

        z_i = z[i]

        if isL:
            for l in range(2 * factor + 1):
                m_idx = l * b_size * b_size + j * b_size + k
                T[t_idx] += matrix_blocks[m_idx] * z_i ** (factor - l)
        else:
            for l in range(2 * factor + 1):
                m_idx = l * b_size * b_size + j * b_size + k
                T[t_idx] += matrix_blocks[m_idx] * z_i ** (l - factor)


@cpx.jit.rawkernel()
def compute_dEk_dk(dEk_dk, kL, kR, phiL, phiR, pRpL, M01, M10, ind_k, Ikmax, N, isL):

    idx = cpx.jit.blockIdx.x * cpx.jit.blockDim.x + cpx.jit.threadIdx.x
    if idx < Ikmax * N * N:

        i = idx // (N * N)
        jk = idx % (N * N)
        j = jk // N
        k = jk % N

        ikr = ind_k[i]
        
        # i_j = i * N + j
        # ikr_j = ikr * N + j
        # k_i = k * N + i
        # k_ikr = k * N + ikr
        # i_ikr = i * N + ikr
        # ikr_i = ikr * N + i

        if isL:

            second_arg = -1j * M10[j, k] * cp.exp(-1j * kL[i]) + 1j * M01[j, k] * cp.exp(1j * kL[i])
            # dEk_dk[idx] = - (phiR[ikr_j] * second_arg * phiL[k_i]) / pRpL[ikr_i]
            dEk_dk[i, j, k] = - (phiR[ikr, j] * second_arg * phiL[k, i]) / pRpL[ikr, i]
        
        else:

            second_arg = -1j * M10[j, k] * cp.exp(-1j * kR[i]) + 1j * M01[j, k] * cp.exp(1j * kR[i])
            # dEk_dk[idx] = - phiR[i_j] * second_arg * phiL[k_ikr] / pRpL[i_ikr]
            dEk_dk[i, j, k] = - phiR[i, j] * second_arg * phiL[k, ikr] / pRpL[i, ikr]


compute_imag_cond_conditions_kernel = cp.RawKernel(r'''
#include <cupy/complex.cuh>
extern "C" __global__
void compute_imag_cond_conditions(bool* cond, complex<double>* k, complex<double>* dEk_dk, long long* ind_neighs, double max_imag, long long N) {
                                                
    int idx = blockDim.x*blockIdx.x + threadIdx.x;
                                                
    if (idx < N) {

        long long neigh_idx = ind_neighs[idx];
        complex<double> k1 = k[idx];
        complex<double> k2 = k[neigh_idx];
        complex<double> dEk1 = dEk_dk[idx];
        complex<double> dEk2 = dEk_dk[neigh_idx];  

        bool cond1 = abs(dEk1 + dEk2) / (abs(dEk1) + 1e-10) < 0.25;
        bool cond2 = abs(k1 + k2) / (abs(k1) + 1e-10) < 0.25;
        bool cond3 = (abs(k1.imag()) + abs(k2.imag())) / 2.0 < 1.5 * max_imag;
        bool cond4 = (k1.imag() / abs(k1.imag())) == (k2.imag() / abs(k2.imag()));

        cond[idx] = cond1 and cond2 and (cond3 or (not cond3 and cond4));                          
    }
}''', "compute_imag_cond_conditions")


# @cpx.jit.rawkernel()
# def compute_imag_cond_conditions(cond, k, dEk_dk, ind_neighs, max_imag, N):
        
#     idx = cpx.jit.blockIdx.x * cpx.jit.blockDim.x + cpx.jit.threadIdx.x
#     if idx < N:

#         neigh_idx = ind_neighs[idx]
#         k1 = k[idx]
#         k2 = k[neigh_idx]
#         dEk1 = dEk_dk[idx]
#         dEk2 = dEk_dk[neigh_idx]
#         k1_img = (k1 - cp.conj(k1)) / 2.0
#         k2_img = (k2 - cp.conj(k2)) / 2.0
#         k1_sgn = k1_img / cp.abs(k1_img)
#         k2_sgn = k2_img / cp.abs(k2_img)

#         cond1 = cp.abs(dEk1 + dEk2) / (cp.abs(dEk1) + 1e-10) < 0.25
#         cond2 = cp.abs(k1 + k2) / (cp.abs(k1) + 1e-10) < 0.25
#         cond3 = (cp.abs(k1_img) + cp.abs(k2_img)) / 2.0 < 1.5 * max_imag
#         cond4 = k1_sgn == k2_sgn

#         cond[idx] = cond1 and cond2 and (cond3 or (not cond3 and cond4))


compute_sort_k_conditions_kernel = cp.RawKernel(r'''
#include <cupy/complex.cuh>
extern "C" __global__
void compute_sort_k_conditions(bool* condA, bool* condB, complex<double>* kref, bool* imag_cond, complex<double>* dEk_dk, long long N, double factor) {
//void compute_sort_k_conditions(bool* cond1, bool* cond2, bool* cond3, bool* cond4, complex<double>* kref, bool* imag_cond, complex<double>* dEk_dk, long long N, double factor) {
                                                
    int idx = blockDim.x*blockIdx.x + threadIdx.x;
                                                
    if (idx < N) {
                                                
        complex<double> kr = kref[idx];
        complex<double> ek = dEk_dk[idx];
        bool icond = imag_cond[idx];
                                                
        bool cond1 = (abs(ek.real()) < (abs(ek.imag()) / 100.0)) and (factor * kr.imag() < 0.0);
        bool cond2 = (abs(ek.real()) >= (abs(ek.imag()) / 100.0)) and (factor * ek.real() < 0.0);
        bool cond3 = not icond and abs(kr.imag()) > 1e-6;
        bool cond4 = factor * kr.imag() < 0.0;
                                                
        //cond1[idx] = (abs(ek.real()) < (abs(ek.imag()) / 100.0)) and (factor * kr.imag() < 0.0);
        //cond2[idx] = (abs(ek.real()) >= (abs(ek.imag()) / 100.0)) and (factor * ek.real() < 0.0);
        //cond3[idx] = not icond and abs(kr.imag()) > 1e-6;
        //cond4[idx] = factor * kr.imag() < 0.0;
                                                
        condA[idx] = (cond3 and cond4) or (not cond3 and (cond1 or cond2));
        condB[idx] = not cond3 and (cond1 or cond2);
    }
}''', "compute_sort_k_conditions")


def extract_small_matrix_blocks_gpu(M00, M01, M10, factor, type, densify: bool = False):
    N = M00.shape[0] // factor
    num_blocks = 2 * factor + 1

    matrix_blocks = cp.empty((num_blocks, N, N), dtype=M00.dtype)

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

    m00 = cp.zeros((N * factor, N * factor), dtype=M00.dtype)
    m01 = cp.zeros((N * factor, N * factor), dtype=M01.dtype)
    m10 = cp.zeros((N * factor, N * factor), dtype=M10.dtype)

    for I in range(1, factor + 1):
        for J in range(1, factor + 1):
            m00[(I - 1) * N:I * N, (J - 1) * N:J * N] = matrix_blocks[factor - I + J]
            if I >= J:
                m01[(I - 1) * N:I * N, (J - 1) * N:J * N] = matrix_blocks[2 * factor - I + J]
            if I <= J:
                m10[(I - 1) * N:I * N, (J - 1) * N:J * N] = matrix_blocks[- I + J]

    if not densify:
        m00 = cp.sparse.csr_matrix(m00)
        m01 = cp.sparse.csr_matrix(m01)
        m10 = cp.sparse.csr_matrix(m10)

    return m00, m01, m10, matrix_blocks

def contour_integral_gpu(factor: int,
                         matrix_blocks: np.ndarray,
                         big_N: int,
                         R: float,
                         side: str,
                         YL=None,
                         YR=None):

    N = big_N // factor

    if factor * N < 100:
        NM = round(3 * N / 4)
    else:
        NM = round(N / 2)
    NM = factor * NM

    if YL is None:
        YL = np.random.rand(N, NM)
    if YR is None:
        YR = np.random.rand(NM, N)
    
    P0C1, P1C1 = ci_gpu_internal(N, factor, matrix_blocks, 3.0, 1.0, side)
    P0C2, P1C2 = ci_gpu_internal(N, factor, matrix_blocks, 1.0 / R, -1.0, side)

    P0 = P0C1 + P0C2
    P1 = P1C1 + P1C2

    LP0 = P0@YL
    LP1 = P1@YL

    RP0 = YR@P0
    RP1 = YR@P1

    return LP0, LP1, RP0, RP1

def contour_integral_batched_gpu(factor: int,
                                 matrix_blocks: np.ndarray,
                                 big_N: int,
                                 R: float,
                                 side: str,
                                 YL=None,
                                 YR=None):

    N = big_N // factor

    if factor * N < 100:
        NM = round(3 * N / 4)
    else:
        NM = round(N / 2)
    NM = factor * NM

    if YL is None:
        YL = np.random.rand(N, NM)
    if YR is None:
        YR = np.random.rand(NM, N)
    
    P0C1, P1C1 = ci_batched_gpu_internal(N, factor, matrix_blocks, 3.0, 1.0, side)
    P0C2, P1C2 = ci_batched_gpu_internal(N, factor, matrix_blocks, 1.0 / R, -1.0, side)

    P0 = P0C1 + P0C2
    P1 = P1C1 + P1C2

    LP0 = P0@YL
    LP1 = P1@YL

    RP0 = YR@P0
    RP1 = YR@P1

    return LP0, LP1, RP0, RP1

def beyn_svd_gpu(LP0, RP0, eps_lim=1e-8):

    LV, LS, LW = cp.linalg.svd(LP0, full_matrices=False)
    Lind = cp.where(cp.abs(LS) > eps_lim)[0]

    RV, RS, RW = cp.linalg.svd(RP0, full_matrices=False)
    Rind = cp.where(cp.abs(RS) > eps_lim)[0]

    LV = LV[:, Lind]
    LS = cp.diag(LS[Lind])
    LW = LW[Lind, :].conj().T

    RV = RV[:, Rind]
    RS = cp.diag(RS[Rind])
    RW = RW[Rind, :].conj().T

    return LV, LS, LW, RV, RS, RW


def contour_svd_gpu(factor: int,
                    matrix_blocks: np.ndarray,
                    big_N: int,
                    R: float,
                    side: str,
                    YL=None,
                    YR=None,
                    eps_lim=1e-8):

    N = big_N // factor

    if factor * N < 100:
        NM = round(3 * N / 4)
    else:
        NM = round(N / 2)
    NM = factor * NM

    if YL is None:
        YL = cp.random.rand(N, NM)
    if YR is None:
        YR = cp.random.rand(NM, N)
    
    P0C1, P1C1 = ci_batched_gpu_internal(N, factor, matrix_blocks, 3.0, 1.0, side)
    P0C2, P1C2 = ci_batched_gpu_internal(N, factor, matrix_blocks, 1.0 / R, -1.0, side)

    P0 = P0C1 + P0C2
    P1 = P1C1 + P1C2

    LP0 = P0@YL
    LP1 = P1@YL

    RP0 = YR@P0
    RP1 = YR@P1

    LV, LS, LW = cp.linalg.svd(LP0, full_matrices=False)
    Lind = cp.where(cp.abs(LS) > eps_lim)[0]

    RV, RS, RW = cp.linalg.svd(RP0, full_matrices=False)
    Rind = cp.where(cp.abs(RS) > eps_lim)[0]

    if len(Lind) == N or len(Rind) == N:

        P0C3, P1C3 = ci_batched_gpu_internal(N, factor, matrix_blocks, 10.0 / R, -1.0, side)

        P0 = P0C1 + P0C3
        P1 = P1C1 + P1C3

        LP0 = P0@YL
        LP1 = P1@YL

        RP0 = YR@P0
        RP1 = YR@P1

        LV, LS, LW = cp.linalg.svd(LP0, full_matrices=False)
        Lind = cp.where(cp.abs(LS) > eps_lim)[0]

        RV, RS, RW = cp.linalg.svd(RP0, full_matrices=False)
        Rind = cp.where(cp.abs(RS) > eps_lim)[0]
    
    if len(Lind) == 0:
        Lind = [0]
    if len(Rind) == 0:
        Rind = [0]

    LV = LV[:, Lind]
    LS = cp.diag(LS[Lind])
    LW = LW[Lind, :].conj().T

    RV = RV[:, Rind]
    RS = cp.diag(RS[Rind])
    RW = RW[Rind, :].conj().T

    return LP1, LV, LS, LW, RP1, RV, RS, RW


def contour_svd_mix(factor: int,
                    matrix_blocks: np.ndarray,
                    big_N: int,
                    R: float,
                    side: str,
                    YL=None,
                    YR=None,
                    eps_lim=1e-8):

    N = big_N // factor

    if factor * N < 100:
        NM = round(3 * N / 4)
    else:
        NM = round(N / 2)
    NM = factor * NM

    if YL is None:
        YL = cp.random.rand(N, NM)
    if YR is None:
        YR = cp.random.rand(NM, N)
    
    P0C1, P1C1 = ci_batched_gpu_internal(N, factor, matrix_blocks, 3.0, 1.0, side)
    P0C2, P1C2 = ci_batched_gpu_internal(N, factor, matrix_blocks, 1.0 / R, -1.0, side)

    P0 = P0C1 + P0C2
    P1 = P1C1 + P1C2

    LP0 = P0@YL
    LP1 = P1@YL

    RP0 = YR@P0
    RP1 = YR@P1

    # start = time.time()
    LV, LS, LW = np.linalg.svd(LP0.get(), full_matrices=False)
    Lind = np.where(np.abs(LS) > eps_lim)[0]

    RV, RS, RW = np.linalg.svd(RP0.get(), full_matrices=False)
    Rind = np.where(np.abs(RS) > eps_lim)[0]
    # finish = time.time()
    # print('time to calculate svd (1): ', finish - start, flush=True)

    if len(Lind) == N or len(Rind) == N:

        P0C3, P1C3 = ci_batched_gpu_internal(N, factor, matrix_blocks, 10.0 / R, -1.0, side)

        P0 = P0C1 + P0C3
        P1 = P1C1 + P1C3

        LP0 = P0@YL
        LP1 = P1@YL

        RP0 = YR@P0
        RP1 = YR@P1

        # start = time.time()
        LV, LS, LW = np.linalg.svd(LP0.get(), full_matrices=False)
        Lind = np.where(np.abs(LS) > eps_lim)[0]

        RV, RS, RW = np.linalg.svd(RP0.get(), full_matrices=False)
        Rind = np.where(np.abs(RS) > eps_lim)[0]
        # finish = time.time()
        # print('time to calculate svd (2): ', finish - start, flush=True)
    
    # start = time.time()
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
    # finish = time.time()
    # print('time to prepare eig: ', finish - start, flush=True)

    return LP1, LV, LS, LW, RP1, RV, RS, RW


def contour_svd_mix_2(factor: int,
                    matrix_blocks: np.ndarray,
                    big_N: int,
                    R: float,
                    side: str,
                    YL=None,
                    YR=None,
                    eps_lim=1e-8):

    N = big_N // factor

    if factor * N < 100:
        NM = round(3 * N / 4)
    else:
        NM = round(N / 2)
    NM = factor * NM

    if YL is None:
        YL = cp.random.rand(N, NM)
    if YR is None:
        YR = cp.random.rand(NM, N)

    P0C1, P0C2, P0C3, P1C1, P1C2, P1C3 = ci_batched_combo_gpu_internal(N, factor, matrix_blocks, [3.0, 1.0 / R, 10.0 / R], [1.0, -1.0, -1.0], side)

    P0 = P0C1 + P0C2
    LP0 = P0@YL
    LV, LS, LW = np.linalg.svd(LP0.get(), full_matrices=False)
    Lind = np.where(np.abs(LS) > eps_lim)[0]
    if len(Lind) != N:
        RP0 = YR@P0
        RV, RS, RW = np.linalg.svd(RP0.get(), full_matrices=False)
        Rind = np.where(np.abs(RS) > eps_lim)[0]
        len_Rind = len(Rind)
    else:
        len_Rind = N
    
    if len_Rind != N:
        P1 = P1C1 + P1C2
        LP1 = P1@YL
        RP1 = YR@P1
    else:
        P0 = P0C1 + P0C3
        P1 = P1C1 + P1C3

        LP0 = P0@YL
        LP1 = P1@YL

        RP0 = YR@P0
        RP1 = YR@P1

        # start = time.time()
        LV, LS, LW = np.linalg.svd(LP0.get(), full_matrices=False)
        Lind = np.where(np.abs(LS) > eps_lim)[0]

        RV, RS, RW = np.linalg.svd(RP0.get(), full_matrices=False)
        Rind = np.where(np.abs(RS) > eps_lim)[0]
    
    # start = time.time()
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
    # finish = time.time()
    # print('time to prepare eig: ', finish - start, flush=True)

    return LP1, LV, LS, LW, RP1, RV, RS, RW


def beyn_eig_gpu(LV, LS, LW, LP1, RV, RS, RW, RP1):

    Llambda, Lu = eig(cp.asnumpy(LV.conj().T @ LP1 @ LW @ cp.linalg.inv(LS)))
    Rlambda, Ru = eig(cp.asnumpy(cp.linalg.inv(RS) @ RV.conj().T @ RP1 @ RW))

    Llambda = cp.asarray(Llambda)
    Lu = cp.asarray(Lu)
    Rlambda = cp.asarray(Rlambda)
    Ru = cp.asarray(Ru)

    return Lu, Llambda, Ru, Rlambda


def beyn_eig_mix(LV, LS, LW, LP1, RV, RS, RW, RP1):

    Llambda, Lu = eig(LV.T.conj() @ LP1.get() @ LW @ np.linalg.inv(LS))
    Rlambda, Ru = eig(np.linalg.inv(RS) @ RV.T.conj() @ RP1.get() @ RW)

    Llambda = cp.asarray(Llambda)
    Lu = cp.asarray(Lu)
    Rlambda = cp.asarray(Rlambda)
    Ru = cp.asarray(Ru)

    return Lu, Llambda, Ru, Rlambda


def beyn_phi_gpu(LV, Lu, Llambda, RW, Ru, Rlambda, factor, type):

    N = LV.shape[0]

    phiL = LV @ Lu
    phiR = cp.linalg.solve(Ru, RW.T.conj())

    if type == 'L':
        kL = 1j * cp.log(Llambda)
        kR = 1j * cp.log(Rlambda)
    else:
        kL = -1j * cp.log(Llambda)
        kR = -1j * cp.log(Rlambda)
    
    ind_sort_kL = cp.argsort(cp.abs(cp.imag(kL)))
    kL = kL[ind_sort_kL]
    phiL = phiL[:, ind_sort_kL]

    ind_sort_kR = cp.argsort(cp.abs(cp.imag(kR)))
    kR = kR[ind_sort_kR]
    phiR = phiR[ind_sort_kR, :]

    phiL_vec = cp.zeros((factor * N, len(kL)), dtype=phiL.dtype)
    phiR_vec = cp.zeros((len(kR), factor * N), dtype=phiR.dtype)

    for i in range(factor):
        phiL_vec[i * N:(i + 1) * N, :] = phiL @ cp.diag(cp.exp(i * 1j * kL))
        phiR_vec[:, i * N:(i + 1) * N] = cp.diag(cp.exp(-i * 1j * kR)) @ phiR
    
    phiL = phiL_vec / (cp.ones((factor * N, 1)) @ cp.sqrt(np.sum(np.abs(phiL_vec) ** 2, axis=0, keepdims=True)))
    phiR = phiR_vec / (cp.sqrt(cp.sum(cp.abs(phiR_vec) ** 2, axis=1, keepdims=True)) @ cp.ones((1, factor * N)))

    kL = factor * kL
    kR = factor * kR

    return kL, kR, phiL, phiR


def beyn_phi_batched_gpu(LV, Lu, Llambda, RW, Ru, Rlambda, factor, type):

    batch_size = len(LV)
    N = LV[0].shape[0]

    for i in range(batch_size):
        # print(Llambda[i].shape, Rlambda[i].shape)
        print(LV[i].shape, Lu[i].shape, RW[i].shape, Ru[i].shape)
    # print(Llambda.shape, Rlambda.shape)

    phiL = LV @ Lu
    # phiR = cp.linalg.solve(Ru, RW.transpose(0, 1, 2).conj())
    phiR = cp.linalg.solve(Ru, RW)

    if type == 'L':
        kL = 1j * cp.log(Llambda)
        kR = 1j * cp.log(Rlambda)
    else:
        kL = -1j * cp.log(Llambda)
        kR = -1j * cp.log(Rlambda)
    
    kL_all = cp.abs(cp.imag(kL))
    kR_all = cp.abs(cp.imag(kR))
    for i in range(batch_size):
        ind_sort_kL = cp.argsort(kL_all[i])
        kL[i] = kL[i][ind_sort_kL]
        phiL[i] = phiL[i][:, ind_sort_kL]

        ind_sort_kR = cp.argsort(kR_all[i])
        kR[i] = kR[i][ind_sort_kR]
        phiR[i] = phiR[i][ind_sort_kR, :]

    phiL_vec = cp.zeros((batch_size, factor * N, len(kL)), dtype=phiL.dtype)
    phiR_vec = cp.zeros((batch_size, len(kR), factor * N), dtype=phiR.dtype)

    for i in range(factor):
        phiL_vec[i * N:(i + 1) * N, :] = phiL @ cp.diag(cp.exp(i * 1j * kL))
        phiR_vec[:, i * N:(i + 1) * N] = cp.diag(cp.exp(-i * 1j * kR)) @ phiR
    
    phiL = phiL_vec / (cp.ones((factor * N, 1)) @ cp.sqrt(np.sum(np.abs(phiL_vec) ** 2, axis=0, keepdims=True)))
    phiR = phiR_vec / (cp.sqrt(cp.sum(cp.abs(phiR_vec) ** 2, axis=1, keepdims=True)) @ cp.ones((1, factor * N)))

    kL = factor * kL
    kR = factor * kR

    return kL, kR, phiL, phiR


def beyn_sigma_gpu(kL, kR, phiL, phiR, M00, M01, M10, imag_lim, ref_iteration, type):

    if type == 'L':

        # start = time.time()
        ksurfL, VsurfL, inv_VsurfL, dEk_dk = prepare_input_data_gpu(kL, kR, phiL, phiR, M01, M10, imag_lim, 1.0)
        # finish = time.time()
        # print('time to prepare input data: ', finish - start)

        # start = time.time()
        gR = cp.linalg.inv(M00 + M10 @ VsurfL @ cp.diag(cp.exp(-1j * ksurfL)) @ inv_VsurfL)

        for _ in range(ref_iteration):
            gR = cp.linalg.inv(M00 - M10 @ gR @ M01)
        
        Sigma = M10 @ gR @ M01
        # finish = time.time()
        # print('time to calculate Sigma: ', finish - start)
    
    else:

        ksurfR, VsurfR, inv_VsurfR, dEk_dk = prepare_input_data_gpu(kL, kR, phiL, phiR, M01, M10, imag_lim, -1.0)
        gR = cp.linalg.inv(M00 + M01 @ VsurfR @ cp.diag(cp.exp(1j * ksurfR)) @ inv_VsurfR)

        for _ in range(ref_iteration):
            gR = cp.linalg.inv(M00 - M01 @ gR @ M10)
        
        Sigma = M01 @ gR @ M10

    # start = time.time()
    ind = np.where(abs(dEk_dk))
    if len(ind[0]) > 0:
        min_dEk = np.min(abs(dEk_dk[ind]))
    else:
        min_dEk = 1e8
    finish = time.time()
    # print('time to calculate min_dEk: ', finish - start)
    
    return Sigma, gR, min_dEk


def beyn_sigma_batched_gpu(kL, kR, phiL, phiR, M00, M01, M10, imag_lim, ref_iteration, side):

    batch_size = len(M00)
    if side == 'L':
        first, second, rfactor, ifactor = M01, M10, 1.0, -1j
    else:
        first, second, rfactor, ifactor = M10, M01, -1.0, 1j
    
    T = cp.empty_like(M00)
    min_dEk = np.zeros(batch_size, dtype=np.float64)
    for i in range(batch_size):
        ksurf, Vsurf, inv_Vsurf, dEk_dk = prepare_input_data_gpu(kL[i], kR[i], phiL[i], phiR[i], M01[i], M10[i], imag_lim, rfactor)
        T[i] = Vsurf @ cp.diag(cp.exp(ifactor * ksurf)) @ inv_Vsurf
        ind = np.where(abs(dEk_dk))
        if len(ind[0]) > 0:
            min_dEk[i] = np.min(abs(dEk_dk[ind]))
        else:
            min_dEk[i] = 1e8

    gR = cp.linalg.inv(M00 + second @ T)
    for _ in range(ref_iteration):
        gR = cp.linalg.inv(M00 - second @ gR @ first)
    Sigma = second @ gR @ first
    
    return Sigma, gR, min_dEk

def check_imag_cond_gpu(kref, kL, kR, phiL, phiR, M10, M01, max_imag, kside):

    # start = time.time()

    Nk = len(kref)
    imag_cond = np.zeros(Nk, dtype=np.bool_)
    dEk_dk = cp.zeros(Nk, dtype=phiL.dtype)

    Ikmax = np.int64(cp.asnumpy(cp.count_nonzero(cp.abs(cp.imag(kref)) < np.max((0.5, max_imag)))))
    Ikmax += Ikmax % 2

    pRpL = phiR @ phiL

    if kside == 'L':
        ind_k = cp.argmin(cp.abs(cp.subtract.outer(kL, kR)), axis=1)
        k = kL
    else:
        ind_k = cp.argmin(cp.abs(cp.subtract.outer(kR, kL)), axis=1)
        k = kR
    Ikmax = min(Ikmax, len(ind_k))

    dEk_dk_d = cp.zeros((Ikmax, *M10.shape), dtype=phiL.dtype)

    num_threads = 1024
    num_blocks = (Ikmax * M10.shape[0] * M10.shape[1] + num_threads - 1) // num_threads
    compute_dEk_dk[num_blocks, num_threads](
        # dEk_dk_d.reshape(-1), kL, kR, phiL.reshape(-1), phiR.reshape(-1), pRpL.reshape(-1),
        dEk_dk_d, kL, kR, phiL, phiR, pRpL,
        M01, M10, ind_k, Ikmax, M10.shape[0], kside=='L')
    cp.cuda.stream.get_current_stream().synchronize()
    cp.sum(dEk_dk_d, axis=(1, 2), out=dEk_dk[:Ikmax])

    # #### Validation against CPU ####

    # dEk_dk_cpu = np.zeros(Nk, dtype=phiL.dtype)

    # Ikmax_cpu = np.count_nonzero(np.abs(np.imag(cp.asnumpy(kref))) < np.max((0.5, max_imag)))
    # Ikmax_cpu += Ikmax_cpu % 2
    # assert Ikmax == Ikmax_cpu

    # phiR_cpu = cp.asnumpy(phiR)
    # phiL_cpu = cp.asnumpy(phiL)
    # kR_cpu = cp.asnumpy(kR)
    # kL_cpu = cp.asnumpy(kL)
    # M10_cpu = cp.asnumpy(M10)
    # M01_cpu = cp.asnumpy(M01)

    # pRpL_cpu = phiR_cpu @ phiL_cpu

    # if kside == 'L':
    #     ind_k_cpu = np.argmin(np.abs(np.subtract.outer(kL_cpu, kR_cpu)), axis=1)
    #     assert np.allclose(cp.asnumpy(ind_k), ind_k_cpu)
    #     for Ik in range(Ikmax_cpu):
    #         ind_kR = ind_k_cpu[Ik]
    #         dEk_dk_cpu[Ik] = -(phiR_cpu[ind_kR, :] @ (-1j * M10_cpu * np.exp(-1j * kL_cpu[Ik]) + 1j * M01_cpu * np.exp(1j * kL_cpu[Ik])) @ phiL_cpu[:, Ik]) / pRpL_cpu[ind_kR, Ik]
    # else:
    #     ind_k_cpu = np.argmin(np.abs(np.subtract.outer(kR_cpu, kL_cpu)), axis=1)
    #     assert np.allclose(cp.asnumpy(ind_k), ind_k_cpu)
    #     for Ik in range(Ikmax_cpu):
    #         ind_kL = ind_k_cpu[Ik]
    #         dEk_dk_cpu[Ik] = -(phiR_cpu[Ik, :] @ (-1j * M10_cpu * np.exp(-1j * kR_cpu[Ik]) + 1j * M01_cpu * np.exp(1j * kR_cpu[Ik])) @ phiL_cpu[:, ind_kL]) / pRpL_cpu[Ik, ind_kL]


    # assert np.allclose(cp.asnumpy(dEk_dk), dEk_dk_cpu)

    # ##################################

    ind_neighs_d = cp.argmin(cp.abs(cp.add.outer(dEk_dk, dEk_dk)), axis=1)
    ind_neighs = cp.asnumpy(ind_neighs_d)
    cond = cp.zeros(Nk, dtype=np.bool_)
    num_threads = 512
    num_blocks = (Nk + num_threads - 1) // num_threads
    compute_imag_cond_conditions_kernel((num_blocks,), (num_threads,), (cond, k, dEk_dk, ind_neighs_d, max_imag, Nk))
    cp.cuda.stream.get_current_stream().synchronize()
    cond = cp.asnumpy(cond)

    # finish = time.time()
    # print('time to calculate dEk_dk: ', finish - start)

    for Ik in range(Ikmax):
        ind_neigh = ind_neighs[Ik]
        if not imag_cond[Ik] and not imag_cond[ind_neigh] and cond[Ik]:
                imag_cond[Ik] = True
                imag_cond[ind_neigh] = True

    # ####### Validation against CPU #######
                
    # imag_cond_cpu = np.zeros(Nk, dtype=np.bool_)

    # for Ik in range(Ikmax_cpu):
    #     if not imag_cond_cpu[Ik]:
    #         ind_neigh = np.argmin(np.abs(dEk_dk_cpu + dEk_dk_cpu[Ik] * np.ones(Nk)))
    #         assert ind_neigh == ind_neighs[Ik]

    #         if kside == 'L':
    #             k1 = kL_cpu[Ik]
    #             k2 = kL_cpu[ind_neigh]
    #         else:
    #             k1 = kR_cpu[Ik]
    #             k2 = kR_cpu[ind_neigh]
    #         dEk1 = dEk_dk_cpu[Ik]
    #         dEk2 = dEk_dk_cpu[ind_neigh]

    #         cond1 = np.abs(dEk1 + dEk2) / (np.abs(dEk1) + 1e-10) < 0.25
    #         cond2 = np.abs(k1 + k2) / (np.abs(k1) + 1e-10) < 0.25
    #         cond3 = (np.abs(np.imag(k1)) + np.abs(np.imag(k2))) / 2.0 < 1.5 * max_imag
    #         cond4 = np.sign(np.imag(k1)) == np.sign(np.imag(k2))

    #         if cond1 and cond2 and (cond3 or (not cond3 and cond4)):
    #             assert cond[Ik]
    #             if not imag_cond_cpu[ind_neigh]:
    #                 imag_cond_cpu[Ik] = True
    #                 imag_cond_cpu[ind_neigh] = True
    
    # assert np.allclose(cp.asnumpy(imag_cond), imag_cond_cpu)

    # ######################################

    return cp.asarray(imag_cond), dEk_dk

def sort_k_gpu(kL, kR, phiL, phiR, M01, M10, imag_limit, factor, kside):
    Nk = max(len(kL), len(kR))
    ksurf = cp.zeros(Nk, dtype=kL.dtype)
    dEk = cp.zeros(Nk, dtype=phiL.dtype)

    if kside == 'L':
        NT = len(phiL[:, 0])
        Vsurf = cp.zeros((NT, Nk), dtype=phiL.dtype)
        kref = kL
    else:
        NT = len(phiR[0, :])
        Vsurf = cp.zeros((Nk, NT), dtype=phiR.dtype)
        kref = kR

    # start = time.time()
    imag_cond, dEk_dk = check_imag_cond_gpu(kref, kL, kR, phiL, phiR, M10, M01, imag_limit, kside)
    # finish = time.time()
    # print('time to check imag cond: ', finish - start)

    Nk = len(kref)
    condA = cp.ndarray(Nk, dtype=np.bool_)
    condB = cp.ndarray(Nk, dtype=np.bool_)
    num_threads = 1024
    num_blocks = (Nk + num_threads - 1) // num_threads
    compute_sort_k_conditions_kernel((num_blocks,), (num_threads,), (condA, condB, kref, imag_cond, dEk_dk, Nk, factor))
    cp.cuda.stream.get_current_stream().synchronize()
    condA = cp.asnumpy(condA)
    condB = cp.asnumpy(condB)


    Nref = 0
    # stream = cp.cuda.stream.get_current_stream()
    for Ik in range(Nk):
        # if (cond3[Ik] and cond4[Ik]) or (not cond3[Ik] and (cond1[Ik] or cond2[Ik])):
        if condA[Ik]:
            ksurf[Nref] = kref[Ik]
            # cp.cuda.runtime.memcpyAsync(ksurf[Nref].data.ptr, kref[Ik].data.ptr, kref[Ik].nbytes, cp.cuda.runtime.memcpyDeviceToDevice, stream)
            if kside == 'L':
                Vsurf[:, Nref] = phiL[:, Ik]
                # cp.cuda.runtime.memcpyAsync(Vsurf[:, Nref].data, phiL[:, Ik].data, phiL[:, Ik].nbytes, cp.cuda.runtime.memcpyDeviceToDevice, stream)
            else:
                Vsurf[Nref, :] = phiR[Ik, :]
                # cp.cuda.runtime.memcpyAsync(Vsurf[Nref, :].data, phiR[Ik, :].data, phiR[Ik, :].nbytes, cp.cuda.runtime.memcpyDeviceToDevice, stream)
            # if not cond3[Ik] and (cond1[Ik] or cond2[Ik]):
            if condB[Ik]:
                dEk[Nref] = dEk_dk[Ik]
                # cp.cuda.runtime.memcpyAsync(dEk[Nref].data, dEk_dk[Ik].data, dEk_dk[Ik].nbytes, cp.cuda.runtime.memcpyDeviceToDevice, stream)
            Nref += 1
    # cp.cuda.stream.get_current_stream().synchronize()

    # ###### Validation against CPU ######

    # Nref_cpu = 0
    # phiL_cpu = cp.asnumpy(phiL)
    # phiR_cpu = cp.asnumpy(phiR)
    # imag_cond_cpu = cp.asnumpy(imag_cond)
    # kref_cpu = cp.asnumpy(kref)
    # dEk_dk_cpu = cp.asnumpy(dEk_dk)
    # ksurf_cpu = np.zeros(Nk, dtype=kL.dtype)
    # if kside == 'L':
    #     NT = len(phiL[:, 0])
    #     Vsurf_cpu = np.zeros((NT, Nk), dtype=phiL.dtype)
    # else:
    #     NT = len(phiR[0, :])
    #     Vsurf_cpu = np.zeros((Nk, NT), dtype=phiR.dtype)
    # dEk_cpu = np.zeros(Nk, dtype=phiL.dtype)

    # for Ik in range(len(kref_cpu)):

    #     if not imag_cond_cpu[Ik] and abs(np.imag(kref_cpu[Ik])) > 1e-6:

    #         if factor * np.imag(kref_cpu[Ik]) < 0:

    #             Nref_cpu += 1
    #             ksurf_cpu[Nref_cpu - 1] = kref_cpu[Ik]

    #             if kside == 'L':
    #                 Vsurf_cpu[:, Nref_cpu - 1] = phiL_cpu[:, Ik]
    #             else:
    #                 Vsurf_cpu[Nref_cpu - 1, :] = phiR_cpu[Ik, :]

    #             dEk_cpu[Nref_cpu - 1] = 0

    #     else:

    #         cond1 = (abs(np.real(dEk_dk_cpu[Ik])) < abs(np.imag(dEk_dk_cpu[Ik])) / 100) and (factor * np.imag(kref_cpu[Ik]) < 0)
    #         cond2 = (abs(np.real(dEk_dk_cpu[Ik])) >= abs(np.imag(dEk_dk_cpu[Ik])) / 100) and (factor * np.real(dEk_dk_cpu[Ik]) < 0)

    #         if cond1 or cond2:

    #             Nref_cpu += 1
    #             ksurf_cpu[Nref_cpu - 1] = kref_cpu[Ik]

    #             if kside == 'L':
    #                 Vsurf_cpu[:, Nref_cpu - 1] = phiL_cpu[:, Ik]
    #             else:
    #                 Vsurf_cpu[Nref_cpu - 1, :] = phiR_cpu[Ik, :]
                
    #             dEk_cpu[Nref_cpu - 1] = dEk_dk_cpu[Ik]
    

    # assert np.allclose(cp.asnumpy(ksurf[:Nref]), ksurf_cpu[:Nref_cpu])
    # if kside == 'L':
    #     assert np.allclose(cp.asnumpy(Vsurf[:, :Nref]), Vsurf_cpu[:, :Nref_cpu])
    # else:
    #     assert np.allclose(cp.asnumpy(Vsurf[:Nref, :]), Vsurf_cpu[:Nref_cpu, :])
    # assert np.allclose(cp.asnumpy(dEk[:Nref]), dEk_cpu[:Nref_cpu])

    # ####################################

    ksurf = ksurf[0:Nref]

    if kside == 'L':
        Vsurf = Vsurf[:, 0:Nref]
    else:
        Vsurf = Vsurf[0:Nref, :]
    
    dEk = dEk[0:Nref]

    return ksurf, Vsurf, dEk

def prepare_input_data_gpu(kL, kR, phiL, phiR, M01, M10, imag_limit, factor):
    ksurfL, VsurfL, dEk = sort_k_gpu(kL, kR, phiL, phiR, M01, M10, imag_limit, factor, 'L')
    ksurfR, VsurfR, _ = sort_k_gpu(kL, kR, phiL, phiR, M01, M10, imag_limit, factor, 'R')

    Nk = min(len(ksurfL), len(ksurfR))

    VsurfL = VsurfL[:, :Nk]
    ksurfL = ksurfL[:Nk]
    VsurfR = VsurfR[:Nk, :]

    # Compute the inverse of VsurfL with the help from the VsurfR eigenvectors
    # inv_VsurfL = np.linalg.pinv(VsurfR @ VsurfL) @ VsurfR
    inv_VsurfL = cp.linalg.solve(VsurfR @ VsurfL, VsurfR)

    return ksurfL, VsurfL, inv_VsurfL, dEk

def beyn_new_gpu(factor: int,
                 matrix_blocks,
                 M00,
                 M01,
                 M10,
                 imag_lim,
                 R,
                 type,
                 YL=None,
                 YR=None):
    

    cond = 0
    min_dEk = 1e8

    
    # LP0, LP1, RP0, RP1 = contour_integral_batched_gpu(factor, matrix_blocks, M00.shape[0], R, type, YL=YL, YR=YR)
    # LV, LS, LW, RV, RS, RW = beyn_svd_gpu(LP0, RP0, eps_lim=1e-8)
    LP1, LV, LS, LW, RP1, RV, RS, RW = contour_svd_gpu(factor, matrix_blocks, M00.shape[0], R, type, YL=YL, YR=YR, eps_lim=1e-8)

    if LS.size == 0 or RS.size == 0:
        cond = np.nan
        Sigma = None
        gR = None
        return Sigma, gR, cond, min_dEk

    Lu, Llambda, Ru, Rlambda = beyn_eig_gpu(LV, LS, LW, LP1, RV, RS, RW, RP1)
    kL, kR, phiL, phiR = beyn_phi_gpu(LV, Lu, Llambda, RW, Ru, Rlambda, factor, type)
    Sigma, gR, min_dEk = beyn_sigma_gpu(kL, kR, phiL, phiR, M00, M01, M10, imag_lim, 2, type)

    return Sigma, gR, cond, min_dEk


def beyn_gpu(factor: int,
             M00,
             M01,
             M10,
             imag_lim,
             R,
             type,
             YL=None,
             YR=None):
    
    N00, N01, N10, matrix_blocks = extract_small_matrix_blocks_gpu(M00, M01, M10, factor, type, densify = True)
    return beyn_new_gpu(factor, matrix_blocks, N00, N01, N10, imag_lim, R, type, YL=YL, YR=YR)


def beyn_new_mix(factor: int,
                 matrix_blocks,
                 M00,
                 M01,
                 M10,
                 imag_lim,
                 R,
                 type,
                 YL=None,
                 YR=None):
    

    cond = 0
    min_dEk = 1e8

    
    # LP0, LP1, RP0, RP1 = contour_integral_batched_gpu(factor, matrix_blocks, M00.shape[0], R, type, YL=YL, YR=YR)
    # LV, LS, LW, RV, RS, RW = beyn_svd_gpu(LP0, RP0, eps_lim=1e-8)
    # start = time.time()
    LP1, LV, LS, LW, RP1, RV, RS, RW = contour_svd_mix(factor, matrix_blocks, M00.shape[0], R, type, YL=YL, YR=YR, eps_lim=1e-8)
    # finish = time.time()
    # print('time to compute contour_svd_mix: ', finish - start, flush=True)

    if LS.size == 0 or RS.size == 0:
        cond = np.nan
        Sigma = None
        gR = None
        return Sigma, gR, cond, min_dEk

    # start = time.time()
    Lu, Llambda, Ru, Rlambda = beyn_eig_mix(LV, LS, LW, LP1, RV, RS, RW, RP1)
    # finish = time.time()
    # print('time to compute beyn_eig_mix: ', finish - start, flush=True)
    # start = time.time()
    kL, kR, phiL, phiR = beyn_phi_gpu(cp.asarray(LV), Lu, Llambda, cp.asarray(RW), Ru, Rlambda, factor, type)
    # finish = time.time()
    # print('time to compute beyn_phi_gpu: ', finish - start, flush=True)
    # start = time.time()
    Sigma, gR, min_dEk = beyn_sigma_gpu(kL, kR, phiL, phiR, M00, M01, M10, imag_lim, 2, type)
    # finish = time.time()
    # print('time to compute beyn_sigma_gpu: ', finish - start, flush=True)

    return Sigma, gR, cond, min_dEk


def beyn_new_mix_2(factor: int,
                 matrix_blocks,
                 M00,
                 M01,
                 M10,
                 imag_lim,
                 R,
                 type,
                 YL=None,
                 YR=None):
    

    cond = 0
    min_dEk = 1e8

    
    # LP0, LP1, RP0, RP1 = contour_integral_batched_gpu(factor, matrix_blocks, M00.shape[0], R, type, YL=YL, YR=YR)
    # LV, LS, LW, RV, RS, RW = beyn_svd_gpu(LP0, RP0, eps_lim=1e-8)
    # start = time.time()
    LP1, LV, LS, LW, RP1, RV, RS, RW = contour_svd_mix_2(factor, matrix_blocks, M00.shape[0], R, type, YL=YL, YR=YR, eps_lim=1e-8)
    # finish = time.time()
    # print('time to compute contour_svd_mix: ', finish - start, flush=True)

    if LS.size == 0 or RS.size == 0:
        cond = np.nan
        Sigma = None
        gR = None
        return Sigma, gR, cond, min_dEk

    # start = time.time()
    Lu, Llambda, Ru, Rlambda = beyn_eig_mix(LV, LS, LW, LP1, RV, RS, RW, RP1)
    # finish = time.time()
    # print('time to compute beyn_eig_mix: ', finish - start, flush=True)
    # start = time.time()
    kL, kR, phiL, phiR = beyn_phi_gpu(cp.asarray(LV), Lu, Llambda, cp.asarray(RW), Ru, Rlambda, factor, type)
    # finish = time.time()
    # print('time to compute beyn_phi_gpu: ', finish - start, flush=True)
    # start = time.time()
    Sigma, gR, min_dEk = beyn_sigma_gpu(kL, kR, phiL, phiR, M00, M01, M10, imag_lim, 2, type)
    # finish = time.time()
    # print('time to compute beyn_sigma_gpu: ', finish - start, flush=True)

    return Sigma, gR, cond, min_dEk


def beyn_mix(factor: int,
             M00,
             M01,
             M10,
             imag_lim,
             R,
             type,
             YL=None,
             YR=None):
    
    # start = time.time()
    N00, N01, N10, matrix_blocks = extract_small_matrix_blocks_gpu(M00, M01, M10, factor, type, densify = True)
    # finish = time.time()
    # print('time to extract_small_matrix_blocks_gpu: ', finish - start, flush=True)
    return beyn_new_mix(factor, matrix_blocks, N00, N01, N10, imag_lim, R, type, YL=YL, YR=YR)


def beyn_mix_2(factor: int,
             M00,
             M01,
             M10,
             imag_lim,
             R,
             type,
             YL=None,
             YR=None):
    
    # start = time.time()
    N00, N01, N10, matrix_blocks = extract_small_matrix_blocks_gpu(M00, M01, M10, factor, type, densify = True)
    # finish = time.time()
    # print('time to extract_small_matrix_blocks_gpu: ', finish - start, flush=True)
    return beyn_new_mix_2(factor, matrix_blocks, N00, N01, N10, imag_lim, R, type, YL=YL, YR=YR)
