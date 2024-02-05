# Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.
import cupy as cp
import cupyx as cpx
import numpy as np
# import skcuda.magma as magma
import time

from scipy.linalg import svd
from numpy.linalg import eig
from scipy.sparse import csr_matrix

from quatrex.OBC.contour_integral import contour_integral as ci_internal, contour_integral_batched as ci_batched_internal
from quatrex.OBC.contour_integral import contour_integral_gpu as ci_gpu_internal, contour_integral_batched_gpu as ci_batched_gpu_internal


# magma.magma_init()

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


# @cpx.jit.rawkernel()
# def compute_sort_k_conditions(condA, condB, kref, imag_cond, dEk_dk, N, factor):

#     idx = cpx.jit.blockIdx.x * cpx.jit.blockDim.x + cpx.jit.threadIdx.x
#     if idx < N:

#         kr = kref[idx]
#         ek = dEk_dk[idx]
#         icond = imag_cond[idx]
#         kr_imag = (kr - cp.conj(kr)) / 2.0
#         ek_real = (ek + cp.conj(ek)) / 2.0
#         ek_imag = (ek - cp.conj(ek)) / 2.0
#         other1 = factor * kr_imag
#         other1_sgn = other1 / cp.abs(other1)
#         other2 = factor * ek_real
#         other2_sgn = other2 / cp.abs(other2)
#         minus_one = -1.0 + 0.0j
        

#         # cond1 = (cp.abs(ek_real) < (cp.abs(ek_imag) / 100.0)) and (factor * kr_imag < 0.0)
#         # cond2 = (cp.abs(ek_real) >= (cp.abs(ek_imag) / 100.0)) and (factor * ek_real < 0.0)
#         cond1 = (cp.abs(ek_real) < (cp.abs(ek_imag) / 100.0)) and (other1_sgn == minus_one)
#         cond2 = (cp.abs(ek_real) >= (cp.abs(ek_imag) / 100.0)) and (other2_sgn == minus_one)
#         cond3 = not icond and cp.abs(kr_imag) > 1e-6
#         cond4 = other1_sgn == minus_one

#         condA[idx] = (cond3 and cond4) or (cond1 or cond2)
#         condB[idx] = not (cond3 and cond4) and (cond1 or cond2)


def extract_small_matrix_blocks(M00, M01, M10, factor, type, sparsify: bool = False):
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

    if sparsify:
        m00 = csr_matrix(m00)
        m01 = csr_matrix(m01)
        m10 = csr_matrix(m10)

    return m00, m01, m10, matrix_blocks


def extract_small_matrix_blocks_gpu(M00, M01, M10, factor, type, sparsify: bool = False):
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

    if sparsify:
        m00 = cp.sparse.csr_matrix(m00)
        m01 = cp.sparse.csr_matrix(m01)
        m10 = cp.sparse.csr_matrix(m10)

    return m00, m01, m10, matrix_blocks


# NOTE: Original version of contour integral for double number of theta points
# def contour_integral_gpu(factor: int,
#                          matrix_blocks: np.ndarray,
#                          big_N: int,
#                          R: float,
#                          type: str,
#                          YL=None,
#                          YR=None):

#     theta_min = 0
#     theta_max = 2 * np.pi
#     NT = 51

#     N = big_N // factor

#     theta = cp.linspace(theta_min, theta_max, NT)
#     dtheta = cp.hstack((theta[1] - theta[0], theta[2:] - theta[:-2], theta[-1] - theta[-2])) / 2

#     c = 0
#     r1 = np.power(3.0, 1.0 / factor)
#     r2 = np.power(1.0 / R, 1.0 / factor)

#     zC1 = c + r1 * cp.exp(1j * theta)
#     zC2 = c + r2 * cp.exp(1j * theta)
#     z = cp.hstack((zC1, zC2))

#     dzC1_dtheta = 1j * r1 * cp.exp(1j * theta)
#     dzC2_dtheta = 1j * r2 * cp.exp(1j * theta)
#     dz_dtheta = cp.hstack((dzC1_dtheta, -dzC2_dtheta))

#     dtheta =  cp.hstack((dtheta, dtheta))

#     if factor * N < 100:
#         NM = round(3 * N / 4)
#     else:
#         NM = round(N / 2)
#     NM = factor * NM

#     if YL is None:
#         YL = cp.random.rand(N, NM)
#     if YR is None:
#         YR = cp.random.rand(NM, N)
#     P0 = cp.zeros((N, N), dtype=np.complex128)
#     P1 = cp.zeros((N, N), dtype=np.complex128)

#     for I in range(len(z)):

#         T = cp.zeros((N, N), dtype=np.complex128)
#         for J in range(2 * factor + 1):
#             if type == 'L':
#                 T = T + matrix_blocks[J] * z[I] ** (factor - J)
#             else:
#                 T = T + matrix_blocks[J] * z[I] ** (J - factor)

#         iT = cp.linalg.inv(T)

#         P0 += iT*dz_dtheta[I]*dtheta[I]/(2*np.pi*1j)
#         P1 += iT*z[I]*dz_dtheta[I]*dtheta[I]/(2*np.pi*1j)

#     LP0 = P0@YL
#     LP1 = P1@YL

#     RP0 = YR@P0
#     RP1 = YR@P1

#     return LP0, LP1, RP0, RP1


# NOTE: Original version of contour integral for double number of theta points
# def contour_integral_batched_gpu(factor: int,
#                                  matrix_blocks: np.ndarray,
#                                  big_N: int,
#                                  R: float,
#                                  type: str,
#                                  YL=None,
#                                  YR=None):

#     theta_min = 0.0
#     theta_max = 2 * np.pi
#     NT = 51

#     N = big_N // factor

#     # theta = cp.linspace(theta_min, theta_max, NT, dtype=np.float64)
#     # dtheta = cp.hstack((theta[1] - theta[0], theta[2:] - theta[:-2], theta[-1] - theta[-2])) / 2

#     # c = 0
#     # r1 = np.power(3.0, 1.0 / factor)
#     # r2 = np.power(1.0 / R, 1.0 / factor)

#     # zC1 = c + r1 * cp.exp(1j * theta)
#     # zC2 = c + r2 * cp.exp(1j * theta)
#     # z = cp.hstack((zC1, zC2))

#     # dzC1_dtheta = 1j * r1 * cp.exp(1j * theta)
#     # dzC2_dtheta = 1j * r2 * cp.exp(1j * theta)
#     # dz_dtheta = cp.hstack((dzC1_dtheta, -dzC2_dtheta))

#     # dtheta = cp.hstack((dtheta, dtheta))

#     # dtheta_dev = cp.ndarray((NT * 2,), dtype=np.float64)
#     # dz_dtheta_dev = cp.ndarray((NT * 2,), dtype=np.complex128)
#     # z_dev = cp.ndarray((NT * 2,), dtype=np.complex128)
#     # # compute_theta[1, NT*2](dtheta_dev, dz_dtheta_dev, z_dev, theta_min, theta_max, NT, factor, R)
#     # compute_theta_kernel((1,), (NT*2,), (dtheta_dev, dz_dtheta_dev, z_dev, theta_min, theta_max, NT, factor, R))
#     # cp.cuda.stream.get_current_stream().synchronize()
#     # print(dtheta - dtheta_dev)
#     # assert cp.allclose(dtheta, dtheta_dev)
#     # assert cp.allclose(dz_dtheta, dz_dtheta_dev)
#     # assert cp.allclose(z, z_dev)


#     dtheta = cp.ndarray((NT * 2,), dtype=np.float64)
#     dz_dtheta = cp.ndarray((NT * 2,), dtype=np.complex128)
#     z = cp.ndarray((NT * 2,), dtype=np.complex128)
#     # compute_theta[1, NT*2](dtheta, dz_dtheta, z, theta_min, theta_max, NT, factor, R)
#     compute_theta_kernel((1,), (NT*2,), (dtheta, dz_dtheta, z, theta_min, theta_max, NT, factor, R))
#     # cp.cuda.stream.get_current_stream().synchronize()

#     if factor * N < 100:
#         NM = round(3 * N / 4)
#     else:
#         NM = round(N / 2)
#     NM = factor * NM

#     if YL is None:
#         YL = cp.random.rand(N, NM)
#     if YR is None:
#         YR = cp.random.rand(NM, N)
#     P0 = cp.zeros((N, N), dtype=np.complex128)
#     P1 = cp.zeros((N, N), dtype=np.complex128)

#     T = cp.zeros((len(z), N, N), dtype=np.complex128)

#     # for I in range(len(z)):
#     #     for J in range(2 * factor + 1):
#     #         if type == 'L':
#     #             T[I] = T[I] + matrix_blocks[J] * z[I] ** (factor - J)
#     #         else:
#     #             T[I] = T[I] + matrix_blocks[J] * z[I] ** (J - factor)
#     num_threads = 1024
#     num_blocks = (len(z) * N * N + num_threads - 1) // num_threads
#     contour[num_blocks, num_threads](T.reshape(-1), matrix_blocks.reshape(-1), z, factor, len(z), N, np.bool_(type=='L'))
#     # cp.cuda.stream.get_current_stream().synchronize()

#     iT = cp.linalg.inv(T)

#     P0 = cp.sum(iT*(dz_dtheta*dtheta/(2*np.pi*1j)).reshape(len(z), 1, 1), axis=0)
#     P1 = cp.sum(iT*(z*dz_dtheta*dtheta/(2*np.pi*1j)).reshape(len(z), 1, 1), axis=0)

#     LP0 = P0@YL
#     LP1 = P1@YL

#     RP0 = YR@P0
#     RP1 = YR@P1

#     return LP0, LP1, RP0, RP1


# NOTE: Original version of contour integral for double number of theta points
# def contour_integral(factor: int,
#                      matrix_blocks: np.ndarray,
#                      big_N: int,
#                      R: float,
#                      type: str,
#                      YL=None,
#                      YR=None):

#     theta_min = 0
#     theta_max = 2 * np.pi
#     NT = 51

#     N = big_N // factor

#     theta = np.linspace(theta_min, theta_max, NT)
#     dtheta = np.hstack((theta[1] - theta[0], theta[2:] - theta[:-2], theta[-1] - theta[-2])) / 2

#     c = 0
#     r1 = np.power(3.0, 1.0 / factor)
#     r2 = np.power(1.0 / R, 1.0 / factor)

#     zC1 = c + r1 * np.exp(1j * theta)
#     zC2 = c + r2 * np.exp(1j * theta)
#     z = np.hstack((zC1, zC2))

#     dzC1_dtheta = 1j * r1 * np.exp(1j * theta)
#     dzC2_dtheta = 1j * r2 * np.exp(1j * theta)
#     dz_dtheta = np.hstack((dzC1_dtheta, -dzC2_dtheta))

#     dtheta = np.hstack((dtheta, dtheta))

#     if factor * N < 100:
#         NM = round(3 * N / 4)
#     else:
#         NM = round(N / 2)
#     NM = factor * NM

#     if YL is None:
#         YL = np.random.rand(N, NM)
#     if YR is None:
#         YR = np.random.rand(NM, N)
#     P0 = np.zeros((N, N), dtype=np.complex128)
#     P1 = np.zeros((N, N), dtype=np.complex128)

#     for I in range(len(z)):

#         T = np.zeros((N, N), dtype=np.complex128)
#         for J in range(2 * factor + 1):
#             if type == 'L':
#                 T = T + matrix_blocks[J] * z[I] ** (factor - J)
#             else:
#                 T = T + matrix_blocks[J] * z[I] ** (J - factor)

#         iT = np.linalg.inv(T)

#         P0 += iT*dz_dtheta[I]*dtheta[I]/(2*np.pi*1j)
#         P1 += iT*z[I]*dz_dtheta[I]*dtheta[I]/(2*np.pi*1j)

#     LP0 = P0@YL
#     LP1 = P1@YL

#     RP0 = YR@P0
#     RP1 = YR@P1

#     return LP0, LP1, RP0, RP1


def contour_integral(factor: int,
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
    
    P0C1, P1C1 = ci_internal(N, factor, matrix_blocks, 3.0, 1.0, side)
    P0C2, P1C2 = ci_internal(N, factor, matrix_blocks, 1.0 / R, -1.0, side)

    P0 = P0C1 + P0C2
    P1 = P1C1 + P1C2

    LP0 = P0@YL
    LP1 = P1@YL

    RP0 = YR@P0
    RP1 = YR@P1

    return LP0, LP1, RP0, RP1


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


# NOTE: Original version of contour integral for double number of theta points
# def contour_integral_batched(factor: int,
#                              matrix_blocks: np.ndarray,
#                              big_N: int,
#                              R: float,
#                              type: str,
#                              YL=None,
#                              YR=None):

#     theta_min = 0
#     theta_max = 2 * np.pi
#     NT = 51

#     N = big_N // factor

#     theta = np.linspace(theta_min, theta_max, NT)
#     dtheta = np.hstack((theta[1] - theta[0], theta[2:] - theta[:-2], theta[-1] - theta[-2])) / 2

#     c = 0
#     r1 = np.power(3.0, 1.0 / factor)
#     r2 = np.power(1.0 / R, 1.0 / factor)

#     zC1 = c + r1 * np.exp(1j * theta)
#     zC2 = c + r2 * np.exp(1j * theta)
#     z = np.hstack((zC1, zC2))

#     dzC1_dtheta = 1j * r1 * np.exp(1j * theta)
#     dzC2_dtheta = 1j * r2 * np.exp(1j * theta)
#     dz_dtheta = np.hstack((dzC1_dtheta, -dzC2_dtheta))

#     dtheta = np.hstack((dtheta, dtheta))

#     if factor * N < 100:
#         NM = round(3 * N / 4)
#     else:
#         NM = round(N / 2)
#     NM = factor * NM

#     if YL is None:
#         YL = np.random.rand(N, NM)
#     if YR is None:
#         YR = np.random.rand(NM, N)
#     P0 = np.zeros((N, N), dtype=np.complex128)
#     P1 = np.zeros((N, N), dtype=np.complex128)

#     T = np.zeros((len(z), N, N), dtype=np.complex128)

#     for I in range(len(z)):
#         for J in range(2 * factor + 1):
#             if type == 'L':
#                 T[I] = T[I] + matrix_blocks[J] * z[I] ** (factor - J)
#             else:
#                 T[I] = T[I] + matrix_blocks[J] * z[I] ** (J - factor)

#     iT = np.linalg.inv(T)

#     P0 = np.sum(iT*(dz_dtheta*dtheta/(2*np.pi*1j)).reshape(len(z), 1, 1), axis=0)
#     P1 = np.sum(iT*(z*dz_dtheta*dtheta/(2*np.pi*1j)).reshape(len(z), 1, 1), axis=0)

#     LP0 = P0@YL
#     LP1 = P1@YL

#     RP0 = YR@P0
#     RP1 = YR@P1

#     return LP0, LP1, RP0, RP1


def contour_integral_batched(factor: int,
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
    
    P0C1, P1C1 = ci_batched_internal(N, factor, matrix_blocks, 3.0, 1.0, side)
    P0C2, P1C2 = ci_batched_internal(N, factor, matrix_blocks, 1.0 / R, -1.0, side)

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


def contour_svd(factor: int,
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
        YL = np.random.rand(N, NM)
    if YR is None:
        YR = np.random.rand(NM, N)
    
    P0C1, P1C1 = ci_batched_internal(N, factor, matrix_blocks, 3.0, 1.0, side)
    P0C2, P1C2 = ci_batched_internal(N, factor, matrix_blocks, 1.0 / R, -1.0, side)

    P0 = P0C1 + P0C2
    P1 = P1C1 + P1C2

    LP0 = P0@YL
    LP1 = P1@YL

    RP0 = YR@P0
    RP1 = YR@P1

    LV, LS, LW = svd(LP0, full_matrices=False)
    Lind = np.where(np.abs(LS) > eps_lim)[0]

    RV, RS, RW = svd(RP0, full_matrices=False)
    Rind = np.where(np.abs(RS) > eps_lim)[0]

    if len(Lind) == N or len(Rind) == N:
        print("CPU: Using 10/R contour")

        P0C3, P1C3 = ci_batched_internal(N, factor, matrix_blocks, 10.0 / R, -1.0, side)

        P0 = P0C1 + P0C3
        P1 = P1C1 + P1C3

        LP0 = P0@YL
        LP1 = P1@YL

        RP0 = YR@P0
        RP1 = YR@P1

        LV, LS, LW = svd(LP0, full_matrices=False)
        Lind = np.where(np.abs(LS) > eps_lim)[0]

        RV, RS, RW = svd(RP0, full_matrices=False)
        Rind = np.where(np.abs(RS) > eps_lim)[0]
    
    if len(Lind) == 0:
        print("CPU: No singular values found for left eigenvectors")
        Lind = 0
    if len(Rind) == 0:
        print("CPU: No singular values found for right eigenvectors")
        Rind = 0

    LV = LV[:, Lind]
    LS = np.diag(LS[Lind])
    LW = LW[Lind, :].conj().T

    RV = RV[:, Rind]
    RS = np.diag(RS[Rind])
    RW = RW[Rind, :].conj().T

    return LP1, LV, LS, LW, RP1, RV, RS, RW


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
        print("GPU: Using 10/R contour")

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
        print("GPU: No singular values found for left eigenvectors")
        Lind = 0
    if len(Rind) == 0:
        print("GPU: No singular values found for right eigenvectors")
        Rind = 0

    LV = LV[:, Lind]
    LS = cp.diag(LS[Lind])
    LW = LW[Lind, :].conj().T

    RV = RV[:, Rind]
    RS = cp.diag(RS[Rind])
    RW = RW[Rind, :].conj().T

    return LP1, LV, LS, LW, RP1, RV, RS, RW


def contour_svd_eig_lumi(factor: int,
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

    LP0 = cp.asnumpy(P0@YL)
    LP1 = cp.asnumpy(P1@YL)

    RP0 = cp.asnumpy(YR@P0)
    RP1 = cp.asnumpy(YR@P1)

    LV, LS, LW = svd(LP0, full_matrices=False)
    Lind = np.where(np.abs(LS) > eps_lim)[0]

    RV, RS, RW = svd(RP0, full_matrices=False)
    Rind = np.where(np.abs(RS) > eps_lim)[0]

    if len(Lind) == N or len(Rind) == N:
        # print("GPU: Using 10/R contour")

        P0C3, P1C3 = ci_batched_gpu_internal(N, factor, matrix_blocks, 10.0 / R, -1.0, side)

        P0 = P0C1 + P0C3
        P1 = P1C1 + P1C3

        LP0 = cp.asnumpy(P0@YL)
        LP1 = cp.asnumpy(P1@YL)

        RP0 = cp.asnumpy(YR@P0)
        RP1 = cp.asnumpy(YR@P1)

        LV, LS, LW = svd(LP0, full_matrices=False)
        Lind = np.where(np.abs(LS) > eps_lim)[0]

        RV, RS, RW = svd(RP0, full_matrices=False)
        Rind = np.where(np.abs(RS) > eps_lim)[0]
    
    if len(Lind) == 0:
        print("GPU: No singular values found for left eigenvectors")
        Lind = 0
    if len(Rind) == 0:
        print("GPU: No singular values found for right eigenvectors")
        Rind = 0

    LV = LV[:, Lind]
    # LS = np.diag(LS[Lind])
    LS = np.diag(1 / LS[Lind])
    LW = LW[Lind, :].conj().T

    RV = RV[:, Rind]
    # RS = np.diag(RS[Rind])
    RS = np.diag(1 / RS[Rind])
    RW = RW[Rind, :].conj().T

    # Llambda, Lu = eig(LV.conj().T @ LP1 @ LW @ np.linalg.inv(LS))
    # Rlambda, Ru = eig(np.linalg.inv(RS) @ RV.conj().T @ RP1 @ RW)
    Llambda, Lu = eig(LV.conj().T @ LP1 @ LW @ LS)
    Rlambda, Ru = eig(RS @ RV.conj().T @ RP1 @ RW)

    LV = cp.asarray(LV)
    RW = cp.asarray(RW)
    Lu = cp.asarray(Lu)
    Ru = cp.asarray(Ru)
    Llambda = cp.asarray(Llambda)
    Rlambda = cp.asarray(Rlambda)

    return LV, RW, Lu, Llambda, Ru, Rlambda


def beyn_eig(LV, LS, LW, LP1, RV, RS, RW, RP1):

    Llambda, Lu = eig(LV.conj().T @ LP1 @ LW @ np.linalg.inv(LS))
    Rlambda, Ru = eig(np.linalg.inv(RS) @ RV.conj().T @ RP1 @ RW)

    return Lu, Llambda, Ru, Rlambda


def beyn_eig_gpu(LV, LS, LW, LP1, RV, RS, RW, RP1):


    # # LA = cp.asnumpy(LV.conj().T @ LP1 @ LW @ cp.linalg.inv(LS))
    # LA = np.transpose(cp.asnumpy(LV.conj().T @ LP1 @ LW @ cp.linalg.inv(LS)))
    # # RA = cp.asnumpy(cp.linalg.inv(RS) @ RV.conj().T @ RP1 @ RW)
    # RA = np.transpose(cp.asnumpy(cp.linalg.inv(RS) @ RV.conj().T @ RP1 @ RW))

    # Nl, Nr = LA.shape[0], RA.shape[0]
    # N = max(Nl, Nr)

    # Llambda = np.zeros((Nl,), np.complex128) # eigenvalues
    # Rlambda = np.zeros((Nr,), np.complex128) # eigenvalues
    # Lu = np.zeros((Nl, Nl), np.complex128)
    # Ru = np.zeros((Nr, Nr), np.complex128)

    # # Set up workspace:
    # nb = magma.magma_get_zgeqrf_nb(N, N)
    # lwork = N*(1 + 2*nb)

    # work = np.zeros((lwork,), np.complex128)
    # rwork= np.zeros((2*N,), np.complex64)

    # # status = magma.magma_zgeev('V', 'V', N, LA.ctypes.data, N, Llambda.ctypes.data, vl.ctypes.data, N, Lu.ctypes.data, N, work.ctypes.data, lwork, rwork.ctypes.data)
    # status = magma.magma_zgeev('V', 'N', Nl, LA.ctypes.data, Nl, Llambda.ctypes.data, Lu.ctypes.data, Nl, 0, Nl, work.ctypes.data, lwork, rwork.ctypes.data)
    # Lu[:] = Lu.conj().T
    # # status = magma.magma_zgeev('N', 'V', N, RA.ctypes.data, N, Rlambda.ctypes.data, vl.ctypes.data, N, Ru.ctypes.data, N, work.ctypes.data, lwork, rwork.ctypes.data)
    # status = magma.magma_zgeev('V', 'N', Nr, RA.ctypes.data, Nr, Rlambda.ctypes.data, Ru.ctypes.data, Nr, 0, N, work.ctypes.data, lwork, rwork.ctypes.data)
    # Ru[:] = Ru.conj().T

    Llambda, Lu = eig(cp.asnumpy(LV.conj().T @ LP1 @ LW @ cp.linalg.inv(LS)))
    Rlambda, Ru = eig(cp.asnumpy(cp.linalg.inv(RS) @ RV.conj().T @ RP1 @ RW))

    Llambda = cp.asarray(Llambda)
    Lu = cp.asarray(Lu)
    Rlambda = cp.asarray(Rlambda)
    Ru = cp.asarray(Ru)

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


def beyn_phi_gpu(LV, Lu, Llambda, RW, Ru, Rlambda, factor, type):

    N = LV.shape[0]

    phiL = LV @ Lu
    phiR = cp.linalg.solve(Ru, RW.conj().T)

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
    # finish = time.time()
    # print('time to calculate min_dEk: ', finish - start)
    
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


def check_imag_cond_vec(kref, kL, kR, phiL, phiR, M10, M01, max_imag, kside):

    Nk = len(kref)
    imag_cond = np.zeros(Nk, dtype=np.bool_)
    dEk_dk = np.zeros(Nk, dtype=phiL.dtype)

    Ikmax = np.count_nonzero(np.abs(np.imag(kref)) < np.max((0.5, max_imag)))
    Ikmax += Ikmax % 2

    pRpL = phiR @ phiL

    if kside == 'L':
        ind_k = np.argmin(np.abs(np.subtract.outer(kL, kR)), axis=1)
        for Ik in range(Ikmax):
            ind_kR = ind_k[Ik]
            dEk_dk[Ik] = -(phiR[ind_kR, :] @ (-1j * M10 * np.exp(-1j * kL[Ik]) + 1j * M01 * np.exp(1j * kL[Ik])) @ phiL[:, Ik]) / pRpL[ind_kR, Ik]
    else:
        ind_k = np.argmin(np.abs(np.subtract.outer(kR, kL)), axis=1)
        for Ik in range(Ikmax):
            ind_kL = ind_k[Ik]
            dEk_dk[Ik] = -(phiR[Ik, :] @ (-1j * M10 * np.exp(-1j * kR[Ik]) + 1j * M01 * np.exp(1j * kR[Ik])) @ phiL[:, ind_kL]) / pRpL[Ik, ind_kL]

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

    # start = time.time()
    imag_cond, dEk_dk = check_imag_cond_vec(kref, kL, kR, phiL, phiR, M10, M01, imag_limit, kside)
    # finish = time.time()
    # print('time to check imag cond: ', finish - start)

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

    # try:
    
    # LP0, LP1, RP0, RP1 = contour_integral(factor, matrix_blocks, M00.shape[0], R, type, YL=YL, YR=YR)
    # LV, LS, LW, RV, RS, RW = beyn_svd(LP0, RP0, eps_lim=1e-8)
    LP1, LV, LS, LW, RP1, RV, RS, RW = contour_svd(factor, matrix_blocks, M00.shape[0], R, type, YL=YL, YR=YR, eps_lim=1e-8)

    if LS.size == 0 or RS.size == 0:
        raise Exception("No singular values above the threshold")

    Lu, Llambda, Ru, Rlambda = beyn_eig(LV, LS, LW, LP1, RV, RS, RW, RP1)
    kL, kR, phiL, phiR = beyn_phi(LV, Lu, Llambda, RW, Ru, Rlambda, factor, type)
    Sigma, gR, min_dEk = beyn_sigma(kL, kR, phiL, phiR, M00, M01, M10, imag_lim, 2, type)
    
    # except Exception as e:

    #     print("Error in Beyn:")
    #     print(e)

    #     cond = np.nan
    #     Sigma = None
    #     gR = None

    return Sigma, gR, cond, min_dEk


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

    # LP1, LV, LS, LW, RP1, RV, RS, RW = contour_svd_gpu(factor, matrix_blocks, M00.shape[0], R, type, YL=YL, YR=YR, eps_lim=1e-8)

    # if LS.size == 0 or RS.size == 0:
    #     raise Exception("No singular values above the threshold")

    # Lu, Llambda, Ru, Rlambda = beyn_eig_gpu(LV, LS, LW, LP1, RV, RS, RW, RP1)
    LV, RW, Lu, Llambda, Ru, Rlambda = contour_svd_eig_lumi(factor, matrix_blocks, M00.shape[0], R, type, YL=YL, YR=YR, eps_lim=1e-8)
    kL, kR, phiL, phiR = beyn_phi_gpu(LV, Lu, Llambda, RW, Ru, Rlambda, factor, type)
    Sigma, gR, min_dEk = beyn_sigma_gpu(kL, kR, phiL, phiR, M00, M01, M10, imag_lim, 2, type)

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


def beyn_gpu(factor: int,
             M00,
             M01,
             M10,
             imag_lim,
             R,
             type,
             YL=None,
             YR=None):
    
    N00, N01, N10, matrix_blocks = extract_small_matrix_blocks_gpu(M00, M01, M10, factor, type)
    return beyn_new_gpu(factor, matrix_blocks, N00, N01, N10, imag_lim, R, type, YL=YL, YR=YR)
