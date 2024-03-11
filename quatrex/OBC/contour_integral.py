# Copyright 2023-2024 ETH Zurich and the QuaTrEx authors. All rights reserved.

import numpy as np

try:
    import cupy as cp
    import cupyx as cpx

    # NOTE: The following kernel is for the original contour integral for double the number of theta points
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
    def contour_batched(T, matrix_blocks, z, factor, batch_size, z_size, b_size, isL):

        idx = cpx.jit.blockIdx.x * cpx.jit.blockDim.x + cpx.jit.threadIdx.x
        if idx < batch_size * z_size * b_size * b_size:

            ie = idx // (z_size * b_size * b_size)
            ijk = idx % (z_size * b_size * b_size)
            i = ijk // (b_size * b_size)
            jk = ijk % (b_size * b_size)
            j = jk // b_size
            k = jk % b_size
            # t_idx = i * b_size * b_size + j * b_size + k
            t_idx = idx
            m_idx_0 = ie * (2 * factor + 1) * b_size * b_size

            z_i = z[i]

            if isL:
                for l in range(2 * factor + 1):
                    m_idx = m_idx_0 + l * b_size * b_size + j * b_size + k
                    T[t_idx] += matrix_blocks[m_idx] * z_i ** (factor - l)
            else:
                for l in range(2 * factor + 1):
                    m_idx = m_idx_0 + l * b_size * b_size + j * b_size + k
                    T[t_idx] += matrix_blocks[m_idx] * z_i ** (l - factor)
    

    @cpx.jit.rawkernel()
    def contour_combo(T, matrix_blocks, theta, R, factor, z_size, b_size, isL):

        idx = cpx.jit.blockIdx.x * cpx.jit.blockDim.x + cpx.jit.threadIdx.x
        if idx < 3 * z_size * b_size * b_size:

            ie = idx // (z_size * b_size * b_size)
            ijk = idx % (z_size * b_size * b_size)
            i = ijk // (b_size * b_size)
            jk = ijk % (b_size * b_size)
            j = jk // b_size
            k = jk % b_size
            # t_idx = i * b_size * b_size + j * b_size + k
            t_idx = idx

            c = 0
            r = cp.power(R[ie], 1.0 / factor)
            z_i = c + r * cp.exp(1j * theta[i])

            if isL:
                for l in range(2 * factor + 1):
                    m_idx = l * b_size * b_size + j * b_size + k
                    T[t_idx] += matrix_blocks[m_idx] * z_i ** (factor - l)
            else:
                for l in range(2 * factor + 1):
                    m_idx = l * b_size * b_size + j * b_size + k
                    T[t_idx] += matrix_blocks[m_idx] * z_i ** (l - factor)
    
    theta_min = 0
    theta_max = 2 * np.pi
    NT = 51

    theta = cp.linspace(theta_min, theta_max, NT)
    dtheta = cp.hstack((theta[1] - theta[0], theta[2:] - theta[:-2], theta[-1] - theta[-2])) / 2

    global_theta = theta
    global_dtheta = dtheta

    # global_z = cp.empty((3 * len(dtheta),), dtype=dtheta.dtype)
    # global_dz_dtheta = cp.empty((3 * len(dtheta),), dtype=dtheta.dtype)

    # rr = 1e4
    # factor = 4
    # for i, (R, csign) in enumerate([(3,0, 1.0), (1.0 / rr, -1.0), (10.0 / rr, -1.0)]):
    #     c = 0
    #     r = cp.power(R, 1.0 / factor)

    #     z = c + r * cp.exp(1j * theta)
    #     dz_dtheta = (csign * 1j * r) * cp.exp(1j * theta)

    #     global_z[i * len(dtheta) : (i+1) * len(dtheta)] = z
    #     global_dz_dtheta[i * len(dtheta) : (i+1) * len(dtheta)] = dz_dtheta
    

except (ImportError, ModuleNotFoundError):
    pass


def contour_integral(N: int,  # Reduced block size, i.e., N = block_size // factor
                     factor: int,  # Block size reduction factor
                     matrix_blocks: np.ndarray,  # Reduced matrix blocks
                     R: float,  # Radius parameter (?)
                     csign: int,  # Sign parameter (?)
                     side: str,  # Left or right side {'L', 'R'}
                     ):

    theta_min = 0
    theta_max = 2 * np.pi
    NT = 51

    theta = np.linspace(theta_min, theta_max, NT)
    dtheta = np.hstack((theta[1] - theta[0], theta[2:] - theta[:-2], theta[-1] - theta[-2])) / 2

    c = 0
    r = np.power(R, 1.0 / factor)

    z = c + r * np.exp(1j * theta)
    dz_dtheta = (csign * 1j * r) * np.exp(1j * theta)

    P0 = np.zeros((N, N), dtype=np.complex128)
    P1 = np.zeros((N, N), dtype=np.complex128)

    for i in range(len(z)):

        T = np.zeros((N, N), dtype=np.complex128)
        for j in range(2 * factor + 1):
            if side == 'L':
                T = T + matrix_blocks[j] * z[i] ** (factor - j)
            else:
                T = T + matrix_blocks[j] * z[i] ** (j - factor)

        iT = np.linalg.inv(T)

        P0 += iT*dz_dtheta[i]*dtheta[i]/(2*np.pi*1j)
        P1 += iT*z[i]*dz_dtheta[i]*dtheta[i]/(2*np.pi*1j)

    return P0, P1


def contour_integral_gpu(N: int,  # Reduced block size, i.e., N = block_size // factor
                         factor: int,  # Block size reduction factor
                         matrix_blocks: np.ndarray,  # Reduced matrix blocks
                         R: float,  # Radius parameter (?)
                         csign: int,  # Sign parameter (?)
                         side: str,  # Left or right side {'L', 'R'}
                         ):

    theta_min = 0
    theta_max = 2 * np.pi
    NT = 51

    theta = cp.linspace(theta_min, theta_max, NT)
    dtheta = cp.hstack((theta[1] - theta[0], theta[2:] - theta[:-2], theta[-1] - theta[-2])) / 2

    c = 0
    r = np.power(R, 1.0 / factor)

    z = c + r * cp.exp(1j * theta)
    dz_dtheta = (csign * 1j * r) * cp.exp(1j * theta)

    P0 = cp.zeros((N, N), dtype=np.complex128)
    P1 = cp.zeros((N, N), dtype=np.complex128)

    for i in range(len(z)):

        T = cp.zeros((N, N), dtype=np.complex128)
        for j in range(2 * factor + 1):
            if side == 'L':
                T = T + matrix_blocks[j] * z[i] ** (factor - j)
            else:
                T = T + matrix_blocks[j] * z[i] ** (j - factor)

        iT = cp.linalg.inv(T)

        P0 += iT*dz_dtheta[i]*dtheta[i]/(2*np.pi*1j)
        P1 += iT*z[i]*dz_dtheta[i]*dtheta[i]/(2*np.pi*1j)

    return P0, P1


def contour_integral_batched(N: int,  # Reduced block size, i.e., N = block_size // factor
                             factor: int,  # Block size reduction factor
                             matrix_blocks: np.ndarray,  # Reduced matrix blocks
                             R: float,  # Radius parameter (?)
                             csign: int,  # Sign parameter (?)
                             side: str,  # Left or right side {'L', 'R'}
                             ):

    theta_min = 0
    theta_max = 2 * np.pi
    NT = 51

    theta = np.linspace(theta_min, theta_max, NT)
    dtheta = np.hstack((theta[1] - theta[0], theta[2:] - theta[:-2], theta[-1] - theta[-2])) / 2

    c = 0
    r = np.power(R, 1.0 / factor)

    z = c + r * np.exp(1j * theta)
    dz_dtheta = (csign * 1j * r) * np.exp(1j * theta)

    T = np.zeros((len(z), N, N), dtype=np.complex128)

    for i in range(len(z)):
        for j in range(2 * factor + 1):
            if side == 'L':
                T[i] = T[i] + matrix_blocks[j] * z[i] ** (factor - j)
            else:
                T[i] = T[i] + matrix_blocks[j] * z[i] ** (j - factor)

    iT = np.linalg.inv(T)

    P0 = np.sum(iT*(dz_dtheta*dtheta/(2*np.pi*1j)).reshape(len(z), 1, 1), axis=0)
    P1 = np.sum(iT*(z*dz_dtheta*dtheta/(2*np.pi*1j)).reshape(len(z), 1, 1), axis=0)

    return P0, P1


def contour_integral_batched_gpu(N: int,  # Reduced block size, i.e., N = block_size // factor
                                 factor: int,  # Block size reduction factor
                                 matrix_blocks: np.ndarray,  # Reduced matrix blocks
                                 R: float,  # Radius parameter (?)
                                 csign: int,  # Sign parameter (?)
                                 side: str,  # Left or right side {'L', 'R'}
                                 ):

    theta_min = 0
    theta_max = 2 * np.pi
    NT = 51

    theta = cp.linspace(theta_min, theta_max, NT)
    dtheta = cp.hstack((theta[1] - theta[0], theta[2:] - theta[:-2], theta[-1] - theta[-2])) / 2

    c = 0
    r = np.power(R, 1.0 / factor)

    z = c + r * cp.exp(1j * theta)
    dz_dtheta = (csign * 1j * r) * cp.exp(1j * theta)

    T = cp.zeros((len(z), N, N), dtype=np.complex128)

    num_threads = 512
    num_blocks = (len(z) * N * N + num_threads - 1) // num_threads
    contour[num_blocks, num_threads](T.reshape(-1),
                                     matrix_blocks.reshape(-1),
                                     z,
                                     factor,
                                     len(z),
                                     N,
                                     np.bool_(side=='L'))

    iT = cp.linalg.inv(T)

    P0 = cp.sum(iT*(dz_dtheta*dtheta/(2*np.pi*1j)).reshape(len(z), 1, 1), axis=0)
    P1 = cp.sum(iT*(z*dz_dtheta*dtheta/(2*np.pi*1j)).reshape(len(z), 1, 1), axis=0)

    return P0, P1


def contour_integral_batched_combo_gpu(N: int,  # Reduced block size, i.e., N = block_size // factor
                                 factor: int,  # Block size reduction factor
                                 matrix_blocks: np.ndarray,  # Reduced matrix blocks
                                 R: float,  # Radius parameter (?)
                                 csign: int,  # Sign parameter (?)
                                 side: str,  # Left or right side {'L', 'R'}
                                 ):

    R_dev = cp.asarray(R)
    # T = cp.zeros((3, len(global_theta), N, N), dtype=np.complex128)
    T = cp.ones((3, len(global_theta), N, N), dtype=np.complex128)

    num_threads = 512
    num_blocks = (3 * len(global_theta) * N * N + num_threads - 1) // num_threads
    contour_combo[num_blocks, num_threads](T.reshape(-1),
                                     matrix_blocks.reshape(-1),
                                     global_theta,
                                     R_dev,
                                     factor,
                                     len(global_theta),
                                     N,
                                     np.bool_(side=='L'))

    iT = cp.linalg.inv(T)

    c = 0
    r = np.power(R[0], 1.0 / factor)
    # z = c + r * cp.exp(1j * theta)
    # dz_dtheta = (csign * 1j * r) * cp.exp(1j * theta)
    multiplier = (csign[0] * 1j * r) / (2.0 * np.pi * 1j)
    P0C0 = cp.sum(iT[0] * (cp.exp(1j * global_theta) * global_dtheta * multiplier).reshape(len(global_theta), 1, 1), axis=0)
    P1C0 = cp.sum(iT[0] * (cp.exp(1j * global_theta) * cp.exp(1j * global_theta) * global_dtheta * (multiplier * r)).reshape(len(global_theta), 1, 1), axis=0)

    r = np.power(R[1], 1.0 / factor)
    multiplier = (csign[1] * 1j * r) / (2.0 * np.pi * 1j)
    P0C1 = cp.sum(iT[1] * (cp.exp(1j * global_theta) * global_dtheta * multiplier).reshape(len(global_theta), 1, 1), axis=0)
    P1C1 = cp.sum(iT[1] * (cp.exp(1j * global_theta) * cp.exp(1j * global_theta) * global_dtheta * (multiplier * r)).reshape(len(global_theta), 1, 1), axis=0)

    r = np.power(R[2], 1.0 / factor)
    multiplier = (csign[2] * 1j * r) / (2.0 * np.pi * 1j)
    P0C2 = cp.sum(iT[2] * (cp.exp(1j * global_theta) * global_dtheta * multiplier).reshape(len(global_theta), 1, 1), axis=0)
    P1C2 = cp.sum(iT[2] * (cp.exp(1j * global_theta) * cp.exp(1j * global_theta) * global_dtheta * (multiplier * r)).reshape(len(global_theta), 1, 1), axis=0)


    return P0C0, P0C1, P0C2, P1C0, P1C1, P1C2


def contour_integral_batched_squared_gpu(N: int,  # Reduced block size, i.e., N = block_size // factor
                                         factor: int,  # Block size reduction factor
                                         matrix_blocks: np.ndarray,  # Reduced matrix blocks
                                         R: float,  # Radius parameter (?)
                                         csign: int,  # Sign parameter (?)
                                         side: str,  # Left or right side {'L', 'R'}
                                         ):

    batch_size = matrix_blocks.shape[0]
    
    theta_min = 0
    theta_max = 2 * np.pi
    NT = 51

    theta = cp.linspace(theta_min, theta_max, NT)
    dtheta = cp.hstack((theta[1] - theta[0], theta[2:] - theta[:-2], theta[-1] - theta[-2])) / 2

    c = 0
    r = np.power(R, 1.0 / factor)

    z = c + r * cp.exp(1j * theta)
    dz_dtheta = (csign * 1j * r) * cp.exp(1j * theta)

    T = cp.zeros((batch_size, len(z), N, N), dtype=np.complex128)

    num_threads = 512
    num_blocks = (batch_size * len(z) * N * N + num_threads - 1) // num_threads
    contour_batched[num_blocks, num_threads](T.reshape(-1),
                                     matrix_blocks.reshape(-1),
                                     z,
                                     factor,
                                     batch_size,
                                     len(z),
                                     N,
                                     np.bool_(side=='L'))

    iT = cp.linalg.inv(T)

    P0 = cp.sum(iT*(dz_dtheta*dtheta/(2*np.pi*1j)).reshape(1, len(z), 1, 1), axis=1)
    P1 = cp.sum(iT*(z*dz_dtheta*dtheta/(2*np.pi*1j)).reshape(1, len(z), 1, 1), axis=1)

    return P0, P1
