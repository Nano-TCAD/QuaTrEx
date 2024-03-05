import cupy as cp
import numpy as np
# import mkl
import scipy.io  as io
import os
import timeit

from concurrent.futures import ThreadPoolExecutor

from quatrex.OBC.beyn_new import extract_small_matrix_blocks, contour_integral, beyn_svd, beyn_eig, beyn_phi, beyn_sigma
from quatrex.OBC.beyn_new import contour_integral_batched
from quatrex.OBC.beyn_new_gpu import contour_integral_gpu, contour_integral_batched_gpu, beyn_gpu, beyn_mix
from quatrex.OBC.beyn_new_gpu import extract_small_matrix_blocks_gpu, beyn_svd_gpu, beyn_eig_gpu, beyn_phi_gpu, beyn_sigma_gpu
from quatrex.OBC.beyn_batched import beyn_batched_gpu, beyn_batched_gpu_2


def _cmplx_random(shape, rng):
    return rng.random(shape) + 1j * rng.random(shape)

def _norm(a, b):
    if np.linalg.norm(a) == 0:
        return np.linalg.norm(b)
    return np.linalg.norm(a - b) / np.linalg.norm(a)


def test_cpu():

    rng = np.random.default_rng(42)

    num_energies = 20

    # for N  in (128, 256, 512, 1024, 2048, 4096):
    for N in (416,):

        M00 = _cmplx_random((num_energies, N, N), rng)
        M01 = _cmplx_random((num_energies, N, N), rng)
        M10 = _cmplx_random((num_energies, N, N), rng)
        imag_lim = 0.5
        R = 1.0

        M00_d = cp.asarray(M00)
        M01_d = cp.asarray(M01)
        M10_d = cp.asarray(M10)
        
        for factor in (1, 2, 4):
        # for factor in (4, ):

            # small_N = N // factor
            # if factor * small_N < 100:
            #     NM = round(3 * small_N / 4)
            # else:
            #     NM = round(small_N / 2)
            # NM = factor * NM
            # YL = cp.random.rand(num_energies, small_N, NM)
            # YR = cp.random.rand(num_energies, NM, small_N)
            YL = [None for _ in range(num_energies)]
            YR = [None for _ in range(num_energies)]

            for type in ('L', 'R'):
            # for type in ('L',):

                print(f'N = {N}, factor = {factor}, type = {type}', flush=True)

                # beyn_batched_gpu(factor, M00_d, M01_d, M10_d, imag_lim, R, type)

                # beyn(factor, M00, M01, M10, imag_lim, R, type)
                # runtimes = timeit.repeat("beyn(factor, M00, M01, M10, imag_lim, R, type)",
                #                          globals={**globals(), **locals()}, number=1, repeat=10)
                # print(f'Beyn CPU: Avg {np.mean(runtimes)}s, Median {np.median(runtimes)}s')

                beyn_mix(factor, M00_d[0], M01_d[0], M10_d[0], imag_lim, R, type)

                # continue

                def _test():
                    Sigma = [None for _ in range(num_energies)]
                    gR = [None for _ in range(num_energies)]
                    cond = [np.nan for _ in range(num_energies)]
                    min_dEk = [1e8 for _ in range(num_energies)]
                    for i in range(num_energies):
                        Sigma[i], gR[i], cond[i], min_dEk[i] = beyn_mix(factor, M00_d[i], M01_d[i], M10_d[i], imag_lim, R, type, YL[i], YR[i])
                        # S, g, c, m = beyn_mix(factor, M00_d[i], M01_d[i], M10_d[i], imag_lim, R, type, YL[i], YR[i])
                        # print(np.allclose(Sigma[i], S), _norm(Sigma[i], S))
                        # print(np.allclose(gR[i], g), _norm(gR[i], g))
                        # print(np.allclose(cond[i], c), _norm(cond[i], c))
                        # print(np.allclose(min_dEk[i], m), _norm(min_dEk[i], m))
                    return Sigma, gR, cond, min_dEk

                # s_ref, g_ref, c_ref, m_ref = _test()
                # s_val, g_val, c_val, m_val = beyn_batched_gpu(factor, M00_d, M01_d, M10_d, imag_lim, R, type, YL, YR)
                # for i in range(num_energies):
                #     print(f'Error at energy {i} Sigma: {_norm(s_ref[i], s_val[i])}')
                #     print(f'Error at energy {i} gR: {_norm(g_ref[i], g_val[i])}')
                #     print(f'Error at energy {i} cond: {_norm(c_ref[i], c_val[i])}')
                #     print(f'Error at energy {i} min_dEk: {_norm(m_ref[i], m_val[i])}')

                # continue
                
                runtimes = timeit.repeat("_test(); cp.cuda.stream.get_current_stream().synchronize()",
                                         setup="cp.cuda.stream.get_current_stream().synchronize()",
                                         globals={**globals(), **locals()}, number=1, repeat=1)
                print(f'Beyn mixed: Avg {np.mean(runtimes)}s, Median {np.median(runtimes)}s', flush=True)

                # def _test2():
                #     # mkl.set_num_threads(1)
                #     with ThreadPoolExecutor(max_workers=20) as executor:
                #         executor.map(lambda x: beyn_gpu(factor, M00_d[x], M01_d[x], M10_d[x], imag_lim, R, type), range(num_energies))
                # runtimes = timeit.repeat("_test2(); cp.cuda.stream.get_current_stream().synchronize()",
                #                             setup="cp.cuda.stream.get_current_stream().synchronize()",
                #                             globals={**globals(), **locals()}, number=1, repeat=3)
                # print(f'Beyn GPU 20 workers: Avg {np.mean(runtimes)}s, Median {np.median(runtimes)}s', flush=True)
                
                # def _test3():
                #     # mkl.set_num_threads(1)
                #     with ThreadPoolExecutor(max_workers=20) as executor:
                #         executor.map(lambda x: beyn_mix(factor, M00_d[x], M01_d[x], M10_d[x], imag_lim, R, type), range(num_energies))
                # runtimes = timeit.repeat("_test3(); cp.cuda.stream.get_current_stream().synchronize()",
                #                             setup="cp.cuda.stream.get_current_stream().synchronize()",
                #                             globals={**globals(), **locals()}, number=1, repeat=3)
                # print(f'Beyn mixed 20 workers: Avg {np.mean(runtimes)}s, Median {np.median(runtimes)}s', flush=True)

                def _test4():
                    # mkl.set_num_threads(28)
                    beyn_batched_gpu(factor, M00_d, M01_d, M10_d, imag_lim, R, type)
                    cp.cuda.stream.get_current_stream().synchronize()
                runtimes = timeit.repeat("_test4()", globals={**globals(), **locals()}, number=1, repeat=1)
                print(f'Beyn batched GPU: Avg {np.mean(runtimes)}s, Median {np.median(runtimes)}s', flush=True)

                def _test5():
                    # mkl.set_num_threads(28)
                    beyn_batched_gpu_2(factor, M00_d, M01_d, M10_d, imag_lim, R, type)
                    cp.cuda.stream.get_current_stream().synchronize()
                runtimes = timeit.repeat("_test5()", globals={**globals(), **locals()}, number=1, repeat=1)
                print(f'Beyn batched GPU 2: Avg {np.mean(runtimes)}s, Median {np.median(runtimes)}s', flush=True)

                continue


                beyn_gpu(factor, M00_d, M01_d, M10_d, imag_lim, R, type)
                runtimes = timeit.repeat("beyn_gpu(factor, M00_d, M01_d, M10_d, imag_lim, R, type); cp.cuda.stream.get_current_stream().synchronize()",
                                         setup="cp.cuda.stream.get_current_stream().synchronize()",
                                         globals={**globals(), **locals()}, number=1, repeat=10)
                print(f'Beyn GPU: Avg {np.mean(runtimes)}s, Median {np.median(runtimes)}s')
                continue

                N00, N01, N10, matrix_blocks = extract_small_matrix_blocks_gpu(M00, M01, M10, factor, type)
                runtimes = timeit.repeat("extract_small_matrix_blocks_gpu(M00, M01, M10, factor, type)",
                                         globals={**globals(), **locals()}, number=1, repeat=10)
                print(f'extract_small_matrix_blocks: Avg {np.mean(runtimes)}s, Median {np.median(runtimes)}s')

                # big_N = N00.shape[0]
                # small_N = big_N // factor
                # if factor * small_N < 100:
                #     NM = round(3 * small_N / 4)
                # else:
                #     NM = round(small_N / 2)
                # NM = factor * NM
                # YL = rng.random((small_N, NM))
                # YR = rng.random((NM, small_N))

                # LP0r, LP1r, RP0r, RP1r = contour_integral(factor, matrix_blocks, N00.shape[0], R, type, YL, YR)
                # # # LP0, LP1, RP0, RP1 = contour_integral_batched(factor, matrix_blocks, N00.shape[0], R, type, YL, YR)
                # matrix_blocks_dev = cp.asarray(matrix_blocks)
                # YL_dev = cp.asarray(YL)
                # YR_dev = cp.asarray(YR)
                # LP0_dev, LP1_dev, RP0_dev, RP1_dev = contour_integral_gpu(factor, matrix_blocks_dev, N00.shape[0], R, type)
                # LP0_dev, LP1_dev, RP0_dev, RP1_dev = contour_integral_batched_gpu(factor, matrix_blocks_dev, N00.shape[0], R, type, YL_dev, YR_dev)
                # LP0 = cp.asnumpy(LP0_dev)
                # LP1 = cp.asnumpy(LP1_dev)
                # RP0 = cp.asnumpy(RP0_dev)
                # RP1 = cp.asnumpy(RP1_dev)
                # for val, ref in zip((LP0r, LP1r, RP0r, RP1r), (LP0, LP1, RP0, RP1)):
                #     assert np.allclose(val, ref)

                LP0, LP1, RP0, RP1 = contour_integral_batched_gpu(factor, matrix_blocks, N00.shape[0], R, type,)

                # LP0, LP1, RP0, RP1 = contour_integral(factor, matrix_blocks, N00.shape[0], R, type)
                runtimes = timeit.repeat("contour_integral_batched_gpu(factor, matrix_blocks, N00.shape[0], R, type); cp.cuda.stream.get_current_stream().synchronize()",
                                         globals={**globals(), **locals()}, number=1, repeat=10)
                # runtimes = timeit.repeat("contour_integral(factor, matrix_blocks, N00.shape[0], R, type)",
                #                          globals={**globals(), **locals()}, number=1, repeat=10)
                print(f'contour_integral: Avg {np.mean(runtimes)}s, Median {np.median(runtimes)}s')

                LV, LS, LW, RV, RS, RW = beyn_svd_gpu(LP0, RP0, eps_lim=1e-8)
                runtimes = timeit.repeat("beyn_svd_gpu(LP0, RP0, eps_lim=1e-8); cp.cuda.stream.get_current_stream().synchronize()",
                                         globals={**globals(), **locals()}, number=1, repeat=10)
                print(f'beyn_svd: Avg {np.mean(runtimes)}s, Median {np.median(runtimes)}s')

                Lu, Llambda, Ru, Rlambda = beyn_eig_gpu(LV, LS, LW, LP1, RV, RS, RW, RP1)
                runtimes = timeit.repeat("beyn_eig_gpu(LV, LS, LW, LP1, RV, RS, RW, RP1); cp.cuda.stream.get_current_stream().synchronize()",
                                         globals={**globals(), **locals()}, number=1, repeat=10)
                print(f'beyn_eig: Avg {np.mean(runtimes)}s, Median {np.median(runtimes)}s')

                kL, kR, phiL, phiR = beyn_phi_gpu(LV, Lu, Llambda, RW, Ru, Rlambda, factor, type)
                runtimes = timeit.repeat("beyn_phi_gpu(LV, Lu, Llambda, RW, Ru, Rlambda, factor, type); cp.cuda.stream.get_current_stream().synchronize()",
                                         globals={**globals(), **locals()}, number=1, repeat=10)
                print(f'beyn_phi: Avg {np.mean(runtimes)}s, Median {np.median(runtimes)}s')

                Sigma, gR, min_dEk = beyn_sigma_gpu(kL, kR, phiL, phiR, N00, N01, N10, imag_lim, 2, type)
                runtimes = timeit.repeat("beyn_sigma_gpu(kL, kR, phiL, phiR, N00, N01, N10, imag_lim, 2, type); cp.cuda.stream.get_current_stream().synchronize()",
                                         globals={**globals(), **locals()}, number=1, repeat=10)
                print(f'beyn_sigma: Avg {np.mean(runtimes)}s, Median {np.median(runtimes)}s')


if __name__ == '__main__':
    test_cpu()
