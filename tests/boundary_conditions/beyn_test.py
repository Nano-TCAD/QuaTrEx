import numpy as np
import scipy.io  as io
import os

from quatrex.OBC.beyn_new import beyn, beyn_gpu
from quatrex.OBC.beyn_new import extract_small_matrix_blocks, extract_small_matrix_blocks_gpu
from quatrex.OBC.beyn_new import contour_integral_batched, contour_integral_batched_gpu
from quatrex.OBC.beyn_new import beyn_svd, beyn_svd_gpu
from quatrex.OBC.beyn_new import beyn_eig, beyn_eig_gpu
from quatrex.OBC.beyn_new import beyn_phi, beyn_phi_gpu
from quatrex.OBC.beyn_new import beyn_sigma, beyn_sigma_gpu
from quatrex.OBC.beyn_new import contour_svd, contour_svd_gpu


def relative_error(a, b):
    return np.linalg.norm(a - b) / np.linalg.norm(a)


def test_beyn_cpu():

    dir_path = os.path.dirname(os.path.realpath(__file__))
    folder = os.path.join(dir_path, 'test_data')
    inputs = dict()
    for name in ('M00', 'M01', 'M10'):
        filename = os.path.join(folder, f'inp_{name}.mat')
        mat = io.loadmat(filename)
        inputs[name] = mat[name]

    M00 = inputs['M00']
    M01 = inputs['M01']
    M10 = inputs['M10']
    imag_lim = 0.5
    R = 1.0

    print(M00.shape)
    
    for factor in (1, 2, 4):
        for type in ('L', 'R'):

            print(f'factor = {factor}, type = {type}')

            filename = os.path.join(folder, f'out_YL_{factor}_{type}.mat')
            mat = io.loadmat(filename)
            YL = mat['YL']
            filename = os.path.join(folder, f'out_YR_{factor}_{type}.mat')
            mat = io.loadmat(filename)
            YR = mat['YR']

            Sigma, gR, cond, min_dEk = beyn(factor, M00, M01, M10, imag_lim, R, type, YL, YR)

            outputs = dict()
            for name in ("Sigma", "gR", "min_dEk"):
                filename = os.path.join(folder, f'out_{name}_{factor}_{type}.mat')
                mat = io.loadmat(filename)
                outputs[name] = mat[name]

            # assert np.allclose(Sigma, outputs['Sigma'])
            # assert np.allclose(gR, outputs['gR'])
            # assert np.allclose(min_dEk, outputs['min_dEk'])
            print(relative_error(Sigma, outputs['Sigma']))
            print(relative_error(gR, outputs['gR']))
            print(relative_error(min_dEk, outputs['min_dEk']))


def test_beyn_gpu():
    import cupy as cp

    dir_path = os.path.dirname(os.path.realpath(__file__))
    folder = os.path.join(dir_path, 'test_data')
    inputs = dict()
    for name in ('M00', 'M01', 'M10'):
        filename = os.path.join(folder, f'inp_{name}.mat')
        mat = io.loadmat(filename)
        inputs[name] = mat[name]

    M00 = cp.asarray(inputs['M00'])
    M01 = cp.asarray(inputs['M01'])
    M10 = cp.asarray(inputs['M10'])
    imag_lim = 0.5
    R = 1.0

    print(M00.shape)
    
    for factor in (1, 2, 4):
        for type in ('L', 'R'):

            print(f'factor = {factor}, type = {type}')

            filename = os.path.join(folder, f'out_YL_{factor}_{type}.mat')
            mat = io.loadmat(filename)
            YL = cp.asarray(mat['YL'])
            filename = os.path.join(folder, f'out_YR_{factor}_{type}.mat')
            mat = io.loadmat(filename)
            YR = cp.asarray(mat['YR'])

            Sigma, gR, cond, min_dEk = beyn_gpu(factor, M00, M01, M10, imag_lim, R, type, YL, YR)
            Sigma = cp.asnumpy(Sigma)
            gR = cp.asnumpy(gR)
            min_dEk = cp.asnumpy(min_dEk)

            outputs = dict()
            for name in ("Sigma", "gR", "min_dEk"):
                filename = os.path.join(folder, f'out_{name}_{factor}_{type}.mat')
                mat = io.loadmat(filename)
                outputs[name] = mat[name]

            # assert np.allclose(Sigma, outputs['Sigma'])
            # assert np.allclose(gR, outputs['gR'])
            # assert np.allclose(min_dEk, outputs['min_dEk'])
            print(relative_error(Sigma, outputs['Sigma']))
            print(relative_error(gR, outputs['gR']))
            print(relative_error(min_dEk, outputs['min_dEk']))


def test_beyn_analytic():
    import cupy as cp

    dir_path = os.path.dirname(os.path.realpath(__file__))
    folder = os.path.join(dir_path, 'test_data')
    inputs = dict()
    for name in ('M00', 'M01', 'M10'):
        filename = os.path.join(folder, f'inp_{name}.mat')
        mat = io.loadmat(filename)
        inputs[name] = mat[name]

    M00_host = inputs['M00']
    M01_host = inputs['M01']
    M10_host = inputs['M10']
    M00_dev = cp.asarray(M00_host)
    M01_dev = cp.asarray(M01_host)
    M10_dev = cp.asarray(M10_host)
    imag_lim = 0.5
    R = 1.0
    cond = 0
    min_dEk = 1e8

    print(M00_host.shape)
    
    for factor in (1, 2, 4):
        for type in ('L', 'R'):

            print(f'factor = {factor}, type = {type}')

            filename = os.path.join(folder, f'out_YL_{factor}_{type}.mat')
            mat = io.loadmat(filename)
            YL_host = mat['YL']
            YL_dev = cp.asarray(YL_host)
            filename = os.path.join(folder, f'out_YR_{factor}_{type}.mat')
            mat = io.loadmat(filename)
            YR_host = mat['YR']
            YR_dev = cp.asarray(YR_host)

            N00_cpu, N01_cpu, N10_cpu, matrix_blocks_cpu = extract_small_matrix_blocks(M00_host, M01_host, M10_host, factor, type)
            N00_gpu, N01_gpu, N10_gpu, matrix_blocks_gpu = extract_small_matrix_blocks_gpu(M00_dev, M01_dev, M10_dev, factor, type)

            assert np.allclose(N00_cpu, cp.asnumpy(N00_gpu))
            assert np.allclose(N01_cpu, cp.asnumpy(N01_gpu))
            assert np.allclose(N10_cpu, cp.asnumpy(N10_gpu))
            assert np.allclose(matrix_blocks_cpu, cp.asnumpy(matrix_blocks_gpu))

            LP1_cpu, LV_cpu, LS_cpu, LW_cpu, RP1_cpu, RV_cpu, RS_cpu, RW_cpu = contour_svd(factor, matrix_blocks_cpu, M00_host.shape[0], R, type, YL=YL_host, YR=YR_host, eps_lim=1e-8)
            LP1_gpu, LV_gpu, LS_gpu, LW_gpu, RP1_gpu, RV_gpu, RS_gpu, RW_gpu = contour_svd_gpu(factor, matrix_blocks_gpu, M00_dev.shape[0], R, type, YL=YL_dev, YR=YR_dev, eps_lim=1e-8)

            # LP0_cpu, LP1_cpu, RP0_cpu, RP1_cpu = contour_integral_batched(factor, matrix_blocks_cpu, N00_cpu.shape[0], R, type, YL=YL_host, YR=YR_host)
            # LP0_gpu, LP1_gpu, RP0_gpu, RP1_gpu = contour_integral_batched_gpu(factor, matrix_blocks_gpu, N00_gpu.shape[0], R, type, YL=YL_dev, YR=YR_dev)

            # assert np.allclose(LP0_cpu, cp.asnumpy(LP0_gpu))
            assert np.allclose(LP1_cpu, cp.asnumpy(LP1_gpu))
            # assert np.allclose(RP0_cpu, cp.asnumpy(RP0_gpu))
            assert np.allclose(RP1_cpu, cp.asnumpy(RP1_gpu))

            # LV_cpu, LS_cpu, LW_cpu, RV_cpu, RS_cpu, RW_cpu = beyn_svd(LP0_cpu, RP0_cpu, eps_lim=1e-8)
            # LV_gpu, LS_gpu, LW_gpu, RV_gpu, RS_gpu, RW_gpu = beyn_svd_gpu(LP0_gpu, RP0_gpu, eps_lim=1e-8)

            # assert np.allclose(LV_cpu @ LS_cpu @ LW_cpu.conj().T, LP0_cpu)
            # assert np.allclose(RV_cpu @ RS_cpu @ RW_cpu.conj().T, RP0_cpu)
            # assert np.allclose(LV_gpu @ LS_gpu @ LW_gpu.conj().T, LP0_gpu)
            # assert np.allclose(RV_gpu @ RS_gpu @ RW_gpu.conj().T, RP0_gpu)

            # assert np.allclose(LV_cpu, cp.asnumpy(LV_gpu))
            assert np.allclose(LS_cpu, cp.asnumpy(LS_gpu))
            # assert np.allclose(LW_cpu, cp.asnumpy(LW_gpu))
            # assert np.allclose(RV_cpu, cp.asnumpy(RV_gpu))
            assert np.allclose(RS_cpu, cp.asnumpy(RS_gpu))
            # assert np.allclose(RW_cpu, cp.asnumpy(RW_gpu))

            Lu_cpu, Llambda_cpu, Ru_cpu, Rlambda_cpu = beyn_eig(LV_cpu, LS_cpu, LW_cpu, LP1_cpu, RV_cpu, RS_cpu, RW_cpu, RP1_cpu)
            Lu_gpu, Llambda_gpu, Ru_gpu, Rlambda_gpu = beyn_eig_gpu(LV_gpu, LS_gpu, LW_gpu, LP1_gpu, RV_gpu, RS_gpu, RW_gpu, RP1_gpu)

            assert np.allclose(Lu_cpu @ np.diag(Llambda_cpu) @ np.linalg.inv(Lu_cpu), LV_cpu.conj().T @ LP1_cpu @ LW_cpu @ np.linalg.inv(LS_cpu))
            assert np.allclose(Ru_cpu @ np.diag(Rlambda_cpu) @ np.linalg.inv(Ru_cpu), np.linalg.inv(RS_cpu) @ RV_cpu.conj().T @ RP1_cpu @ RW_cpu)
            assert np.allclose(Lu_gpu @ cp.diag(Llambda_gpu) @ cp.linalg.inv(Lu_gpu), LV_gpu.conj().T @ LP1_gpu @ LW_gpu @ cp.linalg.inv(LS_gpu))
            assert np.allclose(Ru_gpu @ cp.diag(Rlambda_gpu) @ cp.linalg.inv(Ru_gpu), cp.linalg.inv(RS_gpu) @ RV_gpu.conj().T @ RP1_gpu @ RW_gpu)
            
            # assert np.allclose(Lu_cpu, cp.asnumpy(Lu_gpu))
            # assert np.allclose(Llambda_cpu, cp.asnumpy(Llambda_gpu))
            # assert np.allclose(Ru_cpu, cp.asnumpy(Ru_gpu))
            # assert np.allclose(Rlambda_cpu, cp.asnumpy(Rlambda_gpu))

            kL_cpu, kR_cpu, phiL_cpu, phiR_cpu = beyn_phi(LV_cpu, Lu_cpu, Llambda_cpu, RW_cpu, Ru_cpu, Rlambda_cpu, factor, type)
            kL_gpu, kR_gpu, phiL_gpu, phiR_gpu = beyn_phi_gpu(LV_gpu, Lu_gpu, Llambda_gpu, RW_gpu, Ru_gpu, Rlambda_gpu, factor, type)

            # assert np.allclose(kL_cpu, cp.asnumpy(kL_gpu))
            # assert np.allclose(kR_cpu, cp.asnumpy(kR_gpu))
            # assert np.allclose(phiL_cpu, cp.asnumpy(phiL_gpu))
            # assert np.allclose(phiR_cpu, cp.asnumpy(phiR_gpu))

            Sigma_cpu, gR_cpu, min_dEk_cpu = beyn_sigma(kL_cpu, kR_cpu, phiL_cpu, phiR_cpu, N00_cpu, N01_cpu, N10_cpu, imag_lim, 2, type)
            Sigma_gpu, gR_gpu, min_dEk_gpu = beyn_sigma_gpu(kL_gpu, kR_gpu, phiL_gpu, phiR_gpu, N00_gpu, N01_gpu, N10_gpu, imag_lim, 2, type)

            print(relative_error(cp.asnumpy(Sigma_gpu), Sigma_cpu))
            assert np.allclose(Sigma_cpu, cp.asnumpy(Sigma_gpu))
            assert np.allclose(gR_cpu, cp.asnumpy(gR_gpu))
            assert np.allclose(min_dEk_cpu, cp.asnumpy(min_dEk_gpu))


if __name__ == '__main__':
    # test_beyn_cpu()
    # test_beyn_gpu()
    test_beyn_analytic()
