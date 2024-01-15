import numpy as np
import scipy.io  as io
import os

from quatrex.OBC.beyn_new import extract_small_matrix_blocks
from quatrex.OBC.beyn_new import extract_small_matrix_blocks_gpu
from quatrex.OBC.contour_integral import contour_integral, contour_integral_batched
from quatrex.OBC.contour_integral import contour_integral_gpu, contour_integral_batched_gpu


def _contour_integral_cpu(M00, M01, M10, R, factor, side):
    N = M00.shape[0] // factor
    _, _, _, matrix_blocks= extract_small_matrix_blocks(M00, M01, M10, factor, side)
    P0C1, P1C1 = contour_integral(N, factor, matrix_blocks, 3.0, 1.0, side)
    P0C2, P1C2 = contour_integral(N, factor, matrix_blocks, 1.0 / R, -1.0, side)
    return P0C1, P1C1, P0C2, P1C2


def _contour_integral_batched_cpu(M00, M01, M10, R, factor, side):
    N = M00.shape[0] // factor
    _, _, _, matrix_blocks= extract_small_matrix_blocks(M00, M01, M10, factor, side)
    P0C1, P1C1 = contour_integral_batched(N, factor, matrix_blocks, 3.0, 1.0, side)
    P0C2, P1C2 = contour_integral_batched(N, factor, matrix_blocks, 1.0 / R, -1.0, side)
    return P0C1, P1C1, P0C2, P1C2

def _contour_integral_gpu(M00, M01, M10, R, factor, side):
    import cupy as cp
    N = M00.shape[0] // factor
    M00 = cp.asarray(M00)
    M01 = cp.asarray(M01)
    M10 = cp.asarray(M10)
    _, _, _, matrix_blocks= extract_small_matrix_blocks_gpu(M00, M01, M10, factor, side)
    P0C1, P1C1 = contour_integral_gpu(N, factor, matrix_blocks, 3.0, 1.0, side)
    P0C2, P1C2 = contour_integral_gpu(N, factor, matrix_blocks, 1.0 / R, -1.0, side)
    P0C1 = cp.asnumpy(P0C1)
    P1C1 = cp.asnumpy(P1C1)
    P0C2 = cp.asnumpy(P0C2)
    P1C2 = cp.asnumpy(P1C2)
    return P0C1, P1C1, P0C2, P1C2


def _contour_integral_batched_gpu(M00, M01, M10, R, factor, side):
    import cupy as cp
    N = M00.shape[0] // factor
    M00 = cp.asarray(M00)
    M01 = cp.asarray(M01)
    M10 = cp.asarray(M10)
    _, _, _, matrix_blocks= extract_small_matrix_blocks_gpu(M00, M01, M10, factor, side)
    P0C1, P1C1 = contour_integral_batched_gpu(N, factor, matrix_blocks, 3.0, 1.0, side)
    P0C2, P1C2 = contour_integral_batched_gpu(N, factor, matrix_blocks, 1.0 / R, -1.0, side)
    P0C1 = cp.asnumpy(P0C1)
    P1C1 = cp.asnumpy(P1C1)
    P0C2 = cp.asnumpy(P0C2)
    P1C2 = cp.asnumpy(P1C2)
    return P0C1, P1C1, P0C2, P1C2


def _test(func):

    dir_path = os.path.dirname(os.path.realpath(__file__))
    folder = os.path.join(dir_path, 'data')
    inputs = dict()
    for name in ('M00', 'M01', 'M10'):
        filename = os.path.join(folder, f'inp_{name}.mat')
        mat = io.loadmat(filename)
        inputs[name] = mat[name]

    M00 = inputs['M00']
    M01 = inputs['M01']
    M10 = inputs['M10']
    
    for factor in (1, 2, 4):
        for side in ('L', 'R'):

            print(f'factor = {factor}, side = {side}')

            R = 1.0

            P0C1, P1C1, P0C2, P1C2 = func(M00, M01, M10, R, factor, side)

            outputs = dict()
            for name in ("P0C1", "P1C1", "P0C2", "P1C2"):
                filename = os.path.join(folder, f'out_{name}_{factor}_{side}.mat')
                mat = io.loadmat(filename)
                outputs[name] = mat[name]
            
            assert np.allclose(P0C1, outputs['P0C1'])
            assert np.allclose(P1C1, outputs['P1C1'])
            assert np.allclose(P0C2, outputs['P0C2'])
            assert np.allclose(P1C2, outputs['P1C2'])


def test_contour_integral_cpu():
    _test(_contour_integral_cpu)


def test_contour_integral_batched_cpu():
    _test(_contour_integral_batched_cpu)


def test_contour_integral_gpu():
    _test(_contour_integral_gpu)


def test_contour_integral_batched_gpu():
    _test(_contour_integral_batched_gpu)


if __name__ == '__main__':
    test_contour_integral_cpu()
    test_contour_integral_batched_cpu()
    test_contour_integral_gpu()
    test_contour_integral_batched_gpu()
