import cupy as cp
import cupyx as cpx
import numpy as np
import time

from timeit import repeat as benchmark


def random_complex(shape, rng: np.random.Generator):
    return rng.random(shape) + 1j * rng.random(shape)


def test_matmul(num_matrices: int, matrix_size: int, repeat: int = 100, validate: bool = True):

    rng = np.random.default_rng(42)

    A = random_complex((num_matrices, matrix_size, matrix_size), rng)
    B = random_complex((num_matrices, matrix_size, matrix_size), rng)
    if validate:
        C = np.matmul(A, B)

    A_d = cp.asarray(A)
    B_d = cp.asarray(B)
    C_d = cp.empty_like(A_d)

    def _func(A, B, C):
        for i in range(num_matrices):
            cp.matmul(A[i], B[i], out=C[i])
        return

    _func(A_d, B_d, C_d)
    if validate:
        assert np.allclose(C, C_d.get())

    runtimes = benchmark("_func(A_d, B_d, C_d); cp.cuda.stream.get_current_stream().synchronize()",
                         setup="cp.cuda.stream.get_current_stream().synchronize()",
                         globals={**globals(), **locals()}, number=1, repeat=repeat)
    print(f"Matmul non-batched: Avg {np.mean(runtimes) * 1000} ms, Median {np.median(runtimes) * 1000} ms", flush=True)
    flops = 8 * num_matrices * matrix_size**3
    median = np.median(runtimes)
    print(f"Matmul non-batched: {flops / median / 1e12} Tflop/s", flush=True)

    return


def test_inv(num_matrices: int, matrix_size: int, repeat: int = 100, validate: bool = True):

    rng = np.random.default_rng(42)

    A = random_complex((num_matrices, matrix_size, matrix_size), rng)
    if validate:
        B = np.linalg.inv(A)

    A_d = cp.asarray(A)
    B_d = cp.empty_like(A_d)

    def _func(A, B):
        for i in range(num_matrices):
            B[i] = cp.linalg.inv(A[i])
        return

    _func(A_d, B_d)
    if validate:
        assert np.allclose(B, B_d.get())

    runtimes = benchmark("_func(A_d, B_d); cp.cuda.stream.get_current_stream().synchronize()",
                         setup="cp.cuda.stream.get_current_stream().synchronize()",
                         globals={**globals(), **locals()}, number=1, repeat=repeat)
    print(f"Inv non-batched: Avg {np.mean(runtimes) * 1000} ms, Median {np.median(runtimes) * 1000} ms", flush=True)

    return


def test_bw0(num_matrices: int, matrix_size: int, repeat: int = 100, validate: bool = True):

    rng = np.random.default_rng(42)

    HD = random_complex((num_matrices, matrix_size, matrix_size), rng)
    SLD = random_complex((num_matrices, matrix_size, matrix_size), rng)
    SGD = random_complex((num_matrices, matrix_size, matrix_size), rng)
    gR = np.empty((num_matrices, matrix_size, matrix_size), dtype=np.complex128)
    gL = np.empty((num_matrices, matrix_size, matrix_size), dtype=np.complex128)
    gG = np.empty((num_matrices, matrix_size, matrix_size), dtype=np.complex128)
    if validate:
        gR[:] = np.linalg.inv(HD)
        gL[:] = gR @ SLD @ gR.transpose((0, 2, 1)).conjugate()
        gG[:] = gR @ SGD @ gR.transpose((0, 2, 1)).conjugate()

    HD_d = cp.asarray(HD)
    SLD_d = cp.asarray(SLD)
    SGD_d = cp.asarray(SGD)
    gR_d = cp.empty_like(gR)
    gL_d = cp.empty_like(gL)
    gG_d = cp.empty_like(gG)

    def _func(H, SL, SG, GR, GL, GG):
        for i in range(num_matrices):
            GR[i] = np.linalg.inv(H[i])
            GL[i] = GR[i] @ SL[i] @ GR[i].transpose().conjugate()
            GG[i] = GR[i] @ SG[i] @ GR[i].transpose().conjugate()
        return

    _func(HD_d, SLD_d, SGD_d, gR_d, gL_d, gG_d)
    if validate:
        assert np.allclose(gR, gR_d.get())
        assert np.allclose(gL, gL_d.get())
        assert np.allclose(gG, gG_d.get())

    runtimes = benchmark("_func(HD_d, SLD_d, SGD_d, gR_d, gL_d, gG_d); cp.cuda.stream.get_current_stream().synchronize()",
                         setup="cp.cuda.stream.get_current_stream().synchronize()",
                         globals={**globals(), **locals()}, number=1, repeat=repeat)
    print(f"Backwards pass first iteration non-batched: Avg {np.mean(runtimes) * 1000} ms, Median {np.median(runtimes) * 1000} ms", flush=True)

    return


def test_solve(num_matrices: int, matrix_size: int, repeat: int = 100, validate: bool = True):

    rng = np.random.default_rng(42)

    A = random_complex((num_matrices, matrix_size, matrix_size), rng)
    I = np.repeat(np.eye(matrix_size, dtype=np.complex128)[np.newaxis, :, :], num_matrices, axis = 0)
    if validate:
        B = np.linalg.solve(A, I)

    A_d = cp.asarray(A)
    I_d = cp.eye(matrix_size, dtype=np.complex128)
    B_d = cp.empty_like(A_d)

    def _func(A, B):
        for i in range(num_matrices):
            B[i] = cp.linalg.solve(A[i], I_d)
        return

    _func(A_d, B_d)
    if validate:
        assert np.allclose(B, B_d.get())

    runtimes = benchmark("_func(A_d, B_d); cp.cuda.stream.get_current_stream().synchronize()",
                         setup="cp.cuda.stream.get_current_stream().synchronize()",
                         globals={**globals(), **locals()}, number=1, repeat=repeat)
    print(f"Solve non-batched: Avg {np.mean(runtimes) * 1000} ms, Median {np.median(runtimes) * 1000} ms", flush=True)

    return


def test_matmul_batched(num_matrices: int, matrix_size: int, batch_size: int, repeat: int = 100, validate: bool = True):

    rng = np.random.default_rng(42)

    A = random_complex((num_matrices, matrix_size, matrix_size), rng)
    B = random_complex((num_matrices, matrix_size, matrix_size), rng)
    if validate:
        C = np.matmul(A, B)

    A_d = cp.asarray(A)
    B_d = cp.asarray(B)
    C_d = cp.empty_like(A_d)

    def _func(A, B, C):
        for i in range(0, num_matrices, batch_size):
            j = min(i + batch_size, num_matrices)
            cp.matmul(A[i:j], B[i:j], out=C[i:j])
        return
    
    _func(A_d, B_d, C_d)
    if validate:
        assert np.allclose(C, C_d.get())

    runtimes = benchmark("_func(A_d, B_d, C_d); cp.cuda.stream.get_current_stream().synchronize()",
                         setup="cp.cuda.stream.get_current_stream().synchronize()",
                         globals={**globals(), **locals()}, number=1, repeat=repeat)
    print(f"Matmul batched (size {batch_size}): Avg {np.mean(runtimes) * 1000} ms, Median {np.median(runtimes) * 1000} ms", flush=True)
    flops = 8 * num_matrices * matrix_size**3
    median = np.median(runtimes)
    print(f"Matmul batched (size {batch_size}): {flops / median / 1e12} Tflop/s", flush=True)

    return


def test_inv_batched(num_matrices: int, matrix_size: int, batch_size: int, repeat: int = 100, validate: bool = True):

    rng = np.random.default_rng(42)

    A = random_complex((num_matrices, matrix_size, matrix_size), rng)
    if validate:
        B = np.linalg.inv(A)

    A_d = cp.asarray(A)
    B_d = cp.empty((num_matrices, matrix_size, matrix_size), dtype=np.complex128)

    def _func(A, B):
        for i in range(0, num_matrices, batch_size):
            j = min(i + batch_size, num_matrices)
            B[i:j] = cp.linalg.inv(A[i:j])
        return
    
    _func(A_d, B_d)
    if validate:
        assert np.allclose(B, B_d.get())

    runtimes = benchmark("_func(A_d, B_d); cp.cuda.stream.get_current_stream().synchronize()",
                         setup="cp.cuda.stream.get_current_stream().synchronize()",
                         globals={**globals(), **locals()}, number=1, repeat=repeat)
    print(f"Inv batched (size {batch_size}): Avg {np.mean(runtimes) * 1000} ms, Median {np.median(runtimes) * 1000} ms", flush=True)

    return


def test_bw0_batched(num_matrices: int, matrix_size: int, batch_size: int, repeat: int = 100, validate: bool = True):

    rng = np.random.default_rng(42)

    HD = random_complex((num_matrices, matrix_size, matrix_size), rng)
    SLD = random_complex((num_matrices, matrix_size, matrix_size), rng)
    SGD = random_complex((num_matrices, matrix_size, matrix_size), rng)
    gR = np.empty((num_matrices, matrix_size, matrix_size), dtype=np.complex128)
    gL = np.empty((num_matrices, matrix_size, matrix_size), dtype=np.complex128)
    gG = np.empty((num_matrices, matrix_size, matrix_size), dtype=np.complex128)
    if validate:
        gR[:] = np.linalg.inv(HD)
        gL[:] = gR @ SLD @ gR.transpose((0, 2, 1)).conjugate()
        gG[:] = gR @ SGD @ gR.transpose((0, 2, 1)).conjugate()

    HD_d = cp.asarray(HD)
    SLD_d = cp.asarray(SLD)
    SGD_d = cp.asarray(SGD)
    gR_d = cp.empty_like(gR)
    gL_d = cp.empty_like(gL)
    gG_d = cp.empty_like(gG)

    def _func(H, SL, SG, GR, GL, GG):
        for i in range(0, num_matrices, batch_size):
            j = min(i + batch_size, num_matrices)
            GR[i:j] = np.linalg.inv(H[i:j])
            GL[i:j] = GR[i:j] @ SL[i:j] @ GR[i:j].transpose((0, 2, 1)).conjugate()
            GG[i:j] = GR[i:j] @ SG[i:j] @ GR[i:j].transpose((0, 2, 1)).conjugate()
        return

    _func(HD_d, SLD_d, SGD_d, gR_d, gL_d, gG_d)
    if validate:
        assert np.allclose(gR, gR_d.get())
        assert np.allclose(gL, gL_d.get())
        assert np.allclose(gG, gG_d.get())

    runtimes = benchmark("_func(HD_d, SLD_d, SGD_d, gR_d, gL_d, gG_d); cp.cuda.stream.get_current_stream().synchronize()",
                         setup="cp.cuda.stream.get_current_stream().synchronize()",
                         globals={**globals(), **locals()}, number=1, repeat=repeat)
    print(f"Backwards pass first iteration batched (size {batch_size}): Avg {np.mean(runtimes) * 1000} ms, Median {np.median(runtimes) * 1000} ms", flush=True)

    return


def test_solve_batched(num_matrices: int, matrix_size: int, batch_size: int, repeat: int = 100, validate: bool = True):

    rng = np.random.default_rng(42)

    A = random_complex((num_matrices, matrix_size, matrix_size), rng)
    I = np.repeat(np.eye(matrix_size, dtype=np.complex128)[np.newaxis, :, :], num_matrices, axis = 0)
    if validate:
        B = np.linalg.solve(A, I)

    A_d = cp.asarray(A)
    I_d = cp.repeat(cp.eye(matrix_size, dtype=np.complex128)[np.newaxis, :, :], batch_size, axis = 0)
    B_d = cp.empty((num_matrices, matrix_size, matrix_size), dtype=np.complex128)

    def _func(A, B):
        for i in range(0, num_matrices, batch_size):
            j = min(i + batch_size, num_matrices)
            B[i:j] = cp.linalg.solve(A[i:j], I_d[:j-i])
        return
    
    _func(A_d, B_d)
    if validate:
        assert np.allclose(B, B_d.get())

    runtimes = benchmark("_func(A_d, B_d); cp.cuda.stream.get_current_stream().synchronize()",
                         setup="cp.cuda.stream.get_current_stream().synchronize()",
                         globals={**globals(), **locals()}, number=1, repeat=repeat)
    print(f"Solve batched (size {batch_size}): Avg {np.mean(runtimes) * 1000} ms, Median {np.median(runtimes) * 1000} ms", flush=True)

    return


if __name__ == "__main__":

    # num_matrices = 100
    # # for matrix_size in (16, 32, 64, 128, 256, 512, 1024, 2048):
    # # for matrix_size in (96, 416, 932, 1864):
    # for matrix_size in (2048, ):
    #     print(f"Testing with matrix size {matrix_size}...", flush=True)
    #     # test_matmul(num_matrices, matrix_size, repeat=5, validate=False)
    #     # for batch_size in (1, 2, 10, 20, 50):
    #     #     test_matmul_batched(num_matrices, matrix_size, batch_size, repeat=5, validate=False)
    #     # test_inv(num_matrices, matrix_size, repeat=5, validate=False)
    #     # for batch_size in (1, 2, 10, 20, 50):
    #     #     test_inv_batched(num_matrices, matrix_size, batch_size, repeat=5, validate=False)
    #     # test_solve(num_matrices, matrix_size, repeat=5, validate=False)
    #     # for batch_size in (1, 2, 10, 20, 50):
    #     #     test_solve_batched(num_matrices, matrix_size, batch_size, repeat=5, validate=False)
    #     test_bw0(num_matrices, matrix_size,  repeat=5, validate=False)
    #     for batch_size in (1, 2, 10, 20, 50, 100):
    #         test_bw0_batched(num_matrices, matrix_size, batch_size, repeat=5, validate=False)

    num_matrices = 100
    matrix_size = 512
    rng = np.random.default_rng(42)
    A = random_complex((num_matrices, matrix_size, matrix_size), rng)
    B = random_complex((num_matrices, matrix_size, matrix_size), rng)
    A_dev = cp.asarray(A)
    B_dev = cp.asarray(B)
    BH_dev = cp.empty_like(B_dev)
    C_dev = cp.empty_like(A_dev)

    print("Testing matmul...", flush=True)
    for _ in range(3):
        cp.matmul(A_dev, B_dev, out=C_dev)
    print("Testing matmul with transpose...", flush=True)
    for _ in range(3):
        cp.matmul(A_dev, B_dev.transpose((0, 2, 1)), out=C_dev)
    print("Testing matmul with conjugate transpose...", flush=True)
    for _ in range(3):
        cp.matmul(A_dev, B_dev.transpose((0, 2, 1)).conjugate(), out=C_dev)
    print("Testing matmul with conjugate transpose 2...", flush=True)
    for _ in range(3):
        cp.matmul(A_dev, B_dev.conjugate().transpose((0, 2, 1)), out=C_dev)
    print("Testing matmul with conjugate transpose 3...", flush=True)
    for _ in range(3):
        cp.conjugate(B_dev.transpose((0, 2, 1)), out=BH_dev)
        cp.matmul(A_dev, BH_dev, out=C_dev)

