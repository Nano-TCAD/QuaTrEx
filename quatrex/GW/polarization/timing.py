"""Time the different functions used to calculate polarization"""
import sys
from sparse import helper
from sparse import g2p_sparse
from gold_solution import read_solution
import numpy as np
import cupy as cp
import cupyx
from cupyx.profiler import benchmark
import numba
from scipy import fft
import timeit

# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, "../")
if __name__ == "__main__":

    # limit the number of threads
    if numba.get_num_threads() >= 12:
        numba.config.NUMBA_NUM_THREADS = 12
    print("Number of used threads: ", numba.config.NUMBA_NUM_THREADS)

    # prepare data for timing
    energy, rows, columns, gf, _ = read_solution.load_pg(
        "gold_solution/data_GP.mat")
    energy = np.squeeze(energy)
    rows = np.squeeze(rows)
    columns = np.squeeze(columns)
    ne: int = energy.shape[0]
    ij2ji = read_solution.find_idx_transposed(rows, columns)
    ij2ji_reversal = read_solution.find_idx_rt(rows, columns, ne)
    denergy: np.double = energy[1] - energy[0]

    # generate random data input
    seed = 10
    rng = np.random.default_rng(seed)
    sp = np.shape(gf[0])
    data_1ne_1 = rng.uniform(
        size=(gf[0].shape[0], ne)) + 1j * rng.uniform(size=(gf[0].shape[0], ne))
    data_1ne_2 = rng.uniform(
        size=(gf[0].shape[0], ne)) + 1j * rng.uniform(size=(gf[0].shape[0], ne))
    data_1ne_3 = rng.uniform(
        size=(gf[0].shape[0], ne)) + 1j * rng.uniform(size=(gf[0].shape[0], ne))
    data_2ne_1 = rng.uniform(
        size=(gf[0].shape[0], 2*ne)) + 1j * rng.uniform(size=(gf[0].shape[0], 2*ne))
    data_2ne_2 = rng.uniform(
        size=(gf[0].shape[0], 2*ne)) + 1j * rng.uniform(size=(gf[0].shape[0], 2*ne))
    pre_fatore: np.cdouble = rng.uniform(
        size=1)[0] + 1j * rng.uniform(size=1)[0]

    # number of repeats
    num_run = 10
    num_warm = 5

    # read out cpu/gpu
    try:
        xpu = sys.argv[1]
    except IndexError:
        print("Needs a command line argument")
        print("Use default cpu")
        xpu = "cpu"

    # testing on cpu or gpu
    if xpu == "cpu":
        # call every jit compiled function once before timing
        _ = helper.elementmul(data_2ne_1, data_2ne_2)
        _ = helper.scalarmul(data_1ne_1, pre_fatore)
        _ = helper.reversal_transpose(data_2ne_1, ij2ji)

        time_warm = timeit.timeit(
            lambda: g2p_sparse.g2p_fft_cpu(denergy, ij2ji, data_1ne_1, data_1ne_2, data_1ne_3, numba.get_num_threads()),
            number=num_warm) / num_warm
        time_tot = timeit.timeit(
            lambda: g2p_sparse.g2p_fft_cpu(denergy, ij2ji, data_1ne_1,
            data_1ne_2, data_1ne_3, numba.get_num_threads()),
            number=num_run) / num_run
        print("Time warm up: ", time_warm)
        print("Time total: ", time_tot)


        # print(benchmark(read_solution.load_pg, ("gold_solution/data_GP.mat",),
        #       n_repeat=num_run, n_warmup=num_warm).to_str(show_gpu=False))
        # print(benchmark(read_solution.find_idx_transposed, (rows, columns),
        #       n_repeat=num_run, n_warmup=num_warm).to_str(show_gpu=False))
        # print(benchmark(g2p_sparse.g2p_conv_cpu, (denergy, ij2ji, data_1ne_1, data_1ne_2,
        #       data_1ne_3,  numba.get_num_threads()), n_repeat=num_run, n_warmup=num_warm).to_str(show_gpu=False))
        # print(benchmark(g2p_sparse.g2p_fft_cpu, (denergy, ij2ji, data_1ne_1, data_1ne_2,
        #       data_1ne_3,  numba.get_num_threads()), n_repeat=num_run, n_warmup=num_warm).to_str(show_gpu=False))
        # print(benchmark(helper.scalarmul, (data_1ne_1, pre_fatore),
        #       n_repeat=num_run, n_warmup=num_warm).to_str(show_gpu=False))
        # print(benchmark(helper.elementmul, (data_2ne_1, data_2ne_2),
        #       n_repeat=num_run, n_warmup=num_warm).to_str(show_gpu=False))
        # print(benchmark(helper.reversal_transpose, (data_2ne_1, ij2ji),
        #       n_repeat=num_run, n_warmup=num_warm).to_str(show_gpu=False))
        # print(benchmark(fft.ifft, (data_2ne_1,), {"axis": 1, "workers": numba.get_num_threads()},
        #       n_repeat=num_run, n_warmup=num_warm).to_str(show_gpu=False))
        # print(benchmark(helper.scalarmul, (data_1ne_1, pre_fatore),
        #     n_repeat=num_run, n_warmup=num_warm).to_str(show_gpu=False))
        # print(benchmark(helper.scalarmul_ifft, (data_2ne_1, pre_fatore, ne, numba.get_num_threads()),
        #       n_repeat=num_run, n_warmup=num_warm).to_str(show_gpu=False))

    elif xpu == "gpu":
        # load data to gpu
        ij2ji_gpu = cp.asarray(ij2ji)
        data_1ne_1_gpu = cp.asarray(data_1ne_1)
        data_1ne_2_gpu = cp.asarray(data_1ne_2)
        data_1ne_3_gpu = cp.asarray(data_1ne_3)
        data_2ne_1_gpu = cp.asarray(data_2ne_1)
        data_2ne_2_gpu = cp.asarray(data_2ne_2)

        # benchmark gpu implementation
        print(benchmark(g2p_sparse.g2p_fft_gpu, (denergy, ij2ji_gpu,
              data_1ne_1_gpu, data_1ne_2_gpu, data_1ne_3_gpu), n_repeat=num_run, n_warmup=num_warm).to_str(show_gpu=True))

        # benchmark used cupy functions
        def reversal_transpose(inp: cp.ndarray) -> cp.ndarray:
            return cp.roll(cp.flip(inp, axis=1), 1, axis=1)[ij2ji, :]

        # print(benchmark(cp.fft.fft, (data_1ne_1_gpu,), {"n": 2*ne, "axis": 1},
        #       n_repeat=num_run, n_warmup=num_warm).to_str(show_gpu=True))
        # print(benchmark(reversal_transpose, (data_2ne_1_gpu,),
        #       n_repeat=num_run, n_warmup=num_warm).to_str(show_gpu=True))
        # print(benchmark(cp.multiply, (data_2ne_1_gpu, data_2ne_2_gpu),
        #       n_repeat=num_run, n_warmup=num_warm).to_str(show_gpu=True))
        # print(benchmark(cp.fft.ifft, (data_2ne_1_gpu,), {"axis": 1},
        #       n_repeat=num_run, n_warmup=num_warm).to_str(show_gpu=True))
        # print(benchmark(cp.multiply, (data_1ne_1_gpu, pre_fatore),
        #       n_repeat=num_run, n_warmup=num_warm).to_str(show_gpu=True))

    else:
        raise ValueError(
            "Command line argument has to be either str(cpu) or str(gpu)")
