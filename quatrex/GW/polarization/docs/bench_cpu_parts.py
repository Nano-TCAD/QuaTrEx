"""Generate data for weak scaling cpu plot
"""
import numpy as np
import numpy.typing as npt
import sys
from scipy import fft
import numba
import timeit
import os

# ghetto solution from ghetto coder
main_path = os.path.abspath(os.path.dirname(__file__))
parent_path = os.path.abspath(os.path.join(main_path, ".."))
gggparent_path = os.path.abspath(os.path.join(main_path, "..", "..", ".."))
sys.path.append(parent_path)
sys.path.append(gggparent_path)

from GW_SE_python.polarization.initialization import gf_init
from GW_SE_python.gold_solution import read_solution
from GW_SE_python.polarization.sparse import helper

if __name__ == "__main__":

    # number of repeats
    num_run = 20
    num_warm = 5
    # 28 thread cluster node
    num_threads = 28
    num_parts = 4

    threads: npt.NDArray[np.int32] = np.arange(1, num_threads+1, 1, dtype=np.int32)
    times: npt.NDArray[np.double] = np.empty((4*num_run, num_threads), dtype=np.double)
    speed_ups: npt.NDArray[np.double] = np.empty_like(times, dtype=np.double)


    # number of energy points
    ne = 400
    # number of orbitals -> around 0.0394*nao*nao are the nonzero amount of nnz
    nnz = 50000
    nao = np.int32(np.sqrt(nnz / 0.0394))
    # random seed
    seed = 2
    energy, rows, columns, gg, gl, gr = gf_init.init_sparse(ne, nao, seed)
    energy:     npt.NDArray[np.double]  = np.squeeze(energy)
    rows:       npt.NDArray[np.int32]   = np.squeeze(rows)
    columns:    npt.NDArray[np.int32]   = np.squeeze(columns)
    ij2ji:      npt.NDArray[np.int32]   = read_solution.find_idx_transposed(rows, columns)
    denergy:    np.double               = energy[1] - energy[0]
    ne:         np.int32                = energy.shape[0]
    no:         np.int32                = gg.shape[0]

    print("Number of nnz: ", no)
    print("Number of energy points: ", ne)

    # randomize input data
    seed = 10
    rng = np.random.default_rng(seed)

    data_1ne_1 = rng.uniform(
    size=(no, ne)) + 1j * rng.uniform(size=(no, ne))
    data_2ne_1 = rng.uniform(
    size=(no, 2*ne)) + 1j * rng.uniform(size=(no, 2*ne))
    data_2ne_2 = rng.uniform(
    size=(no, 2*ne)) + 1j * rng.uniform(size=(no, 2*ne))
    pre_factor: np.complex128 = rng.uniform(
    size=1)[0] + 1j * rng.uniform(size=1)[0]

    size_sparse = data_1ne_1.nbytes / (1024**3)
    print(f"Size of one sparse greens function in GB: {size_sparse:.2f} GB")

    for i in range(threads.shape[0]):
        # number of numba threads
        numba.set_num_threads(threads[i])
        print("Number of used threads: ", numba.get_num_threads())


        time_warm = timeit.timeit(
            stmt=lambda: helper.fft_numba(gg,2*ne, no),
            number=num_warm) / num_warm
        time1 = timeit.repeat(
            stmt=lambda: helper.fft_numba(gg,2*ne, no),
            number=1, repeat=num_run)

        time_warm = timeit.timeit(
             stmt=lambda: helper.reversal_transpose(data_2ne_1, ij2ji),
            number=num_warm) / num_warm
        time2 = timeit.repeat(
             stmt=lambda: helper.reversal_transpose(data_2ne_1, ij2ji),
            number=1, repeat=num_run)

        time_warm = timeit.timeit(
            stmt=lambda: helper.elementmul(data_2ne_1, data_2ne_2),
            number=num_warm) / num_warm
        time3 = timeit.repeat(
            stmt=lambda: helper.elementmul(data_2ne_1, data_2ne_2),
            number=1, repeat=num_run)

        time_warm = timeit.timeit(
             stmt=lambda: helper.scalarmul_ifft(data_2ne_1, pre_factor, ne, no),
            number=num_warm) / num_warm
        time4 = timeit.repeat(
            stmt=lambda: helper.scalarmul_ifft(data_2ne_1, pre_factor, ne, no),
            number=1, repeat=num_run)


        times[:num_run,i]               = np.array(time1, dtype=np.double)
        times[num_run:2*num_run,i]      = np.array(time2, dtype=np.double)
        times[2*num_run:3*num_run,i]    = np.array(time3, dtype=np.double)
        times[3*num_run:4*num_run,i]    = np.array(time4, dtype=np.double)


        speed_ups[:num_run,i]               = np.divide(np.mean(times[          :num_run    ,0]),  times[           :num_run    ,i])
        speed_ups[num_run:2*num_run,i]      = np.divide(np.mean(times[num_run   :2*num_run  ,0]),  times[num_run    :2*num_run  ,i])
        speed_ups[2*num_run:3*num_run,i]    = np.divide(np.mean(times[2*num_run :3*num_run  ,0]),  times[2*num_run  :3*num_run  ,i])
        speed_ups[3*num_run:4*num_run,i]    = np.divide(np.mean(times[3*num_run :4*num_run  ,0]),  times[3*num_run  :4*num_run  ,i])

    output: npt.NDArray[np.double] = np.empty((1 + 8 * num_run + 1,threads.shape[0]), dtype=np.double)
    output[0,:] = threads
    output[1:num_run*4+1,:] = times
    output[num_run*4+1:8*num_run + 1,:] = speed_ups
    output[8*num_run + 1,0] = size_sparse

    save_path = os.path.join(main_path, "cpu_parts.npy")
    np.save(save_path, output)
