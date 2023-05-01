"""
Generate data for strong scaling benchmarks for the CPU version of the polarization
"""
import numpy as np
import numpy.typing as npt
import sys
import numba
import timeit
import os
import argparse

main_path = os.path.abspath(os.path.dirname(__file__))
parent_path = os.path.abspath(os.path.join(main_path, "..", "..", ".."))
sys.path.append(parent_path)

from utils import change_format
from GW.polarization.kernel import g2p_cpu
from GW.polarization.initialization import gf_init


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Strong scaling benchmarks"
    )
    parser.add_argument("-t", "--type", default="cpu_fft",
                    choices=["cpu_fft_inlined", "cpu_fft",
                                "cpu_conv"], required=False)

    args = parser.parse_args()
    print("Format: ", args.type)

    # number of repeats
    num_run = 20
    num_warm = 5
    # 28 thread cluster node
    num_threads = 28

    threads: npt.NDArray[np.int32] = np.arange(1, num_threads+1, 1, dtype=np.int32)
    times: npt.NDArray[np.double] = np.empty((num_run, num_threads), dtype=np.double)
    speed_ups: npt.NDArray[np.double] = np.empty_like(times, dtype=np.double)

    # number of energy points
    ne = 400
    # number of orbitals -> around 0.0394*nao*nao are the nonzero amount of nnz
    nnz = 50000
    nao = np.int32(np.sqrt(nnz / 0.0394))
    # random seed
    seed = 2
    energy, rows, columns, gg, gl, gr = gf_init.init_sparse(ne, nao, seed)
    ij2ji:      npt.NDArray[np.int32]   = change_format.find_idx_transposed(rows, columns)
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
    data_1ne_2 = rng.uniform(
    size=(no, ne)) + 1j * rng.uniform(size=(no, ne))
    data_1ne_3 = rng.uniform(
    size=(no, ne)) + 1j * rng.uniform(size=(no, ne))

    size_sparse = data_1ne_1.nbytes / (1024**3)
    print(f"Size of one sparse greens function in GB: {size_sparse:.2f} GB")

    for i in range(threads.shape[0]):
        # number of numba threads
        numba.set_num_threads(threads[i])
        print("Number of used threads: ", numba.get_num_threads())

        if args.type == "cpu_fft":
            time_warm = timeit.timeit(
                lambda: g2p_cpu.g2p_fft_cpu(denergy, ij2ji, data_1ne_1, 
                data_1ne_2, data_1ne_3),
                number=num_warm) / num_warm
            time = timeit.repeat(
                stmt=lambda: g2p_cpu.g2p_fft_cpu(denergy, ij2ji, data_1ne_1,
                data_1ne_2, data_1ne_3),
                number=1, repeat=num_run)
        elif args.type == "cpu_fft_inlined":
            time_warm = timeit.timeit(
                lambda: g2p_cpu.g2p_fft_cpu_inlined(denergy, ij2ji, data_1ne_1, 
                data_1ne_2, data_1ne_3),
                number=num_warm) / num_warm
            time = timeit.repeat(
                stmt=lambda: g2p_cpu.g2p_fft_cpu_inlined(denergy, ij2ji, data_1ne_1,
                data_1ne_2, data_1ne_3),
                number=1, repeat=num_run)
        elif args.type == "cpu_conv":
            time_warm = timeit.timeit(
                lambda: g2p_cpu.g2p_conv_cpu(denergy, ij2ji, data_1ne_1, data_1ne_2, data_1ne_3),
                number=num_warm) / num_warm
            time = timeit.repeat(
                stmt=lambda: g2p_cpu.g2p_conv_cpu(denergy, ij2ji, data_1ne_1,
                data_1ne_2, data_1ne_3),
                number=1, repeat=num_run)
        else:
            raise ValueError(
            "First command line argument has to be either str(fft) or str(conv)")

        times[:,i] = np.array(time, dtype=np.double)
        speed_ups[:,i] = np.divide(np.mean(times[:,0]), times[:,i])
        print("Time: ", np.mean(times[:,i]))

    output: npt.NDArray[np.double] = np.empty((1 + 2 * num_run + 1 + 1,threads.shape[0]), dtype=np.double)
    output[0,:] = threads
    output[1:num_run+1,:] = times
    output[num_run+1:2*num_run + 1,:] = speed_ups
    output[2*num_run + 1,0] = size_sparse
    output[2*num_run + 2,0] = no
    output[2*num_run + 2,1] = ne
    save_path = os.path.join(main_path, "strong_" + args.type + ".npy")
    np.save(save_path, output)
