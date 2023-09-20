# Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.
""" Generates data for strong scaling benchmarks for CPU kernels. """

import numpy as np
import numpy.typing as npt
import sys
import numba
import timeit
import os
import argparse

main_path = os.path.abspath(os.path.dirname(__file__))
parent_path = os.path.abspath(os.path.join(main_path, "..", ".."))
sys.path.append(parent_path)

from quatrex.utils import change_format
from GW.polarization.kernel import g2p_cpu
from GW.selfenergy.kernel import gw2s_cpu
from GW.polarization.initialization import gf_init

if __name__ == "__main__":
    # parse input arguments
    parser = argparse.ArgumentParser(description="Strong scaling benchmarks")
    parser.add_argument("-t",
                        "--type",
                        default="p_cpu_fft",
                        choices=["p_cpu_fft_inlined", "p_cpu_fft", "p_cpu_conv", "s_cpu_fft"],
                        required=False)

    # number of energy points
    ne = 400
    # number of orbitals -> around 0.0394*nao*nao are the nonzero amount of nnz
    nnz = 50000
    # choose number of orbitals and energy points
    parser.add_argument("-ne", "--num_energy", default=ne, required=False, type=int)
    parser.add_argument("-nnz", "--num_nonzero", default=nnz, required=False, type=int)
    parser.add_argument("-th", "--threads", default=28, required=False, type=int)
    parser.add_argument("-r", "--runs", default=20, required=False, type=int)
    args = parser.parse_args()
    print("Format: ", args.type)

    # number of repeats
    num_run = args.runs
    num_warm = 5
    # 28 thread cluster node
    num_threads = args.threads

    threads: npt.NDArray[np.int32] = np.arange(1, num_threads + 1, 1, dtype=np.int32)
    times: npt.NDArray[np.double] = np.empty((num_run, num_threads), dtype=np.double)
    speed_ups: npt.NDArray[np.double] = np.empty_like(times, dtype=np.double)
    nao = np.int32(np.sqrt(args.num_nonzero / 0.0394))
    # random seed
    seed = 2
    # create sparsity pattern
    # that fulfills that the transposed has the same sparsity pattern
    energy, rows, columns, gg, _, _ = gf_init.init_sparse(args.num_energy, nao, seed)
    ij2ji: npt.NDArray[np.int32] = change_format.find_idx_transposed(rows, columns)
    denergy: np.double = energy[1] - energy[0]
    pre_factor: np.complex128 = -1.0j * denergy / (np.pi)
    # number of orbitals can differ from the input
    # this is due to above creation of the sparsity pattern
    ne: np.int32 = energy.shape[0]
    no: np.int32 = gg.shape[0]

    print("Number of nnz: ", no)
    print("Number of energy points: ", ne)

    # randomize input data
    seed = 10
    rng = np.random.default_rng(seed)

    data_1ne_1 = rng.uniform(size=(no, ne)) + 1j * rng.uniform(size=(no, ne))
    data_1ne_2 = rng.uniform(size=(no, ne)) + 1j * rng.uniform(size=(no, ne))
    data_1ne_3 = rng.uniform(size=(no, ne)) + 1j * rng.uniform(size=(no, ne))
    data_1ne_4 = rng.uniform(size=(no, ne)) + 1j * rng.uniform(size=(no, ne))
    data_1ne_5 = rng.uniform(size=(no, ne)) + 1j * rng.uniform(size=(no, ne))
    data_1ne_6 = rng.uniform(size=(no, ne)) + 1j * rng.uniform(size=(no, ne))

    size_sparse = data_1ne_1.nbytes / (1024**3)
    print(f"Size of one sparse greens function in GB: {size_sparse:.2f} GB")

    for i in range(threads.shape[0]):
        # number of numba threads
        numba.set_num_threads(threads[i])
        print("Number of used threads: ", numba.get_num_threads())

        if args.type == "p_cpu_fft":
            time_warm = timeit.timeit(
                lambda: g2p_cpu.g2p_fft_cpu(pre_factor, ij2ji, data_1ne_1, data_1ne_2, data_1ne_3),
                number=num_warm) / num_warm
            time = timeit.repeat(
                stmt=lambda: g2p_cpu.g2p_fft_cpu(pre_factor, ij2ji, data_1ne_1, data_1ne_2, data_1ne_3),
                number=1,
                repeat=num_run)
        elif args.type == "p_cpu_fft_inlined":
            time_warm = timeit.timeit(
                lambda: g2p_cpu.g2p_fft_cpu_inlined(pre_factor, ij2ji, data_1ne_1, data_1ne_2, data_1ne_3),
                number=num_warm) / num_warm
            time = timeit.repeat(
                stmt=lambda: g2p_cpu.g2p_fft_cpu_inlined(pre_factor, ij2ji, data_1ne_1, data_1ne_2, data_1ne_3),
                number=1,
                repeat=num_run)
        elif args.type == "p_cpu_conv":
            time_warm = timeit.timeit(
                lambda: g2p_cpu.g2p_conv_cpu(pre_factor, ij2ji, data_1ne_1, data_1ne_2, data_1ne_3),
                number=num_warm) / num_warm
            time = timeit.repeat(
                stmt=lambda: g2p_cpu.g2p_conv_cpu(pre_factor, ij2ji, data_1ne_1, data_1ne_2, data_1ne_3),
                number=1,
                repeat=num_run)
        elif args.type == "s_cpu_fft":
            time_warm = timeit.timeit(lambda: gw2s_cpu.gw2s_fft_cpu(pre_factor, ij2ji, data_1ne_1, data_1ne_2,
                                                                    data_1ne_3, data_1ne_4, data_1ne_5, data_1ne_6),
                                      number=num_warm) / num_warm
            time = timeit.repeat(stmt=lambda: gw2s_cpu.gw2s_fft_cpu(pre_factor, ij2ji, data_1ne_1, data_1ne_2,
                                                                    data_1ne_3, data_1ne_4, data_1ne_5, data_1ne_6),
                                 number=1,
                                 repeat=num_run)
        else:
            raise ValueError("Invalid input argument")

        times[:, i] = np.array(time, dtype=np.double)
        speed_ups[:, i] = np.divide(np.mean(times[:, 0]), times[:, i])
        print("Time: ", np.mean(times[:, i]))

    output: npt.NDArray[np.double] = np.empty((1 + 2 * num_run + 1 + 1, threads.shape[0]), dtype=np.double)
    output[0, :] = threads
    output[1:num_run + 1, :] = times
    output[num_run + 1:2 * num_run + 1, :] = speed_ups
    output[2 * num_run + 1, 0] = size_sparse
    output[2 * num_run + 2, 0] = no
    output[2 * num_run + 2, 1] = ne
    save_path = os.path.join(main_path, "strong_" + args.type + ".npy")
    np.save(save_path, output)
