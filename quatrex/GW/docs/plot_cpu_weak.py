# Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.
"""Plotting script for weak scaling plots. """

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import os
import argparse
import scipy.stats as st
main_path = os.path.abspath(os.path.dirname(__file__))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Strong scaling benchmark plot"
    )
    parser.add_argument(
                    "-t", "--type",
                    default="p_mpi_cpu_fft_inlined",
                    choices=["p_cpu_fft_inlined",
                             "p_mpi_cpu_fft_inlined",
                             "s_mpi_cpu_fft_3",
                             "p_cpu_fft",
                             "p_cpu_conv",
                             "s_cpu_fft"],
                    required=False)
    parser.add_argument("-d", "--dimension", default="nnz",
                    choices=["energy", "nnz"], required=False)

    args = parser.parse_args()
    print("Format: ", args.type)
    print("Scaling over Energy/nnz: ", args.dimension)

    tmp = "_3500_5000"
    font_type = ""
    # font_size = "14"
    plt.rcParams["font.size"] = 20
    load_path = os.path.join(main_path, "weak_" + args.type + "_" + args.dimension + tmp +".npy")
    data: npt.NDArray[np.double] = np.load(load_path)

    num_run = (data.shape[0] - 4) // 2

    threads = data[0, :]
    num_threads = threads.shape[0]
    speed_up_mean = np.mean(data[num_run+1:2*num_run,:], axis=0)
    speed_up_std = np.std(data[num_run+1:2*num_run + 1,:], axis=0)
    size_sparse = data[2*num_run + 3,0]

    # for i in range(threads.size):
    #     print(st.t.interval(confidence=0.95, df=len(data[num_run+1:2*num_run,i])-1, loc=np.mean(data[num_run+1:2*num_run,i]), scale=st.sem(data[num_run+1:2*num_run,i])) )

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.errorbar(threads, speed_up_mean, yerr=speed_up_std, fmt="_k", capsize=5)

    # add line segment
    # ax.plot(threads, speed_up_mean, ':b', lw=2)

    # add ideal scaling
    ax.plot(threads, np.ones(num_threads))

    ax.set_xlabel("Number of Threads")
    ax.set_ylabel("Speed-up")
    ax.set_title("Weak Scaling of the Self-Energies Calculation")

    ax.text(num_threads - 15, 0.8, f"Initial Runtime: {np.mean(data[1:num_run+1,0]):.2f} s" +
            "\n" + f"Final Runtime: {np.mean(data[1:num_run+1,num_threads-1]):.2f} s" +
            "\n" + f"Initial Size: {np.int32(data[2*num_run + 2,0])} x {np.int32(data[2*num_run + 1,0])}")

    save_path = os.path.join(main_path, args.type + "_" + args.dimension + "_weak_"+ str(np.int32(data[2*num_run + 2,0])) + "_" + str(np.int32(data[2*num_run + 1,0])) + tmp +".pdf")
    plt.savefig(save_path, dpi=600)
