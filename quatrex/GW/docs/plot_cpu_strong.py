# Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import os
import argparse

main_path = os.path.abspath(os.path.dirname(__file__))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Strong scaling benchmark plot"
    )
    parser.add_argument(
                    "-t", "--type",
                    default="s_mpi_cpu_fft_3",
                    choices=["p_cpu_fft_inlined",
                             "p_mpi_cpu_fft_inlined",
                             "s_mpi_cpu_fft_3",
                             "p_cpu_fft",
                             "p_cpu_conv",
                             "s_cpu_fft"],
                    required=False)


    args = parser.parse_args()
    print("Format: ", args.type)

    if args.type in ["p_cpu_fft_inlined", "p_mpi_cpu_fft_inlined", "p_cpu_fft", "p_cpu_conv"]:
        calctype = "Polarizations"
    else:
        calctype = "Self-Energies"


    tmp = "_20000_20000"
    font_type = ""
    # font_size = "14"
    plt.rcParams["font.size"] = 20
    load_path = os.path.join(main_path, "strong_" + args.type + tmp +".npy")
    data: npt.NDArray[np.double] = np.load(load_path)

    num_run = (data.shape[0] - 2) // 2

    threads = data[0, :]
    num_threads = threads.shape[0]
    speed_up_mean = np.mean(data[num_run + 1:2 * num_run, :], axis=0)
    speed_up_std = np.std(data[num_run + 1:2 * num_run + 1, :], axis=0)
    size_sparse = data[2 * num_run + 1, 0]
    no = data[2 * num_run + 2, 0]
    ne = data[2 * num_run + 2, 1]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(threads, speed_up_mean, yerr=speed_up_std, fmt="_k", capsize=5)

    # add line segment
    # ax.plot(threads, speed_up_mean, ':b', lw=2)

    # add ideal scaling
    ax.plot(threads, threads)

    ax.set_xlabel("Number of Threads")
    ax.set_ylabel("Speed-up")
    ax.set_title("Strong Scaling of the Self-Energies Calculation")

    ax.text(1,num_threads - 8, f"Initial Runtime: {np.mean(data[1:num_run+1,0]):.2f} s" +
            "\n" + f"Final Runtime: {np.mean(data[1:num_run+1,num_threads-1]):.2f} s" +
            "\n" + f"Matrix Size: {np.int32(no)} x {np.int32(ne)}")

    save_path = os.path.join(main_path, args.type + "_strong_" + tmp + ".pdf")
    plt.savefig(save_path, dpi=600)