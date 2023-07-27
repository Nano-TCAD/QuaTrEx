# Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import os

main_path = os.path.abspath(os.path.dirname(__file__))

if __name__ == "__main__":

    font_type = ""
    font_size = "14"

    load_path = os.path.join(main_path, "cpu_parts.npy")
    data: npt.NDArray[np.double] = np.load(load_path)

    num_run = (data.shape[0] - 2) // 8

    threads = data[0, :]
    num_threads = threads.shape[0]

    speed_up_mean1 = np.mean(data[4 * num_run + 1:5 * num_run + 1, :], axis=0)
    speed_up_std1 = np.std(data[4 * num_run + 1:5 * num_run + 1, :], axis=0)

    speed_up_mean2 = np.mean(data[5 * num_run + 1:6 * num_run + 1, :], axis=0)
    speed_up_std2 = np.std(data[5 * num_run + 1:6 * num_run + 1, :], axis=0)

    speed_up_mean3 = np.mean(data[6 * num_run + 1:7 * num_run + 1, :], axis=0)
    speed_up_std3 = np.std(data[6 * num_run + 1:7 * num_run + 1, :], axis=0)

    speed_up_mean4 = np.mean(data[7 * num_run + 1:8 * num_run + 1, :], axis=0)
    speed_up_std4 = np.std(data[7 * num_run + 1:8 * num_run + 1, :], axis=0)
    size_sparse = data[8 * num_run + 1, 0]

    fig, ax = plt.subplots()

    ax.errorbar(threads, speed_up_mean1, yerr=speed_up_std1, fmt="_b", capsize=5, label="fft")
    ax.errorbar(threads, speed_up_mean2, yerr=speed_up_std2, fmt="_g", capsize=5, label="reversal+transpose")
    ax.errorbar(threads, speed_up_mean3, yerr=speed_up_std3, fmt="_r", capsize=5, label="elementwise")
    ax.errorbar(threads, speed_up_mean4, yerr=speed_up_std4, fmt="_c", capsize=5, label="ifft+pre-factor")

    # add line segment
    # ax.plot(threads, speed_up_mean, ':b', lw=2)

    # add legend
    plt.legend(loc="upper left", fontsize="large")

    # add ideal scaling
    ax.plot(threads, np.arange(1, num_threads + 1, 1))

    ax.set_xlabel("Number of Threads", fontsize=font_size)
    ax.set_ylabel("Speedup", fontsize=font_size)
    ax.set_title("Strong Scaling Plot of CPU fft Parts", fontsize=font_size)

    ax.tick_params(axis="x", labelsize=font_size)
    ax.tick_params(axis="y", labelsize=font_size)

    save_path = os.path.join(main_path, "cpu_strong_parts.png")
    plt.savefig(save_path, dpi=600)
