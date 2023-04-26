import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import sys
import os
import argparse
main_path = os.path.abspath(os.path.dirname(__file__))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Strong scaling benchmarks"
    )
    parser.add_argument("-t", "--type", default="cpu_fft",
                    choices=["cpu_fft_inlined", "cpu_fft",
                                "cpu_conv"], required=False)

    args = parser.parse_args()
    print("Format: ", args.type)


    font_type = ""
    font_size = "14"

    load_path = os.path.join(main_path, "cpu_strong_" + args.type + ".npy")
    data: npt.NDArray[np.double] = np.load(load_path)

    num_run = (data.shape[0] - 2) // 2

    threads = data[0,:]
    num_threads = threads.shape[0]
    speed_up_mean = np.mean(data[num_run+1:2*num_run,:], axis=0)
    speed_up_std = np.std(data[num_run+1:2*num_run + 1,:], axis=0)
    size_sparse = data[2*num_run + 1,0]
    no = data[2*num_run + 2,0]
    ne = data[2*num_run + 2,1]

    fig, ax = plt.subplots()

    ax.errorbar(threads, speed_up_mean, yerr=speed_up_std, fmt="_k", capsize=5)

    # add line segment
    # ax.plot(threads, speed_up_mean, ':b', lw=2)

    # add ideal scaling
    ax.plot(threads, threads)

    ax.set_xlabel("Number of Threads", fontsize=font_size)
    ax.set_ylabel("Speedup", fontsize=font_size)
    ax.set_title("Strong Scaling Plot of CPU " + args.type + " Implementation", fontsize=font_size)

    ax.text(1,num_threads - 8, f"Single Runtime: {np.mean(data[1:num_run+1,0]):.2f} s" +
            "\n" + f"Final Runtime: {np.mean(data[1:num_run+1,num_threads-1]):.2f} s" +
            "\n" + f"Matrix Size: {np.int32(no)} x {np.int32(ne)}" +
            "\n" + f"Number of Runs per Point {num_run}", fontsize=font_size)

    ax.tick_params(axis="x", labelsize=font_size)
    ax.tick_params(axis="y", labelsize=font_size)

    save_path = os.path.join(main_path, args.type + "_strong_" + str(np.int32(no)) + "_" + str(np.int32(ne)) + ".png")
    plt.savefig(save_path, dpi=600)
