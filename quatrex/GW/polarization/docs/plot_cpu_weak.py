import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
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
    parser.add_argument("-d", "--dimension", default="nnz",
                    choices=["energy", "nnz"], required=False)

    args = parser.parse_args()
    print("Format: ", args.type)
    print("Scaling over Energy/nnz: ", args.dimension)


    font_type = ""
    font_size = "14"

    load_path = os.path.join(main_path, "cpu_weak_" + args.type + "_" + args.dimension + ".npy")
    data: npt.NDArray[np.double] = np.load(load_path)

    num_run = (data.shape[0] - 4) // 2

    threads = data[0,:]
    num_threads = threads.shape[0]
    speed_up_mean = np.mean(data[num_run+1:2*num_run,:], axis=0)
    speed_up_std = np.std(data[num_run+1:2*num_run + 1,:], axis=0)
    size_sparse = data[2*num_run + 3,0]



    fig, ax = plt.subplots()

    ax.errorbar(threads, speed_up_mean, yerr=speed_up_std, fmt="_k", capsize=5)

    # add line segment
    # ax.plot(threads, speed_up_mean, ':b', lw=2)

    # add ideal scaling
    ax.plot(threads, np.ones(num_threads))

    ax.set_xlabel("Number of Threads", fontsize=font_size)
    ax.set_ylabel("Speedup", fontsize=font_size)
    ax.set_title("Weak Scaling Plot over " + args.dimension +" of CPU " + args.type + " Implementation", fontsize=font_size)

    ax.text(num_threads - 16, 0.7, f"Single Runtime: {np.mean(data[1:num_run+1,0]):.2f} s" +
            "\n" + f"Final Runtime: {np.mean(data[1:num_run+1,num_threads-1]):.2f} s" +
            "\n" + f"Init. Size: {np.int32(data[2*num_run + 2,0])} x {np.int32(data[2*num_run + 1,0])}" +
            "\n" + f"Number of Runs per Point {num_run}", fontsize=font_size)

    ax.tick_params(axis="x", labelsize=font_size)
    ax.tick_params(axis="y", labelsize=font_size)

    save_path = os.path.join(main_path, args.type + "_" + args.dimension + "_weak_"+ str(np.int32(data[2*num_run + 2,0])) + "_" + str(np.int32(data[2*num_run + 1,0]))  +".png")
    plt.savefig(save_path, dpi=600)
