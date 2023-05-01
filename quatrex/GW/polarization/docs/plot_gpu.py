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
    parser.add_argument("-t", "--type", default="gpu_fft",
                    choices=["gpu_fft", "gpu_conv"], required=False)
    parser.add_argument("-d", "--dimension", default="nnz",
                    choices=["energy", "nnz"], required=False)

    args = parser.parse_args()
    print("Format: ", args.type)
    print("Scaling over Energy/nnz: ", args.dimension)



    font_type = ""
    font_size = "14"

    load_path = os.path.join(main_path, "gpu_" + args.type + "_" + args.dimension + ".npy")
    data: npt.NDArray[np.double] = np.load(load_path)

    num_run = (data.shape[0] - 4)

    mems = data[0,:]
    num_mems = mems.shape[0]
    time_up_mean = np.mean(data[1:num_run+1,:], axis=0)
    time_std = np.std(data[1:num_run+1,:], axis=0)
    size_sparse = data[num_run + 3,0]


    fig, ax = plt.subplots(figsize=(8, 6))

    ax.errorbar(mems, time_up_mean, yerr=time_std, fmt="_k", capsize=5)

    # add line segment
    # ax.plot(threads, speed_up_mean, ':b', lw=2)

    ax.set_xlabel("Size Scale Factor", fontsize=font_size)
    ax.set_ylabel("Time [s]", fontsize=font_size)
    ax.set_title("Run Time against Matrix Size over " + args.dimension + " of GPU " + args.type + " Implementation", fontsize=font_size)

    ax.text(num_mems - 12, 0.001, f"Single Runtime: {np.mean(data[1:num_run+1,0]):.3f} s" +
            "\n" + f"Final Runtime: {np.mean(data[1:num_run+1,num_mems-1]):.3f} s" +
            "\n" + f"Init. Size: {np.int32(data[num_run + 2,0])} x {np.int32(data[num_run + 1,0])}" +
            "\n" + f"Number of Runs per Point {num_run}", fontsize=font_size)

    ax.tick_params(axis="x", labelsize=font_size)
    ax.tick_params(axis="y", labelsize=font_size)

    save_path = os.path.join(main_path, args.type + "_" + args.dimension + ".png")
    plt.savefig(save_path, dpi=600)
