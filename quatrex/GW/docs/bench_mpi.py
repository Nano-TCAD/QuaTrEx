"""
Generates timing for a different amount of ranks.

"""
import subprocess
import time
import numpy as np
import numpy.typing as npt
import os
import argparse

# ghetto solution from ghetto coder
main_path = os.path.abspath(os.path.dirname(__file__))

if __name__ == "__main__":
    pin_path = os.path.abspath(os.path.join(main_path, "..", "attelas.run"))
    script_path = os.path.abspath(os.path.join(main_path, "..", "test_mpi.py"))
    # parse the possible arguments
    parser = argparse.ArgumentParser(
        description="Benchmark the MPI version of the code"
    )
    parser.add_argument("-pp", "--pin_path", default=pin_path, required=False)
    parser.add_argument("-sp", "--script_path", default=script_path, required=False)
    parser.add_argument("-r", "--ranks", default=4, required=False, type=int)

    args = parser.parse_args()

    # number of ranks to benchmark
    num_ranks = args.ranks
    num_run = 1
    ranks: npt.NDArray[np.int32] = np.arange(1, num_ranks+1, 1, dtype=np.int32)
    times: npt.NDArray[np.double] = np.empty((num_run, num_ranks), dtype=np.double)
    speed_ups: npt.NDArray[np.double] = np.empty((num_run, num_ranks), dtype=np.double)

    for i in range(num_ranks):
        print("Number of ranks: ", ranks[i])
        command = ["mpiexec", "-n", str(ranks[i]), "-f", args.pin_path,
                   "python", args.script_path]

        for run in range(num_run):
            start_time = time.perf_counter()
            # call the mpi file
            subprocess.call(command)
            end_time = time.perf_counter()
            print("Time taken: ", end_time - start_time, " [s]")
            times[run,i] = end_time - start_time

        speed_ups[:,i] = np.divide(np.mean(times[:,0]), times[:,i])

    output: npt.NDArray[np.double] = np.empty((1 + 2 * num_run, num_ranks), dtype=np.double)
    output[0,:] = ranks
    output[1:num_run+1,:] = times
    output[num_run+1:2*num_run + 1,:] = speed_ups
    save_path = os.path.join(main_path, "strong_mpi.npy")
    np.save(save_path, output)
