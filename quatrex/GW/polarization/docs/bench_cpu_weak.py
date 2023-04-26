"""Generate data for weak scaling cpu plot
"""
import numpy as np
import numpy.typing as npt
import sys
import os
import numba
import timeit
import argparse

# ghetto solution from ghetto coder
main_path = os.path.abspath(os.path.dirname(__file__))
parent_path = os.path.abspath(os.path.join(main_path, ".."))
gggparent_path = os.path.abspath(os.path.join(main_path, "..", "..", ".."))
sys.path.append(parent_path)
sys.path.append(gggparent_path)

from GW_SE_python.gold_solution import read_solution
from GW_SE_python.polarization.sparse import g2p_sparse
from GW_SE_python.polarization.initialization import gf_init

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


    # number of repeats
    num_run = 20
    num_warm = 5
    # 28 thread cluster node
    num_threads = 28


    threads: npt.NDArray[np.int32] = np.arange(1, num_threads+1, 1, dtype=np.int32)
    times: npt.NDArray[np.double] = np.empty((num_run, num_threads), dtype=np.double)
    speed_ups: npt.NDArray[np.double] = np.empty_like(times, dtype=np.double)
    energy_sizes = np.empty((num_threads), dtype=np.int32)
    nnz_sizes = np.empty((num_threads), dtype=np.int32)
    gb_sizes = np.empty((num_threads), dtype=np.double)

    # randomize input data
    seed = 10
    rng = np.random.default_rng(seed)

    # number of energy points
    ne_0 = 400
    nnz_0 = 2500
    # number of orbitals -> around 0.04*nao*nao are the nonzero amount of nnz
    # nnz_gold=21888
    # nao = 10 -> nnz = 4
    # nao = 100 -> nnz=393, 0.039437
    # nao = 200 -> nnz=1571
    # nao = 400 -> nnz=6301
    # nao = 600 -> nnz=14205, 0.039437
    # nao = 800 -> nnz=25240
    # nao = 1000 -> nnz=39437, 0.0393


    for i in range(threads.shape[0]):
        # number of numba threads
        numba.set_num_threads(threads[i])
        print("Number of used threads: ", numba.get_num_threads())

        # calculate new size
        if args.dimension == "energy":
            # different scaling depending on fft (NlogN) or conv(N^2)
            if args.type in ("cpu_fft","cpu_fft_inlined"):
                # newtons method it is:
                num_iteration = 10
                fac = 1.0
                def fs(xn: np.double) -> np.double:
                    return xn*ne_0 * np.log(xn*ne_0) - numba.get_num_threads()*ne_0
                def fprime(xn: np.double) -> np.double:
                    return ne_0 * np.log(xn*ne_0) - 1
                for it in range(num_iteration):
                    fac = fac - fs(fac)/fprime(fac)
                print("fac: ", fac)
                ne = np.int32(ne_0 * fac)
            elif args.type in ("cpu_conv"):
                ne = np.int32(ne_0 * np.sqrt(numba.get_num_threads()))
            else:
                raise ValueError(
                "Argument error since not possible input")
            nao = np.int32(np.sqrt(nnz_0 / 0.0394))
        elif args.dimension == "nnz":
            ne = ne_0
            nao = np.int32(np.sqrt(numba.get_num_threads() * nnz_0 / 0.0394))
        else:
            raise ValueError(
            "Argument error since not possible input")

        # generate data in the loop
        energy, rows, columns, gg, gl, gr = gf_init.init_sparse(ne, nao, seed)
        ij2ji:      npt.NDArray[np.int32]   = read_solution.find_idx_transposed(rows, columns)
        denergy:    np.double               = energy[1] - energy[0]
        ne:         np.int32                = energy.shape[0]
        no:         np.int32                = gg.shape[0]
        pre_factor: np.complex128           = -1.0j * denergy / (np.pi)


        data_1ne_1 = rng.uniform(
        size=(no, ne)) + 1j * rng.uniform(size=(no, ne))
        data_1ne_2 = rng.uniform(
        size=(no, ne)) + 1j * rng.uniform(size=(no, ne))
        data_1ne_3 = rng.uniform(
        size=(no, ne)) + 1j * rng.uniform(size=(no, ne))
        size_sparse = data_1ne_1.nbytes / (1024**3)


        print("Number of nnz: ", no)
        print("Number of energy points: ", ne)
        print(f"Size of one sparse greens function in GB: {size_sparse:.2f} GB")
        energy_sizes[i] = ne
        nnz_sizes[i] = no
        gb_sizes[i] = size_sparse


        if args.type == "cpu_fft":
            time_warm = timeit.timeit(
                stmt=lambda: g2p_sparse.g2p_fft_cpu(pre_factor, ij2ji, data_1ne_1, data_1ne_2, data_1ne_3),
                number=num_warm) / num_warm
            time = timeit.repeat(
                stmt=lambda: g2p_sparse.g2p_fft_cpu(pre_factor, ij2ji, data_1ne_1, data_1ne_2, data_1ne_3),
                number=1, repeat=num_run)
        elif args.type == "cpu_fft_inlined":
            time_warm = timeit.timeit(
                stmt=lambda: g2p_sparse.g2p_fft_cpu_inlined(pre_factor, ij2ji, data_1ne_1, data_1ne_2, data_1ne_3),
                number=num_warm) / num_warm
            time = timeit.repeat(
                stmt=lambda: g2p_sparse.g2p_fft_cpu_inlined(pre_factor, ij2ji, data_1ne_1, data_1ne_2, data_1ne_3),
                number=1, repeat=num_run)
        elif args.type  == "conv":
            time_warm = timeit.timeit(
                stmt=lambda: g2p_sparse.g2p_conv_cpu(pre_factor, ij2ji, data_1ne_1, data_1ne_2, data_1ne_3),
                number=num_warm) / num_warm
            time = timeit.repeat(
                stmt=lambda: g2p_sparse.g2p_conv_cpu(pre_factor, ij2ji, data_1ne_1, data_1ne_2, data_1ne_3),
                number=1, repeat=num_run)
        else:
            raise ValueError(
            "Argument error since not possible input")


        times[:,i] = np.array(time, dtype=np.double)
        speed_ups[:,i] = np.divide(np.mean(times[:,0]), times[:,i])
        print("Time warm: ", time_warm)
        print("Time: ", np.mean(times[:,i]))


    output: npt.NDArray[np.double] = np.empty((1 + 2 * num_run + 1  + 1 + 1,threads.shape[0]), dtype=np.double)
    output[0,:] = threads
    output[1:num_run+1,:] = times
    output[num_run+1:2*num_run + 1,:] = speed_ups
    output[2*num_run + 1,:] = energy_sizes
    output[2*num_run + 2,:] = nnz_sizes
    output[2*num_run + 3,:] = gb_sizes

    save_path = os.path.join(main_path, "weak_" + args.type + "_" + args.dimension + ".npy")
    np.save(save_path, output)
