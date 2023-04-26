"""Generate data for weak scaling cpu plot
"""
import numpy as np
import numpy.typing as npt
import sys
import cupy as cp
from cupyx.profiler import benchmark
import os
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
    parser.add_argument("-t", "--type", default="gpu_fft",
                    choices=["gpu_fft", "gpu_conv"], required=False)
    parser.add_argument("-d", "--dimension", default="nnz",
                    choices=["energy", "nnz"], required=False)

    args = parser.parse_args()
    print("Format: ", args.type)
    print("Scaling over Energy/nnz: ", args.dimension)


    # number of repeats
    num_run = 40
    num_warm = 5
    # test for 32 memory sizes on the gpu
    num_mems = 22
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


    mems: npt.NDArray[np.int32] = np.arange(1, num_mems+1, 1, dtype=np.int32)
    times: npt.NDArray[np.double] = np.empty((num_run, num_mems), dtype=np.double)
    energy_sizes = np.empty((num_mems), dtype=np.int32)
    nnz_sizes = np.empty((num_mems), dtype=np.int32)
    gb_sizes = np.empty((num_mems), dtype=np.double)

    # randomize input data
    seed = 10
    rng = np.random.default_rng(seed)


    for i in range(num_mems):

        # calculate new size
        if args.dimension == "energy":
            ne = np.int32(ne_0 * (i+1))
            nao = np.int32(np.sqrt(nnz_0 / 0.0394))
        elif args.dimension == "nnz":
            ne = ne_0
            nao = np.int32(np.sqrt((i+1) * nnz_0 / 0.0394))
        else:
            raise ValueError(
            "Argument error since not possible input")

        # generate data in the loop
        energy, rows, columns, gg, gl, gr = gf_init.init_sparse(ne, nao, seed)
        ij2ji:      npt.NDArray[np.int32]   = read_solution.find_idx_transposed(rows, columns)
        denergy:    np.double               = energy[1] - energy[0]
        ne:         np.int32                = energy.shape[0]
        no:         np.int32                = gg.shape[0]
        prefactor:  np.double               = -1.0j * denergy / (np.pi)

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

        # load data to gpu
        ij2ji_gpu = cp.asarray(ij2ji)
        data_1ne_1_gpu = cp.asarray(data_1ne_1)
        data_1ne_2_gpu = cp.asarray(data_1ne_2)
        data_1ne_3_gpu = cp.asarray(data_1ne_3)

        if args.type == "gpu_fft":

            result = benchmark(g2p_sparse.g2p_fft_gpu, (denergy, ij2ji_gpu,
                data_1ne_1_gpu, data_1ne_2_gpu, data_1ne_3_gpu), n_repeat=num_run, n_warmup=num_warm)

        elif args.type == "gpu_conv":
            # output buffers
            pg_gpu:         cp.ndarray = cp.empty_like(data_1ne_1_gpu, dtype=cp.complex128, order="C")
            pl_gpu:         cp.ndarray = cp.empty_like(data_1ne_1_gpu, dtype=cp.complex128, order="C")
            pr_gpu:         cp.ndarray = cp.empty_like(data_1ne_1_gpu, dtype=cp.complex128, order="C")
            
            # define number of threads
            num_threadsx    = 32
            num_threadsy    = 32
            num_blocksx     = (no + num_threadsx - 1) // num_threadsx
            num_blocksy     = (ne + num_threadsy - 1) // num_threadsy

            gpu_conv = g2p_sparse.g2p_conv_gpu(1)

            def kernel(pre_k, ij_k, gg_k, gl_k, gr_k,
                       pg_k, pl_k, pr_k, ne_k, no_k):
                gpu_conv((num_blocksx, num_blocksy),
                        (num_threadsx, num_threadsy),
                        (pre_k, ij_k,
                        gg_k, gl_k, gr_k,
                        pg_k, pl_k, pr_k,
                        no_k, ne_k))

            result = benchmark(kernel, (prefactor, ij2ji_gpu,
                data_1ne_1_gpu, data_1ne_2_gpu, data_1ne_3_gpu,
                pg_gpu, pl_gpu, pr_gpu, ne, no), n_repeat=num_run, n_warmup=num_warm)
        else:
            raise ValueError(
            "Argument error since not possible input")


        times[:,i] = result.gpu_times
        print("Time: ", np.mean(times[:,i]))


    output: npt.NDArray[np.double] = np.empty((1 + num_run + 1  + 1 + 1, num_mems), dtype=np.double)
    output[0,:] = mems
    output[1:num_run+1,:] = times
    output[num_run + 1,:] = energy_sizes
    output[num_run + 2,:] = nnz_sizes
    output[num_run + 3,:] = gb_sizes

    save_path = os.path.join(main_path, args.type + "_" + args.dimension + ".npy")
    np.save(save_path, output)
