# Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.
""" Tests the different sigma implementation against a reference solution from a matlab code. """
import os
import sys
import argparse
import numba
import numpy as np
import numpy.typing as npt

main_path = os.path.abspath(os.path.dirname(__file__))
parent_path = os.path.abspath(os.path.join(main_path, "..", ".."))
sys.path.append(parent_path)

from GW.selfenergy.kernel import gw2s_cpu
from GW.gold_solution import read_solution
from utils import utils_gpu
from utils import change_format

if utils_gpu.gpu_avail():
    import cupy as cp
    from utils import linalg_gpu
    from GW.selfenergy.kernel import gw2s_gpu

if __name__ == "__main__":
    # parse the possible arguments
    # solution_path = os.path.join("/usr/scratch/mont-fort17/dleonard/CNT/", "data_GPWS_04.mat")
    solution_path = os.path.join("/usr/scratch/mont-fort17/dleonard/IEDM/GNR_pd_unbiased", "data_GPWS_IEDM_GNR_0V.mat")
    parser = argparse.ArgumentParser(description="Tests different implementation of the self-energy calculation")
    parser.add_argument("-t",
                        "--type",
                        default="gpu_fft_mpi_3part",
                        choices=[
                            "gpu_fft", "gpu_fft_mpi", "gpu_fft_mpi_streams", "gpu_fft_mpi_batched", "cpu_fft",
                            "cpu_fft_mpi", "cpu_fft_3part", "cpu_fft_mpi_3part", "gpu_fft_3part", "gpu_fft_mpi_3part"
                        ],
                        required=False)
    parser.add_argument("-f", "--file", default=solution_path, required=False)
    args = parser.parse_args()

    if args.type in ("gpu_fft", "gpu_fft_mpi", "gpu_fft_mpi_streams", "gpu_fft_mpi_batched", "gpu_fft_3part",
                     "gpu_fft_mpi_3part"):
        if not utils_gpu.gpu_avail():
            print("No gpu available")
            sys.exit(1)

    print("Used implementation: ", args.type)
    print("Path to gold solution: ", args.file)
    print("Number of used numba threads: ", numba.get_num_threads())

    # load greens function
    energy, rows, columns, gg_gold, gl_gold, gr_gold = read_solution.load_x(args.file, "g")
    # load screened interaction
    _, _, _, wg_gold, wl_gold, wr_gold = read_solution.load_x(args.file, "w")
    # load sigma
    _, _, _, sg_gold, sl_gold, sr_gold = read_solution.load_x(args.file, "s")

    ij2ji: npt.NDArray[np.int32] = change_format.find_idx_transposed(rows, columns)
    denergy: np.double = energy[1] - energy[0]
    ne: np.int32 = np.int32(energy.shape[0])
    no: np.int32 = np.int32(columns.shape[0])
    pre_factor: np.complex128 = 1.0j * denergy / (2 * np.pi)

    # sanity checks
    assert gg_gold.ndim == 2
    assert gl_gold.ndim == 2
    assert gr_gold.ndim == 2
    assert energy.ndim == 1
    assert np.array_equal(np.shape(gg_gold), np.shape(gl_gold))
    assert np.array_equal(np.shape(gg_gold), np.shape(gr_gold))
    assert np.array_equal(np.shape(gg_gold), np.shape(wg_gold))
    assert np.array_equal(np.shape(gg_gold), np.shape(wl_gold))
    assert np.array_equal(np.shape(gg_gold), np.shape(wr_gold))
    assert np.array_equal(np.shape(gg_gold), np.shape(sg_gold))
    assert np.array_equal(np.shape(gg_gold), np.shape(sl_gold))
    assert np.array_equal(np.shape(gg_gold), np.shape(sr_gold))

    # assume energy is the second index
    assert np.shape(energy)[0] == np.shape(gg_gold)[1]

    # assume energy is evenly spaced
    assert np.allclose(np.diff(energy), np.diff(energy)[0])

    # check physical identities for inputs
    assert np.allclose(gl_gold, -gl_gold[ij2ji, :].conjugate())
    assert np.allclose(gg_gold, -gg_gold[ij2ji, :].conjugate())

    # copy input to test
    energy_copy: npt.NDArray[np.double] = np.copy(energy)
    ij2ji_copy: npt.NDArray[np.int32] = np.copy(ij2ji)
    gg_copy: npt.NDArray[np.complex128] = np.copy(gg_gold)
    gl_copy: npt.NDArray[np.complex128] = np.copy(gl_gold)
    gr_copy: npt.NDArray[np.complex128] = np.copy(gr_gold)
    wg_copy: npt.NDArray[np.complex128] = np.copy(wg_gold)
    wl_copy: npt.NDArray[np.complex128] = np.copy(wl_gold)
    wr_copy: npt.NDArray[np.complex128] = np.copy(wr_gold)
    print("Number of energy points: ", ne)
    print("Number of non zero elements: ", no)

    # check size of arrays
    size_sparse = gg_gold.nbytes / (1024**3)
    print(f"Size of one sparse greens function in GB: {size_sparse:.2f} GB")

    if args.type == "gpu_fft":
        # load data to gpu
        ij2ji_gpu: cp.ndarray = cp.asarray(ij2ji)
        gg_gold_gpu: cp.ndarray = cp.asarray(gg_gold)
        gl_gold_gpu: cp.ndarray = cp.asarray(gl_gold)
        gr_gold_gpu: cp.ndarray = cp.asarray(gr_gold)
        wg_gold_gpu: cp.ndarray = cp.asarray(wg_gold)
        wl_gold_gpu: cp.ndarray = cp.asarray(wl_gold)
        wr_gold_gpu: cp.ndarray = cp.asarray(wr_gold)

        sg_gpu, sl_gpu, sr_gpu = gw2s_gpu.gw2s_fft_gpu(pre_factor, ij2ji_gpu, gg_gold_gpu, gl_gold_gpu, gr_gold_gpu,
                                                       wg_gold_gpu, wl_gold_gpu, wr_gold_gpu)

        # load data to cpu
        sg_cpu: npt.NDArray[np.complex128] = cp.asnumpy(sg_gpu)
        sl_cpu: npt.NDArray[np.complex128] = cp.asnumpy(sl_gpu)
        sr_cpu: npt.NDArray[np.complex128] = cp.asnumpy(sr_gpu)
    elif args.type == "gpu_fft_3part":
        # load data to gpu
        ij2ji_gpu: cp.ndarray = cp.asarray(ij2ji)
        gg_gold_gpu: cp.ndarray = cp.asarray(gg_gold)
        gl_gold_gpu: cp.ndarray = cp.asarray(gl_gold)
        gr_gold_gpu: cp.ndarray = cp.asarray(gr_gold)
        wg_gold_gpu: cp.ndarray = cp.asarray(wg_gold)
        wl_gold_gpu: cp.ndarray = cp.asarray(wl_gold)
        wr_gold_gpu: cp.ndarray = cp.asarray(wr_gold)

        sg_gpu, sl_gpu, sr_gpu = gw2s_gpu.gw2s_fft_gpu_3part_sr(pre_factor, ij2ji_gpu, gg_gold_gpu, gl_gold_gpu,
                                                                gr_gold_gpu, wg_gold_gpu, wl_gold_gpu, wr_gold_gpu)

        # load data to cpu
        sg_cpu: npt.NDArray[np.complex128] = cp.asnumpy(sg_gpu)
        sl_cpu: npt.NDArray[np.complex128] = cp.asnumpy(sl_gpu)
        sr_cpu: npt.NDArray[np.complex128] = cp.asnumpy(sr_gpu)
    elif args.type == "cpu_fft":

        sg_cpu, sl_cpu, sr_cpu = gw2s_cpu.gw2s_fft_cpu(pre_factor, ij2ji, gg_gold, gl_gold, gr_gold, wg_gold, wl_gold,
                                                       wr_gold)
    elif args.type == "cpu_fft_3part":

        sg_cpu, sl_cpu, sr_cpu = gw2s_cpu.gw2s_fft_cpu_3part_sr(pre_factor, ij2ji, gg_gold, gl_gold, gr_gold, wg_gold,
                                                                wl_gold, wr_gold)
    elif args.type == "gpu_fft_mpi_3part":
        wg_transposed = wg_gold[ij2ji, :]
        wl_transposed = wl_gold[ij2ji, :]
        sg_cpu, sl_cpu, sr_cpu = gw2s_gpu.gw2s_fft_mpi_gpu_3part_sr(pre_factor, gg_gold, gl_gold, gr_gold, wg_gold,
                                                                    wl_gold, wr_gold, wg_transposed, wl_transposed)
    elif args.type == "gpu_fft_mpi":
        wg_transposed = wg_gold[ij2ji, :]
        wl_transposed = wl_gold[ij2ji, :]
        sg_cpu, sl_cpu, sr_cpu = gw2s_gpu.gw2s_fft_mpi_gpu(pre_factor, gg_gold, gl_gold, gr_gold, wg_gold, wl_gold,
                                                           wr_gold, wg_transposed, wl_transposed)
    elif args.type == "gpu_fft_mpi_streams":
        wg_transposed = wg_gold[ij2ji, :]
        wl_transposed = wl_gold[ij2ji, :]
        # allocate streams
        # start gpu streams
        streams = [cp.cuda.Stream(non_blocking=True) for i in range(8)]
        # allocate pinned memory
        gg_cpu = linalg_gpu.aloc_pinned_filled(gg_gold)
        gl_cpu = linalg_gpu.aloc_pinned_filled(gl_gold)
        gr_cpu = linalg_gpu.aloc_pinned_filled(gr_gold)
        wg_cpu = linalg_gpu.aloc_pinned_filled(wg_gold)
        wl_cpu = linalg_gpu.aloc_pinned_filled(wl_gold)
        wr_cpu = linalg_gpu.aloc_pinned_filled(wr_gold)
        wg_transposed_cpu = linalg_gpu.aloc_pinned_filled(wg_transposed)
        wl_transposed_cpu = linalg_gpu.aloc_pinned_filled(wl_transposed)
        sg_cpu = linalg_gpu.aloc_pinned_empty_like(gg_gold)
        sl_cpu = linalg_gpu.aloc_pinned_empty_like(gg_gold)
        sr_cpu = linalg_gpu.aloc_pinned_empty_like(gg_gold)
        gw2s_gpu.gw2s_fft_mpi_gpu_streams(pre_factor, gg_cpu, gl_cpu, gr_cpu, wg_cpu, wl_cpu, wr_cpu, wg_transposed_cpu,
                                          wl_transposed_cpu, sg_cpu, sl_cpu, sr_cpu, streams)
    elif args.type == "gpu_fft_mpi_batched":
        wg_transposed = wg_gold[ij2ji, :]
        wl_transposed = wl_gold[ij2ji, :]
        # allocate streams
        # start gpu streams
        streams = [cp.cuda.Stream(non_blocking=True) for i in range(8)]
        # allocate pinned memory
        gg_cpu = linalg_gpu.aloc_pinned_filled(gg_gold)
        gl_cpu = linalg_gpu.aloc_pinned_filled(gl_gold)
        gr_cpu = linalg_gpu.aloc_pinned_filled(gr_gold)
        wg_cpu = linalg_gpu.aloc_pinned_filled(wg_gold)
        wl_cpu = linalg_gpu.aloc_pinned_filled(wl_gold)
        wr_cpu = linalg_gpu.aloc_pinned_filled(wr_gold)
        wg_transposed_cpu = linalg_gpu.aloc_pinned_filled(wg_transposed)
        wl_transposed_cpu = linalg_gpu.aloc_pinned_filled(wl_transposed)
        sg_cpu = linalg_gpu.aloc_pinned_empty_like(gg_gold)
        sl_cpu = linalg_gpu.aloc_pinned_empty_like(gg_gold)
        sr_cpu = linalg_gpu.aloc_pinned_empty_like(gg_gold)

        # chose batch size
        batch_size = no // 10

        gw2s_gpu.gw2s_fft_mpi_gpu_batched(pre_factor, gg_cpu, gl_cpu, gr_cpu, wg_cpu, wl_cpu, wr_cpu, wg_transposed_cpu,
                                          wl_transposed_cpu, sg_cpu, sl_cpu, sr_cpu, streams, batch_size)
    elif args.type == "cpu_fft_mpi":
        wg_transposed = wg_gold[ij2ji, :]
        wl_transposed = wl_gold[ij2ji, :]
        sg_cpu, sl_cpu, sr_cpu = gw2s_cpu.gw2s_fft_mpi_cpu(pre_factor, gg_gold, gl_gold, gr_gold, wg_gold, wl_gold,
                                                           wr_gold, wg_transposed, wl_transposed)
    elif args.type == "cpu_fft_mpi_3part":
        wg_transposed = wg_gold[ij2ji, :]
        wl_transposed = wl_gold[ij2ji, :]
        sg_cpu, sl_cpu, sr_cpu = gw2s_cpu.gw2s_fft_mpi_cpu_3part_sr(pre_factor, gg_gold, gl_gold, gr_gold, wg_gold,
                                                                    wl_gold, wr_gold, wg_transposed, wl_transposed)
    else:
        raise ValueError("Argument error, type input not possible")

    # sanity checks
    assert np.allclose(energy_copy, energy)
    assert np.allclose(ij2ji_copy, ij2ji)
    assert np.allclose(gg_copy, gg_gold)
    assert np.allclose(gl_copy, gl_gold)
    assert np.allclose(gr_copy, gr_gold)
    assert np.allclose(wg_copy, wg_gold)
    assert np.allclose(wl_copy, wl_gold)
    assert np.allclose(wr_copy, wr_gold)
    assert np.array_equal(np.shape(gg_copy), np.shape(sg_cpu))
    assert np.array_equal(np.shape(gg_copy), np.shape(sl_cpu))
    assert np.array_equal(np.shape(gg_copy), np.shape(sr_cpu))

    # assert solution close to real solution
    # use Frobenius norm
    diff_g: np.double = np.linalg.norm(sg_gold - sg_cpu)
    diff_l: np.double = np.linalg.norm(sl_gold - sl_cpu)
    diff_r: np.double = np.linalg.norm(sr_gold - sr_cpu)
    print(f"Differences to Gold Solution g/l/r:  {diff_g:.4f}, {diff_l:.4f}, {diff_r:.4f}")
    abstol = 1e-12
    reltol = 1e-6
    assert diff_g <= abstol + reltol * np.max(np.abs(sg_gold))
    assert diff_l <= abstol + reltol * np.max(np.abs(sl_gold))
    assert diff_r <= abstol + reltol * np.max(np.abs(sr_gold))
    assert np.allclose(sg_gold, sg_cpu)
    assert np.allclose(sl_gold, sl_cpu)
    assert np.allclose(sr_gold, sr_cpu)

    print("The chosen implementation " + args.type + " is correct")
