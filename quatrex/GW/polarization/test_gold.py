"""Tests the different polarization implementation 
a reference solution from a matlab code
"""
import numpy as np
import numpy.typing as npt
import sys
import os
import numba
import argparse
import dace
from dace.transformation.interstate import LoopToMap, StateFusion

main_path = os.path.abspath(os.path.dirname(__file__))
parent_path = os.path.abspath(os.path.join(main_path, "..", ".."))
sys.path.append(parent_path)

from GW.polarization.kernel import g2p_cpu
from GW.gold_solution import read_solution
from utils import utils_gpu
from utils import change_format

if utils_gpu.gpu_avail():
    import cupy as cp
    from GW.polarization.kernel import g2p_gpu
    from utils import linalg_gpu

if __name__ == "__main__":
    # parse the possible arguments
    solution_path = os.path.join("/usr/scratch/mont-fort17/dleonard/CNT/", "data_GPWS_04.mat")
    parser = argparse.ArgumentParser(
        description="Tests different implementation of the polarization calculation"
    )
    parser.add_argument("-t", "--type", default="cpu_fft",
                        choices=["gpu_fft", "gpu_conv", "gpu_fft_mpi",
                                 "gpu_fft_mpi_streams", "gpu_fft_mpi_batched",
                                 "cpu_fft_inlined", "cpu_fft",
                                 "cpu_fft_mpi", "cpu_fft_mpi_inlined",
                                 "cpu_conv", "cpu_conv_dace",
                                 "cpu_dense"], required=False)
    parser.add_argument("-f", "--file", default=solution_path, required=False)
    args = parser.parse_args()

    if args.type in ("gpu_fft", "gpu_conv", "gpu_fft_mpi", "gpu_fft_mpi_streams", "gpu_fft_mpi_batched"):
        if not utils_gpu.gpu_avail():
            print("No gpu available")
            sys.exit(1)

    print("Used implementation: ", args.type)
    print("Path to gold solution: ", args.file)
    # if not set, numba will use max possible threads
    print("Number of used numba threads: ", numba.get_num_threads())

    # load greens function
    energy, rows, columns, gg_gold, gl_gold, gr_gold    = read_solution.load_x(args.file, "g")
    # load polarization
    _, _, _, pg_gold, pl_gold, pr_gold                  = read_solution.load_x(args.file, "p")

    ij2ji:      npt.NDArray[np.int32]   = change_format.find_idx_transposed(rows, columns)
    denergy:    np.double               = energy[1] - energy[0]
    ne:         np.int32                = np.int32(energy.shape[0])
    no:         np.int32                = np.int32(columns.shape[0])
    pre_factor: np.complex128           = -1.0j * denergy / (np.pi)

    # sanity checks
    assert gg_gold.ndim == 2
    assert gl_gold.ndim == 2
    assert gr_gold.ndim == 2
    assert energy.ndim == 1
    assert np.array_equal(np.shape(gg_gold), np.shape(gl_gold))
    assert np.array_equal(np.shape(gg_gold), np.shape(gr_gold))

    # assume energy is the second index
    assert np.shape(energy)[0] == np.shape(gg_gold)[1]
    
    # assume energy is evenly spaced
    assert np.allclose(np.diff(energy), np.diff(energy)[0])

    # check physical identities for inputs
    assert np.allclose(gl_gold, -gl_gold[ij2ji, :].conjugate())
    assert np.allclose(gg_gold, -gg_gold[ij2ji, :].conjugate())

    # todo find out why not hold
    # G^{>} - G^{<} = G^{r} - G^{a}
    # G^{a} = (G^{r})^{H}
    # print(np.linalg.norm(np.real(gg_gold - gl_gold -
    #       gr_gold + gr_gold[ij2ji, :].conjugate())))
    # print(np.linalg.norm(np.imag(gg_gold - gl_gold -
    #       gr_gold + gr_gold[ij2ji, :].conjugate())))
    # assert np.allclose(gg_gold - gl_gold, gr_gold - gr_gold[ij2ji, :].conjugate())

    # copy input to test
    energy_copy:    npt.NDArray[np.double]      = np.copy(energy)
    ij2ji_copy:     npt.NDArray[np.int32]       = np.copy(ij2ji)
    gg_copy:        npt.NDArray[np.complex128]  = np.copy(gg_gold)
    gl_copy:        npt.NDArray[np.complex128]  = np.copy(gl_gold)
    gr_copy:        npt.NDArray[np.complex128]  = np.copy(gr_gold)

    print("Number of energy points: ", ne)
    print("Number of non zero elements: ", no)

    if args.type == "cpu_dense":

        # get dense inputs
        gg_dense: npt.NDArray[np.complex128] = change_format.sparse_to_dense(rows, columns, gg_gold)
        gl_dense: npt.NDArray[np.complex128] = change_format.sparse_to_dense(rows, columns, gl_gold)
        gr_dense: npt.NDArray[np.complex128] = change_format.sparse_to_dense(rows, columns, gr_gold)

        # check size of arrays
        size_dense = gg_dense.nbytes / (1024**3)
        print(f"Size of one dense greens function in GB: {size_dense:.2f} GB")

        pg_cpu_dense, pl_cpu_dense, pr_cpu_dense, ep_dense = g2p_cpu.g2p_dense(
            gg_dense, gl_dense, gr_dense, energy, workers=numba.get_num_threads())

        # define energy interval for dense
        energy_s: np.int32 = ne - 1
        energy_n: np.int32 = 2*ne - 1

        # assert physical identity
        # P^{>}_{ij}\left(E\right) = -P^{<}_{ij}\left(-E\right)^{*}
        assert np.allclose(pg_cpu_dense, -np.conjugate(np.flip(pl_cpu_dense, axis=0)))

        # cutoff
        pg_cpu_dense: npt.NDArray[np.complex128] = pg_cpu_dense[energy_s:energy_n]
        pl_cpu_dense: npt.NDArray[np.complex128] = pl_cpu_dense[energy_s:energy_n]
        pr_cpu_dense: npt.NDArray[np.complex128] = pr_cpu_dense[energy_s:energy_n]

        # transform gold solution to dense
        pg_gold_dense: npt.NDArray[np.complex128] = change_format.sparse_to_dense(rows, columns, pg_gold)
        pl_gold_dense: npt.NDArray[np.complex128] = change_format.sparse_to_dense(rows, columns, pl_gold)
        pr_gold_dense: npt.NDArray[np.complex128] = change_format.sparse_to_dense(rows, columns, pr_gold)

        # assert solution close to real solution
        # use Frobenius norm
        diff_g: np.int32 = np.linalg.norm(pg_gold_dense - pg_cpu_dense)
        diff_l: np.int32 = np.linalg.norm(pl_gold_dense - pl_cpu_dense)
        diff_r: np.int32 = np.linalg.norm(pr_gold_dense - pr_cpu_dense)
        print(f"Differences to Gold Solution g/l/r:  {diff_g:.4f}, {diff_l:.4f}, {diff_r:.4f}")
        abstol = 1e-14
        reltol = 1e-6
        assert diff_g <= abstol + reltol * np.max(np.abs(pg_gold_dense))
        assert diff_l <= abstol + reltol * np.max(np.abs(pl_gold_dense))
        assert diff_r <= abstol + reltol * np.max(np.abs(pr_gold_dense))
        assert np.allclose(pg_gold_dense, pg_cpu_dense)
        assert np.allclose(pl_gold_dense, pl_cpu_dense)
        assert np.allclose(pr_gold_dense, pr_cpu_dense)

    else:
        # check size of arrays
        size_sparse = gg_gold.nbytes / (1024**3)
        print(f"Size of one sparse greens function in GB: {size_sparse:.2f} GB")

        # testing on cpu or gpu
        if args.type == "cpu_fft":

            pg_cpu, pl_cpu, pr_cpu = g2p_cpu.g2p_fft_cpu(
                pre_factor, ij2ji, gg_gold, gl_gold, gr_gold)
            
        elif args.type == "cpu_fft_mpi":

            gl_gold_transposed = gl_gold[ij2ji,:]
            pg_cpu, pl_cpu, pr_cpu = g2p_cpu.g2p_fft_mpi_cpu(
                pre_factor, gg_gold, gl_gold, gr_gold, gl_gold_transposed)

        elif args.type == "cpu_fft_inlined":

            pg_cpu, pl_cpu, pr_cpu = g2p_cpu.g2p_fft_cpu_inlined(
                pre_factor, ij2ji, gg_gold, gl_gold, gr_gold)
        elif args.type == "cpu_fft_mpi_inlined":

            gl_gold_transposed = gl_gold[ij2ji,:]
            pg_cpu, pl_cpu, pr_cpu = g2p_cpu.g2p_fft_mpi_cpu_inlined(
                pre_factor, gg_gold, gl_gold, gr_gold, gl_gold_transposed)
    
        elif args.type == "cpu_conv":

            pg_cpu, pl_cpu, pr_cpu = g2p_cpu.g2p_conv_cpu(
                pre_factor, ij2ji, gg_gold, gl_gold, gr_gold)

        elif args.type == "cpu_conv_dace":

            # create zero outputs
            pg_cpu = np.zeros_like(gg_gold, dtype=np.cdouble)
            pl_cpu = np.zeros_like(gg_gold, dtype=np.cdouble)
            pr_cpu = np.zeros_like(gg_gold, dtype=np.cdouble)


            # compile to SDFG
            sdfg: dace.SDFG = g2p_cpu.g2p_conv_dace.to_sdfg(simplify=True)
            sdfg.apply_transformations(LoopToMap)
            sdfg.apply_transformations_repeated(StateFusion)
            sdfg.apply_transformations(LoopToMap)
            sdfg.simplify()
            csdfg = sdfg.compile()

            # call compiled function
            csdfg(pre_factor=np.array([pre_factor]), ij2ji=ij2ji, gg=gg_gold,
                gl=gl_gold, gr=gr_gold, pg=pg_cpu, pl=pl_cpu, pr=pr_cpu, NE=ne, NO=no)


        elif args.type == "gpu_fft":
            # load data to gpu
            ij2ji_gpu:   cp.ndarray = cp.asarray(ij2ji)
            gg_gold_gpu: cp.ndarray = cp.asarray(gg_gold)
            gl_gold_gpu: cp.ndarray = cp.asarray(gl_gold)
            gr_gold_gpu: cp.ndarray = cp.asarray(gr_gold)

            pg_gpu, pl_gpu, pr_gpu = g2p_gpu.g2p_fft_gpu(
                pre_factor, ij2ji_gpu, gg_gold_gpu, gl_gold_gpu, gr_gold_gpu)

            # load data to cpu
            pg_cpu: npt.NDArray[np.complex128] = cp.asnumpy(pg_gpu)
            pl_cpu: npt.NDArray[np.complex128] = cp.asnumpy(pl_gpu)
            pr_cpu: npt.NDArray[np.complex128] = cp.asnumpy(pr_gpu)

        elif args.type == "gpu_fft_mpi":
            gl_gold_transposed = gl_gold[ij2ji,:]
            pg_cpu, pl_cpu, pr_cpu = g2p_gpu.g2p_fft_mpi_gpu(
                pre_factor, gg_gold, gl_gold, gr_gold, gl_gold_transposed)

        elif args.type == "gpu_fft_mpi_streams":
            gl_gold_transposed = gl_gold[ij2ji,:]
            # allocate streams
            # start gpu streams
            streams = [cp.cuda.Stream(non_blocking=True) for i in range(4)]
            # allocate pinned memory
            gg_cpu = linalg_gpu.aloc_pinned_filled(gg_gold)
            gl_cpu = linalg_gpu.aloc_pinned_filled(gl_gold)
            gr_cpu = linalg_gpu.aloc_pinned_filled(gr_gold)
            gl_transposed_cpu = linalg_gpu.aloc_pinned_filled(gl_gold_transposed)
            pg_cpu = linalg_gpu.aloc_pinned_empty_like(gg_gold)
            pl_cpu = linalg_gpu.aloc_pinned_empty_like(gg_gold)
            pr_cpu = linalg_gpu.aloc_pinned_empty_like(gg_gold)
            g2p_gpu.g2p_fft_mpi_gpu_streams(
                pre_factor, gg_cpu, gl_cpu, gr_cpu, gl_transposed_cpu,
                pg_cpu, pl_cpu, pr_cpu, streams)
            
        elif args.type == "gpu_fft_mpi_batched":
            gl_gold_transposed = gl_gold[ij2ji,:]
            # allocate streams
            # start gpu streams
            streams = [cp.cuda.Stream(non_blocking=True) for i in range(4)]
            # allocate pinned memory
            gg_cpu = linalg_gpu.aloc_pinned_filled(gg_gold)
            gl_cpu = linalg_gpu.aloc_pinned_filled(gl_gold)
            gr_cpu = linalg_gpu.aloc_pinned_filled(gr_gold)
            gl_transposed_cpu = linalg_gpu.aloc_pinned_filled(gl_gold_transposed)
            pg_cpu = linalg_gpu.aloc_pinned_empty_like(gg_gold)
            pl_cpu = linalg_gpu.aloc_pinned_empty_like(gg_gold)
            pr_cpu = linalg_gpu.aloc_pinned_empty_like(gg_gold)

            # chose batch size
            batch_size = no // 5

            g2p_gpu.g2p_fft_mpi_gpu_batched(
                pre_factor, gg_cpu, gl_cpu, gr_cpu, gl_transposed_cpu,
                pg_cpu, pl_cpu, pr_cpu, streams, batch_size)

        elif args.type == "gpu_conv":
            # load data to gpu
            ij2ji_gpu:      cp.ndarray = cp.asarray(ij2ji)
            gg_gold_gpu:    cp.ndarray = cp.asarray(gg_gold, order="C")
            gl_gold_gpu:    cp.ndarray = cp.asarray(gl_gold, order="C")
            gr_gold_gpu:    cp.ndarray = cp.asarray(gr_gold, order="C")
            pg_gpu:         cp.ndarray = cp.empty_like(gg_gold_gpu, dtype=cp.complex128, order="C")
            pl_gpu:         cp.ndarray = cp.empty_like(gg_gold_gpu, dtype=cp.complex128, order="C")
            pr_gpu:         cp.ndarray = cp.empty_like(gg_gold_gpu, dtype=cp.complex128, order="C")
            
            # define number of threads
            num_threadsx    = 32
            num_threadsy    = 32
            num_blocksx     = (no + num_threadsx - 1) // num_threadsx
            num_blocksy     = (ne + num_threadsy - 1) // num_threadsy

            gpu_conv = g2p_gpu.g2p_conv_gpu(1)

            gpu_conv((num_blocksx, num_blocksy),
                    (num_threadsx, num_threadsy),
                    (pre_factor, ij2ji_gpu,
                    gg_gold_gpu, gl_gold_gpu, gr_gold_gpu,
                    pg_gpu, pl_gpu, pr_gpu,
                    no, ne))

            # load data to cpu
            pg_cpu: npt.NDArray[np.complex128] = cp.asnumpy(pg_gpu)
            pl_cpu: npt.NDArray[np.complex128] = cp.asnumpy(pl_gpu)
            pr_cpu: npt.NDArray[np.complex128] = cp.asnumpy(pr_gpu)
        else:
            raise ValueError(
            "Argument error, type input not possible")

        # assert physical identity
        # test: P^{>}_{ij}\left(E\right) = -P^{<}_{ij}\left(-E\right)^{*}
        # assert np.allclose(pg_cpu, -np.conjugate(np.roll(np.flip(pl_cpu, axis=1), 1, axis=1)))
        # does not since we cut off [:, :ne] and or do not take values below energy[0] into account

        # sanity checks
        assert np.allclose(energy_copy, energy)
        assert np.allclose(ij2ji_copy,  ij2ji)
        assert np.allclose(gg_copy,     gg_gold)
        assert np.allclose(gl_copy,     gl_gold)
        assert np.allclose(gr_copy,     gr_gold)
        assert np.array_equal(np.shape(gg_copy), np.shape(pg_cpu))
        assert np.array_equal(np.shape(gg_copy), np.shape(pl_cpu))
        assert np.array_equal(np.shape(gg_copy), np.shape(pr_cpu))

        # assert solution close to real solution
        # use Frobenius norm
        diff_g: np.double = np.linalg.norm(pg_gold - pg_cpu)
        diff_l: np.double = np.linalg.norm(pl_gold - pl_cpu)
        diff_r: np.double = np.linalg.norm(pr_gold - pr_cpu)
        print(f"Differences to Gold Solution g/l/r:  {diff_g:.4f}, {diff_l:.4f}, {diff_r:.4f}")
        abstol = 1e-12
        reltol = 1e-6
        assert diff_g <= abstol + reltol * np.max(np.abs(pg_gold))
        assert diff_l <= abstol + reltol * np.max(np.abs(pl_gold))
        assert diff_r <= abstol + reltol * np.max(np.abs(pr_gold))
        assert np.allclose(pg_gold, pg_cpu)
        assert np.allclose(pl_gold, pl_cpu)
        assert np.allclose(pr_gold, pr_cpu)

    print("The chosen implementation " + args.type + " is correct")
