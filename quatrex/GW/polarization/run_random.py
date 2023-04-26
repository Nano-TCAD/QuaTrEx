"""Calculate the polarization with random input data"""
import numpy as np
import sys
from GW.polarization.sparse import g2p_sparse
from GW.gold_solution import read_solution
from GW.polarization.initialization import gf_init
import cupy as cp
import cupyx
import numba
# caution: path[0] is reserved for script path (or '' in REPL)



if __name__ == "__main__":

    # limit the number of threads
    if numba.get_num_threads() >= 12:
        numba.config.NUMBA_NUM_THREADS = 12
    print("Number of used threads: ", numba.config.NUMBA_NUM_THREADS)


    # number of energy points
    ne = 100
    # number of orbitals -> 0.01*nao*nao are the nonzero amount of nnz
    nao = 50
    # random seed
    seed = 2
    energy, rows, columns, gg, gl, gr = gf_init.init_sparse(ne, nao, seed)
    energy = np.squeeze(energy)
    rows = np.squeeze(rows)
    columns = np.squeeze(columns)
    ij2ji = read_solution.find_idx_transposed(rows, columns)
    denergy: np.double = energy[1] - energy[0]

    # sanity checks
    # assume format of inputs
    assert gg.ndim == 2
    assert gl.ndim == 2
    assert gr.ndim == 2
    assert energy.ndim == 1
    # assume same shape for gg, gl, r
    assert np.array_equal(np.shape(gg), np.shape(gl))
    assert np.array_equal(np.shape(gg), np.shape(gr))
    # assume energy is the second index
    assert np.shape(energy)[0] == np.shape(gg)[1]
    # assume energy is evenly spaced
    assert np.allclose(np.diff(energy), np.diff(energy)[0])

    # check physical identities for inputs
    assert np.allclose(gr, gr[ij2ji,:])
    assert np.allclose(gl, -gl[ij2ji,:].conjugate())
    assert np.allclose(gg, -gg[ij2ji,:].conjugate())
    assert np.allclose(gg - gl, gr - gr[ij2ji,:].conjugate())

    # read out cpu/gpu
    try:
        xpu = sys.argv[1]
    except IndexError:
        print("Needs a command line argument")
        print("Use default cpu")
        xpu = "cpu"

    # testing on cpu or gpu
    if xpu == "cpu":

        pg_cpu, pl_cpu, pr_cpu = g2p_sparse.g2p_fft_cpu(
            denergy, ij2ji, gg, gl, gr, numba.get_num_threads())
        print("cpu_finished")

    elif xpu == "gpu":
        # load data to gpu
        ij2ji_gpu = cp.asarray(ij2ji)
        gg_gpu = cp.asarray(gg)
        gl_gpu = cp.asarray(gl)
        gr_gpu = cp.asarray(gr)

        pg_gpu, pl_gpu, pr_gpu = g2p_sparse.g2p_fft_gpu(
            denergy, ij2ji_gpu, gg_gpu, gl_gpu, gr_gpu)

        # load data to cpu
        pg_cpu = cp.asnumpy(pg_gpu)
        pl_cpu = cp.asnumpy(pl_gpu)
        pr_cpu = cp.asnumpy(pr_gpu)
        print("gpu_finished")
    else:
        raise ValueError(
            "Command line argument has to be either str(cpu) or str(gpu)")

    # test: P^{>}_{ij}\left(E\right) = -P^{<}_{ij}\left(-E\right)^{*}
    # assert np.allclose(pg_cpu, -np.conjugate(np.roll(np.flip(pl_cpu, axis=1), 1, axis=1)))
    # does not since we cut off [:, :ne] and or do not take values below energy[0] into account



