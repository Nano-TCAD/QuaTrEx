"""Tests the different polarization implementation against dense 
solutionof rouyang
    todo merge with test_gold.py
"""
import numpy as np
import sys
from sparse import g2p_sparse
from gold_solution import read_solution
import cupy as cp
import cupyx
import numba
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, "../")
from dense import g2p_dense

if __name__ == "__main__":

    # limit the number of threads
    if numba.get_num_threads() >= 12:
        numba.config.NUMBA_NUM_THREADS = 12
    print("Number of used threads: ", numba.config.NUMBA_NUM_THREADS)

    energy, rows, columns, gf, pol = read_solution.load_pg(
        "gold_solution/data_GP.mat")
    energy = np.squeeze(energy)
    rows = np.squeeze(rows)
    columns = np.squeeze(columns)
    ij2ji = read_solution.find_idx_transposed(rows, columns)
    denergy: np.double = energy[1] - energy[0]
    ne = energy.shape[0]
    # create complex arrays
    gg_gold = gf[0] + 1j * gf[1]
    gl_gold = gf[2] + 1j * gf[3]
    gr_gold = gf[4] + 1j * gf[5]
    pg_gold = pol[0] + 1j * pol[1]
    pl_gold = pol[2] + 1j * pol[3]
    pr_gold = pol[4] + 1j * pol[5]

    # sanity checks
    # assume format of inputs
    assert gg_gold.ndim == 2
    assert gl_gold.ndim == 2
    assert gr_gold.ndim == 2
    assert energy.ndim == 1
    # assume same shape for gg, gl, r
    assert np.array_equal(np.shape(gg_gold), np.shape(gl_gold))
    assert np.array_equal(np.shape(gg_gold), np.shape(gr_gold))
    # assume energy is the second index
    assert np.shape(energy)[0] == np.shape(gg_gold)[1]
    # assume energy is evenly spaced
    assert np.allclose(np.diff(energy), np.diff(energy)[0])

    # check physical identities for inputs
    assert np.allclose(gl_gold, -gl_gold[ij2ji, :].conjugate())
    assert np.allclose(gg_gold, -gg_gold[ij2ji, :].conjugate())

    # get dense inputs
    gg_dense = read_solution.sparse_to_dense(rows, columns, gg_gold)
    gl_dense = read_solution.sparse_to_dense(rows, columns, gl_gold)
    gr_dense = read_solution.sparse_to_dense(rows, columns, gr_gold)

    # check size of arrays
    size_dense = gg_dense.nbytes / (1024**3)
    print(f"Size of one dense greens function in GB: {size_dense:.2f} GB")
    size_sparse = gg_gold.nbytes / (1024**3)
    print(f"Size of one sparse greens function in GB: {size_sparse:.2f} GB")

    pr_dense_dense, pl_dense_dense, pg_dense_dense, ep_dense = g2p_dense.g2p_dense(
        gr_dense, gl_dense, gg_dense, energy, workers=4)
    pg_conv, pl_conv, pr_conv = g2p_sparse.g2p_conv_cpu(
        denergy, ij2ji, gg_gold, gl_gold, gr_gold, numba.get_num_threads())
    pg_fft, pl_fft, pr_fft = g2p_sparse.g2p_fft_cpu(
        denergy, ij2ji, gg_gold, gl_gold, gr_gold, numba.get_num_threads())

    # calculate valid energy points
    energy_s = ne - 1
    energy_n = 2*ne - 1

    # transform to conv solution to dense to compare
    pg_conv_dense = read_solution.sparse_to_dense(rows, columns, pg_conv)
    pl_conv_dense = read_solution.sparse_to_dense(rows, columns, pl_conv)
    pr_conv_dense = read_solution.sparse_to_dense(rows, columns, pr_conv)

    # transform to conv solution to dense to compare
    pg_fft_dense = read_solution.sparse_to_dense(rows, columns, pg_fft)
    pl_fft_dense = read_solution.sparse_to_dense(rows, columns, pl_fft)
    pr_fft_dense = read_solution.sparse_to_dense(rows, columns, pr_fft)

    # calculate norm between solutions
    print(np.linalg.norm(pg_dense_dense[energy_s:energy_n] - pg_conv_dense))
    print(np.linalg.norm(pl_dense_dense[energy_s:energy_n] - pl_conv_dense))
    print(np.linalg.norm(pr_dense_dense[energy_s:energy_n] - pr_conv_dense))

    print(np.linalg.norm(pg_fft_dense - pg_conv_dense))
    print(np.linalg.norm(pl_fft_dense - pl_conv_dense))
    print(np.linalg.norm(pr_fft_dense - pr_conv_dense))

    print(np.linalg.norm(
        pg_fft_dense-pg_dense_dense[energy_s:energy_n]))
    print(np.linalg.norm(
        pl_fft_dense-pl_dense_dense[energy_s:energy_n]))
    print(np.linalg.norm(
        pr_fft_dense-pr_dense_dense[energy_s:energy_n]))

    # assert physical quantity
    assert np.allclose(pg_dense_dense, -np.conjugate(np.flip(pl_dense_dense, axis=0)))
    # todo the following do not hold, since energies bellow the grid are ignored.
    assert np.allclose(pg_conv_dense, -np.conjugate(np.roll(np.flip(pl_conv_dense, axis=0), 1, axis=0)))
    assert np.allclose(pg_fft_dense, -np.conjugate(np.roll(np.flip(pl_fft_dense, axis=0), 1, axis=0)))
