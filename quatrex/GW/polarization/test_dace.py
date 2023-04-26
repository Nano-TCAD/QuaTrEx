"""Test the dace implementation of both gpu and cpu against the gold solution
    todo merge with test_gold.py
"""
import numpy as np
import numpy.typing as npt
import sys
import os
import dace
import argparse
from dace.transformation.interstate import LoopToMap, StateFusion

from GW.polarization.sparse import g2p_sparse
from GW.gold_solution import read_solution

if __name__ == "__main__":
    # read gold solution
    solution_path = os.path.join('parent_path', "GW_SE_python", "gold_solution", "data_GP.mat")

    parser = argparse.ArgumentParser(
        description="Tests different implementation of the polarization calculation"
    )

    parser.add_argument("-f", "--file", default=solution_path, required=False)

    args = parser.parse_args()

    # load greens function
    energy, rows, columns, gg_gold, gl_gold, gr_gold    = read_solution.load_x(args.file, "g")
    # load polarization
    _, _, _, pg_gold, pl_gold, pr_gold                  = read_solution.load_x(args.file, "p")

    ij2ji:      npt.NDArray[np.int32]   = read_solution.find_idx_transposed(rows, columns)
    denergy:    np.double               = energy[1] - energy[0]
    ne:         np.int32                = np.int32(energy.shape[0])
    no:         np.int32                = np.int32(columns.shape[0])
    pre_factor: np.double               = -1.0j * denergy / (np.pi)


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

    # todo find out why not hold
    # G^{>} - G^{<} = G^{r} - G^{a}
    # G^{a} = (G^{r})^{H}
    # print(np.linalg.norm(np.real(gg_gold - gl_gold -
    #       gr_gold + gr_gold[ij2ji, :].conjugate())))
    # print(np.linalg.norm(np.imag(gg_gold - gl_gold -
    #       gr_gold + gr_gold[ij2ji, :].conjugate())))
    # assert np.allclose(gg_gold - gl_gold, gr_gold - gr_gold[ij2ji, :].conjugate())

    # assert physical quantities

    # load data to gpu
    # ij2ji_gpu = cp.asarray(ij2ji)
    # gg_gold_gpu = cp.asarray(gg_gold)
    # gl_gold_gpu = cp.asarray(gl_gold)
    # gr_gold_gpu = cp.asarray(gr_gold)

    # create zero outputs
    pg_cpu = np.zeros_like(gg_gold, dtype=np.cdouble)
    pl_cpu = np.zeros_like(gg_gold, dtype=np.cdouble)
    pr_cpu = np.zeros_like(gg_gold, dtype=np.cdouble)

    # compile to SDFG
    sdfg: dace.SDFG = g2p_sparse.g2p_fft_dace.to_sdfg(simplify=True)
    # sdfg.apply_transformations(LoopToMap)
    # sdfg.apply_transformations_repeated(StateFusion)
    # sdfg.apply_transformations(LoopToMap)
    # sdfg.simplify()
    csdfg = sdfg.compile()

    # call compiled function
    # csdfg(pre_factor=np.array([pre_factor]), ij2ji=ij2ji, gg=gg_gold,
    #     gl=gl_gold, gr=gr_gold, pg=pg_cpu, pl=pl_cpu, pr=pr_cpu, NE=ne, NO=no)

    # load data to cpu
    # pg_cpu = cp.asnumpy(pg_gpu)
    # pl_cpu = cp.asnumpy(pl_gpu)
    # pr_cpu = cp.asnumpy(pr_gpu)

    # assert physical identity
    # test: P^{>}_{ij}\left(E\right) = -P^{<}_{ij}\left(-E\right)^{*}
    # assert np.allclose(np.real(
    #     pg_cpu), np.real(-np.conjugate(np.roll(np.flip(pl_cpu, axis=1), 1, axis=1))))
    # assert np.allclose(np.imag(
    #     pg_cpu), np.imag(-np.conjugate(np.roll(np.flip(pl_cpu, axis=1), 1, axis=1))))
    # assert np.allclose(pg_cpu, -np.conjugate(np.roll(np.flip(pl_cpu, axis=1), 1, axis=1)))
    # does not since we cut off [:, :ne] and or do not take values below energy[0] into account


    # print difference to given solution
    # use Frobenius norm
    # diff_g = np.linalg.norm(pg_gold - pg_cpu)
    # diff_l = np.linalg.norm(pl_gold - pl_cpu)
    # diff_r = np.linalg.norm(pr_gold - pr_cpu)
    # print("Differences to Gold Solution g/l/r: ", diff_g, " ", diff_l, " ",
    #       diff_r)

    # # assert solution close to real solution
    # abstol = 1e-14
    # reltol = 1e-6
    # assert diff_g <= abstol + reltol * np.max(np.abs(pg_gold))
    # assert diff_l <= abstol + reltol * np.max(np.abs(pl_gold))
    # assert diff_r <= abstol + reltol * np.max(np.abs(pr_gold))
    # assert np.allclose(pg_gold, pg_cpu)
    # assert np.allclose(pl_gold, pl_cpu)
    # assert np.allclose(pr_gold, pr_cpu)
