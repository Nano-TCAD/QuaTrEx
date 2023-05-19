"""Tests different implementation of the screened interaction calculations
against a given gold solution
"""
import os
import sys
import numpy as np
from scipy import sparse
import argparse
import numba

main_path = os.path.abspath(os.path.dirname(__file__))
parent_path = os.path.abspath(os.path.join(main_path, "..", ".."))
sys.path.append(parent_path)

from GW.gold_solution import read_solution
from GW.screenedinteraction.kernel import p2w_cpu
# todo if gpu avail
from GW.screenedinteraction.kernel import p2w_gpu
from utils import change_format
from OMEN_structure_matrices import OMENHamClass

if __name__ == "__main__":
    # parse the possible arguments
    solution_path = "/usr/scratch/mont-fort17/dleonard/CNT/"
    solution_path_gw = os.path.join(solution_path, "data_GPWS_04.mat")
    solution_path_vh = os.path.join(solution_path, "data_Vh_4.mat")
    hamiltonian_path = os.path.join(solution_path, "CNT_newwannier")
    parser = argparse.ArgumentParser(
        description="Tests different implementation of the screened interaction calculation"
    )
    parser.add_argument("-t", "--type", default="cpu",
                        choices=["cpu_pool", "cpu",
                            "cpu_alt", "cpu_block"
                            "gpu"], required=False)
    parser.add_argument("-fvh", "--file_vh", default=solution_path_vh, required=False)
    parser.add_argument("-fpw", "--file_gw", default=solution_path_gw, required=False)
    parser.add_argument("-fhm", "--file_hm", default=hamiltonian_path, required=False)
    args = parser.parse_args()

    # set number of threads for the p2w step
    w_mkl_threads = 8
    w_worker_threads = 10

    print("Used implementation: ", args.type)
    print("Path to gold solution vh: ", args.file_vh)
    print("Path to gold solution P/W: ", args.file_gw)
    print("Number of used numba threads: ", numba.get_num_threads())
    print("Number of used mkl threads: ", w_mkl_threads)
    print("Number of used pool workers: ", w_worker_threads)

    # load block sizes
    bmax, bmin                                          = read_solution.load_B(args.file_gw)
    # load greens function
    energy, rows, columns, wg_gold, wl_gold, wr_gold    = read_solution.load_x(args.file_gw, "w")
    # load polarization
    _, _, _, pg_gold, pl_gold, pr_gold                  = read_solution.load_x(args.file_gw, "p")
    # load interaction hat
    rowsRef, columnsRef, vh_gold                        = read_solution.load_v(args.file_vh)
    # mapping to transposed
    ij2ji                                               = change_format.find_idx_transposed(rows, columns)
    
    # one orbital on C atoms, two same types
    no_orb = np.array([1, 1])
    # create hamiltonian object
    hamiltionian_obj = OMENHamClass.Hamiltonian(args.file_hm, no_orb, 0)

    # creating the filtering masks
    w_mask = np.ndarray(shape = (energy.shape[0],), dtype = bool)
    # masks describe if energy point got calculated
    wr_mask = np.sum(np.abs(wr_gold), axis = 0) > 1e-10
    wl_mask = np.sum(np.abs(wl_gold), axis = 0) > 1e-10
    wg_mask = np.sum(np.abs(wg_gold), axis = 0) > 1e-10
    w_mask = np.logical_or(np.logical_or(wr_mask, wl_mask), wg_mask)

    # number of blocks
    nb = bmin.size
    # blocklength
    lb = np.max(bmax - bmin + 1)
    # number of total orbitals
    nao = np.max(bmax) + 1
    # number of energy points
    ne = energy.size
    # number of  non zero elements
    no = rows.size
    # fix nbc to 2 for the given solution
    # todo calculate it
    nbc = 2
    # block sizes after matrix multiplication
    bmax_mm = bmax[nbc-1:nb:nbc]
    bmin_mm = bmin[0:nb:nbc]
    # number of blocks after matrix multiplication
    nb_mm = bmax_mm.size
    # larges block length after matrix multiplication
    lb_max_mm = np.max(bmax_mm - bmin_mm + 1)
    # create map from block format to 2D format after matrix multiplication
    map_diag_mm2m, map_upper_mm2m, map_lower_mm2m = change_format.map_block2sparse_alt(rows, columns,
                                                                                 bmax_mm, bmin_mm)
    # creating the smoothing and filtering factors
    # number of points to smooth the edges of the Green's Function
    dnp = 50
    factor_w = np.ones(ne)
    factor_w[ne-dnp-1:ne] = (np.cos(np.pi*np.linspace(0, 1, dnp+1)) + 1)/2
    factor_w[np.where(np.invert(w_mask))[0]] = 0.0

    # sanity checks
    assert np.max(columns) == nao - 1
    assert np.max(rows) == nao - 1
    assert wg_gold.ndim == 2
    assert energy.ndim == 1
    assert np.array_equal(np.shape(wg_gold), np.shape(wl_gold))
    assert np.array_equal(np.shape(wg_gold), np.shape(wr_gold))
    # check that vh, px/wx have the same sparsity pattern
    assert np.allclose(rowsRef, rows)
    assert np.allclose(columnsRef, columns)
    # assume energy is the second index
    assert np.shape(energy)[0] == np.shape(wg_gold)[1]
    # check if vh is hermitian
    assert np.allclose(vh_gold, np.conjugate(vh_gold[ij2ji]))

    print(f"#Energy: {ne} #nnz: {no} #orbitals: {nao}")

    # make input/output contiguous in orbitals
    pg_gold = pg_gold.transpose()
    pl_gold = pl_gold.transpose()
    pr_gold = pr_gold.transpose()
    wg_gold = wg_gold.transpose()
    wl_gold = wl_gold.transpose()
    wr_gold = wr_gold.transpose()

    energy_copy = np.copy(energy)
    ij2ji_copy = np.copy(ij2ji)
    pg_copy = np.copy(pg_gold)
    pl_copy = np.copy(pl_gold)
    pr_copy = np.copy(pr_gold)
    vh_copy = np.copy(vh_gold)

    if args.type == "cpu_pool":
        # transform from 2D format to list/vector of sparse arrays format
        pg_cpu_vec = change_format.sparse2vecsparse_v2(pg_gold, rows, columns, nao)
        pl_cpu_vec = change_format.sparse2vecsparse_v2(pl_gold, rows, columns, nao)
        pr_cpu_vec = change_format.sparse2vecsparse_v2(pr_gold, rows, columns, nao)
        # from data vector to sparse csr format
        vh = sparse.coo_array((vh_gold, (rows, columns)),
                            shape=(nao, nao), dtype = np.complex128).tocsr()

        wg_diag, wg_upper, wl_diag, wl_upper, wr_diag, wr_upper, _, _ = p2w_cpu.p2w_pool_mpi_cpu(
                                                                                            hamiltionian_obj, energy,
                                                                                            pg_cpu_vec, pl_cpu_vec,
                                                                                            pr_cpu_vec, vh,
                                                                                            factor_w, w_mkl_threads,
                                                                                            w_worker_threads)
        # lower diagonal blocks from physics identity
        wg_lower = -wg_upper.conjugate().transpose((0,1,3,2))
        wl_lower = -wl_upper.conjugate().transpose((0,1,3,2))
        wr_lower = wr_upper.transpose((0,1,3,2))

        wg_cpu = change_format.block2sparse_energy_alt(map_diag_mm2m, map_upper_mm2m,
                                                        map_lower_mm2m, wg_diag, wg_upper,
                                                        wg_lower, no, ne,
                                                        energy_contiguous=False)
        wl_cpu = change_format.block2sparse_energy_alt(map_diag_mm2m, map_upper_mm2m,
                                                        map_lower_mm2m, wl_diag, wl_upper,
                                                        wl_lower, no, ne,
                                                        energy_contiguous=False)
        wr_cpu = change_format.block2sparse_energy_alt(map_diag_mm2m, map_upper_mm2m,
                                                        map_lower_mm2m, wr_diag, wr_upper,
                                                        wr_lower, no, ne,
                                                        energy_contiguous=False)

    elif args.type == "cpu":
        # transform from 2D format to list/vector of sparse arrays format
        pg_cpu_vec = change_format.sparse2vecsparse_v2(pg_gold, rows, columns, nao)
        pl_cpu_vec = change_format.sparse2vecsparse_v2(pl_gold, rows, columns, nao)
        pr_cpu_vec = change_format.sparse2vecsparse_v2(pr_gold, rows, columns, nao)
        # from data vector to sparse csr format
        vh = sparse.coo_array((vh_gold, (rows, columns)),
                            shape=(nao, nao), dtype = np.complex128).tocsr()

        wg_diag, wg_upper, wl_diag, wl_upper, wr_diag, wr_upper, _, _ = p2w_cpu.p2w_mpi_cpu(
                                                                                            hamiltionian_obj, energy,
                                                                                            pg_cpu_vec, pl_cpu_vec,
                                                                                            pr_cpu_vec, vh,
                                                                                            factor_w, mkl_threads=w_mkl_threads
                                                                                            )
        # lower diagonal blocks from physics identity
        wg_lower = -wg_upper.conjugate().transpose((0,1,3,2))
        wl_lower = -wl_upper.conjugate().transpose((0,1,3,2))
        wr_lower = wr_upper.transpose((0,1,3,2))

        wg_cpu = change_format.block2sparse_energy_alt(map_diag_mm2m, map_upper_mm2m,
                                                        map_lower_mm2m, wg_diag, wg_upper,
                                                        wg_lower, no, ne,
                                                        energy_contiguous=False)
        wl_cpu = change_format.block2sparse_energy_alt(map_diag_mm2m, map_upper_mm2m,
                                                        map_lower_mm2m, wl_diag, wl_upper,
                                                        wl_lower, no, ne,
                                                        energy_contiguous=False)
        wr_cpu = change_format.block2sparse_energy_alt(map_diag_mm2m, map_upper_mm2m,
                                                        map_lower_mm2m, wr_diag, wr_upper,
                                                        wr_lower, no, ne,
                                                        energy_contiguous=False)
    elif args.type == "cpu_alt":
        wg_cpu, wl_cpu, wr_cpu = p2w_cpu.p2w_mpi_cpu_alt(
                                            hamiltionian_obj,
                                            ij2ji,
                                            rows,
                                            columns,
                                            pg_gold,
                                            pl_gold,
                                            pr_gold,
                                            vh_gold,
                                            factor_w,
                                            map_diag_mm2m,
                                            map_upper_mm2m,
                                            map_lower_mm2m,
                                            mkl_threads=w_mkl_threads
                                )
    elif args.type == "cpu_block":
        wg_cpu, wl_cpu, wr_cpu = p2w_cpu.p2w_mpi_cpu_block(
                                            hamiltionian_obj,
                                            ij2ji,
                                            rows,
                                            columns,
                                            pg_gold,
                                            pl_gold,
                                            pr_gold,
                                            vh_gold,
                                            factor_w,
                                            map_diag_mm2m,
                                            map_upper_mm2m,
                                            map_lower_mm2m,
                                            mkl_threads=w_mkl_threads
                                )
    elif args.type == "gpu":
        slicing_obj = p2w_gpu.Slice_w2p(bmax, bmin, nao, nbc)
        # todo does not work
        wg_cpu, wl_cpu, wr_cpu = p2w_gpu.p2w_mpi_gpu(
                                            slicing_obj,
                                            ij2ji,
                                            rows,
                                            columns,
                                            pg_gold,
                                            pl_gold,
                                            pr_gold,
                                            vh_gold,
                                            factor_w,
                                            map_diag_mm2m,
                                            map_upper_mm2m,
                                            map_lower_mm2m,
                                            mkl_threads=w_mkl_threads
                                )
    else:
        raise ValueError(
        "Argument error, type input not possible")

    # compare with gold solution and normal matrix inverse
    assert np.allclose(energy_copy, energy)
    assert np.allclose(ij2ji_copy, ij2ji)
    assert np.allclose(pg_copy, pg_gold)
    assert np.allclose(pl_copy, pl_gold)
    assert np.allclose(pr_copy, pr_gold)
    assert np.allclose(vh_copy, vh_gold)
    diff_g = np.linalg.norm(wg_cpu - np.squeeze(wg_gold))
    diff_l = np.linalg.norm(wl_cpu - np.squeeze(wl_gold))
    diff_r = np.linalg.norm(wr_cpu - np.squeeze(wr_gold))
    print(f"Differences to Gold Solution g/l/r:  {diff_g:.4f}, {diff_l:.4f}, {diff_r:.4f}")
    abstol = 1e-1
    reltol = 1e-1
    assert diff_g <= abstol + reltol * np.max(np.abs(wg_gold))
    assert diff_l <= abstol + reltol * np.max(np.abs(wl_gold))
    assert diff_r <= abstol + reltol * np.max(np.abs(wr_gold))
    assert np.allclose(wg_gold, wg_cpu, rtol=1e-2, atol=1e-2)
    assert np.allclose(wl_gold, wl_cpu, atol=1e-6, rtol=1e-6)
    assert np.allclose(wr_gold, wr_cpu, atol=1e-6, rtol=1e-6)

    print("The chosen implementation " + args.type + " is correct")

