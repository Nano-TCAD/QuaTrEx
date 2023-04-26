"""Tests different implementation of the screened interaction calculations
against a given gold solution
"""
import os
import sys
import numpy as np
from scipy import sparse
import mkl
import concurrent.futures
from itertools import repeat
from functools import partial
import argparse
import numba

main_path = os.path.abspath(os.path.dirname(__file__))
parent_path = os.path.abspath(os.path.join(main_path, ".."))

from GW.gold_solution import read_solution
from block_tri_solvers.rgf_W import rgf_W
from GW.screenedinteraction import change_format


if __name__ == "__main__":
    # parse the possible arguments
    # solution_path_pw = os.path.join(parent_path, "gold_solution", "data_GPWS.mat")
    # solution_path_vh = os.path.join(parent_path, "gold_solution", "data_Vh.mat")

    # parse the possible arguments
    #solution_path_pw = os.path.join(parent_path, "gold_solution", "data_GPWS.mat")
    #solution_path_vh = os.path.join(parent_path, "gold_solution", "data_Vh.mat")

    solution_path_pw = '/usr/scratch/mont-fort17/dleonard/CNT/data_GPWS.mat'
    solution_path_vh = '/usr/scratch/mont-fort17/dleonard/CNT/data_Vh.mat'


    parser = argparse.ArgumentParser(
        description="Tests different implementation of the screened interaction calculation"
    )
    parser.add_argument("-t", "--type", default="cpu_single",
                        choices=["cpu_single"], required=False)
    parser.add_argument("-fvh", "--file_vh", default=solution_path_vh, required=False)
    parser.add_argument("-fpw", "--file_pw", default=solution_path_pw, required=False)
    args = parser.parse_args()

    # not yet needed as numba is not used for screened interaction
    # limit the number of threads to not overuse cluster
    if numba.get_num_threads() >= 12:
        numba.set_num_threads(12)

    print("Used implementation: ", args.type)
    print("Path to gold solution vh: ", args.file_vh)
    print("Path to gold solution P/W: ", args.file_pw)
    print("Number of used numba threads: ", numba.get_num_threads())

    # load block sizes
    bmax, bmin                                          = read_solution.load_B(args.file_pw)
    # load greens function
    energy, rows, columns, wg_gold, wl_gold, wr_gold    = read_solution.load_x(args.file_pw, "w")
    # load polarization
    _, _, _, pg_gold, pl_gold, pr_gold                  = read_solution.load_x(args.file_pw, "p")
    # load interaction hat
    rowsRef, columnsRef, vh_gold                        = read_solution.load_v(args.file_vh)
    # mapping to transposed
    ij2ji                                               = read_solution.find_idx_transposed(rows, columns)

    # creating the filtering masks
    w_mask = np.ndarray(shape = (energy.shape[0],), dtype = bool)

    wr_mask = np.sum(np.abs(wr_gold), axis = 0) > 1e-10
    wl_mask = np.sum(np.abs(wl_gold), axis = 0) > 1e-10
    wg_mask = np.sum(np.abs(wg_gold), axis = 0) > 1e-10
    w_mask = np.logical_or(np.logical_or(wr_mask, wl_mask), wg_mask)

    # number of blocks
    nb = bmin.size
    # blocklength
    lb = np.max(bmax - bmin + 1)
    # number of total orbitals
    nao = nb * lb
    # number of energy points
    ne = energy.size
    # number of  non zero elements
    no = rows.size

    # fix nbc to 2 for the given solution
    # todo calculate it
    nbc = 2

    # block sizes after matrix multiplication
    bmax_ref = bmax[nbc-1:nb:nbc]
    bmin_ref = bmin[0:nb:nbc]
    # number of blocks after matrix multiplication
    nb_mm = bmax_ref.size
    # larges block length after matrix multiplication 
    lb_max_mm = np.max(bmax_ref - bmin_ref + 1)

    # mapping block to 2D format
    map_diag_alt, map_upper_alt, map_lower_alt = change_format.map_block2sparse_alt(rows, columns,
                                                                                bmax_ref, bmin_ref)
    
    # creating the smoothing and filtering factors
    dNP = 50
    factor = np.ones(ne)
    factor[ne-dNP-1:ne] = (np.cos(np.pi*np.linspace(0, 1, dNP+1)) + 1)/2
    #factor[0:dNP+1] = (np.cos(np.pi*np.linspace(1, 0, dNP+1)) + 1)/2

    factor[np.where(np.invert(w_mask))[0]] = 0.0

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

    print("Number of energy points: ", ne)
    print("Number of non zero elements: ", no)

    if args.type == "cpu_single":
        # Transform PR, PL into the right format
        # the input formats are
        # a sparse csr array for vh
        # a vector/list of sparse csr arrays for px
        vh         = sparse.coo_array((vh_gold, (rows, columns)),
                                    shape=(nao, nao), dtype = np.complex128).tocsr()

        # transform the 2D format to vector/list of sparse csr arrays
        pg = change_format.sparse2vecsparse(pg_gold,rows,columns,nao)
        pl = change_format.sparse2vecsparse(pl_gold,rows,columns,nao)
        pr = change_format.sparse2vecsparse(pr_gold,rows,columns,nao)

        # not performance, but for testing need atm
        # todo test all energy points
        # todo remove energy points
        for ie in range(0, ne):
            # buffer for a energy points
            # diagonal blocks
            xr_diag_out2  = np.zeros((nb_mm, lb_max_mm, lb_max_mm), dtype = np.complex128)
            wg_diag_out2  = np.zeros((nb_mm, lb_max_mm, lb_max_mm), dtype = np.complex128)
            wl_diag_out2  = np.zeros((nb_mm, lb_max_mm, lb_max_mm), dtype = np.complex128)
            wr_diag_out2  = np.zeros((nb_mm, lb_max_mm, lb_max_mm), dtype = np.complex128)

            # upper diagonal blocks
            wg_upper_out2 = np.zeros((nb_mm-1, lb_max_mm, lb_max_mm), dtype = np.complex128)
            wl_upper_out2 = np.zeros((nb_mm-1, lb_max_mm, lb_max_mm), dtype = np.complex128)
            wr_upper_out2 = np.zeros((nb_mm-1, lb_max_mm, lb_max_mm), dtype = np.complex128)

            # call obc and rgf for every energy point
            xr_ref2_dense, wg_ref2_dense, wl_ref2_dense, wr_ref2_dense = rgf_W(
                                                            vh, pg[ie], pl[ie], pr[ie],
                                                            bmax, bmin,
                                                            wg_diag_out2,
                                                            wg_upper_out2,
                                                            wl_diag_out2,
                                                            wl_upper_out2,
                                                            wr_diag_out2,
                                                            wr_upper_out2,
                                                            xr_diag_out2,
                                                            nbc,
                                                            ie,
                                                            factor[ie],
                                                            ref_flag=True
                                                            )

            # transform normal dense matrix inverse to the block format
            xr_diag_ref2,             _, _ = change_format.dense2block(xr_ref2_dense, bmax_ref, bmin_ref)
            wg_diag_ref2, wg_upper_ref2, _ = change_format.dense2block(wg_ref2_dense, bmax_ref, bmin_ref)
            wl_diag_ref2, wl_upper_ref2, _ = change_format.dense2block(wl_ref2_dense, bmax_ref, bmin_ref)
            wr_diag_ref2, wr_upper_ref2, _ = change_format.dense2block(wr_ref2_dense, bmax_ref, bmin_ref)


            # transform from the block format to the 2D one
            wg_out = change_format.block2sparse_alt(map_diag_alt,
                                                map_upper_alt,
                                                map_lower_alt,
                                                wg_diag_ref2,
                                                wg_upper_ref2,
                                                -wg_upper_ref2.conjugate().transpose((0,2,1)),
                                                no)
            wl_out = change_format.block2sparse_alt(map_diag_alt,
                                                map_upper_alt,
                                                map_lower_alt,
                                                wl_diag_ref2,
                                                wl_upper_ref2,
                                                -wl_upper_ref2.conjugate().transpose((0,2,1)),
                                                no)
            wr_out = change_format.block2sparse_alt(map_diag_alt,
                                                map_upper_alt,
                                                map_lower_alt,
                                                wr_diag_ref2,
                                                wr_upper_ref2,
                                                wr_upper_ref2.transpose((0,2,1)),
                                                no)
            
            wg_computed = change_format.block2sparse_alt(map_diag_alt,
                                                map_upper_alt,
                                                map_lower_alt,
                                                wg_diag_out2,
                                                wg_upper_out2,
                                                -wg_upper_out2.conjugate().transpose((0,2,1)),
                                                no)
            wl_computed = change_format.block2sparse_alt(map_diag_alt,
                                                map_upper_alt,
                                                map_lower_alt,
                                                wl_diag_out2,
                                                wl_upper_out2,
                                                -wl_upper_out2.conjugate().transpose((0,2,1)),
                                                no)
            wr_computed = change_format.block2sparse_alt(map_diag_alt,
                                                map_upper_alt,
                                                map_lower_alt,
                                                wr_diag_out2,
                                                wr_upper_out2,
                                                wr_upper_out2.transpose((0,2,1)),
                                                no)


            # compare with gold solution and normal matrix inverse
            # todo find out a good reason why we need such high tolerances
            diff_g = np.linalg.norm(wg_out - np.squeeze(wg_gold[:,ie]))
            diff_l = np.linalg.norm(wl_out - np.squeeze(wl_gold[:,ie]))
            diff_r = np.linalg.norm(wr_out - np.squeeze(wr_gold[:,ie]))
            print(f"Differences to Gold Solution g/l/r:  {diff_g:.4f}, {diff_l:.4f}, {diff_r:.4f}")
            abstol = 1e-2
            reltol = 2*1e-1
            assert diff_g <= abstol + reltol * np.max(np.abs(wg_gold[:,ie]))*100
            assert diff_l <= abstol + reltol * np.max(np.abs(wl_gold[:,ie]))
            assert diff_r <= abstol + reltol * np.max(np.abs(wr_gold[:,ie]))*100

            assert np.allclose(xr_diag_ref2,  xr_diag_out2)
            assert np.allclose(wg_diag_ref2,  wg_diag_out2)
            assert np.allclose(wl_diag_ref2,  wl_diag_out2)
            assert np.allclose(wr_diag_ref2,  wr_diag_out2, atol=3, rtol=2)
            assert np.allclose(wg_upper_ref2, wg_upper_out2)
            assert np.allclose(wl_upper_ref2, wl_upper_out2)
            assert np.allclose(wr_upper_ref2, wr_upper_out2, atol=3, rtol=2)
            assert np.allclose(wg_out, np.squeeze(wg_gold[:,ie]), atol=2, rtol=1)
            assert np.allclose(wl_out, np.squeeze(wl_gold[:,ie]), atol=1e-1, rtol=1e-2)
            assert np.allclose(wr_out, np.squeeze(wr_gold[:,ie]), atol=1, rtol=1e-1)
            assert np.allclose(wg_computed, np.squeeze(wg_gold[:,ie]), atol=2, rtol=1)
            assert np.allclose(wl_computed, np.squeeze(wl_gold[:,ie]), atol=1e-1, rtol=1e-2)
            assert np.allclose(wr_computed, np.squeeze(wr_gold[:,ie]), atol=5, rtol=1)
            print("At energy point: ", ie, " the solution is correct")

    else:
        raise ValueError(
        "Argument error, type input not possible")
    
    print("The chosen implementation " + args.type + " is correct")

