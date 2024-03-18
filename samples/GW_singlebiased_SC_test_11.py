# Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.
"""
Example a sc-GW iteration with MPI+CUDA.
With transposition through network.
This application is for a tight-binding InAs nanowire. 
See the different GW step folders for more explanations.
"""
import sys
import numpy as np
import cupyx as cpx
import cupy as cp
import numpy.typing as npt
import os
import argparse
import pickle
import mpi4py
from scipy import sparse
import time

mpi4py.rc.initialize = False  # do not initialize MPI automatically
mpi4py.rc.finalize = False  # do not finalize MPI automatically
from mpi4py import MPI

from quatrex.bandstructure.calc_band_edge import get_band_edge_mpi_interpol, get_band_edge_mpi
from quatrex.bandstructure.calc_band_edge import get_band_edge_mpi_interpol_2
from quatrex.GW.polarization.kernel import g2p_cpu
from quatrex.GW.selfenergy.kernel import gw2s_cpu
from quatrex.GW.gold_solution import read_solution
from quatrex.GW.screenedinteraction.kernel import p2w_cpu
from quatrex.GW.coulomb_matrix.read_coulomb_matrix import load_V
from quatrex.GreensFunction import calc_GF_pool
from quatrex.OMEN_structure_matrices import OMENHamClass
from quatrex.OMEN_structure_matrices.construct_CM import construct_coulomb_matrix
from quatrex.utils import change_format
from quatrex.utils import utils_gpu
from quatrex.utils.matrix_creation import get_number_connected_blocks
from quatrex.Phonon import electron_phonon_selfenergy

if utils_gpu.gpu_avail():
    try:
        from quatrex.GreensFunction import calc_GF_pool_GPU, calc_GF_pool_GPU_memopt_2
        from quatrex.GW.screenedinteraction.kernel import p2w_gpu, p2w_gpu_improved, p2w_gpu_improved_2
        from quatrex.GW.polarization.kernel import g2p_gpu
        from quatrex.GW.selfenergy.kernel import gw2s_gpu
    except ImportError:
        print("GPU import error, make sure you have the right GPU driver and CUDA version installed")

if __name__ == "__main__":
    MPI.Init_thread(required=MPI.THREAD_FUNNELED)
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    name = MPI.Get_processor_name()

    # assume every rank has enough memory to read the initial data
    # path to solution
    scratch_path = "/usr/scratch/mont-fort17/dleonard/GW_paper"
    solution_path = os.path.join(scratch_path, "Si_Nanowire_18/")
    solution_path_gw = os.path.join(solution_path, "data_GPWS_cf_ephn_memory2_sinwNBC1_0V.mat")
    #solution_path_gw2 = os.path.join(solution_path, "data_GPWS_IEDM_memory2_GNR_04V.mat")
    solution_path_vh = os.path.join(solution_path, "data_Vh_CF_SINW_0v.mat")
    solution_path_H = os.path.join(solution_path, "data_H_CF_SINW_0v.mat")
    solution_path_S = os.path.join(solution_path, "data_S_CF_SINW_0v.mat")
    hamiltonian_path = solution_path
    parser = argparse.ArgumentParser(description="Example of the first GW iteration with MPI+CUDA")
    parser.add_argument("-fvh", "--file_vh", default=solution_path_vh, required=False)
    parser.add_argument("-fpw", "--file_gw", default=solution_path_gw, required=False)
    parser.add_argument("-fhm", "--file_hm", default=hamiltonian_path, required=False)
    # change manually the used implementation inside the code
    parser.add_argument("-t", "--type", default="gpu", choices=["cpu", "gpu"], required=False)
    parser.add_argument("-nt", "--net_transpose", default=False, type=bool, required=False)
    parser.add_argument("-p", "--pool", default=True, type=bool, required=False)
    args = parser.parse_args()
    # check if gpu is available
    if args.type in ("gpu"):
        if not utils_gpu.gpu_avail():
            print("No gpu available")
            sys.exit(1)
    # print chosen implementation
    if args.type in ("cpu"):
        if rank == 0:
            print("Only GPU implementation for this sample", flush=True)
        sys.exit(1)
    if(rank == 0):
        print(f"Using {args.type} implementation", flush = True)

    # create hamiltonian object
    # one orbital on C atoms, two same types
    no_orb = np.array([1,4])
    # Factor to extract smaller matrix blocks (factor * unit cell size < current block size based on Smin_dat)
    NCpSC = 4
    Vappl = 0.6
    energy = np.linspace(-5, 1, 16, endpoint=True, dtype=float)  # Energy Vector
    Idx_e = np.arange(energy.shape[0])  # Energy Index Vector
    EPHN = np.array([0.0])  # Phonon energy
    DPHN = np.array([2.5e-3])  # Electron-phonon coupling
    hamiltonian_obj = OMENHamClass.Hamiltonian(args.file_hm, no_orb, potential_type = 'atomic', bias_point = 13, Vappl=Vappl, rank=rank, layer_matrix = '/Layer_Matrix.dat', homogenize = True, NCpSC = 4)
    serial_ham = pickle.dumps(hamiltonian_obj)
    broadcasted_ham = comm.bcast(serial_ham, root=0)
    hamiltonian_obj = pickle.loads(broadcasted_ham)
    # Extract neighbor indices
    rows = hamiltonian_obj.rows
    columns = hamiltonian_obj.columns

    # hamiltonian object has 1-based indexing
    bmax = hamiltonian_obj.Bmax - 1
    bmin = hamiltonian_obj.Bmin - 1

    #reading reference solution
    gg_gold = np.load(os.path.join(solution_path, "gg.npy"))
    gl_gold = np.load(os.path.join(solution_path, "gl.npy"))
    pg_gold = np.load(os.path.join(solution_path, "pg.npy"))
    pl_gold = np.load(os.path.join(solution_path, "pl.npy"))
    wg_gold = np.load(os.path.join(solution_path, "wg.npy"))
    wl_gold = np.load(os.path.join(solution_path, "wl.npy"))
    sg_gold = np.load(os.path.join(solution_path, "sg.npy"))
    sl_gold = np.load(os.path.join(solution_path, "sl.npy"))
    sr_gold = np.load(os.path.join(solution_path, "sr.npy"))
    sphg_gold = np.load(os.path.join(solution_path, "sg_phn.npy"))
    sphl_gold = np.load(os.path.join(solution_path, "sl_phn.npy"))
    sphr_gold = np.load(os.path.join(solution_path, "sr_phn.npy"))
    # energy_in, rows, columns, gg_gold, gl_gold, _ = read_solution.load_x_optimized(solution_path_gw, "g")
    # energy_in, rows_p, columns_p, pg_gold, pl_gold, pr_gold = read_solution.load_x_optimized(solution_path_gw, "p")
    # energy_in, rows_w, columns_w, wg_gold, wl_gold, _ = read_solution.load_x_optimized(solution_path_gw, "w")
    # energy_in, rows_s, columns_s, sg_gold, sl_gold, sr_gold = read_solution.load_x_optimized(solution_path_gw, "s")
    # energy_in, rows_sph, columns_sph, sphg_gold, sphl_gold, sphr_gold = read_solution.load_x_optimized(solution_path_gw, "sph")
    # rowsRef, columnsRef, vh_gold = read_solution.load_v(solution_path_vh)
    # rowsRefH, columnsRefH, H_gold = read_solution.load_v(solution_path_H)
    # rowsRefS, columnsRefS, S_gold = read_solution.load_v(solution_path_S)

    ij2ji: npt.NDArray[np.int32] = change_format.find_idx_transposed(rows, columns)
    denergy: npt.NDArray[np.double] = energy[1] - energy[0]
    ne: np.int32 = np.int32(energy.shape[0])
    no: np.int32 = np.int32(columns.shape[0])
    pre_factor: np.complex128 = -1.0j * denergy / (np.pi)
    nao: np.int64 = np.max(bmax) + 1

    # vh = sparse.coo_array((vh_gold, (np.squeeze(rowsRef), np.squeeze(columnsRef))),
    #                       shape=(nao, nao),
    #                       dtype=np.complex128).tocsr()
    # H_in = sparse.coo_array((H_gold, (np.squeeze(rowsRefH), np.squeeze(columnsRefH))),
    #                         shape=(nao, nao),
    #                         dtype=np.complex128).tocsr()
    # S_in = sparse.coo_array((S_gold, (np.squeeze(rowsRefS), np.squeeze(columnsRefS))),
    #                         shape=(nao, nao),
    #                         dtype=np.complex128).tocsr()
    data_shape = np.array([rows.shape[0], energy.shape[0]], dtype=np.int32)

    # Creating the mask for the energy range of the deleted W elements given by the reference solution
    w_mask = np.ndarray(shape=(energy.shape[0], ), dtype=bool)

    # wr_mask = np.sum(np.abs(wr_gold), axis=0) > 1e-10
    # wl_mask = np.sum(np.abs(wl_gold), axis=0) > 1e-10
    # wg_mask = np.sum(np.abs(wg_gold), axis=0) > 1e-10
    # w_mask = np.logical_or(np.logical_or(wr_mask, wl_mask), wg_mask)

    map_diag, map_upper, map_lower = change_format.map_block2sparse_alt(rows, columns, bmax, bmin)

    # number of blocks
    nb = hamiltonian_obj.Bmin.shape[0]
    nbc = get_number_connected_blocks(hamiltonian_obj.NH, bmin, bmax, rows, columns)
    bmax_mm = bmax[nbc - 1:nb:nbc]
    bmin_mm = bmin[0:nb:nbc]

    map_diag_mm, map_upper_mm, map_lower_mm = change_format.map_block2sparse_alt(rows, columns, bmax_mm, bmin_mm)

    # Here we create a new mapping for the m and l matrices. We start by
    # constructing a dummy matrix, which will allow us to determine the
    # sparsity structure a priori.
    from quatrex.GW.screenedinteraction.kernel.p2w_gpu_improved import spgemm, spgemm_direct

    dummy = cp.sparse.csr_matrix(
        sparse.coo_array(
            (np.ones(rows.size, dtype=np.float32), (rows, columns)),
            shape=(nao, nao),
        )
    )

    dummy_m = (cp.sparse.identity(nao) - spgemm(dummy, dummy)).tocoo()
    rows_m = cp.asnumpy(dummy_m.row)
    columns_m = cp.asnumpy(dummy_m.col)

    ij2ji_m: npt.NDArray[np.int32] = change_format.find_idx_transposed(
        rows_m, columns_m
    )
    map_diag_m, map_upper_m, map_lower_m = change_format.map_block2sparse_alt(
        rows_m, columns_m, bmax_mm, bmin_mm
    )

    dummy_l = spgemm(spgemm(dummy, dummy), dummy.T).tocoo()
    rows_l = cp.asnumpy(dummy_l.row)
    columns_l = cp.asnumpy(dummy_l.col)

    ij2ji_l: npt.NDArray[np.int32] = change_format.find_idx_transposed(
        rows_l, columns_l
    )
    map_diag_l, map_upper_l, map_lower_l = change_format.map_block2sparse_alt(
        rows_l, columns_l, bmax_mm, bmin_mm
    )

    # assert np.allclose(rows_p, rows)
    # assert np.allclose(columns, columns_p)
    # assert np.allclose(rows_w, rows)
    # assert np.allclose(columns, columns_w)
    # assert np.allclose(rows_s, rows)
    # assert np.allclose(columns, columns_s)

    # assert pg_gold.shape[0] == rows.shape[0]

    if rank == 0:
        # print size of data
        print(f"#Energy: {data_shape[1]} #nnz: {data_shape[0]}", flush = True)

    # computation parameters----------------------------------------------------
    # set number of threads for the p2w step
    w_mkl_threads = 1
    w_worker_threads = 8
    # set number of threads for the h2g step
    gf_mkl_threads = 1
    gf_worker_threads = 8

    # physical parameter -----------

    # Fermi Level of Left Contact
    energy_fl = -2.0362
    # Fermi Level of Right Contact
    energy_fr = energy_fl - Vappl
    # Temperature in Kelvin
    temp = 300
    # relative permittivity
    epsR = 1.0
    # DFT Conduction Band Minimum
    ECmin = -2.0662

    # Phyiscal Constants -----------

    e = 1.6022e-19
    eps0 = 8.854e-12
    hbar = 1.0546e-34

    # Fermi Level to Band Edge Difference
    dEfL_EC = energy_fl - ECmin
    dEfR_EC = energy_fr - ECmin

    # create the corresponding factor to mask
    # number of points to smooth the edges of the Green's Function
    dnp = 50
    factor_w = np.ones(ne)
    #factor_w[ne - dnp - 1:ne] = (np.cos(np.pi * np.linspace(0, 1, dnp + 1)) + 1) / 2
    #factor_w[np.where(np.invert(w_mask))[0]] = 0.0

    # create factor for the Green's Function
    factor_g = np.ones(ne)
    #factor_g[ne - dnp - 1:ne] = (np.cos(np.pi * np.linspace(0, 1, dnp + 1)) + 1) / 2
    #factor_g[0:dnp + 1] = (np.cos(np.pi * np.linspace(1, 0, dnp + 1)) + 1) / 2

    vh = construct_coulomb_matrix(hamiltonian_obj, epsR, eps0, e, diag=False, orb_uniform = True)
    vh1d = np.squeeze(np.asarray(vh[np.copy(rows), np.copy(columns)].reshape(-1)))

    # assert np.allclose(V_sparse.toarray(), vh.toarray())
    # assert np.allclose(H_in.toarray(), hamiltonian_obj.Hamiltonian['H_4'].toarray())
    # assert np.allclose(S_in.toarray(), hamiltonian_obj.Overlap['H_4'].toarray())

    # calculation of data distribution per rank---------------------------------

    # split nnz/energy per rank
    data_per_rank = data_shape // size

    # create array with energy size distribution
    count = np.repeat(data_per_rank.reshape(-1, 1), size, axis=1)
    count[:, size - 1] += data_shape % size

    # displacements in nnz/energy
    disp = data_per_rank.reshape(-1, 1) * np.arange(size)

    # slice energy vector
    energy_loc = energy[disp[1, rank]:disp[1, rank] + count[1, rank]]
    Idx_e_loc = Idx_e[disp[1, rank]:disp[1, rank] + count[1, rank]]

    # split up the factor between the ranks
    factor_w_loc = factor_w[disp[1, rank]:disp[1, rank] + count[1, rank]]
    factor_g_loc = factor_g[disp[1, rank]:disp[1, rank] + count[1, rank]]

    # print rank distribution
    print(f"Rank: {rank} #Energy/rank: {count[1,rank]} #nnz/rank: {count[0,rank]}", name, flush = True)

    # adding checks
    assert energy_loc.size == count[1, rank]

    # create needed data types--------------------------------------------------

    # arrays filled with complex doubles
    BASE_TYPE = MPI.Datatype(MPI.DOUBLE_COMPLEX)
    base_size = np.dtype(np.complex128).itemsize

    # column type of orginal matrix
    COLUMN = BASE_TYPE.Create_vector(data_shape[0], 1, data_shape[1])
    COLUMN_RIZ = COLUMN.Create_resized(0, base_size)
    MPI.Datatype.Commit(COLUMN_RIZ)
    MPI.Datatype.Commit(COLUMN)

    # row type of original transposed matrix
    ROW = BASE_TYPE.Create_vector(data_shape[1], 1, data_shape[0])
    ROW_RIZ = ROW.Create_resized(0, base_size)
    MPI.Datatype.Commit(ROW_RIZ)
    MPI.Datatype.Commit(ROW)

    # send type g2p
    # column type of split up in nnz
    G2P_S = BASE_TYPE.Create_vector(count[1, rank], 1, data_shape[0])
    G2P_S_RIZ = G2P_S.Create_resized(0, base_size)
    MPI.Datatype.Commit(G2P_S)
    MPI.Datatype.Commit(G2P_S_RIZ)

    # receive types g2p
    # vector of size of #ranks
    # multi column data type for every rank size #energy not divisible
    G2P_R = np.array([BASE_TYPE.Create_vector(count[0, rank], count[1, i], data_shape[1]) for i in range(size)])
    G2P_R_RIZ = np.empty_like(G2P_R)
    for i in range(size):
        G2P_R_RIZ[i] = G2P_R[i].Create_resized(0, base_size)
        MPI.Datatype.Commit(G2P_R[i])
        MPI.Datatype.Commit(G2P_R_RIZ[i])

    # send type p2g
    # column type of split up in energy
    P2G_S = BASE_TYPE.Create_vector(count[0, rank], 1, data_shape[1])
    P2G_S_RIZ = P2G_S.Create_resized(0, base_size)
    MPI.Datatype.Commit(P2G_S)
    MPI.Datatype.Commit(P2G_S_RIZ)

    # receive types p2g
    # vector of size of #ranks
    # multi column data type for every rank size #nnz not divisible
    P2G_R = np.array([BASE_TYPE.Create_vector(count[1, rank], count[0, i], data_shape[0]) for i in range(size)])
    P2G_R_RIZ = np.empty_like(P2G_R)
    for i in range(size):
        P2G_R_RIZ[i] = P2G_R[i].Create_resized(0, base_size)
        MPI.Datatype.Commit(P2G_R[i])
        MPI.Datatype.Commit(P2G_R_RIZ[i])

    # define helper communication functions-------------------------------------
    # captures all variables from the outside (comm/count/disp/rank/size/types)

    def scatter_master(inp: npt.NDArray[np.complex128], outp: npt.NDArray[np.complex128], transpose_net: bool = False):
        if transpose_net:
            comm.Scatterv([inp, count[1, :], disp[1, :], COLUMN_RIZ], outp, root=0)
        else:
            if rank == 0:
                inp_transposed = np.copy(inp.T, order="C")
            else:
                inp_transposed = None
            comm.Scatterv([inp_transposed, count[1, :] * data_shape[0], disp[1, :] * data_shape[0], BASE_TYPE],
                          outp,
                          root=0)

    def gather_master(inp: npt.NDArray[np.complex128], outp: npt.NDArray[np.complex128], transpose_net: bool = False):
        if transpose_net:
            comm.Gatherv(inp, [outp, count[1, :], disp[1, :], COLUMN_RIZ], root=0)
        else:
            if rank == 0:
                out_transposed = np.copy(outp.T, order="C")
            else:
                out_transposed = None
            comm.Gatherv(inp, [out_transposed, count[1, :] * outp.shape[0], disp[1, :] * outp.shape[0], BASE_TYPE],
                         root=0)
            if rank == 0:
                outp[:, :] = out_transposed.T

    def alltoall_g2p(inp: npt.NDArray[np.complex128], outp: npt.NDArray[np.complex128], transpose_net: bool = False):
        if transpose_net:
            comm.Alltoallw([inp, count[0, :], disp[0, :] * base_size,
                            np.repeat(G2P_S_RIZ, size)],
                           [outp, np.repeat([1], size), disp[1, :] * base_size, G2P_R_RIZ])
        else:
            inp_transposed = np.copy(inp.T, order="C")
            comm.Alltoallw([
                inp_transposed, count[0, :] * count[1, rank], disp[0, :] * count[1, rank] * base_size,
                np.repeat(BASE_TYPE, size)
            ], [outp, np.repeat([1], size), disp[1, :] * base_size, G2P_R_RIZ])

    def alltoall_p2g(inp: npt.NDArray[np.complex128], outp: npt.NDArray[np.complex128], transpose_net: bool = False):
        if transpose_net:
            comm.Alltoallw([inp, count[1, :], disp[1, :] * base_size,
                            np.repeat(P2G_S_RIZ, size)],
                           [outp, np.repeat([1], size), disp[0, :] * base_size, P2G_R_RIZ])
        else:
            inp_transposed = np.copy(inp.T, order="C")
            comm.Alltoallw([
                inp_transposed, count[1, :] * count[0, rank], disp[1, :] * count[0, rank] * base_size,
                np.repeat(BASE_TYPE, size)
            ], [outp, np.repeat([1], size), disp[0, :] * base_size, P2G_R_RIZ])

    # initialize self energy----------------------------------------------------
    sg_h2g = np.zeros((count[1, rank], no), dtype=np.complex128)
    sl_h2g = np.zeros((count[1, rank], no), dtype=np.complex128)
    sr_h2g = np.zeros((count[1, rank], no), dtype=np.complex128)

    # phonon self energy. Only diagonal so far----------------------------------
    sg_phn = np.zeros((count[1,rank], nao), dtype=np.complex128)
    sl_phn = np.zeros((count[1,rank], nao), dtype=np.complex128)
    sr_phn = np.zeros((count[1,rank], nao), dtype=np.complex128)

    # # Transform the hamiltonian to a block tri-diagonal format
    # if args.type in ("gpu"):
    #     nb = hamiltonian_obj.Bmin.shape[0]
    #     lb = np.max(hamiltonian_obj.Bmax - hamiltonian_obj.Bmin + 1)
    #     ne_loc = energy_loc.shape[0]
    #     blocked_hamiltonian_diag = np.zeros((nb, ne_loc, lb, lb), dtype=np.complex128)
    #     blocked_hamiltonian_upper = np.zeros((nb-1, ne_loc, lb, lb), dtype=np.complex128)
    #     blocked_hamiltonian_lower = np.zeros((nb-1, ne_loc, lb, lb), dtype=np.complex128)
    #     change_format.sparse2block_energyhamgen_no_map(hamiltonian_obj.Hamiltonian['H_4'], hamiltonian_obj.Overlap['H_4'], blocked_hamiltonian_diag, blocked_hamiltonian_upper, blocked_hamiltonian_lower, bmax, bmin, energy_loc)


    # initialize Green's function------------------------------------------------
    gg_h2g = cpx.zeros_pinned((count[1, rank], no), dtype=np.complex128)
    gl_h2g = cpx.zeros_pinned((count[1, rank], no), dtype=np.complex128)
    gr_h2g = cpx.zeros_pinned((count[1, rank], no), dtype=np.complex128)

    # initialize Screened interaction-------------------------------------------
    wg_p2w = cpx.zeros_pinned((count[1, rank], no), dtype=np.complex128)
    wl_p2w = cpx.zeros_pinned((count[1, rank], no), dtype=np.complex128)
    wr_p2w = cpx.zeros_pinned((count[1, rank], no), dtype=np.complex128)

    # initialize memory factors for Self-Energy, Green's Function and Screened interaction
    mem_s = 0.75
    mem_g = 0.0
    mem_w = 0.0

    # initialize the index of the lowest conduction band of the contact band structure
    ind_ek = -1
    # max number of iterations

    max_iter = 3
    ECmin_vec = np.concatenate((np.array([ECmin]), np.zeros(max_iter)))
    EFL_vec = np.concatenate((np.array([energy_fl]), np.zeros(max_iter)))
    EFR_vec = np.concatenate((np.array([energy_fr]), np.zeros(max_iter)))

    #Start and end index of the energy range
    ne_s = 0
    ne_f = 251

    # Preprocess
    DH = hamiltonian_obj
    from quatrex.block_tri_solvers import rgf_GF_GPU_combo

    bmin = DH.Bmin.copy()
    bmax = DH.Bmax.copy()

    mapping_diag = rgf_GF_GPU_combo.map_to_mapping(map_diag, nb)
    mapping_upper = rgf_GF_GPU_combo.map_to_mapping(map_upper, nb-1)
    mapping_lower = rgf_GF_GPU_combo.map_to_mapping(map_lower, nb-1)
    mapping_diag_dev = cp.asarray(mapping_diag)
    mapping_upper_dev = cp.asarray(mapping_upper)
    mapping_lower_dev = cp.asarray(mapping_lower)

    hamiltonian_diag, hamiltonian_upper, hamiltonian_lower = rgf_GF_GPU_combo.csr_to_block_tridiagonal_csr(DH.Hamiltonian['H_4'], bmin - 1, bmax)
    overlap_diag, overlap_upper, overlap_lower = rgf_GF_GPU_combo.csr_to_block_tridiagonal_csr(DH.Overlap['H_4'], bmin - 1, bmax)

    if rank == 0:
        time_start = -time.perf_counter()
    # output folder
    folder = '/quatrex/results/InAs_biased_sc/'
    for iter_num in range(max_iter):

        # initialize observables----------------------------------------------------
        # density of states
        dos = cpx.zeros_pinned(shape=(ne, nb), dtype=np.complex128)
        dosw = cpx.zeros_pinned(shape=(ne, nb // nbc), dtype=np.complex128)

        # occupied states/unoccupied states
        nE = cpx.zeros_pinned(shape=(ne, nb), dtype=np.complex128)
        nP = cpx.zeros_pinned(shape=(ne, nb), dtype=np.complex128)

        # occupied screening/unoccupied screening
        nEw = cpx.zeros_pinned(shape=(ne, nb // nbc), dtype=np.complex128)
        nPw = cpx.zeros_pinned(shape=(ne, nb // nbc), dtype=np.complex128)

        # current per energy
        ide = cpx.zeros_pinned(shape=(ne, nb), dtype=np.complex128)

        # # transform from 2D format to list/vector of sparse arrays format-----------
        # sg_h2g_vec = change_format.sparse2vecsparse_v2(sg_h2g, rows, columns, nao)
        # sl_h2g_vec = change_format.sparse2vecsparse_v2(sl_h2g, rows, columns, nao)
        # sr_h2g_vec = change_format.sparse2vecsparse_v2(sr_h2g, rows, columns, nao)

       
        # # transform from 2D format to list/vector of sparse arrays format-----------
        # sg_ephn_h2g_vec = change_format.sparse2vecsparse_v2(sg_phn, np.arange(nao), np.arange(nao), nao)
        # sl_ephn_h2g_vec = change_format.sparse2vecsparse_v2(sl_phn, np.arange(nao), np.arange(nao), nao)
        # sr_ephn_h2g_vec = change_format.sparse2vecsparse_v2(sr_phn, np.arange(nao), np.arange(nao), nao)

        comm.Barrier()
        start_symmetrization = time.perf_counter()

        sl_rgf_dev = cp.asarray(sl_h2g)
        sg_rgf_dev = cp.asarray(sg_h2g)
        sr_rgf_dev = cp.asarray(sr_h2g)
        sl_phn_dev = cp.asarray(sl_phn)
        sg_phn_dev = cp.asarray(sg_phn)
        sr_phn_dev = cp.asarray(sr_phn)
        rgf_GF_GPU_combo.self_energy_preprocess_2d(sl_rgf_dev, sg_rgf_dev, sr_rgf_dev, sl_phn_dev, sg_phn_dev, sr_phn_dev, cp.asarray(rows), cp.asarray(columns), cp.asarray(ij2ji))
        sr_rgf = cp.asnumpy(sr_rgf_dev)

        comm.Barrier()
        finish_symmetrization = time.perf_counter()
        if rank == 0:
            print(f"Symmetrization time: {finish_symmetrization - start_symmetrization}", flush = True)
        
        start_band_edge = time.perf_counter()
    
        # Adjusting Fermi Levels of both contacts to the current iteration band minima
        (ECmin_vec[iter_num + 1], ind_ek) = get_band_edge_mpi_interpol_2(ECmin_vec[iter_num],
                                                    energy,
                                                    hamiltonian_obj.Overlap['H_4'],
                                                    hamiltonian_obj.Hamiltonian['H_4'],
                                                    sr_rgf,
                                                    ind_ek,
                                                    bmin,
                                                    bmax,
                                                    comm,
                                                    rank,
                                                    size,
                                                    count,
                                                    disp,
                                                    'left',
                                                    mapping_diag, mapping_upper, mapping_lower, ij2ji)
        
        comm.Barrier()
        finish_band_edge = time.perf_counter()
        if rank == 0:
            print(f"Band edge time: {finish_band_edge - start_band_edge}", flush = True)
        
        # # Adjusting Fermi Levels of both contacts to the current iteration band minima
        # (ECmin_vec[iter_num + 1], ind_ek) = get_band_edge_mpi_interpol(ECmin_vec[iter_num],
        #                                             energy,
        #                                             hamiltonian_obj.Overlap['H_4'],
        #                                             hamiltonian_obj.Hamiltonian['H_4'],
        #                                             sr_h2g_vec,
        #                                             sl_h2g_vec,
        #                                             sg_h2g_vec,
        #                                             sr_ephn_h2g_vec,
        #                                             ind_ek,
        #                                             rows,
        #                                             columns,
        #                                             bmin,
        #                                             bmax,
        #                                             comm,
        #                                             rank,
        #                                             size,
        #                                             count,
        #                                             disp,
        #                                             side='left')
        
        if rank == 0:
            print(f"ECmin: {ECmin_vec[iter_num + 1]}", flush = True)
        
        energy_fl = ECmin_vec[iter_num + 1] + dEfL_EC
        energy_fr = ECmin_vec[iter_num + 1] + dEfR_EC

        EFL_vec[iter_num + 1] = energy_fl
        EFR_vec[iter_num + 1] = energy_fr
        
        # calculate the green's function at every rank------------------------------
        if args.type in ("cpu"):
            gr_diag, gr_upper, gl_diag, gl_upper, gg_diag, gg_upper, sigRBl, sigRBr = calc_GF_pool.calc_GF_pool_mpi_split(
                hamiltonian_obj,
                energy_loc,
                sr_h2g_vec,
                sl_h2g_vec,
                sg_h2g_vec,
                sr_ephn_h2g_vec,
                sl_ephn_h2g_vec,
                sg_ephn_h2g_vec,
                energy_fl,
                energy_fr,
                temp,
                dos[disp[1, rank]:disp[1, rank] + count[1, rank]],
                nE[disp[1, rank]:disp[1, rank] + count[1, rank]],
                nP[disp[1, rank]:disp[1, rank] + count[1, rank]],
                ide[disp[1, rank]:disp[1, rank] + count[1, rank]],
                factor_g_loc,
                comm,
                rank,
                size,
                homogenize=False,
                return_sigma_boundary=True,
                NCpSC=NCpSC,
                mkl_threads=gf_mkl_threads,
                worker_num=gf_worker_threads)
        elif args.type in ("gpu"):
            calc_GF_pool_GPU_memopt_2.calc_GF_pool_mpi_split_memopt(
                hamiltonian_obj,
                hamiltonian_diag, hamiltonian_upper, hamiltonian_lower,
                overlap_diag, overlap_upper, overlap_lower,
                energy_loc,
                sr_rgf_dev,
                sl_rgf_dev,
                sg_rgf_dev,
                gr_h2g,
                gl_h2g,
                gg_h2g,
                mapping_diag_dev,
                mapping_upper_dev,
                mapping_lower_dev,
                # rows,
                # columns,
                # ij2ji,
                energy_fl,
                energy_fr,
                temp,
                dos[disp[1, rank]:disp[1, rank] + count[1, rank]],
                nE[disp[1, rank]:disp[1, rank] + count[1, rank]],
                nP[disp[1, rank]:disp[1, rank] + count[1, rank]],
                ide[disp[1, rank]:disp[1, rank] + count[1, rank]],
                factor_g_loc,
                comm,
                rank,
                size,
                homogenize=False,
                NCpSC=NCpSC,
                mkl_threads=gf_mkl_threads,
                worker_num=gf_worker_threads)

        # lower diagonal blocks from physics identity
        #gg_lower = -gg_upper.conjugate().transpose((0, 1, 3, 2))
        #gl_lower = -gl_upper.conjugate().transpose((0, 1, 3, 2))
        #gr_lower = gr_upper.transpose((0, 1, 3, 2))
        # if iter_num == 0:
        #     gg_h2g = change_format.block2sparse_energy_alt(map_diag,
        #                                                    map_upper,
        #                                                    map_lower,
        #                                                    gg_diag,
        #                                                    gg_upper,
        #                                                    gg_lower,
        #                                                    no,
        #                                                    count[1, rank],
        #                                                    energy_contiguous=False)
        #     gl_h2g = change_format.block2sparse_energy_alt(map_diag,
        #                                                    map_upper,
        #                                                    map_lower,
        #                                                    gl_diag,
        #                                                    gl_upper,
        #                                                    gl_lower,
        #                                                    no,
        #                                                    count[1, rank],
        #                                                    energy_contiguous=False)
        #     gr_h2g = change_format.block2sparse_energy_alt(map_diag,
        #                                                    map_upper,
        #                                                    map_lower,
        #                                                    gr_diag,
        #                                                    gr_upper,
        #                                                    gr_lower,
        #                                                    no,
        #                                                    count[1, rank],
        #                                                    energy_contiguous=False)
        # else:
        #     # add new contribution to the Green's function
        #     gg_h2g = (1.0 - mem_g) * change_format.block2sparse_energy_alt(map_diag,
        #                                                                    map_upper,
        #                                                                    map_lower,
        #                                                                    gg_diag,
        #                                                                    gg_upper,
        #                                                                    gg_lower,
        #                                                                    no,
        #                                                                    count[1, rank],
        #                                                                    energy_contiguous=False) + mem_g * gg_h2g
        #     gl_h2g = (1.0 - mem_g) * change_format.block2sparse_energy_alt(map_diag,
        #                                                                    map_upper,
        #                                                                    map_lower,
        #                                                                    gl_diag,
        #                                                                    gl_upper,
        #                                                                    gl_lower,
        #                                                                    no,
        #                                                                    count[1, rank],
        #                                                                    energy_contiguous=False) + mem_g * gl_h2g
        #     gr_h2g = (1.0 - mem_g) * change_format.block2sparse_energy_alt(map_diag,
        #                                                                    map_upper,
        #                                                                    map_lower,
        #                                                                    gr_diag,
        #                                                                    gr_upper,
        #                                                                    gr_lower,
        #                                                                    no,
        #                                                                    count[1, rank],
        #                                                                    energy_contiguous=False) + mem_g * gr_h2g
        
        comm.Barrier()
        start_g2p_comm = time.perf_counter()
        
        # calculate the transposed
        gl_transposed_h2g = np.copy(gl_h2g[:, ij2ji], order="C")
        # create local buffers
        gg_g2p = np.empty((count[0, rank], data_shape[1]), dtype=np.complex128, order="C")
        gl_g2p = np.empty((count[0, rank], data_shape[1]), dtype=np.complex128, order="C")
        gr_g2p = np.empty((count[0, rank], data_shape[1]), dtype=np.complex128, order="C")
        gl_transposed_g2p = np.empty((count[0, rank], data_shape[1]), dtype=np.complex128, order="C")

        # use of all to all w since not divisible
        alltoall_g2p(gg_h2g, gg_g2p, transpose_net=args.net_transpose)
        alltoall_g2p(gl_h2g, gl_g2p, transpose_net=args.net_transpose)
        alltoall_g2p(gr_h2g, gr_g2p, transpose_net=args.net_transpose)
        alltoall_g2p(gl_transposed_h2g, gl_transposed_g2p, transpose_net=args.net_transpose)

        comm.Barrier()
        finish_g2p_comm = time.perf_counter()
        if rank == 0:
            print(f"G2P communication time: {finish_g2p_comm - start_g2p_comm}", flush = True)
            print("Green's function calculated", flush = True)
        
        start_p_computation = time.perf_counter()

        # calculate the polarization at every rank----------------------------------
        if args.type in ("cpu"):
            pg_g2p, pl_g2p = g2p_cpu.g2p_fft_mpi_cpu_inlined_nopr(
                                                pre_factor,
                                                gg_g2p,
                                                gl_g2p,
                                                gl_transposed_g2p)
        elif args.type in ("gpu"):
            pg_g2p, pl_g2p = g2p_gpu.g2p_fft_mpi_gpu_batched_nopr(
                                                pre_factor,
                                                gg_g2p,
                                                gl_g2p,
                                                gl_transposed_g2p)
        else: 
            raise ValueError("Argument error, input type not possible")
        
        comm.Barrier()
        finish_p_computation = time.perf_counter()
        if rank == 0:
            print(f"Polarization computation time: {finish_p_computation - start_p_computation}", flush = True)
        
        start_p2w_comm = time.perf_counter()

        # distribute polarization function according to p2w step--------------------

        # create local buffers
        pg_p2w = np.empty((count[1, rank], data_shape[0]), dtype=np.complex128, order="C")
        pl_p2w = np.empty((count[1, rank], data_shape[0]), dtype=np.complex128, order="C")
        pr_p2w = np.zeros((count[1, rank], data_shape[0]), dtype=np.complex128, order="C")

        # use of all to all w since not divisible
        alltoall_p2g(pg_g2p, pg_p2w, transpose_net=args.net_transpose)
        alltoall_p2g(pl_g2p, pl_p2w, transpose_net=args.net_transpose)
        alltoall_p2g(pl_g2p, pr_p2w, transpose_net=args.net_transpose)

        # # transform from 2D format to list/vector of sparse arrays format-----------
        # pg_p2w_vec = change_format.sparse2vecsparse_v2(pg_p2w, rows, columns, nao)
        # pl_p2w_vec = change_format.sparse2vecsparse_v2(pl_p2w, rows, columns, nao)
        # pr_p2w_vec = change_format.sparse2vecsparse_v2(pr_p2w, rows, columns, nao)
        pg_p2w_vec = None
        pl_p2w_vec = None
        pr_p2w_vec = None

        comm.Barrier()
        finish_p2w_comm = time.perf_counter()
        if rank == 0:
            print(f"P2W communication time: {finish_p2w_comm - start_p2w_comm}", flush = True)
            print("Polarization calculated", flush = True)

        # calculate the screened interaction on every rank--------------------------
        if args.type in ("cpu"):
            wg_diag, wg_upper, wl_diag, wl_upper, wr_diag, wr_upper, nb_mm, lb_max_mm, ind_zeros = p2w_cpu.p2w_pool_mpi_cpu(
                hamiltonian_obj,
                energy_loc,
                pg_p2w_vec,
                pl_p2w_vec,
                pr_p2w_vec,
                vh,
                dosw[disp[1, rank]:disp[1, rank] + count[1, rank]],
                nEw[disp[1, rank]:disp[1, rank] + count[1, rank]],
                nPw[disp[1, rank]:disp[1, rank] + count[1, rank]],
                Idx_e_loc,
                factor_w_loc,
                comm,
                rank,
                size,
                nbc,
                homogenize=False,
                NCpSC=NCpSC,
                mkl_threads=w_mkl_threads,
                worker_num=w_worker_threads)
        elif args.type in ("gpu"):
            # wg_diag, wg_upper, wl_diag, wl_upper, wr_diag, wr_upper, nb_mm, lb_max_mm, ind_zeros = p2w_gpu.p2w_pool_mpi_gpu_split(
            #     hamiltonian_obj, energy_loc, pg_p2w_vec, pl_p2w_vec, pr_p2w_vec, vh, map_diag_mm,
            #     map_upper_mm, map_lower_mm, rows, columns, ij2ji,
            #     dosw[disp[1, rank]:disp[1, rank] + count[1, rank]], nEw[disp[1, rank]:disp[1, rank] + count[1, rank]],
            #     nPw[disp[1, rank]:disp[1, rank] + count[1, rank]], Idx_e_loc, factor_w_loc, comm, rank, size, nbc, homogenize=False, NCpSC=NCpSC,
            #     mkl_threads = w_mkl_threads, worker_num = w_worker_threads, compute_mode = 1)
            p2w_gpu_improved_2.calc_W_pool_mpi_split(
                hamiltonian_obj,
                # Energy vector.
                energy_loc,
                # Polarization.
                pg_p2w_vec,
                pl_p2w_vec,
                pr_p2w_vec,
                # polarization 2D format.
                pg_p2w,
                pl_p2w,
                pr_p2w,
                # Coulomb matrix.
                vh,
                # Output Green's functions.
                wg_p2w,
                wl_p2w,
                wr_p2w,
                # Sparse-to-dense Mappings.
                map_diag_mm,
                map_upper_mm,
                map_lower_mm,
                map_diag_m,
                map_upper_m,
                map_lower_m,
                map_diag_l,
                map_upper_l,
                map_lower_l,
                # P indices
                rows,
                columns,
                # P transposition indices.
                ij2ji,
                # M and L indices.
                rows_m,
                columns_m,
                rows_l,
                columns_l,
                # M and L transposition indeices.
                ij2ji_m,
                ij2ji_l,
                # Output observables.
                dosw[disp[1, rank] : disp[1, rank] + count[1, rank]],
                nEw[disp[1, rank] : disp[1, rank] + count[1, rank]],
                nPw[disp[1, rank] : disp[1, rank] + count[1, rank]],
                Idx_e_loc,
                # Some factor, idk.
                factor_w_loc,
                # MPI communicator info.
                comm,
                rank,
                size,
                # Number of connected blocks.
                nbc,
                # Options.
                homogenize=False,
                NCpSC=NCpSC,
                mkl_threads=w_mkl_threads,
                worker_num=w_worker_threads,
                compute_mode = 1
            )

        # transform from block format to 2D format-----------------------------------
        # lower diagonal blocks from physics identity
        # wg_lower = -wg_upper.conjugate().transpose((0, 1, 3, 2))
        # wl_lower = -wl_upper.conjugate().transpose((0, 1, 3, 2))
        # wr_lower = wr_upper.transpose((0, 1, 3, 2))

        # memory_mask = np.ones(energy_loc.shape[0], dtype=bool)
        # memory_mask[ind_zeros] = False
        # # add new contribution to the Screened interaction
        # wg_p2w[memory_mask] = (1.0 - mem_w) * change_format.block2sparse_energy_alt(
        #     map_diag_mm,
        #     map_upper_mm,
        #     map_lower_mm,
        #     wg_diag,
        #     wg_upper,
        #     wg_lower,
        #     no,
        #     count[1, rank],
        #     energy_contiguous=False)[memory_mask] + mem_w * wg_p2w[memory_mask]
        # wg_p2w[ind_zeros, :] = 0.0 + 0.0j
        # wl_p2w[memory_mask] = (1.0 - mem_w) * change_format.block2sparse_energy_alt(
        #     map_diag_mm,
        #     map_upper_mm,
        #     map_lower_mm,
        #     wl_diag,
        #     wl_upper,
        #     wl_lower,
        #     no,
        #     count[1, rank],
        #     energy_contiguous=False)[memory_mask] + mem_w * wl_p2w[memory_mask]
        # wl_p2w[ind_zeros, :] = 0.0 + 0.0j
        # wr_p2w[memory_mask] = (1.0 - mem_w) * change_format.block2sparse_energy_alt(
        #     map_diag_mm,
        #     map_upper_mm,
        #     map_lower_mm,
        #     wr_diag,
        #     wr_upper,
        #     wr_lower,
        #     no,
        #     count[1, rank],
        #     energy_contiguous=False)[memory_mask] + mem_w * wr_p2w[memory_mask]

        # distribute screened interaction according to gw2s step--------------------

        comm.Barrier()
        start_w2s_comm = time.perf_counter()
        
        # calculate the transposed
        wg_transposed_p2w = np.copy(wg_p2w[:, ij2ji], order="C")
        wl_transposed_p2w = np.copy(wl_p2w[:, ij2ji], order="C")

        # create local buffers
        wg_gw2s = np.empty((count[0, rank], data_shape[1]), dtype=np.complex128, order="C")
        wl_gw2s = np.empty((count[0, rank], data_shape[1]), dtype=np.complex128, order="C")
        wr_gw2s = np.empty((count[0, rank], data_shape[1]), dtype=np.complex128, order="C")
        wg_transposed_gw2s = np.empty((count[0, rank], data_shape[1]), dtype=np.complex128, order="C")
        wl_transposed_gw2s = np.empty((count[0, rank], data_shape[1]), dtype=np.complex128, order="C")

        # use of all to all w since not divisible
        alltoall_g2p(wg_p2w, wg_gw2s, transpose_net=args.net_transpose)
        alltoall_g2p(wl_p2w, wl_gw2s, transpose_net=args.net_transpose)
        alltoall_g2p(wr_p2w, wr_gw2s, transpose_net=args.net_transpose)
        alltoall_g2p(wg_transposed_p2w, wg_transposed_gw2s, transpose_net=args.net_transpose)
        alltoall_g2p(wl_transposed_p2w, wl_transposed_gw2s, transpose_net=args.net_transpose)

        comm.Barrier()
        finish_w2s_comm = time.perf_counter()
        if rank == 0:
            print(f"W2S communication time: {finish_w2s_comm - start_w2s_comm}", flush = True)
            print("Finish p2w", flush=True)
        
        start_s_computation = time.perf_counter()

        # tod optimize and not load two time green's function to gpu and do twice the fft
        if args.type in ("cpu"):
            sg_gw2s, sl_gw2s, sr_gw2s = gw2s_cpu.gw2s_fft_mpi_cpu_PI_sr(-pre_factor / 2, gg_g2p, gl_g2p,
                                                                           wg_gw2s, wl_gw2s,
                                                                           wg_transposed_gw2s, wl_transposed_gw2s, vh1d, energy, rank, disp, count)
            
        elif args.type in ("gpu"):
            sg_gw2s, sl_gw2s, sr_gw2s = gw2s_gpu.gw2s_fft_mpi_gpu_PI_sr_batched(-pre_factor / 2, gg_g2p, gl_g2p,
                                                                           wg_gw2s, wl_gw2s,
                                                                           wg_transposed_gw2s, wl_transposed_gw2s, vh1d, energy, rank, disp, count)
        else:
            raise ValueError("Argument error, input type not possible")
        
        comm.Barrier()
        finish_s_computation = time.perf_counter()
        if rank == 0:
            print(f"Sigma computation time: {finish_s_computation - start_s_computation}", flush = True)

        start_s2g_comm = time.perf_counter()

        # distribute screened interaction according to h2g step---------------------
        # create local buffers
        sg_h2g_buf = np.empty((count[1, rank], data_shape[0]), dtype=np.complex128, order="C")
        sl_h2g_buf = np.empty((count[1, rank], data_shape[0]), dtype=np.complex128, order="C")
        sr_h2g_buf = np.empty((count[1, rank], data_shape[0]), dtype=np.complex128, order="C")

        # use of all to all w since not divisible
        alltoall_p2g(sg_gw2s, sg_h2g_buf, transpose_net=args.net_transpose)
        alltoall_p2g(sl_gw2s, sl_h2g_buf, transpose_net=args.net_transpose)
        alltoall_p2g(sr_gw2s, sr_h2g_buf, transpose_net=args.net_transpose)

        comm.Barrier()
        finish_s2g_comm = time.perf_counter()
        if rank == 0:
            print(f"S2G communication time: {finish_s2g_comm - start_s2g_comm}", flush = True)

        start_sigma_mem_update = time.perf_counter()

        if iter_num == 0:
            sg_h2g = (1.0 - mem_s) * sg_h2g_buf + mem_s * sg_h2g
            sl_h2g = (1.0 - mem_s) * sl_h2g_buf + mem_s * sl_h2g
            sr_h2g = (1.0 - mem_s) * sr_h2g_buf + mem_s * sr_h2g
        else:
            # add new contribution to the Self-Energy
            sg_h2g = (1.0 - mem_s) * sg_h2g_buf + mem_s * sg_h2g
            sl_h2g = (1.0 - mem_s) * sl_h2g_buf + mem_s * sl_h2g
            sr_h2g = (1.0 - mem_s) * sr_h2g_buf + mem_s * sr_h2g

        comm.Barrier()
        finish_sigma_mem_update = time.perf_counter()
        if rank == 0:
            print(f"Sigma memory update time: {finish_sigma_mem_update - start_sigma_mem_update}", flush = True)
        
        start_sephn = time.perf_counter()

        # Extract diagonal bands
        gg_diag_band = gg_h2g[:, rows == columns]
        gl_diag_band = gl_h2g[:, rows == columns]
        # Add imaginary self energy to broaden peaks (motivated by a zero energy phonon interaction)
        # The Phonon energy (EPHN) is set to zero and the phonon-electron potential (DPHN) is set to 2.5e-3
        # at the beginning of this script. Only diagonal part now!
        sg_phn, sl_phn, sr_phn = electron_phonon_selfenergy.calc_SE_GF_EPHN(energy_loc,
                                                                            gl_diag_band,
                                                                            gg_diag_band,
                                                                            sg_phn,
                                                                            sl_phn,
                                                                            sr_phn,
                                                                            EPHN,
                                                                            DPHN,
                                                                            temp,
                                                                            mem_s)
        
        comm.Barrier()
        finish_sephn = time.perf_counter()
        if rank == 0:
            print(f"SEPHN computation time: {finish_sephn - start_sephn}", flush = True)

        start_observables = time.perf_counter()

        # if iter_num == max_iter - 1:
        #     alltoall_p2g(sg_gw2s, sg_h2g, transpose_net=args.net_transpose)
        #     alltoall_p2g(sl_gw2s, sl_h2g, transpose_net=args.net_transpose)
        #     alltoall_p2g(sr_gw2s, sr_h2g, transpose_net=args.net_transpose)

        # Wrapping up the iteration
        if rank == 0:
            comm.Reduce(MPI.IN_PLACE, dos, op=MPI.SUM, root=0)
            comm.Reduce(MPI.IN_PLACE, ide, op=MPI.SUM, root=0)

        else:
            comm.Reduce(dos, None, op=MPI.SUM, root=0)
            comm.Reduce(ide, None, op=MPI.SUM, root=0)
        
        comm.Barrier()
        finish_observables = time.perf_counter()
        if rank == 0:
            print(f"Observables reduction time: {finish_observables - start_observables}", flush = True)

        # if rank == 0:
        # np.savetxt(parent_path + folder + 'E.dat', energy)
        # np.savetxt(parent_path + folder + 'DOS_' + str(iter_num) + '.dat', dos.view(float))
        # np.savetxt(parent_path + folder + 'IDE_' + str(iter_num) + '.dat', ide.view(float))
    # if rank == 0:
    # np.savetxt(parent_path + folder + 'EFL.dat', EFL_vec)
    # np.savetxt(parent_path + folder + 'EFR.dat', EFR_vec)
    if rank == 0:
        time_start += time.perf_counter()
        print("Finish iteration", flush=True)
        print(f"Time: {time_start:.2f} s")
        # create buffers at master
        gg_mpi = np.empty_like(gg_gold)
        gl_mpi = np.empty_like(gg_gold)
        gr_mpi = np.empty_like(gg_gold)
        pg_mpi = np.empty_like(gg_gold)
        pl_mpi = np.empty_like(gg_gold)
        pr_mpi = np.empty_like(gg_gold)
        wg_mpi = np.empty_like(gg_gold)
        wl_mpi = np.empty_like(gg_gold)
        wr_mpi = np.empty_like(gg_gold)
        sg_mpi = np.empty_like(gg_gold)
        sl_mpi = np.empty_like(gg_gold)
        sr_mpi = np.empty_like(gg_gold)
        sphg_mpi = np.empty((nao, data_shape[1]), dtype=np.complex128)
        sphl_mpi = np.empty((nao, data_shape[1]), dtype=np.complex128)
        sphr_mpi = np.empty((nao, data_shape[1]), dtype=np.complex128)

        gather_master(gg_h2g, gg_mpi, transpose_net=args.net_transpose)
        gather_master(gl_h2g, gl_mpi, transpose_net=args.net_transpose)
        gather_master(gr_h2g, gr_mpi, transpose_net=args.net_transpose)
        gather_master(pg_p2w, pg_mpi, transpose_net=args.net_transpose)
        gather_master(pl_p2w, pl_mpi, transpose_net=args.net_transpose)
        gather_master(pr_p2w, pr_mpi, transpose_net=args.net_transpose)
        gather_master(wg_p2w, wg_mpi, transpose_net=args.net_transpose)
        gather_master(wl_p2w, wl_mpi, transpose_net=args.net_transpose)
        gather_master(wr_p2w, wr_mpi, transpose_net=args.net_transpose)
        gather_master(sg_h2g, sg_mpi, transpose_net=args.net_transpose)
        gather_master(sl_h2g, sl_mpi, transpose_net=args.net_transpose)
        gather_master(sr_h2g, sr_mpi, transpose_net=args.net_transpose)

        gather_master(sg_phn, sphg_mpi, transpose_net=args.net_transpose)
        gather_master(sl_phn, sphl_mpi, transpose_net=args.net_transpose)
        gather_master(sr_phn, sphr_mpi, transpose_net=args.net_transpose)
    else:
        # send time to master
        dummy_array = np.empty((nao, data_shape[1]), dtype=np.complex128)
        gather_master(gg_h2g, dummy_array, transpose_net=args.net_transpose)
        gather_master(gl_h2g, dummy_array, transpose_net=args.net_transpose)
        gather_master(gr_h2g, dummy_array, transpose_net=args.net_transpose)
        gather_master(pg_p2w, dummy_array, transpose_net=args.net_transpose)
        gather_master(pl_p2w, dummy_array, transpose_net=args.net_transpose)
        gather_master(pr_p2w, dummy_array, transpose_net=args.net_transpose)
        gather_master(wg_p2w, dummy_array, transpose_net=args.net_transpose)
        gather_master(wl_p2w, dummy_array, transpose_net=args.net_transpose)
        gather_master(wr_p2w, dummy_array, transpose_net=args.net_transpose)
        gather_master(sg_h2g, dummy_array, transpose_net=args.net_transpose)
        gather_master(sl_h2g, dummy_array, transpose_net=args.net_transpose)
        gather_master(sr_h2g, dummy_array, transpose_net=args.net_transpose)
        gather_master(sg_phn, dummy_array, transpose_net=args.net_transpose)
        gather_master(sl_phn, dummy_array, transpose_net=args.net_transpose)
        gather_master(sr_phn, dummy_array, transpose_net=args.net_transpose)

    # test against gold solution------------------------------------------------

    if rank == 0:
        # print difference to given solution
        # use Frobenius norm
        diff_gg = np.linalg.norm(gg_gold - gg_mpi)
        diff_gl = np.linalg.norm(gl_gold - gl_mpi)
        #diff_gr = np.linalg.norm(gr_gold - gr_mpi)
        diff_pg = np.linalg.norm(pg_gold - pg_mpi)
        diff_pl = np.linalg.norm(pl_gold - pl_mpi)
        #diff_pr = np.linalg.norm(pr_gold - pr_mpi)
        diff_wg = np.linalg.norm(wg_gold - wg_mpi)
        diff_wl = np.linalg.norm(wl_gold - wl_mpi)
        #diff_wr = np.linalg.norm(wr_gold - wr_mpi)
        diff_sg = np.linalg.norm(sg_gold - sg_mpi)
        diff_sl = np.linalg.norm(sl_gold - sl_mpi)
        diff_sr = np.linalg.norm(sr_gold - sr_mpi)
        diff_sphg = np.linalg.norm(sphg_gold - sphg_mpi)
        diff_sphl = np.linalg.norm(sphl_gold - sphl_mpi)
        diff_sphr = np.linalg.norm(sphr_gold - sphr_mpi)
        print(f"Green's Function differences to Gold Solution g/l/r:  {diff_gg:.4f}, {diff_gl:.4f}")
        print(f"Polarization differences to Gold Solution g/l/r:  {diff_pg:.4f}, {diff_pl:.4f}")
        print(f"Screened interaction differences to Gold Solution g/l/r:  {diff_wg:.4f}, {diff_wl:.4f}")
        print(f"Screened self-energy differences to Gold Solution g/l/r:  {diff_sg:.4f}, {diff_sl:.4f}, {diff_sr:.4f}")
        print(f"E-PH self-energy differences to Gold Solution g/l/r:  {diff_sphg:.4f}, {diff_sphl:.4f}, {diff_sphr:.4f}")

        # assert solution close to real solution
        abstol = 1e-2
        reltol = 1e-1
        assert diff_gg <= abstol + reltol * np.max(np.abs(gg_gold))
        assert diff_gl <= abstol + reltol * np.max(np.abs(gl_gold))
        #assert diff_gr <= abstol + reltol * np.max(np.abs(gr_gold))
        assert diff_pg <= abstol + reltol * np.max(np.abs(pg_gold))
        assert diff_pl <= abstol + reltol * np.max(np.abs(pl_gold))
        #assert diff_pr <= abstol + reltol * np.max(np.abs(pr_gold))
        assert diff_wg <= abstol + reltol * np.max(np.abs(wg_gold))
        assert diff_wl <= abstol + reltol * np.max(np.abs(wl_gold))
        #assert diff_wr <= abstol + reltol * np.max(np.abs(wr_gold))
        assert diff_sg <= abstol + reltol * np.max(np.abs(sg_gold))
        assert diff_sl <= abstol + reltol * np.max(np.abs(sl_gold))
        assert diff_sr <= abstol + reltol * np.max(np.abs(sr_gold))
        assert np.allclose(gg_gold, gg_mpi, atol=1e-5, rtol=1e-5)
        assert np.allclose(gl_gold, gl_mpi, atol=1e-5, rtol=1e-5)
        #assert np.allclose(gr_gold, gr_mpi, atol=1e-6, rtol=1e-6)
        assert np.allclose(pg_gold, pg_mpi, atol=1e-6, rtol=1e-6)
        assert np.allclose(pl_gold, pl_mpi, atol=1e-6, rtol=1e-6)
        #assert np.allclose(pr_gold, pr_mpi, atol=1e-6, rtol=1e-6)
        #assert np.allclose(wg_gold, wg_mpi, rtol=1e-6, atol=1e-6)
        assert np.allclose(wg_gold, wg_mpi, rtol=1e-3, atol=1e-3)
        assert np.allclose(wl_gold, wl_mpi, atol=1e-6, rtol=1e-6)
        #assert np.allclose(wr_gold, wr_mpi, atol=1e-6, rtol=1e-6)
        assert np.allclose(sg_gold, sg_mpi, atol=1e-5, rtol=1e-5)
        assert np.allclose(sl_gold, sl_mpi, atol=1e-5, rtol=1e-5)
        assert np.allclose(sr_gold, sr_mpi, atol=1e-6, rtol=1e-6)
        assert np.allclose(sphg_gold, sphg_mpi, atol=1e-5, rtol=1e-5)
        assert np.allclose(sphl_gold, sphl_mpi, atol=1e-5, rtol=1e-5)
        assert np.allclose(sphr_gold, sphr_mpi, atol=1e-5, rtol=1e-5)
        print("The mpi implementation is correct")

    # free datatypes------------------------------------------------------------

    MPI.Datatype.Free(COLUMN_RIZ)
    MPI.Datatype.Free(COLUMN)
    MPI.Datatype.Free(ROW_RIZ)
    MPI.Datatype.Free(ROW)
    MPI.Datatype.Free(G2P_S_RIZ)
    MPI.Datatype.Free(G2P_S)
    MPI.Datatype.Free(P2G_S_RIZ)
    MPI.Datatype.Free(P2G_S)
    for i in range(size):
        MPI.Datatype.Free(G2P_R_RIZ[i])
        MPI.Datatype.Free(G2P_R[i])
        MPI.Datatype.Free(P2G_R_RIZ[i])
        MPI.Datatype.Free(P2G_R[i])

    # finalize
    MPI.Finalize()

    # if rank == 0:
    #     time_start += time.perf_counter()
    #     print(f"Time: {time_start:.2f} s")
