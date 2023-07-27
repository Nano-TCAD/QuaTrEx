# Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.
"""
Example a sc-GW iteration with MPI+CUDA.
With transposition through network.
Applied to a (8-0)-CNT
See the different GW step folders for more explanations.
"""
import sys
import numpy as np
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

from quatrex.bandstructure.calc_band_edge import get_band_edge
from quatrex.GW.polarization.kernel import g2p_cpu
from quatrex.GW.selfenergy.kernel import gw2s_cpu
from quatrex.GW.gold_solution import read_solution
from quatrex.GW.screenedinteraction.kernel import p2w_cpu
from quatrex.GW.coulomb_matrix.read_coulomb_matrix import load_V, load_V_mpi
from quatrex.GreensFunction import calc_GF_pool
from quatrex.OMEN_structure_matrices import OMENHamClass
from quatrex.utils import change_format
from quatrex.utils import utils_gpu

if utils_gpu.gpu_avail():
    from quatrex.GW.polarization.kernel import g2p_gpu
    from quatrex.GW.selfenergy.kernel import gw2s_gpu

if __name__ == "__main__":
    MPI.Init_thread(required=MPI.THREAD_FUNNELED)
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    name = MPI.Get_processor_name()

    # assume every rank has enough memory to read the initial data
    # path to solution
    solution_path = "/usr/scratch/mont-fort17/dleonard/CNT/"
    solution_path_gw = os.path.join(solution_path, "data_GPWS_04.mat")
    solution_path_vh = os.path.join(solution_path, "data_Vh_4.mat")
    hamiltonian_path = os.path.join(solution_path, "CNT_newwannier")
    parser = argparse.ArgumentParser(description="Example of the first GW iteration with MPI+CUDA")
    parser.add_argument("-fvh", "--file_vh", default=solution_path_vh, required=False)
    parser.add_argument("-fpw", "--file_gw", default=solution_path_gw, required=False)
    parser.add_argument("-fhm", "--file_hm", default=hamiltonian_path, required=False)
    # change manually the used implementation inside the code
    parser.add_argument("-t", "--type", default="cpu", choices=["cpu", "gpu"], required=False)
    parser.add_argument("-nt", "--net_transpose", default=False, type=bool, required=False)
    parser.add_argument("-p", "--pool", default=True, type=bool, required=False)
    args = parser.parse_args()
    # check if gpu is available
    if args.type in ("gpu"):
        if not utils_gpu.gpu_avail():
            print("No gpu available")
            sys.exit(1)
    # print chosen implementation
    print(f"Using {args.type} implementation")

    # load greens function as input
    energy, rows, columns, gg_gold, gl_gold, gr_gold = read_solution.load_x(args.file_gw, "g")
    # load polarization to test against
    _, _, _, pg_gold, pl_gold, pr_gold = read_solution.load_x(args.file_gw, "p")
    # load screened interaction to test against
    _, _, _, wg_gold, wl_gold, wr_gold = read_solution.load_x(args.file_gw, "w")
    # load self-energy to test against
    _, _, _, sg_gold, sl_gold, sr_gold = read_solution.load_x(args.file_gw, "s")
    # load block sizes
    bmax, bmin = read_solution.load_B(args.file_gw)

    # create hamiltonian object
    # one orbital on C atoms, two same types
    no_orb = np.array([1, 1])

    hamiltonian_obj = OMENHamClass.Hamiltonian(args.file_hm, no_orb, rank)
    serial_ham = pickle.dumps(hamiltonian_obj)
    broadcasted_ham = comm.bcast(serial_ham, root=0)
    hamiltonian_obj = pickle.loads(broadcasted_ham)
    # Extract neighbor indices
    rows = hamiltonian_obj.rows
    columns = hamiltonian_obj.columns

    # hamiltonian object has 1-based indexing
    bmax = hamiltonian_obj.Bmax - 1
    bmin = hamiltonian_obj.Bmin - 1

    # Read Coulomb matrix and extract the nnz data of the coulomb matrix
    V_sparse = load_V_mpi(args.file_hm, rows, columns, comm, rank)

    vh = V_sparse

    ij2ji: npt.NDArray[np.int32] = change_format.find_idx_transposed(rows, columns)
    denergy: npt.NDArray[np.double] = energy[1] - energy[0]
    ne: np.int32 = np.int32(energy.shape[0])
    no: np.int32 = np.int32(columns.shape[0])
    pre_factor: np.complex128 = -1.0j * denergy / (np.pi)
    nao: np.int64 = np.max(bmax) + 1

    data_shape = np.array(gg_gold.shape)

    if rank == 0:
        # print size of data
        print(f"#Energy: {data_shape[1]} #nnz: {data_shape[0]}")

    # lesser green's function transposed
    gl_gold_t = np.copy(gl_gold[ij2ji, :], order="C")

    # computation parameters----------------------------------------------------
    # set number of threads for the p2w step
    w_mkl_threads = 2
    w_worker_threads = 12
    # set number of threads for the h2g step
    gf_mkl_threads = 1
    gf_worker_threads = 8
    # timing vector, 4 compute + 4 communication + 2 transform
    # order of timing order
    times_compute = np.empty((4), dtype=np.float64)
    times_communicate = np.empty((4), dtype=np.float64)
    times_transform_list = np.empty((2), dtype=np.float64)
    times_transform_block = np.empty((2), dtype=np.float64)

    # physical parameters-------------------------------------------------------

    # Fermi Level of Left Contact
    energy_fl = -3.85
    # Fermi Level of Right Contact
    energy_fr = -3.85
    # Temperature in Kelvin
    temp = 300
    # DFT Conduction Band Minimum
    ECmin = -3.55

    # Fermi Level to Band Edge Difference
    dEfL_EC = energy_fl - ECmin
    dEfR_EC = energy_fr - ECmin

    # creating mask for the energy range of the deleted W elements
    # given by the reference solution
    w_mask = np.ndarray(shape=(energy.shape[0], ), dtype=bool)

    wr_mask = np.sum(np.abs(wr_gold), axis=0) > 1e-10
    wl_mask = np.sum(np.abs(wl_gold), axis=0) > 1e-10
    wg_mask = np.sum(np.abs(wg_gold), axis=0) > 1e-10
    w_mask = np.logical_or(np.logical_or(wr_mask, wl_mask), wg_mask)

    # create the corresponding factor to mask
    # number of points to smooth the edges of the Green's Function
    dnp = 50
    factor_w = np.ones(ne)
    factor_w[ne - dnp - 1:ne] = (np.cos(np.pi * np.linspace(0, 1, dnp + 1)) + 1) / 2
    factor_w[np.where(np.invert(w_mask))[0]] = 0.0

    # create factor for the Green's Function
    factor_g = np.ones(ne)
    factor_g[ne - dnp - 1:ne] = (np.cos(np.pi * np.linspace(0, 1, dnp + 1)) + 1) / 2
    factor_g[0:dnp + 1] = (np.cos(np.pi * np.linspace(1, 0, dnp + 1)) + 1) / 2

    # number of blocks
    nb = hamiltonian_obj.Bmin.shape[0]
    # end and starting indexes of blocks
    # hamiltonian object has 1-based indexing
    bmax = hamiltonian_obj.Bmax - 1
    bmin = hamiltonian_obj.Bmin - 1
    # off diagonal blocks after matrix multiplication
    nbc = 2
    # end and starting indexes of off diagonal blocks after matrix multiplication
    bmax_mm = bmax[nbc - 1:nb:nbc]
    bmin_mm = bmin[0:nb:nbc]

    # create map from block format to 2D format
    map_diag, map_upper, map_lower = change_format.map_block2sparse_alt(rows, columns, bmax, bmin)
    # create map from block format to 2D format after matrix multiplication
    map_diag_mm, map_upper_mm, map_lower_mm = change_format.map_block2sparse_alt(rows, columns, bmax_mm, bmin_mm)

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
    # split up mask among ranks
    w_mask_loc = w_mask[disp[1, rank]:disp[1, rank] + count[1, rank]]
    # split up the factor between the ranks
    factor_w_loc = factor_w[disp[1, rank]:disp[1, rank] + count[1, rank]]
    factor_g_loc = factor_g[disp[1, rank]:disp[1, rank] + count[1, rank]]

    # print rank distribution
    print(f"Rank: {rank} #Energy/rank: {count[1,rank]} #nnz/rank: {count[0,rank]}", name)

    # adding checks
    assert energy_loc.size == count[1, rank]
    assert w_mask_loc.size == count[1, rank]

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
            comm.Gatherv(inp, [out_transposed, count[1, :] * data_shape[0], disp[1, :] * data_shape[0], BASE_TYPE],
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

    # initialize Green's function------------------------------------------------
    gg_h2g = np.zeros((count[1, rank], no), dtype=np.complex128)
    gl_h2g = np.zeros((count[1, rank], no), dtype=np.complex128)
    gr_h2g = np.zeros((count[1, rank], no), dtype=np.complex128)

    # initialize Screened interaction-------------------------------------------
    wg_p2w = np.zeros((count[1, rank], no), dtype=np.complex128)
    wl_p2w = np.zeros((count[1, rank], no), dtype=np.complex128)
    wr_p2w = np.zeros((count[1, rank], no), dtype=np.complex128)

    # initialize memory factors for Self-Energy, Green's Function and Screened interaction
    mem_s = 1.0
    mem_g = 1.0
    mem_w = 1.0
    # max number of iterations
    max_iter = 3
    ECmin_vec = np.concatenate((np.array([ECmin]), np.zeros(max_iter)))
    EFL_vec = np.concatenate((np.array([energy_fl]), np.zeros(max_iter)))
    EFR_vec = np.concatenate((np.array([energy_fr]), np.zeros(max_iter)))

    # output folder
    folder = '/results/CNT_sc/'

    # todo start self consistent loop-------------------------------------------
    for iter_num in range(max_iter):

        # initialize observables----------------------------------------------------
        # density of states
        dos = np.zeros(shape=(ne, nb), dtype=np.complex128)
        # current per energy
        ide = np.zeros(shape=(ne, nb), dtype=np.complex128)
        # timing transform
        times_transform_list[0] = -time.perf_counter()

        # transform from 2D format to list/vector of sparse arrays format-----------
        sg_h2g_vec = change_format.sparse2vecsparse_v2(sg_h2g, rows, columns, nao)
        sl_h2g_vec = change_format.sparse2vecsparse_v2(sl_h2g, rows, columns, nao)
        sr_h2g_vec = change_format.sparse2vecsparse_v2(sr_h2g, rows, columns, nao)

        # timing transform
        times_transform_list[0] += time.perf_counter()

        # Adjusting Fermi Levels of both contacts to the current iteration band minima
        #sr_ephn_h2g_vec = change_format.sparse2vecsparse_v2(np.zeros((count[1,rank], no), dtype=np.complex128), rows, columns, nao)
        #ECmin_vec[iter_num+1] = get_band_edge(ECmin_vec[iter_num], energy, hamiltonian_obj.Overlap['H_4'], hamiltonian_obj.Hamiltonian['H_4'], sr_h2g_vec, sr_ephn_h2g_vec, bmin, bmax, side = 'left')

        EFL_vec[iter_num + 1] = energy_fl
        EFR_vec[iter_num + 1] = energy_fr

        # timing compute
        times_compute[0] = -time.perf_counter()

        # calculate the green's function at every rank------------------------------
        if args.pool:
            gr_diag, gr_upper, gl_diag, gl_upper, gg_diag, gg_upper = calc_GF_pool.calc_GF_pool_mpi(
                hamiltonian_obj, energy_loc, sr_h2g_vec, sl_h2g_vec, sg_h2g_vec, energy_fl, energy_fr, temp,
                dos[disp[1, rank]:disp[1, rank] + count[1, rank]], factor_g_loc, gf_mkl_threads, gf_worker_threads)
        else:
            gr_diag, gr_upper, gl_diag, gl_upper, gg_diag, gg_upper = calc_GF_pool.calc_GF_mpi(
                hamiltonian_obj, energy_loc, sr_h2g_vec, sl_h2g_vec, sg_h2g_vec, energy_fl, energy_fr, temp,
                dos[disp[1, rank]:disp[1, rank] + count[1, rank]], factor_g_loc, gf_mkl_threads, 1)
        # timing compute
        times_compute[0] += time.perf_counter()

        # timing transform
        times_transform_block[0] = -time.perf_counter()

        # transform from block format to 2D format----------------------------------
        # lower diagonal blocks from physics identity
        gg_lower = -gg_upper.conjugate().transpose((0, 1, 3, 2))
        gl_lower = -gl_upper.conjugate().transpose((0, 1, 3, 2))
        gr_lower = gr_upper.transpose((0, 1, 3, 2))
        if iter_num == 0:
            gg_h2g = change_format.block2sparse_energy_alt(map_diag,
                                                           map_upper,
                                                           map_lower,
                                                           gg_diag,
                                                           gg_upper,
                                                           gg_lower,
                                                           no,
                                                           count[1, rank],
                                                           energy_contiguous=False)
            gl_h2g = change_format.block2sparse_energy_alt(map_diag,
                                                           map_upper,
                                                           map_lower,
                                                           gl_diag,
                                                           gl_upper,
                                                           gl_lower,
                                                           no,
                                                           count[1, rank],
                                                           energy_contiguous=False)
            gr_h2g = change_format.block2sparse_energy_alt(map_diag,
                                                           map_upper,
                                                           map_lower,
                                                           gr_diag,
                                                           gr_upper,
                                                           gr_lower,
                                                           no,
                                                           count[1, rank],
                                                           energy_contiguous=False)
        else:
            # add new contribution to the Green's function
            gg_h2g = (1.0 - mem_g) * change_format.block2sparse_energy_alt(map_diag,
                                                                           map_upper,
                                                                           map_lower,
                                                                           gg_diag,
                                                                           gg_upper,
                                                                           gg_lower,
                                                                           no,
                                                                           count[1, rank],
                                                                           energy_contiguous=False) + mem_g * gg_h2g
            gl_h2g = (1.0 - mem_g) * change_format.block2sparse_energy_alt(map_diag,
                                                                           map_upper,
                                                                           map_lower,
                                                                           gl_diag,
                                                                           gl_upper,
                                                                           gl_lower,
                                                                           no,
                                                                           count[1, rank],
                                                                           energy_contiguous=False) + mem_g * gl_h2g
            gr_h2g = (1.0 - mem_g) * change_format.block2sparse_energy_alt(map_diag,
                                                                           map_upper,
                                                                           map_lower,
                                                                           gr_diag,
                                                                           gr_upper,
                                                                           gr_lower,
                                                                           no,
                                                                           count[1, rank],
                                                                           energy_contiguous=False) + mem_g * gr_h2g

        # timing transform
        times_transform_block[0] += time.perf_counter()

        # distribute greens function according to g2p step--------------------------
        # calculate the transposed
        gl_transposed_h2g = np.copy(gl_h2g[:, ij2ji], order="C")

        # create local buffers
        gg_g2p = np.empty((count[0, rank], data_shape[1]), dtype=np.complex128, order="C")
        gl_g2p = np.empty((count[0, rank], data_shape[1]), dtype=np.complex128, order="C")
        gr_g2p = np.empty((count[0, rank], data_shape[1]), dtype=np.complex128, order="C")
        gl_transposed_g2p = np.empty((count[0, rank], data_shape[1]), dtype=np.complex128, order="C")

        # timing communicate
        times_communicate[0] = -time.perf_counter()

        # use of all to all w since not divisible
        alltoall_g2p(gg_h2g, gg_g2p, transpose_net=args.net_transpose)
        alltoall_g2p(gl_h2g, gl_g2p, transpose_net=args.net_transpose)
        alltoall_g2p(gr_h2g, gr_g2p, transpose_net=args.net_transpose)
        alltoall_g2p(gl_transposed_h2g, gl_transposed_g2p, transpose_net=args.net_transpose)

        # timing communicate
        times_communicate[0] += time.perf_counter()

        # timing compute
        times_compute[1] = -time.perf_counter()

        # calculate the polarization at every rank----------------------------------
        if args.type in ("gpu"):
            pg_g2p, pl_g2p, pr_g2p = g2p_gpu.g2p_fft_mpi_gpu(pre_factor, gg_g2p, gl_g2p, gr_g2p, gl_transposed_g2p)
        elif args.type in ("cpu"):
            pg_g2p, pl_g2p, pr_g2p = g2p_cpu.g2p_fft_mpi_cpu_inlined(pre_factor, gg_g2p, gl_g2p, gr_g2p,
                                                                     gl_transposed_g2p)
        else:
            raise ValueError("Argument error, input type not possible")

        # timing compute
        times_compute[1] += time.perf_counter()

        # distribute polarization function according to p2w step--------------------

        # create local buffers
        pg_p2w = np.empty((count[1, rank], data_shape[0]), dtype=np.complex128, order="C")
        pl_p2w = np.empty((count[1, rank], data_shape[0]), dtype=np.complex128, order="C")
        pr_p2w = np.empty((count[1, rank], data_shape[0]), dtype=np.complex128, order="C")

        # timing communicate
        times_communicate[1] = -time.perf_counter()

        # use of all to all w since not divisible
        alltoall_p2g(pg_g2p, pg_p2w, transpose_net=args.net_transpose)
        alltoall_p2g(pl_g2p, pl_p2w, transpose_net=args.net_transpose)
        alltoall_p2g(pr_g2p, pr_p2w, transpose_net=args.net_transpose)

        # timing communicate
        times_communicate[1] += time.perf_counter()

        # timing transform
        times_transform_list[1] = -time.perf_counter()

        # transform from 2D format to list/vector of sparse arrays format-----------
        pg_p2w_vec = change_format.sparse2vecsparse_v2(pg_p2w, rows, columns, nao)
        pl_p2w_vec = change_format.sparse2vecsparse_v2(pl_p2w, rows, columns, nao)
        pr_p2w_vec = change_format.sparse2vecsparse_v2(pr_p2w, rows, columns, nao)

        # timing transform
        times_transform_list[1] += time.perf_counter()

        # timing compute
        times_compute[2] = -time.perf_counter()

        # calculate the screened interaction on every rank--------------------------
        if args.pool:
            wg_diag, wg_upper, wl_diag, wl_upper, wr_diag, wr_upper, nb_mm, lb_max_mm = p2w_cpu.p2w_pool_mpi_cpu(
                hamiltonian_obj, energy_loc, pg_p2w_vec, pl_p2w_vec, pr_p2w_vec, vh, factor_w_loc, w_mkl_threads,
                w_worker_threads)
        else:
            wg_diag, wg_upper, wl_diag, wl_upper, wr_diag, wr_upper, nb_mm, lb_max_mm = p2w_cpu.p2w_mpi_cpu(
                hamiltonian_obj, energy_loc, pg_p2w_vec, pl_p2w_vec, pr_p2w_vec, vh, factor_w_loc, w_mkl_threads)
        # timing compute
        times_compute[2] += time.perf_counter()

        # timing transform
        times_transform_block[1] = -time.perf_counter()

        # transform from block format to 2D format-----------------------------------
        # lower diagonal blocks from physics identity
        wg_lower = -wg_upper.conjugate().transpose((0, 1, 3, 2))
        wl_lower = -wl_upper.conjugate().transpose((0, 1, 3, 2))
        wr_lower = wr_upper.transpose((0, 1, 3, 2))
        if iter_num == 0:
            wg_p2w = change_format.block2sparse_energy_alt(map_diag_mm,
                                                           map_upper_mm,
                                                           map_lower_mm,
                                                           wg_diag,
                                                           wg_upper,
                                                           wg_lower,
                                                           no,
                                                           count[1, rank],
                                                           energy_contiguous=False)
            wl_p2w = change_format.block2sparse_energy_alt(map_diag_mm,
                                                           map_upper_mm,
                                                           map_lower_mm,
                                                           wl_diag,
                                                           wl_upper,
                                                           wl_lower,
                                                           no,
                                                           count[1, rank],
                                                           energy_contiguous=False)
            wr_p2w = change_format.block2sparse_energy_alt(map_diag_mm,
                                                           map_upper_mm,
                                                           map_lower_mm,
                                                           wr_diag,
                                                           wr_upper,
                                                           wr_lower,
                                                           no,
                                                           count[1, rank],
                                                           energy_contiguous=False)
        else:
            # add new contribution to the Screened interaction
            wg_p2w = (1.0 - mem_w) * change_format.block2sparse_energy_alt(map_diag_mm,
                                                                           map_upper_mm,
                                                                           map_lower_mm,
                                                                           wg_diag,
                                                                           wg_upper,
                                                                           wg_lower,
                                                                           no,
                                                                           count[1, rank],
                                                                           energy_contiguous=False) + mem_w * wg_p2w
            wl_p2w = (1.0 - mem_w) * change_format.block2sparse_energy_alt(map_diag_mm,
                                                                           map_upper_mm,
                                                                           map_lower_mm,
                                                                           wl_diag,
                                                                           wl_upper,
                                                                           wl_lower,
                                                                           no,
                                                                           count[1, rank],
                                                                           energy_contiguous=False) + mem_w * wl_p2w
            wr_p2w = (1.0 - mem_w) * change_format.block2sparse_energy_alt(map_diag_mm,
                                                                           map_upper_mm,
                                                                           map_lower_mm,
                                                                           wr_diag,
                                                                           wr_upper,
                                                                           wr_lower,
                                                                           no,
                                                                           count[1, rank],
                                                                           energy_contiguous=False) + mem_w * wr_p2w

        # timing transform
        times_transform_block[1] += time.perf_counter()

        # distribute screened interaction according to gw2s step--------------------

        # calculate the transposed
        wg_transposed_p2w = np.copy(wg_p2w[:, ij2ji], order="C")
        wl_transposed_p2w = np.copy(wl_p2w[:, ij2ji], order="C")

        # create local buffers
        wg_gw2s = np.empty((count[0, rank], data_shape[1]), dtype=np.complex128, order="C")
        wl_gw2s = np.empty((count[0, rank], data_shape[1]), dtype=np.complex128, order="C")
        wr_gw2s = np.empty((count[0, rank], data_shape[1]), dtype=np.complex128, order="C")
        wg_transposed_gw2s = np.empty((count[0, rank], data_shape[1]), dtype=np.complex128, order="C")
        wl_transposed_gw2s = np.empty((count[0, rank], data_shape[1]), dtype=np.complex128, order="C")

        # timing communicate
        times_communicate[2] = -time.perf_counter()

        # use of all to all w since not divisible
        alltoall_g2p(wg_p2w, wg_gw2s, transpose_net=args.net_transpose)
        alltoall_g2p(wl_p2w, wl_gw2s, transpose_net=args.net_transpose)
        alltoall_g2p(wr_p2w, wr_gw2s, transpose_net=args.net_transpose)
        alltoall_g2p(wg_transposed_p2w, wg_transposed_gw2s, transpose_net=args.net_transpose)
        alltoall_g2p(wl_transposed_p2w, wl_transposed_gw2s, transpose_net=args.net_transpose)

        # timing communicate
        times_communicate[2] += time.perf_counter()

        # timing compute
        times_compute[3] = -time.perf_counter()

        # calculate the self-energy on every rank-----------------------------------
        # tod optimize and not load two time green's function to gpu and do twice the fft
        if args.type in ("gpu"):
            sg_gw2s, sl_gw2s, sr_gw2s = gw2s_gpu.gw2s_fft_mpi_gpu(-pre_factor / 2, gg_g2p, gl_g2p, gr_g2p, wg_gw2s,
                                                                  wl_gw2s, wr_gw2s, wg_transposed_gw2s,
                                                                  wl_transposed_gw2s)
        elif args.type in ("cpu"):
            sg_gw2s, sl_gw2s, sr_gw2s = gw2s_cpu.gw2s_fft_mpi_cpu(-pre_factor / 2, gg_g2p, gl_g2p, gr_g2p, wg_gw2s,
                                                                  wl_gw2s, wr_gw2s, wg_transposed_gw2s,
                                                                  wl_transposed_gw2s)
        else:
            raise ValueError("Argument error, input type not possible")

        # timing compute
        times_compute[3] += time.perf_counter()

        # distribute screened interaction according to h2g step---------------------
        # create local buffers
        sg_h2g_buf = np.empty((count[1, rank], data_shape[0]), dtype=np.complex128, order="C")
        sl_h2g_buf = np.empty((count[1, rank], data_shape[0]), dtype=np.complex128, order="C")
        sr_h2g_buf = np.empty((count[1, rank], data_shape[0]), dtype=np.complex128, order="C")

        # timing communicate
        times_communicate[3] = -time.perf_counter()

        # use of all to all w since not divisible
        alltoall_p2g(sg_gw2s, sg_h2g_buf, transpose_net=args.net_transpose)
        alltoall_p2g(sl_gw2s, sl_h2g_buf, transpose_net=args.net_transpose)
        alltoall_p2g(sr_gw2s, sr_h2g_buf, transpose_net=args.net_transpose)

        if iter_num == 0:
            sg_h2g = (1.0 - mem_s) * sg_h2g_buf + mem_s * sg_h2g
            sl_h2g = (1.0 - mem_s) * sl_h2g_buf + mem_s * sl_h2g
            sr_h2g = (1.0 - mem_s) * sr_h2g_buf + mem_s * sr_h2g
        else:
            # add new contribution to the Self-Energy
            sg_h2g = (1.0 - mem_s) * sg_h2g_buf + mem_s * sg_h2g
            sl_h2g = (1.0 - mem_s) * sl_h2g_buf + mem_s * sl_h2g
            sr_h2g = (1.0 - mem_s) * sr_h2g_buf + mem_s * sr_h2g

        if iter_num == max_iter - 1:
            alltoall_p2g(sg_gw2s, sg_h2g, transpose_net=args.net_transpose)
            alltoall_p2g(sl_gw2s, sl_h2g, transpose_net=args.net_transpose)
            alltoall_p2g(sr_gw2s, sr_h2g, transpose_net=args.net_transpose)

        # timing communicate
        times_communicate[3] += time.perf_counter()

        # Wrapping up the iteration
        if rank == 0:
            comm.Reduce(MPI.IN_PLACE, dos, op=MPI.SUM, root=0)
            #average time
            comm.Reduce(MPI.IN_PLACE, times_transform_list, op=MPI.SUM, root=0)
            comm.Reduce(MPI.IN_PLACE, times_transform_block, op=MPI.SUM, root=0)
            comm.Reduce(MPI.IN_PLACE, times_communicate, op=MPI.SUM, root=0)
            comm.Reduce(MPI.IN_PLACE, times_compute, op=MPI.SUM, root=0)
            times_transform_list /= size
            times_transform_block /= size
            times_communicate /= size
            times_compute /= size
        else:
            comm.Reduce(dos, None, op=MPI.SUM, root=0)
            #send time to master
            comm.Reduce(times_transform_list, None, op=MPI.SUM, root=0)
            comm.Reduce(times_transform_block, None, op=MPI.SUM, root=0)
            comm.Reduce(times_communicate, None, op=MPI.SUM, root=0)
            comm.Reduce(times_compute, None, op=MPI.SUM, root=0)
        # print times and write DOS
        if rank == 0:
            print("Finishing Iteration: ", iter_num)
            # print times
            print("Times to transform 2D to sparsematrix_vector: ", times_transform_list)
            print("Times to transform Block-4D to 2D: ", times_transform_block)
            print("Times to communicate: ", times_communicate)
            print("Times to compute: ", times_compute)
            print(
                "Total time: ",
                np.sum(times_transform_list) + np.sum(times_transform_block) + np.sum(times_communicate) +
                np.sum(times_compute))

            #np.savetxt(parent_path + folder + 'E.dat', energy)
            #np.savetxt(parent_path + folder + 'DOS_' + str(iter_num) + '.dat', dos.view(float))

    if rank == 0:
        pass
        #np.savetxt(parent_path + folder + 'EFL.dat', EFL_vec)
        #np.savetxt(parent_path + folder + 'EFR.dat', EFR_vec)
    # culminate on master again-------------------------------------------------
    if rank == 0:
        # average time
        comm.Reduce(MPI.IN_PLACE, times_transform_list, op=MPI.SUM, root=0)
        comm.Reduce(MPI.IN_PLACE, times_transform_block, op=MPI.SUM, root=0)
        comm.Reduce(MPI.IN_PLACE, times_communicate, op=MPI.SUM, root=0)
        comm.Reduce(MPI.IN_PLACE, times_compute, op=MPI.SUM, root=0)
        times_transform_list /= size
        times_transform_block /= size
        times_communicate /= size
        times_compute /= size

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
    else:
        # send time to master
        comm.Reduce(times_transform_list, None, op=MPI.SUM, root=0)
        comm.Reduce(times_transform_block, None, op=MPI.SUM, root=0)
        comm.Reduce(times_communicate, None, op=MPI.SUM, root=0)
        comm.Reduce(times_compute, None, op=MPI.SUM, root=0)

        gather_master(gg_h2g, None, transpose_net=args.net_transpose)
        gather_master(gl_h2g, None, transpose_net=args.net_transpose)
        gather_master(gr_h2g, None, transpose_net=args.net_transpose)
        gather_master(pg_p2w, None, transpose_net=args.net_transpose)
        gather_master(pl_p2w, None, transpose_net=args.net_transpose)
        gather_master(pr_p2w, None, transpose_net=args.net_transpose)
        gather_master(wg_p2w, None, transpose_net=args.net_transpose)
        gather_master(wl_p2w, None, transpose_net=args.net_transpose)
        gather_master(wr_p2w, None, transpose_net=args.net_transpose)
        gather_master(sg_h2g, None, transpose_net=args.net_transpose)
        gather_master(sl_h2g, None, transpose_net=args.net_transpose)
        gather_master(sr_h2g, None, transpose_net=args.net_transpose)

    # test against gold solution------------------------------------------------

    if rank == 0:
        # print times
        print("Times to transform 2D to sparsematrix_vector: ", times_transform_list)
        print("Times to transform Block-4D to 2D: ", times_transform_block)
        print("Times to communicate: ", times_communicate)
        print("Times to compute: ", times_compute)
        print(
            "Total time: ",
            np.sum(times_transform_list) + np.sum(times_transform_block) + np.sum(times_communicate) +
            np.sum(times_compute))

        # print difference to given solution
        # use Frobenius norm
        diff_gg = np.linalg.norm(gg_gold - gg_mpi)
        diff_gl = np.linalg.norm(gl_gold - gl_mpi)
        diff_gr = np.linalg.norm(gr_gold - gr_mpi)
        diff_pg = np.linalg.norm(pg_gold - pg_mpi)
        diff_pl = np.linalg.norm(pl_gold - pl_mpi)
        diff_pr = np.linalg.norm(pr_gold - pr_mpi)
        diff_wg = np.linalg.norm(wg_gold - wg_mpi)
        diff_wl = np.linalg.norm(wl_gold - wl_mpi)
        diff_wr = np.linalg.norm(wr_gold - wr_mpi)
        diff_sg = np.linalg.norm(sg_gold - sg_mpi)
        diff_sl = np.linalg.norm(sl_gold - sl_mpi)
        diff_sr = np.linalg.norm(sr_gold - sr_mpi)
        print(f"Green's Function differences to Gold Solution g/l/r:  {diff_gg:.4f}, {diff_gl:.4f}, {diff_gr:.4f}")
        print(f"Polarization differences to Gold Solution g/l/r:  {diff_pg:.4f}, {diff_pl:.4f}, {diff_pr:.4f}")
        print(f"Screened interaction differences to Gold Solution g/l/r:  {diff_wg:.4f}, {diff_wl:.4f}, {diff_wr:.4f}")
        print(f"Screened self-energy differences to Gold Solution g/l/r:  {diff_sg:.4f}, {diff_sl:.4f}, {diff_sr:.4f}")

        # assert solution close to real solution
        abstol = 1e-2
        reltol = 1e-1
        assert diff_gg <= abstol + reltol * np.max(np.abs(gg_gold))
        assert diff_gl <= abstol + reltol * np.max(np.abs(gl_gold))
        assert diff_gr <= abstol + reltol * np.max(np.abs(gr_gold))
        assert diff_pg <= abstol + reltol * np.max(np.abs(pg_gold))
        assert diff_pl <= abstol + reltol * np.max(np.abs(pl_gold))
        assert diff_pr <= abstol + reltol * np.max(np.abs(pr_gold))
        assert diff_wg <= abstol + reltol * np.max(np.abs(wg_gold))
        assert diff_wl <= abstol + reltol * np.max(np.abs(wl_gold))
        assert diff_wr <= abstol + reltol * np.max(np.abs(wr_gold))
        assert diff_sg <= abstol + reltol * np.max(np.abs(sg_gold))
        assert diff_sl <= abstol + reltol * np.max(np.abs(sl_gold))
        assert diff_sr <= abstol + reltol * np.max(np.abs(sr_gold))
        assert np.allclose(gg_gold, gg_mpi, atol=1e-3, rtol=1e-3)
        assert np.allclose(gl_gold, gl_mpi, atol=1e-3, rtol=1e-3)
        assert np.allclose(gr_gold, gr_mpi, atol=1e-3, rtol=1e-3)
        assert np.allclose(pg_gold, pg_mpi, atol=1e-6, rtol=1e-6)
        assert np.allclose(pl_gold, pl_mpi, atol=1e-6, rtol=1e-6)
        assert np.allclose(pr_gold, pr_mpi, atol=1e-6, rtol=1e-6)
        assert np.allclose(wg_gold, wg_mpi, rtol=1e-2, atol=1e-2)
        assert np.allclose(wl_gold, wl_mpi, atol=1e-6, rtol=1e-6)
        assert np.allclose(wr_gold, wr_mpi, atol=1e-6, rtol=1e-6)
        assert np.allclose(sg_gold, sg_mpi, atol=1e-2, rtol=1e-2)
        assert np.allclose(sl_gold, sl_mpi, atol=1e-2, rtol=1e-2)
        assert np.allclose(sr_gold, sr_mpi, atol=1e-2, rtol=1e-2)
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
