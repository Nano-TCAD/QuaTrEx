# Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.
"""
Example a sc-GW iteration with MPI+CUDA.
With transposition through network.
This application is for a tight-binding InAs nanowire. 
See the different GW step folders for more explanations.
"""
from mpi4py import MPI
import sys
import numpy as np
import os
import argparse
from scipy import sparse
import time
import pickle
from quatrex.utils.communication import TransposeMatrix, CommunicateCompute
from quatrex.utils.matrix_creation import get_number_connected_blocks
from quatrex.utils import utils_gpu
from quatrex.utils import change_format
from quatrex.OMEN_structure_matrices.construct_CM import construct_coulomb_matrix
from quatrex.OMEN_structure_matrices import OMENHamClass
from quatrex.GreensFunction import calc_GF_pool
from quatrex.GW.screenedinteraction.kernel import p2w_cpu
from quatrex.GW.gold_solution import read_solution
from quatrex.GW.selfenergy.kernel import gw2s_cpu
from quatrex.GW.polarization.kernel import g2p_cpu
from quatrex.bandstructure.calc_band_edge import get_band_edge_mpi_interpol


if utils_gpu.gpu_avail():
    try:
        from quatrex.GW.polarization.kernel import g2p_gpu
        from quatrex.GW.selfenergy.kernel import gw2s_gpu
    except ImportError:
        print("GPU import error, make sure you have the right GPU driver and CUDA version installed")

if __name__ == "__main__":

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    proc_name = MPI.Get_processor_name()
    base_type = np.complex128
    gw_num_iter = 2
    is_padded = True
    comm_unblock = True

    if rank == 0:
        time_startup = -time.perf_counter()
        time_read_gold = -time.perf_counter()

    # path to solution
    scratch_path = "/usr/scratch/mont-fort17/dleonard/GW_paper/"
    solution_path = os.path.join(scratch_path, "InAs")
    hamiltonian_path = solution_path

    # coulomb matrix path
    solution_path_vh = os.path.join(
        solution_path, "data_Vh_finalPI_InAs_0v.mat")

    # gw matrices path
    solution_path_gw = os.path.join(
        solution_path, "data_GPWS_big_memory1_InAs_0V.mat")
    # solution_path_gw = "/usr/scratch/mont-fort17/almaeder/test_gw/few_energy.mat"
    # solution_path_gw = "/usr/scratch/mont-fort17/almaeder/test_gw/few_energy_iter2.mat"

    parser = argparse.ArgumentParser(
        description="Reference test of GW iterations with MPI+CUDA")
    parser.add_argument("-fvh", "--file_vh",
                        default=solution_path_vh, required=False)
    parser.add_argument("-fpw", "--file_gw",
                        default=solution_path_gw, required=False)
    parser.add_argument("-fhm", "--file_hm",
                        default=hamiltonian_path, required=False)
    parser.add_argument("-t", "--type", default="cpu",
                        choices=["cpu", "gpu"], required=False)
    parser.add_argument("-nt", "--net_transpose",
                        default=False, type=bool, required=False)
    parser.add_argument("-p", "--pool", default=True, type=bool, required=False)
    args = parser.parse_args()

    # check if gpu is available
    if args.type in ("gpu"):
        if not utils_gpu.gpu_avail():
            print("No gpu available")
            sys.exit(1)
    print(f"Using {args.type} implementation")

    # reading reference solution-------------------------------------------------
    gw_names = ["g", "p", "w", "s"]
    gw_types = ["g", "l", "r"]
    gw_full_names = {"g": "Greens Function", "p": "Polarization",
                     "w": "Screened Interaction", "s": "Self-Energy"}
    matrices_gold = {}
    for gw_name in gw_names:
        _, rows, columns, g_gold, l_gold, r_gold = read_solution.load_x(
            solution_path_gw, gw_name)
        matrices_gold[gw_name + "g"] = g_gold
        matrices_gold[gw_name + "l"] = l_gold
        matrices_gold[gw_name + "r"] = r_gold
        matrices_gold[gw_name + "rows"] = rows
        matrices_gold[gw_name + "columns"] = columns
    for gw_name in gw_names:
        assert np.all(matrices_gold[gw_name + "rows"] == matrices_gold["grows"])
        assert np.all(matrices_gold[gw_name + "columns"]
                      == matrices_gold["gcolumns"])

    rows = matrices_gold["grows"]
    columns = matrices_gold["gcolumns"]
    rowsRef, columnsRef, vh_gold = read_solution.load_v(solution_path_vh)
    # transposition vector
    ij2ji = change_format.find_idx_transposed(
        rows, columns)

    if rank == 0:
        time_read_gold += time.perf_counter()
        time_pre_compute = -time.perf_counter()

    # physics parameters---------------------------------------------------------
    # create hamiltonian object
    # one orbital on C atoms, two same types
    no_orb = np.array([5, 5])
    Vappl = 0.4
    # Fermi Level of Left Contact
    energy_fl = 1.9
    # Fermi Level of Right Contact
    energy_fr = energy_fl - Vappl
    # Temperature in Kelvin
    temp = 300
    # relative permittivity
    epsR = 2.5
    # DFT Conduction Band Minimum
    ECmin = 1.9346

    # Phyiscal Constants -----------
    e = 1.6022e-19
    eps0 = 8.854e-12
    hbar = 1.0546e-34

    # Fermi Level to Band Edge Difference
    dEfL_EC = energy_fl - ECmin
    dEfR_EC = energy_fr - ECmin

    energy = np.linspace(-10.0, 5.0, 376, endpoint=True,
                         dtype=float)
    denergy = energy[1] - energy[0]
    pre_factor = -1.0j * denergy / (np.pi)
    hamiltonian_obj = OMENHamClass.Hamiltonian(
        args.file_hm, no_orb, Vappl=Vappl, rank=rank)
    # broadcast hamiltonian object
    serial_ham = pickle.dumps(hamiltonian_obj)
    broadcasted_ham = comm.bcast(serial_ham, root=0)
    hamiltonian_obj = pickle.loads(broadcasted_ham)

    # hamiltonian object has 1-based indexing
    bmax = hamiltonian_obj.Bmax - 1
    bmin = hamiltonian_obj.Bmin - 1

    # number of energy points/non-zero elements and number of orbitals
    ne = np.int32(energy.shape[0])
    no = np.int32(columns.shape[0])
    nao = np.max(bmax) + 1
    data_shape = np.array([rows.shape[0],
                          energy.shape[0]], dtype=np.int32)

    vh = sparse.coo_array((vh_gold, (np.squeeze(rowsRef), np.squeeze(columnsRef))),
                          shape=(nao, nao),
                          dtype=base_type).tocsr()

    V_sparse = construct_coulomb_matrix(
        hamiltonian_obj, epsR, eps0, e, diag=False, orb_uniform=True)
    vh1d = np.asarray(vh[rows, columns].reshape(-1))

    assert np.allclose(V_sparse.toarray(), vh.toarray())

    ECmin_vec = np.concatenate((np.array([ECmin]), np.zeros(gw_num_iter)))
    EFL_vec = np.concatenate((np.array([energy_fl]), np.zeros(gw_num_iter)))
    EFR_vec = np.concatenate((np.array([energy_fr]), np.zeros(gw_num_iter)))

    # energy depenedent pre-factors
    factor_w = np.ones(ne)
    factor_g = np.ones(ne)

    # memory factors for Self-Energy, Green's Function and Screened interaction
    mem_s = 0.5
    # todo
    # mem_g = 0.0
    mem_w = 0.1

    # maps to transform after rgf for g and w
    map_diag, map_upper, map_lower = change_format.map_block2sparse_alt(
        rows, columns, bmax, bmin)
    map_ = [map_diag, map_upper, map_lower]

    # number of blocks
    nb = hamiltonian_obj.Bmin.shape[0]
    nbc = get_number_connected_blocks(
        hamiltonian_obj.NH, bmin, bmax, rows, columns)

    if rank == 0:
        # print size of data
        print(f"#Energy: {data_shape[1]} #nnz: {data_shape[0]}")

    # computation parameters----------------------------------------------------
    # set number of workers and threads
    w_mkl_threads = 1
    w_worker_threads = 8
    g_mkl_threads = 1
    g_worker_threads = 8
    # greater, lesser, retarded
    gw_num_buffer = 3
    # additional transposed
    g_num_buffer = 4
    p_num_buffer = 3
    w_num_buffer = 5
    s_num_buffer = 3
    num_buffer = max(g_num_buffer, p_num_buffer, w_num_buffer, s_num_buffer)

    # calculation of data distribution per rank---------------------------------
    padding = 0
    extra_elements = np.zeros_like(data_shape)

    if is_padded:
        padding = (size - data_shape % size) % size
        if rank == size-1:
            extra_elements[:] = padding

    data_shape_padded = data_shape + padding
    distribution = TransposeMatrix(comm, data_shape_padded, base_type=base_type)
    distribution_no_padded = TransposeMatrix(
        comm, data_shape, base_type=base_type)

    flag_zeros = np.zeros(distribution.count[1])
    range_local_no_padding = [slice(distribution.displacement[0], min(distribution.displacement[0] + distribution.count[0], data_shape[0])),
                              slice(distribution.displacement[1], min(distribution.displacement[1] + distribution.count[1], data_shape[1]))]
    flag_end = min(distribution.count[1],
                   data_shape[1] - distribution.displacement[1])

    batchsize_row = distribution.count[0] // 2
    batchsize_col = distribution.count[1] // 2
    iteration_row = int(np.ceil(distribution.count[0] / batchsize_row))
    iteration_col = int(np.ceil(distribution.count[1] / batchsize_col))

    distribution_unblock_row = TransposeMatrix(comm, np.array(
        [batchsize_row*size, data_shape_padded[1]]), base_type=base_type)
    distribution_unblock_col = TransposeMatrix(comm, np.array(
        [data_shape_padded[0], batchsize_col*size]), base_type=base_type)

    distributions = [distribution for _ in range(num_buffer)]
    distributions_unblock_row = [
        distribution_unblock_row for _ in range(num_buffer)]
    distributions_unblock_col = [
        distribution_unblock_col for _ in range(num_buffer)]
    iterations_unblock = [iteration_row, iteration_col]

    bmax_mm = bmax[nbc - 1:nb:nbc]
    bmin_mm = bmin[0:nb:nbc]

    map_diag_mm, map_upper_mm, map_lower_mm = change_format.map_block2sparse_alt(
        rows, columns, bmax_mm, bmin_mm)
    map_mm_ = [map_diag_mm, map_upper_mm, map_lower_mm]

    # observables and others----------------------------------------------------
    # Energy Index Vector
    Idx_e = np.arange(energy.shape[0])
    # density of states
    dos = np.zeros(shape=(distribution.shape[1], nb), dtype=base_type)
    dosw = np.zeros(shape=(distribution.shape[1], nb // nbc), dtype=base_type)
    # occupied states/unoccupied states
    nE = np.zeros(shape=(distribution.shape[1], nb), dtype=base_type)
    nP = np.zeros(shape=(distribution.shape[1], nb), dtype=base_type)
    # occupied screening/unoccupied screening
    nEw = np.zeros(shape=(distribution.shape[1], nb // nbc), dtype=base_type)
    nPw = np.zeros(shape=(distribution.shape[1], nb // nbc), dtype=base_type)
    # current per energy
    ide = np.zeros(shape=(distribution.shape[1], nb), dtype=base_type)

    # padding of 1D arrays
    energy_padded = np.zeros((distribution.shape[1]), dtype=energy.dtype)
    energy_padded[:data_shape[1]] = energy
    Idx_e_padded = np.zeros((distribution.shape[1]), dtype=Idx_e.dtype)
    Idx_e_padded[:data_shape[1]] = Idx_e
    factor_w_padded = np.zeros((distribution.shape[1]), dtype=factor_w.dtype)
    factor_w_padded[:data_shape[1]] = factor_w
    factor_g_padded = np.zeros((distribution.shape[1]), dtype=factor_g.dtype)
    factor_g_padded[:data_shape[1]] = factor_g
    vh_padded = np.zeros((distribution.shape[0]), dtype=vh.dtype)
    vh_padded[:data_shape[0]] = vh1d

    # split up the factor between the ranks
    energy_loc = energy_padded[distribution.range_local[1]]
    Idx_e_loc = Idx_e_padded[distribution.range_local[1]]
    factor_w_loc = factor_w_padded[distribution.range_local[1]]
    factor_g_loc = factor_g_padded[distribution.range_local[1]]
    vh_loc = vh_padded[distribution.range_local[0]]

    # print rank distribution
    print(
        f"Rank: {rank} #Energy/rank: {distribution.count[1]} #nnz/rank: {distribution.count[0]}", proc_name)

    if rank == 0:
        time_pre_compute += time.perf_counter()
        time_alloc_buf = -time.perf_counter()

    # create local buffers----------------------------------------------------------
    # order g,l,r and the transposed needed
    g_col = [np.zeros((distribution.count[1], distribution.shape[0]),
                      dtype=base_type, order="C") for _ in range(g_num_buffer)]
    p_col = [np.zeros((distribution.count[1], distribution.shape[0]),
                      dtype=base_type, order="C") for _ in range(p_num_buffer)]
    w_col = [np.zeros((distribution.count[1], distribution.shape[0]),
                      dtype=base_type, order="C") for _ in range(w_num_buffer)]
    w_col_tmp = [np.zeros((distribution.count[1], distribution.shape[0]),
                          dtype=base_type, order="C") for _ in range(w_num_buffer)]
    s_col = [np.zeros((distribution.count[1], distribution.shape[0]),
                      dtype=base_type, order="C") for _ in range(s_num_buffer)]
    s_col_tmp = [np.zeros((distribution.count[1], distribution.shape[0]),
                          dtype=base_type, order="C") for _ in range(s_num_buffer)]

    g_row = [np.zeros((distribution.count[0], distribution.shape[1]),
                      dtype=base_type, order="C") for _ in range(g_num_buffer)]
    p_row = [np.zeros((distribution.count[0], distribution.shape[1]),
                      dtype=base_type, order="C") for _ in range(p_num_buffer)]
    w_row = [np.zeros((distribution.count[0], distribution.shape[1]),
                      dtype=base_type, order="C") for _ in range(w_num_buffer)]
    w_row_tmp = [np.zeros((distribution.count[0], distribution.shape[1]),
                          dtype=base_type, order="C") for _ in range(w_num_buffer)]
    s_row = [np.zeros((distribution.count[0], distribution.shape[1]),
                      dtype=base_type, order="C") for _ in range(s_num_buffer)]
    s_row_tmp = [np.zeros((distribution.count[0], distribution.shape[1]),
                          dtype=base_type, order="C") for _ in range(s_num_buffer)]

    buffer_row_compute = [np.empty((distribution_unblock_row.count[0],
                                    distribution_unblock_row.shape[1]), dtype=base_type, order="C") for _ in range(num_buffer)]
    buffer_row_send = [np.empty((distribution_unblock_row.count[0],
                                 distribution_unblock_row.shape[1]), dtype=base_type, order="C") for _ in range(num_buffer)]
    buffer_col_recv = [np.empty((distribution_unblock_row.count[1],
                                 distribution_unblock_row.shape[0]), dtype=base_type, order="C") for _ in range(num_buffer)]

    buffer_col_compute = [np.empty((distribution_unblock_col.count[1],
                                    distribution_unblock_col.shape[0]), dtype=base_type, order="C") for _ in range(num_buffer)]
    buffer_col_send = [np.empty((distribution_unblock_col.count[1],
                                 distribution_unblock_col.shape[0]), dtype=base_type, order="C") for _ in range(num_buffer)]
    buffer_row_recv = [np.empty((distribution_unblock_col.count[0],
                                 distribution_unblock_col.shape[1]), dtype=base_type, order="C") for _ in range(num_buffer)]

    matrices_loc_row = {}
    matrices_loc_col = {}
    # transposed arrays are not compared in the end
    for i in range(gw_num_buffer):
        matrices_loc_row[gw_names[0] + gw_types[i]] = g_row[i]
        matrices_loc_row[gw_names[1] + gw_types[i]] = p_row[i]
        matrices_loc_row[gw_names[2] + gw_types[i]] = w_row[i]
        matrices_loc_row[gw_names[3] + gw_types[i]] = s_row[i]
        matrices_loc_col[gw_names[0] + gw_types[i]] = g_col[i]
        matrices_loc_col[gw_names[1] + gw_types[i]] = p_col[i]
        matrices_loc_col[gw_names[2] + gw_types[i]] = w_col[i]
        matrices_loc_col[gw_names[3] + gw_types[i]] = s_col[i]

    if rank == 0:
        time_alloc_buf += time.perf_counter()
        time_def_func = -time.perf_counter()

    def greens_function_compute(sgi_vec, sli_vec, sri_vec,
                                energy_loc_batch, dos_loc_batch, nE_loc_batch,
                                nP_loc_batch, ide_loc_batch, factor_g_loc_batch,
                                energy_fermi_left, energy_fermi_right,
                                ggo, glo, gro, glto):
        # calculate the green's function at every rank------------------------------
        ne_loc = energy_loc_batch.shape[0]
        if args.pool:
            gr_diag, gr_upper, gl_diag, gl_upper, gg_diag, gg_upper = calc_GF_pool.calc_GF_pool_mpi_no_filter(
                hamiltonian_obj,
                energy_loc_batch,
                sri_vec, sli_vec, sgi_vec,
                energy_fermi_left,
                energy_fermi_right,
                temp,
                dos_loc_batch,
                nE_loc_batch,
                nP_loc_batch,
                ide_loc_batch,
                factor_g_loc_batch,
                homogenize=False,
                mkl_threads=g_mkl_threads,
                worker_num=g_worker_threads)
        else:
            raise ValueError(
                "Argument error, I will remake this the other option later")

        # lower diagonal blocks from physics identity
        gg_lower = -gg_upper.conjugate().transpose((0, 1, 3, 2))
        gl_lower = -gl_upper.conjugate().transpose((0, 1, 3, 2))
        gr_lower = gr_upper.transpose((0, 1, 3, 2))

        ggo[:ne_loc, :data_shape[0]] = change_format.block2sparse_energy_alt(*map_,
                                                                             gg_diag,
                                                                             gg_upper,
                                                                             gg_lower,
                                                                             data_shape[0],
                                                                             ne_loc,
                                                                             energy_contiguous=False)
        glo[:ne_loc, :data_shape[0]] = change_format.block2sparse_energy_alt(*map_,
                                                                             gl_diag,
                                                                             gl_upper,
                                                                             gl_lower,
                                                                             data_shape[0],
                                                                             ne_loc,
                                                                             energy_contiguous=False)
        gro[:ne_loc, :data_shape[0]] = change_format.block2sparse_energy_alt(*map_,
                                                                             gr_diag,
                                                                             gr_upper,
                                                                             gr_lower,
                                                                             data_shape[0],
                                                                             ne_loc,
                                                                             energy_contiguous=False)

        # calculate the transposed
        glto[:ne_loc, :data_shape[0]] = np.copy(
            glo[:ne_loc, :data_shape[0]][:, ij2ji], order="C")

    def polarization_compute(ggi, gli, gri, glti, pgo, plo, pro):
        no_loc = gri.shape[0]
        if args.type in ("gpu"):
            pgo[:no_loc, :data_shape[1]], plo[:no_loc, :data_shape[1]], pro[:no_loc, :data_shape[1]] = g2p_gpu.g2p_fft_mpi_gpu(
                pre_factor, ggi[:, :data_shape[1]], gli[:, :data_shape[1]], gri[:, :data_shape[1]], glti[:, :data_shape[1]])
        elif args.type in ("cpu"):
            pgo[:no_loc, :data_shape[1]], plo[:no_loc, :data_shape[1]], pro[:no_loc, :data_shape[1]] = g2p_cpu.g2p_fft_mpi_cpu_inlined(
                pre_factor, ggi[:, :data_shape[1]], gli[:, :data_shape[1]], gri[:, :data_shape[1]], glti[:, :data_shape[1]])
        else:
            raise ValueError("Argument error, input type not possible")

    def screened_interaction_compute(pgi, pli, pri, energy_loc_batch,
                                     dosw_loc_batch, nEw_loc_batch, nPw_loc_batch,
                                     Idx_e_loc_batch, factor_w_loc_batch,
                                     wgo, wlo, wro, wgto, wlto):

        ne_loc = energy_loc_batch.shape[0]
        # transform from 2D format to list/vector of sparse arrays format-----------
        pg_col_vec = change_format.sparse2vecsparse_v2(
            pgi[:, :data_shape[0]], rows, columns, nao)
        pl_col_vec = change_format.sparse2vecsparse_v2(
            pli[:, :data_shape[0]], rows, columns, nao)
        pr_col_vec = change_format.sparse2vecsparse_v2(
            pri[:, :data_shape[0]], rows, columns, nao)

        # calculate the screened interaction on every rank--------------------------
        if args.pool:
            wg_diag, wg_upper, wl_diag, wl_upper, wr_diag, wr_upper, _, _ = p2w_cpu.p2w_pool_mpi_cpu_no_filter(
                hamiltonian_obj,
                energy_loc_batch,
                pg_col_vec,
                pl_col_vec,
                pr_col_vec,
                vh,
                dosw_loc_batch,
                nEw_loc_batch,
                nPw_loc_batch,
                Idx_e_loc_batch,
                factor_w_loc_batch,
                nbc,
                homogenize=False,
                mkl_threads=w_mkl_threads,
                worker_num=w_worker_threads)
        else:
            raise ValueError(
                "Argument error, I will remake this the other option later")

        # transform from block format to 2D format-----------------------------------
        # lower diagonal blocks from physics identity
        wg_lower = -wg_upper.conjugate().transpose((0, 1, 3, 2))
        wl_lower = -wl_upper.conjugate().transpose((0, 1, 3, 2))
        wr_lower = wr_upper.transpose((0, 1, 3, 2))

        wgo[:ne_loc, :data_shape[0]] = change_format.block2sparse_energy_alt(
            *map_mm_,
            wg_diag,
            wg_upper,
            wg_lower,
            no,
            ne_loc,
            energy_contiguous=False)
        wlo[:ne_loc, :data_shape[0]] = change_format.block2sparse_energy_alt(
            *map_mm_,
            wl_diag,
            wl_upper,
            wl_lower,
            no,
            ne_loc,
            energy_contiguous=False)
        wro[:ne_loc, :data_shape[0]] = change_format.block2sparse_energy_alt(
            *map_mm_,
            wr_diag,
            wr_upper,
            wr_lower,
            no,
            ne_loc,
            energy_contiguous=False)

        # distribute screened interaction according to gw2s step--------------------

        # calculate the transposed
        wgto[:ne_loc, :data_shape[0]] = np.copy(
            wgo[:ne_loc, :data_shape[0]][:, ij2ji], order="C")
        wlto[:ne_loc, :data_shape[0]] = np.copy(
            wlo[:ne_loc, :data_shape[0]][:, ij2ji], order="C")

    def selfenergy_compute(ggi, gli, gri, wgi, wli, wri, wgti, wlti, vh_pi,  sgo, slo, sro):
        # todo optimize and not load two time green's function to gpu and do twice the fft
        no_loc = gri.shape[0]
        if args.type in ("gpu"):
            sg_tmp, sl_tmp, sr_tmp = gw2s_gpu.gw2s_fft_mpi_gpu_3part_sr(
                -pre_factor / 2, ggi[:, :data_shape[1]], gli[:,
                                                             :data_shape[1]], gri[:, :data_shape[1]],
                wgi[:, :data_shape[1]], wli[:, :data_shape[1]], wri[:, :data_shape[1]], wgti, wlti)
        elif args.type in ("cpu"):
            sg_tmp, sl_tmp, sr_tmp = gw2s_cpu.gw2s_fft_mpi_cpu_PI_sr(
                -pre_factor / 2, ggi[:, :data_shape[1]], gli[:,
                                                             :data_shape[1]], gri[:, :data_shape[1]],
                wgi[:, :data_shape[1]], wli[:, :data_shape[1]], wri[:, :data_shape[1]], wgti[:, :data_shape[1]], wlti[:, :data_shape[1]], vh_pi, energy)
        else:
            raise ValueError("Argument error, input type not possible")
        sgo[:no_loc, :data_shape[1]] = sg_tmp
        slo[:no_loc, :data_shape[1]] = sl_tmp
        sro[:no_loc, :data_shape[1]] = sr_tmp

    # create communication wrapped functions
    greens_function = CommunicateCompute(distributions[:g_num_buffer],
                                         g_num_buffer, "c2r",
                                         g_col,
                                         g_row,
                                         buffer_compute_unblock=buffer_col_compute[:g_num_buffer],
                                         buffer_send_unblock=buffer_col_send[:g_num_buffer],
                                         buffer_recv_unblock=buffer_row_recv[:g_num_buffer],
                                         comm_unblock=comm_unblock,
                                         distributions_unblock=distributions_unblock_col[:g_num_buffer],
                                         batchsize=batchsize_col,
                                         iterations=iterations_unblock[1])(greens_function_compute)

    polarization = CommunicateCompute(distributions[:p_num_buffer],
                                      p_num_buffer, "r2c",
                                      p_row,
                                      p_col,
                                      buffer_compute_unblock=buffer_row_compute[:p_num_buffer],
                                      buffer_send_unblock=buffer_row_send[:p_num_buffer],
                                      buffer_recv_unblock=buffer_col_recv[:p_num_buffer],
                                      comm_unblock=comm_unblock,
                                      distributions_unblock=distributions_unblock_row[:p_num_buffer],
                                      batchsize=batchsize_row,
                                      iterations=iterations_unblock[0])(polarization_compute)

    screened_interaction = CommunicateCompute(distributions[:w_num_buffer],
                                              w_num_buffer, "c2r",
                                              w_col_tmp,
                                              w_row_tmp,
                                              buffer_compute_unblock=buffer_col_compute[:w_num_buffer],
                                              buffer_send_unblock=buffer_col_send[:w_num_buffer],
                                              buffer_recv_unblock=buffer_row_recv[:w_num_buffer],
                                              comm_unblock=comm_unblock,
                                              distributions_unblock=distributions_unblock_col[:w_num_buffer],
                                              batchsize=batchsize_col,
                                              iterations=iterations_unblock[1])(screened_interaction_compute)

    selfenergy = CommunicateCompute(distributions[:s_num_buffer],
                                    s_num_buffer, "r2c",
                                    s_row_tmp,
                                    s_col_tmp,
                                    buffer_compute_unblock=buffer_row_compute[:s_num_buffer],
                                    buffer_send_unblock=buffer_row_send[:s_num_buffer],
                                    buffer_recv_unblock=buffer_col_recv[:s_num_buffer],
                                    comm_unblock=comm_unblock,
                                    distributions_unblock=distributions_unblock_row[:s_num_buffer],
                                    batchsize=batchsize_row,
                                    iterations=iterations_unblock[0])(selfenergy_compute)

    if rank == 0:
        time_def_func += time.perf_counter()
        time_startup += time.perf_counter()
        time_loop = -time.perf_counter()

    for iter_num in range(gw_num_iter):

        # set observables to zero
        dos.fill(0.0)
        dosw.fill(0.0)
        nE.fill(0.0)
        nP.fill(0.0)
        nEw.fill(0.0)
        nPw.fill(0.0)
        ide.fill(0.0)

        if rank == 0:
            time_g = -time.perf_counter()

        # transform from 2D format to list/vector of sparse arrays format-----------
        s_col_vec = [change_format.sparse2vecsparse_v2(
            s_col[i][:, :data_shape[0]], rows, columns, nao) for i in range(gw_num_buffer)]

        # Adjusting Fermi Levels of both contacts to the current iteration band minima
        sr_ephn_col_vec = change_format.sparse2vecsparse_v2(np.zeros((distribution.count[1], no), dtype=base_type), rows,
                                                            columns, nao)

        ECmin_vec[iter_num + 1] = get_band_edge_mpi_interpol(ECmin_vec[iter_num],
                                                             energy,
                                                             hamiltonian_obj.Overlap["H_4"],
                                                             hamiltonian_obj.Hamiltonian["H_4"],
                                                             *(s_col_vec[::-1]),
                                                             sr_ephn_col_vec,
                                                             rows,
                                                             columns,
                                                             bmin,
                                                             bmax,
                                                             comm,
                                                             rank,
                                                             size,
                                                             distribution.counts,
                                                             distribution.displacements,
                                                             side="left")

        energy_fl = ECmin_vec[iter_num + 1] + dEfL_EC
        energy_fr = ECmin_vec[iter_num + 1] + dEfR_EC
        EFL_vec[iter_num + 1] = energy_fl
        EFR_vec[iter_num + 1] = energy_fr

        greens_function_inp_block = [*s_col_vec,
                                     energy_loc,
                                     dos[distribution.range_local[1]],
                                     nE[distribution.range_local[1]],
                                     nP[distribution.range_local[1]],
                                     ide[distribution.range_local[1]],
                                     factor_g_loc]
        greens_function_inp = [energy_fl, energy_fr]
        # greens_function.compute_communicate(greens_function_inp_block, greens_function_inp)
        greens_function(greens_function_inp_block, greens_function_inp)

        # only take part of the greens function

        # filter out peaks
        calc_GF_pool.h2g_observales_mpi(dos[range_local_no_padding[1]],
                                        nE[range_local_no_padding[1]],
                                        nP[range_local_no_padding[1]],
                                        flag_zeros[0:flag_end],
                                        comm, rank, size)
        flag_zeros_global = np.empty(
            distribution.shape[1], dtype=flag_zeros.dtype)
        distribution.gatherall_col(
            flag_zeros, flag_zeros_global, otype=flag_zeros.dtype)
        memory_mask = np.where(flag_zeros_global)[0]

        for i in range(g_num_buffer):
            g_row[i][:, memory_mask] = 0.0

        if rank == 0:
            time_g += time.perf_counter()
            time_p = -time.perf_counter()

        # calculate and communicate the polarization----------------------------------
        polarization_inp_block = [*g_row]
        polarization_inp = []
        polarization(polarization_inp_block, polarization_inp)

        if rank == 0:
            time_p += time.perf_counter()
            time_w = -time.perf_counter()

        # calculate and communicate the screened interaction----------------------------------
        screened_interaction_inp_block = [*p_col,
                                          energy_loc,
                                          dosw[distribution.range_local[1]],
                                          nEw[distribution.range_local[1]],
                                          nPw[distribution.range_local[1]],
                                          Idx_e_loc,
                                          factor_w_loc]
        screened_interaction_inp = []
        screened_interaction(screened_interaction_inp_block,
                             screened_interaction_inp)

        # filter out peaks
        flag_zeros = np.zeros(distribution.count[1])
        p2w_cpu.p2w_observales_mpi(dosw[range_local_no_padding[1]],
                                   nEw[range_local_no_padding[1]],
                                   nPw[range_local_no_padding[1]],
                                   flag_zeros[0:flag_end],
                                   comm, rank, size)
        flag_zeros_global = np.empty(
            distribution.shape[1], dtype=flag_zeros.dtype)
        distribution.gatherall_col(
            flag_zeros, flag_zeros_global, otype=flag_zeros.dtype)
        memory_mask = np.ones(distribution.shape[1], dtype=bool)
        memory_mask[np.where(flag_zeros_global)[0]] = False

        for i in range(w_num_buffer):
            w_row[i][:, memory_mask] = (
                1.0 - mem_w) * w_row_tmp[i][:, memory_mask] + mem_w * w_row[i][:, memory_mask]

        if rank == 0:
            time_w += time.perf_counter()
            time_s = -time.perf_counter()

        # compute and communicate the self-energy------------------------------------
        selfenergy_inp_block = [*g_row[:gw_num_buffer], *w_row, vh_loc]
        selfenergy_inp = []
        selfenergy(selfenergy_inp_block, selfenergy_inp)

        # only take part of the self energy
        for i in range(gw_num_buffer):
            s_col[i][:] = (1.0 - mem_s) * s_col_tmp[i] + mem_s * s_col[i]

        if rank == 0:
            time_s += time.perf_counter()

        # Wrapping up the iteration
        if rank == 0:
            comm.Reduce(MPI.IN_PLACE, dos, op=MPI.SUM, root=0)
            comm.Reduce(MPI.IN_PLACE, ide, op=MPI.SUM, root=0)

        else:
            comm.Reduce(dos, None, op=MPI.SUM, root=0)
            comm.Reduce(ide, None, op=MPI.SUM, root=0)

    if rank == 0:
        time_loop += time.perf_counter()
        time_end = -time.perf_counter()

    # communicate corrected g/w_col since data without filtering is communicated
    for gw_type in gw_types:
        for gw_name in ["g", "w"]:
            distribution.alltoall_r2c(matrices_loc_row[gw_name + gw_type],
                                      matrices_loc_col[gw_name + gw_type], transpose_net=args.net_transpose)

    # gather at master to test against gold solution------------------------------
    if rank == 0:
        # create buffers at master
        matrices_global = {gw_name + gw_type: np.empty(distribution.shape, dtype=distribution.base_type)
                           for gw_type in gw_types for gw_name in gw_names}
        for gw_type in gw_types:
            for gw_name in gw_names:
                distribution.gather_master(matrices_loc_col[gw_name + gw_type],
                                           matrices_global[gw_name + gw_type], transpose_net=args.net_transpose)
    else:
        # send time to master
        for gw_type in gw_types:
            for gw_name in gw_names:
                distribution.gather_master(
                    matrices_loc_col[gw_name + gw_type], None, transpose_net=args.net_transpose)

    # free datatypes------------------------------------------------------------
    distribution.free_datatypes()

    if rank == 0:
        time_end += time.perf_counter()
        print(f"Time Startup: {time_startup:.2f} s")
        print(f"Sub-Time Read Gold: {time_read_gold:.2f} s")
        print(f"Sub-Time Pre Compute: {time_pre_compute:.2f} s")
        print(f"Sub-Time Alloc Buffers: {time_alloc_buf:.2f} s")
        print(f"Sub-Time Def Func: {time_def_func:.2f} s")
        print(f"Time Loop: {time_loop:.2f} s")
        print(f"Sub-Time Greens Function: {time_g:.2f} s")
        print(f"Sub-Time Polarization: {time_p:.2f} s")
        print(f"Sub-Time Screened Interaction: {time_w:.2f} s")
        print(f"Sub-Time Self Energy: {time_s:.2f} s")
        print(f"Time End: {time_end:.2f} s")

    # if rank == 0:
    #     save_path = "/usr/scratch/mont-fort17/almaeder/test_gw/few_energy_iter2.mat"
    #     # todo remove padding from data to save
    #     read_solution.save_all(energy, rows, columns, bmax,
    #                            bmin, save_path, **matrices_global)

    # test against gold solution------------------------------------------------
    if rank == 0:
        # print difference to given solution
        difference = {gw_name + gw_type: np.linalg.norm(matrices_global[gw_name + gw_type][:data_shape[0], :data_shape[1]] -
                                                        matrices_gold[gw_name + gw_type]) for gw_type in gw_types for gw_name in gw_names}

        for gw_name in gw_names:
            print(
                gw_full_names[gw_name] + f" differences to Gold Solution g/l/r:  {difference[gw_name + gw_types[0]]:.4f}, {difference[gw_name + gw_types[1]]:.4f}, {difference[gw_name + gw_types[2]]:.4f}")

        # assert solution close to real solution
        abstol = 1e-2
        reltol = 1e-1
        for gw_type in gw_types:
            for gw_name in gw_names:
                assert difference[gw_name + gw_type] <= abstol + reltol * \
                    np.linalg.norm(matrices_gold[gw_name + gw_type])
                assert np.allclose(matrices_global[gw_name + gw_type][:data_shape[0], :data_shape[1]],
                                   matrices_gold[gw_name + gw_type], atol=1e-3, rtol=1e-3)
        print("The mpi implementation is correct")