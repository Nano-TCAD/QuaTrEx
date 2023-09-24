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
import numpy.typing as npt
import os
import argparse
import pickle
from scipy import sparse
import time
from quatrex.utils.communication import TransposeMatrix, TransposeCompute
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
    name = MPI.Get_processor_name()
    base_type = np.complex128

    if rank == 0:
        time_startup = -time.perf_counter()
        time_read_gold = -time.perf_counter()

    # assume every rank has enough memory to read the initial data
    # path to solution
    scratch_path = "/usr/scratch/mont-fort17/dleonard/GW_paper/"
    solution_path = os.path.join(scratch_path, "InAs")

    solution_path_vh = os.path.join(solution_path, "data_Vh_finalPI_InAs_0v.mat")
    # solution_path_gw2 = os.path.join(solution_path, "data_GPWS_IEDM_memory2_GNR_04V.mat")

    solution_path_gw = os.path.join(solution_path, "data_GPWS_big_memory0_InAs_0V.mat")
    # solution_path_gw = "/usr/scratch/mont-fort17/almaeder/test_gw/few_energy.mat"

    hamiltonian_path = solution_path
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

    # create hamiltonian object
    # one orbital on C atoms, two same types
    no_orb = np.array([5, 5])
    Vappl = 0.4
    energy = np.linspace(-10.0, 5.0, 376, endpoint=True, dtype=float)  # Energy Vector
    Idx_e = np.arange(energy.shape[0])  # Energy Index Vector
    hamiltonian_obj = OMENHamClass.Hamiltonian(args.file_hm, no_orb, Vappl=Vappl, rank=rank)
    serial_ham = pickle.dumps(hamiltonian_obj)
    broadcasted_ham = comm.bcast(serial_ham, root=0)
    hamiltonian_obj = pickle.loads(broadcasted_ham)
    # Extract neighbor indices
    rows_g = hamiltonian_obj.rows
    columns_g = hamiltonian_obj.columns

    # hamiltonian object has 1-based indexing
    bmax = hamiltonian_obj.Bmax - 1
    bmin = hamiltonian_obj.Bmin - 1

    # reading reference solution
    energy_in, rows, columns, gg_gold, gl_gold, gr_gold = read_solution.load_x(solution_path_gw, "g")
    energy_in, rows_p, columns_p, pg_gold, pl_gold, pr_gold = read_solution.load_x(solution_path_gw, "p")
    energy_in, rows_w, columns_w, wg_gold, wl_gold, wr_gold = read_solution.load_x(solution_path_gw, "w")
    energy_in, rows_s, columns_s, sg_gold, sl_gold, sr_gold = read_solution.load_x(solution_path_gw, "s")
    rowsRef, columnsRef, vh_gold = read_solution.load_v(solution_path_vh)

    if rank == 0:
        time_read_gold += time.perf_counter()
        time_pre_compute = -time.perf_counter()

    ij2ji: npt.NDArray[np.int32] = change_format.find_idx_transposed(rows, columns)
    denergy: npt.NDArray[np.double] = energy[1] - energy[0]
    ne: np.int32 = np.int32(energy.shape[0])
    no: np.int32 = np.int32(columns.shape[0])
    pre_factor: base_type = -1.0j * denergy / (np.pi)
    nao: np.int64 = np.max(bmax) + 1

    vh = sparse.coo_array((vh_gold, (np.squeeze(rowsRef), np.squeeze(columnsRef))),
                          shape=(nao, nao),
                          dtype=base_type).tocsr()
    data_shape = np.array([rows.shape[0], energy.shape[0]], dtype=np.int32)

    # Creating the mask for the energy range of the deleted W elements given by the reference solution
    w_mask = np.ndarray(shape=(energy.shape[0], ), dtype=bool)

    wr_mask = np.sum(np.abs(wr_gold), axis=0) > 1e-10
    wl_mask = np.sum(np.abs(wl_gold), axis=0) > 1e-10
    wg_mask = np.sum(np.abs(wg_gold), axis=0) > 1e-10
    w_mask = np.logical_or(np.logical_or(wr_mask, wl_mask), wg_mask)

    map_diag, map_upper, map_lower = change_format.map_block2sparse_alt(rows, columns, bmax, bmin)

    # number of blocks
    nb = hamiltonian_obj.Bmin.shape[0]
    nbc = get_number_connected_blocks(hamiltonian_obj.NH, bmin, bmax, rows, columns)
    bmax_mm = bmax[nbc - 1:nb:nbc]
    bmin_mm = bmin[0:nb:nbc]

    map_diag_mm, map_upper_mm, map_lower_mm = change_format.map_block2sparse_alt(rows, columns, bmax_mm, bmin_mm)

    assert np.allclose(rows_p, rows)
    assert np.allclose(columns, columns_p)
    assert np.allclose(rows_w, rows)
    assert np.allclose(columns, columns_w)
    assert np.allclose(rows_s, rows)
    assert np.allclose(columns, columns_s)

    assert pg_gold.shape[0] == rows.shape[0]

    if rank == 0:
        # print size of data
        print(f"#Energy: {data_shape[1]} #nnz: {data_shape[0]}")

    # computation parameters----------------------------------------------------
    # set number of threads for the p2w step
    w_mkl_threads = 1
    w_worker_threads = 8
    # set number of threads for the h2g step
    gf_mkl_threads = 1
    gf_worker_threads = 8

    # physical parameter -----------

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

    # create the corresponding factor to mask
    # number of points to smooth the edges of the Green's Function
    dnp = 50
    factor_w = np.ones(ne)
    # factor_w[ne - dnp - 1:ne] = (np.cos(np.pi * np.linspace(0, 1, dnp + 1)) + 1) / 2
    # factor_w[np.where(np.invert(w_mask))[0]] = 0.0

    # create factor for the Green's Function
    factor_g = np.ones(ne)
    # factor_g[ne - dnp - 1:ne] = (np.cos(np.pi * np.linspace(0, 1, dnp + 1)) + 1) / 2
    # factor_g[0:dnp + 1] = (np.cos(np.pi * np.linspace(1, 0, dnp + 1)) + 1) / 2

    V_sparse = construct_coulomb_matrix(hamiltonian_obj, epsR, eps0, e, diag=False, orb_uniform=True)
    vh1d = np.asarray(vh[rows, columns].reshape(-1))

    assert np.allclose(V_sparse.toarray(), vh.toarray())

    # calculation of data distribution per rank---------------------------------
    is_padded = True
    comm_unblock = True

    padding = 0
    extra_elements = np.zeros_like(data_shape)

    if is_padded:
        padding = (size - data_shape % size) % size
        if rank == size-1:
            extra_elements[:] = padding

    data_shape_padded = data_shape + padding
    distribution = TransposeMatrix(comm, data_shape_padded, base_type=base_type)
    distribution_no_padded = TransposeMatrix(comm, data_shape, base_type=base_type)

    flag_zeros = np.zeros(distribution.count[1])
    energy_start = distribution.displacement[1]
    energy_end = min(distribution.displacement[1] + distribution.count[1], data_shape[1])
    flag_end = min(distribution.count[1], data_shape[1] - distribution.displacement[1])

    batchsize_row = distribution.count[0] // 2
    batchsize_col = distribution.count[1] // 2
    iteration_row = int(np.ceil(distribution.count[0] / batchsize_row))
    iteration_col = int(np.ceil(distribution.count[1] / batchsize_col))

    # batchsize_col = 3
    distribution_unblock_row = TransposeMatrix(comm, np.array(
        [batchsize_row*size, data_shape_padded[1]]), base_type=base_type)
    distribution_unblock_col = TransposeMatrix(comm, np.array(
        [data_shape_padded[0], batchsize_col*size]), base_type=base_type)

    energy_padded = np.zeros((distribution.shape[1]), dtype=energy.dtype)
    energy_padded[:data_shape[1]] = energy
    Idx_e_padded = np.zeros((distribution.shape[1]), dtype=Idx_e.dtype)
    Idx_e_padded[:data_shape[1]] = Idx_e
    factor_w_padded = np.zeros((distribution.shape[1]), dtype=factor_w.dtype)
    factor_w_padded[:data_shape[1]] = factor_w
    factor_g_padded = np.zeros((distribution.shape[1]), dtype=factor_g.dtype)
    factor_g_padded[:data_shape[1]] = factor_g

    # slice energy vector
    energy_loc = energy_padded[distribution.displacement[1]:distribution.displacement[1] + distribution.count[1]]
    Idx_e_loc = Idx_e_padded[distribution.displacement[1]:distribution.displacement[1] + distribution.count[1]]

    # split up the factor between the ranks
    factor_w_loc = factor_w_padded[distribution.displacement[1]:distribution.displacement[1] + distribution.count[1]]
    factor_g_loc = factor_g_padded[distribution.displacement[1]:distribution.displacement[1] + distribution.count[1]]

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

    # print rank distribution
    print(f"Rank: {rank} #Energy/rank: {distribution.count[1]} #nnz/rank: {distribution.count[0]}", name)

    # adding checks
    assert energy_loc.size == distribution.count[1]

    # initialize memory factors for Self-Energy, Green's Function and Screened interaction
    mem_s = 0.5
    mem_g = 0.0
    mem_w = 0.1
    # max number of iterations

    max_iter = 1
    ECmin_vec = np.concatenate((np.array([ECmin]), np.zeros(max_iter)))
    EFL_vec = np.concatenate((np.array([energy_fl]), np.zeros(max_iter)))
    EFR_vec = np.concatenate((np.array([energy_fr]), np.zeros(max_iter)))

    # Start and end index of the energy range
    ne_s = 0
    ne_f = 251

    # allocate memory
    if rank == 0:
        time_pre_compute += time.perf_counter()
        time_alloc_buf = -time.perf_counter()

    g_num_buffer = 4
    p_num_buffer = 3
    w_num_buffer = 5
    s_num_buffer = 3
    num_buffer = max(g_num_buffer, p_num_buffer, w_num_buffer, s_num_buffer)

    # create local buffers
    gg_col = np.zeros((distribution.count[1], distribution.shape[0]), dtype=base_type, order="C")
    gl_col = np.zeros((distribution.count[1], distribution.shape[0]), dtype=base_type, order="C")
    gr_col = np.zeros((distribution.count[1], distribution.shape[0]), dtype=base_type, order="C")
    glt_col = np.zeros((distribution.count[1], distribution.shape[0]), dtype=base_type, order="C")
    gg_row = np.empty((distribution.count[0], distribution.shape[1]), dtype=base_type, order="C")
    gl_row = np.empty((distribution.count[0], distribution.shape[1]), dtype=base_type, order="C")
    gr_row = np.empty((distribution.count[0], distribution.shape[1]), dtype=base_type, order="C")
    glt_row = np.empty((distribution.count[0], distribution.shape[1]), dtype=base_type, order="C")

    pg_col = np.empty((distribution.count[1], distribution.shape[0]), dtype=base_type, order="C")
    pl_col = np.empty((distribution.count[1], distribution.shape[0]), dtype=base_type, order="C")
    pr_col = np.empty((distribution.count[1], distribution.shape[0]), dtype=base_type, order="C")
    pg_row = np.empty((distribution.count[0], distribution.shape[1]), dtype=base_type, order="C")
    pl_row = np.empty((distribution.count[0], distribution.shape[1]), dtype=base_type, order="C")
    pr_row = np.empty((distribution.count[0], distribution.shape[1]), dtype=base_type, order="C")

    # only needed for testing
    wg_col = np.zeros((distribution.count[1], distribution.shape[0]), dtype=base_type, order="C")
    wl_col = np.zeros((distribution.count[1], distribution.shape[0]), dtype=base_type, order="C")
    wr_col = np.zeros((distribution.count[1], distribution.shape[0]), dtype=base_type, order="C")

    wg_col_tmp = np.zeros((distribution.count[1], distribution.shape[0]), dtype=base_type, order="C")
    wl_col_tmp = np.zeros((distribution.count[1], distribution.shape[0]), dtype=base_type, order="C")
    wr_col_tmp = np.zeros((distribution.count[1], distribution.shape[0]), dtype=base_type, order="C")
    wgt_col_tmp = np.zeros((distribution.count[1], distribution.shape[0]), dtype=base_type, order="C")
    wlt_col_tmp = np.zeros((distribution.count[1], distribution.shape[0]), dtype=base_type, order="C")

    wg_row = np.zeros((distribution.count[0], distribution.shape[1]), dtype=base_type, order="C")
    wl_row = np.zeros((distribution.count[0], distribution.shape[1]), dtype=base_type, order="C")
    wr_row = np.zeros((distribution.count[0], distribution.shape[1]), dtype=base_type, order="C")
    wgt_row = np.zeros((distribution.count[0], distribution.shape[1]), dtype=base_type, order="C")
    wlt_row = np.zeros((distribution.count[0], distribution.shape[1]), dtype=base_type, order="C")
    wg_row_tmp = np.zeros((distribution.count[0], distribution.shape[1]), dtype=base_type, order="C")
    wl_row_tmp = np.zeros((distribution.count[0], distribution.shape[1]), dtype=base_type, order="C")
    wr_row_tmp = np.zeros((distribution.count[0], distribution.shape[1]), dtype=base_type, order="C")
    wgt_row_tmp = np.zeros((distribution.count[0], distribution.shape[1]), dtype=base_type, order="C")
    wlt_row_tmp = np.zeros((distribution.count[0], distribution.shape[1]), dtype=base_type, order="C")

    sg_col = np.zeros((distribution.count[1], distribution.shape[0]), dtype=base_type, order="C")
    sl_col = np.zeros((distribution.count[1], distribution.shape[0]), dtype=base_type, order="C")
    sr_col = np.zeros((distribution.count[1], distribution.shape[0]), dtype=base_type, order="C")
    sg_row = np.empty((distribution.count[0], distribution.shape[1]), dtype=base_type, order="C")
    sl_row = np.empty((distribution.count[0], distribution.shape[1]), dtype=base_type, order="C")
    sr_row = np.empty((distribution.count[0], distribution.shape[1]), dtype=base_type, order="C")

    vh_padded = np.zeros((distribution.shape[0]), dtype=vh.dtype)
    vh_padded[:data_shape[0]] = np.asarray(vh[rows, columns].reshape(-1))
    vh_row = vh_padded[distribution.displacement[0]:distribution.displacement[0]+distribution.count[0]]

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
    if rank == 0:
        time_alloc_buf += time.perf_counter()
        time_def_func = -time.perf_counter()

    distributions = [distribution for _ in range(num_buffer)]
    distributions_unblock_row = [distribution_unblock_row for _ in range(num_buffer)]
    distributions_unblock_col = [distribution_unblock_col for _ in range(num_buffer)]
    iterations_unblock = [iteration_row, iteration_col]

    def greens_function_compute(sri_vec, sli_vec, sgi_vec,
                                gri, gli, ggi,
                                energy_loc_batch, dos_loc_batch, nE_loc_batch,
                                nP_loc_batch, ide_loc_batch, factor_g_loc_batch,
                                itern, energy_fermi_left, energy_fermi_right,
                                gro, glo, ggo, glto):
        # calculate the green's function at every rank------------------------------
        ne_loc = energy_loc_batch.shape[0]
        if args.pool:
            gr_diag, gr_upper, gl_diag, gl_upper, gg_diag, gg_upper = calc_GF_pool.calc_GF_pool_mpi_no_filter(
                hamiltonian_obj,
                energy_loc_batch,
                sri_vec,
                sli_vec,
                sgi_vec,
                energy_fermi_left,
                energy_fermi_right,
                temp,
                dos_loc_batch,
                nE_loc_batch,
                nP_loc_batch,
                ide_loc_batch,
                factor_g_loc_batch,
                homogenize=False,
                mkl_threads=gf_mkl_threads,
                worker_num=gf_worker_threads)
        else:
            raise ValueError("Argument error, I will remake this the other option later")

        # lower diagonal blocks from physics identity
        gg_lower = -gg_upper.conjugate().transpose((0, 1, 3, 2))
        gl_lower = -gl_upper.conjugate().transpose((0, 1, 3, 2))
        gr_lower = gr_upper.transpose((0, 1, 3, 2))

        if itern == 0:
            ggo[:ne_loc, :data_shape[0]] = change_format.block2sparse_energy_alt(map_diag,
                                                                                 map_upper,
                                                                                 map_lower,
                                                                                 gg_diag,
                                                                                 gg_upper,
                                                                                 gg_lower,
                                                                                 data_shape[0],
                                                                                 ne_loc,
                                                                                 energy_contiguous=False)
            glo[:ne_loc, :data_shape[0]] = change_format.block2sparse_energy_alt(map_diag,
                                                                                 map_upper,
                                                                                 map_lower,
                                                                                 gl_diag,
                                                                                 gl_upper,
                                                                                 gl_lower,
                                                                                 data_shape[0],
                                                                                 ne_loc,
                                                                                 energy_contiguous=False)
            gro[:ne_loc, :data_shape[0]] = change_format.block2sparse_energy_alt(map_diag,
                                                                                 map_upper,
                                                                                 map_lower,
                                                                                 gr_diag,
                                                                                 gr_upper,
                                                                                 gr_lower,
                                                                                 data_shape[0],
                                                                                 ne_loc,
                                                                                 energy_contiguous=False)
        else:
            # add new contribution to the Green's function
            ggo[:ne_loc, :data_shape[0]] = (1.0 - mem_g) * change_format.block2sparse_energy_alt(map_diag,
                                                                                                 map_upper,
                                                                                                 map_lower,
                                                                                                 gg_diag,
                                                                                                 gg_upper,
                                                                                                 gg_lower,
                                                                                                 data_shape[0],
                                                                                                 ne_loc,
                                                                                                 energy_contiguous=False) + mem_g * ggi[:ne_loc, :data_shape[0]]
            glo[:ne_loc, :data_shape[0]] = (1.0 - mem_g) * change_format.block2sparse_energy_alt(map_diag,
                                                                                                 map_upper,
                                                                                                 map_lower,
                                                                                                 gl_diag,
                                                                                                 gl_upper,
                                                                                                 gl_lower,
                                                                                                 data_shape[0],
                                                                                                 distribution.count[1],
                                                                                                 energy_contiguous=False) + mem_g * gli[:ne_loc, :data_shape[0]]
            gro[:ne_loc, :data_shape[0]] = (1.0 - mem_g) * change_format.block2sparse_energy_alt(map_diag,
                                                                                                 map_upper,
                                                                                                 map_lower,
                                                                                                 gr_diag,
                                                                                                 gr_upper,
                                                                                                 gr_lower,
                                                                                                 data_shape[0],
                                                                                                 ne_loc,
                                                                                                 energy_contiguous=False) + mem_g * gri[:ne_loc, :data_shape[0]]

        # calculate the transposed
        glto[:ne_loc, :data_shape[0]] = np.copy(glo[:ne_loc, :data_shape[0]][:, ij2ji], order="C")

    greens_function_direction = "c2r"
    greens_function_buffer_row = [gr_row, gl_row, gg_row, glt_row]
    greens_function_buffer_col = [gr_col, gl_col, gg_col, glt_col]

    greens_function = TransposeCompute(distributions[:g_num_buffer], greens_function_compute,
                                       g_num_buffer, greens_function_direction,
                                       comm_unblock=comm_unblock,
                                       distributions_unblock_row=distributions_unblock_row[:g_num_buffer],
                                       distributions_unblock_col=distributions_unblock_col[:g_num_buffer],
                                       batchsize_row=batchsize_row,
                                       batchsize_col=batchsize_col,
                                       iterations=iterations_unblock)

    greens_function.given_buffer(greens_function_buffer_row, greens_function_buffer_col,
                                 buffer_row_compute=buffer_row_compute[:g_num_buffer],
                                 buffer_col_compute=buffer_col_compute[:g_num_buffer],
                                 buffer_row_send=buffer_row_send[:g_num_buffer],
                                 buffer_col_send=buffer_col_send[:g_num_buffer],
                                 buffer_row_recv=buffer_row_recv[:g_num_buffer],
                                 buffer_col_recv=buffer_col_recv[:g_num_buffer])

    def polarization_compute(gri, gli, ggi, glti, pro, plo, pgo):
        no_loc = gri.shape[0]
        if args.type in ("gpu"):
            pgo[:no_loc, :data_shape[1]], plo[:no_loc, :data_shape[1]], pro[:no_loc, :data_shape[1]] = g2p_gpu.g2p_fft_mpi_gpu(
                pre_factor, ggi[:, :data_shape[1]], gli[:, :data_shape[1]], gri[:, :data_shape[1]], glti[:, :data_shape[1]])
        elif args.type in ("cpu"):
            pgo[:no_loc, :data_shape[1]], plo[:no_loc, :data_shape[1]], pro[:no_loc, :data_shape[1]] = g2p_cpu.g2p_fft_mpi_cpu_inlined(
                pre_factor, ggi[:, :data_shape[1]], gli[:, :data_shape[1]], gri[:, :data_shape[1]], glti[:, :data_shape[1]])
        else:
            raise ValueError("Argument error, input type not possible")

    polarization_direction = "r2c"
    polarization_buffer_row = [pr_row, pl_row, pg_row]
    polarization_buffer_col = [pr_col, pl_col, pg_col]

    polarization = TransposeCompute(distributions[:p_num_buffer], polarization_compute,
                                    p_num_buffer, polarization_direction,
                                    comm_unblock=comm_unblock,
                                    distributions_unblock_row=distributions_unblock_row[:p_num_buffer],
                                    distributions_unblock_col=distributions_unblock_col[:p_num_buffer],
                                    batchsize_row=batchsize_row,
                                    batchsize_col=batchsize_col,
                                    iterations=iterations_unblock)

    polarization.given_buffer(polarization_buffer_row, polarization_buffer_col,
                              buffer_row_compute=buffer_row_compute[:p_num_buffer],
                              buffer_col_compute=buffer_col_compute[:p_num_buffer],
                              buffer_row_send=buffer_row_send[:p_num_buffer],
                              buffer_col_send=buffer_col_send[:p_num_buffer],
                              buffer_row_recv=buffer_row_recv[:p_num_buffer],
                              buffer_col_recv=buffer_col_recv[:p_num_buffer])

    def screened_interaction_compute(pri, pli, pgi, energy_loc_batch,
                                     dosw_loc_batch, nEw_loc_batch, nPw_loc_batch,
                                     Idx_e_loc_batch, factor_w_loc_batch,
                                     wro, wlo, wgo, wlto, wgto):

        ne_loc = energy_loc_batch.shape[0]
        # transform from 2D format to list/vector of sparse arrays format-----------
        pg_col_vec = change_format.sparse2vecsparse_v2(pgi[:, :data_shape[0]], rows, columns, nao)
        pl_col_vec = change_format.sparse2vecsparse_v2(pli[:, :data_shape[0]], rows, columns, nao)
        pr_col_vec = change_format.sparse2vecsparse_v2(pri[:, :data_shape[0]], rows, columns, nao)

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
            raise ValueError("Argument error, I will remake this the other option later")

        # transform from block format to 2D format-----------------------------------
        # lower diagonal blocks from physics identity
        wg_lower = -wg_upper.conjugate().transpose((0, 1, 3, 2))
        wl_lower = -wl_upper.conjugate().transpose((0, 1, 3, 2))
        wr_lower = wr_upper.transpose((0, 1, 3, 2))

        wgo[:ne_loc, :data_shape[0]] = change_format.block2sparse_energy_alt(
            map_diag_mm,
            map_upper_mm,
            map_lower_mm,
            wg_diag,
            wg_upper,
            wg_lower,
            no,
            ne_loc,
            energy_contiguous=False)
        wlo[:ne_loc, :data_shape[0]] = change_format.block2sparse_energy_alt(
            map_diag_mm,
            map_upper_mm,
            map_lower_mm,
            wl_diag,
            wl_upper,
            wl_lower,
            no,
            ne_loc,
            energy_contiguous=False)
        wro[:ne_loc, :data_shape[0]] = change_format.block2sparse_energy_alt(
            map_diag_mm,
            map_upper_mm,
            map_lower_mm,
            wr_diag,
            wr_upper,
            wr_lower,
            no,
            ne_loc,
            energy_contiguous=False)

        # distribute screened interaction according to gw2s step--------------------

        # calculate the transposed
        wgto[:ne_loc, :data_shape[0]] = np.copy(wgo[:ne_loc, :data_shape[0]][:, ij2ji], order="C")
        wlto[:ne_loc, :data_shape[0]] = np.copy(wlo[:ne_loc, :data_shape[0]][:, ij2ji], order="C")

    screened_interaction_direction = "c2r"
    screened_interaction_buffer_row = [wr_row_tmp, wl_row_tmp,
                                       wg_row_tmp, wlt_row_tmp, wgt_row_tmp]
    screened_interaction_buffer_col = [wr_col_tmp, wl_col_tmp, wg_col_tmp, wlt_col_tmp, wgt_col_tmp]
    screened_interaction = TransposeCompute(distributions[:w_num_buffer], screened_interaction_compute,
                                            w_num_buffer, screened_interaction_direction,
                                            comm_unblock=comm_unblock,
                                            distributions_unblock_row=distributions_unblock_row[:w_num_buffer],
                                            distributions_unblock_col=distributions_unblock_col[:w_num_buffer],
                                            batchsize_row=batchsize_row,
                                            batchsize_col=batchsize_col,
                                            iterations=iterations_unblock)

    screened_interaction.given_buffer(screened_interaction_buffer_row, screened_interaction_buffer_col,
                                      buffer_row_compute=buffer_row_compute[:w_num_buffer],
                                      buffer_col_compute=buffer_col_compute[:w_num_buffer],
                                      buffer_row_send=buffer_row_send[:w_num_buffer],
                                      buffer_col_send=buffer_col_send[:w_num_buffer],
                                      buffer_row_recv=buffer_row_recv[:w_num_buffer],
                                      buffer_col_recv=buffer_col_recv[:w_num_buffer])

    def selfenergy_compute(gri, gli, ggi, wri, wli, wgi, wlti, wgti, sri, sli, sgi, vh_pi,  sro, slo, sgo):
        # todo optimize and not load two time green's function to gpu and do twice the fft
        no_loc = gri.shape[0]
        if args.type in ("gpu"):
            sg_tmp, sl_tmp, sr_tmp = gw2s_gpu.gw2s_fft_mpi_gpu_3part_sr(
                -pre_factor / 2, ggi[:, :data_shape[1]], gli[:, :data_shape[1]], gri[:, :data_shape[1]],
                wgi[:, :data_shape[1]], wli[:, :data_shape[1]], wri[:, :data_shape[1]], wgti, wlti)
        elif args.type in ("cpu"):
            sg_tmp, sl_tmp, sr_tmp = gw2s_cpu.gw2s_fft_mpi_cpu_PI_sr(
                -pre_factor / 2, ggi[:, :data_shape[1]], gli[:, :data_shape[1]], gri[:, :data_shape[1]],
                wgi[:, :data_shape[1]], wli[:, :data_shape[1]], wri[:, :data_shape[1]], wgti[:, :data_shape[1]], wlti[:, :data_shape[1]], vh_pi, energy)
        else:
            raise ValueError("Argument error, input type not possible")
        sgo[:no_loc, :data_shape[1]] = (1.0 - mem_s) * sg_tmp + mem_s * sgi[:no_loc, :data_shape[1]]
        slo[:no_loc, :data_shape[1]] = (1.0 - mem_s) * sl_tmp + mem_s * sli[:no_loc, :data_shape[1]]
        sro[:no_loc, :data_shape[1]] = (1.0 - mem_s) * sr_tmp + mem_s * sri[:no_loc, :data_shape[1]]

    selfenergy_direction = "r2c"
    selfenergy_buffer_row = [sr_row, sl_row, sg_row]
    selfenergy_buffer_col = [sr_col, sl_col, sg_col]

    selfenergy = TransposeCompute(distributions[:s_num_buffer], selfenergy_compute,
                                  s_num_buffer, selfenergy_direction,
                                  comm_unblock=comm_unblock,
                                  distributions_unblock_row=distributions_unblock_row[:s_num_buffer],
                                  distributions_unblock_col=distributions_unblock_col[:s_num_buffer],
                                  batchsize_row=batchsize_row,
                                  batchsize_col=batchsize_col,
                                  iterations=iterations_unblock)

    selfenergy.given_buffer(selfenergy_buffer_row, selfenergy_buffer_col,
                            buffer_row_compute=buffer_row_compute[:s_num_buffer],
                            buffer_col_compute=buffer_col_compute[:s_num_buffer],
                            buffer_row_send=buffer_row_send[:s_num_buffer],
                            buffer_col_send=buffer_col_send[:s_num_buffer],
                            buffer_row_recv=buffer_row_recv[:s_num_buffer],
                            buffer_col_recv=buffer_col_recv[:s_num_buffer])

    if rank == 0:
        time_def_func += time.perf_counter()
        time_startup += time.perf_counter()
        time_loop = -time.perf_counter()

    for iter_num in range(max_iter):

        # set observables to zero
        dos.fill(0.0)
        dosw.fill(0.0)
        nE.fill(0.0)
        nP.fill(0.0)
        nEw.fill(0.0)
        nPw.fill(0.0)
        ide.fill(0.0)

        # transform from 2D format to list/vector of sparse arrays format-----------

        sg_col_vec = change_format.sparse2vecsparse_v2(sg_col[:, :data_shape[0]], rows, columns, nao)
        sl_col_vec = change_format.sparse2vecsparse_v2(sl_col[:, :data_shape[0]], rows, columns, nao)
        sr_col_vec = change_format.sparse2vecsparse_v2(sr_col[:, :data_shape[0]], rows, columns, nao)

        # Adjusting Fermi Levels of both contacts to the current iteration band minima
        sr_ephn_col_vec = change_format.sparse2vecsparse_v2(np.zeros((distribution.count[1], no), dtype=base_type), rows,
                                                            columns, nao)

        ECmin_vec[iter_num + 1] = get_band_edge_mpi_interpol(ECmin_vec[iter_num],
                                                             energy,
                                                             hamiltonian_obj.Overlap["H_4"],
                                                             hamiltonian_obj.Hamiltonian["H_4"],
                                                             sr_col_vec,
                                                             sl_col_vec,
                                                             sg_col_vec,
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

        greens_function_inp_block = [sr_col_vec, sl_col_vec, sg_col_vec,
                                     gr_row, gl_row, gg_row,
                                     energy_loc,
                                     dos[distribution.displacement[1]:distribution.displacement[1] + distribution.count[1]],
                                     nE[distribution.displacement[1]:distribution.displacement[1] + distribution.count[1]],
                                     nP[distribution.displacement[1]:distribution.displacement[1] + distribution.count[1]],
                                     ide[distribution.displacement[1]:distribution.displacement[1] + distribution.count[1]],
                                     factor_g_loc]
        greens_function_inp = [iter_num, energy_fl, energy_fr]
        # greens_function.compute_communicate(greens_function_inp_block, greens_function_inp)
        greens_function.compute_communicate(greens_function_inp_block, greens_function_inp)

        # filter out peaks
        calc_GF_pool.h2g_observales_mpi(dos[energy_start:energy_end],
                                        nE[energy_start:energy_end],
                                        nP[energy_start:energy_end],
                                        flag_zeros[0:flag_end],
                                        comm, rank, size)
        flag_zeros_global = np.empty(distribution.shape[1], dtype=flag_zeros.dtype)
        distribution.gatherall_col(flag_zeros, flag_zeros_global, otype=flag_zeros.dtype)
        memory_mask = np.where(flag_zeros_global)[0]

        gg_row[:, memory_mask] = 0.0
        gl_row[:, memory_mask] = 0.0
        gr_row[:, memory_mask] = 0.0
        glt_row[:, memory_mask] = 0.0

        # calculate and communicate the polarization----------------------------------
        polarization_inp_block = [gr_row, gl_row, gg_row, glt_row]
        polarization_inp = []
        polarization.compute_communicate(polarization_inp_block, polarization_inp)

        # calculate and communicate the screened interaction----------------------------------
        screened_interaction_inp_block = [pr_col, pl_col, pg_col,
                                          energy_loc,
                                          dosw[distribution.displacement[1]:distribution.displacement[1] + distribution.count[1]],
                                          nEw[distribution.displacement[1]:distribution.displacement[1] + distribution.count[1]],
                                          nPw[distribution.displacement[1]:distribution.displacement[1] + distribution.count[1]],
                                          Idx_e_loc,
                                          factor_w_loc]
        screened_interaction_inp = []
        screened_interaction.compute_communicate(screened_interaction_inp_block, screened_interaction_inp)

        # filter out peaks
        flag_zeros = np.zeros(distribution.count[1])
        p2w_cpu.p2w_observales_mpi(dosw[energy_start:energy_end],
                                   nEw[energy_start:energy_end],
                                   nPw[energy_start:energy_end],
                                   flag_zeros[0:flag_end],
                                   comm, rank, size)
        flag_zeros_global = np.empty(distribution.shape[1], dtype=flag_zeros.dtype)
        distribution.gatherall_col(flag_zeros, flag_zeros_global, otype=flag_zeros.dtype)
        memory_mask = np.ones(distribution.shape[1], dtype=bool)
        memory_mask[np.where(flag_zeros_global)[0]] = False

        wg_row[:, memory_mask] = (1.0 - mem_w) * wg_row_tmp[:, memory_mask] + mem_w * wg_row[:, memory_mask]
        wl_row[:, memory_mask] = (1.0 - mem_w) * wl_row_tmp[:, memory_mask] + mem_w * wl_row[:, memory_mask]
        wr_row[:, memory_mask] = (1.0 - mem_w) * wr_row_tmp[:, memory_mask] + mem_w * wr_row[:, memory_mask]
        wgt_row[:, memory_mask] = (1.0 - mem_w) * wgt_row_tmp[:,
                                                              memory_mask] + mem_w * wgt_row[:, memory_mask]
        wlt_row[:, memory_mask] = (1.0 - mem_w) * wlt_row_tmp[:,
                                                              memory_mask] + mem_w * wlt_row[:, memory_mask]

        # compute and communicate the self-energy------------------------------------
        selfenergy_inp_block = [gr_row, gl_row, gg_row, wr_row, wl_row,
                                wg_row, wlt_row, wgt_row, sr_row, sl_row, sg_row, vh_row]
        selfenergy_inp = []
        selfenergy.compute_communicate(selfenergy_inp_block, selfenergy_inp)

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
    distribution.alltoall_r2c(gg_row, gg_col, transpose_net=args.net_transpose)
    distribution.alltoall_r2c(gl_row, gl_col, transpose_net=args.net_transpose)
    distribution.alltoall_r2c(gr_row, gr_col, transpose_net=args.net_transpose)
    distribution.alltoall_r2c(wg_row, wg_col, transpose_net=args.net_transpose)
    distribution.alltoall_r2c(wl_row, wl_col, transpose_net=args.net_transpose)
    distribution.alltoall_r2c(wr_row, wr_col, transpose_net=args.net_transpose)

    # gather at master to test against gold solution------------------------------
    matrices_loc = [gg_col, gl_col, gr_col, pg_col, pl_col, pr_col, wg_col, wl_col, wr_col, sg_col, sl_col, sr_col]
    matrices_gold = [gg_gold, gl_gold, gr_gold, pg_gold, pl_gold,
                     pr_gold, wg_gold, wl_gold, wr_gold, sg_gold, sl_gold, sr_gold]

    names = ["Greens Function", "Polarization", "Screened Interaction", "Self-Energy"]
    if rank == 0:
        # create buffers at master
        matrices_global = [np.empty(distribution.shape, dtype=distribution.base_type) for _ in range(len(matrices_loc))]
        for i in range(len(matrices_loc)):
            distribution.gather_master(matrices_loc[i], matrices_global[i], transpose_net=args.net_transpose)

    else:
        # send time to master
        for i in range(len(matrices_loc)):
            distribution.gather_master(matrices_loc[i], None, transpose_net=args.net_transpose)

    # if rank == 0:
    #     save_path = "/usr/scratch/mont-fort17/almaeder/test_gw/few_energy.mat"
    #     read_solution.save_all(energy, rows, columns, *matrices_global, bmax, bmin, save_path)

    # test against gold solution------------------------------------------------
    if rank == 0:
        # print difference to given solution
        # use Frobenius norm
        difference = [np.linalg.norm(matrices_global[i][:data_shape[0], :data_shape[1]] -
                                     matrices_gold[i]) for i in range(len(matrices_loc))]

        for i in range(len(matrices_loc)//3):
            print(
                names[i] + f" differences to Gold Solution g/l/r:  {difference[3*i]:.4f}, {difference[3*i+1]:.4f}, {difference[3*i+2]:.4f}")

        # assert solution close to real solution
        abstol = 1e-2
        reltol = 1e-1
        for i in range(len(matrices_loc)):
            assert difference[i] <= abstol + reltol * np.linalg.norm(matrices_gold[i])
            assert np.allclose(matrices_global[i][:data_shape[0], :data_shape[1]],
                               matrices_gold[i], atol=1e-3, rtol=1e-3)
        print("The mpi implementation is correct")

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
        print(f"Time End: {time_end:.2f} s")
