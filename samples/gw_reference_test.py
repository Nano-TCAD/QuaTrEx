# Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.
"""
Example a sc-GW iteration with MPI+CUDA.
With transposition through network.
This application is for a tight-binding InAs nanowire. 
See the different GW step folders for more explanations.
"""
from mpi4py import MPI
import numpy as np
import os
from scipy import sparse
import time
import pickle
from quatrex.utils.communication import TransposeMatrix, CommunicateCompute
from quatrex.utils.matrix_creation import get_number_connected_blocks
from quatrex.utils import change_format
from quatrex.OMEN_structure_matrices.construct_CM import construct_coulomb_matrix
from quatrex.OMEN_structure_matrices import OMENHamClass
from quatrex.GW.gold_solution import read_solution
from quatrex.bandstructure.calc_band_edge import get_band_edge_mpi_interpol


# Import because of refactoring
from quatrex.utils.matrix_creation import initialize_block_G


# Refactored utils
from quatrex.refactored_solvers.fermi_distribution import fermi_distribution

# Refactored functions
from quatrex.refactored_solvers.greens_function_solver import greens_function_solver
from quatrex.refactored_solvers.screened_interaction_solver import screened_interaction_solver
from quatrex.refactored_solvers.polarization_solver import compute_polarization
from quatrex.refactored_solvers.gw_self_energy_solver import compute_gw_self_energy


if __name__ == "__main__":

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    proc_name = MPI.Get_processor_name()
    base_type = np.complex128
    gw_num_iter = 2
    is_padded = False
    comm_unblock = False
    save_result = False
    num_energy = 31
    # memory factors for Self-Energy, Green's Function and Screened interaction
    mem_s = 0.5
    mem_w = 0.1

    if rank == 0:
        time_startup = -time.perf_counter()
        time_read_gold = -time.perf_counter()

    save_path = "/usr/scratch/mont-fort17/almaeder/test_gw/few_energy_iter"

    # path to solution
    solution_path ="/usr/scratch/mont-fort17/dleonard/GW_paper/InAs"
    hamiltonian_path = solution_path

    # coulomb matrix path
    solution_path_vh = os.path.join(
        solution_path, "data_Vh_finalPI_InAs_0v.mat")

    # gw matrices path
    solution_path_gw = "/usr/scratch/mont-fort17/almaeder/test_gw/few_energy_iter" + str(gw_num_iter) + "_no_filter_new_matmult_correction.mat"


    # reading reference solution-------------------------------------------------
    gw_names = ["g", "p", "w", "s"]
    gw_types = ["g", "l"]
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
    rowsRef, columnsRef, Coulomb_matrix_flattened_reference = read_solution.load_v(solution_path_vh)

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
    
    # relative permittivity
    epsR = 2.5
    # DFT Conduction Band Minimum
    ECmin = 1.9346

    # Phyiscal Constants -----------
    # TODO: Move out in config file
    e = 1.6022e-19
    eps0 = 8.854e-12
    hbar = 1.0546e-34
    temperature_in_kelvin = 300
    boltzmann_constant = 1.38e-23
    elementary_charge = 1.6022e-19



    # Fermi Level to Band Edge Difference
    dEfL_EC = energy_fl - ECmin
    dEfR_EC = energy_fr - ECmin

    energy = np.linspace(-10.0, 5.0, num_energy, endpoint=True,
                         dtype=float)
    delta_energy = energy[1] - energy[0]
    pre_factor = -1.0j * delta_energy / (np.pi)
    hamiltonian_obj = OMENHamClass.Hamiltonian(
        solution_path, no_orb, Vappl=Vappl, rank=rank)
    # broadcast hamiltonian object
    serial_ham = pickle.dumps(hamiltonian_obj)
    broadcasted_ham = comm.bcast(serial_ham, root=0)
    hamiltonian_obj = pickle.loads(broadcasted_ham)

    # hamiltonian object has 1-based indexing
    bmax = hamiltonian_obj.Bmax - 1
    bmin = hamiltonian_obj.Bmin - 1

    blocksize = np.max(hamiltonian_obj.Bmax - hamiltonian_obj.Bmin + 1)
    number_of_orbitals = hamiltonian_obj.Hamiltonian['H_4'].shape[0]
    assert number_of_orbitals == hamiltonian_obj.Hamiltonian['H_4'].shape[1]
    assert rows.shape[0] == columns.shape[0]
    assert number_of_orbitals % blocksize == 0
    number_of_blocks = int(number_of_orbitals / blocksize)
    number_of_energy_points = energy.shape[0]
    number_of_nonzero_elements = rows.shape[0]

    
    data_shape = np.array([number_of_nonzero_elements,
                          number_of_energy_points], dtype=np.int32)

    Coulomb_matrix_reference = sparse.coo_array((Coulomb_matrix_flattened_reference,
                                                (np.squeeze(rowsRef), np.squeeze(columnsRef))),
                                                shape=(number_of_orbitals,
                                                        number_of_orbitals),
                                                dtype=base_type).tocsr()

    Coulomb_matrix = construct_coulomb_matrix(
        hamiltonian_obj, epsR, eps0, e, diag=False, orb_uniform=True)
    Coulomb_matrix_flattened = np.asarray(Coulomb_matrix[rows, columns].reshape(-1))

    assert np.allclose(Coulomb_matrix.toarray(), Coulomb_matrix_reference.toarray())

    ECmin_vec = np.concatenate((np.array([ECmin]), np.zeros(gw_num_iter)))
    EFL_vec = np.concatenate((np.array([energy_fl]), np.zeros(gw_num_iter)))
    EFR_vec = np.concatenate((np.array([energy_fr]), np.zeros(gw_num_iter)))


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
    # greater, lesser, retarded
    gw_num_buffer = 2
    # additional transposed
    g_num_buffer = 2
    p_num_buffer = 2
    w_num_buffer = 2
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

    vh_padded = np.zeros((distribution.shape[0]), dtype=Coulomb_matrix_reference.dtype)
    vh_padded[:data_shape[0]] = Coulomb_matrix_flattened

    # split up the factor between the ranks
    energy_loc = energy_padded[distribution.range_local[1]]
    Idx_e_loc = Idx_e_padded[distribution.range_local[1]]
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


    # define functions------------------------------------------------------------
    # in the following functions which will be wrapped with Communications
    # input argument order matters:
    #  1. input matrices which are ij/E sized (have to be sliced for nonblocking communication)
    #  2. other inputs which are "static" (not sliced for nonblocking communication)
    #  3. output matrices which are communicated

    def compute_greens_function(
        Sigma_greater,
        Sigma_lesser,
        Sigma_retarded,
        energy_points_per_rank,
        energy_fermi_left,
        energy_fermi_right,
        G_greater,
        G_lesser
    ):
        # calculate the green's function at every rank------------------------------
        number_of_energy_points_per_rank = energy_points_per_rank.shape[0]

        fermi_distribution_left = fermi_distribution(energy_points_per_rank,
                                                    energy_fermi_left,
                                                    elementary_charge,
                                                    boltzmann_constant,
                                                    temperature_in_kelvin)
        fermi_distribution_right = fermi_distribution(energy_points_per_rank,
                                                    energy_fermi_right,
                                                    elementary_charge,
                                                    boltzmann_constant,
                                                    temperature_in_kelvin)


        # cannot be put into the self-energy kernel because transposed is needed
        for ie in range(number_of_energy_points_per_rank):
            # symmetrize (lesser and greater have to be skewed symmetric)
            Sigma_lesser[ie] = (Sigma_lesser[ie] - Sigma_lesser[ie].T.conj()) / 2
            Sigma_greater[ie] = (Sigma_greater[ie] - Sigma_greater[ie].T.conj()) / 2
            Sigma_retarded[ie] = np.real(Sigma_retarded[ie]) + (Sigma_greater[ie] - Sigma_lesser[ie]) / 2


        G_lesser_diag_blocks = np.zeros((number_of_energy_points,
                                        number_of_blocks,
                                        blocksize,
                                        blocksize),
                                        dtype=np.complex128)
        G_lesser_upper_blocks = np.zeros((number_of_energy_points,
                                        number_of_blocks - 1,
                                        blocksize,
                                        blocksize),
                                        dtype=np.complex128)
        G_greater_diag_blocks = np.zeros((number_of_energy_points,
                                        number_of_blocks,
                                        blocksize,
                                        blocksize),
                                        dtype=np.complex128)
        G_greater_upper_blocks = np.zeros((number_of_energy_points,
                                        number_of_blocks - 1,
                                        blocksize,
                                        blocksize),
                                        dtype=np.complex128)


        greens_function_solver(G_lesser_diag_blocks,
                                G_lesser_upper_blocks,
                                G_greater_diag_blocks, 
                                G_greater_upper_blocks,
                                hamiltonian_obj.Hamiltonian['H_4'],
                                hamiltonian_obj.Overlap['H_4'],
                                Sigma_retarded,
                                Sigma_lesser,
                                Sigma_greater,
                                energy_points_per_rank,
                                fermi_distribution_left,
                                fermi_distribution_right,
                                blocksize)

        
        
        # lower diagonal blocks from physics identity
        G_greater_lower_blocks = -G_greater_upper_blocks.conjugate().transpose((0, 1, 3, 2))
        G_lesser_lower_blocks = -G_lesser_upper_blocks.conjugate().transpose((0, 1, 3, 2))

        G_greater[:number_of_energy_points_per_rank,
                  :number_of_nonzero_elements] = change_format.block2sparse_energy_alt(*map_,
                                                                             G_greater_diag_blocks,
                                                                             G_greater_upper_blocks,
                                                                             G_greater_lower_blocks,
                                                                             number_of_nonzero_elements,
                                                                             number_of_energy_points_per_rank,
                                                                             energy_contiguous=False)
        G_lesser[:number_of_energy_points_per_rank,
                 :number_of_nonzero_elements] = change_format.block2sparse_energy_alt(*map_,
                                                                             G_lesser_diag_blocks,
                                                                             G_lesser_upper_blocks,
                                                                             G_lesser_lower_blocks,
                                                                             number_of_nonzero_elements,
                                                                             number_of_energy_points_per_rank,
                                                                             energy_contiguous=False)



    # TODO: Delete/remove/replace
    def generator_rgf_Hamiltonian(E, DH, SigR):
        for i in range(E.shape[0]):
            yield (E[i] + 1j * 1e-12) * DH.Overlap['H_4'] - DH.Hamiltonian['H_4'] - SigR[i]

    # TODO: Delete/remove/replace
    def generator_rgf_currentdens_Hamiltonian(E, DH):
        for i in range(E.shape[0]):
            yield DH.Hamiltonian['H_4'] - (E[i]) * DH.Overlap['H_4']


    def polarization_compute(
        G_greater,
        G_lesser,
        number_of_energy_points,
        Polarization_greater,
        Polarization_lesser
    ):
        
        number_of_orbitals = G_greater.shape[0]

        # Input matrices can be padded, but the output matrices are not padded
        # Output matrices could be bigger for nonblocking communication than input matrices
        (Polarization_greater[:number_of_orbitals, :number_of_energy_points],
        Polarization_lesser[:number_of_orbitals, :number_of_energy_points]) = compute_polarization(
                                                            G_lesser[:, :number_of_energy_points],
                                                            G_greater[:, :number_of_energy_points],
                                                            delta_energy)
        

    def screened_interaction_compute(
        Polarization_greater,
        Polarization_lesser,
        energy_points_per_rank,
        Screened_interaction_greater_flattened,
        Screened_interaction_lesser_flattened
    ):

        number_of_energy_points_per_rank = energy_points_per_rank.shape[0]
        # transform from 2D format to list/vector of sparse arrays format-----------
        Polarization_greater_list = change_format.sparse2vecsparse_v2(
            Polarization_greater[:, :data_shape[0]], rows, columns, number_of_orbitals)
        Polarization_lesser_list = change_format.sparse2vecsparse_v2(
            Polarization_lesser[:, :data_shape[0]], rows, columns, number_of_orbitals)

        Polarization_retarded_list = []
        # Symmetrization of Polarization (TODO: check if this is needed)
        for ie in range(number_of_energy_points_per_rank):
            # Anti-Hermitian symmetrizing of PL and PG
            # pl[ie] = 1j * np.imag(pl[ie])
            Polarization_lesser_list[ie] = (Polarization_lesser_list[ie] - Polarization_lesser_list[ie].conj().T) / 2

            # pg[ie] = 1j * np.imag(pg[ie])
            Polarization_greater_list[ie] = (Polarization_greater_list[ie] - Polarization_greater_list[ie].conj().T) / 2
            
            # PR has to be derived from PL and PG and then has to be symmetrized
            Polarization_retarded_list.append((Polarization_greater_list[ie] - Polarization_lesser_list[ie]) / 2)
        
        number_of_blocks_after_matmul = int(number_of_blocks / nbc)
        blocksize_after_matmul = int(blocksize * nbc)
        # calculate the screened interaction on every rank--------------------------
        Screened_interaction_lesser_diag_blocks = np.zeros((number_of_energy_points,
                                                            number_of_blocks_after_matmul,
                                                            blocksize_after_matmul,
                                                            blocksize_after_matmul),
                                                            dtype=np.complex128)
        Screened_interaction_lesser_upper_blocks = np.zeros((number_of_energy_points,
                                                             number_of_blocks_after_matmul - 1,
                                                             blocksize_after_matmul,
                                                             blocksize_after_matmul),
                                                             dtype=np.complex128)
        Screened_interaction_greater_diag_blocks = np.zeros((number_of_energy_points,
                                                             number_of_blocks_after_matmul,
                                                             blocksize_after_matmul,
                                                             blocksize_after_matmul),
                                                             dtype=np.complex128)
        Screened_interaction_greater_upper_blocks = np.zeros((number_of_energy_points,
                                                              number_of_blocks_after_matmul - 1,
                                                              blocksize_after_matmul,
                                                              blocksize_after_matmul),
                                                              dtype=np.complex128)


        screened_interaction_solver(
            Screened_interaction_lesser_diag_blocks,
            Screened_interaction_lesser_upper_blocks,
            Screened_interaction_greater_diag_blocks,
            Screened_interaction_greater_upper_blocks,
            Coulomb_matrix,
            Polarization_greater_list,
            Polarization_lesser_list,
            Polarization_retarded_list,
            number_of_energy_points,
            blocksize,
        )

        # Flattening from [E, bblocks, blocksize, blocksize] -> [ij, E]
        
        # Production of lower blocks through symmetry from upper blocks computed
        # in RGF
        Screened_interaction_greater_lower_blocks = -Screened_interaction_greater_upper_blocks.conjugate().transpose((0, 1, 3, 2))
        Screened_interaction_lesser_lower_blocks = -Screened_interaction_lesser_upper_blocks.conjugate().transpose((0, 1, 3, 2))

        Screened_interaction_greater_flattened[:number_of_energy_points_per_rank, :data_shape[0]] = change_format.block2sparse_energy_alt(
            *map_mm_,
            Screened_interaction_greater_diag_blocks,
            Screened_interaction_greater_upper_blocks,
            Screened_interaction_greater_lower_blocks,
            number_of_nonzero_elements,
            number_of_energy_points_per_rank,
            energy_contiguous=False)
        Screened_interaction_lesser_flattened[:number_of_energy_points_per_rank, :data_shape[0]] = change_format.block2sparse_energy_alt(
            *map_mm_,
            Screened_interaction_lesser_diag_blocks,
            Screened_interaction_lesser_upper_blocks,
            Screened_interaction_lesser_lower_blocks,
            number_of_nonzero_elements,
            number_of_energy_points_per_rank,
            energy_contiguous=False)


    def selfenergy_compute(
            G_greater,
            G_lesser,
            Screened_interaction_greater,
            Screened_interaction_lesser,
            Coulomb_matrix,
            number_of_energy_points,
            Sigma_greater,
            Sigma_lesser,
            Sigma_retarded
    ):
        # todo optimize and not load two time green's function to gpu and do twice the fft
        number_of_orbitals = G_greater.shape[0]

        (Sigma_greater[:number_of_orbitals, :number_of_energy_points],
         Sigma_lesser[:number_of_orbitals, :number_of_energy_points],
         Sigma_retarded[:number_of_orbitals, :number_of_energy_points]) = compute_gw_self_energy(
            G_lesser[:, :number_of_energy_points],
            G_greater[:, :number_of_energy_points],
            Screened_interaction_lesser[:, :number_of_energy_points],
            Screened_interaction_greater[:, :number_of_energy_points],
            Coulomb_matrix,
            delta_energy)




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
                                         iterations=iterations_unblock[1])(compute_greens_function)

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
            s_col[i][:, :data_shape[0]], rows, columns, number_of_orbitals) for i in range(s_num_buffer)]

        # Adjusting Fermi Levels of both contacts to the current iteration band minima
        sr_ephn_col_vec = change_format.sparse2vecsparse_v2(np.zeros((distribution.count[1], number_of_nonzero_elements), dtype=base_type), rows,
                                                            columns, number_of_orbitals)

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

        greens_function_inputs_to_slice = [*s_col_vec,
                                     energy_loc]
        greens_function_inputs = [energy_fl, energy_fr]
        greens_function(greens_function_inputs_to_slice, greens_function_inputs)

        # only take part of the greens function


        if rank == 0:
            time_g += time.perf_counter()
            time_p = -time.perf_counter()

        # calculate and communicate the polarization----------------------------------
        polarization_inputs_to_slice = [*g_row]
        polarization_inputs = [data_shape[1]]
        polarization(polarization_inputs_to_slice, polarization_inputs)

        if rank == 0:
            time_p += time.perf_counter()
            time_w = -time.perf_counter()

        # calculate and communicate the screened interaction----------------------------------
        screened_interaction_inputs_to_slice = [*p_col,
                                          energy_loc]
        screened_interaction_inputs = []
        screened_interaction(screened_interaction_inputs_to_slice,
                             screened_interaction_inputs)


        for i in range(w_num_buffer):
            w_row[i][:, :] = (
                1.0 - mem_w) * w_row_tmp[i] + mem_w * w_row[i]

        if rank == 0:
            time_w += time.perf_counter()
            time_s = -time.perf_counter()

        # compute and communicate the self-energy------------------------------------
        selfenergy_inputs_to_slice = [*g_row[:g_num_buffer], *w_row, vh_loc]
        selfenergy_inputs = [data_shape[1]]
        selfenergy(selfenergy_inputs_to_slice, selfenergy_inputs)

        # only take part of the self energy
        for i in range(s_num_buffer):
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
                                      matrices_loc_col[gw_name + gw_type], transpose_net=False)

    # gather at master to test against gold solution------------------------------
    if rank == 0:
        # create buffers at master
        matrices_global = {gw_name + gw_type: np.empty(distribution.shape, dtype=distribution.base_type)
                           for gw_type in gw_types for gw_name in gw_names}
        for gw_type in gw_types:
            for gw_name in gw_names:
                distribution.gather_master(matrices_loc_col[gw_name + gw_type],
                                           matrices_global[gw_name + gw_type], transpose_net=False)
    else:
        # send time to master
        for gw_type in gw_types:
            for gw_name in gw_names:
                distribution.gather_master(
                    matrices_loc_col[gw_name + gw_type], None, transpose_net=False)

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

    if rank == 0 and save_result:
        # todo remove padding from data to save
        matrices_global_save = {}
        for item in matrices_global.items():
            matrices_global_save[item[0]
                                 ] = item[1][:data_shape[0], :data_shape[1]]
        matrices_global_save["Coulomb_matrix_reference"] = Coulomb_matrix_flattened
        read_solution.save_all(energy, rows, columns, bmax,
                               bmin, save_path, **matrices_global_save)


    # test against gold solution------------------------------------------------
    if rank == 0:
        # print difference to given solution
        difference = {gw_name + gw_type: np.linalg.norm(matrices_global[gw_name + gw_type][:data_shape[0], :data_shape[1]] -
                                                        matrices_gold[gw_name + gw_type]) for gw_type in gw_types for gw_name in gw_names}

        for gw_name in gw_names:
            print(
                gw_full_names[gw_name] + f" differences to Gold Solution g/l:  {difference[gw_name + gw_types[0]]:.4f}, {difference[gw_name + gw_types[1]]:.4f}")

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
