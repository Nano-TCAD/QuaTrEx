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
from quatrex.OMEN_structure_matrices.construct_CM import construct_coulomb_matrix
from quatrex.OMEN_structure_matrices import OMENHamClass
from quatrex.refactored_utils.read_solution import load_a_gw_matrix_flattened, load_coulomb_matrix_flattened, save_all
from quatrex.bandstructure.calc_band_edge import get_band_edge_mpi_interpol



# Refactored utils
from quatrex.refactored_utils.fermi_distribution import fermi_distribution
from quatrex.refactored_utils.utils import flattened_to_list_of_csr, map_triple_array_to_flattened, triple_array_to_flattened


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

    save_path = "/usr/scratch/mont-fort17/almaeder/test_gw/reference_iter.mat"

    # path to hamiltonian
    hamiltonian_path ="/usr/scratch/mont-fort17/dleonard/GW_paper/InAs"

    # gw matrices path
    reference_path = "/usr/scratch/mont-fort17/almaeder/test_gw/reference_iter" + str(gw_num_iter) + ".mat"

    # reading reference solution-------------------------------------------------
    gw_names = ["g", "p", "w", "s"]
    gw_types = ["g", "l"]
    gw_full_names = {"g": "Greens Function", "p": "Polarization",
                     "w": "Screened Interaction", "s": "Self-Energy"}
    matrices_gold = {}
    for gw_name in gw_names:
        _, rows, columns, g_gold, l_gold = load_a_gw_matrix_flattened(
            reference_path, gw_name)
        matrices_gold[gw_name + "g"] = g_gold
        matrices_gold[gw_name + "l"] = l_gold
        matrices_gold[gw_name + "rows"] = rows
        matrices_gold[gw_name + "columns"] = columns
    for gw_name in gw_names:
        assert np.all(matrices_gold[gw_name + "rows"] == matrices_gold["grows"])
        assert np.all(matrices_gold[gw_name + "columns"]
                      == matrices_gold["gcolumns"])

    rows = matrices_gold["grows"]
    columns = matrices_gold["gcolumns"]
    rowsRef, columnsRef, Coulomb_matrix_flattened_reference = load_coulomb_matrix_flattened(reference_path)

    if rank == 0:
        time_read_gold += time.perf_counter()
        time_pre_compute = -time.perf_counter()

    # physics parameters---------------------------------------------------------
    # Phyiscal Constants -----------
    # TODO: Move out in config file
    elementary_charge = 1.6022e-19
    vacuum_permittivity = 8.854e-12
    hbar = 1.0546e-34
    temperature_in_kelvin = 300
    boltzmann_constant = 1.38e-23

    
    
    # create hamiltonian object
    # one orbital on C atoms, two same types
    number_of_orbital_per_atom = np.array([5, 5])
    voltage_applied = 0.4
    # Fermi Level of Left Contact
    energy_fermi_left = 1.9
    # Fermi Level of Right Contact
    energy_fermi_right = energy_fermi_left - voltage_applied

    # relative permittivity
    relative_permittivity = 2.5
    # DFT Conduction Band Minimum
    energy_conduction_band_minimum = 1.9346

    # Fermi Level to Band Edge Difference
    energy_difference_fermi_minimum_left = energy_fermi_left - energy_conduction_band_minimum
    energy_difference_fermi_minimum_right = energy_fermi_right - energy_conduction_band_minimum

    energy_points = np.linspace(-10.0, 5.0, num_energy, endpoint=True, dtype=float)
    delta_energy = energy_points[1] - energy_points[0]

    # TODO: refactor
    hamiltonian_obj = OMENHamClass.Hamiltonian(
        hamiltonian_path, number_of_orbital_per_atom, Vappl=voltage_applied, rank=rank)

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
    number_of_energy_points = energy_points.shape[0]
    number_of_nonzero_elements = rows.shape[0]

    
    data_shape = np.array([number_of_nonzero_elements,
                          number_of_energy_points], dtype=np.int32)

    Coulomb_matrix_reference = sparse.coo_array((Coulomb_matrix_flattened_reference,
                                                (np.squeeze(rowsRef), np.squeeze(columnsRef))),
                                                shape=(number_of_orbitals,
                                                        number_of_orbitals),
                                                dtype=base_type).tocsr()

    # TODO refactor
    Coulomb_matrix = construct_coulomb_matrix(
        hamiltonian_obj, relative_permittivity, vacuum_permittivity, elementary_charge, diag=False, orb_uniform=True)
    Coulomb_matrix_flattened = np.asarray(Coulomb_matrix[rows, columns].reshape(-1))

    assert np.allclose(Coulomb_matrix.toarray(), Coulomb_matrix_reference.toarray())

    energy_conduction_band_minimum_over_iterations = np.concatenate((np.array([energy_conduction_band_minimum]), np.zeros(gw_num_iter)))
    energy_fermi_left_over_iterations = np.concatenate((np.array([energy_fermi_left]), np.zeros(gw_num_iter)))
    energy_fermi_right_over_iterations = np.concatenate((np.array([energy_fermi_right]), np.zeros(gw_num_iter)))


    # maps to transform from _blocks to _flattened
    map_blocks_to_flattened = map_triple_array_to_flattened(rows,
                                                            columns,
                                                            number_of_blocks,
                                                            blocksize)

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


    batchsize_nonzero_elements_per_rank = distribution.count[0] // 2
    batchsize_energy_per_rank = distribution.count[1] // 2
    iteration_nonzero_elements_per_rank = int(np.ceil(distribution.count[0] / batchsize_nonzero_elements_per_rank))
    iteration_energy_per_rank = int(np.ceil(distribution.count[1] / batchsize_energy_per_rank))
    iterations_unblock = [iteration_nonzero_elements_per_rank, iteration_energy_per_rank]

    distribution_unblock_nonzero_elements_per_rank = TransposeMatrix(comm, np.array(
        [batchsize_nonzero_elements_per_rank*size, data_shape_padded[1]]), base_type=base_type)
    distribution_unblock_energy_per_rank = TransposeMatrix(comm, np.array(
        [data_shape_padded[0], batchsize_energy_per_rank*size]), base_type=base_type)

    distributions = [distribution for _ in range(num_buffer)]

    distributions_unblock_nonzero_elements_per_rank = [
        distribution_unblock_nonzero_elements_per_rank for _ in range(num_buffer)]

    distributions_unblock_energy_per_rank = [
        distribution_unblock_energy_per_rank for _ in range(num_buffer)]

    
    number_of_energy_points_padded = distribution.shape[1]
    number_of_nonzero_elements_padded = distribution.shape[0]
    number_of_energy_points_per_rank = distribution.count[1]
    number_of_nonzero_elements_per_rank = distribution.count[0]



    # padding of 1D arrays----------------------------------------------------

    energy_points_padded = np.zeros((distribution.shape[1]), dtype=energy_points.dtype)
    energy_points_padded[:data_shape[1]] = energy_points

    Coulomb_matrix_padded = np.zeros((distribution.shape[0]), dtype=Coulomb_matrix_reference.dtype)
    Coulomb_matrix_padded[:number_of_nonzero_elements] = Coulomb_matrix_flattened

    # split up the factor between the ranks
    energy_points_per_rank = energy_points_padded[distribution.range_local[1]]
    Coulomb_matrix_per_rank = Coulomb_matrix_padded[distribution.range_local[0]]

    # print rank distribution
    print(
        f"Rank: {rank} #Energy/rank: {number_of_energy_points_per_rank} #nnz/rank: {number_of_nonzero_elements_per_rank}", proc_name)

    if rank == 0:
        time_pre_compute += time.perf_counter()
        time_alloc_buf = -time.perf_counter()

    # create local buffers----------------------------------------------------------
    g_energy_per_rank = [np.zeros((number_of_energy_points_per_rank, distribution.shape[0]),
                      dtype=base_type, order="C") for _ in range(g_num_buffer)]
    p_energy_per_rank = [np.zeros((number_of_energy_points_per_rank, distribution.shape[0]),
                      dtype=base_type, order="C") for _ in range(p_num_buffer)]
    w_energy_per_rank = [np.zeros((number_of_energy_points_per_rank, distribution.shape[0]),
                      dtype=base_type, order="C") for _ in range(w_num_buffer)]
    w_energy_per_rank_tmp = [np.zeros((number_of_energy_points_per_rank, distribution.shape[0]),
                          dtype=base_type, order="C") for _ in range(w_num_buffer)]
    s_energy_per_rank = [np.zeros((number_of_energy_points_per_rank, distribution.shape[0]),
                      dtype=base_type, order="C") for _ in range(s_num_buffer)]
    s_energy_per_rank_tmp = [np.zeros((number_of_energy_points_per_rank, distribution.shape[0]),
                          dtype=base_type, order="C") for _ in range(s_num_buffer)]

    g_nonzero_elements_per_rank = [np.zeros((number_of_nonzero_elements_per_rank, distribution.shape[1]),
                      dtype=base_type, order="C") for _ in range(g_num_buffer)]
    p_nonzero_elements_per_rank = [np.zeros((number_of_nonzero_elements_per_rank, distribution.shape[1]),
                      dtype=base_type, order="C") for _ in range(p_num_buffer)]
    w_nonzero_elements_per_rank = [np.zeros((number_of_nonzero_elements_per_rank, distribution.shape[1]),
                      dtype=base_type, order="C") for _ in range(w_num_buffer)]
    w_nonzero_elements_per_rank_tmp = [np.zeros((number_of_nonzero_elements_per_rank, distribution.shape[1]),
                          dtype=base_type, order="C") for _ in range(w_num_buffer)]
    s_nonzero_elements_per_rank = [np.zeros((number_of_nonzero_elements_per_rank, distribution.shape[1]),
                      dtype=base_type, order="C") for _ in range(s_num_buffer)]
    s_nonzero_elements_per_rank_tmp = [np.zeros((number_of_nonzero_elements_per_rank, distribution.shape[1]),
                          dtype=base_type, order="C") for _ in range(s_num_buffer)]

    buffer_nonzero_elements_per_rank_compute = [np.empty((distribution_unblock_nonzero_elements_per_rank.count[0],
                                    distribution_unblock_nonzero_elements_per_rank.shape[1]), dtype=base_type, order="C") for _ in range(num_buffer)]
    buffer_nonzero_elements_per_rank_send = [np.empty((distribution_unblock_nonzero_elements_per_rank.count[0],
                                 distribution_unblock_nonzero_elements_per_rank.shape[1]), dtype=base_type, order="C") for _ in range(num_buffer)]
    buffer_energy_per_rank_recv = [np.empty((distribution_unblock_nonzero_elements_per_rank.count[1],
                                 distribution_unblock_nonzero_elements_per_rank.shape[0]), dtype=base_type, order="C") for _ in range(num_buffer)]

    buffer_energy_per_rank_compute = [np.empty((distribution_unblock_energy_per_rank.count[1],
                                    distribution_unblock_energy_per_rank.shape[0]), dtype=base_type, order="C") for _ in range(num_buffer)]
    buffer_energy_per_rank_send = [np.empty((distribution_unblock_energy_per_rank.count[1],
                                 distribution_unblock_energy_per_rank.shape[0]), dtype=base_type, order="C") for _ in range(num_buffer)]
    buffer_nonzero_elements_per_rank_recv = [np.empty((distribution_unblock_energy_per_rank.count[0],
                                 distribution_unblock_energy_per_rank.shape[1]), dtype=base_type, order="C") for _ in range(num_buffer)]

    matrices_nonzero_elements_per_rank = {}
    matrices_energy_per_rank = {}
    # transposed arrays are not compared in the end
    for i in range(gw_num_buffer):
        matrices_nonzero_elements_per_rank[gw_names[0] + gw_types[i]] = g_nonzero_elements_per_rank[i]
        matrices_nonzero_elements_per_rank[gw_names[1] + gw_types[i]] = p_nonzero_elements_per_rank[i]
        matrices_nonzero_elements_per_rank[gw_names[2] + gw_types[i]] = w_nonzero_elements_per_rank[i]
        matrices_nonzero_elements_per_rank[gw_names[3] + gw_types[i]] = s_nonzero_elements_per_rank[i]
        matrices_energy_per_rank[gw_names[0] + gw_types[i]] = g_energy_per_rank[i]
        matrices_energy_per_rank[gw_names[1] + gw_types[i]] = p_energy_per_rank[i]
        matrices_energy_per_rank[gw_names[2] + gw_types[i]] = w_energy_per_rank[i]
        matrices_energy_per_rank[gw_names[3] + gw_types[i]] = s_energy_per_rank[i]

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
        energy_fermi_left_per_iteration,
        energy_fermi_right_per_iteration,
        G_greater,
        G_lesser
    ):
        # calculate the green's function at every rank------------------------------
        fermi_distribution_left = fermi_distribution(energy_points_per_rank,
                                                    energy_fermi_left_per_iteration,
                                                    elementary_charge,
                                                    boltzmann_constant,
                                                    temperature_in_kelvin)
        fermi_distribution_right = fermi_distribution(energy_points_per_rank,
                                                    energy_fermi_right_per_iteration,
                                                    elementary_charge,
                                                    boltzmann_constant,
                                                    temperature_in_kelvin)


        # cannot be put into the self-energy_points kernel because transposed is needed
        for ie in range(number_of_energy_points_per_rank):
            # symmetrize (lesser and greater have to be skewed symmetric)
            Sigma_lesser[ie] = (Sigma_lesser[ie] - Sigma_lesser[ie].T.conj()) / 2
            Sigma_greater[ie] = (Sigma_greater[ie] - Sigma_greater[ie].T.conj()) / 2
            Sigma_retarded[ie] = np.real(Sigma_retarded[ie]) + (Sigma_greater[ie] - Sigma_lesser[ie]) / 2


        G_lesser_diag_blocks = np.zeros((number_of_energy_points_per_rank,
                                        number_of_blocks,
                                        blocksize,
                                        blocksize),
                                        dtype=np.complex128)
        G_lesser_upper_blocks = np.zeros((number_of_energy_points_per_rank,
                                        number_of_blocks - 1,
                                        blocksize,
                                        blocksize),
                                        dtype=np.complex128)
        G_greater_diag_blocks = np.zeros((number_of_energy_points_per_rank,
                                        number_of_blocks,
                                        blocksize,
                                        blocksize),
                                        dtype=np.complex128)
        G_greater_upper_blocks = np.zeros((number_of_energy_points_per_rank,
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

        G_greater[:,:number_of_nonzero_elements] = triple_array_to_flattened(
                                                        map_blocks_to_flattened,
                                                        G_greater_diag_blocks,
                                                        G_greater_upper_blocks,
                                                        G_greater_lower_blocks,
                                                        number_of_nonzero_elements,
                                                        number_of_energy_points_per_rank)

        G_lesser[:,:number_of_nonzero_elements] = triple_array_to_flattened(
                                                        map_blocks_to_flattened,
                                                        G_lesser_diag_blocks,
                                                        G_lesser_upper_blocks,
                                                        G_lesser_lower_blocks,
                                                        number_of_nonzero_elements,
                                                        number_of_energy_points_per_rank)



    def polarization_compute(
        G_greater,
        G_lesser,
        Polarization_greater,
        Polarization_lesser
    ):
        (Polarization_greater[:, :number_of_energy_points],
        Polarization_lesser[:, :number_of_energy_points]) = compute_polarization(
                                                            G_lesser[:, :number_of_energy_points],
                                                            G_greater[:, :number_of_energy_points],
                                                            delta_energy)
        

    def screened_interaction_compute(
        Polarization_greater_flattened,
        Polarization_lesser_flattened,
        Screened_interaction_greater_flattened,
        Screened_interaction_lesser_flattened
    ):
        # transform from 2D format to list/vector of sparse arrays format-----------
        Polarization_greater_list = flattened_to_list_of_csr(
                    Polarization_greater_flattened[:, :number_of_nonzero_elements],
                    rows,
                    columns,
                    number_of_orbitals)
        Polarization_lesser_list = flattened_to_list_of_csr(
                Polarization_lesser_flattened[:, :number_of_nonzero_elements],
                rows, columns,
                number_of_orbitals)

        Polarization_retarded_list = []
        # Symmetrization of Polarization (TODO: check if this is needed)
        for ie in range(number_of_energy_points_per_rank):
            # Anti-Hermitian symmetrizing of PL and PG
            Polarization_lesser_list[ie] = (Polarization_lesser_list[ie] - Polarization_lesser_list[ie].conj().T) / 2

            Polarization_greater_list[ie] = (Polarization_greater_list[ie] - Polarization_greater_list[ie].conj().T) / 2
            
            # PR has to be derived from PL and PG and then has to be symmetrized
            Polarization_retarded_list.append((Polarization_greater_list[ie] - Polarization_lesser_list[ie]) / 2)

        # calculate the screened interaction on every rank--------------------------
        Screened_interaction_lesser_diag_blocks = np.zeros((number_of_energy_points,
                                                            number_of_blocks,
                                                            blocksize,
                                                            blocksize),
                                                            dtype=np.complex128)
        Screened_interaction_lesser_upper_blocks = np.zeros((number_of_energy_points,
                                                             number_of_blocks - 1,
                                                             blocksize,
                                                             blocksize),
                                                             dtype=np.complex128)
        Screened_interaction_greater_diag_blocks = np.zeros((number_of_energy_points,
                                                             number_of_blocks,
                                                             blocksize,
                                                             blocksize),
                                                             dtype=np.complex128)
        Screened_interaction_greater_upper_blocks = np.zeros((number_of_energy_points,
                                                              number_of_blocks - 1,
                                                              blocksize,
                                                              blocksize),
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

        Screened_interaction_greater_flattened[:, :number_of_nonzero_elements] = triple_array_to_flattened(
                                            map_blocks_to_flattened,
                                            Screened_interaction_greater_diag_blocks,
                                            Screened_interaction_greater_upper_blocks,
                                            Screened_interaction_greater_lower_blocks,
                                            number_of_nonzero_elements,
                                            number_of_energy_points_per_rank)
        Screened_interaction_lesser_flattened[:, :number_of_nonzero_elements] = triple_array_to_flattened(
                                            map_blocks_to_flattened,
                                            Screened_interaction_lesser_diag_blocks,
                                            Screened_interaction_lesser_upper_blocks,
                                            Screened_interaction_lesser_lower_blocks,
                                            number_of_nonzero_elements,
                                            number_of_energy_points_per_rank)


    def selfenergy_compute(
            G_greater,
            G_lesser,
            Screened_interaction_greater,
            Screened_interaction_lesser,
            Sigma_greater,
            Sigma_lesser,
            Sigma_retarded
    ):
        # todo optimize and not load two time green's function to gpu and do twice the fft
        (Sigma_greater[:, :number_of_energy_points],
         Sigma_lesser[:, :number_of_energy_points],
         Sigma_retarded[:, :number_of_energy_points]) = compute_gw_self_energy(
            G_lesser[:, :number_of_energy_points],
            G_greater[:, :number_of_energy_points],
            Screened_interaction_lesser[:, :number_of_energy_points],
            Screened_interaction_greater[:, :number_of_energy_points],
            Coulomb_matrix_per_rank,
            delta_energy)




    # create communication wrapped functions
    greens_function = CommunicateCompute(distributions[:g_num_buffer],
                                         g_num_buffer, "c2r",
                                         g_energy_per_rank,
                                         g_nonzero_elements_per_rank,
                                         buffer_compute_unblock=buffer_energy_per_rank_compute[:g_num_buffer],
                                         buffer_send_unblock=buffer_energy_per_rank_send[:g_num_buffer],
                                         buffer_recv_unblock=buffer_nonzero_elements_per_rank_recv[:g_num_buffer],
                                         comm_unblock=comm_unblock,
                                         distributions_unblock=distributions_unblock_energy_per_rank[:g_num_buffer],
                                         batchsize=batchsize_energy_per_rank,
                                         iterations=iterations_unblock[1])(compute_greens_function)

    polarization = CommunicateCompute(distributions[:p_num_buffer],
                                      p_num_buffer, "r2c",
                                      p_nonzero_elements_per_rank,
                                      p_energy_per_rank,
                                      buffer_compute_unblock=buffer_nonzero_elements_per_rank_compute[:p_num_buffer],
                                      buffer_send_unblock=buffer_nonzero_elements_per_rank_send[:p_num_buffer],
                                      buffer_recv_unblock=buffer_energy_per_rank_recv[:p_num_buffer],
                                      comm_unblock=comm_unblock,
                                      distributions_unblock=distributions_unblock_nonzero_elements_per_rank[:p_num_buffer],
                                      batchsize=batchsize_nonzero_elements_per_rank,
                                      iterations=iterations_unblock[0])(polarization_compute)

    screened_interaction = CommunicateCompute(distributions[:w_num_buffer],
                                              w_num_buffer, "c2r",
                                              w_energy_per_rank_tmp,
                                              w_nonzero_elements_per_rank_tmp,
                                              buffer_compute_unblock=buffer_energy_per_rank_compute[:w_num_buffer],
                                              buffer_send_unblock=buffer_energy_per_rank_send[:w_num_buffer],
                                              buffer_recv_unblock=buffer_nonzero_elements_per_rank_recv[:w_num_buffer],
                                              comm_unblock=comm_unblock,
                                              distributions_unblock=distributions_unblock_energy_per_rank[:w_num_buffer],
                                              batchsize=batchsize_energy_per_rank,
                                              iterations=iterations_unblock[1])(screened_interaction_compute)

    selfenergy = CommunicateCompute(distributions[:s_num_buffer],
                                    s_num_buffer, "r2c",
                                    s_nonzero_elements_per_rank_tmp,
                                    s_energy_per_rank_tmp,
                                    buffer_compute_unblock=buffer_nonzero_elements_per_rank_compute[:s_num_buffer],
                                    buffer_send_unblock=buffer_nonzero_elements_per_rank_send[:s_num_buffer],
                                    buffer_recv_unblock=buffer_energy_per_rank_recv[:s_num_buffer],
                                    comm_unblock=comm_unblock,
                                    distributions_unblock=distributions_unblock_nonzero_elements_per_rank[:s_num_buffer],
                                    batchsize=batchsize_nonzero_elements_per_rank,
                                    iterations=iterations_unblock[0])(selfenergy_compute)

    if rank == 0:
        time_def_func += time.perf_counter()
        time_startup += time.perf_counter()
        time_loop = -time.perf_counter()

    for iter_num in range(gw_num_iter):


        if rank == 0:
            time_g = -time.perf_counter()

        # transform from 2D format to list/vector of sparse arrays format-----------
        s_energy_per_rank_list = [flattened_to_list_of_csr(
                    s_energy_per_rank[i][:, :data_shape[0]],
                    rows,
                    columns,
                    number_of_orbitals)
                    for i in range(s_num_buffer)]

        # Adjusting Fermi Levels of both contacts to the current iteration band minima
        sr_ephn_energy_per_rank_list = flattened_to_list_of_csr(
                        np.zeros((number_of_energy_points_per_rank,
                        number_of_nonzero_elements),
                        dtype=base_type),
                        rows,
                        columns,
                        number_of_orbitals)

        energy_conduction_band_minimum_over_iterations[iter_num + 1] = \
            get_band_edge_mpi_interpol(
                        energy_conduction_band_minimum_over_iterations[iter_num],
                        energy_points,
                        hamiltonian_obj.Overlap["H_4"],
                        hamiltonian_obj.Hamiltonian["H_4"],
                        *(s_energy_per_rank_list[::-1]),
                        sr_ephn_energy_per_rank_list,
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

        energy_fermi_left = energy_conduction_band_minimum_over_iterations[iter_num + 1] + energy_difference_fermi_minimum_left
        energy_fermi_right = energy_conduction_band_minimum_over_iterations[iter_num + 1] + energy_difference_fermi_minimum_right
        energy_fermi_left_over_iterations[iter_num + 1] = (energy_conduction_band_minimum_over_iterations[iter_num + 1] 
                                                           + energy_difference_fermi_minimum_left)
        energy_fermi_right_over_iterations[iter_num + 1] = (energy_conduction_band_minimum_over_iterations[iter_num + 1] +
                                                            energy_difference_fermi_minimum_right)

        greens_function_inputs_to_slice = [*s_energy_per_rank_list]
        greens_function_inputs = [energy_fermi_left, energy_fermi_right]
        greens_function(greens_function_inputs_to_slice, greens_function_inputs)

        # only take part of the greens function


        if rank == 0:
            time_g += time.perf_counter()
            time_p = -time.perf_counter()

        # calculate and communicate the polarization----------------------------------
        polarization_inputs_to_slice = [*g_nonzero_elements_per_rank]
        polarization_inputs = []
        polarization(polarization_inputs_to_slice, polarization_inputs)

        if rank == 0:
            time_p += time.perf_counter()
            time_w = -time.perf_counter()

        # calculate and communicate the screened interaction----------------------------------
        screened_interaction_inputs_to_slice = [*p_energy_per_rank]
        screened_interaction_inputs = []
        screened_interaction(screened_interaction_inputs_to_slice,
                             screened_interaction_inputs)


        for i in range(w_num_buffer):
            w_nonzero_elements_per_rank[i][:, :] = (
                1.0 - mem_w) * w_nonzero_elements_per_rank_tmp[i] + mem_w * w_nonzero_elements_per_rank[i]

        if rank == 0:
            time_w += time.perf_counter()
            time_s = -time.perf_counter()

        # compute and communicate the self-energy_points------------------------------------
        selfenergy_inputs_to_slice = [*g_nonzero_elements_per_rank[:g_num_buffer], *w_nonzero_elements_per_rank]
        selfenergy_inputs = []
        selfenergy(selfenergy_inputs_to_slice, selfenergy_inputs)

        # only take part of the self energy_points
        for i in range(s_num_buffer):
            s_energy_per_rank[i][:] = (1.0 - mem_s) * s_energy_per_rank_tmp[i] + mem_s * s_energy_per_rank[i]

        if rank == 0:
            time_s += time.perf_counter()


    if rank == 0:
        time_loop += time.perf_counter()
        time_end = -time.perf_counter()

    # communicate corrected g/w since data without filtering is communicated
    for gw_type in gw_types:
        for gw_name in ["g", "w"]:
            distribution.alltoall_r2c(matrices_nonzero_elements_per_rank[gw_name + gw_type],
                                      matrices_energy_per_rank[gw_name + gw_type], transpose_net=False)

    # gather at master to test against gold solution------------------------------
    if rank == 0:
        # create buffers at master
        matrices_global = {gw_name + gw_type: np.empty(distribution.shape, dtype=distribution.base_type)
                           for gw_type in gw_types for gw_name in gw_names}
        for gw_type in gw_types:
            for gw_name in gw_names:
                distribution.gather_master(matrices_energy_per_rank[gw_name + gw_type],
                                           matrices_global[gw_name + gw_type], transpose_net=False)
    else:
        # send time to master
        for gw_type in gw_types:
            for gw_name in gw_names:
                distribution.gather_master(
                    matrices_energy_per_rank[gw_name + gw_type], None, transpose_net=False)

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
        matrices_global_save["coulomb_matrix"] = Coulomb_matrix_flattened
        save_all(energy_points, rows, columns, blocksize, save_path, **matrices_global_save)


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
