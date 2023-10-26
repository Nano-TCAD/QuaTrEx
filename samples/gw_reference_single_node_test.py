# Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.
import numpy as np
from quatrex.refactored_utils.utils import flattened_to_list_of_csr, map_triple_array_to_flattened
from quatrex.files_to_refactor.adjust_conduction_band_edge import adjust_conduction_band_edge
from quatrex.refactored_utils.fermi_distribution import fermi_distribution
from quatrex.refactored_utils.read_solution import load_a_gw_matrix_flattened

from quatrex.refactored_solvers.greens_function_solver import greens_function_solver
from quatrex.refactored_solvers.screened_interaction_solver import screened_interaction_solver
from quatrex.refactored_solvers.polarization_solver import compute_polarization
from quatrex.refactored_solvers.gw_self_energy_solver import compute_gw_self_energy


# TODO refactor
from quatrex.OMEN_structure_matrices import OMENHamClass
from quatrex.OMEN_structure_matrices.construct_CM import construct_coulomb_matrix

if __name__ == "__main__":
    number_of_gw_iterations = 2
    base_type = np.complex128
    energy_points = np.linspace(-10.0, 5.0, 5, endpoint=True, dtype=float)
    screened_interaction_memory_factor = 0.1
    self_energy_memory_factor = 0.5

    hamiltonian_path = "/usr/scratch/mont-fort17/dleonard/GW_paper/InAs"
    reference_solution_path = "/usr/scratch/mont-fort17/almaeder/test_gw/reference_iter" \
        + str(number_of_gw_iterations) + ".mat"
    reference_solution_path = "/usr/scratch/mont-fort17/almaeder/test_gw/reference_five_energy_points_iter"\
        + str(number_of_gw_iterations) + ".mat"

    voltage_applied = 0.4
    energy_fermi_left = 1.9
    energy_fermi_right = energy_fermi_left - voltage_applied
    energy_conduction_band_minimum = 1.9346
    energy_difference_fermi_minimum_left = energy_fermi_left - \
        energy_conduction_band_minimum
    energy_difference_fermi_minimum_right = energy_fermi_right - \
        energy_conduction_band_minimum

    # TODO: refactor
    number_of_orbital_per_atom = np.array([5, 5])
    hamiltonian_obj = OMENHamClass.Hamiltonian(
        hamiltonian_path, number_of_orbital_per_atom, Vappl=voltage_applied)

    # TODO refactor
    temperature_in_kelvin = 300
    relative_permittivity = 2.5
    Coulomb_matrix = construct_coulomb_matrix(
        hamiltonian_obj, relative_permittivity, diag=False, orb_uniform=True)

    # TODO get rows / columns from neighbour matrix
    _, rows, columns, _, _ = load_a_gw_matrix_flattened(
        reference_solution_path, "g")
    Coulomb_matrix_flattened = np.asarray(
        Coulomb_matrix[rows, columns].reshape(-1))

    Hamiltonian = hamiltonian_obj.Hamiltonian["H_4"]
    Overlap_matrix = hamiltonian_obj.Overlap["H_4"]

    delta_energy = energy_points[1] - energy_points[0]
    blocksize = np.max(hamiltonian_obj.Bmax - hamiltonian_obj.Bmin + 1)
    number_of_orbitals = Hamiltonian.shape[0]
    assert number_of_orbitals == Hamiltonian.shape[1]
    assert rows.shape[0] == columns.shape[0]
    assert number_of_orbitals % blocksize == 0
    number_of_blocks = int(number_of_orbitals / blocksize)
    number_of_energy_points = energy_points.shape[0]
    number_of_nonzero_elements = rows.shape[0]

    # maps to transform from _blocks to _flattened
    map_blocks_to_flattened = map_triple_array_to_flattened(rows,
                                                            columns,
                                                            number_of_blocks,
                                                            blocksize)

    # initial self energies
    Self_energy_retarded_previous_iteration_flattened = np.zeros(
        (number_of_energy_points, number_of_nonzero_elements),
        dtype=base_type)

    Self_energy_lesser_previous_iteration_flattened = np.zeros(
        (number_of_energy_points, number_of_nonzero_elements),
        dtype=base_type)

    Self_energy_greater_previous_iteration_flattened = np.zeros(
        (number_of_energy_points, number_of_nonzero_elements),
        dtype=base_type)

    G_lesser_diag_blocks = np.zeros((number_of_energy_points,
                                    number_of_blocks,
                                    blocksize,
                                    blocksize),
                                    dtype=base_type)
    G_lesser_upper_blocks = np.zeros((number_of_energy_points,
                                      number_of_blocks - 1,
                                      blocksize,
                                      blocksize),
                                     dtype=base_type)
    G_greater_diag_blocks = np.zeros((number_of_energy_points,
                                      number_of_blocks,
                                      blocksize,
                                      blocksize),
                                     dtype=base_type)
    G_greater_upper_blocks = np.zeros((number_of_energy_points,
                                       number_of_blocks - 1,
                                       blocksize,
                                       blocksize),
                                      dtype=base_type)

    Screened_interaction_lesser_diag_blocks = np.zeros((number_of_energy_points,
                                                        number_of_blocks,
                                                        blocksize,
                                                        blocksize),
                                                       dtype=base_type)
    Screened_interaction_lesser_upper_blocks = np.zeros((number_of_energy_points,
                                                        number_of_blocks - 1,
                                                        blocksize,
                                                        blocksize),
                                                        dtype=base_type)
    Screened_interaction_greater_diag_blocks = np.zeros((number_of_energy_points,
                                                        number_of_blocks,
                                                        blocksize,
                                                        blocksize),
                                                        dtype=base_type)
    Screened_interaction_greater_upper_blocks = np.zeros((number_of_energy_points,
                                                          number_of_blocks - 1,
                                                          blocksize,
                                                          blocksize),
                                                         dtype=base_type)
    Screened_interaction_greater_previous_iteration_flattened = np.zeros((number_of_nonzero_elements,
                                                                          number_of_energy_points),
                                                                         dtype=base_type)
    Screened_interaction_lesser_previous_iteration_flattened = np.zeros((number_of_nonzero_elements,
                                                                         number_of_energy_points),
                                                                        dtype=base_type)

    for gw_iteration in range(number_of_gw_iterations):

        # transform the self energy from flattened to list
        Self_energy_retarded_list = flattened_to_list_of_csr(
            Self_energy_retarded_previous_iteration_flattened,
            rows,
            columns,
            number_of_orbitals)
        Self_energy_lesser_list = flattened_to_list_of_csr(
            Self_energy_lesser_previous_iteration_flattened,
            rows,
            columns,
            number_of_orbitals)
        Self_energy_greater_list = flattened_to_list_of_csr(
            Self_energy_greater_previous_iteration_flattened,
            rows,
            columns,
            number_of_orbitals)

        # Adjusting Fermi Levels of both contacts to the current iteration band minima
        energy_conduction_band_minimum = adjust_conduction_band_edge(
            energy_conduction_band_minimum,
            energy_points,
            Overlap_matrix,
            Hamiltonian,
            Self_energy_retarded_list,
            Self_energy_lesser_list,
            Self_energy_greater_list,
            blocksize)

        energy_fermi_left = energy_conduction_band_minimum + \
            energy_difference_fermi_minimum_left
        energy_fermi_right = energy_conduction_band_minimum + \
            energy_difference_fermi_minimum_right

        fermi_distribution_left = fermi_distribution(energy_points,
                                                     energy_fermi_left,
                                                     temperature_in_kelvin)
        fermi_distribution_right = fermi_distribution(energy_points,
                                                      energy_fermi_right,
                                                      temperature_in_kelvin)

        for ie in range(number_of_energy_points):
            # symmetrize (lesser and greater have to be skewed symmetric)
            Self_energy_lesser_list[ie] = (Self_energy_lesser_list[ie] -
                                           Self_energy_lesser_list[ie].T.conj()) / 2
            Self_energy_greater_list[ie] = (Self_energy_greater_list[ie] -
                                            Self_energy_greater_list[ie].T.conj()) / 2
            Self_energy_retarded_list[ie] = (np.real(Self_energy_retarded_list[ie])
                                             + (Self_energy_greater_list[ie] - Self_energy_lesser_list[ie]) / 2)

        (G_greater_flattened,
         G_lesser_flattened) = greens_function_solver(
            Hamiltonian,
            Overlap_matrix,
            Self_energy_retarded_list,
            Self_energy_lesser_list,
            Self_energy_greater_list,
            energy_points,
            fermi_distribution_left,
            fermi_distribution_right,
            rows,
            columns,
            blocksize)

        (Polarization_greater_flattened,
         Polarization_lesser_flattened) = compute_polarization(
            G_lesser_flattened.transpose(),
            G_greater_flattened.transpose(),
            delta_energy)

        (Screened_interaction_greater_flattened,
         Screened_interaction_lesser_flattened) = screened_interaction_solver(
            Coulomb_matrix,
            Polarization_greater_flattened.transpose(),
            Polarization_lesser_flattened.transpose(),
            number_of_energy_points,
            rows,
            columns,
            blocksize)

        # mix with old screened interaction
        Screened_interaction_greater_previous_iteration_flattened = \
            (1.0 - screened_interaction_memory_factor) * Screened_interaction_greater_flattened.transpose() \
            + screened_interaction_memory_factor * \
            Screened_interaction_greater_previous_iteration_flattened
        Screened_interaction_lesser_previous_iteration_flattened = \
            (1.0 - screened_interaction_memory_factor) * Screened_interaction_lesser_flattened.transpose() \
            + screened_interaction_memory_factor * \
            Screened_interaction_lesser_previous_iteration_flattened

        (Self_energy_greater_flattened,
         Self_energy_lesser_flattened,
         Self_energy_retarded_flattened) = compute_gw_self_energy(
            G_lesser_flattened.transpose(),
            G_greater_flattened.transpose(),
            Screened_interaction_lesser_previous_iteration_flattened,
            Screened_interaction_greater_previous_iteration_flattened,
            Coulomb_matrix_flattened,
            delta_energy)

        # mix with old self energy
        Self_energy_greater_previous_iteration_flattened = \
            (1.0 - self_energy_memory_factor) * Self_energy_greater_flattened.transpose() \
            + self_energy_memory_factor * Self_energy_greater_previous_iteration_flattened
        Self_energy_lesser_previous_iteration_flattened = \
            (1.0 - self_energy_memory_factor) * Self_energy_lesser_flattened.transpose() \
            + self_energy_memory_factor * Self_energy_lesser_previous_iteration_flattened
        Self_energy_retarded_previous_iteration_flattened = \
            (1.0 - self_energy_memory_factor) * Self_energy_retarded_flattened.transpose() \
            + self_energy_memory_factor * Self_energy_retarded_previous_iteration_flattened

    # load reference solution
    energy_reference, rows_reference, columns_reference, \
        G_greater_reference, G_lesser_reference = load_a_gw_matrix_flattened(
            reference_solution_path, "g")
    _, _, _, Polarization_greater_reference, Polarization_lesser_reference = load_a_gw_matrix_flattened(
        reference_solution_path, "p")
    _, _, _, Screened_interaction_greater_reference,\
        Screened_interaction_lesser_reference = load_a_gw_matrix_flattened(
            reference_solution_path, "w")
    _, _, _, Self_energy_greater_reference, Self_energy_lesser_reference = load_a_gw_matrix_flattened(
        reference_solution_path, "s")

    assert np.allclose(energy_points, energy_reference)
    assert np.allclose(rows, rows_reference)
    assert np.allclose(columns, columns_reference)

    # print norm difference
    print("differences to reference Green's Function: ",
          f"g: {np.linalg.norm(G_greater_flattened.transpose()-G_greater_reference):.4f}",
          f"l: {np.linalg.norm(G_lesser_flattened.transpose()-G_lesser_reference):.4f}")
    print("differences to reference Polarization:",
          f"g: {np.linalg.norm(Polarization_greater_flattened-Polarization_greater_reference):.4f}",
          f"l: {np.linalg.norm(Polarization_lesser_flattened-Polarization_lesser_reference):.4f}")
    print("differences to reference Screened Interaction:",
          f"g: {np.linalg.norm(Screened_interaction_greater_previous_iteration_flattened-Screened_interaction_greater_reference):.4f}",
          f"{np.linalg.norm(Screened_interaction_lesser_previous_iteration_flattened-Screened_interaction_lesser_reference):.4f}")
    print("differences to reference Self Energy:",
          f"g: {np.linalg.norm(Self_energy_greater_previous_iteration_flattened.T-Self_energy_greater_reference):.4f}",
          f"l: {np.linalg.norm(Self_energy_lesser_previous_iteration_flattened.T-Self_energy_lesser_reference):.4f}")

    # compare with reference solution
    assert np.allclose(G_greater_flattened.transpose(),
                       G_greater_reference)
    assert np.allclose(G_lesser_flattened.transpose(),
                       G_lesser_reference)
    assert np.allclose(Polarization_greater_flattened,
                       Polarization_greater_reference)
    assert np.allclose(Polarization_lesser_flattened,
                       Polarization_lesser_reference)
    assert np.allclose(Screened_interaction_greater_previous_iteration_flattened,
                       Screened_interaction_greater_reference)
    assert np.allclose(Screened_interaction_lesser_previous_iteration_flattened,
                       Screened_interaction_lesser_reference)
    assert np.allclose(Self_energy_greater_previous_iteration_flattened.transpose(),
                       Self_energy_greater_reference)
    assert np.allclose(Self_energy_lesser_previous_iteration_flattened.transpose(),
                       Self_energy_lesser_reference)
    print("All tests passed")
