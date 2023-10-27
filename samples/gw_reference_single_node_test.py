# Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.
import numpy as np

from quatrex.refactored_utils.fermi_distribution import fermi_distribution
from quatrex.refactored_utils.read_reference import load_a_gw_matrix_flattened, save_inputs, save_outputs, save_parameters


from quatrex.refactored_solvers.greens_function_solver import greens_function_solver
from quatrex.refactored_solvers.screened_interaction_solver import screened_interaction_solver
from quatrex.refactored_solvers.polarization_solver import compute_polarization
from quatrex.refactored_solvers.gw_self_energy_solver import compute_gw_self_energy

# TODO refactor
from quatrex.files_to_refactor.adjust_conduction_band_edge import adjust_conduction_band_edge

from quatrex.OMEN_structure_matrices import OMENHamClass
from quatrex.OMEN_structure_matrices.construct_CM import construct_coulomb_matrix

if __name__ == "__main__":
    number_of_gw_iterations = 2
    base_type = np.complex128
    energy_points = np.linspace(-10.0, 5.0, 5, endpoint=True, dtype=float)
    screened_interaction_memory_factor = 0.1
    self_energy_memory_factor = 0.5
    save_reference = False

    save_path = "/usr/scratch/mont-fort17/almaeder/test_gw/InAs/"
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

    # TODO get row_indices_kept / col_indices_kept from neighbour matrix
    _, row_indices_kept, col_indices_kept, _, _ = load_a_gw_matrix_flattened(
        reference_solution_path, "g")
    Coulomb_matrix_flattened = np.asarray(
        Coulomb_matrix[row_indices_kept, col_indices_kept].reshape(-1))

    Hamiltonian = hamiltonian_obj.Hamiltonian["H_4"]
    Overlap_matrix = hamiltonian_obj.Overlap["H_4"]

    delta_energy = energy_points[1] - energy_points[0]
    blocksize = np.max(hamiltonian_obj.Bmax - hamiltonian_obj.Bmin + 1)
    number_of_orbitals = Hamiltonian.shape[0]
    assert number_of_orbitals == Hamiltonian.shape[1]
    assert row_indices_kept.shape[0] == col_indices_kept.shape[0]
    assert number_of_orbitals % blocksize == 0
    number_of_blocks = int(number_of_orbitals / blocksize)
    number_of_energy_points = energy_points.shape[0]
    number_of_elements_kept = row_indices_kept.shape[0]

    # initial self energies
    Self_energy_retarded_previous_iteration_flattened = np.zeros(
        (number_of_energy_points, number_of_elements_kept),
        dtype=base_type)

    Self_energy_lesser_previous_iteration_flattened = np.zeros(
        (number_of_energy_points, number_of_elements_kept),
        dtype=base_type)

    Self_energy_greater_previous_iteration_flattened = np.zeros(
        (number_of_energy_points, number_of_elements_kept),
        dtype=base_type)

    Screened_interaction_greater_previous_iteration_flattened = np.zeros(
        (number_of_elements_kept, number_of_energy_points),
        dtype=base_type)
    Screened_interaction_lesser_previous_iteration_flattened = np.zeros(
        (number_of_elements_kept, number_of_energy_points),
        dtype=base_type)

    for gw_iteration in range(number_of_gw_iterations):

        # Adjusting Fermi Levels of both contacts to the current iteration band minima
        energy_conduction_band_minimum = adjust_conduction_band_edge(
            energy_conduction_band_minimum,
            energy_points,
            Overlap_matrix,
            Hamiltonian,
            Self_energy_retarded_previous_iteration_flattened,
            Self_energy_lesser_previous_iteration_flattened,
            Self_energy_greater_previous_iteration_flattened,
            row_indices_kept,
            col_indices_kept,
            blocksize)


        fermi_distribution_left = fermi_distribution(energy_points,
                                                     energy_conduction_band_minimum
                                                     + energy_difference_fermi_minimum_left,
                                                     temperature_in_kelvin)
        fermi_distribution_right = fermi_distribution(energy_points,
                                                      energy_conduction_band_minimum
                                                      + energy_difference_fermi_minimum_right,
                                                      temperature_in_kelvin)

        (G_greater_flattened,
         G_lesser_flattened) = greens_function_solver(
            Hamiltonian,
            Overlap_matrix,
            Self_energy_retarded_previous_iteration_flattened,
            Self_energy_lesser_previous_iteration_flattened,
            Self_energy_greater_previous_iteration_flattened,
            energy_points,
            fermi_distribution_left,
            fermi_distribution_right,
            row_indices_kept,
            col_indices_kept,
            blocksize)

        (Polarization_lesser_flattened,
         Polarization_greater_flattened) = compute_polarization(
            G_lesser_flattened.T,
            G_greater_flattened.T,
            delta_energy)

        (Screened_interaction_greater_flattened,
         Screened_interaction_lesser_flattened) = screened_interaction_solver(
            Coulomb_matrix,
            Polarization_greater_flattened.T,
            Polarization_lesser_flattened.T,
            number_of_energy_points,
            row_indices_kept,
            col_indices_kept,
            blocksize)

        # mix with old screened interaction
        Screened_interaction_greater_previous_iteration_flattened = \
            (1.0 - screened_interaction_memory_factor) * Screened_interaction_greater_flattened.T \
            + screened_interaction_memory_factor * \
            Screened_interaction_greater_previous_iteration_flattened
        Screened_interaction_lesser_previous_iteration_flattened = \
            (1.0 - screened_interaction_memory_factor) * Screened_interaction_lesser_flattened.T \
            + screened_interaction_memory_factor * \
            Screened_interaction_lesser_previous_iteration_flattened

        (Self_energy_greater_flattened,
         Self_energy_lesser_flattened,
         Self_energy_retarded_flattened) = compute_gw_self_energy(
            G_lesser_flattened.T,
            G_greater_flattened.T,
            Screened_interaction_lesser_previous_iteration_flattened,
            Screened_interaction_greater_previous_iteration_flattened,
            Coulomb_matrix_flattened,
            delta_energy)

        # mix with old self energy
        Self_energy_greater_previous_iteration_flattened = \
            (1.0 - self_energy_memory_factor) * Self_energy_greater_flattened.T \
            + self_energy_memory_factor * Self_energy_greater_previous_iteration_flattened
        Self_energy_lesser_previous_iteration_flattened = \
            (1.0 - self_energy_memory_factor) * Self_energy_lesser_flattened.T \
            + self_energy_memory_factor * Self_energy_lesser_previous_iteration_flattened
        Self_energy_retarded_previous_iteration_flattened = \
            (1.0 - self_energy_memory_factor) * Self_energy_retarded_flattened.T \
            + self_energy_memory_factor * Self_energy_retarded_previous_iteration_flattened

    # load reference solution
    energy_reference, row_indices_kept_reference, column_indices_kept_reference, \
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
    assert np.allclose(row_indices_kept, row_indices_kept_reference)
    assert np.allclose(col_indices_kept, column_indices_kept_reference)

    # print norm difference
    print("differences to reference Green's Function: ",
          f"g: {np.linalg.norm(G_greater_flattened.T-G_greater_reference):.4f}",
          f"l: {np.linalg.norm(G_lesser_flattened.T-G_lesser_reference):.4f}")
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
    assert np.allclose(G_greater_flattened.T,
                       G_greater_reference)
    assert np.allclose(G_lesser_flattened.T,
                       G_lesser_reference)
    assert np.allclose(Polarization_greater_flattened,
                       Polarization_greater_reference)
    assert np.allclose(Polarization_lesser_flattened,
                       Polarization_lesser_reference)
    assert np.allclose(Screened_interaction_greater_previous_iteration_flattened,
                       Screened_interaction_greater_reference)
    assert np.allclose(Screened_interaction_lesser_previous_iteration_flattened,
                       Screened_interaction_lesser_reference)
    assert np.allclose(Self_energy_greater_previous_iteration_flattened.T,
                       Self_energy_greater_reference)
    assert np.allclose(Self_energy_lesser_previous_iteration_flattened.T,
                       Self_energy_lesser_reference)
    print("All tests passed")

    if save_reference:
        inputs_reference = {}
        inputs_reference["hamiltonian"] = Hamiltonian
        inputs_reference["overlap_matrix"] = Overlap_matrix
        inputs_reference["coulomb_matrix"] = Coulomb_matrix
        inputs_reference["row_indices_kept"] = row_indices_kept
        inputs_reference["col_indices_kept"] = col_indices_kept
        parameters_reference = {}
        parameters_reference["energy_points"] = energy_points
        parameters_reference["voltage_applied"] = voltage_applied
        parameters_reference["energy_fermi_left"] = energy_fermi_left
        parameters_reference["energy_conduction_band_minimum"] = energy_conduction_band_minimum
        parameters_reference["number_of_orbital_per_atom"] = number_of_orbital_per_atom
        parameters_reference["temperature_in_kelvin"] = temperature_in_kelvin
        parameters_reference["relative_permittivity"] = relative_permittivity
        parameters_reference["screened_interaction_memory_factor"] = screened_interaction_memory_factor
        parameters_reference["self_energy_memory_factor"] = self_energy_memory_factor
        parameters_reference["blocksize"] = blocksize
        outputs_reference = {}
        outputs_reference["G_greater"] = G_greater_flattened.T
        outputs_reference["G_lesser"] = G_lesser_flattened.T
        outputs_reference["Polarization_greater"] = Polarization_greater_flattened
        outputs_reference["Polarization_lesser"] = Polarization_lesser_flattened
        outputs_reference["Screened_interaction_greater"] = Screened_interaction_greater_previous_iteration_flattened
        outputs_reference["Screened_interaction_lesser"] = Screened_interaction_lesser_previous_iteration_flattened
        outputs_reference["Self_energy_greater"] = Self_energy_greater_previous_iteration_flattened.T
        outputs_reference["Self_energy_lesser"] = Self_energy_lesser_previous_iteration_flattened.T
        save_parameters(save_path, parameters_reference)
        save_inputs(save_path, inputs_reference)
        save_outputs(save_path, outputs_reference)
