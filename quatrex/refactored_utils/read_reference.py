# Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.

import h5py
import numpy as np

# TODO change strings to read, but create first new reference
def load_a_gw_matrix_flattened(
    path: str,
    gw_matrix_name: str
) -> (np.ndarray,
      np.ndarray,
      np.ndarray,
      np.ndarray,
      np.ndarray,
      np.ndarray):
    """
    Load reference Green's Function,
    Polarization, Screened Interaction, or Self Energy from .mat file

    Args:
        path (str): filepath to .mat file
        gw_matrix_name (str): can be either "g"/"p"/"w"/"s"
        GPolarization, Screened Interaction, or Self Energy respectively

    Returns:
        np.ndarray: Energy 
        np.ndarray: Rows 
        np.ndarray: Columns 
        np.ndarray: Greater Function
        np.ndarray: Lesser Function
    """

    # open the file
    hdf5 = h5py.File(path, "r")["formatted"]

    # read the entries
    energy_points = np.array(hdf5["energy_points"], dtype=float)

    rows = np.array(hdf5["rows"], dtype=int)
    columns = np.array(hdf5["columns"], dtype=int)

    energy_points = np.squeeze(energy_points)
    rows = np.squeeze(rows)
    columns = np.squeeze(columns)

    Real_part_greater = np.array(hdf5["real_part_" + gw_matrix_name + "g"], dtype=float)
    Imaginary_part_greater = np.array(hdf5["imaginary_part_" + gw_matrix_name + "g"], dtype=float)
    Real_part_lesser = np.array(hdf5["real_part_" + gw_matrix_name + "l"], dtype=float)
    Imaginary_part_lesser = np.array(hdf5["imaginary_part_" + gw_matrix_name + "l"], dtype=float)

    Greater_flattened = Real_part_greater + 1j * Imaginary_part_greater
    Lesser_flattened = Real_part_lesser + 1j * Imaginary_part_lesser

    return (energy_points, rows, columns, Greater_flattened, Lesser_flattened)


def load_coulomb_matrix_flattened(
        path: str
) -> (np.ndarray, np.ndarray, np.ndarray):
    """Loads v hat/interaction from .mat file created by changeFormatV.m
        All sparse matrices are in the COO format and 
        share the same idx of non-zero elements.
        Such the rows, columns are only saved once.

    Args:
        path (str): filepath to .mat file

    Returns:
        np.ndarray: Rows 
        np.ndarray: Columns 
        np.ndarray: Coulomb Matrix
    """

    # open the file
    hdf5 = h5py.File(path, "r")["formatted"]

    # read the entries
    rows = np.array(hdf5["rows"], dtype=int)
    columns = np.array(hdf5["columns"], dtype=int)
    rows = np.squeeze(rows)
    columns = np.squeeze(columns)

    Real_part_coulomb_matrix = np.array(hdf5["real_part_coulomb_matrix"], dtype=float)
    Imaginary_part_coulomb_matrix = np.array(hdf5["imaginary_part_coulomb_matrix"], dtype=float)
    Coulomb_matrix = Real_part_coulomb_matrix + 1j * Imaginary_part_coulomb_matrix
    Coulomb_matrix = np.squeeze(Coulomb_matrix)

    return (rows, columns, Coulomb_matrix)

def save_parameters(
    path,
    parameters_reference
):
    np.savez(
        path + "parameters.npz",
        energy_points = parameters_reference["energy_points"],
        voltage_applied = parameters_reference["voltage_applied"],
        energy_fermi_left = parameters_reference["energy_fermi_left"],
        energy_conduction_band_minimum = parameters_reference["energy_conduction_band_minimum"],
        number_of_orbital_per_atom = parameters_reference["number_of_orbital_per_atom"],
        temperature_in_kelvin = parameters_reference["temperature_in_kelvin"],
        relative_permittivity = parameters_reference["relative_permittivity"],
        screened_interaction_memory_factor = parameters_reference["screened_interaction_memory_factor"],
        self_energy_memory_factor = parameters_reference["self_energy_memory_factor"],
        blocksize = parameters_reference["blocksize"]
    )


def save_inputs(
    path,
    inputs_reference
):
    np.savez(
        path + "hamiltonian.npz",
        indices = inputs_reference["hamiltonian"].indices,
        indptr = inputs_reference["hamiltonian"].indptr,
        data = inputs_reference["hamiltonian"].data
    )
    np.savez(
        path + "overlap_matrix.npz",
        indices = inputs_reference["overlap_matrix"].indices,
        indptr = inputs_reference["overlap_matrix"].indptr,
        data = inputs_reference["overlap_matrix"].data
    )
    np.savez(
        path + "coulomb_matrix.npz",
        indices = inputs_reference["overlap_matrix"].indices,
        indptr = inputs_reference["overlap_matrix"].indptr,
        data = inputs_reference["overlap_matrix"].data
    )
    np.savez(
        path + "row_indices_kept.npz",
        row_indices_kept = inputs_reference["row_indices_kept"]
    )
    np.savez(
        path + "column_indices_kept.npz",
        column_indices_kept = inputs_reference["column_indices_kept"]
    )

def save_outputs(
    path,
    outputs_reference
):
    gw_names = ["G", "Screened_interaction", "Polarization", "Self_energy"]
    gw_types = ["greater", "lesser"]
    for gw_name in gw_names:
        for gw_type in gw_types:
            np.savez(
                path + gw_name + "_" + gw_type + ".npz",
                dats = outputs_reference[gw_name + "_" + gw_type]
            )


