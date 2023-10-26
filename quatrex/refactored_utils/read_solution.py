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

def save_all(
    energy_points:np.ndarray,
    rows: np.ndarray,
    columns: np.ndarray,
    blocksize: int,
    path: str,
    **kwargs
):
    """
    Saves all the data to a file

    Args:
        energy_points (npt.NDArray[np.float64]): Energy
        rows (np.ndarray): Non-zero row elements
        columns (np.ndarray): Non-zero column elements
        blocksize (int): blocksizes
        path (str): Path+filename where to save
        **kwargs: Dictionary of all the data to save
    """
    # create the file
    hdf5 = h5py.File(path, "w")
    group = hdf5.create_group("formatted")

    # write the entries
    group.create_dataset("rows", data=rows)
    group.create_dataset("columns", data=columns)
    group.create_dataset("energy_points", data=energy_points)
    group.create_dataset("blocksize", data=blocksize)
    for key, value in kwargs.items():
        group.create_dataset("real_part_" + key, data=np.real(value))
        group.create_dataset("imaginary_part_" + key, data=np.imag(value))
