# Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.

import numpy as np
from scipy.sparse import csr_matrix


def csr_to_flattened(
    A: csr_matrix,
    indices: dict[np.ndarray]
):
    return A[indices["row"], indices["col"]]


def csr_to_triple_array(
    A: csr_matrix,
    blocksize: int,
) -> [np.ndarray, np.ndarray, np.ndarray]:

    number_of_blocks = int(A.shape[0] / blocksize)

    A_diag_blocks = np.zeros(
        (number_of_blocks, blocksize, blocksize), dtype=A.dtype)
    A_upper_blocks = np.zeros(
        (number_of_blocks-1, blocksize, blocksize), dtype=A.dtype)

    for j in range(number_of_blocks):
        A_diag_blocks[j] = A[j *
                             blocksize: (j + 1) * blocksize, j * blocksize: (j + 1) * blocksize]
    for j in range(number_of_blocks-1):
        A_upper_blocks[j] = A[j * blocksize: (j + 1) * blocksize,
                              (j + 1) * blocksize: (j + 2) * blocksize]

    return A_diag_blocks, A_upper_blocks


def flattened_to_list_of_csr(
    A_flattened: np.ndarray,
    indices: dict[np.ndarray],
    number_of_orbitals: int
) -> list[csr_matrix]:
    """
        Converts from flattened array of size
        (number_of_energy_points, number_of_nonzero_elements)
        to a list of sparse csr matrices of size 
        number_of_energy_points*(number_of_orbitals,number_of_orbitals).
    """
    # number of energy points
    number_of_energy_points = A_flattened.shape[0]
    A_list = []
    for i in range(number_of_energy_points):
        A_list.append(
            csr_matrix((A_flattened[i, :],
                        (indices["row"], indices["col"])),
                       shape=(number_of_orbitals, number_of_orbitals),
                       dtype=A_flattened.dtype)
        )

    return A_list


def triple_array_to_flattened(
        map_tripple_array_to_flattened: dict,
        A_diag_blocks: np.ndarray,
        A_upper_blocks: np.ndarray,
        A_lower_blocks: np.ndarray,
        number_of_nonzero_elements: int,
        number_of_energy_points: int
) -> np.ndarray:
    """
    Converts from triple array
    of diagonal, upper, lower blocks of size
    (number_of_energy_points, number_of_blocks, blocksize, blocksize)
    (number_of_energy_points, number_of_blocks-1, blocksize, blocksize)
    (number_of_energy_points, number_of_blocks-1, blocksize, blocksize)
    to flattened array of size
    (number_of_energy_points, number_of_nonzero_elements)
    """
    A_flattened = np.empty((number_of_energy_points,
                            number_of_nonzero_elements),
                           dtype=A_diag_blocks.dtype)

    # diagonal elements
    A_flattened[:, map_tripple_array_to_flattened["diag"][3, :]] = A_diag_blocks[:,
                                                                                 map_tripple_array_to_flattened["diag"][0, :],
                                                                                 map_tripple_array_to_flattened["diag"][1, :],
                                                                                 map_tripple_array_to_flattened["diag"][2, :]]
    # off diagonal elements
    A_flattened[:, map_tripple_array_to_flattened["upper"][3, :]] = A_upper_blocks[:,
                                                                                   map_tripple_array_to_flattened[
                                                                                       "upper"][0, :],
                                                                                   map_tripple_array_to_flattened[
                                                                                       "upper"][1, :],
                                                                                   map_tripple_array_to_flattened["upper"][2, :]]

    A_flattened[:, map_tripple_array_to_flattened["lower"][3, :]] = A_lower_blocks[:,
                                                                                   map_tripple_array_to_flattened[
                                                                                       "lower"][0, :],
                                                                                   map_tripple_array_to_flattened[
                                                                                       "lower"][1, :],
                                                                                   map_tripple_array_to_flattened["lower"][2, :]]
    return A_flattened


def map_triple_array_to_flattened(
        rows: np.ndarray,
        columns: np.ndarray,
        number_of_blocks: int,
        blocksize: int
) -> dict:
    """
    Creates a mapping from the triple array to the flattened array.
    The mapping is a dictionary with keys "diag", "upper", "lower".
    The values are arrays of size (4, number_of_nonzero_elements)
    where the first row is the block index,
    the second row is the row index in the block,
    the third row is the column index in the block,
    the fourth row is the location in the flattened array.
    """
    assert columns.size == rows.size

    # a priori the length is unknown
    map_diag = np.empty((4, 0), dtype=int)
    map_upper = np.empty((4, 0), dtype=int)
    map_lower = np.empty((4, 0), dtype=int)

    # count number of nonzero elements
    # to assert correctness
    number_of_nonzero_elements = 0

    # iterate over diagonal blocks
    for i in range(number_of_blocks):
        # start index and end index of the block
        start_index_block = i*blocksize
        end_index_block = (i+1)*blocksize - 1

        # mask_diag if the elements are in the block
        mask_diag = ((rows >= start_index_block) &
                     (rows <= end_index_block) &
                     (columns >= start_index_block) &
                     (columns <= end_index_block))

        # number of to read out elements in the block
        number_of_elements_diag_block = mask_diag.sum()
        number_of_nonzero_elements += number_of_elements_diag_block

        # per block mapping to one vector data
        map_per_diag_block = np.empty(
            (4, number_of_elements_diag_block), dtype=int)

        # sparse elements in the block
        map_per_diag_block[0, :] = i
        map_per_diag_block[1, :] = rows[mask_diag] - start_index_block
        map_per_diag_block[2, :] = columns[mask_diag] - start_index_block
        # location in data vector
        map_per_diag_block[3, :] = np.where(mask_diag)[0]
        # concat to buffer
        map_diag = np.concatenate((map_diag, map_per_diag_block), axis=1)

    # iterate over upper/lower diagonal
    for i in range(number_of_blocks - 1):
        # start index and end index
        start_index_block = i*blocksize
        end_index_block = (i+1)*blocksize - 1
        start_index_next_block = (i+1)*blocksize
        end_index_next_block = (i+2)*blocksize - 1

        # mask_diag if the elements are in the upper/lower block
        mask_upper = ((rows >= start_index_block) &
                      (rows <= end_index_block) &
                      (columns >= start_index_next_block) &
                      (columns <= end_index_next_block))

        mask_lower = ((rows >= start_index_next_block) &
                      (rows <= end_index_next_block) &
                      (columns >= start_index_block) &
                      (columns <= end_index_block))

        # number of to read out elements in the blocks
        number_of_elements_upper_block = mask_upper.sum()
        number_of_elements_lower_block = mask_lower.sum()

        # add to number nonzero
        number_of_nonzero_elements += number_of_elements_upper_block
        number_of_nonzero_elements += number_of_elements_lower_block

        # per block mapping to one vector data
        map_per_upper_block = np.empty(
            (4, number_of_elements_upper_block), dtype=int)
        map_per_lower_block = np.empty(
            (4, number_of_elements_lower_block), dtype=int)

        # sparse elements in the upper/lower block
        map_per_upper_block[0, :] = i
        map_per_lower_block[0, :] = i
        map_per_upper_block[1, :] = rows[mask_upper] - start_index_block
        map_per_upper_block[2, :] = columns[mask_upper] - start_index_next_block
        map_per_lower_block[1, :] = rows[mask_lower] - start_index_next_block
        map_per_lower_block[2, :] = columns[mask_lower] - start_index_block

        # location in data vector
        map_per_upper_block[3, :] = np.where(mask_upper)[0]
        map_per_lower_block[3, :] = np.where(mask_lower)[0]

        # concat to buffer
        map_upper = np.concatenate((map_upper, map_per_upper_block), axis=1)
        map_lower = np.concatenate((map_lower, map_per_lower_block), axis=1)

    assert number_of_nonzero_elements == rows.size

    return {"diag": map_diag, "upper": map_upper, "lower": map_lower}
