# Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.

import numpy as np
import pytest

from quatrex.test_utils import create_matrices
from quatrex.refactored_utils import utils


@pytest.mark.parametrize(
    "matrix_size",
    [(1),
     (21),
     (75),
     ]
)
@pytest.mark.parametrize(
    "density",
    [(0.5),
     (0.1),
     ]
)
def test_csr_to_flattened(
    matrix_size: int,
    density: float
):
    row_indices_to_keep, col_indices_to_keep = \
        create_matrices.create_indices_to_keep(matrix_size, density)
    rng = np.random.default_rng()
    A = rng.uniform(size=(matrix_size, matrix_size)) +\
        1j * rng.uniform(size=(matrix_size, matrix_size))
    data = utils.csr_to_flattened(A, row_indices_to_keep, col_indices_to_keep)
    assert data.size == row_indices_to_keep.size
    assert data.ndim == 1
    assert np.allclose(data, A[row_indices_to_keep, col_indices_to_keep])


@pytest.mark.parametrize(
    "number_of_energy_points",
    [(1),
     (3),
     ]
)
@pytest.mark.parametrize(
    "matrix_size",
    [(1),
     (21),
     (75),
     ]
)
@pytest.mark.parametrize(
    "density",
    [(0.6),
     (0.3),
     ]
)
def test_flattened_to_list_of_csr(
    matrix_size: int,
    density: float,
    number_of_energy_points: int,
):
    row_indices_to_keep, col_indices_to_keep = \
        create_matrices.create_indices_to_keep(matrix_size, density)
    number_of_elements_kept = row_indices_to_keep.size
    rng = np.random.default_rng()
    data_reference = rng.uniform(size=(number_of_energy_points, number_of_elements_kept)) +\
        1j * rng.uniform(size=(number_of_energy_points, number_of_elements_kept))
    A_list = utils.flattened_to_list_of_csr(data_reference,
                                            row_indices_to_keep,
                                            col_indices_to_keep,
                                            matrix_size)
    data_test = np.zeros_like(data_reference)
    for i in range(number_of_energy_points):
        data_test[i,:] = utils.csr_to_flattened(A_list[i],
                                                 row_indices_to_keep,
                                                 col_indices_to_keep)
    assert np.allclose(data_test, data_reference)
