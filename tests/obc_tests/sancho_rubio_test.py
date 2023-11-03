# Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.

import numpy as np
import pytest

from quatrex.open_boundary_conditions.sancho_rubio import sancho_rubio
from quatrex.test_utils.create_matrices import create_invertible_block

@pytest.mark.parametrize(
    "blocksize",
    [(1),
     (21),
     (75),
     ]
)
def test_convergence(
    blocksize: int
):
    """Tests that Sancho-Rubio converges to correct result."""
    rng = np.random.default_rng()
    a_ii = create_invertible_block(blocksize)
    a_ij = rng.uniform(size=(blocksize, blocksize)) +\
            1j * rng.uniform(size=(blocksize, blocksize))
    a_ji = rng.uniform(size=(blocksize, blocksize)) +\
            1j * rng.uniform(size=(blocksize, blocksize))
    x_ii = sancho_rubio(a_ii, a_ij, a_ji)
    assert np.allclose(x_ii, np.linalg.inv(a_ii - a_ji @ x_ii @ a_ij))

@pytest.mark.parametrize(
    "blocksize",
    [(1),
     (21),
     (75),
     ]
)
def test_convergence_hermitian(
    blocksize: int
):
    """Tests that the off_diagonal does not have to be explicit."""
    rng = np.random.default_rng()
    a_ii = create_invertible_block(blocksize)
    a_ij = rng.uniform(size=(blocksize, blocksize)) +\
            1j * rng.uniform(size=(blocksize, blocksize))
    x_ii = sancho_rubio(a_ii, a_ij)
    assert np.allclose(x_ii, np.linalg.inv(a_ii - a_ij.conj().T @ x_ii @ a_ij))

@pytest.mark.parametrize(
    "blocksize",
    [(1),
     (21),
     (75),
     ]
)
def test_max_iterations(
    blocksize: int
):
    """Tests that Sancho-Rubio raises Exception after max_iterations."""
    rng = np.random.default_rng()
    a_ii = create_invertible_block(blocksize)
    a_ij = rng.uniform(size=(blocksize, blocksize)) +\
            1j * rng.uniform(size=(blocksize, blocksize))
    a_ji = rng.uniform(size=(blocksize, blocksize)) +\
            1j * rng.uniform(size=(blocksize, blocksize))
    with pytest.raises(RuntimeError):
        sancho_rubio(a_ii, a_ij, a_ji, max_iterations=1, max_delta=1e-8)
