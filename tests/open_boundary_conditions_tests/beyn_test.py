# Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.

import numpy as np
import pytest

from quatrex.files_to_refactor.beyn_cpu import beyn


@pytest.mark.parametrize(
    "blocksize",
    [(1),
     (21),
     (75),
     ]
)
def test_beyn(
    blocksize: int
):
    # TODO refactor this test
    imaginary_limit = 5e-4
    contour_integration_radius = 1000
    success = np.nan
    rng = np.random.default_rng()
    X_diag = np.zeros((blocksize, blocksize), dtype=np.complex128)
    M_obc_diag = np.zeros((blocksize, blocksize), dtype=np.complex128)
    M_diag = np.zeros((blocksize, blocksize), dtype=np.complex128)
    M_upper = np.zeros((blocksize, blocksize), dtype=np.complex128)
    M_lower = np.zeros((blocksize, blocksize), dtype=np.complex128)
    while np.isnan(success):
        M_diag[:] = rng.uniform(size=(blocksize, blocksize)) +\
            1j * rng.uniform(size=(blocksize, blocksize))
        M_upper[:] = rng.uniform(size=(blocksize, blocksize)) +\
            1j * rng.uniform(size=(blocksize, blocksize))
        M_lower[:] = rng.uniform(size=(blocksize, blocksize)) +\
            1j * rng.uniform(size=(blocksize, blocksize))

        _, success, X_diag[:], M_obc_diag[:], _ = beyn(M_diag,
                                                       M_upper,
                                                       M_lower,
                                                       imaginary_limit,
                                                       contour_integration_radius,
                                                       'L',
                                                       function="G")

    assert np.allclose(M_obc_diag, M_lower @ X_diag @ M_upper)
    relative_error = np.linalg.norm(
        X_diag - np.linalg.inv(M_diag - M_obc_diag))/np.linalg.norm(X_diag)
    assert relative_error < 1e-13
    assert np.allclose(X_diag, np.linalg.inv(
        M_diag - M_obc_diag))
