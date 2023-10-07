# Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.

from quatrex.GW.gold_solution import read_solution
from quatrex.refactored_solvers.polarization_solver import compute_polarization

import numpy as np
import pytest

@pytest.mark.parametrize(
    ", path_reference_solution",
    [
        ("/usr/scratch/mont-fort17/almaeder/test_gw/few_energy_iter1_no_filter.mat"),
        ("/usr/scratch/mont-fort17/almaeder/test_gw/few_energy_iter2_no_filter.mat"),
    ]
)
def test_polarization(
    path_reference_solution: str
):
    # load reference solution
    (energy_array_g, _, _, G_greater_reference, G_lesser_reference, _) = read_solution.load_x(
        path_reference_solution, "g")
    (energy_array_p, _, _, Polarization_greater_reference,
     Polarization_lesser_reference, _) = read_solution.load_x(path_reference_solution, "p")

    assert np.allclose(energy_array_g, energy_array_p)
    delta_energy = energy_array_g[1] - energy_array_g[0]

    (Polarization_greater, Polarization_lesser) = compute_polarization(
                                                        G_lesser_reference,
                                                        G_greater_reference,
                                                        delta_energy)
    # retarded polarization is not compared
    # because it is symmetrized in the screened interaction computation
    assert np.allclose(Polarization_greater, Polarization_greater_reference)
    assert np.allclose(Polarization_lesser, Polarization_lesser_reference)
