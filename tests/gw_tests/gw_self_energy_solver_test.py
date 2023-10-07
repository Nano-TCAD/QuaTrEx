# Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.

from quatrex.GW.gold_solution import read_solution
from quatrex.refactored_solvers.gw_self_energy_solver import compute_gw_self_energy

import numpy as np
import pytest

@pytest.mark.parametrize(
    "path_reference_solution",
    [
        ("/usr/scratch/mont-fort17/almaeder/test_gw/few_energy_iter1_no_mixing.mat"),
        ("/usr/scratch/mont-fort17/almaeder/test_gw/few_energy_iter2_no_mixing.mat"),
    ]
)
def test_compute_gw_self_energy(
    path_reference_solution: str
):
    # load reference solution
    (energy_array_g, _, _, G_greater_reference, G_lesser_reference, _) = read_solution.load_x(
        path_reference_solution, "g")

    (energy_array_w, _, _, Screened_interaction_greater_reference,
     Screened_interaction_lesser_reference, _) = read_solution.load_x(
        path_reference_solution, "w")

    (energy_array_s, _, _, Sigma_greater_reference,
     Sigma_lesser_reference, Sigma_retarded_reference) = read_solution.load_x(path_reference_solution, "s")

    (_, _, Coulomb_matrix) = read_solution.load_v(path_reference_solution)

    assert np.allclose(energy_array_g, energy_array_s)
    assert np.allclose(energy_array_g, energy_array_w)
    delta_energy = energy_array_g[1] - energy_array_g[0]

    (Sigma_greater,  Sigma_lesser, Sigma_retarded) = compute_gw_self_energy(
        G_lesser_reference,
        G_greater_reference,
        Screened_interaction_lesser_reference,
        Screened_interaction_greater_reference,
        Coulomb_matrix,
        delta_energy)
    # retarded polarization is not compared
    # because it is symmetrized in the screened interaction computation
    assert np.allclose(Sigma_greater, Sigma_greater_reference)
    assert np.allclose(Sigma_lesser, Sigma_lesser_reference)
    assert np.allclose(Sigma_retarded, Sigma_retarded_reference)
