# Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.

import numpy as np
import pytest

from quatrex.refactored_solvers.polarization_solver import compute_polarization


def compute_polarization_reference(
    G_lesser: np.ndarray,
    G_greater: np.ndarray,
    delta_energy: float,
):
    scaling_factor = -1.0j * delta_energy / (np.pi)
    number_of_energy_points = G_greater.shape[1]

    # #energy
    number_of_elements_kept = G_greater.shape[0]

    # create polarization arrays
    Polarization_greater = np.empty_like(G_greater)
    Polarization_lesser = np.empty_like(G_greater)

    # evaluate convolution
    for ij in range(number_of_elements_kept):
        for energy in range(number_of_energy_points):
            accumulate_greater = 0
            accumulate_lesser = 0
            for energy_prime in range(energy, number_of_energy_points):
                accumulate_greater += G_greater[ij, energy_prime] *\
                    G_lesser[ij, energy_prime-energy].conj()
                accumulate_lesser += G_lesser[ij, energy_prime] *\
                    G_greater[ij, energy_prime-energy].conj()
            Polarization_greater[ij, energy] = - \
                scaling_factor * accumulate_greater
            Polarization_lesser[ij, energy] = - \
                scaling_factor * accumulate_lesser

    return (Polarization_lesser, Polarization_greater)

@pytest.mark.parametrize(
    "number_of_energy_points",
    [(1),
     (3),
     ]
)
@pytest.mark.parametrize(
    "number_of_elements_kept",
    [(1),
     (3),
     (11),
     ]
)
def test_compute_polarization(
    number_of_energy_points: int,
    number_of_elements_kept: int
):
    rng = np.random.default_rng()

    # not symmetrized, but both should be
    # if a function is used which does not assume the symmetry
    G_lesser = rng.random((number_of_elements_kept, number_of_energy_points)) +\
        1j * rng.random((number_of_elements_kept, number_of_energy_points))
    G_greater = rng.random((number_of_elements_kept, number_of_energy_points)) +\
        1j * rng.random((number_of_elements_kept, number_of_energy_points))

    delta_energy = rng.random()
    Polarization_lesser, Polarization_greater = compute_polarization(
        G_lesser, G_greater, delta_energy)
    Polarization_lesser_reference, Polarization_greater_reference = compute_polarization_reference(
        G_lesser, G_greater, delta_energy)
    assert np.allclose(Polarization_lesser, Polarization_lesser_reference)
    assert np.allclose(Polarization_greater, Polarization_greater_reference)
