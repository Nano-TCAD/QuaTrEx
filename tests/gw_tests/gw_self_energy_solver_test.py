# Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.

import numpy as np
import pytest

from quatrex.refactored_solvers.gw_self_energy_solver import compute_gw_self_energy_lesser_greater, compute_gw_self_energy_lesser_greater_correction


def compute_gw_self_energy_lesser_greater_reference(
    G_lesser: np.ndarray,
    G_greater: np.ndarray,
    Screened_interaction_lesser: np.ndarray,
    Screened_interaction_greater: np.ndarray,
    delta_energy: float
):
    scaling_factor = 1.0j * delta_energy / (2*np.pi)
    number_of_energy_points = G_greater.shape[1]

    # #energy
    number_of_elements_kept = G_greater.shape[0]

    # create self energy arrays
    Self_energy_lesser = np.empty_like(G_greater)
    Self_energy_greater = np.empty_like(G_greater)

    # evaluate convolution
    for ij in range(number_of_elements_kept):
        for energy in range(number_of_energy_points):
            accumulate_greater = 0
            accumulate_lesser = 0
            for energy_prime in range(0, energy+1):
                accumulate_greater += G_greater[ij, energy_prime] *\
                    Screened_interaction_greater[ij, energy-energy_prime]
                accumulate_lesser += G_lesser[ij, energy_prime] *\
                    Screened_interaction_lesser[ij, energy-energy_prime]
            Self_energy_greater[ij, energy] = \
                scaling_factor * accumulate_greater
            Self_energy_lesser[ij, energy] = \
                scaling_factor * accumulate_lesser

    return (Self_energy_lesser, Self_energy_greater)


def compute_gw_self_energy_lesser_greater_correction_reference(
    G_lesser: np.ndarray,
    G_greater: np.ndarray,
    Screened_interaction_lesser: np.ndarray,
    Screened_interaction_greater: np.ndarray,
    delta_energy: float
):
    scaling_factor = 1.0j * delta_energy / (2*np.pi)
    number_of_energy_points = G_greater.shape[1]

    # #energy
    number_of_elements_kept = G_greater.shape[0]

    # create self energy arrays
    Self_energy_lesser = np.empty_like(G_greater)
    Self_energy_greater = np.empty_like(G_greater)

    # both lesser and greater screened interaction are skewed symmetric in orbital space (ij)
    Screened_interaction_lesser_transposed = -Screened_interaction_lesser.conj()
    Screened_interaction_greater_transposed = -Screened_interaction_greater.conj()

    # evaluate convolution
    for ij in range(number_of_elements_kept):
        for energy in range(number_of_energy_points):
            accumulate_greater = 0
            accumulate_lesser = 0
            for energy_prime in range(0, number_of_energy_points-energy):
                accumulate_greater += G_greater[ij, energy + energy_prime] *\
                    Screened_interaction_lesser_transposed[ij, energy_prime]
                accumulate_lesser += G_lesser[ij, energy + energy_prime] *\
                    Screened_interaction_greater_transposed[ij, energy_prime]

            # to not double count the energy_prime=0 term
            accumulate_greater -= G_greater[ij, energy] *\
                Screened_interaction_lesser_transposed[ij, 0]
            accumulate_lesser -= G_lesser[ij, energy] *\
                Screened_interaction_greater_transposed[ij, 0]

            Self_energy_greater[ij, energy] = \
                scaling_factor * accumulate_greater
            Self_energy_lesser[ij, energy] = \
                scaling_factor * accumulate_lesser

    return (Self_energy_lesser, Self_energy_greater)


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
def test_compute_gw_self_energy_lesser_greater(
    number_of_energy_points: int,
    number_of_elements_kept: int
):
    """
    Test the optimized version of the lesser and greater self energy
    against the reference convolution implementation
    """
    rng = np.random.default_rng()

    # not symmetrized, but both should be
    # if a function is used which does not assume the symmetry
    G_lesser = rng.random((number_of_elements_kept, number_of_energy_points)) +\
        1j * rng.random((number_of_elements_kept, number_of_energy_points))
    G_greater = rng.random((number_of_elements_kept, number_of_energy_points)) +\
        1j * rng.random((number_of_elements_kept, number_of_energy_points))
    Screened_interaction_lesser = rng.random((number_of_elements_kept, number_of_energy_points)) +\
        1j * rng.random((number_of_elements_kept, number_of_energy_points))
    Screened_interaction_greater = rng.random((number_of_elements_kept, number_of_energy_points)) +\
        1j * rng.random((number_of_elements_kept, number_of_energy_points))

    delta_energy = rng.random()
    Self_energy_lesser, Self_energy_greater = compute_gw_self_energy_lesser_greater(
        G_lesser, G_greater,
        Screened_interaction_lesser, Screened_interaction_greater, delta_energy)
    Self_energy_lesser_reference, Self_energy_greater_reference = compute_gw_self_energy_lesser_greater_reference(
        G_lesser, G_greater,
        Screened_interaction_lesser, Screened_interaction_greater, delta_energy)
    assert np.allclose(Self_energy_lesser, Self_energy_lesser_reference)
    assert np.allclose(Self_energy_greater, Self_energy_greater_reference)


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
def test_compute_gw_self_energy_lesser_greater_correction(
    number_of_energy_points: int,
    number_of_elements_kept: int
):
    """
    Test the computation of the correction term for the lesser and greater self energy
    against the reference convolution implementation.
    The correction term is due to the energy range cutoff in the polarization calculation.
    """
    rng = np.random.default_rng()

    # not symmetrized, but both should be
    # if a function is used which does not assume the symmetry
    G_lesser = rng.random((number_of_elements_kept, number_of_energy_points)) +\
        1j * rng.random((number_of_elements_kept, number_of_energy_points))
    G_greater = rng.random((number_of_elements_kept, number_of_energy_points)) +\
        1j * rng.random((number_of_elements_kept, number_of_energy_points))
    Screened_interaction_lesser = rng.random((number_of_elements_kept, number_of_energy_points)) +\
        1j * rng.random((number_of_elements_kept, number_of_energy_points))
    Screened_interaction_greater = rng.random((number_of_elements_kept, number_of_energy_points)) +\
        1j * rng.random((number_of_elements_kept, number_of_energy_points))

    delta_energy = rng.random()
    Self_energy_lesser, Self_energy_greater = compute_gw_self_energy_lesser_greater_correction(
        G_lesser, G_greater,
        Screened_interaction_lesser, Screened_interaction_greater, delta_energy)
    Self_energy_lesser_reference, Self_energy_greater_reference = compute_gw_self_energy_lesser_greater_correction_reference(
        G_lesser, G_greater,
        Screened_interaction_lesser, Screened_interaction_greater, delta_energy)
    assert np.allclose(Self_energy_lesser, Self_energy_lesser_reference)
    assert np.allclose(Self_energy_greater, Self_energy_greater_reference)
