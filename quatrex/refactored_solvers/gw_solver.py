# Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.

import numpy as np
from scipy.sparse import csr_matrix

from quatrex.refactored_solvers.polarization_solver import compute_polarization
from quatrex.refactored_solvers.screened_interaction_solver import screened_interaction_solver
from quatrex.refactored_solvers.gw_self_energy_solver import compute_gw_self_energy


def gw_solver(
    Screened_interaction_lesser: np.ndarray,
    Screened_interaction_greater: np.ndarray,
    Self_energy_retarded: np.ndarray,
    Self_energy_lesser: np.ndarray,
    Self_energy_greater: np.ndarray,
    G_lesser: np.ndarray,
    G_greater: np.ndarray,
    Coulomb_matrix: csr_matrix,
    Coulomb_matrix_at_neighbor_indices: np.ndarray,
    Neighboring_matrix_indices: dict[np.ndarray],
    energy_array,
    blocksize,
    screened_interaction_stepping_factor,
    self_energy_stepping_factor
):
    """
    Compute the GW self energy and screened interaction in place.
    """

    delta_energy = energy_array[1] - energy_array[0]
    number_of_energy_points = energy_array.size

    # compute the polarization
    (Polarization_lesser,
        Polarization_greater) = compute_polarization(
        G_lesser.T,
        G_greater.T,
        delta_energy)

    # compute the screened interaction
    (New_screened_interaction_lesser,
        New_screened_interaction_greater) = screened_interaction_solver(
        Coulomb_matrix,
        Polarization_lesser.T,
        Polarization_greater.T,
        number_of_energy_points,
        Neighboring_matrix_indices,
        blocksize)

    # Mix the solution of the previous step
    # with the new screened interaction solution
    # to achieve stability in convergence
    Screened_interaction_lesser[:] = \
        (1.0 - screened_interaction_stepping_factor) * New_screened_interaction_lesser \
        + screened_interaction_stepping_factor * \
        Screened_interaction_lesser
    Screened_interaction_greater[:] = \
        (1.0 - screened_interaction_stepping_factor) * New_screened_interaction_greater \
        + screened_interaction_stepping_factor * \
        Screened_interaction_greater

    # compute the gw self energy
    (New_self_energy_retarded,
        New_self_energy_lesser,
        New_self_energy_greater) = compute_gw_self_energy(
        G_lesser.T,
        G_greater.T,
        Screened_interaction_lesser.T,
        Screened_interaction_greater.T,
        Coulomb_matrix_at_neighbor_indices,
        delta_energy)

    # Mix the solution of the previous step
    # with the new gw self energy solution
    # to achieve stability in convergence
    Self_energy_retarded[:] = \
        (1.0 - self_energy_stepping_factor) * New_self_energy_retarded.T \
        + self_energy_stepping_factor * Self_energy_retarded
    Self_energy_lesser[:] = \
        (1.0 - self_energy_stepping_factor) * New_self_energy_lesser.T \
        + self_energy_stepping_factor * Self_energy_lesser
    Self_energy_greater[:] = \
        (1.0 - self_energy_stepping_factor) * New_self_energy_greater.T \
        + self_energy_stepping_factor * Self_energy_greater
