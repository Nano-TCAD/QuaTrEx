# Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.


import numpy as np
from numpy import fft


def compute_polarization(
    G_lesser: np.ndarray,
    G_greater: np.ndarray,
    delta_energy: float,
):
    scaling_factor = -1.0j * delta_energy / (np.pi)
    number_of_energy_points = G_greater.shape[1]

    # fft
    G_greater_fourier = fft.fft(G_greater, n=2*number_of_energy_points)
    # G_lesser is skewed symmetric
    G_lesser_transposed_fourier = fft.fft(-G_lesser.conj(), n=2*number_of_energy_points)

    # time reversal
    G_lesser_transposed_fourier_reversed = np.empty_like(G_greater_fourier, dtype=G_lesser.dtype)
    for j in range(2*number_of_energy_points):
        G_lesser_transposed_fourier_reversed[:, j] = G_lesser_transposed_fourier[:, -j]

    # multiply elementwise
    Polarization_greater_fourier = G_greater_fourier * G_lesser_transposed_fourier_reversed

    # ifft
    Polarization_greater = fft.ifft(Polarization_greater_fourier)
    # multiply with scaling factor
    Polarization_greater = scaling_factor * Polarization_greater

    # lesser polarization from identity (energy reversal skewed symmetry)
    Polarization_lesser = np.empty_like(G_greater_fourier, dtype=G_lesser.dtype)
    for j in range(2*number_of_energy_points):
        Polarization_lesser[:, j] = -np.conjugate(Polarization_greater[:, -j])

    # cutoff to original size
    return (Polarization_greater[:, :number_of_energy_points], Polarization_lesser[:, :number_of_energy_points])
