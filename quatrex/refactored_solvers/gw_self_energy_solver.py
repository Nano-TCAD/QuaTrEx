# Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.

import numpy as np
from numpy import fft

def compute_gw_self_energy(
    G_lesser: np.ndarray,
    G_greater: np.ndarray,
    Screened_interaction_lesser: np.ndarray,
    Screened_interaction_greater: np.ndarray,
    Coulomb_matrix: np.ndarray,
    delta_energy: float
):
    # number of energy points and non zero elements ij
    number_of_energy_points = G_greater.shape[1]
    number_of_orbitals = G_greater.shape[0]
    energy_array = np.linspace(0, (number_of_energy_points-1)*delta_energy, number_of_energy_points)

    scaling_factor = 1.0j * delta_energy / (2*np.pi)
    scaling_factor_abs = delta_energy / (2*np.pi)

    # both lesser and greater screened interaction are skewed symmetric in orbital space (ij)
    Screened_interaction_lesser_transposed = -Screened_interaction_lesser.conj()
    Screened_interaction_greater_transposed = -Screened_interaction_greater.conj()

    # The screened interaction is extended to the full energy range (negative + positive energy range)
    # symmetric in energy
    Screened_interaction_lesser_full_range = np.zeros((number_of_orbitals,
                        2*number_of_energy_points-1), dtype=G_lesser.dtype)
    Screened_interaction_greater_full_range = np.zeros((number_of_orbitals,
                        2*number_of_energy_points-1), dtype=G_lesser.dtype)

    Screened_interaction_lesser_full_range[:, (number_of_energy_points-1):] = Screened_interaction_lesser
    Screened_interaction_lesser_full_range[:, :number_of_energy_points-1] = \
        np.flip(Screened_interaction_greater_transposed, axis=-1)[:, :(number_of_energy_points-1)]

    Screened_interaction_greater_full_range[:, (number_of_energy_points-1):] = Screened_interaction_greater
    Screened_interaction_greater_full_range[:, :number_of_energy_points-1] = \
        np.flip(Screened_interaction_lesser_transposed, axis=-1)[:, :(number_of_energy_points-1)]

    # fft
    G_greater_fourier = fft.fft(G_greater, n=2*number_of_energy_points)
    G_lesser_fourier = fft.fft(G_lesser, n=2*number_of_energy_points)
    Screened_interaction_greater_fourier = fft.fft(Screened_interaction_greater, n=2*number_of_energy_points)
    Screened_interaction_lesser_fourier = fft.fft(Screened_interaction_lesser, n=2*number_of_energy_points)
    Screened_interaction_greater_transposed_fourier = fft.fft(Screened_interaction_greater_transposed, n=2*number_of_energy_points)
    Screened_interaction_lesser_transposed_fourier = fft.fft(Screened_interaction_lesser_transposed, n=2*number_of_energy_points)

    # fft of energy reversed
    G_greater_reversed_fourier = fft.fft(np.flip(G_greater, axis=-1), n=2*number_of_energy_points)
    G_lesser_reversed_fourier = fft.fft(np.flip(G_lesser, axis=-1), n=2*number_of_energy_points)


    # Sigma lesser greater are computed from two terms
    # because the polarization computation would increase the energy range, but half is cutoff
    # this cutoff is compensated by the second term

    # multiply elementwise for sigma_1 the normal term
    Sigma_greater_fourier = G_greater_fourier*Screened_interaction_greater_fourier
    Sigma_lesser_fourier = G_lesser_fourier*Screened_interaction_lesser_fourier


    # multiply elementwise the energy reversed with difference of transposed and energy zero
    Sigma_greater_mend_fourier = G_greater_reversed_fourier*\
        (Screened_interaction_lesser_transposed_fourier - Screened_interaction_lesser_transposed[:, 0].reshape(-1,1))
    Sigma_lesser_mend_fourier = G_lesser_reversed_fourier*\
        (Screened_interaction_greater_transposed_fourier - Screened_interaction_greater_transposed[:, 0].reshape(-1,1))


    # ifft, multiply with scaling factor and cutoff to original size
    Sigma_greater = scaling_factor*fft.ifft(Sigma_greater_fourier)[:,:number_of_energy_points]
    Sigma_lesser = scaling_factor*fft.ifft(Sigma_lesser_fourier)[:,:number_of_energy_points]

    Sigma_greater_mend = scaling_factor*np.flip(fft.ifft(Sigma_greater_mend_fourier)[:,:number_of_energy_points], axis=-1)
    Sigma_lesser_mend = scaling_factor*np.flip(fft.ifft(Sigma_lesser_mend_fourier)[:,:number_of_energy_points], axis=-1)

    Sigma_greater += Sigma_greater_mend
    Sigma_lesser += Sigma_lesser_mend


    Sigma_retarded = 1.0j*np.imag(Sigma_greater-Sigma_lesser)/2

    # Calculating the truncated Fock part
    # Using the principal value integral method for sigma retarded    
    G_lesser_density = np.imag(np.sum(G_lesser, axis=1))
    Sigma_density = -scaling_factor_abs*(G_lesser_density*Coulomb_matrix).reshape(-1, 1)
    Sigma_retarded_mend = np.repeat(Sigma_density, number_of_energy_points, axis=-1)

    # TODO ask Leo about meaning of below code and naming of variables
    energy_array_inverse = np.concatenate((-1.0/(energy_array[-1:0:-1]), np.array([0.0], dtype=np.float64),
                                  1/(energy_array[1:]), np.array([1/(energy_array[-1] + delta_energy)], dtype=np.float64)))
    energy_array_inverse_fourier = fft.fft(energy_array_inverse)

    SGmSL_fourier = fft.fft(2*Sigma_retarded, n=2*number_of_energy_points)
    rSigma_retarded_fourier = SGmSL_fourier*energy_array_inverse_fourier
    rSigma_retarded = 2*scaling_factor*fft.ifft(rSigma_retarded_fourier)[:, number_of_energy_points-1:-1]


    Sigma_retarded += rSigma_retarded/2 + Sigma_retarded_mend

    return (Sigma_greater, Sigma_lesser, Sigma_retarded)
