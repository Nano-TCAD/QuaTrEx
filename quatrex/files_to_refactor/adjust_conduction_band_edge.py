# Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.

import numpy as np
from scipy.linalg import eig
from quatrex.refactored_utils.utils import flattened_to_list_of_csr


def calc_bandstructure_interpol(
        energy_points,
        Overlap_matrix,
        Hamiltonian,
        target_energy,
        Self_energy_retarded_list,
        Self_energy_lesser_list,
        Self_energy_greater_list,
        index_energy_lower_edge,
        blocksize
):


    energy_lower_target = energy_points[index_energy_lower_edge]
    energy_higher_target = energy_points[index_energy_lower_edge + 1]

    # TODO refacture into functions after knowing what it does
    # does interpolation and symmetrization of self-energy

    # TODO: why only imaginary part
    Self_energy_lesser_lower_energy = 1j * np.imag(Self_energy_lesser_list[index_energy_lower_edge])
    Self_energy_lesser_upper_energy = 1j * np.imag(Self_energy_lesser_list[index_energy_lower_edge + 1])

    # linear interpolation
    Self_energy_lesser = Self_energy_lesser_lower_energy +\
                        (Self_energy_lesser_upper_energy -\
                         Self_energy_lesser_lower_energy) /\
                        (energy_higher_target - energy_lower_target) *\
                        (target_energy - energy_lower_target)

    # anti hermitian symmetrization
    Self_energy_lesser = (Self_energy_lesser - Self_energy_lesser.conj().T) / 2

    # TODO: why only imaginary part
    Self_energy_greater_lower_energy = 1j * np.imag(Self_energy_greater_list[index_energy_lower_edge])
    Self_energy_greater_upper_energy = 1j * np.imag(Self_energy_greater_list[index_energy_lower_edge + 1])

    # linear interpolation
    Self_energy_greater = Self_energy_greater_lower_energy +\
                        (Self_energy_greater_upper_energy -\
                         Self_energy_greater_lower_energy) /\
                        (energy_higher_target - energy_lower_target) *\
                        (target_energy - energy_lower_target)

    # anti hermitian symmetrization
    Self_energy_greater = (Self_energy_greater - Self_energy_greater.conj().T) / 2

    Self_energy_retarded_lower_energy = Self_energy_retarded_list[index_energy_lower_edge]
    Self_energy_retarded_upper_energy = Self_energy_retarded_list[index_energy_lower_edge + 1]

    # linear interpolation
    Self_energy_retarded = Self_energy_retarded_lower_energy +\
                        (Self_energy_retarded_upper_energy -\
                         Self_energy_retarded_lower_energy) /\
                        (energy_higher_target - energy_lower_target) *\
                        (target_energy - energy_lower_target)

    # TODO: reasoning (i.e. the retarded self energy is not necessarily anti hermitian)
    Self_energy_retarded = np.real(Self_energy_retarded) +\
                        1j * np.imag(Self_energy_greater - Self_energy_lesser) / 2
    Self_energy_retarded = (Self_energy_retarded + Self_energy_retarded.T) / 2

    # TODO: avoid full addition if only blocks are used
    Hamiltonian = Hamiltonian + Self_energy_retarded

    # TODO: replace with bsparse
    # TODO: read out first blocks and
    # then do the above symmetrization part (lower comp cost)
    H00 = Hamiltonian[:blocksize, :blocksize].toarray()
    H01 = Hamiltonian[:blocksize, blocksize:2 * blocksize].toarray()
    H10 = Hamiltonian[blocksize:2 * blocksize, :blocksize].toarray()

    S00 = Overlap_matrix[:blocksize, :blocksize].toarray()
    S01 = Overlap_matrix[:blocksize, blocksize:2 * blocksize].toarray()
    S10 = Overlap_matrix[blocksize:2 * blocksize, :blocksize].toarray()

    # calculate eigenvalues
    eigenvalues = np.real(eig(H00 + H01 + H10,
                            b=S00 + S01 + S10,
                            right=False))

    return eigenvalues


def adjust_conduction_band_edge(
        energy_conduction_band_energy,
        energy_points,
        Overlap_matrix,
        Hamiltonian,
        Self_energy_retarded_flattened,
        Self_energy_lesser_flattened,
        Self_energy_greater_flattened,
        Neighboring_matrix_indices,
        blocksize
):
    """
    The whole band strcuture can be shifted in the GW iteration.
    Thus, one has to track the conduction band edge.
    The fermi energy has to be changed accordingly.

    Idea:
    Christian Stieger:
    H_00 + H_01 e^(-ik delta z) + H_10 e^(ik delta z) = H(k)
    solve at k=0 for eigenvalues to get band edge

    energy peaks shifted only by a small amount
    thus the peak closed to the old energy_conduction_band_energy
    is the new energy_conduction_band_energy

    this is done in two steps:
    1. get a first estimate of the correct peaks
    2. refine the position of the correct peaks
    
    The peak closest to the old edge is the new edge.
    """

    # input is flattened, but list of csr is needed
    Self_energy_retarded_list = flattened_to_list_of_csr(
        Self_energy_retarded_flattened,
        Neighboring_matrix_indices,
        Hamiltonian.shape[0])
    Self_energy_lesser_list = flattened_to_list_of_csr(
        Self_energy_lesser_flattened,
        Neighboring_matrix_indices,
        Hamiltonian.shape[0])
    Self_energy_greater_list = flattened_to_list_of_csr(
        Self_energy_greater_flattened,
        Neighboring_matrix_indices,
        Hamiltonian.shape[0])

    # First step: get a first estimate of the CB edge

    # map conduction band energy to the grid
    index_old_conduction_band_energy = np.argmin(np.abs(energy_points -
                                                          energy_conduction_band_energy))

    # get energy index of energy lower than conduction band minimum
    if ((energy_points[index_old_conduction_band_energy] > energy_conduction_band_energy)
            and (index_old_conduction_band_energy > 0)):
        index_old_conduction_band_energy -= 1

    energy_peaks_estimate = calc_bandstructure_interpol(energy_points,
                                     Overlap_matrix,
                                     Hamiltonian,
                                     energy_conduction_band_energy,
                                     Self_energy_retarded_list,
                                     Self_energy_lesser_list,
                                     Self_energy_greater_list,
                                     index_old_conduction_band_energy,
                                     blocksize)


    # find estimate of the conduction band energy
    # peak energy close to the conduction band energy
    # since the peaks are shifted only by a small amount
    index_estimate = np.argmin(np.abs(energy_peaks_estimate - energy_conduction_band_energy))
    conduction_band_energy_estimate = energy_peaks_estimate[index_estimate]


    # Second step: refine the position of the CB edge
    index_estimate_conduction_band_energy = np.argmin(np.abs(energy_points - conduction_band_energy_estimate))

    # get energy index of energy lower than refinded energy
    if ((energy_points[index_estimate_conduction_band_energy] > conduction_band_energy_estimate)
            and (index_estimate_conduction_band_energy > 0)):
        index_estimate_conduction_band_energy -= 1

    energy_peaks = calc_bandstructure_interpol(energy_points,
                                     Overlap_matrix,
                                     Hamiltonian,
                                     conduction_band_energy_estimate,
                                     Self_energy_retarded_list,
                                     Self_energy_lesser_list,
                                     Self_energy_greater_list,
                                     index_estimate_conduction_band_energy,
                                     blocksize)

    # energy close to the conduction band energy
    # of the first refinement step/old conduction band energy
    # is the new conduction band energy
    index_new = np.argmin(np.abs(energy_peaks - conduction_band_energy_estimate))
    conduction_band_energy_new = energy_peaks[index_new]

    return conduction_band_energy_new
