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
        index_energy_close_target,
        blocksize
):

    energy_lower_target = energy_points[index_energy_close_target]
    energy_higher_target = energy_points[index_energy_close_target + 1]

    # TODO refacture into functions after knowing what it does
    SigL1 = 1j * np.imag(Self_energy_lesser_list[index_energy_close_target])
    SigL2 = 1j * np.imag(Self_energy_lesser_list[index_energy_close_target + 1])
    SigL = SigL1 + (SigL2 - SigL1) / (energy_higher_target -
                                      energy_lower_target) * (target_energy - energy_lower_target)
    SigL = (SigL - SigL.conj().T) / 2

    SigG1 = 1j * np.imag(Self_energy_greater_list[index_energy_close_target])
    SigG2 = 1j * \
        np.imag(Self_energy_greater_list[index_energy_close_target + 1])
    SigG = SigG1 + (SigG2 - SigG1) / (energy_higher_target -
                                      energy_lower_target) * (target_energy - energy_lower_target)
    SigG = (SigG - SigG.conj().T) / 2

    SigR1 = Self_energy_retarded_list[index_energy_close_target]
    SigR2 = Self_energy_retarded_list[index_energy_close_target + 1]
    SigR = SigR1 + (SigR2 - SigR1) / (energy_higher_target -
                                      energy_lower_target) * (target_energy - energy_lower_target)
    SigR = np.real(SigR) + 1j * np.imag(SigG - SigL) / 2
    SigR = (SigR + SigR.T) / 2

    Hamiltonian = Hamiltonian + SigR

    H00 = Hamiltonian[:blocksize, :blocksize].toarray()
    H01 = Hamiltonian[:blocksize, blocksize:2 * blocksize].toarray()
    H10 = Hamiltonian[blocksize:2 * blocksize, :blocksize].toarray()

    S00 = Overlap_matrix[:blocksize, :blocksize].toarray()
    S01 = Overlap_matrix[:blocksize, blocksize:2 * blocksize].toarray()
    S10 = Overlap_matrix[blocksize:2 * blocksize, :blocksize].toarray()

    # calculate eigenvalues of
    eigenvalues = np.sort(np.real(eig(H00 + H01 + H10,
                                      b=S00 + S01 + S10,
                                      right=False)))

    return eigenvalues


def adjust_conduction_band_edge(
        energy_conduction_band_minimum,
        energy_points,
        Overlap_matrix,
        Hamiltonian,
        Self_energy_retarded_flattened,
        Self_energy_lesser_flattened,
        Self_energy_greater_flattened,
        Neighboring_matrix_indices,
        blocksize
):

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
    index_energy_close_conduction_band = np.argmin(np.abs(energy_points -
                                                          energy_conduction_band_minimum))

    # get energy index of energy lower than conduction band minimum
    if ((energy_points[index_energy_close_conduction_band] > energy_conduction_band_minimum)
            and (index_energy_close_conduction_band > 0)):
        index_energy_close_conduction_band -= 1

    Ek = calc_bandstructure_interpol(energy_points,
                                     Overlap_matrix,
                                     Hamiltonian,
                                     energy_conduction_band_minimum,
                                     Self_energy_retarded_list,
                                     Self_energy_lesser_list,
                                     Self_energy_greater_list,
                                     index_energy_close_conduction_band,
                                     blocksize)

    ind_ek_plus = np.argmin(np.abs(Ek - energy_conduction_band_minimum))
    refinded_energy = Ek[ind_ek_plus]

    # Second step: refine the position of the CB edge
    index_energy_refined = np.argmin(np.abs(energy_points - refinded_energy))

    if ((energy_points[index_energy_refined] > refinded_energy)
            and (index_energy_refined > 0)):
        index_energy_refined -= 1

    Ek = calc_bandstructure_interpol(energy_points,
                                     Overlap_matrix,
                                     Hamiltonian,
                                     refinded_energy,
                                     Self_energy_retarded_list,
                                     Self_energy_lesser_list,
                                     Self_energy_greater_list,
                                     index_energy_refined,
                                     blocksize)

    ind_ek = np.argmin(np.abs(Ek - refinded_energy))
    ECmin = Ek[ind_ek]

    return ECmin
