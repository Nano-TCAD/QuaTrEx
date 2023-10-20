# Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.
""" Functions to calculate the screened interaction on the cpu. See README.md for more information. """

import numpy as np
import typing
import numpy.typing as npt
from quatrex.block_tri_solvers import rgf_W

from quatrex.refactored_solvers.screened_interaction_solver import screened_interaction_solver

def p2w(
    hamiltionian_obj: object,
    energy: npt.NDArray[np.float64],
    Polarization_greater: npt.NDArray[np.complex128],
    Polarization_lesser: npt.NDArray[np.complex128],
    Polarization_retarded: npt.NDArray[np.complex128],
    Coulomb_matrix: npt.NDArray[np.complex128],
    dosw: npt.NDArray[np.complex128],
    new: npt.NDArray[np.complex128],
    npw: npt.NDArray[np.complex128],
    idx_e: npt.NDArray[np.int32],
    factor: npt.NDArray[np.float64],
    nbc
) -> typing.Tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128], npt.NDArray[np.complex128],
                  npt.NDArray[np.complex128], npt.NDArray[np.complex128], npt.NDArray[np.complex128], int, int]:
    """Calculates the screened interaction on the cpu.
    Uses mkl threading and pool threads.

    Args:
        hamiltionian_obj (object): Class containing the hamiltonian information
        energy (npt.NDArray[np.float64]): energy points
        Polarization_greater (npt.NDArray[np.complex128]): Greater polarization, vector of sparse matrices
        Polarization_lesser (npt.NDArray[np.complex128]): Lesser polarization, vector of sparse matrices
        Polarization_retarded (npt.NDArray[np.complex128]): Retarded polarization, vector of sparse matrices
        Coulomb_matrix (npt.NDArray[np.complex128]): Vh sparse matrix
        dosw (npt.NDArray[np.complex128]): density of state
        new (npt.NDArray[np.complex128]): density of state
        npw (npt.NDArray[np.complex128]): density of state
        factor (npt.NDArray[np.float64]): Smoothing factor
        mkl_threads (int, optional): Number of mkl threads used. Defaults to 1.
        worker_num(int, optional): Number of pool workers used. Defaults to 1.

    Returns:
        typing.Tuple[npt.NDArray[np.complex128],
                    npt.NDArray[np.complex128],
                    npt.NDArray[np.complex128],
                    npt.NDArray[np.complex128],
                    npt.NDArray[np.complex128],
                    npt.NDArray[np.complex128],
                    int, int ]:
    Diagonal/Upper block tensor (#blocks, blocksize, blocksize) of greater, lesser, retarded screened interaction.
    Number of blocks and block size after matrix multiplication
    """
    
    
    # number of energy points
    number_of_energy_points = energy.shape[0]

    # number of blocks
    nb = hamiltionian_obj.Bmin.shape[0]
    # start and end index of each block in python indexing
    bmax = hamiltionian_obj.Bmax - 1
    bmin = hamiltionian_obj.Bmin - 1

    # fix nbc to 2 for the given solution
    # todo calculate it
    # nbc = 2

    # block sizes after matrix multiplication
    bmax_mm = bmax[nbc - 1:nb:nbc]
    bmin_mm = bmin[0:nb:nbc]
    # number of blocks after matrix multiplication
    nb_mm = bmax_mm.size
    # larges block length after matrix multiplication
    lb_max_mm = np.max(bmax_mm - bmin_mm + 1)

    # create empty buffer for screened interaction
    # in block format
    Screened_interactions_retarded_diag = np.zeros((number_of_energy_points, nb_mm, lb_max_mm, lb_max_mm), dtype=np.complex128)
    Screened_interactions_retarded_upper = np.zeros((number_of_energy_points, nb_mm - 1, lb_max_mm, lb_max_mm), dtype=np.complex128)
    Screened_interaction_lesser_diag_blocks = np.zeros((number_of_energy_points, nb_mm, lb_max_mm, lb_max_mm), dtype=np.complex128)
    Screened_interaction_lesser_upper_blocks = np.zeros((number_of_energy_points, nb_mm - 1, lb_max_mm, lb_max_mm), dtype=np.complex128)
    Screened_interaction_greater_diag_blocks = np.zeros((number_of_energy_points, nb_mm, lb_max_mm, lb_max_mm), dtype=np.complex128)
    Screened_interaction_greater_upper_blocks = np.zeros((number_of_energy_points, nb_mm - 1, lb_max_mm, lb_max_mm), dtype=np.complex128)
    Susceptibility_diag = np.zeros((number_of_energy_points, nb_mm, lb_max_mm, lb_max_mm), dtype=np.complex128)

    # for ie in range(number_of_energy_points):
    #     rgf_W.rgf_w(Coulomb_matrix,
    #                 Polarization_greater[ie],
    #                 Polarization_lesser[ie],
    #                 Polarization_retarded[ie],
    #                 bmax,
    #                 bmin,
    #                 Screened_interaction_greater_diag_blocks[ie],
    #                 Screened_interaction_greater_upper_blocks[ie],
    #                 Screened_interaction_lesser_diag_blocks[ie],
    #                 Screened_interaction_lesser_upper_blocks[ie],
    #                 Screened_interactions_retarded_diag[ie],
    #                 Screened_interactions_retarded_upper[ie],
    #                 Susceptibility_diag[ie],
    #                 dosw[ie],
    #                 new[ie],
    #                 npw[ie],
    #                 nbc,
    #                 idx_e[ie],
    #                 factor[ie])
    
    blocksize = np.max(bmax - bmin + 1)
    screened_interaction_solver(
        Screened_interaction_lesser_diag_blocks,
        Screened_interaction_lesser_upper_blocks,
        Screened_interaction_greater_diag_blocks,
        Screened_interaction_greater_upper_blocks,
        Coulomb_matrix,
        Polarization_greater,
        Polarization_lesser,
        Polarization_retarded,
        number_of_energy_points,
        blocksize,
    )
        

    return (Screened_interaction_greater_diag_blocks,
        Screened_interaction_greater_upper_blocks,
        Screened_interaction_lesser_diag_blocks,
        Screened_interaction_lesser_upper_blocks,
        Screened_interactions_retarded_diag,
        Screened_interactions_retarded_upper,
        nb_mm,
        lb_max_mm)
