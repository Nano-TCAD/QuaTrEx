# Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.
"""
Example on analysing a sequence of self-energies arising from a self-consistent GW calculation
Device: off-current state of a CNT TFET.
GW-Self-energy: retarded part and only diagonal elements
Boundary Self-Energy: Calculated for both contacts with beyn

To calculate retarded Green's function:
[E - H - SigmaR_{boundary} - SigmaR_{GW}] G(E)^r = I

SigmaR can be read for 32 iterations for 1000 energy points each.
"""
import sys
import numpy as np
import numpy.typing as npt
import os
import argparse
import pickle
import mpi4py
from scipy import sparse
import time

mpi4py.rc.initialize = False  # do not initialize MPI automatically
mpi4py.rc.finalize = False  # do not finalize MPI automatically
from mpi4py import MPI

from quatrex.bandstructure.calc_band_edge import get_band_edge_mpi, get_band_edge_interpol, get_band_edge_mpi_interpol
from quatrex.GW.gold_solution import read_solution
from quatrex.GW.screenedinteraction.kernel import p2w_cpu
from quatrex.GW.coulomb_matrix.read_coulomb_matrix import load_V
from quatrex.GreensFunction import calc_GF_pool
from quatrex.OMEN_structure_matrices import OMENHamClass
from quatrex.OMEN_structure_matrices.construct_CM import construct_coulomb_matrix
from quatrex.utils import change_format
from quatrex.utils import utils_gpu
from quatrex.utils.bsr import bsr_matrix
from quatrex.utils.matrix_creation import get_number_connected_blocks

if utils_gpu.gpu_avail():
    try:
        from quatrex.GW.polarization.kernel import g2p_gpu
        from quatrex.GW.selfenergy.kernel import gw2s_gpu
    except ImportError:
        print("GPU import error, make sure you have the right GPU driver and CUDA version installed")

main_path = os.path.abspath(os.path.dirname(__file__))
parent_path = os.path.abspath(os.path.join(main_path, ".."))

if __name__ == "__main__":
    MPI.Init_thread(required=MPI.THREAD_FUNNELED)
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    name = MPI.Get_processor_name()

    scratch_path = "/usr/scratch/mont-fort21/chexia/"
    hamiltonian_path = os.path.join(scratch_path, "CNT_evensort48new")

    self_energy_path = "/usr/scratch/mont-fort17/dleonard/QUATREX/results/80TFET72/"

    #Reading Hamiltonian with Hamiltonian class (adding electrostatic potential)
    no_orb = np.array([2, 3])
    Vappl = 0
    energy = np.linspace(-5, -3, 1000, endpoint=True, dtype=float)  # Energy Vector
    Idx_e = np.arange(energy.shape[0])  # Energy Index Vector
    hamiltonian_obj = OMENHamClass.Hamiltonian(hamiltonian_path, no_orb, Vappl=Vappl, rank=rank, potential_type='atomic')
    serial_ham = pickle.dumps(hamiltonian_obj)
    broadcasted_ham = comm.bcast(serial_ham, root=0)
    hamiltonian_obj = pickle.loads(broadcasted_ham)

    # Extract neighbor indices
    rows = hamiltonian_obj.rows
    columns = hamiltonian_obj.columns

    # Only keep diagonals of P and Sigma
    rows = np.arange(hamiltonian_obj.NH, dtype=np.int32)
    columns = np.arange(hamiltonian_obj.NH, dtype=np.int32)

    # hamiltonian object has 1-based indexing, block sizes
    bmax = hamiltonian_obj.Bmax - 1
    bmin = hamiltonian_obj.Bmin - 1

    # Left and right Block sizes for boundary self-energy analysis
    LBsize = bmax[0] - bmin[0] + 1
    RBsize = bmax[-1] - bmin[-1] + 1

    # index information
    ij2ji: npt.NDArray[np.int32] = change_format.find_idx_transposed(rows, columns)
    denergy: npt.NDArray[np.double] = energy[1] - energy[0]
    ne: np.int32 = np.int32(energy.shape[0])
    no: np.int32 = np.int32(columns.shape[0]) # number of elem in GW self energy
    nao: np.int64 = np.max(bmax) + 1 # number of atomic orbitals (matrix size)

    # Analysing an arbitrary iteration
    iteration = 10 # 0->31 iterations performed
    # Reading GW self-energy
    SigmaR_GW = np.load(os.path.join(self_energy_path, "sigma_R_{}.npy".format(iteration)))
    # Reading left boundary self-energy
    SigmaR_Bl = np.load(os.path.join(self_energy_path, "sigma_RBL_{}.npy".format(iteration)))
    # Reading right boundary self-energy
    SigmaR_Br = np.load(os.path.join(self_energy_path, "sigma_RBR_{}.npy".format(iteration)))

    # Printing shapes
    print("SigmaR_GW shape: ", SigmaR_GW.shape)
    print("SigmaR_Bl shape: ", SigmaR_Bl.shape)

    # Bringing GW self-energies to sparse format
    sr_h2g_vec = change_format.sparse2vecsparse_v2(SigmaR_GW, rows, columns, nao)

    # Creating [E - H - SigmaR_GW - SigmaR_B] matrix for an arbitrary energy index
    idx_e = 900 # this is the ernergy point (0->999)

    # [E - H - SigmaR_GW] matrix
    # matrix is CSR
    matrix = (energy[idx_e] + 1j * 1e-12) * hamiltonian_obj.Overlap['H_4'] -  hamiltonian_obj.Hamiltonian['H_4'] - sr_h2g_vec[idx_e]

    # adding boundary elements
    matrix[:LBsize, :LBsize] -= SigmaR_Bl[idx_e]
    matrix[-RBsize:, -RBsize:] -= SigmaR_Br[idx_e]

    # Calculating Green's function
    Gr = np.linalg.inv(matrix.toarray())

    import matplotlib.pyplot as plt
    plt.matshow(np.abs(SigmaR_Bl[idx_e]))
    plt.matshow(np.abs(SigmaR_Br[idx_e]))
    #plt.matshow(np.abs(hamiltonian_obj.Hamiltonian['H_4'].toarray()))
    plt.matshow(np.abs(matrix.toarray()))
    plt.matshow(np.abs(Gr))
    plt.show()

    print(Gr.shape)


