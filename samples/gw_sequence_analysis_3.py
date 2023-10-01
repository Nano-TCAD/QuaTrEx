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
from scipy import linalg
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

global max_iter
max_iter = 0

def print_maxiter(args):
    global max_iter
    max_iter += 1

    
    
    
    
def get_Hamiltonian(
    
):    
    """
    Return Hamiltonian object, energy vector, energy index vector and path to self-energy files.
    """
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
    
    return hamiltonian_obj, energy, Idx_e, self_energy_path



def get_A(
    Hamiltonian_obj : OMENHamClass.Hamiltonian,
    self_energy_path : str,
    energy : npt.NDArray[np.double],
    iteration : int,
    idx_energy : int
):
    """
    Return A = [E - H - SigmaR_GW - SigmaR_B] CSR matrix for an arbitrary energy 
    index and iteration number.
    """
    
    # Extract neighbor indices
    rows = Hamiltonian_obj.rows
    columns = Hamiltonian_obj.columns

    # Only keep diagonals of P and Sigma
    rows = np.arange(Hamiltonian_obj.NH, dtype=np.int32)
    columns = np.arange(Hamiltonian_obj.NH, dtype=np.int32)

    # hamiltonian object has 1-based indexing, block sizes
    bmax = Hamiltonian_obj.Bmax - 1
    bmin = Hamiltonian_obj.Bmin - 1

    # Left and right Block sizes for boundary self-energy analysis
    LBsize = bmax[0] - bmin[0] + 1
    RBsize = bmax[-1] - bmin[-1] + 1

    # index information
    ij2ji: npt.NDArray[np.int32] = change_format.find_idx_transposed(rows, columns)
    denergy: npt.NDArray[np.double] = energy[1] - energy[0]
    ne: np.int32 = np.int32(energy.shape[0])
    no: np.int32 = np.int32(columns.shape[0]) # number of elem in GW self energy
    nao: np.int64 = np.max(bmax) + 1 # number of atomic orbitals (matrix size)
    
    # Reading GW self-energy
    SigmaR_GW = np.load(os.path.join(self_energy_path, "sigma_R_{}.npy".format(iteration)))
    # Reading left boundary self-energy
    SigmaR_Bl = np.load(os.path.join(self_energy_path, "sigma_RBL_{}.npy".format(iteration)))
    # Reading right boundary self-energy
    SigmaR_Br = np.load(os.path.join(self_energy_path, "sigma_RBR_{}.npy".format(iteration)))

    # Bringing GW self-energies to sparse format
    sr_h2g_vec = change_format.sparse2vecsparse_v2(SigmaR_GW, rows, columns, nao)

    # Creating A = [E - H - SigmaR_GW - SigmaR_B] CSR matrix for an arbitrary energy index
    E = (energy[idx_energy] + 1j * 1e-12) * Hamiltonian_obj.Overlap['H_4']
    H = Hamiltonian_obj.Hamiltonian['H_4']
    SigmaR = sr_h2g_vec[idx_energy]
    
    A = E - H - SigmaR

    # Adding boundary elements
    A[:LBsize, :LBsize] -= SigmaR_Bl[idx_energy]
    A[-RBsize:, -RBsize:] -= SigmaR_Br[idx_energy]
    
    return A
            
            
            
if __name__ == "__main__":
    MPI.Init_thread(required=MPI.THREAD_FUNNELED)
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    name = MPI.Get_processor_name()

    hamiltonian_obj, energy, Idx_e, self_energy_path = get_Hamiltonian()

    iteration = 10
    idx_energy = 0

    A_1 = get_A(hamiltonian_obj, self_energy_path, energy, iteration, idx_energy)
    A_2 = get_A(hamiltonian_obj, self_energy_path, energy, iteration+1, idx_energy)

    A_1_dense = A_1.toarray()
    A_2_dense = A_2.toarray()
    
    matrice_size = A_2_dense.shape[0]
    
    A_1_inv = np.linalg.inv(A_1_dense)
    
    A_2_preconditionned = A_1_inv @ A_2_dense
    
    cond_A_2 = np.linalg.cond(A_2_dense)
    cond_A_2_preconditionned = np.linalg.cond(A_2_preconditionned)
    
    print("It: " + str(iteration) + ", IdxE: " + str(idx_energy) + " Cond_A_2: " + str(cond_A_2) + " Cond_A_2_preconditionned: " + str(cond_A_2_preconditionned))

    print("Matrix size: " + str(matrice_size) + "x" + str(matrice_size))

    n_runs = 10
    # Benchmark numpy timing
    start_time = time.time_ns()
    for run in range(n_runs):
        X_2 = np.linalg.inv(A_2_dense)
    end_time = time.time_ns()
    mean_time = (end_time - start_time) / n_runs
    
    print("Np inversion mean time: " + str(mean_time/10e6) + " [ms]")
    
    
    
    # Benchmark GMRES with pre-conditionning
    start_time = time.time_ns()
    
    I = np.eye(matrice_size)
    for run in range(n_runs):
        X_2, info = sparse.linalg.gmres(A_2, 
                                        I[:,0], 
                                        M = A_1_inv,
                                        tol=1e-12, 
                                        restart=1000, 
                                        maxiter=matrice_size, 
                                        callback=print_maxiter)
            
    end_time = time.time_ns()
    mean_time = (end_time - start_time) * matrice_size / n_runs
    
    total_iter_per_run = (max_iter / n_runs) * matrice_size 
    max_iter = 0
    
    print("GMRES with preconditioning inversion mean time: " + str(mean_time/10e6) + " [ms], total_iter_per_run = " + str(total_iter_per_run))
    
    
    
    # Benchmark GMRES without pre-conditionning
    start_time = time.time_ns()
    
    I = np.eye(matrice_size)
    for run in range(n_runs):
        X_2, info = sparse.linalg.gmres(A_2, 
                                        I[:,0], 
                                        tol=1e-12, 
                                        restart=1000, 
                                        maxiter=matrice_size, 
                                        callback=print_maxiter)
            
        
    end_time = time.time_ns()
    mean_time = (end_time - start_time) * matrice_size / n_runs
    
    total_iter_per_run = (max_iter / n_runs) * matrice_size 
    max_iter = 0
    
    print("GMRES without preconditioning inversion mean time: " + str(mean_time/10e6) + " [ms], total_iter_per_run = " + str(total_iter_per_run))
    
    
    

    
    