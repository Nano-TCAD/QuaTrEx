"""
Example a sc-GW iteration with MPI+CUDA.
With transposition through network.
Applied to a (8-0)-CNT and 7 AGNR 
See the different GW step folders for more explanations.
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
mpi4py.rc.finalize = False    # do not finalize MPI automatically
from mpi4py import MPI

main_path = os.path.abspath(os.path.dirname(__file__))
parent_path = os.path.abspath(os.path.join(main_path, ".."))
sys.path.append(parent_path)

from bandstructure.calc_band_edge import get_band_edge_mpi
from GW.polarization.kernel import g2p_cpu
from GW.selfenergy.kernel import gw2s_cpu
from GW.gold_solution import read_solution
from GW.screenedinteraction.kernel import p2w_cpu
from GW.coulomb_matrix.read_coulomb_matrix import load_V
from GreensFunction import calc_GF_pool
from OMEN_structure_matrices import OMENHamClass
from utils import change_format
from utils import utils_gpu

if utils_gpu.gpu_avail():
    from GW.polarization.kernel import g2p_gpu
    from GW.selfenergy.kernel import gw2s_gpu

if __name__ == "__main__":
    MPI.Init_thread(required=MPI.THREAD_FUNNELED)
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    name = MPI.Get_processor_name()

    # assume every rank has enough memory to read the initial data
    # path to solution
    scratch_path = "/usr/scratch/mont-fort17/dleonard/IEDM/"
    solution_path = os.path.join(scratch_path, "CNT_unbiased")
    solution_path_gw = os.path.join(solution_path, "data_GPWS_06.mat")
    solution_path_vh = os.path.join(solution_path, "data_Vh_6.mat")
    hamiltonian_path = solution_path
    parser = argparse.ArgumentParser(
        description="Example of the first GW iteration with MPI+CUDA"
    )
    parser.add_argument("-fvh", "--file_vh", default=solution_path_vh, required=False)
    parser.add_argument("-fpw", "--file_gw", default=solution_path_gw, required=False)
    parser.add_argument("-fhm", "--file_hm", default=hamiltonian_path, required=False)
    # change manually the used implementation inside the code
    parser.add_argument("-t", "--type", default="cpu",
                    choices=["cpu", "gpu"], required=False)
    parser.add_argument("-nt", "--net_transpose", default=False,
                    type=bool, required=False)
    parser.add_argument("-p", "--pool", default=True,
                type=bool, required=False)
    args = parser.parse_args()
    # check if gpu is available
    if args.type in ("gpu"):
        if not utils_gpu.gpu_avail():
            print("No gpu available")
            sys.exit(1)
    # print chosen implementation
    print(f"Using {args.type} implementation")

    # create hamiltonian object
    # one orbital on C atoms, two same types
    no_orb = np.array([1, 1])
    energy = np.linspace(-17.5, 7.5, 251, endpoint = True, dtype = float) # Energy Vector
    hamiltonian_obj = OMENHamClass.Hamiltonian(args.file_hm, no_orb, rank)
    serial_ham = pickle.dumps(hamiltonian_obj)
    broadcasted_ham = comm.bcast(serial_ham, root=0)
    hamiltonian_obj = pickle.loads(broadcasted_ham)
    # Extract neighbor indices
    rows = hamiltonian_obj.rows
    columns = hamiltonian_obj.columns

    # hamiltonian object has 1-based indexing
    bmax = hamiltonian_obj.Bmax - 1
    bmin = hamiltonian_obj.Bmin - 1


    # Read Coulomb matrix and extract the nnz data of the coulomb matrix
    V_sparse = load_V(args.file_hm, rows, columns)
    V_data = V_sparse.data

    vh = V_sparse