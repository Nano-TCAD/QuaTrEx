import cupy as cp
import cupyx as cpx
import numpy as np
import scipy.sparse as sp
import time

from mpi4py import MPI
from quatrex.OMEN_structure_matrices import OMENHamClass
from quatrex.GreensFunction.calc_GF_pool_GPU_memopt_2 import calc_GF_pool_mpi_split_memopt
from quatrex.block_tri_solvers.rgf_GF_GPU_combo import map_to_mapping, csr_to_block_tridiagonal_csr
from quatrex.utils.matrix_creation import get_number_connected_blocks


def random_complex(shape, rng: np.random.Generator):
    result = cpx.empty_pinned(shape, dtype=np.complex128)
    result[:] = rng.random(shape) + 1j * rng.random(shape)
    return result


if __name__ == "__main__":

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    name = MPI.Get_processor_name()

    small_folder = '/users/ziogasal/QuaTrEx/test_data/small'
    large_folder = '/users/ziogasal/QuaTrEx/test_data/large'

    simulations = {
        "small": {
            "folder": small_folder,
            "hamiltonian": "/scratch/project_465000929/Si_Nanowire/",
            "num_energies": 64,
            "num_blocks": 13,
            "block_size": 416,
            "nnz": 491040
        },
        "large": {
            "folder": large_folder,
            "hamiltonian": "/scratch/project_465000929/Si_Nanowire_18/",
            "num_energies": 32,
            "num_blocks": 18,
            "block_size": 416,
            "nnz": 3149584
        },
    }

    simulation = simulations["large"]

    map_diag = np.load(f"{simulation['folder']}/map_diag.npy")
    map_upper = np.load(f"{simulation['folder']}/map_upper.npy")
    map_lower = np.load(f"{simulation['folder']}/map_lower.npy")
    rows = np.load(f"{simulation['folder']}/rows.npy")
    columns = np.load(f"{simulation['folder']}/columns.npy")
    ij2ji = np.load(f"{simulation['folder']}/ij2ji.npy")

    num_energies = simulation["num_energies"]
    num_blocks = simulation["num_blocks"]
    block_size = simulation["block_size"]
    nnz = simulation["nnz"]
    matrix_size = num_blocks * block_size
    assert simulation["nnz"] == len(rows)

    # create hamiltonian object
    # one orbital on C atoms, two same types
    no_orb = np.array([1,4])
    # Factor to extract smaller matrix blocks (factor * unit cell size < current block size based on Smin_dat)
    NCpSC = 4
    Vappl = 0.6
    energy = np.linspace(-5, 1, num_energies, endpoint=True, dtype=float)  # Energy Vector
    Idx_e = np.arange(energy.shape[0])  # Energy Index Vector
    EPHN = np.array([0.0])  # Phonon energy
    DPHN = np.array([2.5e-3])  # Electron-phonon coupling
    hamiltonian_obj = OMENHamClass.Hamiltonian(simulation["hamiltonian"], no_orb, Vappl = Vappl,  potential_type = 'atomic', bias_point = 13, rank = rank, layer_matrix = '/Layer_Matrix.dat')

    mapping_diag = map_to_mapping(map_diag, num_blocks)
    mapping_upper = map_to_mapping(map_upper, num_blocks - 1)
    mapping_lower = map_to_mapping(map_lower, num_blocks - 1)

    mapping_diag_dev = cp.asarray(mapping_diag)
    mapping_upper_dev = cp.asarray(mapping_upper)
    mapping_lower_dev = cp.asarray(mapping_lower)
    rows_dev = cp.asarray(rows)
    columns_dev = cp.asarray(columns)
    ij2ji_dev = cp.asarray(ij2ji)

    DH = hamiltonian_obj
    bmin = DH.Bmin.copy()
    bmax = DH.Bmax.copy()
    nbc = get_number_connected_blocks(hamiltonian_obj.NH, bmin, bmax, rows, columns)
    hamiltonian_diag, hamiltonian_upper, hamiltonian_lower = csr_to_block_tridiagonal_csr(DH.Hamiltonian['H_4'], bmin - 1, bmax)
    overlap_diag, overlap_upper, overlap_lower = csr_to_block_tridiagonal_csr(DH.Overlap['H_4'], bmin - 1, bmax)

    # computation parameters----------------------------------------------------
    # set number of threads for the p2w step
    w_mkl_threads = 1
    w_worker_threads = 6
    # set number of threads for the h2g step
    gf_mkl_threads = 1
    gf_mkl_threads_gpu = 1
    gf_worker_threads = 6

    # physical parameter -----------

    # Fermi Level of Left Contact
    energy_fl = -2.0362
    # Fermi Level of Right Contact
    energy_fr = energy_fl - Vappl
    # Temperature in Kelvin
    temp = 300
    # relative permittivity
    epsR = 2.0
    # DFT Conduction Band Minimum
    ECmin = -2.0662

    # Phyiscal Constants -----------

    e   = 1.6022e-19
    eps0 = 8.854e-12
    hbar = 1.0546e-34

    # Fermi Level to Band Edge Difference
    dEfL_EC = energy_fl - ECmin
    dEfR_EC = energy_fr - ECmin

    # create the corresponding factor to mask 
    # number of points to smooth the edges of the Green's Function
    dnp = 50
    ne = num_energies
    factor_w = np.ones(ne)
    #factor_w[ne-dnp-1:ne] = (np.cos(np.pi*np.linspace(0, 1, dnp+1)) + 1)/2
    #factor_w[np.where(np.invert(w_mask))[0]] = 0.0

    # create factor for the Green's Function
    factor_g = np.ones(ne)
    #factor_g[ne-dnp-1:ne] = (np.cos(np.pi*np.linspace(0, 1, dnp+1)) + 1)/2
    #factor_g[0:dnp+1] = (np.cos(np.pi*np.linspace(1, 0, dnp+1)) + 1)/2

    # vh = construct_coulomb_matrix(hamiltonian_obj, epsR, eps0, e, diag = False, orb_uniform = True)
    # #vh = load_V_mpi(solution_path_vh, rows, columns, comm, rank)/epsR
    # vh1d = cp.asarray(np.squeeze(np.asarray(vh[np.copy(rows), np.copy(columns)].reshape(-1))))

     # calculation of data distribution per rank---------------------------------

    # split nnz/energy per rank
    data_shape = np.array([nnz, num_energies], dtype=np.int32)
    data_per_rank = data_shape // size
    remainders = data_shape % size

    # create array with energy size distribution
    count = np.repeat(data_per_rank.reshape(-1, 1), size, axis=1)
    count[0, :remainders[0]] += 1
    count[1, :remainders[1]] += 1
    # count[:, size-1] += data_shape % size

    # displacements in nnz/energy
    disp = data_per_rank.reshape(-1, 1) * np.arange(size)

    # slice energy vector
    energy_loc = energy[disp[1, rank]:disp[1, rank] + count[1, rank]]
    Idx_e_loc = Idx_e[disp[1, rank]:disp[1, rank] + count[1, rank]]

    # split up the factor between the ranks
    factor_w_loc = factor_w[disp[1, rank]:disp[1, rank] + count[1, rank]]
    factor_g_loc = factor_g[disp[1, rank]:disp[1, rank] + count[1, rank]]

    rng = np.random.default_rng(0)

    sr_dev = cp.asarray(random_complex((num_energies, nnz), rng))
    sl_dev = cp.asarray(random_complex((num_energies, nnz), rng))
    sg_dev = cp.asarray(random_complex((num_energies, nnz), rng))
    sr_phn_dev = cp.asarray(random_complex((num_energies, matrix_size), rng))
    sl_phn_dev = cp.asarray(random_complex((num_energies, matrix_size), rng))
    sg_phn_dev = cp.asarray(random_complex((num_energies, matrix_size), rng))

    no = nnz
    gg_h2g = cp.empty((count[1, rank], no), dtype=np.complex128)
    gl_h2g = cp.empty((count[1, rank], no), dtype=np.complex128)
    gr_h2g = cp.empty((count[1, rank], no), dtype=np.complex128)

    # initialize observables----------------------------------------------------
    # density of states
    nb = num_blocks
    dos = cpx.empty_pinned(shape=(ne, nb), dtype=np.complex128)
    dosw = cpx.empty_pinned(shape=(ne, nb // nbc), dtype=np.complex128)

    # occupied states/unoccupied states
    nE = cpx.empty_pinned(shape=(ne, nb), dtype=np.complex128)
    nP = cpx.empty_pinned(shape=(ne, nb), dtype=np.complex128)

    # occupied screening/unoccupied screening
    nEw = cpx.empty_pinned(shape=(ne, nb // nbc), dtype=np.complex128)
    nPw = cpx.empty_pinned(shape=(ne, nb // nbc), dtype=np.complex128)

    # current per energy
    ide = cpx.empty_pinned(shape=(ne, nb), dtype=np.complex128)

    calc_GF_pool_mpi_split_memopt(
            hamiltonian_obj,
            hamiltonian_diag, hamiltonian_upper, hamiltonian_lower,
            overlap_diag, overlap_upper, overlap_lower,
            energy_loc,
            sr_dev,
            sl_dev,
            sg_dev,
            gr_h2g,
            gl_h2g,
            gg_h2g,
            mapping_diag_dev,
            mapping_upper_dev,
            mapping_lower_dev,
            energy_fl,
            energy_fr,
            temp,
            dos[disp[1, rank]:disp[1, rank] + count[1, rank]],
            nE[disp[1, rank]:disp[1, rank] + count[1, rank]],
            nP[disp[1, rank]:disp[1, rank] + count[1, rank]],
            ide[disp[1, rank]:disp[1, rank] + count[1, rank]],
            factor_g_loc,
            comm,
            rank,
            size,
            homogenize=False,
            NCpSC=NCpSC,
            mkl_threads=gf_mkl_threads,
            worker_num=gf_worker_threads)
