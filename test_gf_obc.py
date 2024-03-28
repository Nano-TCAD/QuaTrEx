import cupy as cp
import cupyx as cpx
import numpy as np
import scipy.sparse as sp
import time

from mpi4py import MPI
from quatrex.OMEN_structure_matrices import OMENHamClass
# from quatrex.GreensFunction.calc_GF_pool_GPU_memopt_2 import calc_GF_pool_mpi_split_memopt
from quatrex.block_tri_solvers import rgf_GF_GPU_combo
from quatrex.block_tri_solvers.rgf_GF_GPU_combo import map_to_mapping, csr_to_block_tridiagonal_csr
from quatrex.utils.matrix_creation import get_number_connected_blocks
from quatrex.OBC.beyn_batched import beyn_batched_gpu_3 as beyn_gpu


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

    ###########################

    comm.Barrier()
    if rank == 0:
        time_pre_OBC = -time.perf_counter()
    
    kB = 1.38e-23
    q = 1.6022e-19

    Temp = temp
    UT = kB * Temp / q

    def fermi_function(E, Ef, UT):
        return 1 / (1 + np.exp((E - Ef) / UT))
    
    vfermi = np.vectorize(fermi_function)
    Efl = energy_fl
    Efr = energy_fr
    fL = cp.asarray(vfermi(energy, Efl, UT)).reshape(len(energy), 1, 1)
    fR = cp.asarray(vfermi(energy, Efr, UT)).reshape(len(energy), 1, 1)

    # initialize the Green's function in block format with zero
    # number of energy points
    ne = energy.shape[0]
    # number of blocks
    nb = DH.Bmin.shape[0]
    # length of the largest block
    lb = np.max(DH.Bmax - DH.Bmin + 1)
    # init

    # index_e = np.arange(ne)
    bmin = DH.Bmin.copy()
    bmax = DH.Bmax.copy()

    LBsize = bmax[0] - bmin[0] + 1
    RBsize = bmax[nb - 1] - bmin[nb - 1] + 1

    SigRBL = cpx.zeros_pinned((ne, LBsize, LBsize), dtype = np.complex128)
    SigRBR = cpx.zeros_pinned((ne, RBsize, RBsize), dtype = np.complex128)
    SigLBL = cpx.zeros_pinned((ne, LBsize, LBsize), dtype = np.complex128)
    SigLBR = cpx.zeros_pinned((ne, RBsize, RBsize), dtype = np.complex128)
    SigGBL = cpx.zeros_pinned((ne, LBsize, LBsize), dtype = np.complex128)
    SigGBR = cpx.zeros_pinned((ne, RBsize, RBsize), dtype = np.complex128)

    bmin = DH.Bmin.copy()
    bmax = DH.Bmax.copy()

    input_stream = cp.cuda.stream.Stream(non_blocking=True)

    M00_left = cp.empty((ne, LBsize, LBsize), dtype = np.complex128)
    M01_left = cp.empty((ne, LBsize, RBsize), dtype = np.complex128)
    M10_left = cp.empty((ne, RBsize, LBsize), dtype = np.complex128)
    M00_right = cp.empty((ne, RBsize, RBsize), dtype = np.complex128)
    M01_right = cp.empty((ne, RBsize, LBsize), dtype = np.complex128)
    M10_right = cp.empty((ne, LBsize, RBsize), dtype = np.complex128)
    rgf_GF_GPU_combo._get_dense_block_batch(sr_dev, mapping_diag_dev, 0, M00_left)
    rgf_GF_GPU_combo._get_dense_block_batch(sr_dev, mapping_upper_dev, 0, M01_left)
    rgf_GF_GPU_combo._get_dense_block_batch(sr_dev, mapping_lower_dev, 0, M10_left)
    rgf_GF_GPU_combo._get_dense_block_batch(sr_dev, mapping_diag_dev, nb-1, M00_right)
    rgf_GF_GPU_combo._get_dense_block_batch(sr_dev, mapping_upper_dev, nb-2, M01_right)
    rgf_GF_GPU_combo._get_dense_block_batch(sr_dev, mapping_lower_dev, nb-2, M10_right)

    csr_matrix = rgf_GF_GPU_combo.csr_matrix
    hdtype = np.complex128
    block_size = max(LBsize, RBsize)

    H_diag_buffer = [csr_matrix(cp.empty(block_size * block_size, hdtype),
                                cp.empty(block_size * block_size, cp.int32),
                                cp.empty(block_size + 1, cp.int32),) for _ in range(2)]
    H_upper_buffer = [csr_matrix(cp.empty(block_size * block_size, hdtype),
                                 cp.empty(block_size * block_size, cp.int32),
                                 cp.empty(block_size + 1, cp.int32),) for _ in range(2)]
    H_lower_buffer = [csr_matrix(cp.empty(block_size * block_size, hdtype),
                                 cp.empty(block_size * block_size, cp.int32),
                                 cp.empty(block_size + 1, cp.int32),) for _ in range(2)]
    S_diag_buffer = [csr_matrix(cp.empty(block_size * block_size, hdtype),
                                cp.empty(block_size * block_size, cp.int32),
                                cp.empty(block_size + 1, cp.int32),) for _ in range(2)]
    S_upper_buffer = [csr_matrix(cp.empty(block_size * block_size, hdtype),
                                 cp.empty(block_size * block_size, cp.int32),
                                 cp.empty(block_size + 1, cp.int32),) for _ in range(2)]
    S_lower_buffer = [csr_matrix(cp.empty(block_size * block_size, hdtype),
                                 cp.empty(block_size * block_size, cp.int32),
                                 cp.empty(block_size + 1, cp.int32),) for _ in range(2)]

    batch_size = len(energy)
    block_size = LBsize
    num_threads = min(1024, block_size)
    num_thread_blocks = batch_size * block_size
    energy_dev = cp.asarray(energy)
    hd, sd = H_diag_buffer[0], S_diag_buffer[0]
    hu, su = H_upper_buffer[0], S_upper_buffer[0]
    hl, sl = H_lower_buffer[0], S_lower_buffer[0]
    rgf_GF_GPU_combo._copy_csr_to_gpu(hamiltonian_diag[0], hd)
    rgf_GF_GPU_combo._copy_csr_to_gpu(overlap_diag[0], sd)
    rgf_GF_GPU_combo._copy_csr_to_gpu(hamiltonian_upper[0], hu)
    rgf_GF_GPU_combo._copy_csr_to_gpu(overlap_upper[0], su)
    rgf_GF_GPU_combo._copy_csr_to_gpu(hamiltonian_lower[0], hl)
    rgf_GF_GPU_combo._copy_csr_to_gpu(overlap_lower[0], sl)
    block_size = LBsize
    num_threads = min(1024, block_size)
    num_thread_blocks = batch_size * block_size
    rgf_GF_GPU_combo._get_system_matrix[num_thread_blocks, num_threads](
        energy_dev, hd.data, hd.indices, hd.indptr, sd.data, sd.indices, sd.indptr, M00_left, M00_left, batch_size, block_size)
    rgf_GF_GPU_combo._get_system_matrix[num_thread_blocks, num_threads](
        energy_dev, hu.data, hu.indices, hu.indptr, su.data, su.indices, su.indptr, M01_left, M01_left, batch_size, block_size)
    rgf_GF_GPU_combo._get_system_matrix[num_thread_blocks, num_threads](
        energy_dev, hl.data, hl.indices, hl.indptr, sl.data, sl.indices, sl.indptr, M10_left, M10_left, batch_size, block_size)
    hd, sd = H_diag_buffer[-1], S_diag_buffer[-1]
    hu, su = H_upper_buffer[-1], S_upper_buffer[-1]
    hl, sl = H_lower_buffer[-1], S_lower_buffer[-1]
    rgf_GF_GPU_combo._copy_csr_to_gpu(hamiltonian_diag[-1], hd)
    rgf_GF_GPU_combo._copy_csr_to_gpu(overlap_diag[-1], sd)
    rgf_GF_GPU_combo._copy_csr_to_gpu(hamiltonian_upper[-1], hu)
    rgf_GF_GPU_combo._copy_csr_to_gpu(overlap_upper[-1], su)
    rgf_GF_GPU_combo._copy_csr_to_gpu(hamiltonian_lower[-1], hl)
    rgf_GF_GPU_combo._copy_csr_to_gpu(overlap_lower[-1], sl)
    block_size = RBsize
    num_threads = min(1024, block_size)
    num_thread_blocks = batch_size * block_size
    rgf_GF_GPU_combo._get_system_matrix[num_thread_blocks, num_threads](
        energy_dev, hd.data, hd.indices, hd.indptr, sd.data, sd.indices, sd.indptr, M00_right, M00_right, batch_size, block_size)
    rgf_GF_GPU_combo._get_system_matrix[num_thread_blocks, num_threads](
        energy_dev, hu.data, hu.indices, hu.indptr, su.data, su.indices, su.indptr, M01_right, M01_right, batch_size, block_size)
    rgf_GF_GPU_combo._get_system_matrix[num_thread_blocks, num_threads](
        energy_dev, hl.data, hl.indices, hl.indptr, sl.data, sl.indices, sl.indptr, M10_right, M10_right, batch_size, block_size)
    cp.cuda.Stream.null.synchronize()

    comm.Barrier() 

    if rank == 0:
        time_pre_OBC += time.perf_counter()
        print("Time for pre-processing OBC: %.3f s" % time_pre_OBC, flush = True)
        time_OBC = -time.perf_counter()
    
    imag_lim = 5e-4
    R = 1000
    SigRBL_gpu, _, condL, _ = beyn_gpu(NCpSC, M00_left, M01_left, M10_left, imag_lim, R, 'L')
    # assert not any(np.isnan(cond) for cond in condL)
    GammaL = 1j * (SigRBL_gpu - SigRBL_gpu.transpose(0, 2, 1).conj())
    (1j * fL * GammaL).get(out=SigLBL)
    (1j * (fL - 1) * GammaL).get(out=SigGBL)
    SigRBL_gpu.get(out=SigRBL)
    SigRBR_gpu, _, condR, _ = beyn_gpu(NCpSC, M00_right, M01_right, M10_right, imag_lim, R, 'R')
    # assert not any(np.isnan(cond) for cond in condR)
    GammaR = 1j * (SigRBR_gpu - SigRBR_gpu.transpose(0, 2, 1).conj())
    (1j * fR * GammaR).get(out=SigLBR)
    (1j * (fR - 1) * GammaR).get(out=SigGBR)
    SigRBR_gpu.get(out=SigRBR)
