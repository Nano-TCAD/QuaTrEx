import cupy as cp
import cupyx as cpx
import numpy as np
import numpy.typing as npt
import scipy.sparse as sp
import time

from mpi4py import MPI
from quatrex.OMEN_structure_matrices import OMENHamClass
from quatrex.GreensFunction.calc_GF_pool_GPU_memopt_2 import calc_GF_pool_mpi_split_memopt
from quatrex.block_tri_solvers.rgf_GF_GPU_combo import map_to_mapping, csr_to_block_tridiagonal_csr
from quatrex.utils.matrix_creation import get_number_connected_blocks
from quatrex.GW.polarization.kernel import g2p_gpu


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
    size = 2048 * 8
    energy = np.linspace(-5, 1, num_energies * size, endpoint=True, dtype=float)  # Energy Vector
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
    ne = num_energies * size
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

    denergy:    npt.NDArray[np.double]  = energy[1] - energy[0]
    pre_factor: np.complex128           = -1.0j * denergy / (np.pi)

     # calculation of data distribution per rank---------------------------------

    # split nnz/energy per rank
    data_shape = np.array([nnz, ne], dtype=np.int32)
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

    sr_dev = cp.asarray(random_complex((count[1, rank], nnz), rng))
    sl_dev = cp.asarray(random_complex((count[1, rank], nnz), rng))
    sg_dev = cp.asarray(random_complex((count[1, rank], nnz), rng))
    sr_phn_dev = cp.asarray(random_complex((count[1, rank], matrix_size), rng))
    sl_phn_dev = cp.asarray(random_complex((count[1, rank], matrix_size), rng))
    sg_phn_dev = cp.asarray(random_complex((count[1, rank], matrix_size), rng))

    no = nnz
    gg_h2g = cp.empty((count[1, rank], no), dtype=np.complex128)
    gl_h2g = cp.empty((count[1, rank], no), dtype=np.complex128)
    gr_h2g = cp.empty((count[1, rank], no), dtype=np.complex128)
    gl_transposed_h2g = cp.empty((count[1, rank], no), dtype=np.complex128)

    gg_g2p = cp.empty((count[0, rank], data_shape[1]), dtype=np.complex128, order="C")
    gl_g2p = cp.empty((count[0, rank], data_shape[1]), dtype=np.complex128, order="C")
    gr_g2p = cp.empty((count[0, rank], data_shape[1]), dtype=np.complex128, order="C")
    gl_transposed_g2p = cp.empty((count[0, rank], data_shape[1]), dtype=np.complex128, order="C")
    pg_p2w = cp.empty((count[1, rank], data_shape[0]), dtype=np.complex128, order="C")
    pl_p2w = cp.empty((count[1, rank], data_shape[0]), dtype=np.complex128, order="C")
    pr_p2w = cp.empty((count[1, rank], data_shape[0]), dtype=np.complex128, order="C")
    pg_g2p = cp.empty((count[0, rank], ne), dtype=np.complex128)
    pl_g2p = cp.empty((count[0, rank], ne), dtype=np.complex128)

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

    # arrays filled with complex doubles
    BASE_TYPE = MPI.Datatype(MPI.DOUBLE_COMPLEX)
    base_size = np.dtype(np.complex128).itemsize

    # column type of orginal matrix
    COLUMN = BASE_TYPE.Create_vector(data_shape[0], 1, data_shape[1])
    COLUMN_RIZ = COLUMN.Create_resized(0, base_size)
    MPI.Datatype.Commit(COLUMN_RIZ)
    MPI.Datatype.Commit(COLUMN)

    # row type of original transposed matrix
    ROW = BASE_TYPE.Create_vector(data_shape[1], 1, data_shape[0])
    ROW_RIZ = ROW.Create_resized(0, base_size)
    MPI.Datatype.Commit(ROW_RIZ)
    MPI.Datatype.Commit(ROW)

    # send type g2p
    # column type of split up in nnz
    G2P_S = BASE_TYPE.Create_vector(count[1, rank], 1, data_shape[0])
    G2P_S_RIZ = G2P_S.Create_resized(0, base_size)
    MPI.Datatype.Commit(G2P_S)
    MPI.Datatype.Commit(G2P_S_RIZ)

    # receive types g2p
    # vector of size of #ranks
    # multi column data type for every rank size #energy not divisible
    G2P_R = np.array([BASE_TYPE.Create_vector(count[0, rank], count[1, i], data_shape[1]) for i in range(size)])
    G2P_R_RIZ = np.empty_like(G2P_R)
    for i in range(size):
        G2P_R_RIZ[i] = G2P_R[i].Create_resized(0, base_size)
        MPI.Datatype.Commit(G2P_R[i])
        MPI.Datatype.Commit(G2P_R_RIZ[i])

    # send type p2g
    # column type of split up in energy
    P2G_S = BASE_TYPE.Create_vector(count[0, rank], 1, data_shape[1])
    P2G_S_RIZ = P2G_S.Create_resized(0, base_size)
    MPI.Datatype.Commit(P2G_S)
    MPI.Datatype.Commit(P2G_S_RIZ)

    # receive types p2g
    # vector of size of #ranks
    # multi column data type for every rank size #nnz not divisible
    P2G_R = np.array([BASE_TYPE.Create_vector(count[1, rank], count[0, i], data_shape[0]) for i in range(size)])
    P2G_R_RIZ = np.empty_like(P2G_R)
    for i in range(size):
        P2G_R_RIZ[i] = P2G_R[i].Create_resized(0, base_size)
        MPI.Datatype.Commit(P2G_R[i])
        MPI.Datatype.Commit(P2G_R_RIZ[i])

    g2p_send_count = count[0, :] * count[1, rank]
    g2p_send_displ = disp[0, :] * count[1, rank] * base_size
    g2p_send_types = np.repeat(BASE_TYPE, size)
    g2p_recv_count = count[1, :] * count[0, rank]
    g2p_recv_displ = disp[1, :] * count[0, rank] * base_size
    g2p_recv_types = np.repeat(BASE_TYPE, size)

    def alltoall_g2p(sendbuf: npt.NDArray[np.complex128], recvbuf: npt.NDArray[np.complex128],
                     transpose_net: bool = False, gpu_aware: bool = False):
        if transpose_net:
            if gpu_aware:
                cp.cuda.get_current_stream().synchronize()
            comm.Alltoallw([sendbuf, count[0, :], disp[0, :] * base_size, np.repeat(G2P_S_RIZ, size)],
                           [recvbuf, np.repeat([1], size), disp[1, :] * base_size, G2P_R_RIZ])
        else:
            if gpu_aware:
                sendbuf = cp.copy(sendbuf.T, order="C")
                cp.cuda.get_current_stream().synchronize()
            else:
                sendbuf = np.copy(sendbuf.T, order="C") 
            tmp0 = recvbuf.reshape(size, count[0, rank], count[1, rank])
            tmp1 = recvbuf.reshape(count[0, rank], size, count[1, rank])
            comm.Alltoallw([sendbuf, g2p_send_count, g2p_send_displ, g2p_send_types],
                           [tmp0, g2p_recv_count, g2p_recv_displ, g2p_recv_types])
            tmp1[:] = tmp0.transpose(1, 0, 2)
            cp.cuda.get_current_stream().synchronize()


    def alltoall_p2g(sendbuf: npt.NDArray[np.complex128], recvbuf: npt.NDArray[np.complex128],
                     transpose_net: bool = False, gpu_aware: bool = False):
        if transpose_net:
            if gpu_aware:
                cp.cuda.get_current_stream().synchronize()
            comm.Alltoallw([sendbuf, count[1, :], disp[1, :] * base_size, np.repeat(P2G_S_RIZ, size)],
                           [recvbuf, np.repeat([1], size), disp[0, :] * base_size, P2G_R_RIZ])
        else:
            tmp1 = sendbuf.reshape(count[0, rank], size, count[1, rank])
            tmp0 = cp.copy(tmp1.transpose(1, 0, 2), order="C")
            tmp2 = recvbuf.reshape(data_shape[0], count[1, rank])
            cp.cuda.get_current_stream().synchronize()
            comm.Alltoallw([tmp0, g2p_recv_count, g2p_recv_displ, g2p_recv_types],
                           [tmp2, g2p_send_count, g2p_send_displ, g2p_send_types])
            recvbuf[:] = tmp2.transpose()
            cp.cuda.get_current_stream().synchronize()
    
    is_mpi_gpu_aware = True

    class dummy:
        def __init__(self):
            self.net_transpose = False

    args = dummy()

    for it in range(10):

        # # cp.cuda.get_current_stream().synchronize()
        # comm.Barrier()
        # start_g2p_comm = time.perf_counter()
        
        # # calculate the transposed
        # if is_mpi_gpu_aware:
        #     # gl_transposed_h2g = cp.copy(gl_h2g[:, ij2ji_dev], order="C")
        #     gl_transposed_h2g[:] = gl_h2g[:, ij2ji_dev]
        # else:
        #     gl_transposed_h2g = np.copy(gl_h2g[:, ij2ji], order="C")

        # # use of all to all w since not divisible
        # # cp.cuda.get_current_stream().synchronize()
        # # comm.Barrier()
        # start = time.perf_counter()
        # alltoall_g2p(gg_h2g, gg_g2p, transpose_net=args.net_transpose, gpu_aware=is_mpi_gpu_aware)
        # # comm.Barrier()
        # # finish = time.perf_counter()
        # # if rank == 0:
        # #     print(f"G2P communication time (1): {finish - start}", flush = True)
        # # start = time.perf_counter()
        # alltoall_g2p(gl_h2g, gl_g2p, transpose_net=args.net_transpose, gpu_aware=is_mpi_gpu_aware)
        # # comm.Barrier()
        # # finish = time.perf_counter()
        # # if rank == 0:
        # #     print(f"G2P communication time (2): {finish - start}", flush = True)
        # # # start = time.perf_counter()
        # # alltoall_g2p(gr_h2g, gr_g2p, transpose_net=args.net_transpose, gpu_aware=is_mpi_gpu_aware)
        # # # comm.Barrier()
        # # # finish = time.perf_counter()
        # # # if rank == 0:
        # # #     print(f"G2P communication time (3): {finish - start}", flush = True)
        # # start = time.perf_counter()
        # alltoall_g2p(gl_transposed_h2g, gl_transposed_g2p, transpose_net=args.net_transpose, gpu_aware=is_mpi_gpu_aware)
        # # comm.Barrier()
        # # finish = time.perf_counter()
        # # if rank == 0:
        # #     print(f"G2P communication time (4): {finish - start}", flush = True)

        # comm.Barrier()
        # finish_g2p_comm = time.perf_counter()
        # if rank == 0:
        #     print(f"G2P communication time: {finish_g2p_comm - start_g2p_comm}", flush = True)
        #     print("Green's function calculated", flush = True)

        print(f" FFT shape: {gg_g2p.shape}, {gl_g2p.shape}, {gl_transposed_g2p.shape}", flush = True)
        
        start_p_computation = time.perf_counter()

        # calculate the polarization at every rank----------------------------------
        # pg_g2p, pl_g2p = g2p_gpu.g2p_fft_mpi_gpu_batched_nopr(
        #                                     pre_factor,
        #                                     gg_g2p,
        #                                     gl_g2p,
        #                                     gl_transposed_g2p)
        
        g2p_gpu.g2p_fft_mpi_gpu_batched_nopr(pg_g2p, pl_g2p,
                                            pre_factor,
                                            gg_g2p,
                                            gl_g2p,
                                            gl_transposed_g2p)
        
        # comm.Barrier()
        finish_p_computation = time.perf_counter()
        if rank == 0:
            print(f"Polarization computation time: {finish_p_computation - start_p_computation}", flush = True)
        
        # start_p2w_comm = time.perf_counter()

        # # distribute polarization function according to p2w step--------------------

        # # use of all to all w since not divisible
        # alltoall_p2g(pg_g2p, pg_p2w, transpose_net=args.net_transpose, gpu_aware=is_mpi_gpu_aware)
        # alltoall_p2g(pl_g2p, pl_p2w, transpose_net=args.net_transpose, gpu_aware=is_mpi_gpu_aware)
        # # alltoall_p2g(pl_g2p, pr_p2w, transpose_net=args.net_transpose, gpu_aware=is_mpi_gpu_aware)

        # comm.Barrier()
        # finish_p2w_comm = time.perf_counter()
        # if rank == 0:
        #     print(f"P2W communication time: {finish_p2w_comm - start_p2w_comm}", flush = True)
        #     print("Polarization calculated", flush = True)
