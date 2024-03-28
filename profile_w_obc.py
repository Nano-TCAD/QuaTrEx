import cupy as cp
import cupyx as cpx
import numpy as np
import scipy.sparse as sp
import time
import concurrent.futures

from mpi4py import MPI
from quatrex.OMEN_structure_matrices import OMENHamClass
from quatrex.OMEN_structure_matrices.construct_CM import construct_coulomb_matrix
# from quatrex.GreensFunction.calc_GF_pool_GPU_memopt_2 import calc_GF_pool_mpi_split_memopt
# from quatrex.GW.screenedinteraction.kernel import p2w_gpu_improved_2
from quatrex.GW.screenedinteraction.kernel.p2w_gpu_improved_2 import _toarray, spgemm, spgemm_direct
from quatrex.block_tri_solvers.rgf_GF_GPU_combo import map_to_mapping, csr_to_block_tridiagonal_csr
from quatrex.utils.matrix_creation import get_number_connected_blocks
from quatrex.GW.screenedinteraction.polarization_preprocess import polarization_preprocess_2d
from quatrex.OBC import obc_w_gpu
from quatrex.OBC.beyn_batched import beyn_new_batched_gpu_3 as beyn_gpu


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

    map_diag_mm = np.load(f"{simulation['folder']}/map_diag_mm.npy")
    map_upper_mm = np.load(f"{simulation['folder']}/map_upper_mm.npy")
    map_lower_mm = np.load(f"{simulation['folder']}/map_lower_mm.npy")
    map_diag_m = np.load(f"{simulation['folder']}/map_diag_m.npy")
    map_upper_m = np.load(f"{simulation['folder']}/map_upper_m.npy")
    map_lower_m = np.load(f"{simulation['folder']}/map_lower_m.npy")
    map_diag_l = np.load(f"{simulation['folder']}/map_diag_l.npy")
    map_upper_l = np.load(f"{simulation['folder']}/map_upper_l.npy")
    map_lower_l = np.load(f"{simulation['folder']}/map_lower_l.npy")
    rows_m = np.load(f"{simulation['folder']}/rows_m.npy")
    columns_m = np.load(f"{simulation['folder']}/columns_m.npy")
    rows_l = np.load(f"{simulation['folder']}/rows_l.npy")
    columns_l = np.load(f"{simulation['folder']}/columns_l.npy")
    ij2ji_m = np.load(f"{simulation['folder']}/ij2ji_m.npy")
    ij2ji_l = np.load(f"{simulation['folder']}/ij2ji_l.npy")

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

    vh = construct_coulomb_matrix(hamiltonian_obj, epsR, eps0, e, diag = False, orb_uniform = True)
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

    # sr_dev = cp.asarray(random_complex((num_energies, nnz), rng))
    # sl_dev = cp.asarray(random_complex((num_energies, nnz), rng))
    # sg_dev = cp.asarray(random_complex((num_energies, nnz), rng))
    # sr_phn_dev = cp.asarray(random_complex((num_energies, matrix_size), rng))
    # sl_phn_dev = cp.asarray(random_complex((num_energies, matrix_size), rng))
    # sg_phn_dev = cp.asarray(random_complex((num_energies, matrix_size), rng))

    pg_p2w = cp.asarray(random_complex((count[1, rank], data_shape[0]), rng))
    pl_p2w = cp.asarray(random_complex((count[1, rank], data_shape[0]), rng))
    pr_p2w = cp.asarray(random_complex((count[1, rank], data_shape[0]), rng))

    no = nnz
    wg_p2w = cp.empty((count[1, rank], no), dtype=np.complex128)
    wl_p2w = cp.empty((count[1, rank], no), dtype=np.complex128)
    wr_p2w = cp.empty((count[1, rank], no), dtype=np.complex128)

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

    pg_p2w_vec = None
    pl_p2w_vec = None
    pr_p2w_vec = None

    mempool = cp.get_default_memory_pool()

    comm.Barrier()
    if rank == 0:
        time_pre_OBC = -time.perf_counter()
        print(f"Used bytes: {mempool.used_bytes()}", flush=True)
        print(f"Total bytes: {mempool.total_bytes()}", flush=True)

    # --- Preprocessing ------------------------------------------------

    for it in range(10):

        print("Iteration: ", it, flush=True)

        start_time = time.perf_counter()
        mapping_time = -time.perf_counter()

        # Number of energy points.
        ne = energy.shape[0]

        # Number of blocks.
        nb = hamiltonian_obj.Bmin.shape[0]
        # Start and end indices of the blocks.
        bmax = hamiltonian_obj.Bmax - 1
        bmin = hamiltonian_obj.Bmin - 1

        # fix nbc to 2 for the given solution
        # todo calculate it (?)
        # nbc = 2
        lb_vec = bmax - bmin + 1
        lb_start = lb_vec[0]
        lb_end = lb_vec[nb - 1]

        # Block sizes after matrix multiplication.
        bmax_mm = bmax[nbc - 1 : nb : nbc]
        bmin_mm = bmin[0:nb:nbc]
        # Number of blocks after matrix multiplication.
        nb_mm = bmax_mm.size
        # larges block length after matrix multiplication
        lb_max_mm = np.max(bmax_mm - bmin_mm + 1)
        lb_vec_mm = bmax_mm - bmin_mm + 1
        lb_start_mm = lb_vec_mm[0]
        lb_end_mm = lb_vec_mm[nb_mm - 1]

        mapping_diag_l = map_to_mapping(map_diag_l, nb_mm)
        mapping_upper_l = map_to_mapping(map_upper_l, nb_mm - 1)
        mapping_lower_l = map_to_mapping(map_lower_l, nb_mm - 1)

        mapping_diag_m = map_to_mapping(map_diag_m, nb_mm)
        mapping_upper_m = map_to_mapping(map_upper_m, nb_mm - 1)
        mapping_lower_m = map_to_mapping(map_lower_m, nb_mm - 1)

        mapping_diag_mm = map_to_mapping(map_diag_mm, nb_mm)
        mapping_upper_mm = map_to_mapping(map_upper_mm, nb_mm - 1)
        mapping_lower_mm = map_to_mapping(map_lower_mm, nb_mm - 1)

        vh_diag, vh_upper, vh_lower = csr_to_block_tridiagonal_csr(
            vh, bmin_mm, bmax_mm + 1
        )

        mapping_time += time.perf_counter()
        print("    Mapping: %.3f s" % mapping_time, flush=True)

        obc_w_batchsize = 8

        # --- Boundary conditions ------------------------------------------

        preprocess_time = -time.perf_counter()

        homogenize = False
        pl_rgf, pg_rgf, pr_rgf = polarization_preprocess_2d(pl_p2w, pg_p2w, pr_p2w, rows_dev, columns_dev, ij2ji_dev, NCpSC, bmin, bmax, homogenize)

        preprocess_time += time.perf_counter()
        print("    Preprocessing: %.3f s" % preprocess_time, flush=True)

        sparse_init_time = -time.perf_counter()
        nao = vh.shape[0]
        vh_dev = cp.sparse.csr_matrix(vh)
        vh_ct_dev = vh_dev.T.conj(copy=False)
        identity = cp.sparse.identity(nao)
        sparse_init_time += time.perf_counter()
        print("    Sparse init: %.3f s" % sparse_init_time, flush=True)

        # slice block start diagonal and slice block start off diagonal
        slb_sd = slice(0, lb_start)
        slb_so = slice(lb_start, 2 * lb_start)
        # slice block end diagonal and slice block end off diagonal
        slb_ed = slice(nao - lb_end, nao)
        slb_eo = slice(nao - 2 * lb_end, nao - lb_end)

        allocation_time = -time.perf_counter()

        mr_dev = cp.empty((obc_w_batchsize, len(rows_m)), dtype=np.complex128)
        lg_dev = cp.empty((obc_w_batchsize, len(rows_l)), dtype=np.complex128)
        ll_dev = cp.empty((obc_w_batchsize, len(rows_l)), dtype=np.complex128)
        mr_host = mr_dev
        lg_host = lg_dev
        ll_host = ll_dev

        vh_s1 = cp.empty((obc_w_batchsize, lb_start, lb_start), dtype=np.complex128)
        vh_s2 = cp.empty((obc_w_batchsize, lb_start, lb_start), dtype=np.complex128)
        pg_s1 = cp.empty((obc_w_batchsize, lb_start, lb_start), dtype=np.complex128)
        pg_s2 = cp.empty((obc_w_batchsize, lb_start, lb_start), dtype=np.complex128)
        pl_s1 = cp.empty((obc_w_batchsize, lb_start, lb_start), dtype=np.complex128)
        pl_s2 = cp.empty((obc_w_batchsize, lb_start, lb_start), dtype=np.complex128)
        pr_s1 = cp.empty((obc_w_batchsize, lb_start, lb_start), dtype=np.complex128)
        pr_s2 = cp.empty((obc_w_batchsize, lb_start, lb_start), dtype=np.complex128)
        pr_s3 = cp.empty((obc_w_batchsize, lb_start, lb_start), dtype=np.complex128)

        vh_e1 = cp.empty((obc_w_batchsize, lb_end, lb_end), dtype=np.complex128)
        vh_e2 = cp.empty((obc_w_batchsize, lb_end, lb_end), dtype=np.complex128)
        pg_e1 = cp.empty((obc_w_batchsize, lb_end, lb_end), dtype=np.complex128)
        pg_e2 = cp.empty((obc_w_batchsize, lb_end, lb_end), dtype=np.complex128)
        pl_e1 = cp.empty((obc_w_batchsize, lb_end, lb_end), dtype=np.complex128)
        pl_e2 = cp.empty((obc_w_batchsize, lb_end, lb_end), dtype=np.complex128)
        pr_e1 = cp.empty((obc_w_batchsize, lb_end, lb_end), dtype=np.complex128)
        pr_e2 = cp.empty((obc_w_batchsize, lb_end, lb_end), dtype=np.complex128)
        pr_e3 = cp.empty((obc_w_batchsize, lb_end, lb_end), dtype=np.complex128)

        pr_dev = None
        pg_dev = None
        pl_dev = None

        condL, condR = [], []

        allocation_time += time.perf_counter()
        print("    Allocation: %.3f s" % allocation_time, flush=True)

        start_spgemm = time.perf_counter()
        time_copy_in = 0
        time_spgemm = 0
        time_copy_out = 0
        time_dense = 0
        time_obc_mm = 0
        time_beyn_w = 0
        time_obc_l = 0
        time_w = 0

        for i in range(0, ne, obc_w_batchsize):
            j = min(i + obc_w_batchsize, ne)

            # for ie in range(ne):
            for ie in range(i, j):

                time_copy_in -= time.perf_counter()

                if pr_dev is None:
                    pr_dev = cp.sparse.csr_matrix((cp.asarray(pr_rgf[ie]), (rows_dev, columns_dev)), shape = (nao, nao))
                    pg_dev = cp.sparse.csr_matrix((cp.asarray(pg_rgf[ie]), (rows_dev, columns_dev)), shape = (nao, nao))
                    pl_dev = cp.sparse.csr_matrix((cp.asarray(pl_rgf[ie]), (rows_dev, columns_dev)), shape = (nao, nao))
                else:
                    pr_tmp = cp.asarray(pr_rgf[ie])
                    pr_dev.data[:] = pr_tmp[ij2ji_dev]
                    pg_tmp = cp.asarray(pg_rgf[ie])
                    pg_dev.data[:] = pg_tmp[ij2ji_dev]
                    pl_tmp = cp.asarray(pl_rgf[ie])
                    pl_dev.data[:] = pl_tmp[ij2ji_dev]
                
                time_copy_in += time.perf_counter()

                time_spgemm -= time.perf_counter()

                cp.cuda.Stream.null.synchronize()

                mr_dev[ie-i] = (identity - spgemm(vh_dev, pr_dev)).data
                spgemm_direct(spgemm(vh_dev, pg_dev), vh_ct_dev, lg_dev[ie-i])
                spgemm_direct(spgemm(vh_dev, pl_dev), vh_ct_dev, ll_dev[ie-i])

                time_spgemm += time.perf_counter()

                time_dense -= time.perf_counter()

                num_threads = min(1024, lb_start)
                num_thread_blocks = lb_start
                _toarray[num_thread_blocks, num_threads](vh_dev.data, vh_dev.indices, vh_dev.indptr, vh_s1[ie-i], slb_sd.start, slb_sd.stop, slb_sd.start, slb_sd.stop)
                _toarray[num_thread_blocks, num_threads](vh_dev.data, vh_dev.indices, vh_dev.indptr, vh_s2[ie-i], slb_sd.start, slb_sd.stop, slb_so.start, slb_so.stop)
                _toarray[num_thread_blocks, num_threads](pg_dev.data, pg_dev.indices, pg_dev.indptr, pg_s1[ie-i], slb_sd.start, slb_sd.stop, slb_sd.start, slb_sd.stop)
                _toarray[num_thread_blocks, num_threads](pg_dev.data, pg_dev.indices, pg_dev.indptr, pg_s2[ie-i], slb_sd.start, slb_sd.stop, slb_so.start, slb_so.stop)
                _toarray[num_thread_blocks, num_threads](pl_dev.data, pl_dev.indices, pl_dev.indptr, pl_s1[ie-i], slb_sd.start, slb_sd.stop, slb_sd.start, slb_sd.stop)
                _toarray[num_thread_blocks, num_threads](pl_dev.data, pl_dev.indices, pl_dev.indptr, pl_s2[ie-i], slb_sd.start, slb_sd.stop, slb_so.start, slb_so.stop)
                _toarray[num_thread_blocks, num_threads](pr_dev.data, pr_dev.indices, pr_dev.indptr, pr_s1[ie-i], slb_sd.start, slb_sd.stop, slb_sd.start, slb_sd.stop)
                _toarray[num_thread_blocks, num_threads](pr_dev.data, pr_dev.indices, pr_dev.indptr, pr_s2[ie-i], slb_sd.start, slb_sd.stop, slb_so.start, slb_so.stop)
                _toarray[num_thread_blocks, num_threads](pr_dev.data, pr_dev.indices, pr_dev.indptr, pr_s3[ie-i], slb_so.start, slb_so.stop, slb_sd.start, slb_sd.stop)

                num_threads = min(1024, lb_end)
                num_thread_blocks = lb_end
                _toarray[num_thread_blocks, num_threads](vh_dev.data, vh_dev.indices, vh_dev.indptr, vh_e1[ie-i], slb_ed.start, slb_ed.stop, slb_ed.start, slb_ed.stop)
                _toarray[num_thread_blocks, num_threads](vh_dev.data, vh_dev.indices, vh_dev.indptr, vh_e2[ie-i], slb_ed.start, slb_ed.stop, slb_eo.start, slb_eo.stop)
                _toarray[num_thread_blocks, num_threads](pg_dev.data, pg_dev.indices, pg_dev.indptr, pg_e1[ie-i], slb_ed.start, slb_ed.stop, slb_ed.start, slb_ed.stop)
                _toarray[num_thread_blocks, num_threads](pg_dev.data, pg_dev.indices, pg_dev.indptr, pg_e2[ie-i], slb_ed.start, slb_ed.stop, slb_eo.start, slb_eo.stop)
                _toarray[num_thread_blocks, num_threads](pl_dev.data, pl_dev.indices, pl_dev.indptr, pl_e1[ie-i], slb_ed.start, slb_ed.stop, slb_ed.start, slb_ed.stop)
                _toarray[num_thread_blocks, num_threads](pl_dev.data, pl_dev.indices, pl_dev.indptr, pl_e2[ie-i], slb_ed.start, slb_ed.stop, slb_eo.start, slb_eo.stop)
                _toarray[num_thread_blocks, num_threads](pr_dev.data, pr_dev.indices, pr_dev.indptr, pr_e1[ie-i], slb_ed.start, slb_ed.stop, slb_ed.start, slb_ed.stop)
                _toarray[num_thread_blocks, num_threads](pr_dev.data, pr_dev.indices, pr_dev.indptr, pr_e2[ie-i], slb_eo.start, slb_eo.stop, slb_ed.start, slb_ed.stop)
                _toarray[num_thread_blocks, num_threads](pr_dev.data, pr_dev.indices, pr_dev.indptr, pr_e3[ie-i], slb_ed.start, slb_ed.stop, slb_eo.start, slb_eo.stop)

                vh_e2[ie-i] = vh_e2[ie-i].T.conj()
                pg_e2[ie-i] = -pg_e2[ie-i].T.conj()
                pl_e2[ie-i] = -pl_e2[ie-i].T.conj()

                time_dense += time.perf_counter()
            
            
            # start_obc_mm = time.perf_counter()

            # obc_w_gpu.obc_w_mm_batched_gpu(vh_s1, vh_s2, pg_s1, pg_s2, pl_s1, pl_s2, pr_s1, pr_s2, pr_s3,
            #                                vh_e1, vh_e2, pg_e1, pg_e2, pl_e1, pl_e2, pr_e1, pr_e2, pr_e3,
            #                                dmr_sd, dmr_ed, dlg_sd, dlg_ed, dll_sd, dll_ed,
            #                             #    mr_s, mr_e, lg_s, lg_e, ll_s, ll_e,
            #                                mr_s0, mr_s1, mr_s2, mr_e0, mr_e1, mr_e2, lg_s0, lg_s1, lg_e0, lg_e1, ll_s0, ll_s1, ll_e0, ll_e1,
            #                                vh_s, vh_e, mb00, mbNN, nbc, NCpSC)
                
            time_obc_mm -= time.perf_counter()

            # obc_w_gpu.obc_w_mm_batched_gpu(vh_s1[:j-i], vh_s2[:j-i], pg_s1[:j-i], pg_s2[:j-i], pl_s1[:j-i], pl_s2[:j-i], pr_s1[:j-i], pr_s2[:j-i], pr_s3[:j-i],
            #                             vh_e1[:j-i], vh_e2[:j-i], pg_e1[:j-i], pg_e2[:j-i], pl_e1[:j-i], pl_e2[:j-i], pr_e1[:j-i], pr_e2[:j-i], pr_e3[:j-i],
            #                             dmr_sd[:j-i], dmr_ed[:j-i], dlg_sd[:j-i], dlg_ed[:j-i], dll_sd[:j-i], dll_ed[:j-i],
            #                             mr_s0[:j-i], mr_s1[:j-i], mr_s2[:j-i], mr_e0[:j-i], mr_e1[:j-i], mr_e2[:j-i],
            #                             lg_s0[:j-i], lg_s1[:j-i], lg_e0[:j-i], lg_e1[:j-i], ll_s0[:j-i], ll_s1[:j-i], ll_e0[:j-i], ll_e1[:j-i],
            #                             vh_s[:j-i], vh_e[:j-i], mb00[:j-i], mbNN[:j-i], nbc, NCpSC)
            (
                dmr_sd, dmr_ed, dlg_sd, dlg_ed, dll_sd, dll_ed,
                mr_s0, mr_s1, mr_s2, mr_e0, mr_e1, mr_e2,
                lg_s0, lg_s1, lg_e0, lg_e1, ll_s0, ll_s1, ll_e0, ll_e1,
                vh_s, vh_e, mb00, mbNN
            ) = obc_w_gpu.obc_w_mm_batched_gpu(vh_s1[:j-i], vh_s2[:j-i], pg_s1[:j-i], pg_s2[:j-i], pl_s1[:j-i], pl_s2[:j-i], pr_s1[:j-i], pr_s2[:j-i], pr_s3[:j-i],
                                        vh_e1[:j-i], vh_e2[:j-i], pg_e1[:j-i], pg_e2[:j-i], pl_e1[:j-i], pl_e2[:j-i], pr_e1[:j-i], pr_e2[:j-i], pr_e3[:j-i],
                                        nbc, NCpSC)
            
            time_obc_mm += time.perf_counter()

            time_beyn_w -= time.perf_counter()

            imag_lim = 1e-4
            R = 1e4
            # matrix_blocks_left = cp.asarray(mb00[:j-i])
            # M00_left = cp.asarray(mr_s0[:j-i])
            # M01_left = cp.asarray(mr_s1[:j-i])
            # M10_left = cp.asarray(mr_s2[:j-i])
            matrix_blocks_left = mb00
            M00_left = mr_s0
            M01_left = mr_s1
            M10_left = mr_s2
            dmr, dxr_sd_gpu, condl, _ = beyn_gpu(nbc * NCpSC, matrix_blocks_left, M00_left, M01_left, M10_left, imag_lim, R, 'L')
            # dxr_sd_gpu.get(out=dxr_sd[:j-i])
            dxr_sd = dxr_sd_gpu
            # dmr_sd[:j-i] -= dmr.get()
            dmr_sd -= dmr
            # (M10_left @ dxr_sd_gpu @ cp.asarray(vh_s[:j-i])).get(out=dvh_sd[:j-i])
            dvh_sd = M10_left @ dxr_sd_gpu @ vh_s
            # matrix_blocks_right = cp.asarray(mbNN[:j-i])
            # M00_right = cp.asarray(mr_e0[:j-i])
            # M01_right = cp.asarray(mr_e1[:j-i])
            # M10_right = cp.asarray(mr_e2[:j-i])
            matrix_blocks_right = mbNN[:j-i]
            M00_right = mr_e0
            M01_right = mr_e1
            M10_right = mr_e2
            dmr, dxr_ed_gpu, condr, _ = beyn_gpu(NCpSC, matrix_blocks_right, M00_right, M01_right, M10_right, imag_lim, R, 'R')
            # dxr_ed_gpu.get(out=dxr_ed[:j-i])
            dxr_ed = dxr_ed_gpu
            # dmr_ed[:j-i] -= dmr.get()
            dmr_ed -= dmr
            # (M01_right @ dxr_ed_gpu @ cp.asarray(vh_e[:j-i])).get(out=dvh_ed[:j-i])
            dvh_ed = M01_right @ dxr_ed_gpu @ vh_e

            condL.extend(condl)
            condR.extend(condr)

            time_beyn_w += time.perf_counter()

            time_obc_l -= time.perf_counter()

            worker_num = obc_w_batchsize

            # with concurrent.futures.ThreadPoolExecutor(max_workers=worker_num) as executor:
            #     executor.map(obc_w_gpu.obc_w_L_lg_2,
            #         dlg_sd[:j-i],
            #         dlg_ed[:j-i],
            #         dll_sd[:j-i],
            #         dll_ed[:j-i],
            #         mr_s0[:j-i], mr_s1[:j-i], mr_s2[:j-i],
            #         mr_e0[:j-i], mr_e1[:j-i], mr_e2[:j-i],
            #         lg_s0[:j-i], lg_s1[:j-i],
            #         lg_e0[:j-i], lg_e1[:j-i],
            #         ll_s0[:j-i], ll_s1[:j-i],
            #         ll_e0[:j-i], ll_e1[:j-i],
            #         dxr_sd[:j-i],
            #         dxr_ed[:j-i],
            #     )
            with concurrent.futures.ThreadPoolExecutor(max_workers=worker_num) as executor:
                executor.map(obc_w_gpu.obc_w_L_lg_2,
                    dlg_sd,
                    dlg_ed,
                    dll_sd,
                    dll_ed,
                    mr_s0, mr_s1, mr_s2,
                    mr_e0, mr_e1, mr_e2,
                    lg_s0, lg_s1,
                    lg_e0, lg_e1,
                    ll_s0, ll_s1,
                    ll_e0, ll_e1,
                    dxr_sd,
                    dxr_ed,
                )

            mr_s0, mr_s1, mr_s2 = None, None, None
            mr_e0, mr_e1, mr_e2 = None, None, None
            lg_s0, lg_s1 = None, None
            lg_e0, lg_e1 = None, None
            ll_s0, ll_s1 = None, None
            ll_e0, ll_e1 = None, None
            dxr_sd, dxr_ed = None, None
            mb00, mbNN = None, None
            matrix_blocks_left, matrix_blocks_right = None, None
            M00_left, M01_left, M10_left = None, None, None
            M00_right, M01_right, M10_right = None, None, None
            
            time_obc_l += time.perf_counter()

            time_w -= time.perf_counter()

            # rgf_W_GPU_combo.rgf_batched_GPU(
            #     energies=energy[i:j],
            #     map_diag_mm=mapping_diag_mm,
            #     map_upper_mm=mapping_upper_mm,
            #     map_lower_mm=mapping_lower_mm,
            #     map_diag_m=mapping_diag_m,
            #     map_upper_m=mapping_upper_m,
            #     map_lower_m=mapping_lower_m,
            #     map_diag_l=mapping_diag_l,
            #     map_upper_l=mapping_upper_l,
            #     map_lower_l=mapping_lower_l,
            #     vh_diag_host=vh_diag,
            #     vh_upper_host=vh_upper,
            #     vh_lower_host=vh_lower,
            #     mr_host=mr_host[:j-i],
            #     ll_host=ll_host[:j-i],
            #     lg_host=lg_host[:j-i],
            #     dvh_left_host=dvh_sd[:j-i],
            #     dvh_right_host=dvh_ed[:j-i],
            #     dmr_left_host=dmr_sd[:j-i],
            #     dmr_right_host=dmr_ed[:j-i],
            #     dlg_left_host=dlg_sd[:j-i],
            #     dlg_right_host=dlg_ed[:j-i],
            #     dll_left_host=dll_sd[:j-i],
            #     dll_right_host=dll_ed[:j-i],
            #     wr_host=wr_p2w[i:j],
            #     wl_host=wl_p2w[i:j],
            #     wg_host=wg_p2w[i:j],
            #     dosw=dosw[i:j],
            #     nEw=new[i:j],
            #     nPw=npw[i:j],
            #     bmax=bmax_mm,
            #     bmin=bmin_mm,
            #     solve=False,
            #     input_stream=input_stream,
            # )

            time_w += time.perf_counter()

            # print(f"Used bytes: {mempool.used_bytes()}", flush=True)
            # print(f"Total bytes: {mempool.total_bytes()}", flush=True)
        

        vh_s1 = None
        vh_s2 = None
        pg_s1 = None
        pg_s2 = None
        pl_s1 = None
        pl_s2 = None
        pr_s1 = None
        pr_s2 = None
        pr_s3 = None

        vh_e1 = None
        vh_e2 = None
        pg_e1 = None
        pg_e2 = None
        pl_e1 = None
        pl_e2 = None
        pr_e1 = None
        pr_e2 = None
        pr_e3 = None

        # comm.Barrier()
        # finish_obc_mm = time.perf_counter()
        # if rank == 0:
        #     print("    Time for obc mm: %.3f s" % (finish_obc_mm - start_obc_mm), flush=True)

        comm.Barrier()
        finish_spgemm = time.perf_counter()
        if rank == 0:
            print("        Time for CopyToDevice: %.3f s" % time_copy_in, flush=True)
            print("        Time for SpGEMM: %.3f s" % time_spgemm, flush=True)
            print("        Time for CopyToHost: %.3f s" % time_copy_out, flush=True)
            print("        Time for Densification: %.3f s" % time_dense, flush=True)
            print("        Time for OBC MM: %.3f s" % time_obc_mm, flush=True)
            print("        Time for Beyn W: %.3f s" % time_beyn_w, flush=True)
            print("        Time for OBC L: %.3f s" % time_obc_l, flush=True)
            print("        Time for RGF W: %.3f s" % time_w, flush=True)
    
        end_time = time.perf_counter()
        print(f"Total time: {end_time - start_time:.3f} s", flush = True)
        print()
