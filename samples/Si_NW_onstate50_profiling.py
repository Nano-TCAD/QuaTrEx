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

from quatrex.bandstructure.calc_band_edge import get_band_edge_mpi, get_band_edge_mpi_interpol
from quatrex.GW.polarization.kernel import g2p_cpu
from quatrex.GW.selfenergy.kernel import gw2s_cpu
from quatrex.GW.gold_solution import read_solution
from quatrex.GW.screenedinteraction.kernel import p2w_cpu
from quatrex.GW.coulomb_matrix.read_coulomb_matrix import load_V, load_V_mpi
from quatrex.GreensFunction import calc_GF_pool
from quatrex.OMEN_structure_matrices import OMENHamClass
from quatrex.OMEN_structure_matrices.construct_CM import construct_coulomb_matrix
from quatrex.utils import change_format
from quatrex.utils import utils_gpu
from quatrex.utils.bsr import bsr_matrix
from quatrex.utils.matrix_creation import get_number_connected_blocks
from quatrex.Phonon import electron_phonon_selfenergy

if utils_gpu.gpu_avail():
    try:
        from quatrex.GreensFunction import calc_GF_pool_GPU
    except ImportError:
        print("GPU import error, make sure you have the right GPU driver and CUDA version installed")

if __name__ == "__main__":
    MPI.Init_thread(required=MPI.THREAD_FUNNELED)
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    name = MPI.Get_processor_name()

    # assume every rank has enough memory to read the initial data
    # path to solution
    scratch_path = "/usr/scratch/mont-fort17/dleonard/GW_paper/"
    # scratch_path = "/scratch/aziogas/IEDM/"
    solution_path = os.path.join(scratch_path, "Si_Nanowire/")
    solution_path_gw = os.path.join(solution_path, "data_GPWS_IEDM_GNR_04V.mat")
    solution_path_gw2 = os.path.join(solution_path, "data_GPWS_IEDM_it2_GNR_04V.mat")
    solution_path_vh = os.path.join(solution_path, "V.dat")
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
    # parser.add_argument("-p", "--pool", default=True, type=bool, required=False)
    parser.add_argument('--pool', action='store_true', help='If True, use thread-pool')
    parser.add_argument('--no-pool', dest='pool', action='store_false')
    parser.set_defaults(pool=True)
    parser.add_argument('--block-inv', action='store_true', help='If True, use block inversion in Beyn')
    parser.add_argument('--no-block-inv', dest='block-inv', action='store_false')
    parser.set_defaults(block_inv=False)
    parser.add_argument('--bsr', action='store_true', help='If True, use bsr format for W')
    parser.add_argument('--no-bsr', dest='bsr', action='store_false')
    parser.set_defaults(bsr=False)
    parser.add_argument('--validate-bsr', action='store_true', help='If True, validate W with BSR')
    parser.add_argument('--no-validate-bsr', dest='validate-bsr', action='store_false')
    parser.set_defaults(validate_bsr=False)
    parser.add_argument('--dace', action='store_true', help='If True, use dace for Beyn')
    parser.add_argument('--no-dace', dest='dace', action='store_false')
    parser.set_defaults(dace=False)
    parser.add_argument('--validate-dace', action='store_true', help='If True, validate DaCe')
    parser.add_argument('--no-validate-dace', dest='validate-dace', action='store_false')
    parser.set_defaults(validate_dace=False)
    args = parser.parse_args()
    # check if gpu is available
    if args.type in ("gpu"):
        if not utils_gpu.gpu_avail():
            print("No gpu available")
            sys.exit(1)
    # print chosen implementation
    print(f"Using {args.type} implementation")


    if args.dace:
        import dace
        from dace.sdfg import utils
        if rank == 0:
            print("Using dace for Beyn")
            from quatrex.OBC.beyn_dace import contour_integral_dace, contour_integral_block_dace, sort_k_dace
            from dace.transformation.auto.auto_optimize import auto_optimize
            ci_sdfg = contour_integral_dace.to_sdfg(simplify=True)
            auto_optimize(ci_sdfg, dace.DeviceType.CPU, thread_safe=True)
            # ci_func = ci_sdfg.compile()
            ci_block_sdfg = contour_integral_block_dace.to_sdfg(simplify=True)
            auto_optimize(ci_block_sdfg, dace.DeviceType.CPU, thread_safe=True)
            # ci_block_func = ci_block_sdfg.compile()
            sk_sdfg = sort_k_dace.to_sdfg(simplify=True)
            auto_optimize(sk_sdfg, dace.DeviceType.CPU, thread_safe=True)
            # sk_func = sk_sdfg.compile()
        else:
            ci_sdfg, ci_block_sdfg, sk_sdfg = None, None, None
        comm.Barrier()
        import quatrex.OBC.beyn_globals as bg
        bg.contour_integral = utils.distributed_compile(ci_sdfg, comm)
        bg.contour_integral_block = utils.distributed_compile(ci_block_sdfg, comm)
        bg.sort_k = utils.distributed_compile(sk_sdfg, comm)
        comm.Barrier()

    # create hamiltonian object
    # one orbital on C atoms, two same types
    no_orb = np.array([1, 4])
    NCpSC = 4
    Vappl = 0.6
    energy = np.linspace(-4.695, 1.391, 208, endpoint = True, dtype = float) # Energy Vector
    Idx_e = np.arange(energy.shape[0]) # Energy Index Vector
    EPHN = np.array([0.0])  # Phonon energy
    DPHN = np.array([2.5e-3])  # Electron-phonon coupling
    hamiltonian_obj = OMENHamClass.Hamiltonian(args.file_hm, no_orb, Vappl = Vappl,  potential_type = 'atomic', bias_point = 13, rank = rank, layer_matrix = '/Layer_Matrix.dat')
    serial_ham = pickle.dumps(hamiltonian_obj)
    broadcasted_ham = comm.bcast(serial_ham, root=0)
    hamiltonian_obj = pickle.loads(broadcasted_ham)
    # Extract neighbor indices
    rows = hamiltonian_obj.rows
    columns = hamiltonian_obj.columns

    #Only keep diagonals of P and Sigma
    #rows = np.arange(hamiltonian_obj.NH, dtype = np.int32)
    #columns = np.arange(hamiltonian_obj.NH, dtype = np.int32)


    # hamiltonian object has 1-based indexing
    bmax = hamiltonian_obj.Bmax - 1
    bmin = hamiltonian_obj.Bmin - 1

    ij2ji:      npt.NDArray[np.int32]   = change_format.find_idx_transposed(rows, columns)
    denergy:    npt.NDArray[np.double]  = energy[1] - energy[0]
    ne:         np.int32                = np.int32(energy.shape[0])
    no:         np.int32                = np.int32(columns.shape[0])
    pre_factor: np.complex128           = -1.0j * denergy / (np.pi)
    nao:        np.int64                = np.max(bmax) + 1

    data_shape = np.array([rows.shape[0], energy.shape[0]], dtype=np.int32)

    map_diag, map_upper, map_lower = change_format.map_block2sparse_alt(rows, columns,
                                                                    bmax, bmin)

    # number of blocks
    nb = hamiltonian_obj.Bmin.shape[0]
    #nbc = 2
    nbc = get_number_connected_blocks(hamiltonian_obj.NH, bmin, bmax, rows, columns)
    bmax_mm = bmax[nbc-1:nb:nbc]
    bmin_mm = bmin[0:nb:nbc]

    map_diag_mm, map_upper_mm, map_lower_mm = change_format.map_block2sparse_alt(rows, columns, bmax_mm, bmin_mm)

    if rank == 0:
        # print size of data
        print(f"#Energy: {data_shape[1]} #nnz: {data_shape[0]}")


    # computation parameters----------------------------------------------------
    # set number of threads for the p2w step
    w_mkl_threads = 1
    w_worker_threads = 3
    # set number of threads for the h2g step
    gf_mkl_threads = 1
    gf_mkl_threads_gpu = 1
    gf_worker_threads = 3

    # physical parameter -----------

    # Fermi Level of Left Contact
    energy_fl = -2.0362
    # Fermi Level of Right Contact
    energy_fr = energy_fl - Vappl
    # Temperature in Kelvin
    temp = 300
    # relative permittivity
    epsR = 1.0
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
    factor_w = np.ones(ne)
    #factor_w[ne-dnp-1:ne] = (np.cos(np.pi*np.linspace(0, 1, dnp+1)) + 1)/2
    #factor_w[np.where(np.invert(w_mask))[0]] = 0.0

    # create factor for the Green's Function
    factor_g = np.ones(ne)
    #factor_g[ne-dnp-1:ne] = (np.cos(np.pi*np.linspace(0, 1, dnp+1)) + 1)/2
    #factor_g[0:dnp+1] = (np.cos(np.pi*np.linspace(1, 0, dnp+1)) + 1)/2

    vh = construct_coulomb_matrix(hamiltonian_obj, epsR, eps0, e, diag = False, orb_uniform = True)
    #vh = load_V_mpi(solution_path_vh, rows, columns, comm, rank)/epsR
    vh1d = np.squeeze(np.asarray(vh[np.copy(rows), np.copy(columns)].reshape(-1)))
    if args.bsr:
        w_bsize = vh.shape[0] // hamiltonian_obj.Bmin.shape[0]
        vh = bsr_matrix(vh.tobsr(blocksize=(w_bsize, w_bsize)))

     # calculation of data distribution per rank---------------------------------

    # split nnz/energy per rank
    data_per_rank = data_shape // size

    # create array with energy size distribution
    count = np.repeat(data_per_rank.reshape(-1, 1), size, axis=1)
    count[:, size-1] += data_shape % size

    # displacements in nnz/energy
    disp = data_per_rank.reshape(-1, 1) * np.arange(size)

    # slice energy vector
    energy_loc = energy[disp[1, rank]:disp[1, rank] + count[1, rank]]
    Idx_e_loc = Idx_e[disp[1, rank]:disp[1, rank] + count[1, rank]]

    # split up the factor between the ranks
    factor_w_loc = factor_w[disp[1, rank]:disp[1, rank] + count[1, rank]]
    factor_g_loc = factor_g[disp[1, rank]:disp[1, rank] + count[1, rank]]

    # print rank distribution
    print(
    f"Rank: {rank} #Energy/rank: {count[1,rank]} #nnz/rank: {count[0,rank]}", 
    name)

    # adding checks
    assert energy_loc.size == count[1,rank]

    # create needed data types--------------------------------------------------

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
    G2P_R = np.array([BASE_TYPE.Create_vector(
        count[0, rank], count[1, i], data_shape[1]) for i in range(size)])
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
    P2G_R = np.array([BASE_TYPE.Create_vector(
        count[1, rank], count[0, i], data_shape[0]) for i in range(size)])
    P2G_R_RIZ = np.empty_like(P2G_R)
    for i in range(size):
        P2G_R_RIZ[i] = P2G_R[i].Create_resized(0, base_size)
        MPI.Datatype.Commit(P2G_R[i])
        MPI.Datatype.Commit(P2G_R_RIZ[i])


    # define helper communication functions-------------------------------------
    # captures all variables from the outside (comm/count/disp/rank/size/types)

    def scatter_master(inp: npt.NDArray[np.complex128],
                       outp: npt.NDArray[np.complex128],
                       transpose_net: bool = False):
        if transpose_net:
            comm.Scatterv([inp, count[1, :], disp[1, :], COLUMN_RIZ], outp, root=0)
        else:
            if rank == 0:
                inp_transposed = np.copy(inp.T, order="C")
            else:
                inp_transposed = None
            comm.Scatterv([inp_transposed, count[1, :]*data_shape[0], disp[1, :]*data_shape[0], BASE_TYPE], outp, root=0)

    def gather_master(inp: npt.NDArray[np.complex128],
                      outp: npt.NDArray[np.complex128],
                      transpose_net: bool = False):
        if transpose_net:
            comm.Gatherv(inp, [outp, count[1, :], disp[1, :], COLUMN_RIZ], root=0)
        else:
            if rank == 0:
                out_transposed = np.copy(outp.T, order="C")
            else:
                out_transposed = None
            comm.Gatherv(inp, [out_transposed, count[1, :]*data_shape[0], disp[1, :]*data_shape[0], BASE_TYPE], root=0)
            if rank == 0:
                outp[:,:] = out_transposed.T
    
    def alltoall_g2p(inp: npt.NDArray[np.complex128],
                     outp: npt.NDArray[np.complex128],
                     transpose_net: bool = False):
        if transpose_net:
            comm.Alltoallw(
            [inp, count[0, :], disp[0, :]*base_size, np.repeat(G2P_S_RIZ, size)],
            [outp, np.repeat([1], size), disp[1, :]*base_size, G2P_R_RIZ])
        else:
            inp_transposed = np.copy(inp.T, order="C")
            comm.Alltoallw(
            [inp_transposed, count[0,:]*count[1, rank], disp[0, :]*count[1, rank]*base_size, np.repeat(BASE_TYPE, size)],
            [outp, np.repeat([1], size), disp[1, :]*base_size, G2P_R_RIZ])

    def alltoall_p2g(inp: npt.NDArray[np.complex128],
                     outp: npt.NDArray[np.complex128],
                     transpose_net: bool = False):
        if transpose_net:
            comm.Alltoallw(
            [inp, count[1, :], disp[1, :]*base_size, np.repeat(P2G_S_RIZ, size)],
            [outp, np.repeat([1], size), disp[0, :]*base_size, P2G_R_RIZ])
        else:
            inp_transposed = np.copy(inp.T, order="C")
            comm.Alltoallw(
            [inp_transposed, count[1,:]*count[0, rank], disp[1, :]*count[0, rank]*base_size, np.repeat(BASE_TYPE, size)],
            [outp, np.repeat([1], size), disp[0, :]*base_size, P2G_R_RIZ])



    # initialize self energy----------------------------------------------------
    sg_h2g = np.zeros((count[1,rank], no), dtype=np.complex128)
    sl_h2g = np.zeros((count[1,rank], no), dtype=np.complex128)
    sr_h2g = np.zeros((count[1,rank], no), dtype=np.complex128)

    # phonon self energy. Only diagonal so far----------------------------------
    sg_phn = np.zeros((count[1,rank], nao), dtype=np.complex128)
    sl_phn = np.zeros((count[1,rank], nao), dtype=np.complex128)
    sr_phn = np.zeros((count[1,rank], nao), dtype=np.complex128)

    # Transform the hamiltonian to a block tri-diagonal format
    if args.type in ("gpu"):
        nb = hamiltonian_obj.Bmin.shape[0]
        lb = np.max(hamiltonian_obj.Bmax - hamiltonian_obj.Bmin + 1)
        ne_loc = energy_loc.shape[0]
        blocked_hamiltonian_diag = np.zeros((nb, ne_loc, lb, lb), dtype=np.complex128)
        blocked_hamiltonian_upper = np.zeros((nb-1, ne_loc, lb, lb), dtype=np.complex128)
        blocked_hamiltonian_lower = np.zeros((nb-1, ne_loc, lb, lb), dtype=np.complex128)
        change_format.sparse2block_energyhamgen_no_map(hamiltonian_obj.Hamiltonian['H_4'], hamiltonian_obj.Overlap['H_4'], blocked_hamiltonian_diag, blocked_hamiltonian_upper, blocked_hamiltonian_lower, bmax, bmin, energy_loc)

    # initialize Green's function------------------------------------------------
    gg_h2g = np.zeros((count[1,rank], no), dtype=np.complex128)
    gl_h2g = np.zeros((count[1,rank], no), dtype=np.complex128)
    #gr_h2g = np.zeros((count[1,rank], no), dtype=np.complex128)

    # initialize Screened interaction-------------------------------------------
    wg_p2w = np.zeros((count[1,rank], no), dtype=np.complex128)
    wl_p2w = np.zeros((count[1,rank], no), dtype=np.complex128)
    #wr_p2w = np.zeros((count[1,rank], no), dtype=np.complex128)

    # initialize memory factors for Self-Energy, Green's Function and Screened interaction
    mem_s = 0.9
    mem_g = 0.0
    mem_w = 0.0
    # max number of iterations

    max_iter = 10
    ECmin_vec = np.concatenate((np.array([ECmin]), np.zeros(max_iter)))
    EFL_vec = np.concatenate((np.array([energy_fl]), np.zeros(max_iter)))
    EFR_vec = np.concatenate((np.array([energy_fr]), np.zeros(max_iter)))
    ind_ek = -1

    # Communication buffers
    # G2P
    gg_g2p = np.empty((count[0, rank], data_shape[1]),
                    dtype=np.complex128, order="C")
    gl_g2p = np.empty((count[0, rank], data_shape[1]),
                    dtype=np.complex128, order="C")
    # gr_g2p = np.empty((count[0, rank], data_shape[1]),
    #                 dtype=np.complex128, order="C")
    gl_transposed_g2p = np.empty((count[0, rank], data_shape[1]),
                    dtype=np.complex128, order="C")
    # P2W
    pg_p2w = np.empty((count[1, rank], data_shape[0]),
                    dtype=np.complex128, order="C")
    pl_p2w = np.empty((count[1, rank], data_shape[0]),
                    dtype=np.complex128, order="C")
    # pr_p2w = np.empty((count[1, rank], data_shape[0]),
    #                 dtype=np.complex128, order="C")
    # GW2S
    wg_gw2s = np.empty((count[0, rank], data_shape[1]),
                    dtype=np.complex128, order="C")
    wl_gw2s = np.empty((count[0, rank], data_shape[1]),
                    dtype=np.complex128, order="C")
    # wr_gw2s = np.empty((count[0, rank], data_shape[1]),
    #                 dtype=np.complex128, order="C")
    wg_transposed_gw2s = np.empty((count[0, rank], data_shape[1]),
                    dtype=np.complex128, order="C")
    # wg_gw2s = gg_g2p
    # wl_gw2s = gl_g2p
    # wr_gw2s = gr_g2p
    # wg_transposed_gw2s = gl_transposed_g2p
    wl_transposed_gw2s = np.empty((count[0, rank], data_shape[1]),
                    dtype=np.complex128, order="C")
    # H2G
    sg_h2g_buf = np.empty((count[1, rank], data_shape[0]),
                    dtype=np.complex128, order="C")
    sl_h2g_buf = np.empty((count[1, rank], data_shape[0]),
                    dtype=np.complex128, order="C")
    sr_h2g_buf = np.empty((count[1, rank], data_shape[0]),
                    dtype=np.complex128, order="C")
    # sg_h2g_buf = pg_p2w
    # sl_h2g_buf = pl_p2w
    # sr_h2g_buf = pr_p2w

    comm.Barrier()

    if rank == 0:
        time_start = -time.perf_counter()
    # output folder
    folder = '/quatrex/results/SINW_biased_epsR1_n50/'
    for iter_num in range(max_iter):

        comm.Barrier()

        if rank == 0:
            iter_time = -time.perf_counter()
            pre_gf_time = -time.perf_counter()
            print(f"Iteration {iter_num+1} of {max_iter}:", flush=True)
            #print(f" norm of sg_h2g: {np.linalg.norm(sg_h2g):.3e}", flush=True)

        # initialize observables----------------------------------------------------
        # density of states
        dos = np.zeros(shape=(ne,nb), dtype = np.complex128)
        dosw = np.zeros(shape=(ne,nb//nbc), dtype = np.complex128)

        # occupied states/unoccupied states
        nE = np.zeros(shape=(ne,nb), dtype = np.complex128)
        nP = np.zeros(shape=(ne,nb), dtype = np.complex128)

        # occupied screening/unoccupied screening
        nEw = np.zeros(shape=(ne,nb//nbc), dtype = np.complex128)
        nPw = np.zeros(shape=(ne,nb//nbc), dtype = np.complex128)

        # current per energy
        ide = np.zeros(shape=(ne,nb), dtype = np.complex128)

        

        # transform from 2D format to list/vector of sparse arrays format-----------
        sg_h2g_vec = change_format.sparse2vecsparse_v2(sg_h2g, rows, columns, nao)
        sl_h2g_vec = change_format.sparse2vecsparse_v2(sl_h2g, rows, columns, nao)
        sr_h2g_vec = change_format.sparse2vecsparse_v2(sr_h2g, rows, columns, nao)

        # transform from 2D format to list/vector of sparse arrays format-----------
        sg_ephn_h2g_vec = change_format.sparse2vecsparse_v2(sg_phn, np.arange(nao), np.arange(nao), nao)
        sl_ephn_h2g_vec = change_format.sparse2vecsparse_v2(sl_phn, np.arange(nao), np.arange(nao), nao)
        sr_ephn_h2g_vec = change_format.sparse2vecsparse_v2(sr_phn, np.arange(nao), np.arange(nao), nao)

        # Adjusting Fermi Levels of both contacts to the current iteration band minima
        (ECmin_vec[iter_num+1], ind_ek) = get_band_edge_mpi_interpol(ECmin_vec[iter_num],
                                                            energy,
                                                            hamiltonian_obj.Overlap['H_4'], 
                                                            hamiltonian_obj.Hamiltonian['H_4'], 
                                                            sr_h2g_vec,
                                                            sl_h2g_vec,
                                                            sg_h2g_vec,
                                                            sr_ephn_h2g_vec,
                                                            ind_ek,
                                                            rows, 
                                                            columns, 
                                                            bmin, 
                                                            bmax, 
                                                            comm, 
                                                            rank, 
                                                            size, 
                                                            count, 
                                                            disp, 
                                                            side = 'left')
        if iter_num == 0:
            dEfL_EC = energy_fl - ECmin_vec[iter_num + 1]
            dEfR_EC = energy_fr - ECmin_vec[iter_num + 1]
        else:
            energy_fl = ECmin_vec[iter_num + 1] + dEfL_EC
            energy_fr = ECmin_vec[iter_num + 1] + dEfR_EC

        EFL_vec[iter_num+1] = energy_fl
        EFR_vec[iter_num+1] = energy_fr

        comm.Barrier()
            
        if rank == 0:
            pre_gf_time += time.perf_counter()
            print(f"    Pre-GF time: {pre_gf_time:.3f} s", flush=True)
            gf_time = -time.perf_counter()

        # calculate the green's function at every rank------------------------------
        if args.type in ("cpu"):
            _, _, gl_diag, gl_upper, gg_diag, gg_upper = calc_GF_pool.calc_GF_pool_mpi_split(
                hamiltonian_obj,
                energy_loc,
                sr_h2g_vec,
                sl_h2g_vec,
                sg_h2g_vec,
                sr_ephn_h2g_vec,
                sl_ephn_h2g_vec,
                sg_ephn_h2g_vec,
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
                return_sigma_boundary=False,
                NCpSC=NCpSC,
                mkl_threads=gf_mkl_threads,
                worker_num=gf_worker_threads)
        elif args.type in ("gpu"):
            gr_diag, gr_upper, gl_diag, gl_upper, gg_diag, gg_upper = calc_GF_pool_GPU.calc_GF_pool_mpi_split(
                hamiltonian_obj,
                blocked_hamiltonian_diag,
                blocked_hamiltonian_upper,
                blocked_hamiltonian_lower,
                energy_loc,
                sr_h2g_vec,
                sl_h2g_vec,
                sg_h2g_vec,
                sr_ephn_h2g_vec,
                sl_ephn_h2g_vec,
                sg_ephn_h2g_vec,
                sr_h2g,
                sl_h2g,
                sg_h2g,
                sr_phn,
                sl_phn,
                sg_phn,
                map_diag,
                map_upper,
                map_lower,
                rows,
                columns,
                ij2ji,
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
                return_sigma_boundary=False,
                NCpSC=NCpSC,
                mkl_threads=gf_mkl_threads_gpu,
                worker_num=gf_worker_threads)
        
        comm.Barrier()

        if rank == 0:
            gf_time += time.perf_counter()
            print(f"    GF time: {gf_time:.3f} s", flush=True)
            pre_comm0_time = -time.perf_counter()
            

        # lower diagonal blocks from physics identity
        gg_lower = -gg_upper.conjugate().transpose((0,1,3,2))
        gl_lower = -gl_upper.conjugate().transpose((0,1,3,2))
        #gr_lower = gr_upper.transpose((0,1,3,2))
        if iter_num == 0:
            gg_h2g = change_format.block2sparse_energy_alt(map_diag, map_upper,
                                                            map_lower, gg_diag, gg_upper,
                                                            gg_lower, no, count[1,rank],
                                                            energy_contiguous=False)
            gl_h2g = change_format.block2sparse_energy_alt(map_diag, map_upper,
                                                            map_lower, gl_diag, gl_upper,
                                                            gl_lower, no, count[1,rank],
                                                            energy_contiguous=False)
            # gr_h2g = change_format.block2sparse_energy_alt(map_diag, map_upper,
            #                                                 map_lower, gr_diag, gr_upper,
            #                                                 gr_lower, no, count[1,rank],
            #                                                 energy_contiguous=False)
        else:   
            # add new contribution to the Green's function
            gg_h2g = (1.0 - mem_g) * change_format.block2sparse_energy_alt(map_diag, map_upper,
                                                            map_lower, gg_diag, gg_upper,
                                                            gg_lower, no, count[1,rank],
                                                            energy_contiguous=False) + mem_g * gg_h2g
            gl_h2g = (1.0 - mem_g) * change_format.block2sparse_energy_alt(map_diag, map_upper,
                                                            map_lower, gl_diag, gl_upper,
                                                            gl_lower, no, count[1,rank],
                                                            energy_contiguous=False) + mem_g * gl_h2g
            # gr_h2g = (1.0 - mem_g) * change_format.block2sparse_energy_alt(map_diag, map_upper,
            #                                                 map_lower, gr_diag, gr_upper,
            #                                                 gr_lower, no, count[1,rank],
            #                                                 energy_contiguous=False) + mem_g * gr_h2g 


        # calculate the transposed
        gl_transposed_h2g = np.copy(gl_h2g[:,ij2ji], order="C")
        # # create local buffers
        # gg_g2p = np.empty((count[0, rank], data_shape[1]),
        #                 dtype=np.complex128, order="C")
        # gl_g2p = np.empty((count[0, rank], data_shape[1]),
        #                 dtype=np.complex128, order="C")
        # gr_g2p = np.empty((count[0, rank], data_shape[1]),
        #                 dtype=np.complex128, order="C")
        # gl_transposed_g2p = np.empty((count[0, rank], data_shape[1]),
        #                 dtype=np.complex128, order="C")
        
        comm.Barrier()

        if rank == 0:
            pre_comm0_time += time.perf_counter()
            print(f"    Pre-Comm-0 time: {pre_comm0_time:.3f} s", flush=True)
            comm0_time = -time.perf_counter()

        # use of all to all w since not divisible
        alltoall_g2p(gg_h2g, gg_g2p, transpose_net=args.net_transpose)
        alltoall_g2p(gl_h2g, gl_g2p, transpose_net=args.net_transpose)
        #alltoall_g2p(gr_h2g, gr_g2p, transpose_net=args.net_transpose)
        alltoall_g2p(gl_transposed_h2g, gl_transposed_g2p, transpose_net=args.net_transpose)

        comm.Barrier()

        if rank == 0:
            comm0_time += time.perf_counter()
            print(f"    Comm-0 time: {comm0_time:.3f} s", flush=True)
            g2p_time = -time.perf_counter()

        # calculate the polarization at every rank----------------------------------
        if args.type in ("gpu") or args.type in ("cpu"):
            pg_g2p, pl_g2p = g2p_cpu.g2p_fft_mpi_cpu_inlined_nopr(
                                                pre_factor,
                                                gg_g2p,
                                                gl_g2p,
                                                gl_transposed_g2p)
        else:
            raise ValueError("Argument error, input type not possible")

        comm.Barrier()

        if rank == 0:
            g2p_time += time.perf_counter()
            print(f"    G2P time: {g2p_time:.3f} s", flush=True)
            pre_comm1_time = -time.perf_counter()

        # distribute polarization function according to p2w step--------------------

        # # create local buffers
        # pg_p2w = np.empty((count[1, rank], data_shape[0]),
        #                 dtype=np.complex128, order="C")
        # pl_p2w = np.empty((count[1, rank], data_shape[0]),
        #                 dtype=np.complex128, order="C")
        # pr_p2w = np.empty((count[1, rank], data_shape[0]),
        #                 dtype=np.complex128, order="C")

        comm.Barrier()

        if rank == 0:
            pre_comm1_time += time.perf_counter()
            print(f"    Pre-Comm-1 time: {pre_comm1_time:.3f} s", flush=True)
            comm1_time = -time.perf_counter()

        # use of all to all w since not divisible
        alltoall_p2g(pg_g2p, pg_p2w, transpose_net=args.net_transpose)
        alltoall_p2g(pl_g2p, pl_p2w, transpose_net=args.net_transpose)
        #alltoall_p2g(pr_g2p, pr_p2w, transpose_net=args.net_transpose)

        comm.Barrier()

        if rank == 0:
            comm1_time += time.perf_counter()
            print(f"    Comm-1 time: {comm1_time:.3f} s", flush=True)
            p2w_time = -time.perf_counter()

        if args.bsr:

            # transform from 2D format to list/vector of sparse arrays format-----------
            pg_p2w_vec = change_format.sparse2vecbsr_v2(pg_p2w, rows, columns, nao, w_bsize)
            pl_p2w_vec = change_format.sparse2vecbsr_v2(pl_p2w, rows, columns, nao, w_bsize)
            pr_p2w_vec = change_format.sparse2vecbsr_v2(pg_p2w, rows, columns, nao, w_bsize)

            # calculate the screened interaction on every rank--------------------------
            if args.pool:
                wg_diag_bsr, wg_upper_bsr, wl_diag_bsr, wl_upper_bsr, wr_diag_bsr, wr_upper_bsr, nb_mm, lb_max_mm, ind_zeros = p2w_cpu.p2w_pool_mpi_cpu(
                                                                                                    hamiltonian_obj, energy_loc,
                                                                                                    pg_p2w_vec, pl_p2w_vec,
                                                                                                    pr_p2w_vec, vh, dosw[disp[1, rank]:disp[1, rank] + count[1, rank]],
                                                                                                    nEw[disp[1, rank]:disp[1, rank] + count[1, rank]], nPw[disp[1, rank]:disp[1, rank] + count[1, rank]],
                                                                                                    Idx_e_loc,   
                                                                                                    factor_w_loc,
                                                                                                    comm,
                                                                                                    rank,
                                                                                                    size,
                                                                                                    nbc,
                                                                                                    mkl_threads = w_mkl_threads,
                                                                                                    worker_threads = w_worker_threads,
                                                                                                    block_inv=args.block_inv,
                                                                                                    use_dace=args.dace,
                                                                                                    validate_dace=args.validate_dace)
            else:
                wg_diag_bsr, wg_upper_bsr, wl_diag_bsr, wl_upper_bsr, wr_diag_bsr, wr_upper_bsr, nb_mm, lb_max_mm = p2w_cpu.p2w_mpi_cpu(
                                                                                                    hamiltonian_obj, energy_loc,
                                                                                                    pg_p2w_vec, pl_p2w_vec,
                                                                                                    pr_p2w_vec, vh, dosw[disp[1, rank]:disp[1, rank] + count[1, rank]],  
                                                                                                    nEw[disp[1, rank]:disp[1, rank] + count[1, rank]], nPw[disp[1, rank]:disp[1, rank] + count[1, rank]],    
                                                                                                    factor_w_loc,
                                                                                                    comm,
                                                                                                    rank,
                                                                                                    size,
                                                                                                    w_mkl_threads,
                                                                                                    block_inv=args.block_inv,
                                                                                                    use_dace=args.dace,
                                                                                                    validate_dace=args.validate_dace)
        
        if not args.bsr or (args.bsr and args.validate_bsr):

            # transform from 2D format to list/vector of sparse arrays format-----------
            pg_p2w_vec = change_format.sparse2vecsparse_v2(pg_p2w, rows, columns, nao)
            pl_p2w_vec = change_format.sparse2vecsparse_v2(pl_p2w, rows, columns, nao)
            pr_p2w_vec = change_format.sparse2vecsparse_v2(pg_p2w, rows, columns, nao)


            # calculate the screened interaction on every rank--------------------------
            if args.pool:
                wg_diag, wg_upper, wl_diag, wl_upper, _, _, nb_mm, lb_max_mm, ind_zeros = p2w_cpu.p2w_pool_mpi_cpu(
                                                                                                    hamiltonian_obj,
                                                                                                    energy_loc,
                                                                                                    pg_p2w_vec, 
                                                                                                    pl_p2w_vec,
                                                                                                    pr_p2w_vec, 
                                                                                                    vh, 
                                                                                                    dosw[disp[1, rank]:disp[1, rank] + count[1, rank]],
                                                                                                    nEw[disp[1, rank]:disp[1, rank] + count[1, rank]], 
                                                                                                    nPw[disp[1, rank]:disp[1, rank] + count[1, rank]],
                                                                                                    Idx_e_loc,   
                                                                                                    factor_w_loc,
                                                                                                    comm,
                                                                                                    rank,
                                                                                                    size,
                                                                                                    nbc,
                                                                                                    homogenize = False,
                                                                                                    NCpSC = NCpSC,
                                                                                                    mkl_threads = w_mkl_threads,
                                                                                                    worker_num = w_worker_threads,
                                                                                                    block_inv=args.block_inv,
                                                                                                    use_dace=args.dace,
                                                                                                    validate_dace=args.validate_dace)
            else:
                wg_diag, wg_upper, wl_diag, wl_upper, wr_diag, wr_upper, nb_mm, lb_max_mm, ind_zeros = p2w_cpu.p2w_mpi_cpu(
                                                                                                    hamiltonian_obj, energy_loc,
                                                                                                    pg_p2w_vec, pl_p2w_vec,
                                                                                                    pr_p2w_vec, vh, dosw[disp[1, rank]:disp[1, rank] + count[1, rank]],  
                                                                                                    nEw[disp[1, rank]:disp[1, rank] + count[1, rank]], nPw[disp[1, rank]:disp[1, rank] + count[1, rank]],    
                                                                                                    factor_w_loc,
                                                                                                    comm,
                                                                                                    rank,
                                                                                                    size,
                                                                                                    w_mkl_threads,
                                                                                                    block_inv=args.block_inv,
                                                                                                    use_dace=args.dace,
                                                                                                    validate_dace=args.validate_dace)
            
            if args.bsr and args.validate_bsr:
                assert np.allclose(wg_diag, wg_diag_bsr)
                assert np.allclose(wg_upper, wg_upper_bsr)
                assert np.allclose(wl_diag, wl_diag_bsr)
                assert np.allclose(wl_upper, wl_upper_bsr)
                assert np.allclose(wr_diag, wr_diag_bsr)
                assert np.allclose(wr_upper, wr_upper_bsr)
                if rank == 0:
                    print("Validation passed!")
        
        if args.bsr:
            wg_diag, wg_upper, wl_diag, wl_upper, wr_diag, wr_upper = wg_diag_bsr, wg_upper_bsr, wl_diag_bsr, wl_upper_bsr, wr_diag_bsr, wr_upper_bsr
            
        comm.Barrier()

        if rank == 0:
            p2w_time += time.perf_counter()
            print(f"    P2W time: {p2w_time:.3f} s", flush=True)
            print(f"    # of ind_zeros: {ind_zeros.shape[0]}", flush=True)
            pre_comm2_time = -time.perf_counter()

        memory_mask = np.ones(energy_loc.shape[0], dtype=bool)
        memory_mask[ind_zeros] = False

        # transform from block format to 2D format-----------------------------------
        # lower diagonal blocks from physics identity
        wg_lower = -wg_upper.conjugate().transpose((0,1,3,2))
        wl_lower = -wl_upper.conjugate().transpose((0,1,3,2))
        #wr_lower = wr_upper.transpose((0,1,3,2))
        # if iter_num == 0:
        #     wg_p2w = change_format.block2sparse_energy_alt(map_diag_mm, map_upper_mm,
        #                                                     map_lower_mm, wg_diag, wg_upper,
        #                                                     wg_lower, no, count[1,rank],
        #                                                     energy_contiguous=False)
        #     wl_p2w = change_format.block2sparse_energy_alt(map_diag_mm, map_upper_mm,
        #                                                     map_lower_mm, wl_diag, wl_upper,
        #                                                     wl_lower, no, count[1,rank],
        #                                                     energy_contiguous=False)
        #     wr_p2w = change_format.block2sparse_energy_alt(map_diag_mm, map_upper_mm,
        #                                                     map_lower_mm, wr_diag, wr_upper,
        #                                                     wr_lower, no, count[1,rank],
        #                                                     energy_contiguous=False)
        # else:
        # add new contribution to the Screened interaction
        wg_p2w[memory_mask] = (1.0 - mem_w) * change_format.block2sparse_energy_alt(map_diag_mm, map_upper_mm,
                                                        map_lower_mm, wg_diag, wg_upper,
                                                        wg_lower, no, count[1,rank],
                                                        energy_contiguous=False)[memory_mask] + mem_w * wg_p2w[memory_mask]
        wg_p2w[ind_zeros, :] = 0.0 + 0.0j
        wl_p2w[memory_mask] = (1.0 - mem_w) * change_format.block2sparse_energy_alt(map_diag_mm, map_upper_mm,
                                                        map_lower_mm, wl_diag, wl_upper,
                                                        wl_lower, no, count[1,rank],
                                                        energy_contiguous=False)[memory_mask] + mem_w * wl_p2w[memory_mask]
        wl_p2w[ind_zeros, :] = 0.0 + 0.0j
        # wr_p2w[memory_mask] = (1.0 - mem_w) * change_format.block2sparse_energy_alt(map_diag_mm, map_upper_mm,
        #                                                 map_lower_mm, wr_diag, wr_upper,
        #                                                 wr_lower, no, count[1,rank],
        #                                                 energy_contiguous=False)[memory_mask] + mem_w * wr_p2w[memory_mask]


        # distribute screened interaction according to gw2s step--------------------

        # calculate the transposed
        wg_transposed_p2w = np.copy(wg_p2w[:,ij2ji], order="C")
        wl_transposed_p2w = np.copy(wl_p2w[:,ij2ji], order="C")

        # # create local buffers
        # wg_gw2s = np.empty((count[0, rank], data_shape[1]),
        #                 dtype=np.complex128, order="C")
        # wl_gw2s = np.empty((count[0, rank], data_shape[1]),
        #                 dtype=np.complex128, order="C")
        # wr_gw2s = np.empty((count[0, rank], data_shape[1]),
        #                 dtype=np.complex128, order="C")
        # wg_transposed_gw2s = np.empty((count[0, rank], data_shape[1]),
        #                 dtype=np.complex128, order="C")
        # wl_transposed_gw2s = np.empty((count[0, rank], data_shape[1]),
        #                 dtype=np.complex128, order="C")
        
        comm.Barrier()

        if rank == 0:
            pre_comm2_time += time.perf_counter()
            print(f"    Pre-Comm-2 time: {pre_comm2_time:.3f} s", flush=True)
            #print(f" Norm of wg_p2w: {np.linalg.norm(wg_p2w):.3e}", flush=True)
            comm2_time = -time.perf_counter()
        

        # use of all to all w since not divisible
        alltoall_g2p(wg_p2w, wg_gw2s, transpose_net=args.net_transpose)
        alltoall_g2p(wl_p2w, wl_gw2s, transpose_net=args.net_transpose)
        #alltoall_g2p(wr_p2w, wr_gw2s, transpose_net=args.net_transpose)
        alltoall_g2p(wg_transposed_p2w, wg_transposed_gw2s, transpose_net=args.net_transpose)
        alltoall_g2p(wl_transposed_p2w, wl_transposed_gw2s, transpose_net=args.net_transpose)
    
        comm.Barrier()

        if rank == 0:
            comm2_time += time.perf_counter()
            print(f"    Comm-2 time: {comm2_time:.3f} s", flush=True)
            gw2s_time = -time.perf_counter()

    # tod optimize and not load two time green's function to gpu and do twice the fft
        if args.type in ("gpu") or args.type in ("cpu"):
            sg_gw2s, sl_gw2s, sr_gw2s = gw2s_cpu.gw2s_fft_mpi_cpu_PI_sr(-pre_factor / 2, gg_g2p, gl_g2p, 
                                                                           wg_gw2s, wl_gw2s, 
                                                                            wg_transposed_gw2s, wl_transposed_gw2s, vh1d, energy, rank, disp, count)
            # sg_gw2s, sl_gw2s, sr_gw2s = gw2s_cpu.gw2s_fft_mpi_cpu_3part_sr(
            #                                                     -pre_factor/2,
            #                                                     gg_g2p,
            #                                                     gl_g2p,
            #                                                     gr_g2p,
            #                                                     wg_gw2s,
            #                                                     wl_gw2s,
            #                                                     wr_gw2s,
            #                                                     wg_transposed_gw2s,
            #                                                     wl_transposed_gw2s
            #                                                     )

            # sg_gw2s, sl_gw2s, sr_gw2s = gw2s_cpu.gw2s_fft_mpi_cpu(
            #                                                     -pre_factor/2,
            #                                                     gg_g2p,
            #                                                     gl_g2p,
            #                                                     gr_g2p,
            #                                                     wg_gw2s,
            #                                                     wl_gw2s,
            #                                                     wr_gw2s,
            #                                                     wg_transposed_gw2s,
            #                                                     wl_transposed_gw2s
            #                                                     )
        else:
            raise ValueError("Argument error, input type not possible")
        
        comm.Barrier()

        if rank == 0:
            gw2s_time += time.perf_counter()
            print(f"    GW2S time: {gw2s_time:.3f} s", flush=True)
            pre_comm3_time = -time.perf_counter()

        # distribute screened interaction according to h2g step---------------------
        # # create local buffers
        # sg_h2g_buf = np.empty((count[1, rank], data_shape[0]),
        #                 dtype=np.complex128, order="C")
        # sl_h2g_buf = np.empty((count[1, rank], data_shape[0]),
        #                 dtype=np.complex128, order="C")
        # sr_h2g_buf = np.empty((count[1, rank], data_shape[0]),
        #                 dtype=np.complex128, order="C")
        
        comm.Barrier()

        if rank == 0:
            pre_comm3_time += time.perf_counter()
            print(f"    Pre-comm-3 time: {pre_comm3_time:.3f} s", flush=True)
            comm3_time = -time.perf_counter()

        # use of all to all w since not divisible
        alltoall_p2g(sg_gw2s, sg_h2g_buf, transpose_net=args.net_transpose)
        alltoall_p2g(sl_gw2s, sl_h2g_buf, transpose_net=args.net_transpose)
        alltoall_p2g(sr_gw2s, sr_h2g_buf, transpose_net=args.net_transpose)

        comm.Barrier()

        if rank == 0:
            comm3_time += time.perf_counter()
            print(f"    Comm-3 time: {comm3_time:.3f} s", flush=True)
            wrapping_up_time = -time.perf_counter()

        if iter_num == 0:
            sg_h2g = (1.0 - mem_s) * sg_h2g_buf + mem_s * sg_h2g
            sl_h2g = (1.0 - mem_s) * sl_h2g_buf + mem_s * sl_h2g
            sr_h2g = (1.0 - mem_s) * sr_h2g_buf + mem_s * sr_h2g
        else:
            # add new contribution to the Self-Energy
            sg_h2g = (1.0 - mem_s) * sg_h2g_buf + mem_s * sg_h2g
            sl_h2g = (1.0 - mem_s) * sl_h2g_buf + mem_s * sl_h2g
            sr_h2g = (1.0 - mem_s) * sr_h2g_buf + mem_s * sr_h2g
        
        # Extract diagonal bands
        gg_diag_band = gg_h2g[:, rows == columns]
        gl_diag_band = gl_h2g[:, rows == columns]
        # Add imaginary self energy to broaden peaks (motivated by a zero energy phonon interaction)
        # The Phonon energy (EPHN) is set to zero and the phonon-electron potential (DPHN) is set to 2.5e-3
        # at the beginning of this script. Only diagonal part now!
        sg_phn, sl_phn, sr_phn = electron_phonon_selfenergy.calc_SE_GF_EPHN(energy_loc,
                                                                            gl_diag_band,
                                                                            gg_diag_band,
                                                                            sg_phn,
                                                                            sl_phn,
                                                                            sr_phn,
                                                                            EPHN,
                                                                            DPHN,
                                                                            temp,
                                                                            mem_s)


        # if iter_num == max_iter - 1:
        #     alltoall_p2g(sg_gw2s, sg_h2g, transpose_net=args.net_transpose)
        #     alltoall_p2g(sl_gw2s, sl_h2g, transpose_net=args.net_transpose)
        #     alltoall_p2g(sr_gw2s, sr_h2g, transpose_net=args.net_transpose)

        
        # Wrapping up the iteration
        if rank == 0:
            comm.Reduce(MPI.IN_PLACE, dos, op=MPI.SUM, root=0)
            comm.Reduce(MPI.IN_PLACE, ide, op=MPI.SUM, root=0)

        else:
            comm.Reduce(dos, None, op=MPI.SUM, root=0)
            comm.Reduce(ide, None, op=MPI.SUM, root=0)
        
        if rank == 0:
            wrapping_up_time += time.perf_counter()
            print(f"    Wrapping-up time: {wrapping_up_time:.3f} s", flush=True)
            iter_time += time.perf_counter()
            print(f"Iteration time: {iter_time:.3f} s", flush=True)
            print()

        if rank == 0:
            np.savetxt(parent_path + folder + 'E.dat', energy)
            np.savetxt(parent_path + folder + 'DOS_' + str(iter_num) + '.dat', dos.view(float))
            np.savetxt(parent_path + folder + 'IDE_' + str(iter_num) + '.dat', ide.view(float))
            np.savetxt(parent_path + folder + 'EFL.dat', EFL_vec)
            np.savetxt(parent_path + folder + 'EFR.dat', EFR_vec)
            np.savetxt(parent_path + folder + 'ECmin.dat', ECmin_vec)
    if rank == 0:
        np.savetxt(parent_path + folder + 'EFL.dat', EFL_vec)
        np.savetxt(parent_path + folder + 'EFR.dat', EFR_vec)
        np.savetxt(parent_path + folder + 'ECmin.dat', ECmin_vec)

    # free datatypes------------------------------------------------------------

    MPI.Datatype.Free(COLUMN_RIZ)
    MPI.Datatype.Free(COLUMN)
    MPI.Datatype.Free(ROW_RIZ)
    MPI.Datatype.Free(ROW)
    MPI.Datatype.Free(G2P_S_RIZ)
    MPI.Datatype.Free(G2P_S)
    MPI.Datatype.Free(P2G_S_RIZ)
    MPI.Datatype.Free(P2G_S)
    for i in range(size):
        MPI.Datatype.Free(G2P_R_RIZ[i])
        MPI.Datatype.Free(G2P_R[i])
        MPI.Datatype.Free(P2G_R_RIZ[i])
        MPI.Datatype.Free(P2G_R[i])


    # finalize
    MPI.Finalize()

    if rank == 0:
        time_start += time.perf_counter()
        print(f"Time: {time_start:.2f} s")
