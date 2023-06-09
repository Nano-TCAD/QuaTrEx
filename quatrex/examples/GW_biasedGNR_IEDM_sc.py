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
from OMEN_structure_matrices.construct_CM import construct_coulomb_matrix
from utils import change_format
from utils import utils_gpu
from utils.bsr import bsr_matrix

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
    solution_path = os.path.join(scratch_path, "GNR_pd")
    solution_path_gw = os.path.join(solution_path, "data_GPWS_IEDM_GNR_04V.mat")
    solution_path_gw2 = os.path.join(solution_path, "data_GPWS_IEDM_it2_GNR_04V.mat")
    solution_path_vh = os.path.join(solution_path, "data_Vh_IEDM_GNR_0v.mat")
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
    parser.add_argument('--bsr', action='store_true', help='If True, use bsr format for W')
    parser.add_argument('--no-bsr', dest='bsr', action='store_false')
    parser.set_defaults(bsr=False)
    parser.add_argument('--validate', action='store_true', help='If True, validate W')
    parser.add_argument('--no-validate', dest='validate', action='store_false')
    parser.set_defaults(validate=False)
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
    no_orb = np.array([3, 3, 3])
    Vappl = 0.4
    energy = np.linspace(-8, 12.0, 4001, endpoint = True, dtype = float) # Energy Vector
    Idx_e = np.arange(energy.shape[0]) # Energy Index Vector
    hamiltonian_obj = OMENHamClass.Hamiltonian(args.file_hm, no_orb, Vappl = Vappl, rank = rank)
    serial_ham = pickle.dumps(hamiltonian_obj)
    broadcasted_ham = comm.bcast(serial_ham, root=0)
    hamiltonian_obj = pickle.loads(broadcasted_ham)
    # Extract neighbor indices
    rows = hamiltonian_obj.rows
    columns = hamiltonian_obj.columns

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
    nbc = 2
    bmax_mm = bmax[nbc-1:nb:nbc]
    bmin_mm = bmin[0:nb:nbc]

    map_diag_mm, map_upper_mm, map_lower_mm = change_format.map_block2sparse_alt(rows, columns, bmax_mm, bmin_mm)

    if rank == 0:
        # print size of data
        print(f"#Energy: {data_shape[1]} #nnz: {data_shape[0]}")


    # computation parameters----------------------------------------------------
    # set number of threads for the p2w step
    w_mkl_threads = 2
    w_worker_threads = 12
    # set number of threads for the h2g step
    gf_mkl_threads = 1
    gf_worker_threads = 8

    # physical parameter -----------

    # Fermi Level of Left Contact
    energy_fl = 1.0
    # Fermi Level of Right Contact
    energy_fr = energy_fl - Vappl
    # Temperature in Kelvin
    temp = 300
    # relative permittivity
    epsR = 5.0
    # DFT Conduction Band Minimum
    ECmin = 1.0133

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
    factor_w[ne-dnp-1:ne] = (np.cos(np.pi*np.linspace(0, 1, dnp+1)) + 1)/2
    #factor_w[np.where(np.invert(w_mask))[0]] = 0.0

    # create factor for the Green's Function
    factor_g = np.ones(ne)
    factor_g[ne-dnp-1:ne] = (np.cos(np.pi*np.linspace(0, 1, dnp+1)) + 1)/2
    factor_g[0:dnp+1] = (np.cos(np.pi*np.linspace(1, 0, dnp+1)) + 1)/2

    vh = construct_coulomb_matrix(hamiltonian_obj, epsR, eps0, e)
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

    # initialize Green's function------------------------------------------------
    gg_h2g = np.zeros((count[1,rank], no), dtype=np.complex128)
    gl_h2g = np.zeros((count[1,rank], no), dtype=np.complex128)
    gr_h2g = np.zeros((count[1,rank], no), dtype=np.complex128)

    # initialize Screened interaction-------------------------------------------
    wg_p2w = np.zeros((count[1,rank], no), dtype=np.complex128)
    wl_p2w = np.zeros((count[1,rank], no), dtype=np.complex128)
    wr_p2w = np.zeros((count[1,rank], no), dtype=np.complex128)

    # initialize memory factors for Self-Energy, Green's Function and Screened interaction
    mem_s = 0.75
    mem_g = 0.0
    mem_w = 0.75
    # max number of iterations

    max_iter = 200
    ECmin_vec = np.concatenate((np.array([ECmin]), np.zeros(max_iter)))
    EFL_vec = np.concatenate((np.array([energy_fl]), np.zeros(max_iter)))
    EFR_vec = np.concatenate((np.array([energy_fr]), np.zeros(max_iter)))

    if rank == 0:
        time_start = -time.perf_counter()
    # output folder
    folder = '/results/GNR_biased_sc/'
    for iter_num in range(max_iter):

        if rank == 0:
            iter_time = -time.perf_counter()

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

        
        # Adjusting Fermi Levels of both contacts to the current iteration band minima
        sr_ephn_h2g_vec = change_format.sparse2vecsparse_v2(np.zeros((count[1,rank], no), dtype=np.complex128), rows, columns, nao)

        # calculate the green's function at every rank------------------------------
        if args.pool:
            gr_diag, gr_upper, gl_diag, gl_upper, gg_diag, gg_upper = calc_GF_pool.calc_GF_pool_mpi(
                                                                hamiltonian_obj,
                                                                energy_loc,
                                                                sr_h2g_vec,
                                                                sl_h2g_vec,
                                                                sg_h2g_vec,
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
                                                                homogenize = False,
                                                                mkl_threads = gf_mkl_threads,
                                                                worker_num = gf_worker_threads
                                                            )
        else:
            gr_diag, gr_upper, gl_diag, gl_upper, gg_diag, gg_upper = calc_GF_pool.calc_GF_mpi(
                                                                hamiltonian_obj,   
                                                                energy_loc,
                                                                sr_h2g_vec,
                                                                sl_h2g_vec,
                                                                sg_h2g_vec,
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
                                                                gf_mkl_threads,
                                                                1
                                                            )
            
        ECmin_vec[iter_num+1] = get_band_edge_mpi(ECmin_vec[iter_num], energy, hamiltonian_obj.Overlap['H_4'], hamiltonian_obj.Hamiltonian['H_4'], sr_h2g_vec, sr_ephn_h2g_vec, rows, columns, bmin, bmax, comm, rank, size, count, disp, side = 'left')

        energy_fl = ECmin_vec[iter_num+1] + dEfL_EC
        energy_fr = ECmin_vec[iter_num+1] + dEfR_EC

        EFL_vec[iter_num+1] = energy_fl
        EFR_vec[iter_num+1] = energy_fr
        # lower diagonal blocks from physics identity
        gg_lower = -gg_upper.conjugate().transpose((0,1,3,2))
        gl_lower = -gl_upper.conjugate().transpose((0,1,3,2))
        gr_lower = gr_upper.transpose((0,1,3,2))
        if iter_num == 0:
            gg_h2g = change_format.block2sparse_energy_alt(map_diag, map_upper,
                                                            map_lower, gg_diag, gg_upper,
                                                            gg_lower, no, count[1,rank],
                                                            energy_contiguous=False)
            gl_h2g = change_format.block2sparse_energy_alt(map_diag, map_upper,
                                                            map_lower, gl_diag, gl_upper,
                                                            gl_lower, no, count[1,rank],
                                                            energy_contiguous=False)
            gr_h2g = change_format.block2sparse_energy_alt(map_diag, map_upper,
                                                            map_lower, gr_diag, gr_upper,
                                                            gr_lower, no, count[1,rank],
                                                            energy_contiguous=False)
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
            gr_h2g = (1.0 - mem_g) * change_format.block2sparse_energy_alt(map_diag, map_upper,
                                                            map_lower, gr_diag, gr_upper,
                                                            gr_lower, no, count[1,rank],
                                                            energy_contiguous=False) + mem_g * gr_h2g
        # calculate the transposed
        gl_transposed_h2g = np.copy(gl_h2g[:,ij2ji], order="C")
        # create local buffers
        gg_g2p = np.empty((count[0, rank], data_shape[1]),
                        dtype=np.complex128, order="C")
        gl_g2p = np.empty((count[0, rank], data_shape[1]),
                        dtype=np.complex128, order="C")
        gr_g2p = np.empty((count[0, rank], data_shape[1]),
                        dtype=np.complex128, order="C")
        gl_transposed_g2p = np.empty((count[0, rank], data_shape[1]),
                        dtype=np.complex128, order="C")

        # use of all to all w since not divisible
        alltoall_g2p(gg_h2g, gg_g2p, transpose_net=args.net_transpose)
        alltoall_g2p(gl_h2g, gl_g2p, transpose_net=args.net_transpose)
        alltoall_g2p(gr_h2g, gr_g2p, transpose_net=args.net_transpose)
        alltoall_g2p(gl_transposed_h2g, gl_transposed_g2p, transpose_net=args.net_transpose)

        # calculate the polarization at every rank----------------------------------
        if args.type in ("gpu"):
            pg_g2p, pl_g2p, pr_g2p = g2p_gpu.g2p_fft_mpi_gpu(
                                                pre_factor,
                                                gg_g2p,
                                                gl_g2p,
                                                gr_g2p,
                                                gl_transposed_g2p)
        elif args.type in ("cpu"):
            pg_g2p, pl_g2p, pr_g2p = g2p_cpu.g2p_fft_mpi_cpu_inlined(
                                                pre_factor,
                                                gg_g2p,
                                                gl_g2p,
                                                gr_g2p,
                                                gl_transposed_g2p)
        else:
            raise ValueError("Argument error, input type not possible")



        # distribute polarization function according to p2w step--------------------

        # create local buffers
        pg_p2w = np.empty((count[1, rank], data_shape[0]),
                        dtype=np.complex128, order="C")
        pl_p2w = np.empty((count[1, rank], data_shape[0]),
                        dtype=np.complex128, order="C")
        pr_p2w = np.empty((count[1, rank], data_shape[0]),
                        dtype=np.complex128, order="C")



        # use of all to all w since not divisible
        alltoall_p2g(pg_g2p, pg_p2w, transpose_net=args.net_transpose)
        alltoall_p2g(pl_g2p, pl_p2w, transpose_net=args.net_transpose)
        alltoall_p2g(pr_g2p, pr_p2w, transpose_net=args.net_transpose)

        if rank == 0:
            w_time = -time.perf_counter()

        if args.bsr:

            # transform from 2D format to list/vector of sparse arrays format-----------
            pg_p2w_vec = change_format.sparse2vecbsr_v2(pg_p2w, rows, columns, nao, w_bsize)
            pl_p2w_vec = change_format.sparse2vecbsr_v2(pl_p2w, rows, columns, nao, w_bsize)
            pr_p2w_vec = change_format.sparse2vecbsr_v2(pr_p2w, rows, columns, nao, w_bsize)

            # calculate the screened interaction on every rank--------------------------
            if args.pool:
                wg_diag_bsr, wg_upper_bsr, wl_diag_bsr, wl_upper_bsr, wr_diag_bsr, wr_upper_bsr, nb_mm, lb_max_mm = p2w_cpu.p2w_pool_mpi_cpu(
                                                                                                    hamiltonian_obj, energy_loc,
                                                                                                    pg_p2w_vec, pl_p2w_vec,
                                                                                                    pr_p2w_vec, vh, dosw[disp[1, rank]:disp[1, rank] + count[1, rank]],
                                                                                                    nEw[disp[1, rank]:disp[1, rank] + count[1, rank]], nPw[disp[1, rank]:disp[1, rank] + count[1, rank]],
                                                                                                    Idx_e_loc,   
                                                                                                    factor_w_loc,
                                                                                                    comm,
                                                                                                    rank,
                                                                                                    size,
                                                                                                    w_mkl_threads,
                                                                                                    w_worker_threads)
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
                                                                                                    w_mkl_threads
                                                                                                    )
        
        if not args.bsr or (args.bsr and args.validate):

            # transform from 2D format to list/vector of sparse arrays format-----------
            pg_p2w_vec = change_format.sparse2vecsparse_v2(pg_p2w, rows, columns, nao)
            pl_p2w_vec = change_format.sparse2vecsparse_v2(pl_p2w, rows, columns, nao)
            pr_p2w_vec = change_format.sparse2vecsparse_v2(pr_p2w, rows, columns, nao)


            # calculate the screened interaction on every rank--------------------------
            if args.pool:
                wg_diag, wg_upper, wl_diag, wl_upper, wr_diag, wr_upper, nb_mm, lb_max_mm = p2w_cpu.p2w_pool_mpi_cpu(
                                                                                                    hamiltonian_obj, energy_loc,
                                                                                                    pg_p2w_vec, pl_p2w_vec,
                                                                                                    pr_p2w_vec, vh, dosw[disp[1, rank]:disp[1, rank] + count[1, rank]],
                                                                                                    nEw[disp[1, rank]:disp[1, rank] + count[1, rank]], nPw[disp[1, rank]:disp[1, rank] + count[1, rank]],
                                                                                                    Idx_e_loc,   
                                                                                                    factor_w_loc,
                                                                                                    comm,
                                                                                                    rank,
                                                                                                    size,
                                                                                                    w_mkl_threads,
                                                                                                    w_worker_threads)
            else:
                wg_diag, wg_upper, wl_diag, wl_upper, wr_diag, wr_upper, nb_mm, lb_max_mm = p2w_cpu.p2w_mpi_cpu(
                                                                                                    hamiltonian_obj, energy_loc,
                                                                                                    pg_p2w_vec, pl_p2w_vec,
                                                                                                    pr_p2w_vec, vh, dosw[disp[1, rank]:disp[1, rank] + count[1, rank]],  
                                                                                                    nEw[disp[1, rank]:disp[1, rank] + count[1, rank]], nPw[disp[1, rank]:disp[1, rank] + count[1, rank]],    
                                                                                                    factor_w_loc,
                                                                                                    comm,
                                                                                                    rank,
                                                                                                    size,
                                                                                                    w_mkl_threads
                                                                                                    )
            
            if args.validate:
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
            
        if rank == 0:
            w_time += time.perf_counter()
            print(f"w time: {w_time:.2f} s")

        # transform from block format to 2D format-----------------------------------
        # lower diagonal blocks from physics identity
        wg_lower = -wg_upper.conjugate().transpose((0,1,3,2))
        wl_lower = -wl_upper.conjugate().transpose((0,1,3,2))
        wr_lower = wr_upper.transpose((0,1,3,2))
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
        wg_p2w = (1.0 - mem_w) * change_format.block2sparse_energy_alt(map_diag_mm, map_upper_mm,
                                                        map_lower_mm, wg_diag, wg_upper,
                                                        wg_lower, no, count[1,rank],
                                                        energy_contiguous=False) + mem_w * wg_p2w
        wl_p2w = (1.0 - mem_w) * change_format.block2sparse_energy_alt(map_diag_mm, map_upper_mm,
                                                        map_lower_mm, wl_diag, wl_upper,
                                                        wl_lower, no, count[1,rank],
                                                        energy_contiguous=False) + mem_w * wl_p2w
        wr_p2w = (1.0 - mem_w) * change_format.block2sparse_energy_alt(map_diag_mm, map_upper_mm,
                                                        map_lower_mm, wr_diag, wr_upper,
                                                        wr_lower, no, count[1,rank],
                                                        energy_contiguous=False) + mem_w * wr_p2w


        # distribute screened interaction according to gw2s step--------------------

        # calculate the transposed
        wg_transposed_p2w = np.copy(wg_p2w[:,ij2ji], order="C")
        wl_transposed_p2w = np.copy(wl_p2w[:,ij2ji], order="C")

        # create local buffers
        wg_gw2s = np.empty((count[0, rank], data_shape[1]),
                        dtype=np.complex128, order="C")
        wl_gw2s = np.empty((count[0, rank], data_shape[1]),
                        dtype=np.complex128, order="C")
        wr_gw2s = np.empty((count[0, rank], data_shape[1]),
                        dtype=np.complex128, order="C")
        wg_transposed_gw2s = np.empty((count[0, rank], data_shape[1]),
                        dtype=np.complex128, order="C")
        wl_transposed_gw2s = np.empty((count[0, rank], data_shape[1]),
                        dtype=np.complex128, order="C")
        

        # use of all to all w since not divisible
        alltoall_g2p(wg_p2w, wg_gw2s, transpose_net=args.net_transpose)
        alltoall_g2p(wl_p2w, wl_gw2s, transpose_net=args.net_transpose)
        alltoall_g2p(wr_p2w, wr_gw2s, transpose_net=args.net_transpose)
        alltoall_g2p(wg_transposed_p2w, wg_transposed_gw2s, transpose_net=args.net_transpose)
        alltoall_g2p(wl_transposed_p2w, wl_transposed_gw2s, transpose_net=args.net_transpose)

    # tod optimize and not load two time green's function to gpu and do twice the fft
        if args.type in ("gpu"):
            sg_gw2s, sl_gw2s, sr_gw2s = gw2s_gpu.gw2s_fft_mpi_gpu_3part_sr(
                                                                -pre_factor/2,
                                                                gg_g2p,
                                                                gl_g2p,
                                                                gr_g2p,
                                                                wg_gw2s,
                                                                wl_gw2s,
                                                                wr_gw2s,
                                                                wg_transposed_gw2s,
                                                                wl_transposed_gw2s
                                                                )
        elif args.type in ("cpu"):
            sg_gw2s, sl_gw2s, sr_gw2s = gw2s_cpu.gw2s_fft_mpi_cpu_3part_sr(
                                                                -pre_factor/2,
                                                                gg_g2p,
                                                                gl_g2p,
                                                                gr_g2p,
                                                                wg_gw2s,
                                                                wl_gw2s,
                                                                wr_gw2s,
                                                                wg_transposed_gw2s,
                                                                wl_transposed_gw2s
                                                                )
        else:
            raise ValueError("Argument error, input type not possible")

        # distribute screened interaction according to h2g step---------------------
        # create local buffers
        sg_h2g_buf = np.empty((count[1, rank], data_shape[0]),
                        dtype=np.complex128, order="C")
        sl_h2g_buf = np.empty((count[1, rank], data_shape[0]),
                        dtype=np.complex128, order="C")
        sr_h2g_buf = np.empty((count[1, rank], data_shape[0]),
                        dtype=np.complex128, order="C")

        # use of all to all w since not divisible
        alltoall_p2g(sg_gw2s, sg_h2g_buf, transpose_net=args.net_transpose)
        alltoall_p2g(sl_gw2s, sl_h2g_buf, transpose_net=args.net_transpose)
        alltoall_p2g(sr_gw2s, sr_h2g_buf, transpose_net=args.net_transpose)

        if iter_num == 0:
            sg_h2g = (1.0 - mem_s) * sg_h2g_buf + mem_s * sg_h2g
            sl_h2g = (1.0 - mem_s) * sl_h2g_buf + mem_s * sl_h2g
            sr_h2g = (1.0 - mem_s) * sr_h2g_buf + mem_s * sr_h2g
        else:
            # add new contribution to the Self-Energy
            sg_h2g = (1.0 - mem_s) * sg_h2g_buf + mem_s * sg_h2g
            sl_h2g = (1.0 - mem_s) * sl_h2g_buf + mem_s * sl_h2g
            sr_h2g = (1.0 - mem_s) * sr_h2g_buf + mem_s * sr_h2g

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
            iter_time += time.perf_counter()
            print(f"iter time: {iter_time:.2f} s")

        if rank == 0:
            np.savetxt(parent_path + folder + 'E.dat', energy)
            np.savetxt(parent_path + folder + 'DOS_' + str(iter_num) + '.dat', dos.view(float))
            np.savetxt(parent_path + folder + 'IDE_' + str(iter_num) + '.dat', ide.view(float))
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