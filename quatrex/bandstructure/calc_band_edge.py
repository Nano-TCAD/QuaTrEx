# Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.

import numpy as np
from scipy.sparse import csc_array

import mpi4py

mpi4py.rc.initialize = False  # do not initialize MPI automatically
mpi4py.rc.finalize = False  # do not finalize MPI automatically
from mpi4py import MPI
import os
import pickle

from quatrex.OMEN_structure_matrices import OMENHamClass
from quatrex.bandstructure.calc_contact_bs import calc_bandstructure, calc_bandstructure_mpi
from quatrex.utils import change_format


def get_band_edge(ECmin_DFT, E, S, H, SigmaR_GW, SigmaR_PHN, Bmin, Bmax, side='left'):
    # First step: get a first estimate of the CB edge
    min_ind = np.argmin(np.abs(E - ECmin_DFT))
    Ek = calc_bandstructure(S, H, SigmaR_GW, SigmaR_PHN, min_ind, Bmin, Bmax, side)
    ind_ek_plus = np.argmin(np.abs(Ek - ECmin_DFT))
    ECmin_int = Ek[ind_ek_plus]

    # Second step: refine the position of the CB edge
    min_ind = np.argmin(np.abs(E - ECmin_int))
    Ek = calc_bandstructure(S, H, SigmaR_GW, SigmaR_PHN, min_ind, Bmin, Bmax, side)
    ind_ek = np.argmin(np.abs(Ek - ECmin_int))
    ECmin = Ek[ind_ek]

    return ECmin


def get_band_edge_mpi(ECmin_DFT,
                      E,
                      S,
                      H,
                      SigmaR_GW,
                      SigmaR_PHN,
                      rows,
                      columns,
                      Bmin,
                      Bmax,
                      comm,
                      rank,
                      size,
                      count,
                      disp,
                      side='left',
                      mode='regular'):
    nao = Bmax[-1] + 1

    # First step: get a first estimate of the CB edge
    ## Broadcasting minimum index
    if rank == 0:
        min_ind = np.argmin(np.abs(E - ECmin_DFT))
        comm.Bcast([min_ind, MPI.INT], root=0)
    else:
        min_ind = np.empty(1, dtype=np.int64)
        comm.Bcast([min_ind, MPI.INT], root=0)
        min_ind = min_ind[0]

    ## Checking which rank has the minimum index
    send_rank = 0

    for check_rank in range(size):
        if min_ind in range(disp[1, check_rank], disp[1, check_rank] + count[1, check_rank]):
            send_rank = check_rank
            break

    ## Sending the SigmaR_GW and SigmaR_PHN from the rank with the minimum index to rank 0
    if rank == 0:
        if send_rank == 0:
            Ek = calc_bandstructure_mpi(S, H, SigmaR_GW[min_ind - disp[1, rank]], SigmaR_PHN[min_ind - disp[1, rank]],
                                        Bmin, Bmax, side)
            ind_ek_plus = np.argmin(np.abs(Ek - ECmin_DFT))
            ECmin_int = Ek[ind_ek_plus]

        else:
            sr_gw_buf = np.empty(rows.shape[0], dtype=np.complex128)
            sr_phn_buf = np.empty(rows.shape[0], dtype=np.complex128)
            comm.Recv([sr_gw_buf, MPI.DOUBLE_COMPLEX], source=send_rank, tag=0)
            comm.Recv([sr_phn_buf, MPI.DOUBLE_COMPLEX], source=send_rank, tag=1)

            SigmaR_recv = csc_array((sr_gw_buf, (rows, columns)), shape=(nao, nao))
            SigmaR_PHN_recv = csc_array((sr_phn_buf, (rows, columns)), shape=(nao, nao))
            Ek = calc_bandstructure_mpi(S, H, SigmaR_recv, SigmaR_PHN_recv, Bmin, Bmax, side)
            ind_ek_plus = np.argmin(np.abs(Ek - ECmin_DFT))
            ECmin_int = Ek[ind_ek_plus]

    else:
        if send_rank == rank:
            sr_gw_buf = SigmaR_GW[min_ind - disp[1, rank]].toarray()[rows, columns]
            sr_phn_buf = SigmaR_PHN[min_ind - disp[1, rank]].toarray()[rows, columns]
            comm.Send([sr_gw_buf, MPI.DOUBLE_COMPLEX], dest=0, tag=0)
            comm.Send([sr_phn_buf, MPI.DOUBLE_COMPLEX], dest=0, tag=1)

    # Second step: refine the position of the CB edge
    ## Broadcasting minimum index
    if rank == 0:
        min_ind = np.argmin(np.abs(E - ECmin_int))
        comm.Bcast(min_ind, root=0)
    else:
        min_ind = np.empty(1, dtype=np.int64)
        comm.Bcast(min_ind, root=0)
        min_ind = min_ind[0]

    ## Checking which rank has the minimum index
    send_rank = 0

    for check_rank in range(size):
        if min_ind in range(disp[1, check_rank], disp[1, check_rank] + count[1, check_rank]):
            send_rank = check_rank
            break

    ## Sending the SigmaR_GW and SigmaR_PHN from the rank with the minimum index to rank 0
    if rank == 0:
        if send_rank == 0:
            Ek = calc_bandstructure_mpi(S, H, SigmaR_GW[min_ind - disp[1, rank]], SigmaR_PHN[min_ind - disp[1, rank]],
                                        Bmin, Bmax, side)
            ind_ek = np.argmin(np.abs(Ek - ECmin_int))
            ECmin = Ek[ind_ek]

        else:
            sr_gw_buf = np.empty(rows.shape[0], dtype=np.complex128)
            sr_phn_buf = np.empty(rows.shape[0], dtype=np.complex128)
            comm.Recv([sr_gw_buf, MPI.DOUBLE_COMPLEX], source=send_rank, tag=0)
            comm.Recv([sr_phn_buf, MPI.DOUBLE_COMPLEX], source=send_rank, tag=1)

            SigmaR_recv = csc_array((sr_gw_buf, (rows, columns)), shape=(nao, nao))
            SigmaR_PHN_recv = csc_array((sr_phn_buf, (rows, columns)), shape=(nao, nao))
            Ek = calc_bandstructure_mpi(S, H, SigmaR_recv, SigmaR_PHN_recv, Bmin, Bmax, side)
            ind_ek = np.argmin(np.abs(Ek - ECmin_DFT))
            ECmin = Ek[ind_ek]

    else:
        if send_rank == rank:
            sr_gw_buf = SigmaR_GW[min_ind - disp[1, rank]].toarray()[rows, columns]
            sr_phn_buf = SigmaR_PHN[min_ind - disp[1, rank]].toarray()[rows, columns]
            comm.Send([sr_gw_buf, MPI.DOUBLE_COMPLEX], dest=0, tag=0)
            comm.Send([sr_phn_buf, MPI.DOUBLE_COMPLEX], dest=0, tag=1)

    # Broadcasting the band edge
    if rank == 0:
        comm.Bcast(ECmin, root=0)
        print("new band edge: " + str(ECmin))
    else:
        ECmin = np.empty(1, dtype=np.float64)
        comm.Bcast(ECmin, root=0)
    # returning the band edge
    return ECmin


if __name__ == '__main__':
    MPI.Init_thread(required=MPI.THREAD_MULTIPLE)
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    name = MPI.Get_processor_name()

    # path to solution
    solution_path = "/usr/scratch/mont-fort17/dleonard/CNT/"
    hamiltonian_path = os.path.join(solution_path, "CNT_newwannier")

    # one orbital on C atoms, two same types
    no_orb = np.array([1, 1])
    energy = np.linspace(-17.5, 7.5, 251, endpoint=True, dtype=float)  # Energy Vector
    hamiltonian_obj = OMENHamClass.Hamiltonian(hamiltonian_path, no_orb, rank)
    serial_ham = pickle.dumps(hamiltonian_obj)
    broadcasted_ham = comm.bcast(serial_ham, root=0)
    hamiltonian_obj = pickle.loads(broadcasted_ham)
    # Extract neighbor indices
    rows = hamiltonian_obj.rows
    columns = hamiltonian_obj.columns

    # hamiltonian object has 1-based indexing
    bmax = hamiltonian_obj.Bmax - 1
    bmin = hamiltonian_obj.Bmin - 1

    # DFT Conduction Band Minimum
    ECmin = -3.55

    nao: np.int64 = np.max(bmax) + 1
    no: np.int32 = np.int32(columns.shape[0])

    # calculation of data distribution per rank---------------------------------
    data_shape = np.array([rows.shape[0], energy.shape[0]], dtype=np.int32)

    # split nnz/energy per rank
    data_per_rank = data_shape // size

    # create array with energy size distribution
    count = np.repeat(data_per_rank.reshape(-1, 1), size, axis=1)
    count[:, size - 1] += data_shape % size

    # displacements in nnz/energy
    disp = data_per_rank.reshape(-1, 1) * np.arange(size)

    # slice energy vector
    energy_loc = energy[disp[1, rank]:disp[1, rank] + count[1, rank]]

    # initialize self energy
    sr_h2g_full = np.loadtxt(solution_path + 'python_testfiles/calc_band_edge/sr_gw.dat').view(complex)

    #sr_h2g = np.zeros((count[1,rank], no), dtype=np.complex128)
    sr_h2g = sr_h2g_full[disp[1, rank]:disp[1, rank] + count[1, rank], :]
    # transform from 2D format to list/vector of sparse arrays format-----------
    sr_h2g_vec = change_format.sparse2vecsparse_v2(sr_h2g, rows, columns, nao)

    sr_ephn_h2g_vec = change_format.sparse2vecsparse_v2(np.zeros((count[1, rank], no), dtype=np.complex128), rows,
                                                        columns, nao)

    #ECmin_single = get_band_edge(-3.52400739, energy, hamiltonian_obj.Overlap['H_4'], hamiltonian_obj.Hamiltonian['H_4'], sr_h2g_vec, sr_ephn_h2g_vec, bmin, bmax, side = 'left')
    ECmin_MPI = get_band_edge_mpi(-3.52400739,
                                  energy,
                                  hamiltonian_obj.Overlap['H_4'],
                                  hamiltonian_obj.Hamiltonian['H_4'],
                                  sr_h2g_vec,
                                  sr_ephn_h2g_vec,
                                  rows,
                                  columns,
                                  bmin,
                                  bmax,
                                  comm,
                                  rank,
                                  size,
                                  count,
                                  disp,
                                  side='left')

    if rank == 0:
        #print(ECmin_single)
        print(ECmin_MPI)
