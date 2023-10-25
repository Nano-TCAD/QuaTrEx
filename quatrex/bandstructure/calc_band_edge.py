# Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.

import numpy as np
from scipy.sparse import csc_array

import mpi4py

mpi4py.rc.initialize = False  # do not initialize MPI automatically
mpi4py.rc.finalize = False  # do not finalize MPI automatically
from mpi4py import MPI

from quatrex.bandstructure.calc_contact_bs import calc_bandstructure, calc_bandstructure_interpol, calc_bandstructure_mpi_interpol



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

def get_band_edge_interpol(ECmin_DFT,
                           E,
                           S,
                           H,
                           SigmaR_GW,
                           SigmaL_GW,
                           SigmaG_GW,
                           SigmaR_PHN,
                           Bmin,
                           Bmax,
                           side='left'):
    # First step: get a first estimate of the CB edge
    min_ind = np.argmin(np.abs(E - ECmin_DFT))

    if(E[min_ind] > ECmin_DFT and min_ind > 0):
        min_ind -= 1

    Ek = calc_bandstructure_interpol(E, S, H, ECmin_DFT, SigmaR_GW, SigmaL_GW, SigmaG_GW, SigmaR_PHN, min_ind, Bmin, Bmax, side)
    ind_ek_plus = np.argmin(np.abs(Ek - ECmin_DFT))
    ECmin_int = Ek[ind_ek_plus]

    # Second step: refine the position of the CB edge
    min_ind = np.argmin(np.abs(E - ECmin_int))

    if (E[min_ind] > ECmin_int and min_ind > 0):
        min_ind -= 1

    Ek = calc_bandstructure_interpol(E, S, H, ECmin_int, SigmaR_GW, SigmaL_GW, SigmaG_GW, SigmaR_PHN, min_ind, Bmin, Bmax, side)
    ind_ek = np.argmin(np.abs(Ek - ECmin_int))
    ECmin = Ek[ind_ek]

    return ECmin



def get_band_edge_mpi_interpol(ECmin_DFT,
                      E,
                      S,
                      H,
                      SigmaR_GW,
                      SigmaL_GW,
                      SigmaG_GW,
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
                      side='left'):
    nao = Bmax[-1] + 1
    SigmaR_GW_vec = np.ndarray((2, ), dtype=object)
    SigmaL_GW_vec = np.ndarray((2, ), dtype=object)
    SigmaG_GW_vec = np.ndarray((2, ), dtype=object)
    SigmaR_PHN_vec = np.ndarray((2, ), dtype=object)

    # First step: get a first estimate of the CB edge
    (min_ind, send_rank_1, send_rank_2) = get_send_ranks_interpol(ECmin_DFT, E, comm, rank, size, count, disp)
    send_sigmas_GWRGL_PHNR_to_root(SigmaR_GW_vec, SigmaL_GW_vec, SigmaG_GW_vec, SigmaR_PHN_vec,
                                   send_rank_1, send_rank_2, min_ind, rank, comm, disp,  SigmaR_GW, SigmaL_GW, SigmaG_GW, SigmaR_PHN, rows, columns, nao)
    if rank == 0:
        Ek = calc_bandstructure_mpi_interpol(E, S, H, ECmin_DFT, SigmaR_GW_vec, SigmaL_GW_vec, SigmaG_GW_vec, SigmaR_PHN_vec, min_ind, Bmin, Bmax, side)
        ind_ek_plus = np.argmin(np.abs(Ek - ECmin_DFT))
        ECmin_int = Ek[ind_ek_plus]
        # broadcasting the band edge (this is actually not necessary, but it is done for consistency), non-root nodes will not use it in get_send_ranks_interpol 
        comm.Bcast([ECmin_int, MPI.DOUBLE], root=0)
    else:   
        ECmin_int = np.empty(1, dtype=np.float64)
        comm.Bcast([ECmin_int, MPI.DOUBLE], root=0)
        ECmin_int = ECmin_int[0]

    # Second step: refine the position of the CB edge
    (min_ind, send_rank_1, send_rank_2) = get_send_ranks_interpol(ECmin_int, E, comm, rank, size, count, disp)
    send_sigmas_GWRGL_PHNR_to_root(SigmaR_GW_vec, SigmaL_GW_vec, SigmaG_GW_vec, SigmaR_PHN_vec,
                                   send_rank_1, send_rank_2, min_ind, rank, comm, disp, SigmaR_GW, SigmaL_GW, SigmaG_GW, SigmaR_PHN, rows, columns, nao)
    if rank == 0:
        Ek = calc_bandstructure_mpi_interpol(E, S, H, ECmin_int, SigmaR_GW_vec, SigmaL_GW_vec, SigmaG_GW_vec, SigmaR_PHN_vec, min_ind, Bmin, Bmax, side)
        ind_ek_plus = np.argmin(np.abs(Ek - ECmin_int))
        ECmin = Ek[ind_ek_plus]
        # broadcasting the band edge
        comm.Bcast([ECmin, MPI.DOUBLE], root=0)
        print("send rank 1 was: " + str(send_rank_1) + " send rank 2 was: " + str(send_rank_2) + " new band edge: " + str(ECmin))
    else:   
        ECmin = np.empty(1, dtype=np.float64)
        comm.Bcast([ECmin, MPI.DOUBLE], root=0)
        ECmin = ECmin[0]
    # returning the band edge
    return ECmin




def get_send_ranks_interpol(ECmin_DFT, E, comm,
                      rank,
                      size,
                      count,
                      disp):
    """
    This function returns the ranks that need to send their SigmaR_GW and SigmaR_PHN to rank 0
    First, the minimum index is broadcasted to all ranks
    Then, the rank that has the minimum index is determined
    Then, the rank that has the minimum index + 1 is determined

    :input: 
    ECmin_DFT: DFT conduction band minimum or target energy on root node (input on non-root nodes is irrelevant!!): float
    E: energy vector (global): npt.NDArray[np.float64]
    comm: MPI communicator: MPI communicator
    rank: rank of the current process: int
    size: number of processes: int
    count: number of elements per rank: npt.NDArray[np.int32]
    disp: displacement of elements per rank: npt.NDArray[np.int32]

    :return: send_ranks: list of ranks that need to send their SigmaR_GW and SigmaR_PHN to rank 0
    """
    ## Broadcasting minimum index
    if rank == 0:
        min_ind = np.argmin(np.abs(E - ECmin_DFT))
        if (E[min_ind] > ECmin_DFT and min_ind > 0):
            min_ind -= 1
        comm.Bcast([min_ind, MPI.INT], root=0)
    else:
        min_ind = np.empty(1, dtype=np.int64)
        comm.Bcast([min_ind, MPI.INT], root=0)
        min_ind = min_ind[0]

    ## Checking which rank has the minimum index
    send_rank_1 = 0
    send_rank_2 = 0

    for check_rank in range(size):
        if min_ind in range(disp[1, check_rank], disp[1, check_rank] + count[1, check_rank]):
            send_rank_1 = check_rank
            break

    for check_rank in range(size):
        if min_ind + 1 in range(disp[1, check_rank], disp[1, check_rank] + count[1, check_rank]):
            send_rank_2 = check_rank
            break
    return (min_ind, send_rank_1, send_rank_2)

def send_sigmas_GWRGL_PHNR_to_root(SigmaR_GW_vec, SigmaL_GW_vec, SigmaG_GW_vec, SigmaR_PHN_vec,
                                   send_rank_1, send_rank_2, min_ind, rank, comm, disp, SigmaR_GW, SigmaL_GW, SigmaG_GW, SigmaR_PHN, rows, columns, nao):
    """
    This function sends the SigmaR_GW, SigmaL_GW and SigmaG_GW and SigmaR_PHN from the ranks that have the minimum index and the minimum index + 1 to rank 0
    It will be stored in the _vec datastructures.
    input: 
    SigmaR_GW_vec: SigmaR_GW vector of size 2 of the target self-energies in sparse format: np.ndarray[np.object]
    SigmaL_GW_vec: SigmaL_GW vector of size 2 of the target self-energies in sparse format: np.ndarray[np.object]
    SigmaG_GW_vec: SigmaG_GW vector of size 2 of the target self-energies in sparse format: np.ndarray[np.object]
    SigmaR_PHN_vec: SigmaR_PHN vector of size 2 of the target self-energies in sparse format: np.ndarray[np.object]
    send_rank_1: rank that has the minimum index: int
    send_rank_2: rank that has the minimum index + 1: int
    min_ind: global minimum index: int
    rank: rank of the current process: int
    comm: MPI communicator: MPI communicator
    SigmaR_GW: local SigmaR_GW vector in sparse format: np.ndarray[np.object]
    SigmaL_GW: local SigmaL_GW vector in sparse format: np.ndarray[np.object]
    SigmaG_GW: local SigmaG_GW vector in sparse format: np.ndarray[np.object]
    SigmaR_PHN: local SigmaR_PHN vector in sparse format: np.ndarray[np.object]
    """
    if rank == 0:
        if send_rank_1 == 0:
            SigmaR_GW_vec[0] = SigmaR_GW[min_ind - disp[1, rank]]
            SigmaL_GW_vec[0] = SigmaL_GW[min_ind - disp[1, rank]]   
            SigmaG_GW_vec[0] = SigmaG_GW[min_ind - disp[1, rank]]
            SigmaR_PHN_vec[0] = SigmaR_PHN[min_ind - disp[1, rank]]

        else:
            sr_gw_buf = np.empty(rows.shape[0], dtype=np.complex128)
            sl_gw_buf = np.empty(rows.shape[0], dtype=np.complex128)
            sg_gw_buf = np.empty(rows.shape[0], dtype=np.complex128)
            sr_phn_buf = np.empty(rows.shape[0], dtype=np.complex128)
            comm.Recv([sr_gw_buf, MPI.DOUBLE_COMPLEX], source=send_rank_1, tag=0)
            comm.Recv([sl_gw_buf, MPI.DOUBLE_COMPLEX], source=send_rank_1, tag=1)
            comm.Recv([sg_gw_buf, MPI.DOUBLE_COMPLEX], source=send_rank_1, tag=2)
            comm.Recv([sr_phn_buf, MPI.DOUBLE_COMPLEX], source=send_rank_1, tag=3)

            SigmaR_GW_vec[0] = csc_array((sr_gw_buf, (rows, columns)), shape=(nao, nao))
            SigmaL_GW_vec[0] = csc_array((sl_gw_buf, (rows, columns)), shape=(nao, nao))
            SigmaG_GW_vec[0] = csc_array((sg_gw_buf, (rows, columns)), shape=(nao, nao))
            SigmaR_PHN_vec[0] = csc_array((sr_phn_buf, (rows, columns)), shape=(nao, nao))

    else:
        if send_rank_1 == rank:
            sr_gw_buf = SigmaR_GW[min_ind - disp[1, rank]].toarray()[rows, columns]
            sl_gw_buf = SigmaL_GW[min_ind - disp[1, rank]].toarray()[rows, columns]
            sg_gw_buf = SigmaG_GW[min_ind - disp[1, rank]].toarray()[rows, columns]
            sr_phn_buf = SigmaR_PHN[min_ind - disp[1, rank]].toarray()[rows, columns]
            comm.Send([sr_gw_buf, MPI.DOUBLE_COMPLEX], dest=0, tag=0)
            comm.Send([sl_gw_buf, MPI.DOUBLE_COMPLEX], dest=0, tag=1)
            comm.Send([sg_gw_buf, MPI.DOUBLE_COMPLEX], dest=0, tag=2)
            comm.Send([sr_phn_buf, MPI.DOUBLE_COMPLEX], dest=0, tag=3)

    
    if rank == 0:
        if send_rank_2 == 0:
            SigmaR_GW_vec[1] = SigmaR_GW[min_ind + 1 - disp[1, rank]]
            SigmaL_GW_vec[1] = SigmaL_GW[min_ind + 1 - disp[1, rank]]
            SigmaG_GW_vec[1] = SigmaG_GW[min_ind + 1 - disp[1, rank]]
            SigmaR_PHN_vec[1] = SigmaR_PHN[min_ind + 1 - disp[1, rank]]
        else:
            sr_gw_buf = np.empty(rows.shape[0], dtype=np.complex128)
            sl_gw_buf = np.empty(rows.shape[0], dtype=np.complex128)
            sg_gw_buf = np.empty(rows.shape[0], dtype=np.complex128)
            sr_phn_buf = np.empty(rows.shape[0], dtype=np.complex128)
            comm.Recv([sr_gw_buf, MPI.DOUBLE_COMPLEX], source=send_rank_2, tag=0)
            comm.Recv([sl_gw_buf, MPI.DOUBLE_COMPLEX], source=send_rank_2, tag=1)
            comm.Recv([sg_gw_buf, MPI.DOUBLE_COMPLEX], source=send_rank_2, tag=2)
            comm.Recv([sr_phn_buf, MPI.DOUBLE_COMPLEX], source=send_rank_2, tag=3)

            SigmaR_GW_vec[1] = csc_array((sr_gw_buf, (rows, columns)), shape=(nao, nao))
            SigmaL_GW_vec[1] = csc_array((sl_gw_buf, (rows, columns)), shape=(nao, nao))
            SigmaG_GW_vec[1] = csc_array((sg_gw_buf, (rows, columns)), shape=(nao, nao))
            SigmaR_PHN_vec[1] = csc_array((sr_phn_buf, (rows, columns)), shape=(nao, nao))

    else:
        if send_rank_2 == rank:
            sr_gw_buf = SigmaR_GW[min_ind + 1 - disp[1, rank]].toarray()[rows, columns]
            sl_gw_buf = SigmaL_GW[min_ind + 1 - disp[1, rank]].toarray()[rows, columns]
            sg_gw_buf = SigmaG_GW[min_ind + 1 - disp[1, rank]].toarray()[rows, columns]
            sr_phn_buf = SigmaR_PHN[min_ind + 1 - disp[1, rank]].toarray()[rows, columns]
            comm.Send([sr_gw_buf, MPI.DOUBLE_COMPLEX], dest=0, tag=0)
            comm.Send([sl_gw_buf, MPI.DOUBLE_COMPLEX], dest=0, tag=1)
            comm.Send([sg_gw_buf, MPI.DOUBLE_COMPLEX], dest=0, tag=2)
            comm.Send([sr_phn_buf, MPI.DOUBLE_COMPLEX], dest=0, tag=3)
