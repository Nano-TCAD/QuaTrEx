# Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.

from quatrex.utils import change_format
from quatrex.bandstructure.calc_contact_bs import calc_bandstructure, calc_bandstructure_interpol, calc_bandstructure_mpi, calc_bandstructure_mpi_interpol
from quatrex.OMEN_structure_matrices import OMENHamClass
import numpy.typing as npt
import typing
import pickle
import os
from mpi4py import MPI
import numpy as np
from scipy.sparse import csc_array, csr_array

import mpi4py

mpi4py.rc.initialize = False  # do not initialize MPI automatically
mpi4py.rc.finalize = False  # do not finalize MPI automatically


def get_band_edge(ECmin_DFT, E, S, H, SigmaR_GW, SigmaR_PHN, Bmin, Bmax, side='left'):
    # First step: get a first estimate of the CB edge
    min_ind = np.argmin(np.abs(E - ECmin_DFT))
    Ek = calc_bandstructure(S, H, SigmaR_GW, SigmaR_PHN,
                            min_ind, Bmin, Bmax, side)
    ind_ek_plus = np.argmin(np.abs(Ek - ECmin_DFT))
    ECmin_int = Ek[ind_ek_plus]

    # Second step: refine the position of the CB edge
    min_ind = np.argmin(np.abs(E - ECmin_int))
    Ek = calc_bandstructure(S, H, SigmaR_GW, SigmaR_PHN,
                            min_ind, Bmin, Bmax, side)
    ind_ek = np.argmin(np.abs(Ek - ECmin_int))
    ECmin = Ek[ind_ek]

    return ECmin


def get_band_edge_interpol(ECmin_DFT, E, S, H, SigmaR_GW, SigmaL_GW, SigmaG_GW, SigmaR_PHN, Bmin, Bmax, side='left'):
    # First step: get a first estimate of the CB edge
    min_ind = np.argmin(np.abs(E - ECmin_DFT))

    if (E[min_ind] > ECmin_DFT and min_ind > 0):
        min_ind -= 1

    Ek = calc_bandstructure_interpol(
        E, S, H, ECmin_DFT, SigmaR_GW, SigmaL_GW, SigmaG_GW, SigmaR_PHN, min_ind, Bmin, Bmax, side)
    ind_ek_plus = np.argmin(np.abs(Ek - ECmin_DFT))
    ECmin_int = Ek[ind_ek_plus]

    # Second step: refine the position of the CB edge
    min_ind = np.argmin(np.abs(E - ECmin_int))

    if (E[min_ind] > ECmin_int and min_ind > 0):
        min_ind -= 1

    Ek = calc_bandstructure_interpol(
        E, S, H, ECmin_int, SigmaR_GW, SigmaL_GW, SigmaG_GW, SigmaR_PHN, min_ind, Bmin, Bmax, side)
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
    # Broadcasting minimum index
    if rank == 0:
        min_ind = np.argmin(np.abs(E - ECmin_DFT))
        comm.Bcast([min_ind, MPI.INT], root=0)
    else:
        min_ind = np.empty(1, dtype=np.int64)
        comm.Bcast([min_ind, MPI.INT], root=0)
        min_ind = min_ind[0]

    # Checking which rank has the minimum index
    send_rank = 0

    for check_rank in range(size):
        if min_ind in range(disp[1, check_rank], disp[1, check_rank] + count[1, check_rank]):
            send_rank = check_rank
            break

    # Sending the SigmaR_GW and SigmaR_PHN from the rank with the minimum index to rank 0
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
            comm.Recv([sr_phn_buf, MPI.DOUBLE_COMPLEX],
                      source=send_rank, tag=1)

            SigmaR_recv = csc_array(
                (sr_gw_buf, (rows, columns)), shape=(nao, nao))
            SigmaR_PHN_recv = csc_array(
                (sr_phn_buf, (rows, columns)), shape=(nao, nao))
            Ek = calc_bandstructure_mpi(
                S, H, SigmaR_recv, SigmaR_PHN_recv, Bmin, Bmax, side)
            ind_ek_plus = np.argmin(np.abs(Ek - ECmin_DFT))
            ECmin_int = Ek[ind_ek_plus]

    else:
        if send_rank == rank:
            sr_gw_buf = SigmaR_GW[min_ind -
                                  disp[1, rank]].toarray()[rows, columns]
            sr_phn_buf = SigmaR_PHN[min_ind -
                                    disp[1, rank]].toarray()[rows, columns]
            comm.Send([sr_gw_buf, MPI.DOUBLE_COMPLEX], dest=0, tag=0)
            comm.Send([sr_phn_buf, MPI.DOUBLE_COMPLEX], dest=0, tag=1)

    # Second step: refine the position of the CB edge
    # Broadcasting minimum index
    if rank == 0:
        min_ind = np.argmin(np.abs(E - ECmin_int))
        comm.Bcast(min_ind, root=0)
    else:
        min_ind = np.empty(1, dtype=np.int64)
        comm.Bcast(min_ind, root=0)
        min_ind = min_ind[0]

    # Checking which rank has the minimum index
    send_rank = 0

    for check_rank in range(size):
        if min_ind in range(disp[1, check_rank], disp[1, check_rank] + count[1, check_rank]):
            send_rank = check_rank
            break

    # Sending the SigmaR_GW and SigmaR_PHN from the rank with the minimum index to rank 0
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
            comm.Recv([sr_phn_buf, MPI.DOUBLE_COMPLEX],
                      source=send_rank, tag=1)

            SigmaR_recv = csc_array(
                (sr_gw_buf, (rows, columns)), shape=(nao, nao))
            SigmaR_PHN_recv = csc_array(
                (sr_phn_buf, (rows, columns)), shape=(nao, nao))
            Ek = calc_bandstructure_mpi(
                S, H, SigmaR_recv, SigmaR_PHN_recv, Bmin, Bmax, side)
            ind_ek = np.argmin(np.abs(Ek - ECmin_DFT))
            ECmin = Ek[ind_ek]

    else:
        if send_rank == rank:
            sr_gw_buf = SigmaR_GW[min_ind -
                                  disp[1, rank]].toarray()[rows, columns]
            sr_phn_buf = SigmaR_PHN[min_ind -
                                    disp[1, rank]].toarray()[rows, columns]
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


def get_band_edge_mpi_interpol(ECmin_DFT: float,
                               E: npt.NDArray[np.float64],
                               S: csr_array,
                               H: csr_array,
                               SigmaR_GW: npt.NDArray[csr_array],
                               SigmaL_GW: npt.NDArray[csr_array],
                               SigmaG_GW: npt.NDArray[csr_array],
                               SigmaR_PHN: npt.NDArray[csr_array],
                               ind_ek_plus: int,
                               rows: npt.NDArray[np.int64],
                               columns: npt.NDArray[np.int64],
                               Bmin: npt.NDArray[np.int64],
                               Bmax: npt.NDArray[np.int64],
                               comm: MPI.Comm,
                               rank: int,
                               size: int,
                               count: npt.NDArray[np.int32],
                               disp: npt.NDArray[np.int32],
                               side: str = 'left'):
    """
    This function calculates the conduction band edge of the system.

    Parameters
    ----------
    ECmin_DFT : float
        DFT conduction band minimum
    E : npt.NDArray[np.float64]
        Energy vector
    S : scipy.sparse.csr_array
        Overlap matrix
    H : scipy.sparse.csr_array
        Hamiltonian matrix
    SigmaR_GW : npt.NDArray[csr_array]
        Local SigmaR_GW vector in sparse format
    SigmaL_GW : npt.NDArray[csr_array]
        Local SigmaL_GW vector in sparse format
    SigmaG_GW : npt.NDArray[csr_array]
        Local SigmaG_GW vector in sparse format
    SigmaR_PHN : npt.NDArray[csr_array]
        Local SigmaR_PHN vector in sparse format
    ind_ek_plus : int
        Index of the band edge in the eigenvalue problem
    rows : npt.NDArray[np.int64]
        Row indices of the sparse matrices
    columns : npt.NDArray[np.int64]
        Column indices of the sparse matrices
    Bmin : npt.NDArray[np.int64]
        Minimum indices of the blocks
    Bmax : npt.NDArray[np.int64]
        Maximum indices of the blocks
    comm : MPI.Comm
        MPI communicator
    rank : int
        Rank of the current process
    size : int
        Number of processes
    count : npt.NDArray[np.int32]
        Number of elements per rank
    disp : npt.NDArray[np.int32]
        Displacement of elements per rank
    side : str, optional
        Side of the band edge, by default 'left'
    Returns
    -------
    ECmin : float
        Conduction band minimum
    ind_ek_plus : int
        Index of the band edge in the eigenvalue problem

    """
    nao = Bmax[-1] + 1
    SigmaR_GW_vec = np.ndarray((2, ), dtype=object)
    SigmaL_GW_vec = np.ndarray((2, ), dtype=object)
    SigmaG_GW_vec = np.ndarray((2, ), dtype=object)
    SigmaR_PHN_vec = np.ndarray((2, ), dtype=object)

    # First step: get a first estimate of the CB edge
    (min_ind, send_rank_1, send_rank_2) = get_send_ranks_interpol(
        ECmin_DFT, E, comm, rank, size, count, disp)
    send_sigmas_GWRGL_PHNR_to_root(SigmaR_GW_vec, SigmaL_GW_vec, SigmaG_GW_vec, SigmaR_PHN_vec,
                                   send_rank_1, send_rank_2, min_ind, rank, comm, disp,  SigmaR_GW, SigmaL_GW, SigmaG_GW, SigmaR_PHN, rows, columns, nao)
    if rank == 0:
        Ek = calc_bandstructure_mpi_interpol(
            E, S, H, ECmin_DFT, SigmaR_GW_vec, SigmaL_GW_vec, SigmaG_GW_vec, SigmaR_PHN_vec, min_ind, Bmin, Bmax, side)
        if (ind_ek_plus == -1):
            ind_ek_plus = np.argmin(np.abs(Ek - ECmin_DFT))
        ECmin_int = Ek[ind_ek_plus]
        # broadcasting the band edge (this is actually not necessary, but it is done for consistency), non-root nodes will not use it in get_send_ranks_interpol
        comm.Bcast([ECmin_int, MPI.DOUBLE], root=0)
    else:
        ECmin_int = np.empty(1, dtype=np.float64)
        comm.Bcast([ECmin_int, MPI.DOUBLE], root=0)
        ECmin_int = ECmin_int[0]

    # Second step: refine the position of the CB edge
    (min_ind, send_rank_1, send_rank_2) = get_send_ranks_interpol(
        ECmin_int, E, comm, rank, size, count, disp)
    send_sigmas_GWRGL_PHNR_to_root(SigmaR_GW_vec, SigmaL_GW_vec, SigmaG_GW_vec, SigmaR_PHN_vec,
                                   send_rank_1, send_rank_2, min_ind, rank, comm, disp, SigmaR_GW, SigmaL_GW, SigmaG_GW, SigmaR_PHN, rows, columns, nao)
    if rank == 0:
        Ek = calc_bandstructure_mpi_interpol(
            E, S, H, ECmin_int, SigmaR_GW_vec, SigmaL_GW_vec, SigmaG_GW_vec, SigmaR_PHN_vec, min_ind, Bmin, Bmax, side)
        # ind_ek_plus = np.argmin(np.abs(Ek - ECmin_int))
        ECmin = Ek[ind_ek_plus]
        # broadcasting the band edge
        comm.Bcast([ECmin, MPI.DOUBLE], root=0)
        print("send rank 1 was: " + str(send_rank_1) + " send rank 2 was: " +
              str(send_rank_2) + " new band edge: " + str(ECmin))
    else:
        ECmin = np.empty(1, dtype=np.float64)
        comm.Bcast([ECmin, MPI.DOUBLE], root=0)
        ECmin = ECmin[0]
    # returning the band edge and the index of the band edge in the eigenvalue problem
    return ECmin, ind_ek_plus


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
    # Broadcasting minimum index
    if rank == 0:
        min_ind = np.argmin(np.abs(E - ECmin_DFT))
        if (E[min_ind] > ECmin_DFT and min_ind > 0):
            min_ind -= 1
        comm.Bcast([min_ind, MPI.INT], root=0)
    else:
        min_ind = np.empty(1, dtype=np.int64)
        comm.Bcast([min_ind, MPI.INT], root=0)
        min_ind = min_ind[0]

    # Checking which rank has the minimum index
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
            comm.Recv([sr_gw_buf, MPI.DOUBLE_COMPLEX],
                      source=send_rank_1, tag=0)
            comm.Recv([sl_gw_buf, MPI.DOUBLE_COMPLEX],
                      source=send_rank_1, tag=1)
            comm.Recv([sg_gw_buf, MPI.DOUBLE_COMPLEX],
                      source=send_rank_1, tag=2)
            comm.Recv([sr_phn_buf, MPI.DOUBLE_COMPLEX],
                      source=send_rank_1, tag=3)

            SigmaR_GW_vec[0] = csc_array(
                (sr_gw_buf, (rows, columns)), shape=(nao, nao))
            SigmaL_GW_vec[0] = csc_array(
                (sl_gw_buf, (rows, columns)), shape=(nao, nao))
            SigmaG_GW_vec[0] = csc_array(
                (sg_gw_buf, (rows, columns)), shape=(nao, nao))
            SigmaR_PHN_vec[0] = csc_array(
                (sr_phn_buf, (rows, columns)), shape=(nao, nao))

    else:
        if send_rank_1 == rank:
            sr_gw_buf = SigmaR_GW[min_ind -
                                  disp[1, rank]].toarray()[rows, columns]
            sl_gw_buf = SigmaL_GW[min_ind -
                                  disp[1, rank]].toarray()[rows, columns]
            sg_gw_buf = SigmaG_GW[min_ind -
                                  disp[1, rank]].toarray()[rows, columns]
            sr_phn_buf = SigmaR_PHN[min_ind -
                                    disp[1, rank]].toarray()[rows, columns]
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
            comm.Recv([sr_gw_buf, MPI.DOUBLE_COMPLEX],
                      source=send_rank_2, tag=0)
            comm.Recv([sl_gw_buf, MPI.DOUBLE_COMPLEX],
                      source=send_rank_2, tag=1)
            comm.Recv([sg_gw_buf, MPI.DOUBLE_COMPLEX],
                      source=send_rank_2, tag=2)
            comm.Recv([sr_phn_buf, MPI.DOUBLE_COMPLEX],
                      source=send_rank_2, tag=3)

            SigmaR_GW_vec[1] = csc_array(
                (sr_gw_buf, (rows, columns)), shape=(nao, nao))
            SigmaL_GW_vec[1] = csc_array(
                (sl_gw_buf, (rows, columns)), shape=(nao, nao))
            SigmaG_GW_vec[1] = csc_array(
                (sg_gw_buf, (rows, columns)), shape=(nao, nao))
            SigmaR_PHN_vec[1] = csc_array(
                (sr_phn_buf, (rows, columns)), shape=(nao, nao))

    else:
        if send_rank_2 == rank:
            sr_gw_buf = SigmaR_GW[min_ind + 1 -
                                  disp[1, rank]].toarray()[rows, columns]
            sl_gw_buf = SigmaL_GW[min_ind + 1 -
                                  disp[1, rank]].toarray()[rows, columns]
            sg_gw_buf = SigmaG_GW[min_ind + 1 -
                                  disp[1, rank]].toarray()[rows, columns]
            sr_phn_buf = SigmaR_PHN[min_ind + 1 -
                                    disp[1, rank]].toarray()[rows, columns]
            comm.Send([sr_gw_buf, MPI.DOUBLE_COMPLEX], dest=0, tag=0)
            comm.Send([sl_gw_buf, MPI.DOUBLE_COMPLEX], dest=0, tag=1)
            comm.Send([sg_gw_buf, MPI.DOUBLE_COMPLEX], dest=0, tag=2)
            comm.Send([sr_phn_buf, MPI.DOUBLE_COMPLEX], dest=0, tag=3)


if __name__ == '__main__':
    MPI.Init_thread(required=MPI.THREAD_MULTIPLE)
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    name = MPI.Get_processor_name()

    # path to solution
    solution_path = "/usr/scratch2/tortin12/chexia/"
    hamiltonian_path = os.path.join(solution_path, "CNT_evensort48")

    # one orbital on C atoms, two same types
    no_orb = np.array([2, 3])
    Vappl = 0.0
    energy = np.linspace(-15, 10, 6, endpoint=True,
                         dtype=float)  # Energy Vector
    hamiltonian_obj = OMENHamClass.Hamiltonian(
        hamiltonian_path, no_orb, Vappl=Vappl, rank=rank, potential_type='atomic')
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
    ECmin = -3.447

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
    print("rank: " + str(rank) + " posess energies: " + repr(energy_loc))
    # initialize self energy
    sr_h2g_full = np.loadtxt(
        '/usr/scratch/mont-fort17/dleonard/CNT/python_testfiles/calc_band_edge_interpol/sr_gw.dat').view(complex)
    sl_h2g_full = np.loadtxt(
        '/usr/scratch/mont-fort17/dleonard/CNT/python_testfiles/calc_band_edge_interpol/sl_gw.dat').view(complex)
    sg_h2g_full = np.loadtxt(
        '/usr/scratch/mont-fort17/dleonard/CNT/python_testfiles/calc_band_edge_interpol/sg_gw.dat').view(complex)

    # sr_h2g = np.zeros((count[1,rank], no), dtype=np.complex128)
    sr_h2g = sr_h2g_full[disp[1, rank]:disp[1, rank] + count[1, rank], :]
    sl_h2g = sl_h2g_full[disp[1, rank]:disp[1, rank] + count[1, rank], :]
    sg_h2g = sg_h2g_full[disp[1, rank]:disp[1, rank] + count[1, rank], :]
    # transform from 2D format to list/vector of sparse arrays format-----------
    sr_h2g_vec = change_format.sparse2vecsparse_v2(sr_h2g, rows, columns, nao)
    sl_h2g_vec = change_format.sparse2vecsparse_v2(sl_h2g, rows, columns, nao)
    sg_h2g_vec = change_format.sparse2vecsparse_v2(sg_h2g, rows, columns, nao)

    sr_ephn_h2g_vec = change_format.sparse2vecsparse_v2(np.zeros((count[1, rank], no), dtype=np.complex128), rows,
                                                        columns, nao)

    # ECmin_single = get_band_edge(-3.52400739, energy, hamiltonian_obj.Overlap['H_4'], hamiltonian_obj.Hamiltonian['H_4'], sr_h2g_vec, sr_ephn_h2g_vec, bmin, bmax, side = 'left')
    ECmin_MPI = get_band_edge_mpi_interpol(-3.67759569,
                                           energy,
                                           hamiltonian_obj.Overlap['H_4'],
                                           hamiltonian_obj.Hamiltonian['H_4'],
                                           sr_h2g_vec,
                                           sl_h2g_vec,
                                           sg_h2g_vec,
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
        # print(ECmin_single)
        # for the interpol case: from: -3.67759569 to: -3.68866105
        print(ECmin_MPI)
