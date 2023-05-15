import numpy as np
from scipy.sparse import csc_array
from bandstructure.calc_contact_bs import calc_bandstructure, calc_bandstructure_mpi
from mpi4py import MPI

def get_band_edge(ECmin_DFT, E, S, H, SigmaR_GW, SigmaR_PHN, Bmin, Bmax, side = 'left'):
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

def get_band_edge_mpi(ECmin_DFT, E, S, H, SigmaR_GW, SigmaR_PHN, rows, columns, Bmin, Bmax, comm, rank, size, count, disp, side = 'left'):
    nao = Bmax[-1] + 1
    # First step: get a first estimate of the CB edge

    ## Broadcasting minimum index
    if rank == 0:
        min_ind = np.argmin(np.abs(E - ECmin_DFT))    
        comm.Bcast(min_ind, root = 0)
    else:
        min_ind = np.empty(1, dtype = np.int64)
        comm.Bcast(min_ind, root = 0)
    
    ## Checking which rank has the minimum index
    send_rank = 0

    for check_rank in range(size):
        if check_rank in range(disp[1, check_rank], disp[1, check_rank] + count[1, check_rank]):
            send_rank = check_rank
            break
    
    ## Sending the SigmaR_GW and SigmaR_PHN from the rank with the minimum index to rank 0
    if rank == 0:
        if send_rank == 0:
            Ek = calc_bandstructure_mpi(S, H, SigmaR_GW[min_ind % disp[1, rank]], SigmaR_PHN[min_ind % disp[1, rank]], Bmin, Bmax, side)
            ind_ek_plus = np.argmin(np.abs(Ek - ECmin_DFT))
            ECmin_int = Ek[ind_ek_plus]
            
        else:
            sr_gw_buf = np.empty(count[1, send_rank], dtype = np.float64)
            sr_phn_buf = np.empty(count[1, send_rank], dtype = np.float64)
            comm.Receive([sr_gw_buf, MPI.DOUBLE_COMPLEX], dest = send_rank, tag = 0)
            comm.Receive([sr_phn_buf, MPI.DOUBLE_COMPLEX], dest = send_rank, tag = 1)

            SigmaR_recv = csc_array((sr_gw_buf, (rows, columns)), shape = (nao, nao))
            SigmaR_PHN_recv = csc_array((sr_phn_buf, (rows, columns)), shape = (nao, nao))
            Ek = calc_bandstructure_mpi(S, H, SigmaR_recv, SigmaR_PHN_recv, Bmin, Bmax, side)
            ind_ek_plus = np.argmin(np.abs(Ek - ECmin_DFT))
            ECmin_int = Ek[ind_ek_plus]

    else:
        if send_rank == rank:
            sr_gw_buf = SigmaR_GW[min_ind % disp[1, rank]].data
            sr_phn_buf = SigmaR_PHN[min_ind % disp[1, rank]].data
            comm.Send([sr_gw_buf, MPI.DOUBLE_COMPLEX], dest = 0, tag = 0)
            comm.Send([sr_phn_buf, MPI.DOUBLE_COMPLEX], dest = 0, tag = 1)

    # Second step: refine the position of the CB edge
     ## Broadcasting minimum index
    if rank == 0:
        min_ind = np.argmin(np.abs(E - ECmin_int))    
        comm.Bcast(min_ind, root = 0)
    else:
        min_ind = np.empty(1, dtype = np.int64)
        comm.Bcast(min_ind, root = 0)

    ## Checking which rank has the minimum index
    send_rank = 0

    for check_rank in range(size):
        if check_rank in range(disp[1, check_rank], disp[1, check_rank] + count[1, check_rank]):
            send_rank = check_rank
            break

    ## Sending the SigmaR_GW and SigmaR_PHN from the rank with the minimum index to rank 0
    if rank == 0:
        if send_rank == 0:
            Ek = calc_bandstructure_mpi(S, H, SigmaR_GW[min_ind % disp[1, rank]], SigmaR_PHN[min_ind % disp[1, rank]], Bmin, Bmax, side)
            ind_ek = np.argmin(np.abs(Ek - ECmin_int))
            ECmin = Ek[ind_ek]
            
        else:
            sr_gw_buf = np.empty(count[1, send_rank], dtype = np.float64)
            sr_phn_buf = np.empty(count[1, send_rank], dtype = np.float64)
            comm.Receive([sr_gw_buf, MPI.DOUBLE_COMPLEX], dest = send_rank, tag = 0)
            comm.Receive([sr_phn_buf, MPI.DOUBLE_COMPLEX], dest = send_rank, tag = 1)

            SigmaR_recv = csc_array((sr_gw_buf, (rows, columns)), shape = (nao, nao))
            SigmaR_PHN_recv = csc_array((sr_phn_buf, (rows, columns)), shape = (nao, nao))
            Ek = calc_bandstructure_mpi(S, H, SigmaR_GW[min_ind % disp[1, rank]], SigmaR_PHN[min_ind % disp[1, rank]], Bmin, Bmax, side)
            ind_ek = np.argmin(np.abs(Ek - ECmin_DFT))
            ECmin = Ek[ind_ek]

    else:
        if send_rank == rank:
            sr_gw_buf = SigmaR_GW[min_ind % disp[1, rank]].data
            sr_phn_buf = SigmaR_PHN[min_ind % disp[1, rank]].data
            comm.Send([sr_gw_buf, MPI.DOUBLE_COMPLEX], dest = 0, tag = 0)
            comm.Send([sr_phn_buf, MPI.DOUBLE_COMPLEX], dest = 0, tag = 1)

    # Broadcasting the band edge
    if rank == 0:
        comm.Bcast(ECmin, root = 0)
    else:
        ECmin = np.empty(1, dtype = np.float64)
        comm.Bcast(ECmin, root = 0)
    # returning the band edge
    return ECmin