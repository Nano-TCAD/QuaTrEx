import numpy as np
from bandstructure.calc_contact_bs import calc_bandstructure

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
    # First step: get a first estimate of the CB edge
    if rank == 0:
        min_ind = np.argmin(np.abs(E - ECmin_DFT))
        Ek = calc_bandstructure(S, H, SigmaR_GW, SigmaR_PHN, min_ind, Bmin, Bmax, side)
        ind_ek_plus = np.argmin(np.abs(Ek - ECmin_DFT))
        ECmin_int = Ek[ind_ek_plus]
    
        # Second step: refine the position of the CB edge
        min_ind = np.argmin(np.abs(E - ECmin_int))
        comm.Bcast(min_ind, root = 0)
    else:
        min_ind = np.empty(1, dtype = np.int64)
        comm.Bcast(min_ind, root = 0)
    
    send_rank = 0
    for check_rank in range(size):
        if check_rank in range(disp[1, rank], disp[1, rank] + count[1, rank]):
            send_rank = check_rank
            break
    
            
    
