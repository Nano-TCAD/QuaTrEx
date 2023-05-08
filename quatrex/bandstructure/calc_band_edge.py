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
