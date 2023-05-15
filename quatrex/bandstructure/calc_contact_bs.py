import numpy as np
from scipy.sparse import csc_matrix
from scipy.linalg import eig

def calc_bandstructure(S, H, SigmaR_GW, SigmaR_PHN, indE, Bmin, Bmax,side):

    
    #SigR = SigR + np.diag(SigmaR_PHN[indE, :])
    H = H + SigmaR_GW[indE] + SigmaR_PHN[indE]
    
    if side == 'left':
        LBsize = Bmax[0] - Bmin[0] + 1
        H00 = H[:LBsize, :LBsize].toarray()
        H01 = H[:LBsize, LBsize:2*LBsize].toarray()
        H10 = H[LBsize:2*LBsize, :LBsize].toarray()
        
        S00 = S[:LBsize, :LBsize].toarray()
        S01 = S[:LBsize, LBsize:2*LBsize].toarray()
        S10 = S[LBsize:2*LBsize, :LBsize].toarray()
        
    elif side == 'right':
        RBsize = Bmax[-1] - Bmin[-1] + 1
        H00 = H[-RBsize:, -RBsize:].toarray()
        H01 = H[-RBsize:, -2*RBsize:-RBsize].toarray()
        H10 = H[-2*RBsize:-RBsize, -RBsize:].toarray()

        S00 = S[-RBsize:, -RBsize:].toarray()
        S01 = S[-RBsize:, -2*RBsize:-RBsize].toarray()
        S10 = S[-2*RBsize:-RBsize, -RBsize:].toarray()
        
    Ek = np.sort(np.real(eig(H00+H01+H10, b=S00+S01+S10, right = False)))
    
    return Ek

def calc_bandstructure_mpi(S, H, SigmaR_GW, SigmaR_PHN, Bmin, Bmax,side):

    
    #SigR = SigR + np.diag(SigmaR_PHN[indE, :])
    H = H + SigmaR_GW + SigmaR_PHN
    
    if side == 'left':
        LBsize = Bmax[0] - Bmin[0] + 1
        H00 = H[:LBsize, :LBsize].toarray()
        H01 = H[:LBsize, LBsize:2*LBsize].toarray()
        H10 = H[LBsize:2*LBsize, :LBsize].toarray()
        
        S00 = S[:LBsize, :LBsize].toarray()
        S01 = S[:LBsize, LBsize:2*LBsize].toarray()
        S10 = S[LBsize:2*LBsize, :LBsize].toarray()
        
    elif side == 'right':
        RBsize = Bmax[-1] - Bmin[-1] + 1
        H00 = H[-RBsize:, -RBsize:].toarray()
        H01 = H[-RBsize:, -2*RBsize:-RBsize].toarray()
        H10 = H[-2*RBsize:-RBsize, -RBsize:].toarray()

        S00 = S[-RBsize:, -RBsize:].toarray()
        S01 = S[-RBsize:, -2*RBsize:-RBsize].toarray()
        S10 = S[-2*RBsize:-RBsize, -RBsize:].toarray()
        
    Ek = np.sort(np.real(eig(H00+H01+H10, b=S00+S01+S10, right = False)))
    
    return Ek