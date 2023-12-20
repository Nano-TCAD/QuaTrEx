import numpy as np
from scipy import sparse
from quatrex.utils.matrix_creation import create_matrices_H, initialize_block_G, mat_assembly_fullG
from quatrex.OBC.beyn_new import beyn
from quatrex.OBC.sancho import open_boundary_conditions
from functools import partial

import cupy as cp

def rgf_standaloneGF_GPU(
           ham_diag,
           ham_upper,
           ham_lower,
           sg_diag,
           sg_upper,
           sg_lower,
           sl_diag,
           sl_upper,
           sl_lower,
           GR,
           GRnn1,
           GL,
           GLnn1,
           GG,
           GGnn1,
           DOS,
           nE,
           nP,
           idE,
           Bmin_fi,
           Bmax_fi,
           factor=1.0,
):
    # rgf_GF(DH, E, EfL, EfR, Temp) This could be the function call considering Leo's code
    '''
    Working!
    
    '''
    Bmax = Bmax_fi - 1
    Bmin = Bmin_fi - 1
    Bsize = max(Bmax - Bmin + 1)  # Used for declaration of variables
    NB = len(Bmin)
    NT = Bmax[NB - 1] + 1  # Not used in this fcn

    LBsize = Bmax[0] - Bmin[0] + 1
    RBsize = Bmax[NB - 1] - Bmin[NB - 1] + 1

    energy_batchsize = ham_diag.shape[1]

    gR = np.zeros((NB, energy_batchsize, Bsize, Bsize), dtype=np.cfloat)  # Retarded (right)
    gL = np.zeros((NB, energy_batchsize, Bsize, Bsize), dtype=np.cfloat)  # Lesser (right)
    gG = np.zeros((NB, energy_batchsize, Bsize, Bsize), dtype=np.cfloat)  # Greater (right)
    SigLB = np.zeros((NB - 1, energy_batchsize, Bsize, Bsize), dtype=np.cfloat)  # Lesser boundary self-energy
    SigGB = np.zeros((NB - 1, energy_batchsize, Bsize, Bsize), dtype=np.cfloat)  # Greater boundary self-energy

    IdE = np.zeros((NB, energy_batchsize))
    n = np.zeros((NB, energy_batchsize))
    p = np.zeros((NB, energy_batchsize))

    # First step of iteration
    NN = Bmax[-1] - Bmin[-1] + 1
    gR[-1, :, 0:NN, 0:NN] = np.linalg.inv(ham_diag[-1, :, 0:NN, 0:NN])
    gL[-1, :, 0:NN, 0:NN] = gR[-1, :, 0:NN, 0:NN] @ (sl_diag[-1, :, 0:NN, 0:NN]) @ gR[-1, :, 0:NN, 0:NN].conjugate().transpose((0,2,1))
    gG[-1, :, 0:NN, 0:NN] = gR[-1, :, 0:NN, 0:NN] @ (sg_diag[-1, :, 0:NN, 0:NN]) @ gR[-1, :, 0:NN, 0:NN].conjugate().transpose((0,2,1))

    for IB in range(NB - 2, -1, -1):
        NI = Bmax[IB] - Bmin[IB] + 1
        NP = Bmax[IB + 1] - Bmin[IB + 1] + 1

        # Extracting diagonal Hamiltonian block
        if(IB == 0):
            M_c = M[Bmin[IB]:Bmax[IB] + 1, Bmin[IB]:Bmax[IB] + 1].toarray() - SigRBL
            SigL_c = SigL[Bmin[IB]:Bmax[IB] + 1, Bmin[IB]:Bmax[IB] + 1].toarray() + SigLBL
            SigG_c = SigG[Bmin[IB]:Bmax[IB] + 1, Bmin[IB]:Bmax[IB] + 1].toarray() + SigGBL
        else: 
            M_c = M[Bmin[IB]:Bmax[IB] + 1, Bmin[IB]:Bmax[IB] + 1].toarray()
            SigL_c = SigL[Bmin[IB]:Bmax[IB] + 1, Bmin[IB]:Bmax[IB] + 1].toarray()
            SigG_c = SigG[Bmin[IB]:Bmax[IB] + 1, Bmin[IB]:Bmax[IB] + 1].toarray()
        #M_c = M[Bmin[IB]:Bmax[IB] + 1, Bmin[IB]:Bmax[IB] + 1].toarray()

        # Extracting off-diagonal Hamiltonian block (right)
        M_r = M[Bmin[IB]:Bmax[IB] + 1, Bmin[IB + 1]:Bmax[IB + 1] + 1].toarray()

        # Extracting off-diagonal Hamiltonian block (lower)
        M_d = M[Bmin[IB + 1]:Bmax[IB + 1] + 1, Bmin[IB]:Bmax[IB] + 1].toarray()

        # Extracting diagonal lesser Self-energy block
        #SigL_c = SigL[Bmin[IB]:Bmax[IB] + 1, Bmin[IB]:Bmax[IB] + 1].toarray()

        # Extracting off-diagonal lesser Self-energy block (lower)
        SigL_l = SigL[Bmin[IB + 1]:Bmax[IB + 1] + 1, Bmin[IB]:Bmax[IB] + 1].toarray()

        # Extracting diagonal greater Self-energy block
        #SigG_c = SigG[Bmin[IB]:Bmax[IB] + 1, Bmin[IB]:Bmax[IB] + 1].toarray()

        # Extracting off-diagonal greater Self-energy block (lower)
        SigG_l = SigG[Bmin[IB + 1]:Bmax[IB + 1] + 1, Bmin[IB]:Bmax[IB] + 1].toarray()


        gR[IB, 0:NI, 0:NI] = np.linalg.inv(M_c \
                            - M_r \
                            @ gR[IB+1, 0:NP, 0:NP] \
                            @ M_d)#######
        # AL, What is this? Handling off-diagonal sigma elements?
        AL = M_r \
            @ gR[IB+1, 0:NP, 0:NP] \
            @ SigL_l
        
        SigLB[IB, 0:NI, 0:NI] = M_r \
                            @ gL[IB+1, 0:NP, 0:NP] \
                            @ M_r.T.conj() \
                            - (AL - AL.T.conj())

        # gL[IB, 0:NI, 0:NI] = gR[IB, 0:NI, 0:NI] \
        #                     @ (SigL_c \
        #                     + M_r \
        #                     @ gL[IB+1, 0:NP, 0:NP] \
        #                     @ M_r.T.conj() \
        #                     - (AL - AL.T.conj()))  \
        #                     @ gR[IB, 0:NI, 0:NI].T.conj() # Confused about the AL

        gL[IB, 0:NI, 0:NI] = gR[IB, 0:NI, 0:NI] \
                            @ (SigL_c \
                            + SigLB[IB, 0:NI, 0:NI])  \
                            @ gR[IB, 0:NI, 0:NI].T.conj() # Confused about the AL

        ### What is this?
        AG = M_r \
            @ gR[IB+1, 0:NP, 0:NP] \
            @ SigG_l     # Handling off-diagonal sigma elements? Prob. need to check

        # gG[IB, 0:NI, 0:NI] = gR[IB, 0:NI, 0:NI] \
        #                     @ (SigG_c \
        #                     + M_r \
        #                     @ gG[IB+1, 0:NP, 0:NP] \
        #                     @ M_r.T.conj() \
        #                     - (AG - AG.T.conj())) \
        #                     @ gR[IB, 0:NI, 0:NI].T.conj() # Confused about the AG. 
        SigGB[IB, 0:NI, 0:NI] = M_r \
                            @ gG[IB+1, 0:NP, 0:NP] \
                            @ M_r.T.conj() \
                            - (AG - AG.T.conj())

        gG[IB, 0:NI, 0:NI] = gR[IB, 0:NI, 0:NI] \
                            @ (SigG_c \
                                + SigGB[IB, 0:NI, 0:NI]) \
                                @ gR[IB, 0:NI, 0:NI].T.conj() # Confused about the AG. 
    
    #Second step of iteration
    GR[0, :NI, :NI] = gR[0, :NI, :NI]
    GRnn1[0, :NI, :NP] = -GR[0, :NI, :NI] @ M[Bmin[0]:Bmax[0] + 1, Bmin[1]:Bmax[1] + 1].toarray() @ gR[1, :NP, :NP]

    GL[0, :NI, :NI] = gL[0, :NI, :NI]
    GLnn1[0, :NI, :NP] = GR[0, :NI, :NI] @ SigL[Bmin[0]:Bmax[0]+1, Bmin[1]:Bmax[1]+1].toarray() @ gR[1, :NP, :NP].T.conj() \
                - GR[0,:NI,:NI] @ M[Bmin[0]:Bmax[0]+1, Bmin[1]:Bmax[1]+1].toarray() @ gL[1, :NP, :NP] \
                - GL[0,:NI,:NI] @ M[Bmin[1]:Bmax[1]+1, Bmin[0]:Bmax[0]+1].toarray().T.conj() @ gR[1, :NP, :NP].T.conj()

    GG[0, :NI, :NI] = gG[0, :NI, :NI]
    GGnn1[0, :NI, :NP] = GR[0, :NI, :NI] @ SigG[Bmin[0]:Bmax[0]+1, Bmin[1]:Bmax[1]+1].toarray() @ gR[1, :NP, :NP].T.conj() \
                - GR[0,:NI,:NI] @ M[Bmin[0]:Bmax[0]+1, Bmin[1]:Bmax[1]+1].toarray() @ gG[1, :NP, :NP] \
                - GG[0,:NI,:NI] @ M[Bmin[1]:Bmax[1]+1, Bmin[0]:Bmax[0]+1].toarray().T.conj() @ gR[1, :NP, :NP].T.conj() 
    
    idE[0] = np.real(np.trace(SigGB[0, :NI, :NI] @ GL[0, :NI, :NI] - GG[0, :NI, :NI] @ SigLB[0, :NI, :NI]))

    for IB in range(1, NB):

        NM = Bmax[IB - 1] - Bmin[IB - 1] + 1
        NI = Bmax[IB] - Bmin[IB] + 1

        # # Extracting off-diagonal Hamiltonian block (upper)
        M_u = M[Bmin[IB - 1]:Bmax[IB - 1] + 1, Bmin[IB]:Bmax[IB] + 1].toarray()

        # # Extracting off-diagonal Hamiltonian block (left)
        M_l = M[Bmin[IB]:Bmax[IB] + 1, Bmin[IB - 1]:Bmax[IB - 1] + 1].toarray()

        # Extracting off-diagonal lesser Self-energy block (left)
        SigL_l = SigL[Bmin[IB]:Bmax[IB] + 1, Bmin[IB - 1]:Bmax[IB - 1] + 1].toarray()

        # Extracting off-diagonal greater Self-energy block (left)
        SigG_l = SigG[Bmin[IB]:Bmax[IB] + 1, Bmin[IB - 1]:Bmax[IB - 1] + 1].toarray()

        GR[IB, :NI, :NI] = gR[IB, :NI, :NI] + gR[IB, :NI, :NI] \
                        @ M_l \
                        @ GR[IB-1,0:NM, 0:NM] \
                        @ M_u \
                        @ gR[IB, :NI, :NI]
        # What is this? Handling off-diagonal elements?
        AL = gR[IB, :NI, :NI] \
            @ SigL_l \
            @ GR[IB-1, 0:NM, 0:NM].T.conj() \
            @ M_l.T.conj() \
            @ gR[IB, 0:NI, 0:NI].T.conj()
        # What is this?
        BL = gR[IB, :NI, :NI] \
            @ M_l \
            @ GR[IB-1, 0:NM, 0:NM] \
            @ M_u \
            @ gL[IB, :NI, :NI]

        GL[IB, 0:NI, 0:NI] = gL[IB, :NI, :NI] \
                            + gR[IB, 0:NI, 0:NI] \
                            @ M_l \
                            @ GL[IB-1, 0:NM, 0:NM] \
                            @ M_l.T.conj() \
                            @ gR[IB, :NI, :NI].T.conj() \
                            - (AL - AL.T.conj()) + (BL - BL.T.conj())


        AG = gR[IB, 0:NI, 0:NI] \
            @ SigG_l \
            @ GR[IB-1, 0:NM, 0:NM].T.conj() \
            @ M_l.T.conj() \
            @ gR[IB, 0:NI, 0:NI].T.conj()

        BG = gR[IB, 0:NI, 0:NI] \
            @ M_l \
            @ GR[IB-1, 0:NM, 0:NM] \
            @ M_u \
            @ gG[IB, 0:NI, 0:NI]

        GG[IB, 0:NI, 0:NI] = gG[IB, 0:NI, 0:NI] \
                            + gR[IB, 0:NI, 0:NI] \
                            @ M_l \
                            @ GG[IB-1, 0:NM, 0:NM] \
                            @ M_l.T.conj() \
                            @ gR[IB, 0:NI, 0:NI].T.conj() \
                            - (AG - AG.T.conj()) + (BG - BG.T.conj()) #

        if IB < NB - 1:  #Off-diagonal are only interesting for IdE!

            # # Extracting off-diagonal Hamiltonian block (right)
            M_r = M[Bmin[IB]:Bmax[IB] + 1, Bmin[IB + 1]:Bmax[IB + 1] + 1].toarray()

            # # Extracting off-diagonal Hamiltonian block (lower)
            M_d = M[Bmin[IB + 1]:Bmax[IB + 1] + 1, Bmin[IB]:Bmax[IB] + 1].toarray()

            # Extracting off-diagonal lesser Self-energy block (right)
            SigL_r = SigL[Bmin[IB]:Bmax[IB] + 1, Bmin[IB + 1]:Bmax[IB + 1] + 1].toarray()

            # Extracting off-diagonal greater Self-energy block (right)
            SigG_r = SigG[Bmin[IB]:Bmax[IB] + 1, Bmin[IB + 1]:Bmax[IB + 1] + 1].toarray()

            NP = Bmax[IB + 1] - Bmin[IB + 1] + 1

            GRnn1[IB, 0:NI, 0:NP] = - GR[IB, 0:NI, 0:NI] \
                                    @ M_r \
                                    @ gR[IB+1, 0:NP, 0:NP]

            GLnn1[IB, 0:NI, 0:NP] = GR[IB, 0:NI, 0:NI] \
                                    @ SigL_r \
                                    @ gR[IB+1, 0:NP, 0:NP].T.conj() \
                                    - GR[IB, 0:NI, 0:NI] \
                                    @ M_r \
                                    @ gL[IB+1, 0:NP, 0:NP] \
                                    - GL[IB, :NI, :NI] \
                                    @ M_d.T.conj() \
                                    @ gR[IB+1, 0:NP, 0:NP].T.conj()
            GGnn1[IB, 0:NI, 0:NP] = GR[IB, 0:NI, 0:NI] \
                                    @ SigG_r \
                                    @ gR[IB+1, 0:NP, 0:NP].T.conj() \
                                    - GR[IB, 0:NI, 0:NI] \
                                    @ M_r \
                                    @ gG[IB+1, 0:NP, 0:NP] \
                                    - GG[IB, 0:NI, 0:NI] \
                                    @ M_d.T.conj() \
                                    @ gR[IB+1, 0:NP, 0:NP].T.conj()   
            idE[IB] = np.real(np.trace(SigGB[IB, :NI, :NI] @ GL[IB, :NI, :NI] - GG[IB, :NI, :NI] @ SigLB[IB, :NI, :NI]))    
    for IB in range(NB):
        
        NI = Bmax[IB] - Bmin[IB] + 1
        GR[IB, :, :] *= factor
        GL[IB, :, :] *= factor
        GG[IB, :, :] *= factor
        DOS[IB] = 1j * np.trace(GR[IB, :, :] - GR[IB, :, :].T.conj())
        nE[IB] = -1j * np.trace(GL[IB, :, :])
        nP[IB] = 1j * np.trace(GG[IB, :, :])

        if IB < NB-1:
            NP = Bmax[IB+1] - Bmin[IB+1] + 1
            #idE[IB] = -2 * np.trace(np.real(H[Bmin[IB+1]:Bmax[IB+1]+1, Bmin[IB]:Bmax[IB]+1].toarray() @ GLnn1[IB, 0:NI, 0:NP]))
            GRnn1[IB, :, :] *= factor
            GLnn1[IB, :, :] *= factor
            GGnn1[IB, :, :] *= factor

    
    #idE[NB - 1] = idE[NB - 2]
    idE[NB-1] = np.real(np.trace(SigGBR @ GL[NB-1, :NI, :NI] - GG[NB-1, :NI, :NI] @ SigLBR))
