import numpy as np
# from scipy import sparse
# from quatrex.utils.matrix_creation import create_matrices_H, initialize_block_G, mat_assembly_fullG
# from quatrex.OBC.beyn_new import beyn
# from quatrex.OBC.sancho import open_boundary_conditions
# from functools import partial

import cupy as cp

def rgf_standaloneGF_batched(
           ham_diag,
           ham_upper,
           ham_lower,
           sg_diag,
           sg_upper,
           sg_lower,
           sl_diag,
           sl_upper,
           sl_lower,
           SigGBR,
           SigLBR,
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
           Bmax_fi
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

        # # Extracting diagonal Hamiltonian block
        # if(IB == 0):
        #     M_c = M[Bmin[IB]:Bmax[IB] + 1, Bmin[IB]:Bmax[IB] + 1].toarray() - SigRBL
        #     SigL_c = SigL[Bmin[IB]:Bmax[IB] + 1, Bmin[IB]:Bmax[IB] + 1].toarray() + SigLBL
        #     SigG_c = SigG[Bmin[IB]:Bmax[IB] + 1, Bmin[IB]:Bmax[IB] + 1].toarray() + SigGBL
        # else: 
        #     M_c = M[Bmin[IB]:Bmax[IB] + 1, Bmin[IB]:Bmax[IB] + 1].toarray()
        #     SigL_c = SigL[Bmin[IB]:Bmax[IB] + 1, Bmin[IB]:Bmax[IB] + 1].toarray()
        #     SigG_c = SigG[Bmin[IB]:Bmax[IB] + 1, Bmin[IB]:Bmax[IB] + 1].toarray()
        # #M_c = M[Bmin[IB]:Bmax[IB] + 1, Bmin[IB]:Bmax[IB] + 1].toarray()

        # # Extracting off-diagonal Hamiltonian block (right)
        # M_r = M[Bmin[IB]:Bmax[IB] + 1, Bmin[IB + 1]:Bmax[IB + 1] + 1].toarray()

        # # Extracting off-diagonal Hamiltonian block (lower)
        # M_d = M[Bmin[IB + 1]:Bmax[IB + 1] + 1, Bmin[IB]:Bmax[IB] + 1].toarray()

        # # Extracting diagonal lesser Self-energy block
        # #SigL_c = SigL[Bmin[IB]:Bmax[IB] + 1, Bmin[IB]:Bmax[IB] + 1].toarray()

        # # Extracting off-diagonal lesser Self-energy block (lower)
        # SigL_l = SigL[Bmin[IB + 1]:Bmax[IB + 1] + 1, Bmin[IB]:Bmax[IB] + 1].toarray()

        # # Extracting diagonal greater Self-energy block
        # #SigG_c = SigG[Bmin[IB]:Bmax[IB] + 1, Bmin[IB]:Bmax[IB] + 1].toarray()

        # # Extracting off-diagonal greater Self-energy block (lower)
        # SigG_l = SigG[Bmin[IB + 1]:Bmax[IB + 1] + 1, Bmin[IB]:Bmax[IB] + 1].toarray()


        gR[IB, :, 0:NI, 0:NI] = np.linalg.inv(ham_diag[IB, :, 0:NN, 0:NN] \
                            - ham_upper[IB, :, 0:NI, 0:NP] \
                            @ gR[IB+1, :, 0:NP, 0:NP] \
                            @ ham_lower[IB, :, :NP, 0:NI])#######
        # AL, What is this? Handling off-diagonal sigma elements?
        AL = ham_upper[IB, :, 0:NI, 0:NP] \
            @ gR[IB+1, :, 0:NP, 0:NP] \
            @ sl_lower[IB, :, 0:NP, 0:NI]
        
        SigLB[IB, :, 0:NI, 0:NI] = ham_upper[IB, :, 0:NI, 0:NP] \
                            @ gL[IB+1, :, 0:NP, 0:NP] \
                            @ ham_upper[IB, :, 0:NI, 0:NP].transpose((0,2,1)).conj() \
                            - (AL - AL.transpose((0,2,1)).conj())

        # gL[IB, 0:NI, 0:NI] = gR[IB, 0:NI, 0:NI] \
        #                     @ (SigL_c \
        #                     + M_r \
        #                     @ gL[IB+1, 0:NP, 0:NP] \
        #                     @ M_r.T.conj() \
        #                     - (AL - AL.T.conj()))  \
        #                     @ gR[IB, 0:NI, 0:NI].T.conj() # Confused about the AL

        gL[IB, :, 0:NI, 0:NI] = gR[IB, :, 0:NI, 0:NI] \
                            @ (sl_diag[IB, :, 0:NI, 0:NI] \
                            + SigLB[IB, :, 0:NI, 0:NI])  \
                            @ gR[IB, :, 0:NI, 0:NI].transpose((0, 2, 1)).conj() # Confused about the AL

        ### What is this?
        AG = ham_upper[IB, :, 0:NI, 0:NP] \
            @ gR[IB+1, :, 0:NP, 0:NP] \
            @ sg_lower[IB, :, 0:NP, 0:NI]     # Handling off-diagonal sigma elements? Prob. need to check

        # gG[IB, 0:NI, 0:NI] = gR[IB, 0:NI, 0:NI] \
        #                     @ (SigG_c \
        #                     + M_r \
        #                     @ gG[IB+1, 0:NP, 0:NP] \
        #                     @ M_r.T.conj() \
        #                     - (AG - AG.T.conj())) \
        #                     @ gR[IB, 0:NI, 0:NI].T.conj() # Confused about the AG. 
        SigGB[IB, :, 0:NI, 0:NI] = ham_upper[IB, :, 0:NI, 0:NP] \
                            @ gG[IB+1, :,  0:NP, 0:NP] \
                            @ ham_upper[IB, :, 0:NI, 0:NP].transpose((0,2,1)).conj() \
                            - (AG - AG.transpose((0,2,1)).conj())

        gG[IB, 0:NI, 0:NI] = gR[IB, :, 0:NI, 0:NI] \
                            @ (sg_diag[IB, :, 0:NI, 0:NI] \
                                + SigGB[IB, :, 0:NI, 0:NI]) \
                                @ gR[IB, :, 0:NI, 0:NI].transpose((0,2,1)).conj() # Confused about the AG. 
    
    #Second step of iteration
    GR[0, :,  :NI, :NI] = gR[0, :, :NI, :NI]
    GRnn1[0, :,  :NI, :NP] = -GR[0, :, :NI, :NI] @ ham_upper[0, :, :NI, :NP] @ gR[1, :, :NP, :NP]

    GL[0, :, :NI, :NI] = gL[0, :, :NI, :NI]
    GLnn1[0, :, :NI, :NP] = GR[0, :, :NI, :NI] @ sl_upper[0, :, :NI, :NP] @ gR[1, :, :NP, :NP].transpose((0,2,1)).conj() \
                - GR[0,:, :NI,:NI] @ ham_upper[0, :, :NI, :NP] @ gL[1,:, :NP, :NP] \
                - GL[0,:, :NI,:NI] @ ham_lower[0, :, :NP, :NI].transpose((0,2,1)).conj() @ gR[1,:, :NP, :NP].transpose((0,2,1)).conj()

    GG[0, :, :NI, :NI] = gG[0, :, :NI, :NI]
    GGnn1[0, :, :NI, :NP] = GR[0, :,:NI, :NI] @ sg_upper[0, :, :NI, :NP] @ gR[1, :,:NP, :NP].transpose((0,2,1)).conj() \
                - GR[0,:, :NI,:NI] @ ham_upper[0, :, :NI, :NP] @ gG[1, :, :NP, :NP] \
                - GG[0,:,:NI,:NI] @ ham_lower[0, :, :NP, :NI].transpose((0,2,1)).conj() @ gR[1, :, :NP, :NP].transpose((0,2,1)).conj() 
    
    idE[:, 0] = np.real(np.trace(SigGB[0, :, :NI, :NI] @ GL[0, :, :NI, :NI] - GG[0, :, :NI, :NI] @ SigLB[0, :, :NI, :NI], axis1 = 1, axis2 = 2))

    for IB in range(1, NB):

        NM = Bmax[IB - 1] - Bmin[IB - 1] + 1
        NI = Bmax[IB] - Bmin[IB] + 1

        # # Extracting off-diagonal Hamiltonian block (upper)
        # M_u = M[Bmin[IB - 1]:Bmax[IB - 1] + 1, Bmin[IB]:Bmax[IB] + 1].toarray()

        # # # Extracting off-diagonal Hamiltonian block (left)
        # M_l = M[Bmin[IB]:Bmax[IB] + 1, Bmin[IB - 1]:Bmax[IB - 1] + 1].toarray()

        # # Extracting off-diagonal lesser Self-energy block (left)
        # SigL_l = SigL[Bmin[IB]:Bmax[IB] + 1, Bmin[IB - 1]:Bmax[IB - 1] + 1].toarray()

        # # Extracting off-diagonal greater Self-energy block (left)
        # SigG_l = SigG[Bmin[IB]:Bmax[IB] + 1, Bmin[IB - 1]:Bmax[IB - 1] + 1].toarray()

        GR[IB, :, :NI, :NI] = gR[IB, :, :NI, :NI] + gR[IB, :,  :NI, :NI] \
                        @ ham_lower[IB-1, :, :NI, :NM] \
                        @ GR[IB-1, :, 0:NM, 0:NM] \
                        @ ham_upper[IB-1, :, :NM, :NI] \
                        @ gR[IB, :,  :NI, :NI]
        # What is this? Handling off-diagonal elements?
        AL = gR[IB, :, :NI, :NI] \
            @ sl_lower[IB-1, :, :NI, :NM] \
            @ GR[IB-1, :, 0:NM, 0:NM].transpose((0,2,1)).conj() \
            @ ham_lower[IB-1, :, :NI, :NM].transpose((0,2,1)).conj() \
            @ gR[IB,:, 0:NI, 0:NI].transpose((0,2,1)).conj()
        # What is this?
        BL = gR[IB, :, :NI, :NI] \
            @ ham_lower[IB-1, :, :NI, :NM] \
            @ GR[IB-1, :, 0:NM, 0:NM] \
            @ ham_upper[IB-1, :, :NM, :NI] \
            @ gL[IB, :, :NI, :NI]

        GL[IB, :, 0:NI, 0:NI] = gL[IB, :, :NI, :NI] \
                            + gR[IB, :, 0:NI, 0:NI] \
                            @ ham_lower[IB-1, :, :NI, :NM] \
                            @ GL[IB-1, :, 0:NM, 0:NM] \
                            @ ham_lower[IB-1, :, :NI, :NM].transpose((0,2,1)).conj() \
                            @ gR[IB, :, :NI, :NI].transpose((0,2,1)).conj() \
                            - (AL - AL.transpose((0,2,1)).conj()) + (BL - BL.transpose((0,2,1)).conj())


        AG = gR[IB, :, 0:NI, 0:NI] \
            @ sg_lower[IB-1, :, :NI, :NM] \
            @ GR[IB-1, :, 0:NM, 0:NM].transpose((0,2,1)).conj() \
            @ ham_lower[IB-1, :, :NI, :NM].transpose((0,2,1)).conj() \
            @ gR[IB, :, :NI, :NI].transpose((0,2,1)).conj()

        BG = gR[IB, :, 0:NI, 0:NI] \
            @ ham_lower[IB-1, :, :NI, :NM] \
            @ GR[IB-1, :, 0:NM, 0:NM] \
            @ ham_upper[IB-1, :, :NM, :NI] \
            @ gG[IB, :, 0:NI, 0:NI]

        GG[IB, :, 0:NI, 0:NI] = gG[IB, :, 0:NI, 0:NI] \
                            + gR[IB, :, 0:NI, 0:NI] \
                            @ ham_lower[IB-1, :, :NI, :NM] \
                            @ GG[IB-1, :, 0:NM, 0:NM] \
                            @ ham_lower[IB-1, :, :NI, :NM].transpose((0,2,1)).conj() \
                            @ gR[IB, :, 0:NI, 0:NI].transpose((0,2,1)).conj() \
                            - (AG - AG.transpose((0,2,1)).conj()) + (BG - BG.transpose((0,2,1)).conj()) #

        if IB < NB - 1:  #Off-diagonal are only interesting for IdE!

            # # Extracting off-diagonal Hamiltonian block (right)
            # M_r = M[Bmin[IB]:Bmax[IB] + 1, Bmin[IB + 1]:Bmax[IB + 1] + 1].toarray()

            # # # Extracting off-diagonal Hamiltonian block (lower)
            # M_d = M[Bmin[IB + 1]:Bmax[IB + 1] + 1, Bmin[IB]:Bmax[IB] + 1].toarray()

            # # Extracting off-diagonal lesser Self-energy block (right)
            # SigL_r = SigL[Bmin[IB]:Bmax[IB] + 1, Bmin[IB + 1]:Bmax[IB + 1] + 1].toarray()

            # # Extracting off-diagonal greater Self-energy block (right)
            # SigG_r = SigG[Bmin[IB]:Bmax[IB] + 1, Bmin[IB + 1]:Bmax[IB + 1] + 1].toarray()

            NP = Bmax[IB + 1] - Bmin[IB + 1] + 1

            GRnn1[IB, :, 0:NI, 0:NP] = - GR[IB, :,  0:NI, 0:NI] \
                                    @ ham_upper[IB, :, :NM, :NI] \
                                    @ gR[IB+1, :, 0:NP, 0:NP]

            GLnn1[IB, :, 0:NI, 0:NP] = GR[IB, :, 0:NI, 0:NI] \
                                    @ sl_upper[IB, :, :NM, :NI] \
                                    @ gR[IB+1, :, 0:NP, 0:NP].transpose((0,2,1)).conj() \
                                    - GR[IB, :, 0:NI, 0:NI] \
                                    @ ham_upper[IB, :, :NM, :NI] \
                                    @ gL[IB+1, :, 0:NP, 0:NP] \
                                    - GL[IB, :, :NI, :NI] \
                                    @ ham_lower[IB, :, :NI, :NM].transpose((0,2,1)).conj() \
                                    @ gR[IB+1, :, 0:NP, 0:NP].transpose((0,2,1)).conj()
            GGnn1[IB, :, 0:NI, 0:NP] = GR[IB, :, 0:NI, 0:NI] \
                                    @ sg_upper[IB, :, :NM, :NI] \
                                    @ gR[IB+1, :, 0:NP, 0:NP].transpose((0,2,1)).conj() \
                                    - GR[IB, :, 0:NI, 0:NI] \
                                    @ ham_upper[IB, :, :NM, :NI]  \
                                    @ gG[IB+1, :, 0:NP, 0:NP] \
                                    - GG[IB, :, 0:NI, 0:NI] \
                                    @ ham_lower[IB, :, :NI, :NM].transpose((0,2,1)).conj() \
                                    @ gR[IB+1, :, 0:NP, 0:NP].transpose((0,2,1)).conj()
            idE[:, IB] = np.real(np.trace(SigGB[IB, :NI, :NI] @ GL[IB, :NI, :NI] - GG[IB, :NI, :NI] @ SigLB[IB, :NI, :NI], axis1 = 1, axis2 = 2))  
    for IB in range(NB):
        
        NI = Bmax[IB] - Bmin[IB] + 1
        # GR[IB, :, :, :] *= factor
        # GL[IB, :, :, :] *= factor
        # GG[IB, :, :, :] *= factor
        DOS[:, IB] = 1j * np.trace(GR[IB, :, :, :] - GR[IB, :, :, :].transpose((0,2,1)).conj(), axis1= 1, axis2 = 2)
        nE[:, IB] = -1j * np.trace(GL[IB, :, :, :], axis1= 1, axis2 = 2)
        nP[:, IB] = 1j * np.trace(GG[IB, :, :, :], axis1= 1, axis2 = 2)

        if IB < NB-1:
            NP = Bmax[IB+1] - Bmin[IB+1] + 1
            #idE[IB] = -2 * np.trace(np.real(H[Bmin[IB+1]:Bmax[IB+1]+1, Bmin[IB]:Bmax[IB]+1].toarray() @ GLnn1[IB, 0:NI, 0:NP]))
            # GRnn1[IB, :, :, :] *= factor
            # GLnn1[IB, :, :, :] *= factor
            # GGnn1[IB, :, :, :] *= factor

    
    #idE[NB - 1] = idE[NB - 2]
    idE[:, NB-1] = np.real(np.trace(SigGBR[:, :NI, :NI] @ GL[NB-1, :, :NI, :NI] - GG[NB-1, :, :NI, :NI] @ SigLBR[:, :NI, :NI], axis1 = 1, axis2 = 2))


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
           SigGBR,
           SigLBR,
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
           Bmax_fi
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

    # Upload to GPU
    ham_diag_gpu = cp.asarray(ham_diag)
    ham_upper_gpu = cp.asarray(ham_upper)
    ham_lower_gpu = cp.asarray(ham_lower)

    sg_diag_gpu = cp.asarray(sg_diag)
    sg_upper_gpu = cp.asarray(sg_upper)
    sg_lower_gpu = cp.asarray(sg_lower)

    sl_diag_gpu = cp.asarray(sl_diag)
    sl_upper_gpu = cp.asarray(sl_upper)
    sl_lower_gpu = cp.asarray(sl_lower)

    SigGBR_gpu = cp.asarray(SigGBR)
    SigLBR_gpu = cp.asarray(SigLBR)
    


    gR_gpu = cp.zeros((NB, Bsize, Bsize), dtype=cp.cfloat)  # Retarded (right)
    gL_gpu = cp.zeros((NB, Bsize, Bsize), dtype=cp.cfloat)  # Lesser (right)
    gG_gpu = cp.zeros((NB, Bsize, Bsize), dtype=cp.cfloat)  # Greater (right)
    SigLB_gpu = cp.zeros((NB - 1, Bsize, Bsize), dtype=cp.cfloat)  # Lesser boundary self-energy
    SigGB_gpu = cp.zeros((NB - 1, Bsize, Bsize), dtype=cp.cfloat)  # Greater boundary self-energy

    # IdE = np.zeros((NB, energy_batchsize))
    # n = np.zeros((NB, energy_batchsize))
    # p = np.zeros((NB, energy_batchsize))

    # First step of iteration
    NN = Bmax[-1] - Bmin[-1] + 1
    # gpu_identity = cp.identity(NN, dtype = cp.cfloat)
    # gpu_identity_batch = cp.repeat(gpu_identity[cp.newaxis, :, :], energy_batchsize, axis = 0)
    # gR_gpu[-1, :, 0:NN, 0:NN] = cp.linalg.solve(ham_diag_gpu[-1, :, 0:NN, 0:NN], gpu_identity_batch)
    gR_gpu[-1, 0:NN, 0:NN] = cp.linalg.inv(ham_diag_gpu[-1, 0:NN, 0:NN])
    gL_gpu[-1, 0:NN, 0:NN] = gR_gpu[-1, 0:NN, 0:NN] @ (sl_diag_gpu[-1, 0:NN, 0:NN]) @ gR_gpu[-1, 0:NN, 0:NN].conjugate().transpose()
    gG_gpu[-1, 0:NN, 0:NN] = gR_gpu[-1, 0:NN, 0:NN] @ (sg_diag_gpu[-1, 0:NN, 0:NN]) @ gR_gpu[-1, 0:NN, 0:NN].conjugate().transpose()

    for IB in range(NB - 2, -1, -1):
        NI = Bmax[IB] - Bmin[IB] + 1
        NP = Bmax[IB + 1] - Bmin[IB + 1] + 1

        # # Extracting diagonal Hamiltonian block
        # if(IB == 0):
        #     M_c = M[Bmin[IB]:Bmax[IB] + 1, Bmin[IB]:Bmax[IB] + 1].toarray() - SigRBL
        #     SigL_c = SigL[Bmin[IB]:Bmax[IB] + 1, Bmin[IB]:Bmax[IB] + 1].toarray() + SigLBL
        #     SigG_c = SigG[Bmin[IB]:Bmax[IB] + 1, Bmin[IB]:Bmax[IB] + 1].toarray() + SigGBL
        # else: 
        #     M_c = M[Bmin[IB]:Bmax[IB] + 1, Bmin[IB]:Bmax[IB] + 1].toarray()
        #     SigL_c = SigL[Bmin[IB]:Bmax[IB] + 1, Bmin[IB]:Bmax[IB] + 1].toarray()
        #     SigG_c = SigG[Bmin[IB]:Bmax[IB] + 1, Bmin[IB]:Bmax[IB] + 1].toarray()
        # #M_c = M[Bmin[IB]:Bmax[IB] + 1, Bmin[IB]:Bmax[IB] + 1].toarray()

        # # Extracting off-diagonal Hamiltonian block (right)
        # M_r = M[Bmin[IB]:Bmax[IB] + 1, Bmin[IB + 1]:Bmax[IB + 1] + 1].toarray()

        # # Extracting off-diagonal Hamiltonian block (lower)
        # M_d = M[Bmin[IB + 1]:Bmax[IB + 1] + 1, Bmin[IB]:Bmax[IB] + 1].toarray()

        # # Extracting diagonal lesser Self-energy block
        # #SigL_c = SigL[Bmin[IB]:Bmax[IB] + 1, Bmin[IB]:Bmax[IB] + 1].toarray()

        # # Extracting off-diagonal lesser Self-energy block (lower)
        # SigL_l = SigL[Bmin[IB + 1]:Bmax[IB + 1] + 1, Bmin[IB]:Bmax[IB] + 1].toarray()

        # # Extracting diagonal greater Self-energy block
        # #SigG_c = SigG[Bmin[IB]:Bmax[IB] + 1, Bmin[IB]:Bmax[IB] + 1].toarray()

        # # Extracting off-diagonal greater Self-energy block (lower)
        # SigG_l = SigG[Bmin[IB + 1]:Bmax[IB + 1] + 1, Bmin[IB]:Bmax[IB] + 1].toarray()

        # gpu_identity = cp.identity(NI, dtype = cp.cfloat)
        # gpu_identity_batch = cp.repeat(gpu_identity[cp.newaxis, :, :], energy_batchsize, axis = 0)
        gR_gpu[IB, 0:NI, 0:NI] = cp.linalg.inv(ham_diag_gpu[IB, 0:NN, 0:NN] \
                                                  - ham_upper_gpu[IB, 0:NI, 0:NP] \
                                                  @ gR_gpu[IB+1, 0:NP, 0:NP] \
                                                  @ ham_lower_gpu[IB, :NP, 0:NI])
        # gR_gpu[IB, :, 0:NI, 0:NI] = cp.linalg.solve(ham_diag_gpu[IB, :, 0:NN, 0:NN] \
        #                     - ham_upper_gpu[IB, :, 0:NI, 0:NP] \
        #                     @ gR_gpu[IB+1, :, 0:NP, 0:NP] \
        #                     @ ham_lower_gpu[IB, :, :NP, 0:NI], gpu_identity_batch)#######
        # AL, What is this? Handling off-diagonal sigma elements?
        AL = ham_upper_gpu[IB, 0:NI, 0:NP] \
            @ gR_gpu[IB+1, 0:NP, 0:NP] \
            @ sl_lower_gpu[IB, 0:NP, 0:NI]
        
        SigLB_gpu[IB, 0:NI, 0:NI] = ham_upper_gpu[IB, 0:NI, 0:NP] \
                            @ gL_gpu[IB+1, 0:NP, 0:NP] \
                            @ ham_upper_gpu[IB, 0:NI, 0:NP].transpose().conj() \
                            - (AL - AL.transpose().conj())

        # gL[IB, 0:NI, 0:NI] = gR[IB, 0:NI, 0:NI] \
        #                     @ (SigL_c \
        #                     + M_r \
        #                     @ gL[IB+1, 0:NP, 0:NP] \
        #                     @ M_r.T.conj() \
        #                     - (AL - AL.T.conj()))  \
        #                     @ gR[IB, 0:NI, 0:NI].T.conj() # Confused about the AL

        gL_gpu[IB, 0:NI, 0:NI] = gR_gpu[IB, 0:NI, 0:NI] \
                            @ (sl_diag_gpu[IB, 0:NI, 0:NI] \
                            + SigLB_gpu[IB, 0:NI, 0:NI])  \
                            @ gR_gpu[IB, 0:NI, 0:NI].transpose().conj() # Confused about the AL

        ### What is this?
        AG = ham_upper_gpu[IB, 0:NI, 0:NP] \
            @ gR_gpu[IB+1, 0:NP, 0:NP] \
            @ sg_lower_gpu[IB, 0:NP, 0:NI]     # Handling off-diagonal sigma elements? Prob. need to check

        # gG[IB, 0:NI, 0:NI] = gR[IB, 0:NI, 0:NI] \
        #                     @ (SigG_c \
        #                     + M_r \
        #                     @ gG[IB+1, 0:NP, 0:NP] \
        #                     @ M_r.T.conj() \
        #                     - (AG - AG.T.conj())) \
        #                     @ gR[IB, 0:NI, 0:NI].T.conj() # Confused about the AG. 
        SigGB_gpu[IB, 0:NI, 0:NI] = ham_upper_gpu[IB, 0:NI, 0:NP] \
                            @ gG_gpu[IB+1, 0:NP, 0:NP] \
                            @ ham_upper_gpu[IB, 0:NI, 0:NP].transpose().conj() \
                            - (AG - AG.transpose().conj())

        gG_gpu[IB, 0:NI, 0:NI] = gR_gpu[IB, 0:NI, 0:NI] \
                            @ (sg_diag_gpu[IB, 0:NI, 0:NI] \
                                + SigGB_gpu[IB, 0:NI, 0:NI]) \
                                @ gR_gpu[IB, 0:NI, 0:NI].transpose().conj() # Confused about the AG. 
    
    #Second step of iteration
    GR_gpu = cp.zeros_like(gR_gpu)
    GRnn1_gpu = cp.zeros_like(SigLB_gpu)
    GL_gpu = cp.zeros_like(gL_gpu)
    GLnn1_gpu = cp.zeros_like(SigLB_gpu)
    GG_gpu = cp.zeros_like(gG_gpu)
    GGnn1_gpu = cp.zeros_like(SigLB_gpu)



    GR_gpu[0, :NI, :NI] = gR_gpu[0, :NI, :NI]
    GRnn1_gpu[0, :NI, :NP] = -GR_gpu[0, :NI, :NI] @ ham_upper_gpu[0, :NI, :NP] @ gR_gpu[1, :NP, :NP]

    GL_gpu[0, :NI, :NI] = gL_gpu[0, :NI, :NI]
    GLnn1_gpu[0, :NI, :NP] = GR_gpu[0, :NI, :NI] @ sl_upper_gpu[0, :NI, :NP] @ gR_gpu[1, :NP, :NP].transpose().conj() \
                - GR_gpu[0, :NI,:NI] @ ham_upper_gpu[0, :NI, :NP] @ gL_gpu[1, :NP, :NP] \
                - GL_gpu[0, :NI,:NI] @ ham_lower_gpu[0, :NP, :NI].transpose().conj() @ gR_gpu[1, :NP, :NP].transpose().conj()

    GG_gpu[0, :NI, :NI] = gG_gpu[0, :NI, :NI]
    GGnn1_gpu[0, :NI, :NP] = GR_gpu[0, :NI, :NI] @ sg_upper_gpu[0, :NI, :NP] @ gR_gpu[1, :NP, :NP].transpose().conj() \
                - GR_gpu[0, :NI,:NI] @ ham_upper_gpu[0, :NI, :NP] @ gG_gpu[1, :NP, :NP] \
                - GG_gpu[0, :NI,:NI] @ ham_lower_gpu[0, :NP, :NI].transpose().conj() @ gR_gpu[1, :NP, :NP].transpose().conj() 
    
    idE[0] = cp.real(cp.trace(SigGB_gpu[0, :NI, :NI] @ GL_gpu[0, :NI, :NI] - GG_gpu[0, :NI, :NI] @ SigLB_gpu[0, :NI, :NI])).get()

    for IB in range(1, NB):

        NM = Bmax[IB - 1] - Bmin[IB - 1] + 1
        NI = Bmax[IB] - Bmin[IB] + 1

        # # Extracting off-diagonal Hamiltonian block (upper)
        # M_u = M[Bmin[IB - 1]:Bmax[IB - 1] + 1, Bmin[IB]:Bmax[IB] + 1].toarray()

        # # # Extracting off-diagonal Hamiltonian block (left)
        # M_l = M[Bmin[IB]:Bmax[IB] + 1, Bmin[IB - 1]:Bmax[IB - 1] + 1].toarray()

        # # Extracting off-diagonal lesser Self-energy block (left)
        # SigL_l = SigL[Bmin[IB]:Bmax[IB] + 1, Bmin[IB - 1]:Bmax[IB - 1] + 1].toarray()

        # # Extracting off-diagonal greater Self-energy block (left)
        # SigG_l = SigG[Bmin[IB]:Bmax[IB] + 1, Bmin[IB - 1]:Bmax[IB - 1] + 1].toarray()

        GR_gpu[IB, :NI, :NI] = gR_gpu[IB, :NI, :NI] + gR_gpu[IB, :NI, :NI] \
                        @ ham_lower_gpu[IB-1, :NI, :NM] \
                        @ GR_gpu[IB-1, 0:NM, 0:NM] \
                        @ ham_upper_gpu[IB-1, :NM, :NI] \
                        @ gR_gpu[IB, :NI, :NI]
        # What is this? Handling off-diagonal elements?
        AL = gR_gpu[IB, :NI, :NI] \
            @ sl_lower_gpu[IB-1, :NI, :NM] \
            @ GR_gpu[IB-1, 0:NM, 0:NM].transpose().conj() \
            @ ham_lower_gpu[IB-1, :NI, :NM].transpose().conj() \
            @ gR_gpu[IB, 0:NI, 0:NI].transpose().conj()
        # What is this?
        BL = gR_gpu[IB, :NI, :NI] \
            @ ham_lower_gpu[IB-1, :NI, :NM] \
            @ GR_gpu[IB-1, 0:NM, 0:NM] \
            @ ham_upper_gpu[IB-1, :NM, :NI] \
            @ gL_gpu[IB, :NI, :NI]

        GL_gpu[IB, 0:NI, 0:NI] = gL_gpu[IB, :NI, :NI] \
                            + gR_gpu[IB, 0:NI, 0:NI] \
                            @ ham_lower_gpu[IB-1, :NI, :NM] \
                            @ GL_gpu[IB-1, 0:NM, 0:NM] \
                            @ ham_lower_gpu[IB-1, :NI, :NM].transpose().conj() \
                            @ gR_gpu[IB, :NI, :NI].transpose().conj() \
                            - (AL - AL.transpose().conj()) + (BL - BL.transpose().conj())


        AG = gR_gpu[IB, 0:NI, 0:NI] \
            @ sg_lower_gpu[IB-1, :NI, :NM] \
            @ GR_gpu[IB-1, 0:NM, 0:NM].transpose().conj() \
            @ ham_lower_gpu[IB-1, :NI, :NM].transpose().conj() \
            @ gR_gpu[IB, :NI, :NI].transpose().conj()

        BG = gR_gpu[IB, 0:NI, 0:NI] \
            @ ham_lower_gpu[IB-1, :NI, :NM] \
            @ GR_gpu[IB-1, 0:NM, 0:NM] \
            @ ham_upper_gpu[IB-1, :NM, :NI] \
            @ gG_gpu[IB, 0:NI, 0:NI]

        GG_gpu[IB, 0:NI, 0:NI] = gG_gpu[IB, 0:NI, 0:NI] \
                            + gR_gpu[IB, 0:NI, 0:NI] \
                            @ ham_lower_gpu[IB-1, :NI, :NM] \
                            @ GG_gpu[IB-1, 0:NM, 0:NM] \
                            @ ham_lower_gpu[IB-1, :NI, :NM].transpose().conj() \
                            @ gR_gpu[IB, 0:NI, 0:NI].transpose().conj() \
                            - (AG - AG.transpose().conj()) + (BG - BG.transpose().conj()) #

        if IB < NB - 1:  #Off-diagonal are only interesting for IdE!

            # # Extracting off-diagonal Hamiltonian block (right)
            # M_r = M[Bmin[IB]:Bmax[IB] + 1, Bmin[IB + 1]:Bmax[IB + 1] + 1].toarray()

            # # # Extracting off-diagonal Hamiltonian block (lower)
            # M_d = M[Bmin[IB + 1]:Bmax[IB + 1] + 1, Bmin[IB]:Bmax[IB] + 1].toarray()

            # # Extracting off-diagonal lesser Self-energy block (right)
            # SigL_r = SigL[Bmin[IB]:Bmax[IB] + 1, Bmin[IB + 1]:Bmax[IB + 1] + 1].toarray()

            # # Extracting off-diagonal greater Self-energy block (right)
            # SigG_r = SigG[Bmin[IB]:Bmax[IB] + 1, Bmin[IB + 1]:Bmax[IB + 1] + 1].toarray()

            NP = Bmax[IB + 1] - Bmin[IB + 1] + 1

            GRnn1_gpu[IB, 0:NI, 0:NP] = - GR_gpu[IB, 0:NI, 0:NI] \
                                    @ ham_upper_gpu[IB, :NM, :NI] \
                                    @ gR_gpu[IB+1, 0:NP, 0:NP]

            GLnn1_gpu[IB, 0:NI, 0:NP] = GR_gpu[IB, 0:NI, 0:NI] \
                                    @ sl_upper_gpu[IB, :NM, :NI] \
                                    @ gR_gpu[IB+1, 0:NP, 0:NP].transpose().conj() \
                                    - GR_gpu[IB, 0:NI, 0:NI] \
                                    @ ham_upper_gpu[IB, :NM, :NI] \
                                    @ gL_gpu[IB+1, 0:NP, 0:NP] \
                                    - GL_gpu[IB, :NI, :NI] \
                                    @ ham_lower_gpu[IB, :NI, :NM].transpose().conj() \
                                    @ gR_gpu[IB+1, 0:NP, 0:NP].transpose().conj()
            GGnn1_gpu[IB, 0:NI, 0:NP] = GR_gpu[IB, 0:NI, 0:NI] \
                                    @ sg_upper_gpu[IB, :NM, :NI] \
                                    @ gR_gpu[IB+1, 0:NP, 0:NP].transpose().conj() \
                                    - GR_gpu[IB, 0:NI, 0:NI] \
                                    @ ham_upper_gpu[IB, :NM, :NI]  \
                                    @ gG_gpu[IB+1, 0:NP, 0:NP] \
                                    - GG_gpu[IB, 0:NI, 0:NI] \
                                    @ ham_lower_gpu[IB, :NI, :NM].transpose().conj() \
                                    @ gR_gpu[IB+1, 0:NP, 0:NP].transpose().conj()
            idE[IB] = cp.real(cp.trace(SigGB_gpu[IB, :NI, :NI] @ GL_gpu[IB, :NI, :NI] - GG_gpu[IB, :NI, :NI] @ SigLB_gpu[IB, :NI, :NI])).get() 
    for IB in range(NB):
        
        NI = Bmax[IB] - Bmin[IB] + 1
        # GR[IB, :, :, :] *= factor
        # GL[IB, :, :, :] *= factor
        # GG[IB, :, :, :] *= factor
        DOS[IB] = 1j * cp.trace(GR_gpu[IB, :, :] - GR_gpu[IB, :, :].transpose().conj()).get()
        nE[IB] = -1j * cp.trace(GL_gpu[IB, :, :]).get()
        nP[IB] = 1j * cp.trace(GG_gpu[IB, :, :]).get()

        # if IB < NB-1:
        #     NP = Bmax[IB+1] - Bmin[IB+1] + 1
        #     #idE[IB] = -2 * np.trace(np.real(H[Bmin[IB+1]:Bmax[IB+1]+1, Bmin[IB]:Bmax[IB]+1].toarray() @ GLnn1[IB, 0:NI, 0:NP]))
        #     # GRnn1[IB, :, :, :] *= factor
        #     # GLnn1[IB, :, :, :] *= factor
        #     # GGnn1[IB, :, :, :] *= factor

    
    #idE[NB - 1] = idE[NB - 2]
    idE[NB-1] = cp.real(cp.trace(SigGBR_gpu[:NI, :NI] @ GL_gpu[NB-1, :NI, :NI] - GG_gpu[NB-1, :NI, :NI] @ SigLBR_gpu[:NI, :NI])).get()

    #Final Data Transfer
    #GR[:, :, :, :] = GR_gpu.get()
    GL[:, :, :] = GL_gpu.get()
    GG[:, :, :] = GG_gpu.get()
    #GRnn1[:, :, :, :] = GRnn1_gpu.get()
    GLnn1[:, :, :] = GLnn1_gpu.get()
    GGnn1[:, :, :] = GGnn1_gpu.get()


def rgf_standaloneGF_batched_GPU(
           ham_diag,
           ham_upper,
           ham_lower,
           sg_diag,
           sg_upper,
           sg_lower,
           sl_diag,
           sl_upper,
           sl_lower,
           SigGBR,
           SigLBR,
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
           Bmax_fi
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

    # Upload to GPU
    ham_diag_gpu = cp.asarray(ham_diag)
    ham_upper_gpu = cp.asarray(ham_upper)
    ham_lower_gpu = cp.asarray(ham_lower)

    sg_diag_gpu = cp.asarray(sg_diag)
    sg_upper_gpu = cp.asarray(sg_upper)
    sg_lower_gpu = cp.asarray(sg_lower)

    sl_diag_gpu = cp.asarray(sl_diag)
    sl_upper_gpu = cp.asarray(sl_upper)
    sl_lower_gpu = cp.asarray(sl_lower)

    SigGBR_gpu = cp.asarray(SigGBR)
    SigLBR_gpu = cp.asarray(SigLBR)
    


    gR_gpu = cp.zeros((NB, energy_batchsize, Bsize, Bsize), dtype=cp.cfloat)  # Retarded (right)
    gL_gpu = cp.zeros((NB, energy_batchsize, Bsize, Bsize), dtype=cp.cfloat)  # Lesser (right)
    gG_gpu = cp.zeros((NB, energy_batchsize, Bsize, Bsize), dtype=cp.cfloat)  # Greater (right)
    SigLB_gpu = cp.zeros((NB - 1, energy_batchsize, Bsize, Bsize), dtype=cp.cfloat)  # Lesser boundary self-energy
    SigGB_gpu = cp.zeros((NB - 1, energy_batchsize, Bsize, Bsize), dtype=cp.cfloat)  # Greater boundary self-energy

    # IdE = np.zeros((NB, energy_batchsize))
    # n = np.zeros((NB, energy_batchsize))
    # p = np.zeros((NB, energy_batchsize))

    # First step of iteration
    NN = Bmax[-1] - Bmin[-1] + 1
    # gpu_identity = cp.identity(NN, dtype = cp.cfloat)
    # gpu_identity_batch = cp.repeat(gpu_identity[cp.newaxis, :, :], energy_batchsize, axis = 0)
    # gR_gpu[-1, :, 0:NN, 0:NN] = cp.linalg.solve(ham_diag_gpu[-1, :, 0:NN, 0:NN], gpu_identity_batch)
    gR_gpu[-1, :, 0:NN, 0:NN] = cp.linalg.inv(ham_diag_gpu[-1, :, 0:NN, 0:NN])
    gL_gpu[-1, :, 0:NN, 0:NN] = gR_gpu[-1, :, 0:NN, 0:NN] @ (sl_diag_gpu[-1, :, 0:NN, 0:NN]) @ gR_gpu[-1, :, 0:NN, 0:NN].conjugate().transpose((0,2,1))
    gG_gpu[-1, :, 0:NN, 0:NN] = gR_gpu[-1, :, 0:NN, 0:NN] @ (sg_diag_gpu[-1, :, 0:NN, 0:NN]) @ gR_gpu[-1, :, 0:NN, 0:NN].conjugate().transpose((0,2,1))

    for IB in range(NB - 2, -1, -1):
        NI = Bmax[IB] - Bmin[IB] + 1
        NP = Bmax[IB + 1] - Bmin[IB + 1] + 1

        # # Extracting diagonal Hamiltonian block
        # if(IB == 0):
        #     M_c = M[Bmin[IB]:Bmax[IB] + 1, Bmin[IB]:Bmax[IB] + 1].toarray() - SigRBL
        #     SigL_c = SigL[Bmin[IB]:Bmax[IB] + 1, Bmin[IB]:Bmax[IB] + 1].toarray() + SigLBL
        #     SigG_c = SigG[Bmin[IB]:Bmax[IB] + 1, Bmin[IB]:Bmax[IB] + 1].toarray() + SigGBL
        # else: 
        #     M_c = M[Bmin[IB]:Bmax[IB] + 1, Bmin[IB]:Bmax[IB] + 1].toarray()
        #     SigL_c = SigL[Bmin[IB]:Bmax[IB] + 1, Bmin[IB]:Bmax[IB] + 1].toarray()
        #     SigG_c = SigG[Bmin[IB]:Bmax[IB] + 1, Bmin[IB]:Bmax[IB] + 1].toarray()
        # #M_c = M[Bmin[IB]:Bmax[IB] + 1, Bmin[IB]:Bmax[IB] + 1].toarray()

        # # Extracting off-diagonal Hamiltonian block (right)
        # M_r = M[Bmin[IB]:Bmax[IB] + 1, Bmin[IB + 1]:Bmax[IB + 1] + 1].toarray()

        # # Extracting off-diagonal Hamiltonian block (lower)
        # M_d = M[Bmin[IB + 1]:Bmax[IB + 1] + 1, Bmin[IB]:Bmax[IB] + 1].toarray()

        # # Extracting diagonal lesser Self-energy block
        # #SigL_c = SigL[Bmin[IB]:Bmax[IB] + 1, Bmin[IB]:Bmax[IB] + 1].toarray()

        # # Extracting off-diagonal lesser Self-energy block (lower)
        # SigL_l = SigL[Bmin[IB + 1]:Bmax[IB + 1] + 1, Bmin[IB]:Bmax[IB] + 1].toarray()

        # # Extracting diagonal greater Self-energy block
        # #SigG_c = SigG[Bmin[IB]:Bmax[IB] + 1, Bmin[IB]:Bmax[IB] + 1].toarray()

        # # Extracting off-diagonal greater Self-energy block (lower)
        # SigG_l = SigG[Bmin[IB + 1]:Bmax[IB + 1] + 1, Bmin[IB]:Bmax[IB] + 1].toarray()

        # gpu_identity = cp.identity(NI, dtype = cp.cfloat)
        # gpu_identity_batch = cp.repeat(gpu_identity[cp.newaxis, :, :], energy_batchsize, axis = 0)
        gR_gpu[IB, :, 0:NI, 0:NI] = cp.linalg.inv(ham_diag_gpu[IB, :, 0:NN, 0:NN] \
                                                  - ham_upper_gpu[IB, :, 0:NI, 0:NP] \
                                                  @ gR_gpu[IB+1, :, 0:NP, 0:NP] \
                                                  @ ham_lower_gpu[IB, :, :NP, 0:NI])
        # gR_gpu[IB, :, 0:NI, 0:NI] = cp.linalg.solve(ham_diag_gpu[IB, :, 0:NN, 0:NN] \
        #                     - ham_upper_gpu[IB, :, 0:NI, 0:NP] \
        #                     @ gR_gpu[IB+1, :, 0:NP, 0:NP] \
        #                     @ ham_lower_gpu[IB, :, :NP, 0:NI], gpu_identity_batch)#######
        # AL, What is this? Handling off-diagonal sigma elements?
        AL = ham_upper_gpu[IB, :, 0:NI, 0:NP] \
            @ gR_gpu[IB+1, :, 0:NP, 0:NP] \
            @ sl_lower_gpu[IB, :, 0:NP, 0:NI]
        
        SigLB_gpu[IB, :, 0:NI, 0:NI] = ham_upper_gpu[IB, :, 0:NI, 0:NP] \
                            @ gL_gpu[IB+1, :, 0:NP, 0:NP] \
                            @ ham_upper_gpu[IB, :, 0:NI, 0:NP].transpose((0,2,1)).conj() \
                            - (AL - AL.transpose((0,2,1)).conj())

        # gL[IB, 0:NI, 0:NI] = gR[IB, 0:NI, 0:NI] \
        #                     @ (SigL_c \
        #                     + M_r \
        #                     @ gL[IB+1, 0:NP, 0:NP] \
        #                     @ M_r.T.conj() \
        #                     - (AL - AL.T.conj()))  \
        #                     @ gR[IB, 0:NI, 0:NI].T.conj() # Confused about the AL

        gL_gpu[IB, :, 0:NI, 0:NI] = gR_gpu[IB, :, 0:NI, 0:NI] \
                            @ (sl_diag_gpu[IB, :, 0:NI, 0:NI] \
                            + SigLB_gpu[IB, :, 0:NI, 0:NI])  \
                            @ gR_gpu[IB, :, 0:NI, 0:NI].transpose((0, 2, 1)).conj() # Confused about the AL

        ### What is this?
        AG = ham_upper_gpu[IB, :, 0:NI, 0:NP] \
            @ gR_gpu[IB+1, :, 0:NP, 0:NP] \
            @ sg_lower_gpu[IB, :, 0:NP, 0:NI]     # Handling off-diagonal sigma elements? Prob. need to check

        # gG[IB, 0:NI, 0:NI] = gR[IB, 0:NI, 0:NI] \
        #                     @ (SigG_c \
        #                     + M_r \
        #                     @ gG[IB+1, 0:NP, 0:NP] \
        #                     @ M_r.T.conj() \
        #                     - (AG - AG.T.conj())) \
        #                     @ gR[IB, 0:NI, 0:NI].T.conj() # Confused about the AG. 
        SigGB_gpu[IB, :, 0:NI, 0:NI] = ham_upper_gpu[IB, :, 0:NI, 0:NP] \
                            @ gG_gpu[IB+1, :,  0:NP, 0:NP] \
                            @ ham_upper_gpu[IB, :, 0:NI, 0:NP].transpose((0,2,1)).conj() \
                            - (AG - AG.transpose((0,2,1)).conj())

        gG_gpu[IB, :, 0:NI, 0:NI] = gR_gpu[IB, :, 0:NI, 0:NI] \
                            @ (sg_diag_gpu[IB, :, 0:NI, 0:NI] \
                                + SigGB_gpu[IB, :, 0:NI, 0:NI]) \
                                @ gR_gpu[IB, :, 0:NI, 0:NI].transpose((0,2,1)).conj() # Confused about the AG. 
    
    #Second step of iteration
    GR_gpu = cp.zeros_like(gR_gpu)
    GRnn1_gpu = cp.zeros_like(SigLB_gpu)
    GL_gpu = cp.zeros_like(gL_gpu)
    GLnn1_gpu = cp.zeros_like(SigLB_gpu)
    GG_gpu = cp.zeros_like(gG_gpu)
    GGnn1_gpu = cp.zeros_like(SigLB_gpu)



    GR_gpu[0, :,  :NI, :NI] = gR_gpu[0, :, :NI, :NI]
    GRnn1_gpu[0, :,  :NI, :NP] = -GR_gpu[0, :, :NI, :NI] @ ham_upper_gpu[0, :, :NI, :NP] @ gR_gpu[1, :, :NP, :NP]

    GL_gpu[0, :, :NI, :NI] = gL_gpu[0, :, :NI, :NI]
    GLnn1_gpu[0, :, :NI, :NP] = GR_gpu[0, :, :NI, :NI] @ sl_upper_gpu[0, :, :NI, :NP] @ gR_gpu[1, :, :NP, :NP].transpose((0,2,1)).conj() \
                - GR_gpu[0,:, :NI,:NI] @ ham_upper_gpu[0, :, :NI, :NP] @ gL_gpu[1,:, :NP, :NP] \
                - GL_gpu[0,:, :NI,:NI] @ ham_lower_gpu[0, :, :NP, :NI].transpose((0,2,1)).conj() @ gR_gpu[1,:, :NP, :NP].transpose((0,2,1)).conj()

    GG_gpu[0, :, :NI, :NI] = gG_gpu[0, :, :NI, :NI]
    GGnn1_gpu[0, :, :NI, :NP] = GR_gpu[0, :,:NI, :NI] @ sg_upper_gpu[0, :, :NI, :NP] @ gR_gpu[1, :,:NP, :NP].transpose((0,2,1)).conj() \
                - GR_gpu[0,:, :NI,:NI] @ ham_upper_gpu[0, :, :NI, :NP] @ gG_gpu[1, :, :NP, :NP] \
                - GG_gpu[0,:,:NI,:NI] @ ham_lower_gpu[0, :, :NP, :NI].transpose((0,2,1)).conj() @ gR_gpu[1, :, :NP, :NP].transpose((0,2,1)).conj() 
    
    idE[:, 0] = cp.real(cp.trace(SigGB_gpu[0, :, :NI, :NI] @ GL_gpu[0, :, :NI, :NI] - GG_gpu[0, :, :NI, :NI] @ SigLB_gpu[0, :, :NI, :NI], axis1 = 1, axis2 = 2)).get()

    for IB in range(1, NB):

        NM = Bmax[IB - 1] - Bmin[IB - 1] + 1
        NI = Bmax[IB] - Bmin[IB] + 1

        # # Extracting off-diagonal Hamiltonian block (upper)
        # M_u = M[Bmin[IB - 1]:Bmax[IB - 1] + 1, Bmin[IB]:Bmax[IB] + 1].toarray()

        # # # Extracting off-diagonal Hamiltonian block (left)
        # M_l = M[Bmin[IB]:Bmax[IB] + 1, Bmin[IB - 1]:Bmax[IB - 1] + 1].toarray()

        # # Extracting off-diagonal lesser Self-energy block (left)
        # SigL_l = SigL[Bmin[IB]:Bmax[IB] + 1, Bmin[IB - 1]:Bmax[IB - 1] + 1].toarray()

        # # Extracting off-diagonal greater Self-energy block (left)
        # SigG_l = SigG[Bmin[IB]:Bmax[IB] + 1, Bmin[IB - 1]:Bmax[IB - 1] + 1].toarray()

        GR_gpu[IB, :, :NI, :NI] = gR_gpu[IB, :, :NI, :NI] + gR_gpu[IB, :,  :NI, :NI] \
                        @ ham_lower_gpu[IB-1, :, :NI, :NM] \
                        @ GR_gpu[IB-1, :, 0:NM, 0:NM] \
                        @ ham_upper_gpu[IB-1, :, :NM, :NI] \
                        @ gR_gpu[IB, :,  :NI, :NI]
        # What is this? Handling off-diagonal elements?
        AL = gR_gpu[IB, :, :NI, :NI] \
            @ sl_lower_gpu[IB-1, :, :NI, :NM] \
            @ GR_gpu[IB-1, :, 0:NM, 0:NM].transpose((0,2,1)).conj() \
            @ ham_lower_gpu[IB-1, :, :NI, :NM].transpose((0,2,1)).conj() \
            @ gR_gpu[IB,:, 0:NI, 0:NI].transpose((0,2,1)).conj()
        # What is this?
        BL = gR_gpu[IB, :, :NI, :NI] \
            @ ham_lower_gpu[IB-1, :, :NI, :NM] \
            @ GR_gpu[IB-1, :, 0:NM, 0:NM] \
            @ ham_upper_gpu[IB-1, :, :NM, :NI] \
            @ gL_gpu[IB, :, :NI, :NI]

        GL_gpu[IB, :, 0:NI, 0:NI] = gL_gpu[IB, :, :NI, :NI] \
                            + gR_gpu[IB, :, 0:NI, 0:NI] \
                            @ ham_lower_gpu[IB-1, :, :NI, :NM] \
                            @ GL_gpu[IB-1, :, 0:NM, 0:NM] \
                            @ ham_lower_gpu[IB-1, :, :NI, :NM].transpose((0,2,1)).conj() \
                            @ gR_gpu[IB, :, :NI, :NI].transpose((0,2,1)).conj() \
                            - (AL - AL.transpose((0,2,1)).conj()) + (BL - BL.transpose((0,2,1)).conj())


        AG = gR_gpu[IB, :, 0:NI, 0:NI] \
            @ sg_lower_gpu[IB-1, :, :NI, :NM] \
            @ GR_gpu[IB-1, :, 0:NM, 0:NM].transpose((0,2,1)).conj() \
            @ ham_lower_gpu[IB-1, :, :NI, :NM].transpose((0,2,1)).conj() \
            @ gR_gpu[IB, :, :NI, :NI].transpose((0,2,1)).conj()

        BG = gR_gpu[IB, :, 0:NI, 0:NI] \
            @ ham_lower_gpu[IB-1, :, :NI, :NM] \
            @ GR_gpu[IB-1, :, 0:NM, 0:NM] \
            @ ham_upper_gpu[IB-1, :, :NM, :NI] \
            @ gG_gpu[IB, :, 0:NI, 0:NI]

        GG_gpu[IB, :, 0:NI, 0:NI] = gG_gpu[IB, :, 0:NI, 0:NI] \
                            + gR_gpu[IB, :, 0:NI, 0:NI] \
                            @ ham_lower_gpu[IB-1, :, :NI, :NM] \
                            @ GG_gpu[IB-1, :, 0:NM, 0:NM] \
                            @ ham_lower_gpu[IB-1, :, :NI, :NM].transpose((0,2,1)).conj() \
                            @ gR_gpu[IB, :, 0:NI, 0:NI].transpose((0,2,1)).conj() \
                            - (AG - AG.transpose((0,2,1)).conj()) + (BG - BG.transpose((0,2,1)).conj()) #

        if IB < NB - 1:  #Off-diagonal are only interesting for IdE!

            # # Extracting off-diagonal Hamiltonian block (right)
            # M_r = M[Bmin[IB]:Bmax[IB] + 1, Bmin[IB + 1]:Bmax[IB + 1] + 1].toarray()

            # # # Extracting off-diagonal Hamiltonian block (lower)
            # M_d = M[Bmin[IB + 1]:Bmax[IB + 1] + 1, Bmin[IB]:Bmax[IB] + 1].toarray()

            # # Extracting off-diagonal lesser Self-energy block (right)
            # SigL_r = SigL[Bmin[IB]:Bmax[IB] + 1, Bmin[IB + 1]:Bmax[IB + 1] + 1].toarray()

            # # Extracting off-diagonal greater Self-energy block (right)
            # SigG_r = SigG[Bmin[IB]:Bmax[IB] + 1, Bmin[IB + 1]:Bmax[IB + 1] + 1].toarray()

            NP = Bmax[IB + 1] - Bmin[IB + 1] + 1

            GRnn1_gpu[IB, :, 0:NI, 0:NP] = - GR_gpu[IB, :,  0:NI, 0:NI] \
                                    @ ham_upper_gpu[IB, :, :NM, :NI] \
                                    @ gR_gpu[IB+1, :, 0:NP, 0:NP]

            GLnn1_gpu[IB, :, 0:NI, 0:NP] = GR_gpu[IB, :, 0:NI, 0:NI] \
                                    @ sl_upper_gpu[IB, :, :NM, :NI] \
                                    @ gR_gpu[IB+1, :, 0:NP, 0:NP].transpose((0,2,1)).conj() \
                                    - GR_gpu[IB, :, 0:NI, 0:NI] \
                                    @ ham_upper_gpu[IB, :, :NM, :NI] \
                                    @ gL_gpu[IB+1, :, 0:NP, 0:NP] \
                                    - GL_gpu[IB, :, :NI, :NI] \
                                    @ ham_lower_gpu[IB, :, :NI, :NM].transpose((0,2,1)).conj() \
                                    @ gR_gpu[IB+1, :, 0:NP, 0:NP].transpose((0,2,1)).conj()
            GGnn1_gpu[IB, :, 0:NI, 0:NP] = GR_gpu[IB, :, 0:NI, 0:NI] \
                                    @ sg_upper_gpu[IB, :, :NM, :NI] \
                                    @ gR_gpu[IB+1, :, 0:NP, 0:NP].transpose((0,2,1)).conj() \
                                    - GR_gpu[IB, :, 0:NI, 0:NI] \
                                    @ ham_upper_gpu[IB, :, :NM, :NI]  \
                                    @ gG_gpu[IB+1, :, 0:NP, 0:NP] \
                                    - GG_gpu[IB, :, 0:NI, 0:NI] \
                                    @ ham_lower_gpu[IB, :, :NI, :NM].transpose((0,2,1)).conj() \
                                    @ gR_gpu[IB+1, :, 0:NP, 0:NP].transpose((0,2,1)).conj()
            idE[:, IB] = cp.real(cp.trace(SigGB_gpu[IB, :NI, :NI] @ GL_gpu[IB, :NI, :NI] - GG_gpu[IB, :NI, :NI] @ SigLB_gpu[IB, :NI, :NI], axis1 = 1, axis2 = 2)).get() 
    for IB in range(NB):
        
        NI = Bmax[IB] - Bmin[IB] + 1
        # GR[IB, :, :, :] *= factor
        # GL[IB, :, :, :] *= factor
        # GG[IB, :, :, :] *= factor
        DOS[:, IB] = 1j * cp.trace(GR_gpu[IB, :, :, :] - GR_gpu[IB, :, :, :].transpose((0,2,1)).conj(), axis1= 1, axis2 = 2).get()
        nE[:, IB] = -1j * cp.trace(GL_gpu[IB, :, :, :], axis1= 1, axis2 = 2).get()
        nP[:, IB] = 1j * cp.trace(GG_gpu[IB, :, :, :], axis1= 1, axis2 = 2).get()

        # if IB < NB-1:
        #     NP = Bmax[IB+1] - Bmin[IB+1] + 1
        #     #idE[IB] = -2 * np.trace(np.real(H[Bmin[IB+1]:Bmax[IB+1]+1, Bmin[IB]:Bmax[IB]+1].toarray() @ GLnn1[IB, 0:NI, 0:NP]))
        #     # GRnn1[IB, :, :, :] *= factor
        #     # GLnn1[IB, :, :, :] *= factor
        #     # GGnn1[IB, :, :, :] *= factor

    
    #idE[NB - 1] = idE[NB - 2]
    idE[:, NB-1] = cp.real(cp.trace(SigGBR_gpu[:, :NI, :NI] @ GL_gpu[NB-1, :, :NI, :NI] - GG_gpu[NB-1, :, :NI, :NI] @ SigLBR_gpu[:, :NI, :NI], axis1 = 1, axis2 = 2)).get()

    #Final Data Transfer
    #GR[:, :, :, :] = GR_gpu.get()
    GL[:, :, :, :] = GL_gpu.get()
    GG[:, :, :, :] = GG_gpu.get()
    #GRnn1[:, :, :, :] = GRnn1_gpu.get()
    GLnn1[:, :, :, :] = GLnn1_gpu.get()
    GGnn1[:, :, :, :] = GGnn1_gpu.get()


def rgf_standaloneGF_batched_GPU_part1(
           ham_diag,
           ham_upper,
           ham_lower,
           sg_diag,
           sg_upper,
           sg_lower,
           sl_diag,
           sl_upper,
           sl_lower,
           SigGBR,
           SigLBR,
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
           Bmax_fi
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

    # Upload to GPU
    ham_diag_gpu = cp.asarray(ham_diag)
    ham_upper_gpu = cp.asarray(ham_upper)
    ham_lower_gpu = cp.asarray(ham_lower)

    sg_diag_gpu = cp.asarray(sg_diag)
    sg_upper_gpu = cp.asarray(sg_upper)
    sg_lower_gpu = cp.asarray(sg_lower)

    sl_diag_gpu = cp.asarray(sl_diag)
    sl_upper_gpu = cp.asarray(sl_upper)
    sl_lower_gpu = cp.asarray(sl_lower)

    SigGBR_gpu = cp.asarray(SigGBR)
    SigLBR_gpu = cp.asarray(SigLBR)
    


    gR_gpu = cp.zeros((NB, energy_batchsize, Bsize, Bsize), dtype=cp.cfloat)  # Retarded (right)
    gL_gpu = cp.zeros((NB, energy_batchsize, Bsize, Bsize), dtype=cp.cfloat)  # Lesser (right)
    gG_gpu = cp.zeros((NB, energy_batchsize, Bsize, Bsize), dtype=cp.cfloat)  # Greater (right)
    SigLB_gpu = cp.zeros((NB - 1, energy_batchsize, Bsize, Bsize), dtype=cp.cfloat)  # Lesser boundary self-energy
    SigGB_gpu = cp.zeros((NB - 1, energy_batchsize, Bsize, Bsize), dtype=cp.cfloat)  # Greater boundary self-energy

    # IdE = np.zeros((NB, energy_batchsize))
    # n = np.zeros((NB, energy_batchsize))
    # p = np.zeros((NB, energy_batchsize))

    # First step of iteration
    NN = Bmax[-1] - Bmin[-1] + 1
    gpu_identity = cp.identity(NN, dtype = cp.cfloat)
    gpu_identity_batch = cp.repeat(gpu_identity[cp.newaxis, :, :], energy_batchsize, axis = 0)
    gR_gpu[-1, :, 0:NN, 0:NN] = cp.linalg.solve(ham_diag_gpu[-1, :, 0:NN, 0:NN], gpu_identity_batch)
    gL_gpu[-1, :, 0:NN, 0:NN] = gR_gpu[-1, :, 0:NN, 0:NN] @ (sl_diag_gpu[-1, :, 0:NN, 0:NN]) @ gR_gpu[-1, :, 0:NN, 0:NN].conjugate().transpose((0,2,1))
    gG_gpu[-1, :, 0:NN, 0:NN] = gR_gpu[-1, :, 0:NN, 0:NN] @ (sg_diag_gpu[-1, :, 0:NN, 0:NN]) @ gR_gpu[-1, :, 0:NN, 0:NN].conjugate().transpose((0,2,1))

    for IB in range(NB - 2, -1, -1):
        NI = Bmax[IB] - Bmin[IB] + 1
        NP = Bmax[IB + 1] - Bmin[IB + 1] + 1

        # # Extracting diagonal Hamiltonian block
        # if(IB == 0):
        #     M_c = M[Bmin[IB]:Bmax[IB] + 1, Bmin[IB]:Bmax[IB] + 1].toarray() - SigRBL
        #     SigL_c = SigL[Bmin[IB]:Bmax[IB] + 1, Bmin[IB]:Bmax[IB] + 1].toarray() + SigLBL
        #     SigG_c = SigG[Bmin[IB]:Bmax[IB] + 1, Bmin[IB]:Bmax[IB] + 1].toarray() + SigGBL
        # else: 
        #     M_c = M[Bmin[IB]:Bmax[IB] + 1, Bmin[IB]:Bmax[IB] + 1].toarray()
        #     SigL_c = SigL[Bmin[IB]:Bmax[IB] + 1, Bmin[IB]:Bmax[IB] + 1].toarray()
        #     SigG_c = SigG[Bmin[IB]:Bmax[IB] + 1, Bmin[IB]:Bmax[IB] + 1].toarray()
        # #M_c = M[Bmin[IB]:Bmax[IB] + 1, Bmin[IB]:Bmax[IB] + 1].toarray()

        # # Extracting off-diagonal Hamiltonian block (right)
        # M_r = M[Bmin[IB]:Bmax[IB] + 1, Bmin[IB + 1]:Bmax[IB + 1] + 1].toarray()

        # # Extracting off-diagonal Hamiltonian block (lower)
        # M_d = M[Bmin[IB + 1]:Bmax[IB + 1] + 1, Bmin[IB]:Bmax[IB] + 1].toarray()

        # # Extracting diagonal lesser Self-energy block
        # #SigL_c = SigL[Bmin[IB]:Bmax[IB] + 1, Bmin[IB]:Bmax[IB] + 1].toarray()

        # # Extracting off-diagonal lesser Self-energy block (lower)
        # SigL_l = SigL[Bmin[IB + 1]:Bmax[IB + 1] + 1, Bmin[IB]:Bmax[IB] + 1].toarray()

        # # Extracting diagonal greater Self-energy block
        # #SigG_c = SigG[Bmin[IB]:Bmax[IB] + 1, Bmin[IB]:Bmax[IB] + 1].toarray()

        # # Extracting off-diagonal greater Self-energy block (lower)
        # SigG_l = SigG[Bmin[IB + 1]:Bmax[IB + 1] + 1, Bmin[IB]:Bmax[IB] + 1].toarray()

        gpu_identity = cp.identity(NI, dtype = cp.cfloat)
        gpu_identity_batch = cp.repeat(gpu_identity[cp.newaxis, :, :], energy_batchsize, axis = 0)
        #gR_gpu[IB, :, 0:NI, 0:NI] = cp.linalg.inv(ham_diag_gpu[IB, :, 0:NN, 0:NN] \
        gR_gpu[IB, :, 0:NI, 0:NI] = cp.linalg.solve(ham_diag_gpu[IB, :, 0:NN, 0:NN] \
                            - ham_upper_gpu[IB, :, 0:NI, 0:NP] \
                            @ gR_gpu[IB+1, :, 0:NP, 0:NP] \
                            @ ham_lower_gpu[IB, :, :NP, 0:NI], gpu_identity_batch)#######
        # # AL, What is this? Handling off-diagonal sigma elements?
        AL = ham_upper_gpu[IB, :, 0:NI, 0:NP] \
            @ gR_gpu[IB+1, :, 0:NP, 0:NP] \
            @ sl_lower_gpu[IB, :, 0:NP, 0:NI]
        
        SigLB_gpu[IB, :, 0:NI, 0:NI] = ham_upper_gpu[IB, :, 0:NI, 0:NP] \
                            @ gL_gpu[IB+1, :, 0:NP, 0:NP] \
                            @ ham_upper_gpu[IB, :, 0:NI, 0:NP].transpose((0,2,1)).conj() \
                            - (AL - AL.transpose((0,2,1)).conj())

        # gL[IB, 0:NI, 0:NI] = gR[IB, 0:NI, 0:NI] \
        #                     @ (SigL_c \
        #                     + M_r \
        #                     @ gL[IB+1, 0:NP, 0:NP] \
        #                     @ M_r.T.conj() \
        #                     - (AL - AL.T.conj()))  \
        #                     @ gR[IB, 0:NI, 0:NI].T.conj() # Confused about the AL

        gL_gpu[IB, :, 0:NI, 0:NI] = gR_gpu[IB, :, 0:NI, 0:NI] \
                            @ (sl_diag_gpu[IB, :, 0:NI, 0:NI] \
                            + SigLB_gpu[IB, :, 0:NI, 0:NI])  \
                            @ gR_gpu[IB, :, 0:NI, 0:NI].transpose((0, 2, 1)).conj() # Confused about the AL

        ### What is this?
        AG = ham_upper_gpu[IB, :, 0:NI, 0:NP] \
            @ gR_gpu[IB+1, :, 0:NP, 0:NP] \
            @ sg_lower_gpu[IB, :, 0:NP, 0:NI]     # Handling off-diagonal sigma elements? Prob. need to check

        # gG[IB, 0:NI, 0:NI] = gR[IB, 0:NI, 0:NI] \
        #                     @ (SigG_c \
        #                     + M_r \
        #                     @ gG[IB+1, 0:NP, 0:NP] \
        #                     @ M_r.T.conj() \
        #                     - (AG - AG.T.conj())) \
        #                     @ gR[IB, 0:NI, 0:NI].T.conj() # Confused about the AG. 
        SigGB_gpu[IB, :, 0:NI, 0:NI] = ham_upper_gpu[IB, :, 0:NI, 0:NP] \
                            @ gG_gpu[IB+1, :,  0:NP, 0:NP] \
                            @ ham_upper_gpu[IB, :, 0:NI, 0:NP].transpose((0,2,1)).conj() \
                            - (AG - AG.transpose((0,2,1)).conj())

        gG_gpu[IB, :, 0:NI, 0:NI] = gR_gpu[IB, :, 0:NI, 0:NI] \
                            @ (sg_diag_gpu[IB, :, 0:NI, 0:NI] \
                                + SigGB_gpu[IB, :, 0:NI, 0:NI]) \
                                @ gR_gpu[IB, :, 0:NI, 0:NI].transpose((0,2,1)).conj() # Confused about the AG. 
        
    #Second step of iteration
    GR_gpu = cp.zeros_like(gR_gpu)
    GRnn1_gpu = cp.zeros_like(SigLB_gpu)
    GL_gpu = cp.zeros_like(gL_gpu)
    GLnn1_gpu = cp.zeros_like(SigLB_gpu)
    GG_gpu = cp.zeros_like(gG_gpu)
    GGnn1_gpu = cp.zeros_like(SigLB_gpu)



    GR_gpu[0, :,  :NI, :NI] = gR_gpu[0, :, :NI, :NI]
    GRnn1_gpu[0, :,  :NI, :NP] = -GR_gpu[0, :, :NI, :NI] @ ham_upper_gpu[0, :, :NI, :NP] @ gR_gpu[1, :, :NP, :NP]

    GL_gpu[0, :, :NI, :NI] = gL_gpu[0, :, :NI, :NI]
    GLnn1_gpu[0, :, :NI, :NP] = GR_gpu[0, :, :NI, :NI] @ sl_upper_gpu[0, :, :NI, :NP] @ gR_gpu[1, :, :NP, :NP].transpose((0,2,1)).conj() \
                - GR_gpu[0,:, :NI,:NI] @ ham_upper_gpu[0, :, :NI, :NP] @ gL_gpu[1,:, :NP, :NP] \
                - GL_gpu[0,:, :NI,:NI] @ ham_lower_gpu[0, :, :NP, :NI].transpose((0,2,1)).conj() @ gR_gpu[1,:, :NP, :NP].transpose((0,2,1)).conj()

    GG_gpu[0, :, :NI, :NI] = gG_gpu[0, :, :NI, :NI]
    GGnn1_gpu[0, :, :NI, :NP] = GR_gpu[0, :,:NI, :NI] @ sg_upper_gpu[0, :, :NI, :NP] @ gR_gpu[1, :,:NP, :NP].transpose((0,2,1)).conj() \
                - GR_gpu[0,:, :NI,:NI] @ ham_upper_gpu[0, :, :NI, :NP] @ gG_gpu[1, :, :NP, :NP] \
                - GG_gpu[0,:,:NI,:NI] @ ham_lower_gpu[0, :, :NP, :NI].transpose((0,2,1)).conj() @ gR_gpu[1, :, :NP, :NP].transpose((0,2,1)).conj() 
    
    idE[:, 0] = cp.real(cp.trace(SigGB_gpu[0, :, :NI, :NI] @ GL_gpu[0, :, :NI, :NI] - GG_gpu[0, :, :NI, :NI] @ SigLB_gpu[0, :, :NI, :NI], axis1 = 1, axis2 = 2)).get()

    for IB in range(1, NB):

        NM = Bmax[IB - 1] - Bmin[IB - 1] + 1
        NI = Bmax[IB] - Bmin[IB] + 1

        # # Extracting off-diagonal Hamiltonian block (upper)
        # M_u = M[Bmin[IB - 1]:Bmax[IB - 1] + 1, Bmin[IB]:Bmax[IB] + 1].toarray()

        # # # Extracting off-diagonal Hamiltonian block (left)
        # M_l = M[Bmin[IB]:Bmax[IB] + 1, Bmin[IB - 1]:Bmax[IB - 1] + 1].toarray()

        # # Extracting off-diagonal lesser Self-energy block (left)
        # SigL_l = SigL[Bmin[IB]:Bmax[IB] + 1, Bmin[IB - 1]:Bmax[IB - 1] + 1].toarray()

        # # Extracting off-diagonal greater Self-energy block (left)
        # SigG_l = SigG[Bmin[IB]:Bmax[IB] + 1, Bmin[IB - 1]:Bmax[IB - 1] + 1].toarray()

        GR_gpu[IB, :, :NI, :NI] = gR_gpu[IB, :, :NI, :NI] + gR_gpu[IB, :,  :NI, :NI] \
                        @ ham_lower_gpu[IB-1, :, :NI, :NM] \
                        @ GR_gpu[IB-1, :, 0:NM, 0:NM] \
                        @ ham_upper_gpu[IB-1, :, :NM, :NI] \
                        @ gR_gpu[IB, :,  :NI, :NI]
        # # What is this? Handling off-diagonal elements?
        AL = gR_gpu[IB, :, :NI, :NI] \
            @ sl_lower_gpu[IB-1, :, :NI, :NM] \
            @ GR_gpu[IB-1, :, 0:NM, 0:NM].transpose((0,2,1)).conj() \
            @ ham_lower_gpu[IB-1, :, :NI, :NM].transpose((0,2,1)).conj() \
            @ gR_gpu[IB,:, 0:NI, 0:NI].transpose((0,2,1)).conj()
        # What is this?
        BL = gR_gpu[IB, :, :NI, :NI] \
            @ ham_lower_gpu[IB-1, :, :NI, :NM] \
            @ GR_gpu[IB-1, :, 0:NM, 0:NM] \
            @ ham_upper_gpu[IB-1, :, :NM, :NI] \
            @ gL_gpu[IB, :, :NI, :NI]

        GL_gpu[IB, :, 0:NI, 0:NI] = gL_gpu[IB, :, :NI, :NI] \
                            + gR_gpu[IB, :, 0:NI, 0:NI] \
                            @ ham_lower_gpu[IB-1, :, :NI, :NM] \
                            @ GL_gpu[IB-1, :, 0:NM, 0:NM] \
                            @ ham_lower_gpu[IB-1, :, :NI, :NM].transpose((0,2,1)).conj() \
                            @ gR_gpu[IB, :, :NI, :NI].transpose((0,2,1)).conj() \
                            - (AL - AL.transpose((0,2,1)).conj()) + (BL - BL.transpose((0,2,1)).conj())


        AG = gR_gpu[IB, :, 0:NI, 0:NI] \
            @ sg_lower_gpu[IB-1, :, :NI, :NM] \
            @ GR_gpu[IB-1, :, 0:NM, 0:NM].transpose((0,2,1)).conj() \
            @ ham_lower_gpu[IB-1, :, :NI, :NM].transpose((0,2,1)).conj() \
            @ gR_gpu[IB, :, :NI, :NI].transpose((0,2,1)).conj()

        BG = gR_gpu[IB, :, 0:NI, 0:NI] \
            @ ham_lower_gpu[IB-1, :, :NI, :NM] \
            @ GR_gpu[IB-1, :, 0:NM, 0:NM] \
            @ ham_upper_gpu[IB-1, :, :NM, :NI] \
            @ gG_gpu[IB, :, 0:NI, 0:NI]

        GG_gpu[IB, :, 0:NI, 0:NI] = gG_gpu[IB, :, 0:NI, 0:NI] \
                            + gR_gpu[IB, :, 0:NI, 0:NI] \
                            @ ham_lower_gpu[IB-1, :, :NI, :NM] \
                            @ GG_gpu[IB-1, :, 0:NM, 0:NM] \
                            @ ham_lower_gpu[IB-1, :, :NI, :NM].transpose((0,2,1)).conj() \
                            @ gR_gpu[IB, :, 0:NI, 0:NI].transpose((0,2,1)).conj() \
                            - (AG - AG.transpose((0,2,1)).conj()) + (BG - BG.transpose((0,2,1)).conj()) #

        if IB < NB - 1:  #Off-diagonal are only interesting for IdE!

            # # Extracting off-diagonal Hamiltonian block (right)
            # M_r = M[Bmin[IB]:Bmax[IB] + 1, Bmin[IB + 1]:Bmax[IB + 1] + 1].toarray()

            # # # Extracting off-diagonal Hamiltonian block (lower)
            # M_d = M[Bmin[IB + 1]:Bmax[IB + 1] + 1, Bmin[IB]:Bmax[IB] + 1].toarray()

            # # Extracting off-diagonal lesser Self-energy block (right)
            # SigL_r = SigL[Bmin[IB]:Bmax[IB] + 1, Bmin[IB + 1]:Bmax[IB + 1] + 1].toarray()

            # # Extracting off-diagonal greater Self-energy block (right)
            # SigG_r = SigG[Bmin[IB]:Bmax[IB] + 1, Bmin[IB + 1]:Bmax[IB + 1] + 1].toarray()

            NP = Bmax[IB + 1] - Bmin[IB + 1] + 1

            GRnn1_gpu[IB, :, 0:NI, 0:NP] = - GR_gpu[IB, :,  0:NI, 0:NI] \
                                    @ ham_upper_gpu[IB, :, :NM, :NI] \
                                    @ gR_gpu[IB+1, :, 0:NP, 0:NP]

            GLnn1_gpu[IB, :, 0:NI, 0:NP] = GR_gpu[IB, :, 0:NI, 0:NI] \
                                    @ sl_upper_gpu[IB, :, :NM, :NI] \
                                    @ gR_gpu[IB+1, :, 0:NP, 0:NP].transpose((0,2,1)).conj() \
                                    - GR_gpu[IB, :, 0:NI, 0:NI] \
                                    @ ham_upper_gpu[IB, :, :NM, :NI] \
                                    @ gL_gpu[IB+1, :, 0:NP, 0:NP] \
                                    - GL_gpu[IB, :, :NI, :NI] \
                                    @ ham_lower_gpu[IB, :, :NI, :NM].transpose((0,2,1)).conj() \
                                    @ gR_gpu[IB+1, :, 0:NP, 0:NP].transpose((0,2,1)).conj()
            GGnn1_gpu[IB, :, 0:NI, 0:NP] = GR_gpu[IB, :, 0:NI, 0:NI] \
                                    @ sg_upper_gpu[IB, :, :NM, :NI] \
                                    @ gR_gpu[IB+1, :, 0:NP, 0:NP].transpose((0,2,1)).conj() \
                                    - GR_gpu[IB, :, 0:NI, 0:NI] \
                                    @ ham_upper_gpu[IB, :, :NM, :NI]  \
                                    @ gG_gpu[IB+1, :, 0:NP, 0:NP] \
                                    - GG_gpu[IB, :, 0:NI, 0:NI] \
                                    @ ham_lower_gpu[IB, :, :NI, :NM].transpose((0,2,1)).conj() \
                                    @ gR_gpu[IB+1, :, 0:NP, 0:NP].transpose((0,2,1)).conj()
            idE[:, IB] = cp.real(cp.trace(SigGB_gpu[IB, :NI, :NI] @ GL_gpu[IB, :NI, :NI] - GG_gpu[IB, :NI, :NI] @ SigLB_gpu[IB, :NI, :NI], axis1 = 1, axis2 = 2)).get() 
    for IB in range(NB):
        
        NI = Bmax[IB] - Bmin[IB] + 1
        # GR[IB, :, :, :] *= factor
        # GL[IB, :, :, :] *= factor
        # GG[IB, :, :, :] *= factor
        DOS[:, IB] = 1j * cp.trace(GR_gpu[IB, :, :, :] - GR_gpu[IB, :, :, :].transpose((0,2,1)).conj(), axis1= 1, axis2 = 2).get()
        nE[:, IB] = -1j * cp.trace(GL_gpu[IB, :, :, :], axis1= 1, axis2 = 2).get()
        nP[:, IB] = 1j * cp.trace(GG_gpu[IB, :, :, :], axis1= 1, axis2 = 2).get()

    #     if IB < NB-1:
    #         NP = Bmax[IB+1] - Bmin[IB+1] + 1
    #         #idE[IB] = -2 * np.trace(np.real(H[Bmin[IB+1]:Bmax[IB+1]+1, Bmin[IB]:Bmax[IB]+1].toarray() @ GLnn1[IB, 0:NI, 0:NP]))
    #         # GRnn1[IB, :, :, :] *= factor
    #         # GLnn1[IB, :, :, :] *= factor
    #         # GGnn1[IB, :, :, :] *= factor

    
    #idE[NB - 1] = idE[NB - 2]
    idE[:, NB-1] = cp.real(cp.trace(SigGBR_gpu[:, :NI, :NI] @ GL_gpu[NB-1, :, :NI, :NI] - GG_gpu[NB-1, :, :NI, :NI] @ SigLBR_gpu[:, :NI, :NI], axis1 = 1, axis2 = 2)).get()

    #Final Data Transfer
    #GR[:, :, :, :] = GR_gpu.get()
    GL[:, :, :, :] = GL_gpu.get()
    GG[:, :, :, :] = GG_gpu.get()
    #GRnn1[:, :, :, :] = GRnn1_gpu.get()
    GLnn1[:, :, :, :] = GLnn1_gpu.get()
    GGnn1[:, :, :, :] = GGnn1_gpu.get()