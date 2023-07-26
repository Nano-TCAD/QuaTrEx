# Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.

import numpy as np
from scipy import sparse
from utils.matrix_creation import create_matrices_H, initialize_block_G, mat_assembly_fullG
from OBC.beyn_cpu import beyn
from OBC.sancho import open_boundary_conditions
from functools import partial


def rgf_GF(M,
           H,
           SigL,
           SigG,
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
           fL,
           fR,
           Bmin_fi,
           Bmax_fi,
           factor=1.0,
           index_E=0,
           block_inv=False,
           use_dace=False,
           validate_dace=False,
           sancho=False,
           min_dEk=1e8):
    # rgf_GF(DH, E, EfL, EfR, Temp) This could be the function call considering Leo's code
    '''
    Working!
    
    '''

    beyn_func = beyn
    if use_dace:
        from OBC import beyn_dace
        beyn_func = partial(beyn_dace.beyn, validate=validate_dace)

    imag_lim = 5e-4
    R = 1000

    min_dEkL = 1e8
    min_dEkR = 1e8
    Bmax = Bmax_fi - 1
    Bmin = Bmin_fi - 1
    Bsize = max(Bmax - Bmin + 1)  # Used for declaration of variables
    NB = len(Bmin)
    NT = Bmax[NB - 1] + 1  # Not used in this fcn

    LBsize = Bmax[0] - Bmin[0] + 1
    RBsize = Bmax[NB - 1] - Bmin[NB - 1] + 1

    gR = np.zeros((NB, Bsize, Bsize), dtype=np.cfloat)  # Retarded (right)
    gL = np.zeros((NB, Bsize, Bsize), dtype=np.cfloat)  # Lesser (right)
    gG = np.zeros((NB, Bsize, Bsize), dtype=np.cfloat)  # Greater (right)
    #GR = np.zeros((NB, Bsize, Bsize), dtype=np.cfloat) # Retarded GF
    #GRnn1 = np.zeros((NB - 1, Bsize, Bsize), dtype=np.cfloat) #Off-diagonal GR
    #GL = np.zeros((NB, Bsize, Bsize), dtype=np.cfloat) # Lesser GF
    #GLnn1 = np.zeros((NB - 1, Bsize, Bsize), dtype=np.cfloat) # Off-diagonal GL
    #GG = np.zeros((NB, Bsize, Bsize), dtype=np.cfloat) # Greater GF
    #GGnn1 = np.zeros((NB - 1, Bsize, Bsize), dtype=np.cfloat) # Off-diagonal GG
    SigLB = np.zeros((NB - 1, Bsize, Bsize), dtype=np.cfloat)  # Lesser boundary self-energy
    SigGB = np.zeros((NB - 1, Bsize, Bsize), dtype=np.cfloat)  # Greater boundary self-energy

    IdE = np.zeros(NB)
    n = np.zeros(NB)
    p = np.zeros(NB)
    condL = 0.0
    condR = 0.0

    #GR/GL/GG OBC Left
    #_, SigRBL, _, condL = open_boundary_conditions(M[:LBsize, :LBsize].toarray(), M[LBsize:2*LBsize, :LBsize].toarray(),
    #                                                    M[:LBsize, LBsize:2*LBsize].toarray(), np.eye(LBsize, LBsize))
    if not sancho:
        _, condL, _, SigRBL, min_dEkL = beyn_func(M[:LBsize, :LBsize].toarray(),
                                                  M[:LBsize, LBsize:2 * LBsize].toarray(),
                                                  M[LBsize:2 * LBsize, :LBsize].toarray(),
                                                  imag_lim,
                                                  R,
                                                  'L',
                                                  function='G',
                                                  block=block_inv)

    if np.isnan(condL) or sancho:
        _, SigRBL, _, condL = open_boundary_conditions(M[:LBsize, :LBsize].toarray(),
                                                       M[LBsize:2 * LBsize, :LBsize].toarray(),
                                                       M[:LBsize, LBsize:2 * LBsize].toarray(), np.eye(LBsize, LBsize))

    #condL = np.nan
    if not np.isnan(condL):
        M[:LBsize, :LBsize] -= SigRBL
        GammaL = 1j * (SigRBL - SigRBL.conj().T)
        SigLBL = 1j * fL * GammaL
        SigGBL = 1j * (fL - 1) * GammaL
        SigL[:LBsize, :LBsize] += SigLBL
        SigG[:LBsize, :LBsize] += SigGBL

    #GR/GL/GG OBC right
    if not sancho:
        _, condR, _, SigRBR, min_dEkR = beyn_func(M[NT - RBsize:NT, NT - RBsize:NT].toarray(),
                                                  M[NT - 2 * RBsize:NT - RBsize, NT - RBsize:NT].toarray(),
                                                  M[NT - RBsize:NT, NT - 2 * RBsize:NT - RBsize].toarray(),
                                                  imag_lim,
                                                  R,
                                                  'R',
                                                  function='G',
                                                  block=block_inv)

    if np.isnan(condR) or sancho:
        _, SigRBR, _, condR = open_boundary_conditions(M[NT - RBsize:NT, NT - RBsize:NT].toarray(),
                                                       M[NT - 2 * RBsize:NT - RBsize, NT - RBsize:NT].toarray(),
                                                       M[NT - RBsize:NT, NT - 2 * RBsize:NT - RBsize].toarray(),
                                                       np.eye(RBsize, RBsize))
    #condR = np.nan
    if not np.isnan(condR):
        M[NT - RBsize:NT, NT - RBsize:NT] -= SigRBR
        GammaR = 1j * (SigRBR - SigRBR.conj().T)
        SigLBR = 1j * fR * GammaR
        SigGBR = 1j * (fR - 1) * GammaR
        SigL[NT - RBsize:NT, NT - RBsize:NT] += SigLBR
        SigG[NT - RBsize:NT, NT - RBsize:NT] += SigGBR

    min_dEk = np.min((min_dEkL, min_dEkR))

    if not (np.isnan(condL) or np.isnan(condR)):
        # First step of iteration
        NN = Bmax[-1] - Bmin[-1] + 1
        gR[-1, 0:NN, 0:NN] = np.linalg.inv(M[Bmin[-1]:Bmax[-1] + 1, Bmin[-1]:Bmax[-1] + 1].toarray())
        gL[-1, 0:NN, 0:NN] = gR[-1, 0:NN, 0:NN] @ SigL[Bmin[-1]:Bmax[-1] + 1,
                                                       Bmin[-1]:Bmax[-1] + 1].toarray() @ gR[-1, 0:NN, 0:NN].T.conj()
        gG[-1, 0:NN, 0:NN] = gR[-1, 0:NN, 0:NN] @ SigG[Bmin[-1]:Bmax[-1] + 1,
                                                       Bmin[-1]:Bmax[-1] + 1].toarray() @ gR[-1, 0:NN, 0:NN].T.conj()

        for IB in range(NB - 2, -1, -1):
            NI = Bmax[IB] - Bmin[IB] + 1
            NP = Bmax[IB + 1] - Bmin[IB + 1] + 1

            # Extracting diagonal Hamiltonian block
            M_c = M[Bmin[IB]:Bmax[IB] + 1, Bmin[IB]:Bmax[IB] + 1].toarray()

            # Extracting off-diagonal Hamiltonian block (right)
            M_r = M[Bmin[IB]:Bmax[IB] + 1, Bmin[IB + 1]:Bmax[IB + 1] + 1].toarray()

            # Extracting off-diagonal Hamiltonian block (lower)
            M_d = M[Bmin[IB + 1]:Bmax[IB + 1] + 1, Bmin[IB]:Bmax[IB] + 1].toarray()

            # Extracting diagonal lesser Self-energy block
            SigL_c = SigL[Bmin[IB]:Bmax[IB] + 1, Bmin[IB]:Bmax[IB] + 1].toarray()

            # Extracting off-diagonal lesser Self-energy block (lower)
            SigL_l = SigL[Bmin[IB + 1]:Bmax[IB + 1] + 1, Bmin[IB]:Bmax[IB] + 1].toarray()

            # Extracting diagonal greater Self-energy block
            SigG_c = SigG[Bmin[IB]:Bmax[IB] + 1, Bmin[IB]:Bmax[IB] + 1].toarray()

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
    
    return 0


# write if name == __main__:
if __name__ == '__main__':
    # write the code that you want to run when you run this file
    # as a script
    n = 128
    nBlocks = 6
    NE = 1

    H, SL, SG = create_matrices_H(n, nBlocks)

    SBL = sparse.random(n, n, 1, dtype=np.cfloat)
    SBL = (SBL + SBL.T) / 2
    SBR = sparse.random(n, n, 1, dtype=np.cfloat)
    SBR = (SBR + SBR.T) / 2
    SB = sparse.block_diag((SBL, sparse.csc_matrix((n * (nBlocks - 2), n * (nBlocks - 2))), SBR))
    M = H + SB

    nt = n * nBlocks

    block_starts = np.array(range(0, nt, n))
    block_ends = np.array(range(n - 1, nt, n))

    if len(block_ends) != len(block_starts):
        raise Exception("specified number of block starts does not match with specified block ends!")

    (GR_3D_E, GRnn1_3D_E, GL_3D_E, GLnn1_3D_E, GG_3D_E, GGnn1_3D_E) = initialize_block_G(NE, nBlocks, n)

    rgf_GF(M.copy(),
           SL.copy(),
           SG.copy(),
           GR_3D_E[0],
           GRnn1_3D_E[0],
           GL_3D_E[0],
           GLnn1_3D_E[0],
           GG_3D_E[0],
           GGnn1_3D_E[0],
           0.5,
           0.5,
           block_starts.copy() + 1,
           block_ends.copy() + 1,
           sancho=True)

    # Reference Solution
    LBsize = block_ends[0] - block_starts[0] + 1
    RBsize = block_ends[nBlocks - 1] - block_starts[nBlocks - 1] + 1
    NT = block_ends[nBlocks - 1] + 1
    fL = 0.5
    fR = 0.5

    #GR/GL/GG OBC Left
    _, SigRBL, _, condL = open_boundary_conditions(M[:LBsize, :LBsize].toarray(),
                                                   M[LBsize:2 * LBsize, :LBsize].toarray(),
                                                   M[:LBsize, LBsize:2 * LBsize].toarray(), np.eye(LBsize, LBsize))
    #_, condL, _, SigRBL, min_dEkL  = beyn(M[:LBsize, :LBsize].toarray(), M[:LBsize, LBsize:2*LBsize].toarray(), M[LBsize:2*LBsize, :LBsize].toarray(), 5e-4, 1000, 'L')
    if not np.isnan(condL):
        M[:LBsize, :LBsize] -= SigRBL
        GammaL = 1j * (SigRBL - SigRBL.conj().T)
        SigLBL = 1j * fL * GammaL
        SigGBL = 1j * (fL - 1) * GammaL
        SL[:LBsize, :LBsize] += SigLBL
        SG[:LBsize, :LBsize] += SigGBL

    #GR/GL/GG OBC right
    _, SigRBR, _, condR = open_boundary_conditions(M[NT - RBsize:NT, NT - RBsize:NT].toarray(),
                                                   M[NT - 2 * RBsize:NT - RBsize, NT - RBsize:NT].toarray(),
                                                   M[NT - RBsize:NT, NT - 2 * RBsize:NT - RBsize].toarray(),
                                                   np.eye(RBsize, RBsize))
    #_, condR, _, SigRBR, min_dEkR  = beyn(M[NT - RBsize:NT, NT - RBsize:NT].toarray(), M[NT - 2*RBsize:NT - RBsize, NT - RBsize:NT].toarray(), M[NT - RBsize:NT, NT - 2*RBsize:NT - RBsize].toarray(),  5e-4, 1000, 'R')
    if not np.isnan(condR):
        M[NT - RBsize:NT, NT - RBsize:NT] -= SigRBR
        GammaR = 1j * (SigRBR - SigRBR.conj().T)
        SigLBR = 1j * fR * GammaR
        SigGBR = 1j * (fR - 1) * GammaR
        SL[NT - RBsize:NT, NT - RBsize:NT] += SigLBR
        SG[NT - RBsize:NT, NT - RBsize:NT] += SigGBR

    GR_full = np.linalg.inv(M.toarray())
    GL_full = GR_full @ SL @ GR_full.T.conj()
    GG_full = GR_full @ SG @ GR_full.T.conj()

    # Validation of rgf green's function with full green's function
    # Use the following code to validate the results: use mat_assemble to assemble the 3D matrix into a 2D matrix
    # and then use np.allclose to compare the two matrices

    GL_2D = mat_assembly_fullG(GL_3D_E[0],
                               GLnn1_3D_E[0],
                               block_starts.copy() + 1,
                               block_ends.copy() + 1,
                               format='sparse',
                               type='L')
    GG_2D = mat_assembly_fullG(GG_3D_E[0],
                               GGnn1_3D_E[0],
                               block_starts.copy() + 1,
                               block_ends.copy() + 1,
                               format='sparse',
                               type='G')
    GR_2D = mat_assembly_fullG(GR_3D_E[0],
                               GRnn1_3D_E[0],
                               block_starts.copy() + 1,
                               block_ends.copy() + 1,
                               format='sparse',
                               type='R')

    np.testing.assert_allclose(GL_full[n:2 * n, :3 * n], GL_2D[n:2 * n, :3 * n].toarray(), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(GR_full[n:2 * n, :3 * n], GR_2D[n:2 * n, :3 * n].toarray(), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(GG_full[n:2 * n, :3 * n], GG_2D[n:2 * n, :3 * n].toarray(), rtol=1e-6, atol=1e-6)
