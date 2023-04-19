# Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.
""" Recursive Green's Function (RGF) solver. """

import numpy as np

from numpy import typing as npt
from typing import TypeVar
# from scipy import sparse
# from utils.linalg import invert
# from utils.matrix_creation import create_matrices_H, initialize_block_G, mat_assembly_fullG
# from OBC.beyn import beyn
# from OBC.sancho import open_boundary_conditions

IT = TypeVar("IT", bound=npt.NBitBase)
FT = TypeVar("FT", bound=npt.NBitBase)
IntT = np.integer[IT]
FloatT = np.floating[FT]
ComplexT = np.complexfloating[FT]
IntArray = npt.NDArray[IntT]
FloatArray = npt.NDArray[FloatT]
ComplexArray = npt.NDArray[ComplexT]


def rgf(M: ComplexArray, SigL: ComplexArray, SigG: ComplexArray, GR: ComplexArray, GRnn1: ComplexArray,
        GL: ComplexArray, GLnn1: ComplexArray, GG: ComplexArray, GGnn1: ComplexArray, fL: FloatT, fR: FloatT,
        Bmin: IntArray, Bmax: IntArray):
    """
    Computes Green's Functions using the Recursive Green's Function (RGF) method.

    :param M: Hamiltonian matrix (?)
    :param SigL: Lesser Self-Energy (?)
    :param SigG: Greater Self-Energy (?)
    :param GR: Retarded Green's Function (?)
    :param GRnn1: (?)
    :param GL: Lesser Green's Function (?)
    :param GLnn1: (?)
    :param GG: Greater Green's Function (?)
    :param GGnn1: (?)
    :param fL: (?)
    :param fR: (?)
    :param Bmin: (?)
    :param Bmax: (?)
    """
    # rgf_GF(DH, E, EfL, EfR, Temp) This could be the function call considering Leo's code
    '''
    Working!
    
    '''
    imag_lim = 5e-4
    R = 1000

    min_dEkL = 1e8
    min_dEkR = 1e8
    Bmax -= 1
    Bmin -= 1
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

    IdE = np.zeros(NB)
    DOS = np.zeros(NB)
    n = np.zeros(NB)
    p = np.zeros(NB)

    #GR/GL/GG OBC Left
    #_, SigRBL, _, condL = open_boundary_conditions(M[:LBsize, :LBsize].toarray(), M[LBsize:2*LBsize, :LBsize].toarray(),
    #                                                    M[:LBsize, LBsize:2*LBsize].toarray(), np.eye(LBsize, LBsize))

    _, condL, _, SigRBL, min_dEk = beyn(M[:LBsize, :LBsize].toarray(), M[:LBsize, LBsize:2 * LBsize].toarray(),
                                        M[LBsize:2 * LBsize, :LBsize].toarray(), imag_lim, R, 'L')

    #condL = np.nan
    if not np.isnan(condL):
        M[:LBsize, :LBsize] -= SigRBL
        GammaL = 1j * (SigRBL - SigRBL.conj().T)
        SigLBL = 1j * fL * GammaL
        SigGBL = 1j * (fL - 1) * GammaL
        SigL[:LBsize, :LBsize] += SigLBL
        SigG[:LBsize, :LBsize] += SigGBL

    #GR/GL/GG OBC right
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

    # First step of iteration
    NN = Bmax[-1] - Bmin[-1] + 1
    gR[-1, 0:NN, 0:NN] = np.linalg.inv(M[Bmin[-1]:Bmax[-1] + 1, Bmin[-1]:Bmax[-1] + 1].toarray())
    gL[-1, 0:NN,
       0:NN] = gR[-1, 0:NN, 0:NN] @ SigL[Bmin[-1]:Bmax[-1] + 1, Bmin[-1]:Bmax[-1] + 1] @ gR[-1, 0:NN, 0:NN].T.conj()
    gG[-1, 0:NN,
       0:NN] = gR[-1, 0:NN, 0:NN] @ SigG[Bmin[-1]:Bmax[-1] + 1, Bmin[-1]:Bmax[-1] + 1] @ gR[-1, 0:NN, 0:NN].T.conj()

    for IB in range(NB - 2, -1, -1):
        NI = Bmax[IB] - Bmin[IB] + 1
        NP = Bmax[IB + 1] - Bmin[IB + 1] + 1

        gR[IB, 0:NI, 0:NI] = np.linalg.inv(M[Bmin[IB]:Bmax[IB]+1, Bmin[IB]:Bmax[IB]+1].toarray() \
                             - M[Bmin[IB]:Bmax[IB]+1, Bmin[IB+1]:Bmax[IB+1]+1] \
                             @ gR[IB+1, 0:NP, 0:NP] \
                             @ M[Bmin[IB+1]:Bmax[IB+1]+1, Bmin[IB]:Bmax[IB]+1])#######
        # AL, What is this? Handling off-diagonal sigma elements?
        AL = M[Bmin[IB]:Bmax[IB]+1, Bmin[IB+1]:Bmax[IB+1]+1] \
             @ gR[IB+1, 0:NP, 0:NP] \
             @ SigL[Bmin[IB+1]:Bmax[IB+1]+1, Bmin[IB]:Bmax[IB]+1]

        gL[IB, 0:NI, 0:NI] = gR[IB, 0:NI, 0:NI] \
                             @ (SigL[Bmin[IB]:Bmax[IB]+1, Bmin[IB]:Bmax[IB]+1] \
                             + M[Bmin[IB]:Bmax[IB]+1, Bmin[IB+1]:Bmax[IB+1]+1] \
                             @ gL[IB+1, 0:NP, 0:NP] \
                             @ M[Bmin[IB]:Bmax[IB]+1, Bmin[IB+1]:Bmax[IB+1]+1].T.conj() \
                             - (AL - AL.T.conj()))  \
                             @ gR[IB, 0:NI, 0:NI].T.conj() # Confused about the AL
        ### What is this?
        AG = M[Bmin[IB]:Bmax[IB]+1, Bmin[IB+1]:Bmax[IB+1]+1] \
             @ gR[IB+1, 0:NP, 0:NP] \
             @ SigG[Bmin[IB+1]:Bmax[IB+1]+1, Bmin[IB]:Bmax[IB]+1]     # Handling off-diagonal sigma elements? Prob. need to check

        gG[IB, 0:NI, 0:NI] = gR[IB, 0:NI, 0:NI] \
                             @ (SigG[Bmin[IB]:Bmax[IB]+1, Bmin[IB]:Bmax[IB]+1] \
                             + M[Bmin[IB]:Bmax[IB]+1, Bmin[IB+1]:Bmax[IB+1]+1] \
                             @ gG[IB+1, 0:NP, 0:NP] \
                             @ M[Bmin[IB]:Bmax[IB]+1, Bmin[IB+1]:Bmax[IB+1]+1].T.conj() \
                             - (AG - AG.T.conj())) \
                             @ gR[IB, 0:NI, 0:NI].T.conj() # Confused about the AG.

    #Second step of iteration
    GR[0, :NI, :NI] = gR[0, :NI, :NI]
    GRnn1[0, :NI, :NP] = -GR[0, :NI, :NI] @ M[Bmin[0]:Bmax[0] + 1, Bmin[1]:Bmax[1] + 1] @ gR[1, :NP, :NP]

    GL[0, :NI, :NI] = gL[0, :NI, :NI]
    GLnn1[0, :NI, :NP] = GR[0, :NI, :NI] @ SigL[Bmin[0]:Bmax[0]+1, Bmin[1]:Bmax[1]+1] @ gR[1, :NP, :NP].T.conj() \
                 - GR[0,:NI,:NI] @ M[Bmin[0]:Bmax[0]+1, Bmin[1]:Bmax[1]+1] @ gL[1, :NP, :NP] \
                 - GL[0,:NI,:NI] @ M[Bmin[1]:Bmax[1]+1, Bmin[0]:Bmax[0]+1].T.conj() @ gR[1, :NP, :NP].T.conj()

    GG[0, :NI, :NI] = gG[0, :NI, :NI]
    GGnn1[0, :NI, :NP] = GR[0, :NI, :NI] @ SigG[Bmin[0]:Bmax[0]+1, Bmin[1]:Bmax[1]+1] @ gR[1, :NP, :NP].T.conj() \
                - GR[0,:NI,:NI] @ M[Bmin[0]:Bmax[0]+1, Bmin[1]:Bmax[1]+1] @ gG[1, :NP, :NP] \
                - GG[0,:NI,:NI] @ M[Bmin[1]:Bmax[1]+1, Bmin[0]:Bmax[0]+1].T.conj() @ gR[1, :NP, :NP].T.conj()
    for IB in range(1, NB):

        NM = Bmax[IB - 1] - Bmin[IB - 1] + 1
        NI = Bmax[IB] - Bmin[IB] + 1

        GR[IB, :NI, :NI] = gR[IB, :NI, :NI] + gR[IB, :NI, :NI] \
                           @ M[Bmin[IB]:Bmax[IB]+1, Bmin[IB-1]:Bmax[IB-1]+1] \
                           @ GR[IB-1,0:NM, 0:NM] \
                           @ M[Bmin[IB-1]:Bmax[IB-1]+1, Bmin[IB]:Bmax[IB]+1] \
                           @ gR[IB, :NI, :NI]
        # What is this? Handling off-diagonal elements?
        AL = gR[IB, :NI, :NI] \
             @ SigL[Bmin[IB]:Bmax[IB]+1, Bmin[IB-1]:Bmax[IB-1]+1] \
             @ GR[IB-1, 0:NM, 0:NM].T.conj() \
             @ M[Bmin[IB]:Bmax[IB]+1, Bmin[IB-1]:Bmax[IB-1]+1].T.conj() \
             @ gR[IB, 0:NI, 0:NI].T.conj()
        # What is this?
        BL = gR[IB, :NI, :NI] \
             @ M[Bmin[IB]:Bmax[IB]+1, Bmin[IB-1]:Bmax[IB-1]+1] \
             @ GR[IB-1, 0:NM, 0:NM] \
             @ M[Bmin[IB-1]:Bmax[IB-1]+1, Bmin[IB]:Bmax[IB]+1] \
             @ gL[IB, :NI, :NI]

        GL[IB, 0:NI, 0:NI] = gL[IB, :NI, :NI] \
                             + gR[IB, 0:NI, 0:NI] \
                             @ M[Bmin[IB]:Bmax[IB]+1, Bmin[IB-1]:Bmax[IB-1]+1] \
                             @ GL[IB-1, 0:NM, 0:NM] \
                             @ M[Bmin[IB]:Bmax[IB]+1, Bmin[IB-1]:Bmax[IB-1]+1].T.conj() \
                             @ gR[IB, :NI, :NI].T.conj() \
                             - (AL - AL.T.conj()) + (BL - BL.T.conj())


        AG = gR[IB, 0:NI, 0:NI] \
             @ SigG[Bmin[IB]:Bmax[IB]+1, Bmin[IB-1]:Bmax[IB-1]+1] \
             @ GR[IB-1, 0:NM, 0:NM].T.conj() \
             @ M[Bmin[IB]:Bmax[IB]+1, Bmin[IB-1]:Bmax[IB-1]+1].T.conj() \
             @ gR[IB, 0:NI, 0:NI].T.conj()

        BG = gR[IB, 0:NI, 0:NI] \
             @ M[Bmin[IB]:Bmax[IB]+1, Bmin[IB-1]:Bmax[IB-1]+1] \
             @ GR[IB-1, 0:NM, 0:NM] \
             @ M[Bmin[IB-1]:Bmax[IB-1]+1, Bmin[IB]:Bmax[IB]+1] \
             @ gG[IB, 0:NI, 0:NI]

        GG[IB, 0:NI, 0:NI] = gG[IB, 0:NI, 0:NI] \
                             + gR[IB, 0:NI, 0:NI] \
                             @ M[Bmin[IB]:Bmax[IB]+1, Bmin[IB-1]:Bmax[IB-1]+1] \
                             @ GG[IB-1, 0:NM, 0:NM] \
                             @ M[Bmin[IB]:Bmax[IB]+1, Bmin[IB-1]:Bmax[IB-1]+1].T.conj() \
                             @ gR[IB, 0:NI, 0:NI].T.conj() \
                             - (AG - AG.T.conj()) + (BG - BG.T.conj()) #

        if IB < NB - 1:  #Off-diagonal are only interesting for IdE!
            NP = Bmax[IB + 1] - Bmin[IB + 1] + 1

            GRnn1[IB, 0:NI, 0:NP] = - GR[IB, 0:NI, 0:NI] \
                                    @ M[Bmin[IB]:Bmax[IB]+1, Bmin[IB+1]:Bmax[IB+1]+1] \
                                    @ gR[IB+1, 0:NP, 0:NP]

            GLnn1[IB, 0:NI, 0:NP] = GR[IB, 0:NI, 0:NI] \
                                    @ SigL[Bmin[IB]:Bmax[IB]+1, Bmin[IB+1]:Bmax[IB+1]+1] \
                                    @ gR[IB+1, 0:NP, 0:NP].T.conj() \
                                    - GR[IB, 0:NI, 0:NI] \
                                    @ M[Bmin[IB]:Bmax[IB]+1, Bmin[IB+1]:Bmax[IB+1]+1] \
                                    @ gL[IB+1, 0:NP, 0:NP] \
                                    - GL[IB, :NI, :NI] \
                                    @ M[Bmin[IB+1]:Bmax[IB+1]+1, Bmin[IB]:Bmax[IB]+1].T.conj() \
                                    @ gR[IB+1, 0:NP, 0:NP].T.conj()
            GGnn1[IB, 0:NI, 0:NP] = GR[IB, 0:NI, 0:NI] \
                                    @ SigG[Bmin[IB]:Bmax[IB]+1, Bmin[IB+1]:Bmax[IB+1]+1] \
                                    @ gR[IB+1, 0:NP, 0:NP].T.conj() \
                                    - GR[IB, 0:NI, 0:NI] \
                                    @ M[Bmin[IB]:Bmax[IB]+1, Bmin[IB+1]:Bmax[IB+1]+1] \
                                    @ gG[IB+1, 0:NP, 0:NP] \
                                    - GG[IB, 0:NI, 0:NI] \
                                    @ M[Bmin[IB+1]:Bmax[IB+1]+1, Bmin[IB]:Bmax[IB]+1].T.conj() \
                                    @ gR[IB+1, 0:NP, 0:NP].T.conj()
    print('done')
