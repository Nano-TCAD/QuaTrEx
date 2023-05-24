#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 18:02:03 2023

@author: leonard
"""
import numpy as np
from scipy import sparse

from utils.matrix_creation import initialize_block_G, mat_assembly_fullG
from GreensFunction.fermi import fermi_function
from block_tri_solvers.rgf_GF import *

import mkl


def calc_GF(DH, E, GR, GL, GG, SigR, SigL, SigG, Efl, Efr, Temp):
    kB = 1.38e-23
    q = 1.6022e-19
    
    UT = kB * Temp / q
    
    vfermi = np.vectorize(fermi_function)
    fL = vfermi(E, Efl, UT)
    fR = vfermi(E, Efr, UT)
    
    NE = E.shape[0]
    NA = DH.NA

    dNP = 50 # number of points to smooth the edges of the Green's Function
    
    factor = np.ones(NE)
    factor[NE-dNP-1:NE] = (np.cos(np.pi*np.linspace(0, 1, dNP+1)) + 1)/2
    factor[0:dNP+1] = (np.cos(np.pi*np.linspace(1, 0, dNP+1)) + 1)/2

    NB = DH.Bmin.shape[0]
    NT = DH.Bmax[-1] + 1
    Bsize = np.max(DH.Bmax - DH.Bmin + 1)

    (GR_3D_E, GRnn1_3D_E, GL_3D_E, GLnn1_3D_E, GG_3D_E, GGnn1_3D_E) = initialize_block_G(NE, NB, Bsize)
    #initialize a vector of NE number of scipy sparse matrices for the GFs

    #GF = sparse.csr_matrix((NE, NH * (DH.NB + 1)*DH.TB))
    #GR = sparse.csr_matrix((NE, NH * (DH.NB + 1)*DH.TB))
    for IE in range(NE):
        print(IE)
        M = (E[IE]+1j*1e-12)*DH.Overlap['H_4']-DH.Hamiltonian['H_4']-SigR[IE]
        rgf_GF(M, SigG[IE], SigL[IE], GR_3D_E[IE], GRnn1_3D_E[IE], GL_3D_E[IE], GLnn1_3D_E[IE], GG_3D_E[IE], GGnn1_3D_E[IE], fL[IE], fR[IE], DH.Bmin.copy(), DH.Bmax.copy())
        # we want to create a list of sparse matrices, one for each type of Green's function
        GR[IE] = factor[IE] * mat_assembly_fullG(GR_3D_E[IE], GRnn1_3D_E[IE], DH.Bmin.copy(), DH.Bmax.copy(), format = 'sparse', type = 'R')
        GL[IE] = factor[IE] * mat_assembly_fullG(GL_3D_E[IE], GLnn1_3D_E[IE], DH.Bmin.copy(), DH.Bmax.copy(), format = 'sparse', type = 'L')
        GG[IE] = factor[IE] * mat_assembly_fullG(GG_3D_E[IE], GGnn1_3D_E[IE], DH.Bmin.copy(), DH.Bmax.copy(), format = 'sparse', type = 'G')

