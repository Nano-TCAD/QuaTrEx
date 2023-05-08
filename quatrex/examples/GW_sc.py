# Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.
"""Computing a single iteration of the GW iteration for a carbon nanotube configuration.
Timing Results are reported at the end of the script."""

import time

import copy
import numpy as np
from scipy import ndimage as nd
from scipy.interpolate import RegularGridInterpolator
from scipy import sparse
import numba

import matplotlib.pyplot as plt
from matplotlib import cm

import sys
import os
main_path = os.path.abspath(os.path.dirname(__file__))
parent_path = os.path.abspath(os.path.join(main_path, ".."))
sys.path.append(parent_path)

from OMEN_structure_matrices.construct_CM import construct_coulomb_matrix
from OMEN_structure_matrices.OMENHamClass import Hamiltonian

from bandstructure.calc_contact_bs import calc_bandstructure
from bandstructure.calc_band_edge import get_band_edge

from GreensFunction.calc_GF import calc_GF
from GreensFunction.calc_GF_pool import calc_GF_pool

from utils import matrix_creation
from utils import change_format

from GW.gold_solution import read_solution
from GW.polarization.kernel import g2p_cpu
from utils.change_format import map_block2sparse, block2sparse_energy, \
                    block2sparse_energy_alt, map_block2sparse_alt, sparse2vecsparse
from GW.screenedinteraction.kernel.calc_W import calc_W
from GW.screenedinteraction.kernel.calc_W_pool import calc_W_pool
from GW.selfenergy.kernel import gw2s_cpu

"""
Input parameters for the Carbon Nanotube.

Structure Properties: 
    CNT Unit Cell: 32 atoms -> 32 Wannier Orbitals
    Block Size (Super Cell): 4 x Unit Cell -> 128 Atoms and 128 Wannier Orbitals
    Number of Blocks: 6
    
    Total Nanowire Atoms = 768
    Total Nanowire Orbitals = 768
    
"""


Path = '/usr/scratch/mont-fort17/dleonard/CNT/CNT_newwannier'
no_orb = np.array([1, 1])  # 1 Orbital on C atoms, two same types
E = np.linspace(-17.5, 7.5, 251, endpoint = True, dtype = float) # Energy Vector
dE = E[1]-E[0] # Energy spacing
NE = E.shape[0]
EfL = -3.85 # Fermi Level of Left Contact
EfR = -3.85 # Fermi Level of Right Contact

ECmin = -3.55 # DFT conuction band minimum
EVmax = -4.05 # DFT valence band maximum

Temp = 300 # Temperature in Kelvin
epsR = 11.9 # Dielectric Constant 

max_iteration = 1 # Max number of iteration of the self-consistent Born Iteration
max_error = 1e-3 # Max error

memory_factor_s = 0.5
memory_factor_p = 0.5
use_fft = 1 # Use fft to compute energy convolution
retarded_from_lg = 0
sancho_rubio = 0

gf_mkl_threads = 1
gf_worker_threads = 8

w_mkl_threads = 1
w_worker_threads = 12

"""
Physical Constants
"""

e = 1.6022e-19
eps0 = 8.854e-12
hbar = 1.0546e-34

"""
Structure Generation
"""
tic = time.perf_counter()

DH = Hamiltonian(Path, no_orb)
H = DH.Hamiltonian['H_4']
S = DH.Overlap['H_4']

V_sparse = construct_coulomb_matrix(DH, epsR, eps0, e)

iteration = 1
crit = np.inf

NB = DH.Bmin.shape[0]
Bsize = np.max(DH.Bmax - DH.Bmin + 1)
bmin = DH.Bmin - 1 
bmax = DH.Bmax - 1
nao = DH.Bmax[-1]

SigR_GW = np.ndarray(shape = (E.shape[0],),dtype = object)
SigR_EPHN = np.ndarray(shape = (E.shape[0],),dtype = object)

for i in range(E.shape[0]):
            SigR_GW[i] = sparse.csr_matrix((NB * Bsize, NB * Bsize), dtype = np.cfloat)
            SigR_EPHN[i] = sparse.csr_matrix((NB * Bsize, NB * Bsize), dtype = np.cfloat)

ECmin_newL = get_band_edge(ECmin, E, S, H, SigR_GW, SigR_EPHN, bmin, bmax, side = 'left')
ECmin_newR = get_band_edge(ECmin, E, S, H, SigR_GW, SigR_EPHN, bmin, bmax, side = 'right')

print(ECmin_newL)
print(ECmin_newR)