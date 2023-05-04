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
Input parameters for the Si Nanowire.

Structure Properties: 
    Si Unit Cell: 21 Si-Atoms and 20 H Atoms for passivation -> 104 Wannier Orbitals
    Block Size (Super Cell): 3 x Unit Cell -> 123 Atoms and 312 Wannier Orbitals
    Number of Blocks: 10
    
    Total Nanowire Atoms = 1230
    Total Nanowire Orbitals = 3120
    
"""

""" Path = 'Si_nanowire'
no_orb = np.array([1, 4])  # 1 Orbital on H atoms, 4 orbitals on Si atoms.
E = np.linspace(-15.5, 5.5, 1051, endpoint = True, dtype = np.float64) # Energy Vector
dE = E[1]-E[0] # Energy spacing
NE = E.shape[0]
EfL = -2.1 # Fermi Level of Left Contact
EfR = -2.101 # Fermi Level of Right Contact """

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

V_sparse = construct_coulomb_matrix(DH, epsR, eps0, e)

iteration = 1
crit = np.inf

#reading reference solution
energy, rows, columns, gg_gold, gl_gold, gr_gold = read_solution.load_x("/usr/scratch/mont-fort17/dleonard/CNT/data_GPWS_04.mat", "g")
energy, rows_p, columns_p, pg_gold, pl_gold, pr_gold = read_solution.load_x("/usr/scratch/mont-fort17/dleonard/CNT/data_GPWS_04.mat", "p")
energy, rows_w, columns_w, wg_gold, wl_gold, wr_gold = read_solution.load_x("/usr/scratch/mont-fort17/dleonard/CNT/data_GPWS_04.mat", "w")
energy, rows_s, columns_s, sg_gold, sl_gold, sr_gold = read_solution.load_x("/usr/scratch/mont-fort17/dleonard/CNT/data_GPWS_04.mat", "s")
rowsRef, columnsRef, vh_gold                        = read_solution.load_v("/usr/scratch/mont-fort17/dleonard/CNT/data_Vh_4.mat")
ij2ji = change_format.find_idx_transposed(rows, columns)
pre_factor = -1.0j * dE / (np.pi)

# Creating the mask for the energy range of the deleted W elements given by the reference solution
w_mask = np.ndarray(shape = (energy.shape[0],), dtype = bool)

wr_mask = np.sum(np.abs(wr_gold), axis = 0) > 1e-10
wl_mask = np.sum(np.abs(wl_gold), axis = 0) > 1e-10
wg_mask = np.sum(np.abs(wg_gold), axis = 0) > 1e-10
w_mask = np.logical_or(np.logical_or(wr_mask, wl_mask), wg_mask)

map_diag, map_upper, map_lower = map_block2sparse_alt(rows, columns,
                                                                DH.Bmax-1, DH.Bmin-1)

# Parameters to create the map for the W block structures
NB = DH.Bmin.shape[0]
nbc = 2
bmax_pyt = DH.Bmax - 1
bmin_pyt = DH.Bmin - 1
bmax_ref = bmax_pyt[nbc-1:NB:nbc]
bmin_ref = bmin_pyt[0:NB:nbc]

map_diag_W, map_upper_W, map_lower_W = map_block2sparse_alt(rows, columns, bmax_ref, bmin_ref)

assert np.allclose(rows_p, rows)
assert np.allclose(columns, columns_p)
assert np.allclose(rows_w, rows)
assert np.allclose(columns, columns_w)
assert np.allclose(rowsRef, rows)
assert np.allclose(columnsRef, columns)
assert np.allclose(rows_s, rows)
assert np.allclose(columns, columns_s)

assert pg_gold.shape[0] == rows.shape[0]

#Start and end index of the energy range
ne_s = 0
ne_f = 251

# Maximum Block size
Bsize = np.max(DH.Bmax - DH.Bmin + 1)
# Number of atomic orbitals
nao = DH.Bmax[-1]

print(numba.get_num_threads())

# Observables: DOS, Transmission, Current
DOS = np.zeros(shape = (E.shape[0],DH.Bmin.shape[0]), dtype = np.cfloat)
IdE = np.zeros(shape = (E.shape[0],DH.Bmin.shape[0]), dtype = np.cfloat)

# Starting the scGW iterations: Currently only testing one iteration
while (iteration <= max_iteration) and (crit > max_error):
    if(iteration == 1):
        SigR = np.ndarray(shape = (E[ne_s:ne_f].shape[0],),dtype = object)
        SigG = np.ndarray(shape = (E[ne_s:ne_f].shape[0],),dtype = object)
        SigL = np.ndarray(shape = (E[ne_s:ne_f].shape[0],),dtype = object)

        for i in range(ne_s, ne_f):
            SigR[i-ne_s] = sparse.csr_matrix((NB * Bsize, NB * Bsize), dtype = np.cfloat)
            SigG[i-ne_s] = sparse.csr_matrix((NB * Bsize, NB * Bsize), dtype = np.cfloat)
            SigL[i-ne_s] = sparse.csr_matrix((NB * Bsize, NB * Bsize), dtype = np.cfloat)

    # Calculating the Green's function and OBC (Sigmas should not change)
    tic_rgf_G = time.perf_counter()
    GR_3D_E, GRnn1_3D_E, GL_3D_E, GLnn1_3D_E, GG_3D_E, GGnn1_3D_E = \
                calc_GF_pool(DH, E[ne_s:ne_f], copy.deepcopy(SigR), copy.deepcopy(SigG), copy.deepcopy(SigG), EfL, EfR, Temp, DOS, gf_mkl_threads, gf_worker_threads)
    toc_rgf_G = time.perf_counter()

    # Transforming the Green's function to energy contiguous arrays
    tic_trafo_g2p = time.perf_counter()
    GRn1n_reduced = np.array(list(map(np.transpose, GRnn1_3D_E.reshape(((NB-1)*(ne_f - ne_s), Bsize, Bsize)))))
    GRn1n_3D_E = GRn1n_reduced.reshape((ne_f - ne_s, NB-1, Bsize, Bsize))

    GLn1n_reduced = np.array(list(map(matrix_creation.negative_hermitian_transpose,
                                       GLnn1_3D_E.reshape(((NB-1)*(ne_f - ne_s), Bsize, Bsize)))))
    GLn1n_3D_E = GLn1n_reduced.reshape((ne_f - ne_s, NB-1, Bsize, Bsize))

    GGn1n_reduced = np.array(list(map(matrix_creation.negative_hermitian_transpose,
                                        GGnn1_3D_E.reshape(((NB-1)*(ne_f - ne_s), Bsize, Bsize)))))
    GGn1n_3D_E = GGn1n_reduced.reshape((ne_f - ne_s, NB-1, Bsize, Bsize))

    gr_computed = block2sparse_energy_alt(map_diag,
                                         map_upper,
                                         map_lower,
                                         GR_3D_E,
                                         GRnn1_3D_E,
                                         GRn1n_3D_E, rows.size, ne_f-ne_s)
    
    gl_computed = block2sparse_energy_alt(map_diag,
                                            map_upper,
                                            map_lower,
                                            GL_3D_E,
                                            GLnn1_3D_E,
                                            GLn1n_3D_E, rows.size, ne_f-ne_s)
    
    gg_computed = block2sparse_energy_alt(map_diag,
                                            map_upper,
                                            map_lower,
                                            GG_3D_E,
                                            GGnn1_3D_E,
                                            GGn1n_3D_E, rows.size, ne_f-ne_s)
    
    
    toc_trafo_g2p = time.perf_counter()

    # Calculating the polarization
    tic_polarization = time.perf_counter()
    pg_computed, pl_computed, pr_computed = g2p_cpu.g2p_fft_cpu_inlined(pre_factor, ij2ji,
                                                                            gg_computed, gl_computed, gr_computed)

    toc_polarization = time.perf_counter()

    # Transforming the polarization to vector of sparse matrices
    tic_trafo_p2w = time.perf_counter()
    pg_vecsparse = sparse2vecsparse(pg_computed, rows, columns, nao)
    pl_vecsparse = sparse2vecsparse(pl_computed, rows, columns, nao)
    pr_vecsparse = sparse2vecsparse(pr_computed, rows, columns, nao)
    vh = sparse.coo_array((vh_gold, (rows, columns)),
                                    shape=(nao, nao), dtype = np.complex128).tocsr()
    toc_trafo_p2w = time.perf_counter()

    # Calculating the screened interaction
    tic_rgf_W = time.perf_counter()
    #calc_GF(DH, E[ne_s:ne_f], GR, GL, GG, SigR, SigG, SigL, EfL, EfR, Temp)
    #WR_3D_E, WRnn1_3D_E, WL_3D_E, WLnn1_3D_E, WG_3D_E, WGnn1_3D_E, nb_mm, lb_max_mm = \
    #            calc_W(DH, E[ne_s:ne_f], pg_vecsparse, pl_vecsparse, pr_vecsparse, vh, w_mask, w_mkl_threads, w_worker_threads)
    WR_3D_E, WRnn1_3D_E, WL_3D_E, WLnn1_3D_E, WG_3D_E, WGnn1_3D_E, nb_mm, lb_max_mm = \
                calc_W_pool(DH, E[ne_s:ne_f], pg_vecsparse, pl_vecsparse, pr_vecsparse, vh, w_mask, w_mkl_threads, w_worker_threads)
    toc_rgf_W = time.perf_counter()

    # Transforming the screened interaction to energy contiguous arrays
    tic_trafo_w2s = time.perf_counter()
    WRn1n_reduced = np.array(list(map(np.transpose, WRnn1_3D_E.reshape(((nb_mm-1)*(ne_f - ne_s), lb_max_mm, lb_max_mm)))))
    WRn1n_3D_E = WRn1n_reduced.reshape(ne_f - ne_s, nb_mm-1, lb_max_mm, lb_max_mm)

    WLn1n_reduced = np.array(list(map(matrix_creation.negative_hermitian_transpose,
                                       WLnn1_3D_E.reshape(((nb_mm-1)*(ne_f - ne_s), lb_max_mm, lb_max_mm)))))
    WLn1n_3D_E = WLn1n_reduced.reshape((ne_f - ne_s, nb_mm-1, lb_max_mm, lb_max_mm))

    WGn1n_reduced = np.array(list(map(matrix_creation.negative_hermitian_transpose,
                                        WGnn1_3D_E.reshape(((nb_mm-1)*(ne_f - ne_s), lb_max_mm, lb_max_mm)))))
    WGn1n_3D_E = WGn1n_reduced.reshape((ne_f - ne_s, nb_mm-1, lb_max_mm, lb_max_mm))

    wr_computed = block2sparse_energy_alt(map_diag_W,
                                         map_upper_W,
                                         map_lower_W,
                                         WR_3D_E,
                                         WRnn1_3D_E,
                                         WRn1n_3D_E, rows.size, ne_f-ne_s)
    
    wl_computed = block2sparse_energy_alt(map_diag_W,
                                            map_upper_W,
                                            map_lower_W,
                                            WL_3D_E,
                                            WLnn1_3D_E,
                                            WLn1n_3D_E, rows.size, ne_f-ne_s)
    
    wg_computed = block2sparse_energy_alt(map_diag_W,
                                            map_upper_W,
                                            map_lower_W,
                                            WG_3D_E,
                                            WGnn1_3D_E,
                                            WGn1n_3D_E, rows.size, ne_f-ne_s)
    
    
    toc_trafo_w2s = time.perf_counter()

    # Calculating the self-energy
    tic_sigma = time.perf_counter()
    sg_cpu, sl_cpu, sr_cpu = gw2s_cpu.gw2s_fft_cpu(
            -pre_factor/2, ij2ji,
            gg_computed, gl_computed, gr_computed,
            wg_computed, wl_computed, wr_computed)

    toc_sigma = time.perf_counter()

    iteration+=1

# Running the test with the reference solution
for i in range(ne_s, ne_f):
    print("Energy: " + "%.2f" % (i) + "  " + "%.2f" % (E[i]) + " [eV]" )
    # filtered_GL = GL[i-ne_s][rows, columns]
    # #assert np.allclose(np.squeeze(filtered_GL), gl_gold[:,i], rtol = 1e-3, atol = 1e-3)
    # filtered_GG = GG[i-ne_s][rows, columns]
    # #assert np.allclose(np.squeeze(filtered_GG), gg_gold[:,i], rtol = 1e-3, atol = 1e-3)
    # filtered_GR = GR[i-ne_s][rows, columns]
    # #assert np.allclose(np.squeeze(filtered_GR), gr_gold[:,i], rtol = 1e-3, atol = 1e-3)
    assert np.allclose(gr_computed[:,i-ne_s], gr_gold[:,i], rtol = 1e-3, atol = 1e-3)
    assert np.allclose(gl_computed[:,i-ne_s], gl_gold[:,i], rtol = 1e-3, atol = 1e-3)
    assert np.allclose(gg_computed[:,i-ne_s], gg_gold[:,i], rtol = 1e-3, atol = 1e-3)

    assert np.allclose(pg_computed[:,i-ne_s], pg_gold[:,i], rtol = 1e-6, atol = 1e-6)
    assert np.allclose(pl_computed[:,i-ne_s], pl_gold[:,i], rtol = 1e-6, atol = 1e-6)
    assert np.allclose(pr_computed[:,i-ne_s], pr_gold[:,i], rtol = 1e-6, atol = 1e-6)
    print('checking W')
    assert np.allclose(wg_computed[:, i-ne_s], wg_gold[:,i], atol=1e-2, rtol=1e-2)
    assert np.allclose(wl_computed[:, i-ne_s], wl_gold[:,i], atol=1e-6, rtol=1e-6)
    assert np.allclose(wr_computed[:, i-ne_s], wr_gold[:,i], atol=1e-6, rtol=1e-6)
    print('checking sigma')
    assert np.allclose(sg_gold[:, i-ne_s], sg_cpu[:, i], atol=1e-2, rtol=1e-2)
    assert np.allclose(sl_gold[:, i-ne_s], sl_cpu[:, i], atol=1e-2, rtol=1e-2)
    assert np.allclose(sr_gold[:, i-ne_s], sr_cpu[:, i], atol=1e-2, rtol=1e-2)

toc = time.perf_counter()

# print timings
print("Calculating GF took: " + "%.2f" % (toc_rgf_G-tic_rgf_G) + " [s]" )
print("Transforming GF into energy-contiguous array took: " + "%.2f" % (toc_trafo_g2p-tic_trafo_g2p) + " [s]" )
print("Calculating polarization took: " + "%.2f" % (toc_polarization-tic_polarization) + " [s]" )
print("Transforming polarization into vector of sparse matrices took: " + "%.2f" % (toc_trafo_p2w-tic_trafo_p2w) + " [s]" )
print("Calculating W  took: " + "%.2f" % (toc_rgf_W-tic_rgf_W) + " [s]" )
print("Transforming W into energy-contiguous array took: " + "%.2f" % (toc_trafo_w2s-tic_trafo_w2s) + " [s]" )
print("Calculating Sigma took: " + "%.2f" % (toc_sigma-tic_sigma) + " [s]" )
print("Total Time: " + "%.2f" % (toc_sigma - tic_rgf_G) + " [s]" )

folder = 'results/CNT_single/'
np.savetxt( folder + 'E.dat', E)
np.savetxt( folder + 'DOS_.dat', DOS.view(float))


