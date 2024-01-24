import numpy as np
from scipy import sparse
from scipy.sparse import csr_matrix, lil_matrix, block_diag, find

import cupy as cp
import cupyx as cpx

def initialize_block_sigma_batched(NE, NB, Bsize):
    mr_3D_E = cp.zeros((NB, NE, Bsize, Bsize), dtype=np.complex128)
    mrnn1_3D_E = cp.zeros((NB - 1, NE, Bsize, Bsize), dtype=np.complex128)
    mrn1n_3D_E = cp.zeros((NB - 1, NE, Bsize, Bsize), dtype=np.complex128)
    ll_3D_E = cp.zeros((NB, NE, Bsize, Bsize), dtype=np.complex128)
    llnn1_3D_E = cp.zeros((NB - 1 , NE, Bsize, Bsize), dtype=np.complex128)
    lln1n_3D_E = cp.zeros((NB - 1, NE, Bsize, Bsize), dtype=np.complex128)
    lg_3D_E = cp.zeros((NB, NE, Bsize, Bsize), dtype=np.complex128)
    lgnn1_3D_E = cp.zeros((NB - 1, NE, Bsize, Bsize), dtype=np.complex128)
    lgn1n_3D_E = cp.zeros((NB - 1, NE, Bsize, Bsize), dtype=np.complex128)

    return (mr_3D_E, mrnn1_3D_E, mrn1n_3D_E, ll_3D_E, llnn1_3D_E, lln1n_3D_E, lg_3D_E, lgnn1_3D_E, lgn1n_3D_E)