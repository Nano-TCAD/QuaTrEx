# Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.

import numpy as np
from scipy import sparse
from quatrex.files_to_refactor.read_utils import *


def construct_coulomb_matrix(DH, eps_r, eps0, e, diag=False, orb_uniform=False):
    """
    This function computes a placeholder for the 2-index Coulomb matrix. It
    assumes that the atomic orbitals are point charges and computes their 
    coulomb repulsion based on their mutual distance. The coulomb matrix
    should be computed using the localized basis on a spacial grid.

    Parameters
    ----------
    DH : Device_Hamiltonian
        OMEN Hamiltonian Class with Block Properties
    epsR : float
        Relative Dielectric permittivity
    eps0 : float
        Dielectric permittivity of vacuum
    e   : float
        Elementary charge

    Returns
    -------
    V_Col : scipy.sparse.csc matix of type cfloat, same dimension as
    Hamiltonian Matrix in Hamiltonian class. (n_orbs x n_orbs)
        The coulomb Matrix

    """
    factor = e / (4 * np.pi * eps0 * eps_r) * 1e9
    V_atomic = np.zeros((DH.NA, DH.NB + 1, DH.TB, DH.TB), dtype=np.cfloat)
    SF = np.outer(np.arange(1, -0.1, -0.1), np.arange(1, -0.1, -0.1))
    Vmax = float(0.0)

    for ia in range(DH.NA):
        orbA = DH.orb_per_at[ia + 1] - DH.orb_per_at[ia]
        for ib in range(DH.NB):
            if DH.LM[ia, 4 + ib] > 0:
                neigh = int(DH.LM[ia, 4 + ib] - 1)
                orbB = DH.orb_per_at[neigh + 1] - DH.orb_per_at[neigh]

                dist = np.linalg.norm(DH.LM[neigh, 0:3] - DH.LM[ia, 0:3])
                LM = DH.LM

                if abs(dist) < 1e-24:
                    print(ia)
                    print(DH.LM[ia, 4 + ib])
                    print(dist)

                Vact = factor / dist

                if Vact > Vmax:
                    Vmax = Vact

                if (orb_uniform):
                    V_atomic[ia, ib + 1, 0:orbA, 0:orbB] = Vact * np.ones((orbA, orbB), dtype=np.cfloat)
                else:
                    V_atomic[ia, ib + 1, 0:orbA, 0:orbB] = Vact * SF[0:orbA, 0:orbB]

    for ia in range(DH.NA):

        orbA = DH.orb_per_at[ia+1] - DH.orb_per_at[ia]
        if (diag):
            if (orb_uniform):
                # V_atomic[ia,0, 0:orbA, 0:orbA] = 1.5 * Vmax * (np.ones((orbA, orbA), dtype = np.cfloat) - np.eye(int(orbA), dtype = np.cfloat))
                V_atomic[ia, 0, 0:orbA, 0:orbA] = 1.5 * Vmax * (np.ones((orbA, orbA), dtype=np.cfloat))
            else:
                V_atomic[ia, 0, 0:orbA, 0:orbA] = 1.5 * Vmax * SF[0:orbA, 0:orbA]
        elif (orbA > 1):
            if (orb_uniform):
                V_atomic[ia, 0, 0:orbA, 0:orbA] = Vmax * \
                    (np.ones((orbA, orbA), dtype=np.cfloat) - np.eye(int(orbA), dtype=np.cfloat))
            else:
                pass  # not changing this as it will break the test unfortunately.
        # V_atomic[ia,0, :orbA, :orbA] = 1.5 * Vmax * np.eye(int(orbA), dtype = np.cfloat)

    V_sparse = map_4D_to_sparse(V_atomic, DH)
    return V_sparse


def map_4D_to_sparse(V_atomic, DH):
    """


    Parameters
    ----------
    V_atomic : 4-D cfloat array of coulomb elements
        First dimension specifies atom index, second dimension specifies the
        selected neighbor. The remaining two dimensions are of size of the TB order 
        (maximum number of orbitals over atoms in the structure), this means
        one can select all possible orbital combinations (for each atom and its
                                                          selected neighbor)
    DH : Device_Hamiltonian
        OMEN Hamiltonian Class with Block Properties

    Returns
    -------
    V_Col : scipy.sparse.csc matix of type complex128, same dimension as
    Hamiltonian Matrix in Hamiltonian class. (n_orbs x n_orbs)
        The coulomb Matrix

    """
    indI = np.zeros((DH.NA * (DH.NB + 1) * DH.TB * DH.TB, ), dtype=int)
    indJ = np.zeros((DH.NA * (DH.NB + 1) * DH.TB * DH.TB, ), dtype=int)
    NNZ = np.zeros((DH.NA * (DH.NB + 1) * DH.TB * DH.TB, ), dtype=complex)

    ind = 0

    for IA in range(DH.NA):

        indR = DH.orb_per_at[IA]
        orbA = DH.orb_per_at[IA + 1] - DH.orb_per_at[IA]

        for IB in range(DH.NB + 1):

            add_element = 1

            if IB == 0:
                indC = indR
                orbB = orbA

            else:
                if DH.LM[IA, 4 + IB - 1] > 0:

                    neigh = int(DH.LM[IA, 4 + IB - 1] - 1)

                    indC = DH.orb_per_at[neigh]
                    orbB = DH.orb_per_at[neigh + 1] - DH.orb_per_at[neigh]

                else:
                    add_element = 0

            if add_element:
                indI[ind:ind + orbA * orbB] = np.reshape(np.outer(np.ones((1, orbB)), np.arange(indR, indR + orbA)),
                                                         (1, orbA * orbB))
                indJ[ind:ind + orbA * orbB] = np.reshape(np.outer(np.arange(indC, indC + orbB), np.ones((orbA, 1))),
                                                         (1, orbA * orbB))
                NNZ[ind:ind + orbA * orbB] = np.reshape(np.squeeze(V_atomic[IA, IB, 0:orbA, 0:orbB]), (1, orbA * orbB))

                ind = ind + orbA * orbB

    sparse_shape = np.max(DH.orb_per_at) - DH.orb_per_at[0]
    indI_sparse = indI[:ind] - 1
    indJ_sparse = indJ[:ind] - 1
    NNZ_sparse = NNZ[:ind]

    return sparse.csr_matrix((NNZ_sparse, (indI_sparse, indJ_sparse)), shape=(sparse_shape, sparse_shape))
