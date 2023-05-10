"""Loads the coulomb matrix computed from wannier orbitals and makes modification for stability."""
import h5py
import numpy as np
import numpy.typing as npt
import typing
from scipy.sparse import csc_array

def load_V(
        path: str,
        rows_ni: npt.NDArray[np.int32],
        columns_ni: npt.NDArray[np.int32],
        reduce_to_neighbor_indices: bool = True
                        ):
    """Expects a .dat with 4 columns: rows, cols, real, imag """
    # Loading the coulomb matrix and extracting rows, columns, real, imag
    V_sparse = np.loadtxt(path + "/V.dat", dtype={'names': ('col1', 'col2', 'col3', 'col4'), 'formats': ('i4', 'i4', 'f8', 'f8')})
    rows = V_sparse['col1'] - 1 # -1 because python starts counting at 0
    cols = V_sparse['col2'] - 1 # -1 because python starts counting at 0
    real_parts = V_sparse['col3']
    imag_parts = V_sparse['col4']

    # Creating a coo_matrix from the data above
    V_sparse = csc_array((real_parts + 1j * imag_parts, (rows, cols)))
    nao = V_sparse.shape[0]

    # Reducing the coulomb matrix to the neighbor indices
    if reduce_to_neighbor_indices:
        V_sparse.eliminate_zeros()
        V_sparse = V_sparse.toarray()[rows_ni, columns_ni]
        V_sparse = csc_array((V_sparse, (rows_ni, columns_ni)), shape = (nao, nao))

    # Making the coulomb matrix hermitian
    V_sparse = (V_sparse + V_sparse.conj().T)/2

    # Adding a small real part to the diagonal to make the matrix positive definite
    V_sparse = V_sparse + 0.5 * csc_array((np.ones(nao), (np.arange(nao), np.arange(nao))))
    
    return V_sparse