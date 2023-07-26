"""Loads the coulomb matrix computed from wannier orbitals and makes modification for stability."""
import h5py
import numpy as np
import numpy.typing as npt
import typing
from scipy.sparse import csc_array
import mpi4py
mpi4py.rc.initialize = False  # do not initialize MPI automatically
mpi4py.rc.finalize = False    # do not finalize MPI automatically
from mpi4py import MPI

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
    #V_sparse = V_sparse + 0.5 * csc_array((np.ones(nao), (np.arange(nao), np.arange(nao))))
    
    return V_sparse

def load_V_mpi(
        path: str,
        rows_ni: npt.NDArray[np.int32],
        columns_ni: npt.NDArray[np.int32],
        comm : MPI.Comm,
        rank : np.int32,
        reduce_to_neighbor_indices: bool = True
):

    """Expects a .dat with 4 columns: rows, cols, real, imag """
    if(rank == 0):
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
        #V_sparse = V_sparse + 0.5 * csc_array((np.ones(nao), (np.arange(nao), np.arange(nao))))
        comm.Bcast([np.array(nao), MPI.INT], root = 0)
        comm.Bcast([V_sparse.data, MPI.DOUBLE_COMPLEX], root = 0)
        return V_sparse
    else:
        nao = np.zeros((1,), dtype = int)
        comm.Bcast([nao, MPI.INT], root = 0)
        nao = nao[0]
        V_sparse_data = np.empty(rows_ni.shape[0], dtype = np.complex128)
        comm.Bcast([V_sparse_data, MPI.DOUBLE_COMPLEX], root = 0)
        V_sparse = csc_array((np.array(V_sparse_data), (rows_ni, columns_ni)), shape = (nao, nao))
        return V_sparse
if __name__ == "__main__":
    MPI.Init_thread(required=MPI.THREAD_FUNNELED)
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    name = MPI.Get_processor_name()

    rows_ni = np.arange(768)
    columns_ni = np.arange(768)
    V_sparse = load_V_mpi("/usr/scratch/mont-fort17/dleonard/CNT/CNT_newwannier", rows_ni, columns_ni, comm, rank)