# Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.
"""Loads the coulomb matrix computed from wannier orbitals and makes modification for stability."""

import h5py
import numpy as np
import numpy.typing as npt
import typing
from scipy import sparse
from scipy.sparse import csc_array, find
import mpi4py

mpi4py.rc.initialize = False  # do not initialize MPI automatically
mpi4py.rc.finalize = False  # do not finalize MPI automatically
from mpi4py import MPI


def load_V(path: str,
           rows_ni: npt.NDArray[np.int32],
           columns_ni: npt.NDArray[np.int32],
           reduce_to_neighbor_indices: bool = True):
    """Expects a .dat with 4 columns: rows, cols, real, imag """
    # Loading the coulomb matrix and extracting rows, columns, real, imag
    V_sparse = np.loadtxt(path + "/V.dat",
                          dtype={
                              'names': ('col1', 'col2', 'col3', 'col4'),
                              'formats': ('i4', 'i4', 'f8', 'f8')
                          })
    rows = V_sparse['col1'] - 1  # -1 because python starts counting at 0
    cols = V_sparse['col2'] - 1  # -1 because python starts counting at 0
    real_parts = V_sparse['col3']
    imag_parts = V_sparse['col4']

    # Creating a coo_matrix from the data above
    V_sparse = csc_array((real_parts + 1j * imag_parts, (rows, cols)))
    nao = V_sparse.shape[0]

    # Reducing the coulomb matrix to the neighbor indices
    if reduce_to_neighbor_indices:
        V_sparse.eliminate_zeros()
        V_sparse = V_sparse.toarray()[rows_ni, columns_ni]
        V_sparse = csc_array((V_sparse, (rows_ni, columns_ni)), shape=(nao, nao))

    # Making the coulomb matrix hermitian
    V_sparse = (V_sparse + V_sparse.conj().T) / 2

    # Adding a small real part to the diagonal to make the matrix positive definite
    V_sparse = V_sparse + 0.5 * csc_array((np.ones(nao), (np.arange(nao), np.arange(nao))))

    return V_sparse


def load_V_mpi(path: str,
               rows_ni: npt.NDArray[np.int32],
               columns_ni: npt.NDArray[np.int32],
               comm: MPI.Comm,
               rank: np.int32,
               reduce_to_neighbor_indices: bool = True):
    """Expects a .dat with 4 columns: rows, cols, real, imag """
    if (rank == 0):
        # Loading the coulomb matrix and extracting rows, columns, real, imag
        V_sparse = np.loadtxt(path,
                              dtype={
                                  'names': ('col1', 'col2', 'col3', 'col4'),
                                  'formats': ('i4', 'i4', 'f8', 'f8')
                              })
        rows = V_sparse['col1'] - 1  # -1 because python starts counting at 0
        cols = V_sparse['col2'] - 1  # -1 because python starts counting at 0
        real_parts = V_sparse['col3']
        imag_parts = V_sparse['col4']

        # Creating a coo_matrix from the data above
        V_sparse = csc_array((real_parts + 1j * imag_parts, (rows, cols)))
        # Making the coulomb matrix hermitian
        V_sparse = (V_sparse + V_sparse.conj().T) / 2
        nao = V_sparse.shape[0]
        comm.Bcast([np.array(nao), MPI.INT], root=0)

        # Reducing the coulomb matrix to the neighbor indices
        if reduce_to_neighbor_indices:
            V_sparse.eliminate_zeros()
            V_sparse = V_sparse.toarray()[rows_ni, columns_ni]
            V_sparse = csc_array((V_sparse, (rows_ni, columns_ni)), shape=(nao, nao))
            comm.Bcast([V_sparse.data, MPI.DOUBLE_COMPLEX], root=0)

        else:
            (I, J, V) = find(V_sparse)
            data_size = I.shape[0]
            comm.Bcast([np.array(data_size), MPI.INT], root=0)
            comm.Bcast([I, MPI.INT], root=0)
            comm.Bcast([J, MPI.INT], root=0)
            comm.Bcast([V, MPI.DOUBLE_COMPLEX], root=0)

        # Adding a small real part to the diagonal to make the matrix positive definite
        #V_sparse = V_sparse + 0.5 * csc_array((np.ones(nao), (np.arange(nao), np.arange(nao))))


        return V_sparse
    else:
        nao = np.zeros((1, ), dtype=int)
        comm.Bcast([nao, MPI.INT], root=0)
        nao = nao[0]

        if reduce_to_neighbor_indices:
            V_sparse_data = np.empty(rows_ni.shape[0], dtype=np.complex128)
            comm.Bcast([V_sparse_data, MPI.DOUBLE_COMPLEX], root=0)
            V_sparse = csc_array((np.array(V_sparse_data), (rows_ni, columns_ni)), shape=(nao, nao))

        else:
            data_size = np.zeros((1, ), dtype=int)
            comm.Bcast([data_size, MPI.INT], root=0)
            data_size = data_size[0]
            I = np.empty(data_size, dtype=np.int32)
            J = np.empty(data_size, dtype=np.int32)
            V = np.empty(data_size, dtype=np.complex128)
            comm.Bcast([I, MPI.INT], root=0)
            comm.Bcast([J, MPI.INT], root=0)
            comm.Bcast([V, MPI.DOUBLE_COMPLEX], root=0)
            V_sparse = csc_array((V, (I, J)), shape=(nao, nao))
        return V_sparse

def load_binary_V_mpi(path: str,
               rows_ni: npt.NDArray[np.int32],
               columns_ni: npt.NDArray[np.int32],
               comm: MPI.Comm,
               rank: np.int32,
               reduce_to_neighbor_indices: bool = True):
        if (rank == 0):
            V_sparse = read_sparse_matrix(path)
            # rows = V_sparse['col1'] - 1  # -1 because python starts counting at 0
            # cols = V_sparse['col2'] - 1  # -1 because python starts counting at 0
            # real_parts = V_sparse['col3']
            # imag_parts = V_sparse['col4']

            # # Creating a coo_matrix from the data above
            # V_sparse = csc_array((real_parts + 1j * imag_parts, (rows, cols)))
            # Making the coulomb matrix hermitian
            V_sparse = (V_sparse + V_sparse.conj().T) / 2
            nao = V_sparse.shape[0]
            comm.Bcast([np.array(nao), MPI.INT], root=0)

            # Reducing the coulomb matrix to the neighbor indices
            if reduce_to_neighbor_indices:
                V_sparse.eliminate_zeros()
                V_array = V_sparse.toarray()[rows_ni, columns_ni]
                V_sparse1 = sparse.coo_array((V_array, (rows_ni, columns_ni)), shape=(nao, nao))
                comm.Bcast([V_sparse1.data, MPI.DOUBLE_COMPLEX], root=0)
            else:
                (I, J, V) = find(V_sparse)
                data_size = I.shape[0]
                comm.Bcast([np.array(data_size), MPI.INT], root=0)
                comm.Bcast([I, MPI.INT], root=0)
                comm.Bcast([J, MPI.INT], root=0)
                comm.Bcast([V, MPI.DOUBLE_COMPLEX], root=0)

            return V_sparse1.tocsc()
        
        else:
            nao = np.zeros((1, ), dtype=int)
            comm.Bcast([nao, MPI.INT], root=0)
            nao = nao[0]

            if reduce_to_neighbor_indices:
                V_sparse_data = np.empty(rows_ni.shape[0], dtype=np.complex128)
                comm.Bcast([V_sparse_data, MPI.DOUBLE_COMPLEX], root=0)
                V_sparse = sparse.coo_array((np.array(V_sparse_data), (rows_ni, columns_ni)), shape=(nao, nao))

            else:
                data_size = np.zeros((1, ), dtype=int)
                comm.Bcast([data_size, MPI.INT], root=0)
                data_size = data_size[0]
                I = np.empty(data_size, dtype=np.int32)
                J = np.empty(data_size, dtype=np.int32)
                V = np.empty(data_size, dtype=np.complex128)
                comm.Bcast([I, MPI.INT], root=0)
                comm.Bcast([J, MPI.INT], root=0)
                comm.Bcast([V, MPI.DOUBLE_COMPLEX], root=0)
                V_sparse = csc_array((V, (I, J)), shape=(nao, nao))
            return V_sparse.tocsc()
    


#Helper function to initialise all hamiltonians
def read_sparse_matrix(fname='./H_4.bin'):

    fid = open(fname, 'rb')

    #Read the Header head[0]: Contains the format 0 if indexing starts at 0, 1 if indexing starts at 1
    #                head[1]: number of non-zero elements
    #                head[2]: matrix size
    head = matlab_fread(fid, 3, 'double')

    #Read the data and reshape it
    M = matlab_fread(fid, 4 * int(head[1]), 'double')
    blob = np.reshape(M, (int(head[1]), 4))
    fid.close()

    #Matrix indexes start with zero
    if head[2] == 0:
        H = sparse.csc_matrix((blob[:, 2] + 1j * blob[:, 3], (blob[:, 0], blob[:, 1])),
                                shape=(int(head[0]), int(head[0])))
    #Matrix indexes start with one
    else:
        H = sparse.csc_matrix((blob[:, 2] + 1j * blob[:, 3], (blob[:, 0] - 1, blob[:, 1] - 1)),
                                shape=(int(head[0]), int(head[0])))

    return H

def matlab_fread(fid, nelements, dtype):
    """Equivalent to Matlab fread function"""

    if dtype is np.str_:
        dt = np.uint8  # WARNING: assuming 8-bit ASCII for np.str!
    else:
        dt = dtype

    data_array = np.fromfile(fid, dt, nelements)
    data_array.shape = (nelements, 1)

    return data_array


if __name__ == "__main__":
    MPI.Init_thread(required=MPI.THREAD_FUNNELED)
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    name = MPI.Get_processor_name()

    rows_ni = np.concatenate((1 + np.arange(767), [0]))
    columns_ni = np.arange(768)
    V_sparse = load_V_mpi("/usr/scratch/mont-fort17/dleonard/GW_paper/CNT_32/V.dat", rows_ni, columns_ni, comm, rank)
    print(np.sum(V_sparse.data))

    V_sparse = load_V_mpi("/usr/scratch/mont-fort17/dleonard/GW_paper/CNT_32/V.dat", rows_ni, columns_ni, comm, rank, reduce_to_neighbor_indices=False)
    print(np.sum(V_sparse.data))
