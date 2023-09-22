"""
@author: Alexander Maeder (almaeder@ethz.ch)
@date: 2023-09

Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.
"""
import numpy as np
from mpi4py import MPI
from mpi4py.util import dtlib
from typing import Callable, List


class TransposeMatrix():
    """
    Distributes a m*n matrix A in either row or column slices among the ranks
    and is able to switch between both distribution.

    r2c: row to column distribution
    c2r: column to row distribution


    """

    def __init__(self, comm, shape, base_type=np.complex128) -> None:
        self.comm = comm
        self.shape = shape
        self.base_type = base_type

        self.comm_size = comm.Get_size()
        self.comm_rank = comm.Get_rank()

        self.split_data()
        self.init_datatypes()

    def split_data(self) -> None:
        self.data_per_rank = self.shape // self.comm_size
        self.counts = np.repeat(self.data_per_rank.reshape(-1, 1), self.comm_size, axis=1)
        self.counts[:, self.comm_size - 1] += self.shape % self.comm_size
        self.count = self.counts[:, self.comm_rank]
        self.displacements = self.data_per_rank.reshape(-1, 1) * np.arange(self.comm_size)
        self.displacement = self.displacements[:, self.comm_rank]

    def init_datatypes(self) -> None:
        self.BASE_TYPE = MPI.Datatype(dtlib.from_numpy_dtype(self.base_type))
        self.base_size = np.dtype(self.base_type).itemsize

        # column type of matrix in C order
        self.COLUMN = self.BASE_TYPE.Create_vector(self.shape[0], 1, self.shape[1])
        self.COLUMN_RESIZED = self.COLUMN.Create_resized(0, self.base_size)
        MPI.Datatype.Commit(self.COLUMN)
        MPI.Datatype.Commit(self.COLUMN_RESIZED)

        # row type of matrix in Fortran order
        self.ROW = self.BASE_TYPE.Create_vector(self.shape[1], 1, self.shape[0])
        self.ROW_RESIZED = self.ROW.Create_resized(0, self.base_size)
        MPI.Datatype.Commit(self.ROW)
        MPI.Datatype.Commit(self.ROW_RESIZED)

        # send type r2c
        # local array is row contiguous
        self.R2C_SEND = self.BASE_TYPE.Create_vector(self.counts[1, self.comm_rank], 1, self.shape[0])
        self.R2C_SEND_RESIZED = self.R2C_SEND.Create_resized(0, self.base_size)
        MPI.Datatype.Commit(self.R2C_SEND)
        MPI.Datatype.Commit(self.R2C_SEND_RESIZED)

        # receive types r2c
        # vector of size of #ranks
        # multi column data type for every rank since n is not divisible
        self.R2C_RECV = np.array([self.BASE_TYPE.Create_vector(self.counts[0, self.comm_rank],
                                                               self.counts[1, i], self.shape[1]) for i in range(self.comm_size)])
        self.R2C_RECV_RESIZED = np.empty_like(self.R2C_RECV)
        for i in range(self.comm_size):
            self.R2C_RECV_RESIZED[i] = self.R2C_RECV[i].Create_resized(0, self.base_size)
            MPI.Datatype.Commit(self.R2C_RECV[i])
            MPI.Datatype.Commit(self.R2C_RECV_RESIZED[i])

        # send type c2r
        # local array is column contiguous
        self.C2R_SEND = self.BASE_TYPE.Create_vector(self.counts[0, self.comm_rank], 1, self.shape[1])
        self.C2R_SEND_RESIZED = self.C2R_SEND.Create_resized(0, self.base_size)
        MPI.Datatype.Commit(self.C2R_SEND)
        MPI.Datatype.Commit(self.C2R_SEND_RESIZED)

        # receive types c2r
        # vector of size of #ranks
        # multi column data type for every rank since m is not divisible
        self.C2R_RECV = np.array([self.BASE_TYPE.Create_vector(self.counts[1, self.comm_rank],
                                                               self.counts[0, i], self.shape[0]) for i in range(self.comm_size)])
        self.C2R_RECV_RESIZED = np.empty_like(self.C2R_RECV)
        for i in range(self.comm_size):
            self.C2R_RECV_RESIZED[i] = self.C2R_RECV[i].Create_resized(0, self.base_size)
            MPI.Datatype.Commit(self.C2R_RECV[i])
            MPI.Datatype.Commit(self.C2R_RECV_RESIZED[i])

    def free_datatypes(self) -> None:
        MPI.Datatype.Free(self.COLUMN_RESIZED)
        MPI.Datatype.Free(self.COLUMN)
        MPI.Datatype.Free(self.ROW_RESIZED)
        MPI.Datatype.Free(self.ROW)
        MPI.Datatype.Free(self.R2C_SEND_RESIZED)
        MPI.Datatype.Free(self.R2C_SEND)
        MPI.Datatype.Free(self.C2R_SEND_RESIZED)
        MPI.Datatype.Free(self.C2R_SEND)
        for i in range(self.comm_size):
            MPI.Datatype.Free(self.R2C_RECV_RESIZED[i])
            MPI.Datatype.Free(self.R2C_RECV[i])
            MPI.Datatype.Free(self.C2R_RECV_RESIZED[i])
            MPI.Datatype.Free(self.C2R_RECV[i])

    def scatter_master(self, inp: np.ndarray, outp: np.ndarray, transpose_net: bool = False) -> None:
        if transpose_net:
            self.comm.Scatterv([inp, self.counts[1, :], self.displacements[1, :], self.COLUMN_RESIZED], outp, root=0)
        else:
            if self.comm_rank == 0:
                inp_transposed = np.copy(inp.T, order="C")
            else:
                inp_transposed = None
            self.comm.Scatterv([inp_transposed, self.counts[1, :] * self.shape[0], self.displacements[1, :] * self.shape[0], self.BASE_TYPE],
                               outp,
                               root=0)

    def gather_master(self, inp: np.ndarray, outp: np.ndarray, transpose_net: bool = False) -> None:
        if transpose_net:
            self.comm.Gatherv(inp, [outp, self.counts[1, :], self.displacements[1, :], self.COLUMN_RESIZED], root=0)
        else:
            if self.comm_rank == 0:
                out_transposed = np.copy(outp.T, order="C")
            else:
                out_transposed = None
            self.comm.Gatherv(inp, [out_transposed, self.counts[1, :] * self.shape[0], self.displacements[1, :] * self.shape[0], self.BASE_TYPE],
                              root=0)
            if self.comm_rank == 0:
                outp[:, :] = out_transposed.T

    def alltoall_c2r(self, inp: np.ndarray, outp: np.ndarray, transpose_net: bool = False) -> None:
        if transpose_net:
            self.comm.Alltoallw([inp, self.counts[0, :], self.displacements[0, :] * self.base_size,
                                 np.repeat(self.R2C_SEND_RESIZED, self.comm_size)],
                                [outp, np.repeat([1], self.comm_size), self.displacements[1, :] * self.base_size, self.R2C_RECV_RESIZED])
        else:
            inp_transposed = np.copy(inp.T, order="C")
            self.comm.Alltoallw([
                inp_transposed, self.counts[0, :] * self.counts[1, self.comm_rank], self.displacements[0, :] *
                self.counts[1, self.comm_rank] * self.base_size,
                np.repeat(self.BASE_TYPE, self.comm_size)
            ], [outp, np.repeat([1], self.comm_size), self.displacements[1, :] * self.base_size, self.R2C_RECV_RESIZED])

    def alltoall_r2c(self, inp: np.ndarray, outp: np.ndarray, transpose_net: bool = False) -> None:
        if transpose_net:
            self.comm.Alltoallw([inp, self.counts[1, :], self.displacements[1, :] * self.base_size,
                                 np.repeat(self.C2R_SEND_RESIZED, self.comm_size)],
                                [outp, np.repeat([1], self.comm_size), self.displacements[0, :] * self.base_size, self.C2R_RECV_RESIZED])
        else:
            inp_transposed = np.copy(inp.T, order="C")
            self.comm.Alltoallw([
                inp_transposed, self.counts[1, :] * self.counts[0, self.comm_rank], self.displacements[1, :] *
                self.counts[0, self.comm_rank] * self.base_size,
                np.repeat(self.BASE_TYPE, self.comm_size)
            ], [outp, np.repeat([1], self.comm_size), self.displacements[0, :] * self.base_size, self.C2R_RECV_RESIZED])

    def gatherall_col(self, inp: np.ndarray, outp: np.ndarray, otype=None) -> None:
        if otype is None:
            self.comm.Allgatherv(inp, [outp, self.counts[1, :], self.displacements[1, :], self.BASE_TYPE])
        else:
            mpi_otype = MPI.Datatype(dtlib.from_numpy_dtype(otype))
            self.comm.Allgatherv(inp, [outp, self.counts[1, :], self.displacements[1, :], mpi_otype])


class TransposeCompute():
    def __init__(self, distributions: List[TransposeMatrix], function: Callable, num_buffer: int, direction: str, comm_unblock: bool = False, batchsizes: List = None) -> None:
        self.distributions = distributions
        self.function = function
        self.num_buffer = num_buffer
        self.direction = direction
        if batchsizes is None and comm_unblock:
            raise ValueError("If comm_unblock is True, batchsizes must be given")
        self.comm_unblock = comm_unblock
        self.batchsizes = batchsizes

    def alloc_buffer(self):
        self.buffer_row = [np.zeros((self.distributions[i].count[0], self.distributions[i].shape[1]),
                                    dtype=self.distributions[i].base_type) for i in range(self.num_buffer)]
        self.buffer_col = [np.zeros((self.distributions[i].count[1], self.distributions[i].shape[0]),
                                    dtype=self.distributions[i].base_type) for i in range(self.num_buffer)]

        if self.comm_unblock:
            self.buffer_row_compute = [np.zeros(
                (self.distributions[i].count[0], self.distributions[i].shape[1]), dtype=self.distributions[i].base_type) for i in range(self.num_buffer)]
            self.buffer_col_compute = [np.zeros(
                (self.distributions[i].count[1], self.distributions[i].shape[0]), dtype=self.distributions[i].base_type) for i in range(self.num_buffer)]

    def free_buffer(self):
        self.buffer_row = None
        self.buffer_col = None

        if self.comm_unblock:
            self.buffer_row_compute = None
            self.buffer_col_compute = None

    def given_buffer(self, buffer_row, buffer_col, buffer_row_compute=None, buffer_col_compute=None):
        self.buffer_row = buffer_row
        self.buffer_col = buffer_col

        if self.comm_unblock:
            if buffer_row_compute is None or buffer_col_compute is None:
                raise ValueError("If comm_unblock is True, buffer_row_compute and buffer_col_compute must be given")
            self.buffer_row_compute = buffer_row_compute
            self.buffer_col_compute = buffer_col_compute

    def compute_communicate(self, inp_block, inp):
        if self.comm_unblock:
            self.compute_communicate_unblocking(inp_block, inp)
        else:
            self.compute_communicate_blocking(inp_block, inp)

    def compute_communicate_blocking(self, inp_block, inp):
        if self.direction == "r2c":
            self.function(*inp_block, *inp, *self.buffer_row)
            for i in range(self.num_buffer):
                self.distributions[i].alltoall_r2c(self.buffer_row[i], self.buffer_col[i])
        elif self.direction == "c2r":
            self.function(*inp_block, *inp, *self.buffer_col)
            for i in range(self.num_buffer):
                self.distributions[i].alltoall_c2r(self.buffer_col[i], self.buffer_row[i])
        else:
            raise ValueError("Direction must be either r2c or c2r")
