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

    ToDo:
    Inplace transpose and not copy

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
        rest = self.shape % self.comm_size
        self.counts[:, self.comm_size - 1] += rest
        self.count = self.counts[:, self.comm_rank]
        self.displacements = self.data_per_rank.reshape(-1, 1) * np.arange(self.comm_size)
        self.displacement = self.displacements[:, self.comm_rank]
        self.is_divisible = not np.any(rest)
        self.is_divisible_row = not rest[0]
        self.is_divisible_col = not rest[1]
        if self.is_divisible:
            assert all(self.counts[0, 0] == self.counts[0, i] for i in range(self.comm_size))
            assert all(self.counts[1, 0] == self.counts[1, i] for i in range(self.comm_size))

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
        self.R2C_SEND = self.BASE_TYPE.Create_vector(self.count[1], 1, self.shape[0])
        self.R2C_SEND_RESIZED = self.R2C_SEND.Create_resized(0, self.base_size)
        MPI.Datatype.Commit(self.R2C_SEND)
        MPI.Datatype.Commit(self.R2C_SEND_RESIZED)

        # receive types r2c
        # vector of size of #ranks
        # multi column data type for every rank since n is not divisible
        self.R2C_RECV = np.array([self.BASE_TYPE.Create_vector(self.count[0],
                                                               self.counts[1, i], self.shape[1]) for i in range(self.comm_size)])
        self.R2C_RECV_RESIZED = np.empty_like(self.R2C_RECV)
        for i in range(self.comm_size):
            self.R2C_RECV_RESIZED[i] = self.R2C_RECV[i].Create_resized(0, self.base_size)
            MPI.Datatype.Commit(self.R2C_RECV[i])
            MPI.Datatype.Commit(self.R2C_RECV_RESIZED[i])

        # send type c2r
        # local array is column contiguous
        self.C2R_SEND = self.BASE_TYPE.Create_vector(self.count[0], 1, self.shape[1])
        self.C2R_SEND_RESIZED = self.C2R_SEND.Create_resized(0, self.base_size)
        MPI.Datatype.Commit(self.C2R_SEND)
        MPI.Datatype.Commit(self.C2R_SEND_RESIZED)

        # receive types c2r
        # vector of size of #ranks
        # multi column data type for every rank since m is not divisible
        self.C2R_RECV = np.array([self.BASE_TYPE.Create_vector(self.count[1],
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

        # todo why does this not work?
        # if transpose_net:
        #     self.comm.Alltoall([inp, self.count[0],
        #                         self.R2C_SEND_RESIZED],
        #                        [outp, 1, self.R2C_RECV_RESIZED[0]])
        # else:
        #     inp_transposed = np.copy(inp.T, order="C")
        #     self.comm.Alltoall([
        #         inp_transposed, self.count[0] * self.count[1],
        #         self.BASE_TYPE,
        #     ], [outp, 1, self.R2C_RECV_RESIZED[0]])
        if self.is_divisible:
            if transpose_net:
                self.comm.Alltoallv([inp, self.counts[0, :], self.displacements[0, :],
                                    self.R2C_SEND_RESIZED],
                                    [outp, np.repeat([1], self.comm_size), self.displacements[1, :], self.R2C_RECV_RESIZED[0]])
            else:
                inp_transposed = np.copy(inp.T, order="C")
                self.comm.Alltoallv([
                    inp_transposed, self.counts[0, :] * self.count[1], self.displacements[0, :] *
                    self.count[1],
                    self.BASE_TYPE,
                ], [outp, np.repeat([1], self.comm_size), self.displacements[1, :], self.R2C_RECV_RESIZED[0]])
        else:
            if transpose_net:
                self.comm.Alltoallw([inp, self.counts[0, :], self.displacements[0, :] * self.base_size,
                                    np.repeat(self.R2C_SEND_RESIZED, self.comm_size)],
                                    [outp, np.repeat([1], self.comm_size), self.displacements[1, :] * self.base_size, self.R2C_RECV_RESIZED])
            else:
                inp_transposed = np.copy(inp.T, order="C")
                self.comm.Alltoallw([
                    inp_transposed, self.counts[0, :] * self.count[1], self.displacements[0, :] *
                    self.count[1] * self.base_size,
                    np.repeat(self.BASE_TYPE, self.comm_size)
                ], [outp, np.repeat([1], self.comm_size), self.displacements[1, :] * self.base_size, self.R2C_RECV_RESIZED])

    def ialltoall_c2r(self, inp: np.ndarray, outp: np.ndarray, transpose_net: bool = False) -> None:
        if transpose_net:
            request = self.comm.Ialltoallw([inp, self.counts[0, :], self.displacements[0, :] * self.base_size,
                                            np.repeat(self.R2C_SEND_RESIZED, self.comm_size)],
                                           [outp, np.repeat([1], self.comm_size), self.displacements[1, :] * self.base_size, self.R2C_RECV_RESIZED])
        else:
            inp_transposed = np.copy(inp.T, order="C")
            request = self.comm.Ialltoallw([
                inp_transposed, self.counts[0, :] * self.count[1], self.displacements[0, :] *
                self.count[1] * self.base_size,
                np.repeat(self.BASE_TYPE, self.comm_size)
            ], [outp, np.repeat([1], self.comm_size), self.displacements[1, :] * self.base_size, self.R2C_RECV_RESIZED])
        return request

    def alltoall_r2c(self, inp: np.ndarray, outp: np.ndarray, transpose_net: bool = False) -> None:
        if self.is_divisible:
            if transpose_net:
                self.comm.Alltoallv([inp, self.counts[1, :], self.displacements[1, :],
                                    self.C2R_SEND_RESIZED],
                                    [outp, np.repeat([1], self.comm_size), self.displacements[0, :], self.C2R_RECV_RESIZED[0]])
            else:
                inp_transposed = np.copy(inp.T, order="C")
                self.comm.Alltoallv([
                    inp_transposed, self.counts[1, :] * self.count[0], self.displacements[1, :] *
                    self.count[0],
                    self.BASE_TYPE
                ], [outp, np.repeat([1], self.comm_size), self.displacements[0, :], self.C2R_RECV_RESIZED[0]])
        else:
            if transpose_net:
                self.comm.Alltoallw([inp, self.counts[1, :], self.displacements[1, :] * self.base_size,
                                    np.repeat(self.C2R_SEND_RESIZED, self.comm_size)],
                                    [outp, np.repeat([1], self.comm_size), self.displacements[0, :] * self.base_size, self.C2R_RECV_RESIZED])
            else:
                inp_transposed = np.copy(inp.T, order="C")
                self.comm.Alltoallw([
                    inp_transposed, self.counts[1, :] * self.count[0], self.displacements[1, :] *
                    self.count[0] * self.base_size,
                    np.repeat(self.BASE_TYPE, self.comm_size)
                ], [outp, np.repeat([1], self.comm_size), self.displacements[0, :] * self.base_size, self.C2R_RECV_RESIZED])

    def ialltoall_r2c(self, inp: np.ndarray, outp: np.ndarray, transpose_net: bool = False) -> None:
        if transpose_net:
            request = self.comm.Ialltoallw([inp, self.counts[1, :], self.displacements[1, :] * self.base_size,
                                            np.repeat(self.C2R_SEND_RESIZED, self.comm_size)],
                                           [outp, np.repeat([1], self.comm_size), self.displacements[0, :] * self.base_size, self.C2R_RECV_RESIZED])
        else:
            inp_transposed = np.copy(inp.T, order="C")
            request = self.comm.Ialltoallw([
                inp_transposed, self.counts[1, :] * self.count[0], self.displacements[1, :] *
                self.count[0] * self.base_size,
                np.repeat(self.BASE_TYPE, self.comm_size)
            ], [outp, np.repeat([1], self.comm_size), self.displacements[0, :] * self.base_size, self.C2R_RECV_RESIZED])
        return request

    def gatherall_col(self, inp: np.ndarray, outp: np.ndarray, otype=None) -> None:
        if otype is None:
            self.comm.Allgatherv(inp, [outp, self.counts[1, :], self.displacements[1, :], self.BASE_TYPE])
        else:
            mpi_otype = MPI.Datatype(dtlib.from_numpy_dtype(otype))
            self.comm.Allgatherv(inp, [outp, self.counts[1, :], self.displacements[1, :], mpi_otype])


class TransposeCompute():
    def __init__(self, distributions: List[TransposeMatrix], function: Callable,
                 num_buffer: int, direction: str, comm_unblock: bool = False,
                 distributions_unblock_row: List[TransposeMatrix] = None,
                 distributions_unblock_col: List[TransposeMatrix] = None,
                 batchsize_row=None, batchsize_col=None,
                 iterations=None) -> None:
        self.distributions = distributions
        self.function = function
        self.num_buffer = num_buffer
        self.direction = direction
        self.comm_unblock = comm_unblock

        # addional information for unblocking
        if (batchsize_row is None or batchsize_col is None
            or iterations is None
            or distributions_unblock_row is None
                or distributions_unblock_col is None) and comm_unblock:
            raise ValueError("If comm_unblock is True, all optional inputs must be given")

        if comm_unblock:
            if not all(distribution.is_divisible for distribution in distributions):
                raise ValueError("If comm_unblock is True, all distributions must be divisible by the grid")
            self.distributions_unblock_row = distributions_unblock_row
            self.distributions_unblock_col = distributions_unblock_col
            self.batchsize_row = batchsize_row
            self.batchsize_col = batchsize_col
            self.iterations = iterations

    def alloc_buffer(self):
        self.buffer_row = [np.zeros((self.distributions[i].count[0], self.distributions[i].shape[1]),
                                    dtype=self.distributions[i].base_type) for i in range(self.num_buffer)]
        self.buffer_col = [np.zeros((self.distributions[i].count[1], self.distributions[i].shape[0]),
                                    dtype=self.distributions[i].base_type) for i in range(self.num_buffer)]

        if self.comm_unblock:
            self.buffer_row_compute = [np.zeros(
                (self.distributions_unblock_row[i].count[0], self.distributions_unblock_row[i].shape[1]),
                dtype=self.distributions_unblock_row[i].base_type) for i in range(self.num_buffer)]
            self.buffer_row_send = [np.zeros(
                (self.distributions_unblock_row[i].count[0], self.distributions_unblock_row[i].shape[1]),
                dtype=self.distributions_unblock_row[i].base_type) for i in range(self.num_buffer)]
            self.buffer_col_recv = [np.zeros(
                (self.distributions_unblock_row[i].count[1], self.distributions_unblock_row[i].shape[0]),
                dtype=self.distributions_unblock_row[i].base_type) for i in range(self.num_buffer)]

            self.buffer_col_compute = [np.zeros(
                (self.distributions_unblock_col[i].count[1], self.distributions_unblock_col[i].shape[0]),
                dtype=self.distributions_unblock_col[i].base_type) for i in range(self.num_buffer)]
            self.buffer_col_send = [np.zeros(
                (self.distributions_unblock_col[i].count[1], self.distributions_unblock_col[i].shape[0]),
                dtype=self.distributions_unblock_col[i].base_type) for i in range(self.num_buffer)]
            self.buffer_row_recv = [np.zeros(
                (self.distributions_unblock_col[i].count[0], self.distributions_unblock_col[i].shape[1]),
                dtype=self.distributions_unblock_col[i].base_type) for i in range(self.num_buffer)]

    def free_buffer(self):
        self.buffer_row = None
        self.buffer_col = None

        if self.comm_unblock:
            self.buffer_row_compute = None
            self.buffer_row_send = None
            self.buffer_col_recv = None

            self.buffer_col_compute = None
            self.buffer_col_send = None
            self.buffer_row_recv = None

    def given_buffer(self, buffer_row, buffer_col,
                     buffer_row_compute=None, buffer_col_compute=None,
                     buffer_row_send=None, buffer_col_send=None,
                     buffer_row_recv=None, buffer_col_recv=None):
        self.buffer_row = buffer_row
        self.buffer_col = buffer_col

        if self.comm_unblock:
            if (buffer_row_compute is None
                or buffer_col_compute is None
                or buffer_row_send is None
                    or buffer_col_send is None):
                raise ValueError("If comm_unblock is True, buffer_row_compute and buffer_col_compute must be given")
            self.buffer_row_compute = buffer_row_compute
            self.buffer_row_send = buffer_row_send
            self.buffer_col_recv = buffer_col_recv

            self.buffer_col_compute = buffer_col_compute
            self.buffer_col_send = buffer_col_send
            self.buffer_row_recv = buffer_row_recv

            for i in range(self.num_buffer):
                assert self.buffer_row_compute[i].shape[0] == self.batchsize_row
                assert self.buffer_row_send[i].shape[0] == self.batchsize_row
                assert self.buffer_col_compute[i].shape[0] == self.batchsize_col
                assert self.buffer_col_send[i].shape[0] == self.batchsize_col

    def compute_communicate(self, inp_block, inp, transpose_net: bool = False):
        if self.comm_unblock:
            self.compute_communicate_unblocking(inp_block, inp, transpose_net=transpose_net)
        else:
            self.compute_communicate_blocking(inp_block, inp, transpose_net=transpose_net)

    def compute_communicate_blocking(self, inp_block, inp, transpose_net: bool = False):
        if self.direction == "r2c":
            self.function(*inp_block, *inp, *self.buffer_row)
            for i in range(self.num_buffer):
                self.distributions[i].alltoall_r2c(self.buffer_row[i], self.buffer_col[i], transpose_net=transpose_net)
        elif self.direction == "c2r":
            self.function(*inp_block, *inp, *self.buffer_col)
            for i in range(self.num_buffer):
                self.distributions[i].alltoall_c2r(self.buffer_col[i], self.buffer_row[i], transpose_net=transpose_net)
        else:
            raise ValueError("Direction must be either r2c or c2r")

    def compute_communicate_unblocking(self, inp_block, inp, transpose_net: bool = False):
        if self.direction == "r2c":
            for distribution in self.distributions:
                # same number of columns to be able to batch over them
                assert distribution.shape[0] == self.distributions[0].shape[0]
                assert distribution.count[0] == self.distributions[0].count[0]
                assert distribution.comm_size == self.distributions[0].comm_size
                assert distribution.comm_rank == self.distributions[0].comm_rank
                assert distribution.is_divisible_row
                for j in range(distribution.comm_size):
                    assert distribution.displacements[0, j] == self.distributions[0].displacements[0, j]

            for i in range(0, self.iterations[0]):

                idx_batch_start = i*self.batchsize_row
                idx_batch_end = min((i+1)*self.batchsize_row, self.distributions[0].shape[0])

                # slice a range of the input
                inp_block_slice = [inp_block_i[idx_batch_start:idx_batch_end, ...] for inp_block_i in inp_block]

                self.function(*inp_block_slice, *inp, *self.buffer_row_compute)
                for k in range(self.num_buffer):
                    # todo replace with pointer swap
                    self.buffer_row_send[k][:] = self.buffer_row_compute[k]
                if i > 0:
                    MPI.Request.Waitall(requests)
                    for j in range(self.distributions[k].comm_size):
                        idx_start = self.distributions[0].displacements[0, j] + \
                            (i-1) * self.batchsize_row
                        idx_range = self.batchsize_row
                        for k in range(self.num_buffer):
                            self.buffer_col[k][:, idx_start:idx_start+idx_range] = \
                                self.buffer_col_recv[k][:, j*self.batchsize_row:j*self.batchsize_row+idx_range]

                requests = []
                for k in range(self.num_buffer):
                    requests.append(self.distributions_unblock_row[k].ialltoall_r2c(
                        self.buffer_row_send[k], self.buffer_col_recv[k], transpose_net=transpose_net))

            MPI.Request.Waitall(requests)
            for j in range(self.distributions[k].comm_size):
                idx_start = self.distributions[0].displacements[0, j] + \
                    (self.iterations[0]-1) * self.batchsize_row
                if self.distributions[0].count[0] % self.batchsize_row == 0:
                    idx_range = self.batchsize_row
                else:
                    idx_range = self.distributions[0].count[0] % self.batchsize_row
                for k in range(self.num_buffer):
                    self.buffer_col[k][:, idx_start:idx_start+idx_range] = \
                        self.buffer_col_recv[k][:, j*self.batchsize_row:j*self.batchsize_row+idx_range]
        elif self.direction == "c2r":
            for distribution in self.distributions:
                # same number of columns to be able to batch over them
                assert distribution.shape[1] == self.distributions[0].shape[1]
                assert distribution.count[1] == self.distributions[0].count[1]
                assert distribution.comm_size == self.distributions[0].comm_size
                assert distribution.comm_rank == self.distributions[0].comm_rank
                assert distribution.is_divisible_col
                for j in range(distribution.comm_size):
                    assert distribution.displacements[1, j] == self.distributions[0].displacements[1, j]

            for i in range(0, self.iterations[1]):

                idx_batch_start = i*self.batchsize_col
                idx_batch_end = min((i+1)*self.batchsize_col, self.distributions[0].shape[1])

                # slice a range of the input
                inp_block_slice = [inp_block_i[idx_batch_start:idx_batch_end, ...] for inp_block_i in inp_block]
                self.function(*inp_block_slice, *inp, *self.buffer_col_compute)

                for k in range(self.num_buffer):
                    # todo replace with pointer swap
                    self.buffer_col_send[k][:] = self.buffer_col_compute[k]
                if i > 0:
                    MPI.Request.Waitall(requests)
                    for j in range(self.distributions[k].comm_size):
                        idx_start = self.distributions[0].displacements[1, j] + \
                            (i-1) * self.batchsize_col
                        idx_range = self.batchsize_col
                        for k in range(self.num_buffer):
                            self.buffer_row[k][:, idx_start:idx_start+idx_range] = \
                                self.buffer_row_recv[k][:, j*self.batchsize_col:j*self.batchsize_col+idx_range]

                requests = []
                for k in range(self.num_buffer):
                    requests.append(self.distributions_unblock_col[k].ialltoall_c2r(
                        self.buffer_col_send[k], self.buffer_row_recv[k], transpose_net=transpose_net))

            MPI.Request.Waitall(requests)
            for j in range(self.distributions[k].comm_size):
                idx_start = self.distributions[0].displacements[1, j] + \
                    (self.iterations[1]-1) * self.batchsize_col
                if self.distributions[0].count[1] % self.batchsize_col == 0:
                    idx_range = self.batchsize_col
                else:
                    idx_range = self.distributions[0].count[1] % self.batchsize_col
                for k in range(self.num_buffer):
                    self.buffer_row[k][:, idx_start:idx_start+idx_range] = \
                        self.buffer_row_recv[k][:, j*self.batchsize_col:j*self.batchsize_col+idx_range]
        else:
            raise ValueError("Direction must be either r2c or c2r")
