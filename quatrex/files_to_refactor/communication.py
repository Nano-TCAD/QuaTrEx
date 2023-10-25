"""
@author: Alexander Maeder (almaeder@ethz.ch)
@date: 2023-09

Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.
"""
import numpy as np
from mpi4py import MPI
from mpi4py.util import dtlib
from typing import List
from functools import wraps


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
        """
        Split the data among the ranks.
        """
        self.data_per_rank = self.shape // self.comm_size
        self.counts = np.repeat(
            self.data_per_rank.reshape(-1, 1), self.comm_size, axis=1)
        rest = self.shape % self.comm_size
        # add rest to last rank
        # would be better for load balancing to evenly distribute the rest
        self.counts[:, self.comm_size - 1] += rest
        self.count = self.counts[:, self.comm_rank]
        self.displacements = self.data_per_rank.reshape(
            -1, 1) * np.arange(self.comm_size)
        self.displacement = self.displacements[:, self.comm_rank]
        self.is_divisible = not np.any(rest)
        self.is_divisible_row = not rest[0]
        self.is_divisible_col = not rest[1]
        self.range_local = [slice(self.displacement[0], self.displacement[0]+self.count[0]),
                            slice(self.displacement[1], self.displacement[1]+self.count[1])]
        if self.is_divisible:
            assert all(self.counts[0, 0] == self.counts[0, i]
                       for i in range(self.comm_size))
            assert all(self.counts[1, 0] == self.counts[1, i]
                       for i in range(self.comm_size))

    def init_datatypes(self) -> None:
        """
        Initialize all necessary datatypes for communication.
        """
        self.BASE_TYPE = MPI.Datatype(dtlib.from_numpy_dtype(self.base_type))
        self.base_size = np.dtype(self.base_type).itemsize

        # column type of matrix in C order
        self.COLUMN = self.BASE_TYPE.Create_vector(
            self.shape[0], 1, self.shape[1])
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
        self.R2C_SEND = self.BASE_TYPE.Create_vector(
            self.count[1], 1, self.shape[0])
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
            self.R2C_RECV_RESIZED[i] = self.R2C_RECV[i].Create_resized(
                0, self.base_size)
            MPI.Datatype.Commit(self.R2C_RECV[i])
            MPI.Datatype.Commit(self.R2C_RECV_RESIZED[i])

        # send type c2r
        # local array is column contiguous
        self.C2R_SEND = self.BASE_TYPE.Create_vector(
            self.count[0], 1, self.shape[1])
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
            self.C2R_RECV_RESIZED[i] = self.C2R_RECV[i].Create_resized(
                0, self.base_size)
            MPI.Datatype.Commit(self.C2R_RECV[i])
            MPI.Datatype.Commit(self.C2R_RECV_RESIZED[i])

    def free_datatypes(self) -> None:
        """
        Free all datatypes.
        """
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
        """
        Scatter the data from the master rank to all other ranks.
        Creates a columnwise distribution of the data.
        """
        if inp is not None:
            assert np.all(inp.shape == self.shape)
        assert outp.shape[0] == self.count[0]
        assert outp.shape[1] == self.shape[1]

        if transpose_net:
            self.comm.Scatterv(
                [inp, self.counts[1, :], self.displacements[1, :], self.COLUMN_RESIZED], outp, root=0)
        else:
            if self.comm_rank == 0:
                inp_transposed = np.copy(inp.T, order="C")
            else:
                inp_transposed = None
            self.comm.Scatterv([inp_transposed, self.counts[1, :] * self.shape[0], self.displacements[1, :] * self.shape[0], self.BASE_TYPE],
                               outp,
                               root=0)

    def gather_master(self, inp: np.ndarray, outp: np.ndarray, transpose_net: bool = False) -> None:
        """
        Gather the data from all ranks to the master rank.
        Data is in a columnwise distribution.
        """
        if outp is not None:
            assert np.all(outp.shape == outp.shape)
        assert inp.shape[0] == self.count[1]
        assert inp.shape[1] == self.shape[0]
        if transpose_net:
            self.comm.Gatherv(inp, [
                              outp, self.counts[1, :], self.displacements[1, :], self.COLUMN_RESIZED], root=0)
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
        """
        Blocking communicates the data from a columnwise distribution to a rowwise distribution.
        """

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
        """
        Unblocking communicates the data from a columnwise distribution to a rowwise distribution.
        """
        if self.is_divisible:
            if transpose_net:
                request = self.comm.Ialltoallv([inp, self.counts[0, :], self.displacements[0, :],
                                                self.R2C_SEND_RESIZED],
                                               [outp, np.repeat([1], self.comm_size), self.displacements[1, :], self.R2C_RECV_RESIZED[0]])
            else:
                inp_transposed = np.copy(inp.T, order="C")
                request = self.comm.Ialltoallv([
                    inp_transposed, self.counts[0, :] * self.count[1], self.displacements[0, :] *
                    self.count[1],
                    self.BASE_TYPE,
                ], [outp, np.repeat([1], self.comm_size), self.displacements[1, :], self.R2C_RECV_RESIZED[0]])
        else:
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
        """
        Blocking communicates the data from a rowwise distribution to a columnwise distribution.
        """
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
        """
        Unblocking communicates the data from a rowwise distribution to a columnwise distribution.
        """
        if self.is_divisible:
            if transpose_net:
                request = self.comm.Ialltoallv([inp, self.counts[1, :], self.displacements[1, :],
                                                self.C2R_SEND_RESIZED],
                                               [outp, np.repeat([1], self.comm_size), self.displacements[0, :], self.C2R_RECV_RESIZED[0]])
            else:
                inp_transposed = np.copy(inp.T, order="C")
                request = self.comm.Ialltoallv([
                    inp_transposed, self.counts[1, :] * self.count[0], self.displacements[1, :] *
                    self.count[0],
                    self.BASE_TYPE
                ], [outp, np.repeat([1], self.comm_size), self.displacements[0, :], self.C2R_RECV_RESIZED[0]])
        else:
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
        """
        Allgather a 1D columnwise distributed array from all ranks to all ranks.
        Example: Gets called with local energy such that every rank has the whole energy grid.
        """
        assert inp.size == self.count[1]
        assert outp.size == self.shape[1]
        if otype is None:
            self.comm.Allgatherv(
                inp, [outp, self.counts[1, :], self.displacements[1, :], self.BASE_TYPE])
        else:
            mpi_otype = MPI.Datatype(dtlib.from_numpy_dtype(otype))
            self.comm.Allgatherv(
                inp, [outp, self.counts[1, :], self.displacements[1, :], mpi_otype])

    def gatherall_row(self, inp: np.ndarray, outp: np.ndarray, otype=None) -> None:
        """
        Allgather a 1D columnwise distributed array from all ranks to all ranks.
        Example: Gets called with local energy such that every rank has the whole energy grid.
        """
        assert inp.size == self.count[0]
        assert outp.size == self.shape[0]
        if otype is None:
            self.comm.Allgatherv(
                inp, [outp, self.counts[0, :], self.displacements[0, :], self.BASE_TYPE])
        else:
            mpi_otype = MPI.Datatype(dtlib.from_numpy_dtype(otype))
            self.comm.Allgatherv(
                inp, [outp, self.counts[0, :], self.displacements[0, :], mpi_otype])


class CommunicateCompute():
    """
    Wrapper class for communication and computation
    of/on a row or columnwise distributed matrix.
    After the computation on a row/column slice, the matrix is redistributed
    to a column/rowwise distribution.
    The communication can be either blocking or non-blocking.
    """

    def __init__(self, distributions: List[TransposeMatrix],
                 num_buffer: int, direction: str,
                 buffer_send, buffer_recv,
                 buffer_compute_unblock=None,
                 buffer_send_unblock=None,
                 buffer_recv_unblock=None,
                 comm_unblock: bool = False,
                 distributions_unblock: List[TransposeMatrix] = None,
                 batchsize=None,
                 iterations=None,
                 transpose_net=False) -> None:
        self.distributions = distributions
        self.num_buffer = num_buffer
        self.direction = direction
        self.buffer_send = buffer_send
        self.buffer_recv = buffer_recv
        self.comm_unblock = comm_unblock
        self.transpose_net = transpose_net

        if self.direction == "r2c":
            self.contiguous_dim = 0
            self.comm_func_block = [
                distribution.alltoall_r2c for distribution in distributions]
            self.comm_func_unblock = [
                distribution.ialltoall_r2c for distribution in distributions_unblock]
        elif self.direction == "c2r":
            self.contiguous_dim = 1
            self.comm_func_block = [
                distribution.alltoall_c2r for distribution in distributions]
            self.comm_func_unblock = [
                distribution.ialltoall_c2r for distribution in distributions_unblock]
        else:
            raise ValueError("Direction must be either r2c or c2r")
        # addional information for unblocking
        if (batchsize is None
            or iterations is None
            or buffer_compute_unblock is None
            or buffer_send_unblock is None
            or buffer_recv_unblock is None
                or distributions_unblock is None) and comm_unblock:
            raise ValueError(
                "If comm_unblock is True, all optional inputs must be given")
        if comm_unblock:
            if not all(distribution.is_divisible for distribution in distributions):
                raise ValueError(
                    "If comm_unblock is True, all distributions must be divisible by the grid")
            self.buffer_compute_unblock = buffer_compute_unblock
            self.buffer_send_unblock = buffer_send_unblock
            self.buffer_recv_unblock = buffer_recv_unblock
            self.distributions_unblock = distributions_unblock
            self.batchsize = batchsize
            self.iterations = iterations
            for i in range(self.num_buffer):
                assert self.buffer_compute_unblock[i].shape[0] == self.batchsize
                assert self.buffer_send_unblock[i].shape[0] == self.batchsize

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if self.comm_unblock:
                self.compute_communicate_unblocking(func, *args, **kwargs)
            else:
                self.compute_communicate_blocking(func, *args, **kwargs)
        return wrapper

    def compute_communicate_blocking(self, func, inp_block, inp):
        """
        Computes and communicates in a blocking fashion.
        """
        func(*inp_block, *inp, *self.buffer_send)
        for i in range(self.num_buffer):
            self.comm_func_block[i](
                self.buffer_send[i], self.buffer_recv[i], transpose_net=self.transpose_net)

    def compute_communicate_unblocking(self, func, inp_block, inp):
        """
        Computes and communicates in a non-blocking fashion.
        Batched compute and communication loop.
        """
        for distribution in self.distributions:
            assert distribution.shape[self.contiguous_dim] == self.distributions[0].shape[self.contiguous_dim]
            assert distribution.count[self.contiguous_dim] == self.distributions[0].count[self.contiguous_dim]
            assert distribution.comm_size == self.distributions[0].comm_size
            assert distribution.comm_rank == self.distributions[0].comm_rank
            assert distribution.is_divisible
            for j in range(distribution.comm_size):
                assert distribution.displacements[self.contiguous_dim,
                                                  j] == self.distributions[0].displacements[self.contiguous_dim, j]

        for i in range(0, self.iterations):

            idx_batch_start = i*self.batchsize
            idx_batch_end = min((i+1)*self.batchsize,
                                self.distributions[0].shape[self.contiguous_dim])

            # slice a range of the input
            inp_block_slice = [
                inp_block_i[idx_batch_start:idx_batch_end, ...] for inp_block_i in inp_block]

            func(*inp_block_slice, *inp, *self.buffer_compute_unblock)
            for k in range(self.num_buffer):
                # todo replace with pointer swap
                self.buffer_send_unblock[k][:] = self.buffer_compute_unblock[k]
            if i > 0:
                MPI.Request.Waitall(requests)
                for j in range(self.distributions[k].comm_size):
                    idx_start = self.distributions[0].displacements[self.contiguous_dim, j] + \
                        (i-1) * self.batchsize
                    idx_range = self.batchsize
                    for k in range(self.num_buffer):
                        self.buffer_recv[k][:, idx_start:idx_start+idx_range] = \
                            self.buffer_recv_unblock[k][:, j *
                                                        self.batchsize:j*self.batchsize+idx_range]

            requests = []
            for k in range(self.num_buffer):
                requests.append(self.comm_func_unblock[k](
                    self.buffer_send_unblock[k], self.buffer_recv_unblock[k], transpose_net=self.transpose_net))

        MPI.Request.Waitall(requests)
        for j in range(self.distributions[k].comm_size):
            idx_start = self.distributions[0].displacements[self.contiguous_dim, j] + \
                (self.iterations-1) * self.batchsize
            if self.distributions[0].count[self.contiguous_dim] % self.batchsize == 0:
                idx_range = self.batchsize
            else:
                idx_range = self.distributions[0].count[self.contiguous_dim] % self.batchsize
            for k in range(self.num_buffer):
                self.buffer_recv[k][:, idx_start:idx_start+idx_range] = \
                    self.buffer_recv_unblock[k][:, j *
                                                self.batchsize:j*self.batchsize+idx_range]
