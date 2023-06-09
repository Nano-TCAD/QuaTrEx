import numpy as np
import scipy.sparse as sp

from typing import Sequence, Union


class bsr_matrix(object):

    _matrix: sp.bsr_matrix

    def __init__(self, matrix: sp.bsr_matrix):
        self._matrix = matrix

    @property
    def data(self):
        return self._matrix.data

    @property
    def indices(self):
        return self._matrix.indices

    @property
    def indptr(self):
        return self._matrix.indptr

    @property
    def shape(self):
        return self._matrix.shape

    @property
    def blocksize(self):
        return self._matrix.blocksize

    @property
    def ndim(self):
        return self._matrix.ndim

    @property
    def nnz(self):
        return self._matrix.nnz

    @property
    def dtype(self):
        return self._matrix.dtype

    @property
    def T(self):
        return bsr_matrix(self._matrix.T)

    def __getitem__(self, key: Sequence[slice]):
        """ Returns a single block of the matrix. """

        frow, lrow = key[0].start or 0, key[0].stop or self.shape[0]
        fcol, lcol = key[1].start or 0, key[1].stop or self.shape[1]
        assert lrow - frow == self.blocksize[0]
        assert lcol - fcol == self.blocksize[1]

        brow, bcol = frow // self.blocksize[0], fcol // self.blocksize[1]
        bsize = self.blocksize[0] * self.blocksize[1]

        data_idx = -1
        for i in range(self.indptr[brow], self.indptr[brow + 1]):
            if self.indices[i] == bcol:
                data_idx = i
                break
        assert data_idx >= 0

        return self.data[data_idx]

    def copy(self):
        return bsr_matrix(self._matrix.copy())

    def real(self):
        data = np.real(self.data)
        indices = self.indices.copy()
        indptr = self.indptr.copy()
        return bsr_matrix(sp.bsr_matrix((data, indices, indptr), shape=self.shape, blocksize=self.blocksize))

    def imag(self):
        data = np.imag(self.data)
        indices = self.indices.copy()
        indptr = self.indptr.copy()
        return bsr_matrix(sp.bsr_matrix((data, indices, indptr), shape=self.shape, blocksize=self.blocksize))

    def conj(self):
        return bsr_matrix(self._matrix.conj())

    def conjugate(self):
        return bsr_matrix(self._matrix.conjugate())

    def transpose(self):
        return bsr_matrix(self._matrix.transpose())

    def __mul__(self, other):
        data = self.data * other
        indices = self.indices.copy()
        indptr = self.indptr.copy()
        return bsr_matrix(sp.bsr_matrix((data, indices, indptr), shape=self.shape, blocksize=self.blocksize))

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        data = self.data / other
        indices = self.indices.copy()
        indptr = self.indptr.copy()
        return bsr_matrix(sp.bsr_matrix((data, indices, indptr), shape=self.shape, blocksize=self.blocksize))

    def __rtruediv__(self, other):
        data = other / self.data
        indices = self.indices.copy()
        indptr = self.indptr.copy()
        return bsr_matrix(sp.bsr_matrix((data, indices, indptr), shape=self.shape, blocksize=self.blocksize))

    def __add__(self, other):
        if isinstance(other, bsr_matrix):
            other = other._matrix
        return bsr_matrix(self._matrix + other)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, bsr_matrix):
            other = other._matrix
        return bsr_matrix(self._matrix - other)

    def __rsub__(self, other):
        if isinstance(other, bsr_matrix):
            other = other._matrix
        return bsr_matrix(other - self._matrix)

    def _dot(self, A, B):

        indptr = np.empty_like(A.indptr)
        indptr[0] = 0
        indices = []

        for i in range(A.shape[0] // A.blocksize[0]):
            current_length = len(indices)
            for j in range(B.shape[1] // B.blocksize[1]):
                found = False
                ka = A.indptr[i]
                while not found and ka < A.indptr[i + 1]:
                    k = A.indices[ka]
                    for kb in range(B.indptr[k], B.indptr[k + 1]):
                        if B.indices[kb] == j:
                            indices.append(j)
                            found = True
                            break
                    ka += 1
            indptr[i + 1] = indptr[i] + len(indices) - current_length

        nnz = len(indices)
        indices = np.array(indices, dtype=np.int32)
        data = np.empty((nnz, *A.blocksize), dtype=A.dtype)

        for i in range(A.shape[0] // A.blocksize[0]):
            for jidx in range(indptr[i], indptr[i + 1]):
                j = indices[jidx]
                out = None
                for ka in range(A.indptr[i], A.indptr[i + 1]):
                    k = A.indices[ka]
                    for kb in range(B.indptr[k], B.indptr[k + 1]):
                        if B.indices[kb] == j:
                            if out is not None:
                                out[:] += A.data[ka] @ B.data[kb]
                            else:
                                out = data[jidx]
                                out[:] = A.data[ka] @ B.data[kb]

        return bsr_matrix(
            sp.bsr_matrix((data, indices, indptr), shape=(A.shape[0], B.shape[1]), blocksize=self.blocksize))

    def __matmul__(self, other):
        if isinstance(other, bsr_matrix):
            other = other._matrix
        if isinstance(other, sp.bsr_matrix) and self.blocksize == other.blocksize:
            return self._dot(self, other)
        return bsr_matrix(self._matrix @ other)

    def __rmatmul__(self, other):
        if isinstance(other, bsr_matrix):
            other = other._matrix
        if isinstance(other, sp.bsr_matrix) and self.blocksize == other.blocksize:
            return self._dot(other, self)
        return bsr_matrix(other @ self._matrix)
