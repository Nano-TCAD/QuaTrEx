"""
An implementation of a variable block sparse row matrix.

"""

import sys
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from numpy.typing import ArrayLike


class vbsr:
    """A variable block sparse row (VBSR) matrix.

    This is a square matrix with aligned blocks of different sizes.
    Naturally, all blocks on the diagonal are square.

    Parameters
    ----------
    data : ArrayLike
        Data of the matrix. This is a list or an array of different
        sized blocks.
    indices : ArrayLike
        Column indices.
    indptr : ArrayLike
        Row index pointers.
    blocksizes : ArrayLike, optional
        Size of the blocks in the matrix. If not given, the blocksizes
        are inferred from the data. To infer the block sizes, the
        provided diagonal blocks must be square and non-zero.

    Attributes
    ----------
    dtype : np.dtype
        Data type of the matrix blocks.
    data : list[np.ndarray]
        Data of the matrix. Rather than a single array, this is a list
        of arrays of different sizes.
    indices : np.ndarray[int]
        Column indices of the matrix.
    indptr : np.ndarray[int]
        Index pointers of the matrix.
    blocksizes : np.ndarray[int]
        Size of each of the blocks in the matrix.
    nnz : int
        Number of nonzero elements in the matrix.
    size : int
        Size of the matrix, i.e. the number of rows/columns of the total
        square matrix.
    shape : tuple[int, int]
        Shape of the matrix. This is the same as (size, size).
    blockoffsets : np.ndarray[int]
        Offsets of the blocks in the matrix.
    num_blocks : int
        Number of block rows/columns in the matrix.
    nbytes : int
        Number of bytes of the matrix.

    """

    def __init__(
        self,
        data: ArrayLike,
        indices: ArrayLike,
        indptr: ArrayLike,
        blocksizes: ArrayLike = None,
    ) -> None:
        """Initializes a VBSR matrix."""
        self.dtype = np.find_common_type([block.dtype for block in data], [])
        # NOTE: Using ndarrays to store ragged nested sequences is
        # deprecated (see NEP 34). Best option is to use lists.
        self.data = [np.asarray(block, dtype=self.dtype) for block in data]
        self.indices = np.asarray(indices, dtype=int)
        self.indptr = np.asarray(indptr, dtype=int)

        self.num_blocks = self.indptr.size - 1
        self.nnz = sum(block.size for block in self.data)
        self.nbytes = (
            sys.getsizeof(self.data)
            + sum(block.nbytes for block in self.data)
            + self.indices.nbytes
            + self.indptr.nbytes
        )

        if blocksizes is None:
            self.blocksizes = self._get_blocksizes()
            # Currently, to infer the block sizes, the diagonal blocks
            # must be square and non-zero.
            # TODO: There is another way to infer block size from the
            # data. If the blocks are aligned, the block sizes can be
            # inferred from the data.
            self._check_diagonal()
        else:
            self.blocksizes = np.asarray(blocksizes, dtype=int)

        self.blockoffsets = np.cumsum(self.blocksizes) - self.blocksizes

        self.size = sum(self.blocksizes)
        self.shape = (self.size, self.size)
        # Check that the matrix is valid.
        self._check_alignment()
        self._check_indices()

    def __str__(self) -> str:
        """Returns a string representation of the matrix."""
        return (
            f"{self.shape[0]}x{self.shape[1]} VBSR matrix of type '{self.dtype}'"
            f" with {self.nnz} nonzero elements ({len(self.data)} blocks)."
        )

    def __repr__(self) -> str:
        """Returns a string representation of the matrix."""
        return self.__str__()

    def _get_col_indices(self, row: int) -> np.ndarray:
        """Returns the indices of the blocks in the given block row."""
        return self.indices[self.indptr[row] : self.indptr[row + 1]]

    def _get_data_blocks(self, row: int) -> list[np.ndarray]:
        """Returns the data of the blocks in the given block row."""
        return self.data[self.indptr[row] : self.indptr[row + 1]]

    def _check_diagonal(self) -> None:
        """Checks that all diagonal blocks are non-zero and square."""
        if any(np.diff(self.indptr) == 0):
            raise ValueError("Blocks on diagonal must be non-zero to infer blocksize.")
        if any(block.shape[0] != block.shape[1] for block in self.diagonal_blocks()):
            raise ValueError("Blocks on diagonal must be square to infer blocksize.")

    def _check_block_in_bounds(self, row: int, col: int) -> None:
        """Checks that the block index is in bounds."""
        if row >= self.num_blocks or col >= self.num_blocks:
            raise IndexError("Block index out of bounds.")

    def _check_alignment(self) -> None:
        """Checks that blocks are aligned correctly."""
        for row, blocksize in enumerate(self.blocksizes):
            blocks = self._get_data_blocks(row)
            if len(blocks) == 0:  # No blocks in this row.
                continue
            shapes = np.array([block.shape for block in blocks])
            if not np.all(shapes[:, 0] == blocksize):
                raise ValueError("Inconsistent row alignment.")
            indices = self._get_col_indices(row)
            if not np.all(shapes[:, 1] == self.blocksizes[indices]):
                raise ValueError("Inconsistent column alignment.")

    def _check_indices(self) -> None:
        """Checks that all indices are in bounds."""
        if any(self.indices < 0):
            raise ValueError("All indices must be positive.")
        if any(self.indices >= self.num_blocks):
            raise ValueError("All indices must be less than the number of blocks.")

    def _unsign_block_indices(self, row: int, col: int) -> tuple[int, int]:
        """Calculates effective positive index for negative indices."""
        if row < 0:
            row = self.num_blocks + row
        if col < 0:
            col = self.num_blocks + col
        if row < 0 or col < 0:
            raise IndexError("Block index out of bounds.")

        return row, col

    def _get_blocksizes(self) -> np.ndarray:
        """Calculates the block sizes from the diagonal blocks."""
        # TODO: This is a naive implementation. It is possible to infer
        # the block sizes from the data without the diagonal blocks
        # being square and non-zero.
        blocksizes = np.zeros(self.num_blocks, dtype=int)
        for i in range(self.num_blocks):
            blocksizes[i] = self.data[self.indptr[i]].shape[0]

        return blocksizes

    def get_block(self, row: int, col: int) -> np.ndarray:
        """Returns the block at the given row and column.

        If the block is not present in the sparse matrix, a zero block
        of the appropriate shape is returned.

        Supports negative indices.

        Parameters
        ----------
        row : int
            The row index of the block.
        col : int
            The column index of the block.

        Returns
        -------
        block : np.ndarray
            The block at the given row and column.

        """
        row, col = self._unsign_block_indices(row, col)
        self._check_block_in_bounds(row, col)

        col_indices = self._get_col_indices(row)
        if col not in col_indices:
            blockshape = (self.blocksizes[row], self.blocksizes[col])
            return np.zeros(blockshape, dtype=self.dtype)

        blocks = self._get_data_blocks(row)
        block_index = np.where(col_indices == col)[0][0]
        return blocks[block_index]

    def get_blocks(self, *indices: ArrayLike) -> tuple[np.ndarray]:
        """Returns the blocks at the given indices."""
        indices = np.asarray(indices)
        if not indices.ndim == 2:
            raise ValueError("Indices must be 2D.")

        if len(indices) == 1:
            return self.get_block(*indices[0])

        return tuple(self.get_block(*index) for index in indices)

    def set_block(self, row: int, col: int, block: ArrayLike) -> None:
        """Sets the block at the given row and column.

        It is not possible to change the sparsity pattern of the matrix.

        """
        row, col = self._unsign_block_indices(row, col)
        self._check_block_in_bounds(row, col)

        block = np.asarray(block)
        if block.shape != (self.blocksizes[row], self.blocksizes[col]):
            raise ValueError("Invalid block size.")

        col_indices = self._get_col_indices(row)

        if col not in col_indices:
            raise NotImplementedError("Changing the sparsity pattern is not supported.")

        block_index = np.where(col_indices == col)[0][0]
        self.data[self.indptr[row] + block_index] = block

    def diagonal_blocks(self, k: int = 0) -> np.ndarray:
        """Returns the k-th block-diagonal.

        Parameters
        ----------
        k : int, optional
            The block-diagonal to return. Defaults to 0, i.e. the main
            diagonal.

        Returns
        -------
        blocks : np.ndarray
            The blocks on the k-th block-diagonal.

        """
        if abs(k) >= self.num_blocks:
            raise IndexError("Block index out of bounds.")
        blocks = [self.get_block(i, i + k) for i in range(self.num_blocks - abs(k))]
        return np.array(blocks, dtype=object)

    def show(self, **kwargs) -> plt.Axes:
        """Displays the absolute value of the matrix.

        Parameters
        ----------
        ax : plt.Axes, optional
            The axes to plot on. If not given, a new figure is created.
        kwargs : dict, optional
            Additional keyword arguments are passed to `plt.matshow`.

        Returns
        -------
        ax : plt.Axes
            The axes on which the matrix is plotted.

        """
        ax = kwargs.pop("ax", None)
        if ax is None:
            __, ax = plt.subplots()

        ax.matshow(np.abs(self.toarray()), **kwargs)
        ax.set_aspect("equal")
        ticks = self.blockoffsets - 0.5
        ticklabels = self.blockoffsets
        ax.set_xticks(ticks)
        ax.set_xticklabels(ticklabels)
        ax.set_yticks(ticks)
        ax.set_yticklabels(ticklabels)
        ax.grid(which="both", color="black", linestyle="-")

        return ax

    def _tosliceable(self, init: Callable):
        """Converts the sparse matrix to a sliceable matrix."""
        mat = init(self.shape, dtype=self.dtype)
        for row, m in enumerate(self.blockoffsets):
            col_indices = self._get_col_indices(row)
            blocks = self._get_data_blocks(row)
            for col, block in zip(col_indices, blocks):
                n = np.sum(self.blocksizes[:col])
                mat[m : m + block.shape[0], n : n + block.shape[1]] = block

        return mat

    def tolil(self) -> sp.lil_array:
        """Converts the sparse matrix to a LIL matrix."""
        return self._tosliceable(sp.lil_array)

    def toarray(self) -> np.ndarray:
        """Converts the sparse matrix to a dense array."""
        return self._tosliceable(np.zeros)

    @staticmethod
    def _blocksize_heuristic(arr: ArrayLike) -> np.ndarray:
        """Returns blocksizes for the given array."""
        # Start by constructing a sparsity mask.
        mask = np.isclose(arr, 0.0)
        # Determine where zeros start along each row.
        transients = np.argwhere(np.diff(mask, axis=1))[:, 1]
        unique, counts = np.unique(transients, return_counts=True)
        # Remove spurious transients that occur only once.
        # NOTE: This should probably be somehow dependent on matrix size
        # and the average count. Oh well.
        unique = unique[counts > 1]
        # Determine block sizes from these block offsets.
        blockoffsets = np.concatenate(([0], unique + 1, [arr.shape[0]]))
        blocksizes = np.diff(blockoffsets)
        return blocksizes

    @classmethod
    def from_array(cls, arr: ArrayLike, blocksizes: ArrayLike = None) -> "vbsr":
        """Creates a matrix from an array.

        Parameters
        ----------
        arr : ArrayLike
            The array to convert to a vbsr matrix.
        blocksizes : ArrayLike, optional
            The block sizes. If not given, they are determined by a
            heuristic.

        Returns
        -------
        mat : vbsr
            The vbsr matrix.

        """
        if arr.ndim != 2:
            raise ValueError("Array must be two-dimensional.")
        if arr.shape[0] != arr.shape[1]:
            raise ValueError("Array must be square.")

        if blocksizes is None:
            blocksizes = cls._blocksize_heuristic(arr)

        blockoffsets = np.cumsum(blocksizes) - blocksizes
        data = []
        indices = []
        indptr = [0]
        for i, m in enumerate(blockoffsets):
            for j, n in enumerate(blockoffsets):
                block = arr[m : m + blocksizes[i], n : n + blocksizes[j]]
                if not np.allclose(block, 0.0):
                    data.append(block)
                    indices.append(j)
            indptr.append(len(data))

        return cls(data, indices, indptr, blocksizes)

    @classmethod
    def from_spmatrix(cls, mat: sp.spmatrix, blocksizes: ArrayLike) -> "vbsr":
        """Creates a matrix from any sp.spmatrix.

        Since lil_arrays are sliceable, this just calls 'mat.tolil()'
        first. No blocksize heuristic is implemented.

        Parameters
        ----------
        mat : sp.spmatrix
            The spmatrix to convert to a vbsr matrix.
        blocksizes : ArrayLike
            The block sizes.

        Returns
        -------
        mat : vbsr
            The vbsr matrix.

        """
        mat = mat.tolil()
        if mat.shape[0] != mat.shape[1]:
            raise ValueError("Array must be square.")

        blockoffsets = np.cumsum(blocksizes) - blocksizes
        data = []
        indices = []
        indptr = [0]
        for i, m in enumerate(blockoffsets):
            for j, n in enumerate(blockoffsets):
                block = mat[m : m + blocksizes[i], n : n + blocksizes[j]]
                if block.nnz > 0:
                    data.append(block.toarray())
                    indices.append(j)
            indptr.append(len(data))

        return cls(data, indices, indptr, blocksizes)

    @classmethod
    def diag(cls, blocks: ArrayLike, overlap: int = 0) -> "vbsr":
        """Creates a matrix from the given blocks with an overlap.

        In analogy to scipy.block_diag, this function creates a block
        diagonal matrix from the given arrays. The overlap of the
        matrices is given by the last n rows and columns of the first
        matrix and the first n rows and columns of the second matrix.

        Parameters
        ----------
        blocks : list of array_like
            The diagonal blocks of the matrix. Squareness is assumed.
        overlap : int, optional
            The overlap of the blocks. Default is 0.

        """
        blocks = [np.asarray(block) for block in blocks]
        sizes = np.array([block.shape[0] for block in blocks])
        out_dtype = np.find_common_type([block.dtype for block in blocks], [])
        total_overlap = (len(blocks) - 1) * overlap
        out_size = np.sum(sizes, axis=0) - total_overlap
        out = sp.lil_array((out_size, out_size), dtype=out_dtype)

        blockoffsets = [0]
        m = 0
        for i, size in enumerate(sizes):
            out[m : m + size, m : m + size] += blocks[i]
            m += size - overlap
            blockoffsets.append(m)

        blocksizes = np.diff(blockoffsets)
        return cls.from_spmatrix(out, blocksizes=blocksizes)
