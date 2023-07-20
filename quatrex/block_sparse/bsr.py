# Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.
"""
An extension to SciPy's sparse bsr_array type with other useful methods.

"""
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from numpy.typing import ArrayLike


class bsr(sp.bsr_array):
    """A block sparse row matrix.

    This class extends scipy.sparse.bsr_array with some useful methods.

    """

    def __init__(self, arg1, shape=None, dtype=None, copy=False, blocksize=None):
        """Initializes a block sparse matrix."""
        super().__init__(
            arg1,
            shape=shape,
            dtype=dtype,
            copy=copy,
            blocksize=blocksize,
        )

    def __getitem__(self, key):
        """Returns the matrix value at the given row and column."""
        item = self.tocsr()[key]
        if isinstance(item, sp.spmatrix):
            return bsr(item)
        return item

    def _get_col_indices(self, row: int) -> np.ndarray:
        """Returns the indices of the blocks in the given block row."""
        return self.indices[self.indptr[row]:self.indptr[row + 1]]

    def _get_data_blocks(self, row: int) -> np.ndarray:
        """Returns the data of the blocks in the given block row."""
        return self.data[self.indptr[row]:self.indptr[row + 1]]

    def _check_block_in_bounds(self, row: int, col: int):
        """Checks if the given block is in bounds."""
        m, n = self.shape
        r, c = self.blocksize
        if row >= m // r or col >= n // c:
            raise IndexError("Block index out of bounds.")

    def _unsign_block_indices(self, row: int, col: int) -> tuple[int, int]:
        """Calculates effective positive index for negative indices."""
        m, n = self.shape
        r, c = self.blocksize

        if row < 0:
            row = m // r + row
        if col < 0:
            col = n // c + col
        if row < 0 or col < 0:
            raise IndexError("Block index out of bounds.")

        return row, col

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
            return np.zeros(self.blocksize)

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
        if block.shape != self.blocksize:
            raise ValueError("Invalid block size.")

        col_indices = self._get_col_indices(row)

        if col not in col_indices:
            raise NotImplementedError("Changing the sparsity pattern is not supported.")

        block_index = np.where(col_indices == col)[0][0]
        self.data[self.indptr[row] + block_index] = block

    def set_blocksize(self, blocksize: tuple[int, int], enforce: bool = False) -> "bsr":
        """Sets the block size of the matrix.

        This essentially returns a new matrix with the given block size.

        Parameters
        ----------
        blocksize : tuple of int
            The new block size.
        enforce : bool, optional
            If True, the block size is enforced. Any blocks that do not
            fit the new block size are discarded. Default is False. This
            can be rather expensive.
        """
        try:
            return bsr(self, blocksize=blocksize)
        except TypeError:
            # For some reason, setting an incompatible blocksize raises
            # a TypeError instead of a ValueError.
            if not enforce:
                raise ValueError(f"Invalid block size {blocksize}. "
                                 "Use enforce=True to enforce the block size.")
            return self._enforce_blocksize(blocksize)

    def _enforce_blocksize(self, blocksize: tuple[int, int]) -> "bsr":
        """Enforces the given block size.

        This discards any blocks that do not fit the new block size.

        """
        m, n = self.shape
        r, c = blocksize

        new_m = (m // r) * r
        new_n = (n // c) * c

        return self[:new_m, :new_n].set_blocksize(blocksize, enforce=False)

    def make_periodic(self) -> "bsr":
        """Makes the matrix periodic.

        This essentially overwrites the first block with the second one
        and the last block with the second last one.

        Returns
        -------
        bsr
            The periodic matrix.

        """
        self.set_block(0, 0, self.get_block(1, 1))
        self.set_block(-1, -1, self.get_block(-2, -2))

        return self.copy()

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
        if kwargs.pop("ax", None) is None:
            __, ax = plt.subplots()

        ax.matshow(np.abs(self.toarray()), **kwargs)
        ax.set_aspect("equal")
        ax.set_yticks(np.arange(-0.5, self.shape[0] - 0.5, step=self.blocksize[0]))
        ax.set_yticklabels(np.arange(0, self.shape[0], step=self.blocksize[0]))
        ax.set_xticks(np.arange(-0.5, self.shape[1] - 0.5, step=self.blocksize[1]))
        ax.set_xticklabels(np.arange(0, self.shape[1], step=self.blocksize[1]))
        ax.grid(which="both", color="black", linestyle="-")

        return ax

    @classmethod
    def diag(
        cls,
        blocks: ArrayLike,
        overlap: int = 0,
        blocksize: tuple[int, int] = None,
    ) -> "bsr":
        """Creates a matrix from the given blocks with an overlap.

        In analogy to scipy.block_diag, this function creates a block
        diagonal matrix from the given arrays. The overlap of the
        matrices is given by the last n rows and columns of the first
        matrix and the first n rows and columns of the second matrix.

        Parameters
        ----------
        blocks : list of array_like
            The blocks of the matrix.
        overlap : int, optional
            The overlap of the blocks. Default is 0.

        """
        blocks = [np.asarray(block) for block in blocks]
        shapes = np.array([block.shape for block in blocks])
        out_dtype = np.find_common_type([block.dtype for block in blocks], [])
        total_overlap = (len(blocks) - 1) * overlap
        out_shape = tuple(np.sum(shapes, axis=0) - total_overlap)
        out = sp.lil_array(out_shape, dtype=out_dtype)

        r, c = 0, 0
        for i, (rr, cc) in enumerate(shapes):
            out[r:r + rr, c:c + cc] += blocks[i]
            r += rr - overlap
            c += cc - overlap
        return bsr(out, blocksize=blocksize)
