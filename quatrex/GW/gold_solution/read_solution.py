"""Loads the gold solution created with MATLAB codebase
    and creates a mapping to transposed matrix"""
import h5py
import numpy as np
import numpy.typing as npt
import typing
from scipy import sparse

def load_x(
    path: str,
    tensor_name: str
) -> typing.Tuple[
        npt.NDArray[np.double],
        npt.NDArray[np.int32],
        npt.NDArray[np.int32],
        npt.NDArray[np.complex128],
        npt.NDArray[np.complex128],
        npt.NDArray[np.complex128]
        ]:
    """Loads Green's Functions from .mat file created by changeFormatGPWS.m or changeFormatGP.m 
        All sparse matrices are in the COO format and 
        share the same idx of non-zero elements.
        Such the rows, columns are only saved once.

    Args:
        path (str): filepath to .mat file
        tensor_name (str): can be either "g"/"p"/"w"/"s" to read 
        Green's Function, Polarization, W, Sigma

    Returns:
        typing.Tuple[ 
        npt.NDArray[np.double]:     Energy 
        npt.NDArray[np.int32]:      Rows 
        npt.NDArray[np.int32]:      Columns 
        npt.NDArray[np.complex128]: Greater Function
        npt.NDArray[np.complex128]: Lesser Function
        npt.NDArray[np.complex128]: Retarded Function 
    """

    # open the file
    hdf5 = h5py.File(path, "r")["formatted"]

    # read the entries
    energy:     npt.NDArray[np.double] = np.array(hdf5["E"], dtype=np.double)
    # go from matlab to python
    rows:       npt.NDArray[np.int32]  = np.array(hdf5["rows"], dtype=np.int32) - 1
    columns:    npt.NDArray[np.int32]  = np.array(hdf5["columns"], dtype=np.int32) - 1

    energy:     npt.NDArray[np.double] = np.squeeze(energy)
    rows:       npt.NDArray[np.int32]  = np.squeeze(rows)
    columns:    npt.NDArray[np.int32]  = np.squeeze(columns)


    realg: npt.NDArray[np.double] = np.array(hdf5["real"   + tensor_name + "g"], dtype=np.double)
    imgg:  npt.NDArray[np.double] = np.array(hdf5["img"    + tensor_name + "g"], dtype=np.double)
    reall: npt.NDArray[np.double] = np.array(hdf5["real"   + tensor_name + "l"], dtype=np.double)
    imgl:  npt.NDArray[np.double] = np.array(hdf5["img"    + tensor_name + "l"], dtype=np.double)
    realr: npt.NDArray[np.double] = np.array(hdf5["real"   + tensor_name + "r"], dtype=np.double)
    imgr:  npt.NDArray[np.double] = np.array(hdf5["img"    + tensor_name + "r"], dtype=np.double)

    xg = realg + 1j * imgg
    xl = reall + 1j * imgl
    xr = realr + 1j * imgr

    return (energy, rows, columns, xg, xl, xr)

def load_v(
    path: str
) -> typing.Tuple[
        npt.NDArray[np.int32],
        npt.NDArray[np.int32],
        npt.NDArray[np.complex128]
        ]:
    """Loads v hat/interaction from .mat file created by changeFormatV.m
        All sparse matrices are in the COO format and 
        share the same idx of non-zero elements.
        Such the rows, columns are only saved once.

    Args:
        path (str): filepath to .mat file

    Returns:
        typing.Tuple[ 
        npt.NDArray[np.int32]:      Rows 
        npt.NDArray[np.int32]:      Columns 
        npt.NDArray[np.complex128]: Interaction hat
    """

    # open the file
    hdf5 = h5py.File(path, "r")["formatted"]

    # read the entries
    # go from matlab to python
    rows:       npt.NDArray[np.int32]   = np.array(hdf5["rows"], dtype=np.int32) - 1
    columns:    npt.NDArray[np.int32]   = np.array(hdf5["columns"], dtype=np.int32) - 1

    realvh: npt.NDArray[np.double]      = np.array(hdf5["realvh"], dtype=np.double)
    imgvh:  npt.NDArray[np.double]      = np.array(hdf5["imgvh"], dtype=np.double)
    vh:     npt.NDArray[np.complex128]  = realvh + 1j * imgvh
    vh:     npt.NDArray[np.complex128]  = np.squeeze(vh)

    return (rows, columns, vh)

def load_B(
    path: str
) -> typing.Tuple[
        npt.NDArray[np.int32],
        npt.NDArray[np.int32]
        ]:
    """Loads Bmax and Bmin from .mat file created by changeFormatGPWS.m or changeFormatGP.m 

    Args:
        path (str): filepath to .mat file

    Returns:
        typing.Tuple[ 
        npt.NDArray[np.int32]: Bmax, vector of end indexes of the blocks
        npt.NDArray[np.int32]: Bmin, vector of start indexes of the blocks 
    """

    # open the file
    hdf5 = h5py.File(path, "r")["formatted"]

    # read the entries
    # go from matlab to python
    bmax: npt.NDArray[np.int32]  = np.array(hdf5["Bmax"], dtype=np.int32) - 1
    bmin: npt.NDArray[np.int32]  = np.array(hdf5["Bmin"], dtype=np.int32) - 1
    bmax: npt.NDArray[np.int32]  = np.squeeze(bmax)
    bmin: npt.NDArray[np.int32]  = np.squeeze(bmin)
    return (bmax, bmin)


def load_pg(
    path: str
) -> typing.Tuple[
        npt.NDArray[np.double], npt.NDArray[np.int32], npt.NDArray[np.int32],
        typing.Tuple[npt.NDArray[np.double], npt.NDArray[np.double],
                     npt.NDArray[np.double], npt.NDArray[np.double],
                     npt.NDArray[np.double], npt.NDArray[np.double]],
        typing.Tuple[npt.NDArray[np.double], npt.NDArray[np.double],
                     npt.NDArray[np.double], npt.NDArray[np.double],
                     npt.NDArray[np.double], npt.NDArray[np.double]]]:
    """Loads data from .mat file created by changeFormatGP.m
        All sparse matrices are in the COO format and 
        share the same idx of non-zero elements.
        Such the rows, columns are only saved once.

    Args:
        path (str): filepath to .mat file

    Returns:
        typing.Tuple[ 
        npt.NDArray[np.double]: energy, 
        npt.NDArray[np.int32]:   rows, 
        npt.NDArray[np.int32]:   columns, 
        typing.Tuple[npt.NDArray[np.double],
                     npt.NDArray[np.double], 
                     npt.NDArray[np.double], 
                     npt.NDArray[np.double], 
                     npt.NDArray[np.double], 
                     npt.NDArray[np.double]]: greens function, 
        typing.Tuple[npt.NDArray[np.double], 
                     npt.NDArray[np.double], 
                     npt.NDArray[np.double], 
                     npt.NDArray[np.double], 
                     npt.NDArray[np.double], 
                     npt.NDArray[np.double]]]: polarization                 
        Inner Tuples ordered real/img of greater/lesser/retarded
    """

    # open the file
    hdf5 = h5py.File(path, "r")["formatted"]

    # read the entries
    
    energy:     npt.NDArray[np.double] = np.array(hdf5["E"], dtype=np.double)
    # go from matlab to python
    rows:       npt.NDArray[np.int32]  = np.array(hdf5["rows"], dtype=np.int32) - 1
    columns:    npt.NDArray[np.int32]  = np.array(hdf5["columns"], dtype=np.int32) - 1

    realgg: npt.NDArray[np.double] = np.array(hdf5["realgg"], dtype=np.double)
    imggg:  npt.NDArray[np.double] = np.array(hdf5["imggg"], dtype=np.double)
    realgl: npt.NDArray[np.double] = np.array(hdf5["realgl"], dtype=np.double)
    imggl:  npt.NDArray[np.double] = np.array(hdf5["imggl"], dtype=np.double)
    realgr: npt.NDArray[np.double] = np.array(hdf5["realgr"], dtype=np.double)
    imggr:  npt.NDArray[np.double] = np.array(hdf5["imggr"], dtype=np.double)

    realpg: npt.NDArray[np.double] = np.array(hdf5["realpg"], dtype=np.double)
    imgpg:  npt.NDArray[np.double] = np.array(hdf5["imgpg"], dtype=np.double)
    realpl: npt.NDArray[np.double] = np.array(hdf5["realpl"], dtype=np.double)
    imgpl:  npt.NDArray[np.double] = np.array(hdf5["imgpl"], dtype=np.double)
    realpr: npt.NDArray[np.double] = np.array(hdf5["realpr"], dtype=np.double)
    imgpr:  npt.NDArray[np.double] = np.array(hdf5["imgpr"], dtype=np.double)

    gf  = (realgg, imggg, realgl, imggl, realgr, imggr)
    pol = (realpg, imgpg, realpl, imgpl, realpr, imgpr)
    return (energy, rows, columns, gf, pol)


def sparse_to_dense(rows: npt.NDArray[np.int32], 
                    columns: npt.NDArray[np.int32], 
                    data: npt.NDArray[np.complex128]
) -> npt.NDArray[np.complex128]:
    """Transform the sparse format rows, columns, 
        data to a dense format (#energy, #orbitals, #orbitals).
        Format of input is coo, but data is 2D 
        (meaning #energy * sparse matrices with all the same position of nnz elements)

    Args:
        rows (npt.NDArray[np.int32]):       rows idx of sparse matrix (nnz)
        columns (npt.NDArray[np.int32]):    column idx of sparse matrix (nnz)
        data (npt.NDArray[np.complex128]):     data vector (nnz, #energy)

    Returns:
        npt.NDArray[np.complex128]: dense matrix (#energy, nao, nao)
    """

    # number of energy points
    ne: np.int32 = np.shape(data)[1]
    no: np.int32 = np.shape(data)[0]

    # same number of rows, columns, nnz elements
    assert no == np.shape(rows)[0]
    assert no == np.shape(columns)[0]


    data_1 = sparse.coo_matrix((data[:, 0], (rows, columns)), dtype=np.complex128)

    # assert square matrix
    assert data_1.shape[0] == data_1.shape[1]

    out: npt.NDArray[np.complex128] = np.empty((ne, data_1.shape[0], data_1.shape[1]), dtype=np.complex128)

    for i in range(ne):
        out[i, :, :] = sparse.coo_matrix(
            (data[:, i], (rows, columns)),
            dtype=np.complex128).todense("C")

    return out


def find_idx_transposed(rows: npt.NDArray[np.int32],
                        columns: npt.NDArray[np.int32]) -> npt.NDArray[np.int32]:
    """Creates vector with mapping between element ij and ji of a sparse matrix.
        It is assumed that the matrix is either symmetric/hermitian such that if 
        ij exists also ji is non-zero.

    Args:
        rows (npt.NDArray[np.int32]): row index of non-zero values of a matrix
        columns (npt.NDArray[np.int32]): column index of non-zero values of a matrix

    Returns:
        npt.NDArray[np.int32]: Array with mapping from ij to ji 
    """

    assert np.array_equal(np.shape(rows), np.shape(columns))

    # stack to 2D arrays
    ij: npt.NDArray[np.int32] = np.stack((rows, columns), axis=1)
    ji: npt.NDArray[np.int32] = np.stack((columns, rows), axis=1)

    # create type for sorting, a column is 8 byte
    rowtype = np.dtype((np.void, 8))
    # make contiguous in memory, view with new type and flatten
    ij_n = np.ascontiguousarray(ij).view(rowtype).ravel()
    ji_n = np.ascontiguousarray(ji).view(rowtype).ravel()

    # return idx that would sort
    ij_to_ijs = np.argsort(ij_n)

    # find indexes how to insert ji_n into ij_n such it remains sorted
    # ij_n is sorted with ij_to_ijs
    ijs_to_ji = ij_n.searchsorted(ji_n, sorter=ij_to_ijs)

    # take the elements from ij_to_ijs at ijs_to_ji
    # ij -> ij sorted -> ji mapping
    idx_transposed: npt.NDArray[np.int32] = ij_to_ijs[ijs_to_ji]

    # assert right shape
    assert np.array_equal(np.shape(rows), np.shape(idx_transposed))
    # assert transformation is correct
    assert np.array_equal(ij, ji[idx_transposed])
    assert np.array_equal(ij[idx_transposed], ji)
    assert idx_transposed.size == np.unique(idx_transposed).size
    idx_transposed = np.int32(idx_transposed)
    return idx_transposed


def find_idx_rt(rows: npt.NDArray[np.int32], columns: npt.NDArray[np.int32],
                ne: np.int32) -> typing.Tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
    """Creates 2D array with mapping between element ij and ji of a sparse matrix.
        It is assumed that the matrix is either symmetric/hermitian such that if 
        ij exists also ji is non-zero. The second axis already creates a time reversal
        to omit this operation later.
        Not used and np.ix_ is kinda whack

    Args:
        rows (npt.NDArray[np.int32]): row index of non-zero values of a matrix
        columns (npt.NDArray[np.int32]): column index of non-zero values of a matrix
        ne: np.int32: number of energy points

    Returns:
        typing.Tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]: 2D Array with mapping from ij to ji and reversal
    """

    assert np.array_equal(np.shape(rows), np.shape(columns))

    # stack to 2D arrays
    ij: npt.NDArray[np.int32] = np.stack((rows, columns), axis=1)
    ji: npt.NDArray[np.int32] = np.stack((columns, rows), axis=1)

    # create type for sorting, a column is 16 byte
    rowtype = np.dtype((np.void, 16))
    # make contiguous in memory, view with new type and flatten
    ij_n = np.ascontiguousarray(ij).view(rowtype).ravel()
    ji_n = np.ascontiguousarray(ji).view(rowtype).ravel()

    # return idx that would sort
    ij_to_ijs = np.argsort(ij_n)

    # find indexes how to insert ji_n into ij_n such it remains sorted
    # ij_n is sorted with ij_to_ijs
    ijs_to_ji = ij_n.searchsorted(ji_n, sorter=ij_to_ijs)

    # take the elements from ij_to_ijs at ijs_to_ji
    # ij -> ij sorted -> ji mapping
    idx_transposed: npt.NDArray[np.int32] = ij_to_ijs[ijs_to_ji]

    # sanity checks
    # assert right shape
    assert np.array_equal(np.shape(rows), np.shape(idx_transposed))
    # assert transformation is correct
    assert np.array_equal(ij, ji[idx_transposed])
    assert np.array_equal(ij[idx_transposed], ji)
    assert idx_transposed.size == np.unique(idx_transposed).size

    # time reversal idx, 2*ne due to the padding in fft
    reversal = np.roll(np.flip(np.arange(2*ne), axis=0), 1, axis=0)

    return np.ix_(idx_transposed, reversal)
