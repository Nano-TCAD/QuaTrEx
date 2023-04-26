"""
Contains helper functions to transform 
between different formats used in the GW solver parts.

- Dense to block wise for single energy point


"""
import numpy as np
import numpy.typing as npt
import typing
from scipy import sparse

def dense2block(
    xx: npt.NDArray[np.complex128],
    bmax: npt.NDArray[np.int32],
    bmin: npt.NDArray[np.int32]
) -> typing.Tuple[
    npt.NDArray[np.complex128],
    npt.NDArray[np.complex128],
    npt.NDArray[np.complex128]
]:
    """Read out certain blocks from a dense matrix.
       The blocks are the tri diagonal blocks with varying size
       given through bmax and bmin.

    Args:
        xx (npt.NDArray[np.complex128]): dense matrix to read from (#orbitals, #orbitals)
        bmax (npt.NDArray[np.int32]): end indexes of the diagonal blocks
        bmin (npt.NDArray[np.int32]): start indexes of the diagonal blocks

    Returns:
        typing.Tuple[
        npt.NDArray[np.complex128], Diagonal blocks (#blocks, block lengths, block lengths)
        npt.NDArray[np.complex128], Upper diagonal blocks (#blocks, block lengths, block lengths')
        npt.NDArray[np.complex128]  Lower diagonal blocks (#blocks, block lengths', block lengths)
        ]
    """
    # number of blocks
    nb = bmin.size
    assert nb == bmax.size
    assert xx.shape[0] == xx.shape[1]
    assert xx.shape[0] == bmax[nb-1] + 1

    # create buffer to read out
    blocks_diag     = [np.empty((bmax[i]-bmin[i] + 1, bmax[i]-bmin[i] + 1),
                                dtype=np.complex128) for i in range(nb)]
    blocks_upper    = [np.empty((bmax[i]-bmin[i] + 1, bmax[i+1]-bmin[i+1] + 1),
                                dtype=np.complex128) for i in range(nb-1)]
    blocks_lower    = [np.empty((bmax[i+1]-bmin[i+1] + 1, bmax[i]-bmin[i] + 1),
                                dtype=np.complex128) for i in range(nb-1)]
    blocks_diag     = np.array(blocks_diag, dtype=np.complex128)
    blocks_upper    = np.array(blocks_upper, dtype=np.complex128)
    blocks_lower    = np.array(blocks_lower, dtype=np.complex128)


    # read out from dense array
    for i in range(nb):
        # block length
        bl = bmax[i]-bmin[i]+1
        # start index
        idx = bmin[i]

        blocks_diag[i,:,:] = xx[idx:idx+bl,idx:idx+bl]

    for i in range(nb-1):
        # block lengths
        bl1 = bmax[i]-bmin[i]+1
        bl2 = bmax[i+1]-bmin[i+1]+1
        # starting indexes
        id1 = bmin[i]
        id2 = bmin[i+1]

        blocks_upper[i,:,:] = xx[id1:id1+bl1,id2:id2+bl2]
        blocks_lower[i,:,:] = xx[id2:id2+bl2,id1:id1+bl1]

    return blocks_diag, blocks_upper, blocks_lower

def dense2block_energy(
    xx: npt.NDArray[np.complex128],
    bmax: npt.NDArray[np.int32],
    bmin: npt.NDArray[np.int32]
) -> typing.Tuple[
    npt.NDArray[np.complex128],
    npt.NDArray[np.complex128],
    npt.NDArray[np.complex128]
]:
    """Read out certain blocks from a dense matrix.
       The blocks are the tri diagonal blocks with varying size
       given through bmax and bmin.
       Where the given matrix is 2D plus additional energy axis
       
    Args:
        xx (npt.NDArray[np.complex128]): dense matrix to read from (#energy, #orbitals, #orbitals)
        bmax (npt.NDArray[np.int32]): end indexes of the diagonal blocks
        bmin (npt.NDArray[np.int32]): start indexes of the diagonal blocks

    Returns:
        typing.Tuple[
        npt.NDArray[np.complex128], Diagonal blocks (#energy, #blocks, block lengths, block lengths)
        npt.NDArray[np.complex128], Upper diagonal blocks (#energy, #blocks, block lengths, block lengths')
        npt.NDArray[np.complex128]  Lower diagonal blocks (#energy, #blocks, block lengths', block lengths)
        ]
    """
    # number of energy points
    ne = xx.shape[0]

    # number of blocks
    nb = bmin.size
    assert nb == bmax.size
    assert xx.shape[1] == xx.shape[2]
    assert xx.shape[1] == bmax[nb-1] + 1

    # create buffer to read out
    blocks_diag     = [np.empty((ne, bmax[i]-bmin[i] + 1, bmax[i]-bmin[i] + 1),
                                dtype=np.complex128) for i in range(nb)]
    blocks_upper    = [np.empty((ne, bmax[i]-bmin[i] + 1, bmax[i+1]-bmin[i+1] + 1),
                                dtype=np.complex128) for i in range(nb-1)]
    blocks_lower    = [np.empty((ne, bmax[i+1]-bmin[i+1] + 1, bmax[i]-bmin[i] + 1),
                                dtype=np.complex128) for i in range(nb-1)]
    blocks_diag     = np.array(blocks_diag,  dtype=np.complex128).transpose((1,0,2,3))
    blocks_upper    = np.array(blocks_upper, dtype=np.complex128).transpose((1,0,2,3))
    blocks_lower    = np.array(blocks_lower, dtype=np.complex128).transpose((1,0,2,3))


    # read out from dense array
    for i in range(nb):
        # block length
        bl = bmax[i]-bmin[i]+1
        # start index
        idx = bmin[i]

        blocks_diag[:,i,:,:] = xx[:,idx:idx+bl,idx:idx+bl]

    for i in range(nb-1):
        # block lengths
        bl1 = bmax[i]-bmin[i]+1
        bl2 = bmax[i+1]-bmin[i+1]+1
        # starting indexes
        id1 = bmin[i]
        id2 = bmin[i+1]

        blocks_upper[:,i,:,:] = xx[:,id1:id1+bl1,id2:id2+bl2]
        blocks_lower[:,i,:,:] = xx[:,id2:id2+bl2,id1:id1+bl1]

    return blocks_diag, blocks_upper, blocks_lower

def sparse2vecsparse(
    inp: npt.NDArray[np.complex128],
    rows: npt.NDArray[np.int32],
    columns: npt.NDArray[np.int32],
    nao: np.int64
) -> np.ndarray:
    """Convert from the 2D type of (nnz,ne) to a vector 
        of spare csr matrices, where the vector has size ne

    Args:
        inp (npt.npt.NDArray[np.complex128]): Dense 2D input of size (nnz,ne)
        rows (npt.NDArray[np.int32]): row indexes of non zeros (nnz)
        columns (npt.NDArray[np.int32]): column indexes of non zeros (nnz)
        nao (np.int64): Number of atomic orbitals, size of the hamiltonian (nao,nao)

    Returns:
        np.ndarray: vector containing ne times sparse csr matrices (nao,nao)
    """
    # number of energy points
    ne = inp.shape[1]
    # output buffer
    out = np.ndarray((ne,), dtype=object)
    for i in range(ne):
        out[i] = sparse.coo_array((inp[:,i], (rows, columns)),
                                  shape=(nao, nao), dtype = np.complex128).tocsr()
    return out

def map_block2sparse(
    rows: npt.NDArray[np.int32],
    columns: npt.NDArray[np.int32],
    bmax: npt.NDArray[np.int32],
    bmin: npt.NDArray[np.int32]
) -> typing.Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray
]:
    """Creates mapping from the three block vector diagonal, upper, lower
        to the form of a flattened 2D vector with the indexes given by rows/columns.
        The mapping is a vector of local block wise mappings

    Args:
        rows (npt.NDArray[np.int32]): non-zero element row indexes (coo) of green's function and others
        columns (npt.NDArray[np.int32]): non-zero element column indexes (coo) of green's function and others
        bmax (npt.NDArray[np.int32]): end indexes of the blocks
        bmin (npt.NDArray[np.int32]): start indexes of the blocks

    Returns:
        typing.Tuple[np.ndarray, mapping for diagonal blocks
                     np.ndarray, mapping for upper diagonal blocks
                     np.ndarray  mapping for lower diagonal blocks
        ]
    """
    assert columns.size == rows.size
    assert bmax.size == bmin.size

    # number of blocks
    nb = bmin.size
    # map buffers for the tridiagonal blocks
    map_diag  = np.ndarray((nb,), dtype=np.ndarray)
    map_upper = np.ndarray((nb-1,), dtype=np.ndarray)
    map_lower = np.ndarray((nb-1,), dtype=np.ndarray)

    # init number of non zero elements
    no = 0

    # iterate over diagonal
    for i in range(nb):
        # start index and end index
        idx_start = bmin[i]
        idx_end = bmax[i]

        # mask if the elements are in the block
        mask = (rows >= idx_start) & (rows <= idx_end) & (columns >= idx_start) & (columns <= idx_end)
        # number of to read out elements in the block
        nelements = mask.sum()
        # add to number nonzero
        no += nelements

        # local block mapping to one vector data
        map_loc = np.empty((3,nelements), dtype=np.int32)

        # sparse elements in the block
        map_loc[0,:] = rows[mask]       - idx_start
        map_loc[1,:] = columns[mask]    - idx_start
        # location in data vector
        map_loc[2,:] = np.where(mask)[0]
        # assign to buffer
        map_diag[i] = map_loc

    # iterate over upper/lower diagonal
    for i in range(nb-1):
        # start index and end index
        idx1_start  = bmin[i]
        idx1_end    = bmax[i]
        idx2_start  = bmin[i+1]
        idx2_end    = bmax[i+1]
         # mask if the elements are in the upper/lower block
        mask_upper = (rows >= idx1_start) & (rows <= idx1_end) & (columns >= idx2_start) & (columns <= idx2_end)
        mask_lower = (rows >= idx2_start) & (rows <= idx2_end) & (columns >= idx1_start) & (columns <= idx1_end)
        # number of to read out elements in the block
        nelements_upper = mask_upper.sum()
        nelements_lower = mask_lower.sum()
        # add to number nonzero
        no += nelements_upper
        no += nelements_lower

        # local block mapping to one vector data
        map_loc_upper = np.empty((3,nelements_upper), dtype=np.int32)
        map_loc_lower = np.empty((3,nelements_lower), dtype=np.int32)

        # sparse elements in the upper/lower block
        map_loc_upper[0,:] = rows[mask_upper] - idx1_start
        map_loc_upper[1,:] = columns[mask_upper] - idx2_start
        map_loc_lower[0,:] = rows[mask_lower] - idx2_start
        map_loc_lower[1,:] = columns[mask_lower] - idx1_start

        # location in data vector
        map_loc_upper[2,:] = np.where(mask_upper)[0]
        map_loc_lower[2,:] = np.where(mask_lower)[0]

        # assign to buffer
        map_upper[i] = map_loc_upper
        map_lower[i] = map_loc_lower

    assert no == rows.size

    return map_diag, map_upper, map_lower

def block2sparse(
    map_diag: np.ndarray,
    map_upper: np.ndarray,
    map_lower: np.ndarray,
    x_diag: npt.NDArray[np.complex128],
    x_upper: npt.NDArray[np.complex128],
    x_lower: npt.NDArray[np.complex128],
    no: np.int64
) -> npt.NDArray[np.complex128]:
    """Applies the map to get from block to sparse form
       Map created by map_block2sparse

    Args:
        map_diag (np.ndarray): Diagonal map, #blocks length vector of local maps
        map_upper (np.ndarray): Upper Diagonal map, #blocks-1 length vector of local maps
        map_lower (np.ndarray): Lower Diagonal map, #blocks-1 length vector of local maps
        x_diag (npt.NDArray[np.complex128]): Diagonal blocks
        x_upper (npt.NDArray[np.complex128]): Upper blocks
        x_lower (npt.NDArray[np.complex128]): Lower blocks
        no (np.int64): Number of non zero elements of final form

    Returns:
        npt.NDArray[np.complex128]: 1D output of the mapped elements
    """
    assert map_diag.shape[0] == map_upper.shape[0] + 1
    assert map_diag.shape[0] == map_lower.shape[0] + 1
    # number of blocks
    nb = map_diag.shape[0]
    output = np.empty((no), dtype=np.complex128)
    # diagonal elements
    for i in range(nb):
        # read out local map
        map_loc = map_diag[i]
        # apply map
        output[map_loc[2,:]] = x_diag[i,map_loc[0,:],map_loc[1,:]]
    # off diagonal elements
    for i in range(nb-1):
        # read out local map
        map_loc_upper = map_upper[i]
        map_loc_lower = map_lower[i]
        # apply map
        output[map_loc_upper[2,:]] = x_upper[i,map_loc_upper[0,:],map_loc_upper[1,:]]
        output[map_loc_lower[2,:]] = x_lower[i,map_loc_lower[0,:],map_loc_lower[1,:]]
    return output

def block2sparse_energy(
    map_diag: np.ndarray,
    map_upper: np.ndarray,
    map_lower: np.ndarray,
    x_diag: npt.NDArray[np.complex128],
    x_upper: npt.NDArray[np.complex128],
    x_lower: npt.NDArray[np.complex128],
    no: np.int64,
    ne: np.int64
) -> npt.NDArray[np.complex128]:
    """Applies the map to get from block to sparse form
       Map created by map_block2sparse

    Args:
        map_diag (np.ndarray): Diagonal map, 4 x number of non zeros contained in diag (3*x_diag indexes and last for output)
        map_upper (np.ndarray): Upper Diagonal map, 4 x number of non zeros contained in upper (3*x_upper indexes and last for output)
        map_lower (np.ndarray): Lower Diagonal map, 4 x number of non zeros contained in lower (3*x_lower indexes and last for output)
        x_diag (npt.NDArray[np.complex128]): Diagonal blocks
        x_upper (npt.NDArray[np.complex128]): Upper blocks
        x_lower (npt.NDArray[np.complex128]): Lower blocks
        no (np.int64): Number of non zero elements of final form
        ne (np.int64): Number of energy points

    Returns:
        npt.NDArray[np.complex128]: 2D output of the mapped elements (nnz,#energy)
    """
    assert map_diag.shape[0] == map_upper.shape[0] + 1
    assert map_diag.shape[0] == map_lower.shape[0] + 1
    # number of blocks
    nb = map_diag.shape[0]
    output = np.empty((no, ne), dtype=np.complex128)
    # diagonal elements
    for i in range(nb):
        # read out local map
        map_loc = map_diag[i]
        # apply map
        output[map_loc[2,:],:] = x_diag[:,i,map_loc[0,:],map_loc[1,:]].transpose()
    # off diagonal elements
    for i in range(nb-1):
        # read out local map
        map_loc_upper = map_upper[i]
        map_loc_lower = map_lower[i]
        # apply map
        output[map_loc_upper[2,:],:] = x_upper[:,i,map_loc_upper[0,:],map_loc_upper[1,:]].transpose()
        output[map_loc_lower[2,:],:] = x_lower[:,i,map_loc_lower[0,:],map_loc_lower[1,:]].transpose()
    return output


def map_block2sparse_alt(
    rows: npt.NDArray[np.int32],
    columns: npt.NDArray[np.int32],
    bmax: npt.NDArray[np.int32],
    bmin: npt.NDArray[np.int32]
) -> typing.Tuple[
    npt.NDArray[np.int32],
    npt.NDArray[np.int32],
    npt.NDArray[np.int32]
]:
    """Creates mapping from the three block vector diagonal, upper, lower
        to the form of a flattened 2D vector with the indexes given by rows/columns
        The mapping is just random access pattern for the 3D block tensor

    Args:
        rows (npt.NDArray[np.int32]): non-zero element row indexes (coo) of green's function and others
        columns (npt.NDArray[np.int32]): non-zero element column indexes (coo) of green's function and others
        bmax (npt.NDArray[np.int32]): end indexes of the blocks
        bmin (npt.NDArray[np.int32]): start indexes of the blocks

    Returns:
        typing.Tuple[npt.NDArray[np.int32], mapping for diagonal blocks
                     npt.NDArray[np.int32], mapping for upper diagonal blocks
                     npt.NDArray[np.int32]  mapping for lower diagonal blocks
        ]
    """
    assert columns.size == rows.size
    assert bmax.size == bmin.size

    # number of blocks
    nb = bmin.size
    # map buffers for the tridiagonal blocks
    #a priori the length is unknown
    map_diag  = np.empty((4,0), dtype=np.int32)
    map_upper = np.empty((4,0), dtype=np.int32)
    map_lower = np.empty((4,0), dtype=np.int32)

    # init number of non zero elements
    no = 0

    # iterate over diagonal
    for i in range(nb):
        # start index and end index
        idx_start = bmin[i]
        idx_end = bmax[i]

        # mask if the elements are in the block
        mask = (rows >= idx_start) & (rows <= idx_end) & (columns >= idx_start) & (columns <= idx_end)
        # number of to read out elements in the block
        nelements = mask.sum()
        # add to number nonzero
        no += nelements

        # local block mapping to one vector data
        map_loc = np.empty((4,nelements), dtype=np.int32)

        # sparse elements in the block
        map_loc[0,:] = i
        map_loc[1,:] = rows[mask]       - idx_start
        map_loc[2,:] = columns[mask]    - idx_start
        # location in data vector
        map_loc[3,:] = np.where(mask)[0]
        # concat to buffer
        map_diag = np.concatenate((map_diag, map_loc), axis=1)

    # iterate over upper/lower diagonal
    for i in range(nb-1):
        # start index and end index
        idx1_start  = bmin[i]
        idx1_end    = bmax[i]
        idx2_start  = bmin[i+1]
        idx2_end    = bmax[i+1]
         # mask if the elements are in the upper/lower block
        mask_upper = (rows >= idx1_start) & (rows <= idx1_end) & (columns >= idx2_start) & (columns <= idx2_end)
        mask_lower = (rows >= idx2_start) & (rows <= idx2_end) & (columns >= idx1_start) & (columns <= idx1_end)
        # number of to read out elements in the block
        nelements_upper = mask_upper.sum()
        nelements_lower = mask_lower.sum()
        # add to number nonzero
        no += nelements_upper
        no += nelements_lower

        # local block mapping to one vector data
        map_loc_upper = np.empty((4,nelements_upper), dtype=np.int32)
        map_loc_lower = np.empty((4,nelements_lower), dtype=np.int32)

        # sparse elements in the upper/lower block
        map_loc_upper[0,:] = i
        map_loc_lower[0,:] = i
        map_loc_upper[1,:] = rows[mask_upper] - idx1_start
        map_loc_upper[2,:] = columns[mask_upper] - idx2_start
        map_loc_lower[1,:] = rows[mask_lower] - idx2_start
        map_loc_lower[2,:] = columns[mask_lower] - idx1_start

        # location in data vector
        map_loc_upper[3,:] = np.where(mask_upper)[0]
        map_loc_lower[3,:] = np.where(mask_lower)[0]

        # concat to buffer
        map_upper = np.concatenate((map_upper, map_loc_upper), axis=1)
        map_lower = np.concatenate((map_lower, map_loc_lower), axis=1)

    assert no == rows.size

    return map_diag, map_upper, map_lower

def block2sparse_alt(
    map_diag: np.ndarray,
    map_upper: np.ndarray,
    map_lower: np.ndarray,
    x_diag: npt.NDArray[np.complex128],
    x_upper: npt.NDArray[np.complex128],
    x_lower: npt.NDArray[np.complex128],
    no: np.int64
) -> npt.NDArray[np.complex128]:
    """Applies the map to get from block to sparse form
       Alternative map created by map_block2sparse_alt

    Args:
        map_diag (np.ndarray): Diagonal map, 4 x number of non zeros contained in diag (3*x_diag indexes and last for output)
        map_upper (np.ndarray): Upper Diagonal map, 4 x number of non zeros contained in upper (3*x_upper indexes and last for output)
        map_lower (np.ndarray): Lower Diagonal map, 4 x number of non zeros contained in lower (3*x_lower indexes and last for output)
        x_diag (npt.NDArray[np.complex128]): Diagonal blocks
        x_upper (npt.NDArray[np.complex128]): Upper blocks
        x_lower (npt.NDArray[np.complex128]): Lower blocks
        no (np.int64): Number of non zero elements of final form

    Returns:
        npt.NDArray[np.complex128]: 1D output of the mapped elements
    """
    assert map_diag.shape[0] == map_upper.shape[0]
    assert map_diag.shape[0] == map_lower.shape[0]
    # number of blocks
    output = np.empty((no), dtype=np.complex128)

    # diagonal elements
    output[map_diag[3,:]] = x_diag[map_diag[0,:],map_diag[1,:],map_diag[2,:]]
    # off diagonal elements
    output[map_upper[3,:]] = x_upper[map_upper[0,:],map_upper[1,:],map_upper[2,:]]
    output[map_lower[3,:]] = x_lower[map_lower[0,:],map_lower[1,:],map_lower[2,:]]
    return output

def block2sparse_energy_alt(
    map_diag: np.ndarray,
    map_upper: np.ndarray,
    map_lower: np.ndarray,
    x_diag: npt.NDArray[np.complex128],
    x_upper: npt.NDArray[np.complex128],
    x_lower: npt.NDArray[np.complex128],
    no: np.int64,
    ne: np.int64
) -> npt.NDArray[np.complex128]:
    """Applies the map to get from block to sparse form
       Alternative map created by map_block2sparse_alt

    Args:
        map_diag (np.ndarray): Diagonal map, 4 x number of non zeros contained in diag (3*x_diag indexes and last for output)
        map_upper (np.ndarray): Upper Diagonal map, 4 x number of non zeros contained in upper (3*x_upper indexes and last for output)
        map_lower (np.ndarray): Lower Diagonal map, 4 x number of non zeros contained in lower (3*x_lower indexes and last for output)
        x_diag (npt.NDArray[np.complex128]): Diagonal blocks
        x_upper (npt.NDArray[np.complex128]): Upper blocks
        x_lower (npt.NDArray[np.complex128]): Lower blocks
        no (np.int64): Number of non zero elements of final form
        ne (np.int64): Number of energy points

    Returns:
        npt.NDArray[np.complex128]: 2D output of the mapped elements (nnz,#energy)
    """
    assert map_diag.shape[0] == map_upper.shape[0]
    assert map_diag.shape[0] == map_lower.shape[0]
    assert map_lower[0,:].size + map_diag[0,:].size + map_upper[0,:].size == no
    # number of blocks
    output = np.empty((no, ne), dtype=np.complex128)

    # diagonal elements
    output[map_diag[3,:],:] = x_diag[:,map_diag[0,:],map_diag[1,:],map_diag[2,:]].transpose()
    # off diagonal elements
    output[map_upper[3,:],:] = x_upper[:,map_upper[0,:],map_upper[1,:],map_upper[2,:]].transpose()
    output[map_lower[3,:],:] = x_lower[:,map_lower[0,:],map_lower[1,:],map_lower[2,:]].transpose()
    return output

def sparse2block_alt(
    map_diag: np.ndarray,
    map_upper: np.ndarray,
    map_lower: np.ndarray,
    x_s: npt.NDArray[np.complex128],
    bmax,
    bmin
) -> typing.Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray
]:
    # non zero elements
    no = x_s.shape[0]
    # number of blocks
    nb = bmin.size
    assert nb == bmax.size
    assert nb == bmin.size
    # create buffer to write
    blocks_diag     = [np.zeros((bmax[i]-bmin[i] + 1, bmax[i]-bmin[i] + 1),
                                dtype=np.complex128) for i in range(nb)]
    blocks_upper    = [np.zeros((bmax[i]-bmin[i] + 1, bmax[i+1]-bmin[i+1] + 1),
                                dtype=np.complex128) for i in range(nb-1)]
    blocks_lower    = [np.zeros((bmax[i+1]-bmin[i+1] + 1, bmax[i]-bmin[i] + 1),
                                dtype=np.complex128) for i in range(nb-1)]
    blocks_diag     = np.array(blocks_diag, dtype=np.complex128)
    blocks_upper    = np.array(blocks_upper, dtype=np.complex128)
    blocks_lower    = np.array(blocks_lower, dtype=np.complex128)

    assert map_diag.shape[0] == map_upper.shape[0]
    assert map_diag.shape[0] == map_lower.shape[0]
    assert map_lower[0,:].size + map_diag[0,:].size + map_upper[0,:].size == no

    # diagonal elements
    blocks_diag[map_diag[0,:],map_diag[1,:],map_diag[2,:]] = x_s[map_diag[3,:]]
    # off diagonal elements
    blocks_upper[map_upper[0,:],map_upper[1,:],map_upper[2,:]] = x_s[map_upper[3,:]]
    blocks_lower[map_lower[0,:],map_lower[1,:],map_lower[2,:]] = x_s[map_lower[3,:]]
    return blocks_diag, blocks_upper, blocks_lower

def sparse2block_energy_alt(
    map_diag: np.ndarray,
    map_upper: np.ndarray,
    map_lower: np.ndarray,
    x_s: npt.NDArray[np.complex128],
    bmax,
    bmin
) -> typing.Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray
]:
    """Applies the map to get from sparse to block form
       Alternative map created by map_block2sparse_alt
       The other form does not exist, because 
       the alt implementation will be faster and 
       I (almaeder) am lazy

    Args:
        map_diag (np.ndarray): Diagonal map, 4 x number of non zeros contained in diag (3*x_diag indexes and last for output)
        map_upper (np.ndarray): Upper Diagonal map, 4 x number of non zeros contained in upper (3*x_upper indexes and last for output)
        map_lower (np.ndarray): Lower Diagonal map, 4 x number of non zeros contained in lower (3*x_lower indexes and last for output)
        x_s (npt.NDArray[np.complex128]): 2D array (nnz,#energy)
        bmax (_type_): vector of end indexes of blocks
        bmin (_type_): vector of start indexes of blocks

    Returns:
        typing.Tuple[ np.ndarray, np.ndarray, np.ndarray ]: _description_
    """
    # number of energy and non zero elements
    ne = x_s.shape[1]
    no = x_s.shape[0]
    # number of blocks
    nb = bmin.size
    assert nb == bmax.size
    assert nb == bmin.size
    # create buffer to write
    blocks_diag     = [np.empty((ne, bmax[i]-bmin[i] + 1, bmax[i]-bmin[i] + 1),
                                dtype=np.complex128) for i in range(nb)]
    blocks_upper    = [np.empty((ne, bmax[i]-bmin[i] + 1, bmax[i+1]-bmin[i+1] + 1),
                                dtype=np.complex128) for i in range(nb-1)]
    blocks_lower    = [np.empty((ne, bmax[i+1]-bmin[i+1] + 1, bmax[i]-bmin[i] + 1),
                                dtype=np.complex128) for i in range(nb-1)]
    blocks_diag     = np.array(blocks_diag, dtype=np.complex128).transpose((1,0,2,3))
    blocks_upper    = np.array(blocks_upper, dtype=np.complex128).transpose((1,0,2,3))
    blocks_lower    = np.array(blocks_lower, dtype=np.complex128).transpose((1,0,2,3))

    assert map_diag.shape[0] == map_upper.shape[0]
    assert map_diag.shape[0] == map_lower.shape[0]
    assert map_lower[0,:].size + map_diag[0,:].size + map_upper[0,:].size == no

    # diagonal elements
    blocks_diag[:,map_diag[0,:],map_diag[1,:],map_diag[2,:]] = x_s[map_diag[3,:],:].transpose()
    # off diagonal elements
    blocks_upper[:,map_upper[0,:],map_upper[1,:],map_upper[2,:]] = x_s[map_upper[3,:],:].transpose()
    blocks_lower[:,map_lower[0,:],map_lower[1,:],map_lower[2,:]] = x_s[map_lower[3,:],:].transpose()
    return blocks_diag, blocks_upper, blocks_lower
