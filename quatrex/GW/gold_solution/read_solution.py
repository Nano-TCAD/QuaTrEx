"""Loads the gold solution created with MATLAB codebase
    and creates a mapping to transposed matrix"""
import h5py
import numpy as np
import numpy.typing as npt
import typing

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
