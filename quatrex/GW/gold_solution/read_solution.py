# Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.
""" Loads the gold solution created with MATLAB codebase and creates a mapping to transposed matrix. """

import h5py
import numpy as np
import numpy.typing as npt
import typing


def load_x(
    path: str, tensor_name: str
) -> typing.Tuple[npt.NDArray[np.double], npt.NDArray[np.int32], npt.NDArray[np.int32], npt.NDArray[np.complex128],
                  npt.NDArray[np.complex128], npt.NDArray[np.complex128]]:
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
    energy: npt.NDArray[np.double] = np.array(hdf5["E"], dtype=np.double)
    # go from matlab to python
    rows: npt.NDArray[np.int32] = np.array(hdf5["rows"], dtype=np.int32) - 1
    columns: npt.NDArray[np.int32] = np.array(hdf5["columns"], dtype=np.int32) - 1

    energy: npt.NDArray[np.double] = np.squeeze(energy)
    rows: npt.NDArray[np.int32] = np.squeeze(rows)
    columns: npt.NDArray[np.int32] = np.squeeze(columns)

    realg: npt.NDArray[np.double] = np.array(hdf5["real" + tensor_name + "g"], dtype=np.double)
    imgg: npt.NDArray[np.double] = np.array(hdf5["img" + tensor_name + "g"], dtype=np.double)
    reall: npt.NDArray[np.double] = np.array(hdf5["real" + tensor_name + "l"], dtype=np.double)
    imgl: npt.NDArray[np.double] = np.array(hdf5["img" + tensor_name + "l"], dtype=np.double)
    realr: npt.NDArray[np.double] = np.array(hdf5["real" + tensor_name + "r"], dtype=np.double)
    imgr: npt.NDArray[np.double] = np.array(hdf5["img" + tensor_name + "r"], dtype=np.double)

    xg = realg + 1j * imgg
    xl = reall + 1j * imgl
    xr = realr + 1j * imgr

    return (energy, rows, columns, xg, xl, xr)


def load_v(path: str) -> typing.Tuple[npt.NDArray[np.int32], npt.NDArray[np.int32], npt.NDArray[np.complex128]]:
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
    rows: npt.NDArray[np.int32] = np.array(hdf5["rows"], dtype=np.int32) - 1
    columns: npt.NDArray[np.int32] = np.array(hdf5["columns"], dtype=np.int32) - 1

    realvh: npt.NDArray[np.double] = np.array(hdf5["realvh"], dtype=np.double)
    imgvh: npt.NDArray[np.double] = np.array(hdf5["imgvh"], dtype=np.double)
    vh: npt.NDArray[np.complex128] = realvh + 1j * imgvh
    vh: npt.NDArray[np.complex128] = np.squeeze(vh)

    return (rows, columns, vh)


def load_B(path: str) -> typing.Tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
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
    bmax: npt.NDArray[np.int32] = np.array(hdf5["Bmax"], dtype=np.int32) - 1
    bmin: npt.NDArray[np.int32] = np.array(hdf5["Bmin"], dtype=np.int32) - 1
    bmax: npt.NDArray[np.int32] = np.squeeze(bmax)
    bmin: npt.NDArray[np.int32] = np.squeeze(bmin)
    return (bmax, bmin)


def save_all(
    energy: npt.NDArray[np.float64],
    rows: npt.NDArray[np.int32],
    columns: npt.NDArray[np.int32],
    gg: npt.NDArray[np.complex128],
    gl: npt.NDArray[np.complex128],
    gr: npt.NDArray[np.complex128],
    pg: npt.NDArray[np.complex128],
    pl: npt.NDArray[np.complex128],
    pr: npt.NDArray[np.complex128],
    wg: npt.NDArray[np.complex128],
    wl: npt.NDArray[np.complex128],
    wr: npt.NDArray[np.complex128],
    sg: npt.NDArray[np.complex128],
    sl: npt.NDArray[np.complex128],
    sr: npt.NDArray[np.complex128],
    bmax: npt.NDArray[np.int32],
    bmin: npt.NDArray[np.int32],
    path: str
):
    """
    Saves all the data to a file in the format of changeFormatGPWS.m

    Args:
        energy (npt.NDArray[np.float64]): Energy
        rows (npt.NDArray[np.int32]): Non-zero row elements
        columns (npt.NDArray[np.int32]): Non-zero column elements
        gg (npt.NDArray[np.complex128]): 2D energy contiguous greater green's function
        gl (npt.NDArray[np.complex128]): 2D energy contiguous lesser green's function
        gr (npt.NDArray[np.complex128]): 2D energy contiguous retarded green's function
        pg (npt.NDArray[np.complex128]): 2D energy contiguous greater polarization
        pl (npt.NDArray[np.complex128]): 2D energy contiguous lesser polarization
        pr (npt.NDArray[np.complex128]): 2D energy contiguous retarded polarization
        wg (npt.NDArray[np.complex128]): 2D energy contiguous greater screened interaction
        wl (npt.NDArray[np.complex128]): 2D energy contiguous lesser screened interaction
        wr (npt.NDArray[np.complex128]): 2D energy contiguous retarded screened interaction
        sg (npt.NDArray[np.complex128]): 2D energy contiguous greater self energy
        sl (npt.NDArray[np.complex128]): 2D energy contiguous lesser self energy
        sr (npt.NDArray[np.complex128]): 2D energy contiguous retarded self energy
        bmax (npt.NDArray[np.int32]): End indices of blocks
        bmin (npt.NDArray[np.int32]): Start indices of blocks
        path (str): Path+filename where to save
    """

    # create the file
    hdf5 = h5py.File(path, "w")
    group = hdf5.create_group("formatted")

    # write the entries
    # +1 to go from python to matlab (legacy on how the file is read)
    group.create_dataset("rows", data=rows+1)
    group.create_dataset("columns", data=columns+1)
    group.create_dataset("E", data=energy)
    # group.create_dataset("realvh", data=np.real(vh))
    # group.create_dataset("imgvh", data=np.imag(vh))
    group.create_dataset("realgg", data=np.real(gg))
    group.create_dataset("imggg", data=np.imag(gg))
    group.create_dataset("realgl", data=np.real(gl))
    group.create_dataset("imggl", data=np.imag(gl))
    group.create_dataset("realgr", data=np.real(gr))
    group.create_dataset("imggr", data=np.imag(gr))
    group.create_dataset("realpg", data=np.real(pg))
    group.create_dataset("imgpg", data=np.imag(pg))
    group.create_dataset("realpl", data=np.real(pl))
    group.create_dataset("imgpl", data=np.imag(pl))
    group.create_dataset("realpr", data=np.real(pr))
    group.create_dataset("imgpr", data=np.imag(pr))
    group.create_dataset("realwg", data=np.real(wg))
    group.create_dataset("imgwg", data=np.imag(wg))
    group.create_dataset("realwl", data=np.real(wl))
    group.create_dataset("imgwl", data=np.imag(wl))
    group.create_dataset("realwr", data=np.real(wr))
    group.create_dataset("imgwr", data=np.imag(wr))
    group.create_dataset("realsg", data=np.real(sg))
    group.create_dataset("imgsg", data=np.imag(sg))
    group.create_dataset("realsl", data=np.real(sl))
    group.create_dataset("imgsl", data=np.imag(sl))
    group.create_dataset("realsr", data=np.real(sr))
    group.create_dataset("imgsr", data=np.imag(sr))
    group.create_dataset("Bmax", data=bmax+1)
    group.create_dataset("Bmin", data=bmin+1)


def load_pg(
    path: str
) -> typing.Tuple[npt.NDArray[np.double], npt.NDArray[np.int32], npt.NDArray[np.int32],
                  typing.Tuple[npt.NDArray[np.double], npt.NDArray[np.double], npt.NDArray[np.double],
                               npt.NDArray[np.double], npt.NDArray[np.double], npt.NDArray[np.double]], typing.Tuple[
                                   npt.NDArray[np.double], npt.NDArray[np.double], npt.NDArray[np.double],
                                   npt.NDArray[np.double], npt.NDArray[np.double], npt.NDArray[np.double]]]:
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

    energy: npt.NDArray[np.double] = np.array(hdf5["E"], dtype=np.double)
    # go from matlab to python
    rows: npt.NDArray[np.int32] = np.array(hdf5["rows"], dtype=np.int32) - 1
    columns: npt.NDArray[np.int32] = np.array(hdf5["columns"], dtype=np.int32) - 1

    realgg: npt.NDArray[np.double] = np.array(hdf5["realgg"], dtype=np.double)
    imggg: npt.NDArray[np.double] = np.array(hdf5["imggg"], dtype=np.double)
    realgl: npt.NDArray[np.double] = np.array(hdf5["realgl"], dtype=np.double)
    imggl: npt.NDArray[np.double] = np.array(hdf5["imggl"], dtype=np.double)
    realgr: npt.NDArray[np.double] = np.array(hdf5["realgr"], dtype=np.double)
    imggr: npt.NDArray[np.double] = np.array(hdf5["imggr"], dtype=np.double)

    realpg: npt.NDArray[np.double] = np.array(hdf5["realpg"], dtype=np.double)
    imgpg: npt.NDArray[np.double] = np.array(hdf5["imgpg"], dtype=np.double)
    realpl: npt.NDArray[np.double] = np.array(hdf5["realpl"], dtype=np.double)
    imgpl: npt.NDArray[np.double] = np.array(hdf5["imgpl"], dtype=np.double)
    realpr: npt.NDArray[np.double] = np.array(hdf5["realpr"], dtype=np.double)
    imgpr: npt.NDArray[np.double] = np.array(hdf5["imgpr"], dtype=np.double)

    gf = (realgg, imggg, realgl, imggl, realgr, imggr)
    pol = (realpg, imgpg, realpl, imgpl, realpr, imgpr)
    return (energy, rows, columns, gf, pol)
