"""
Functions to calculate the screened interaction on the cpu.
See README.md for more information. 
"""
import concurrent.futures
from itertools import repeat
import numpy as np
import mkl
import typing
import numpy.typing as npt
from utils import matrix_creation
from block_tri_solvers import rgf_W


def p2w_pool_mpi_cpu(
    hamiltionian_obj: object,
    energy: npt.NDArray[np.float64],
    pg: npt.NDArray[np.complex128],
    pl: npt.NDArray[np.complex128],
    pr: npt.NDArray[np.complex128],
    vh: npt.NDArray[np.complex128],
    factor: npt.NDArray[np.float64],
    mkl_threads: int = 1,
    worker_num: int = 1
) -> typing.Tuple[
        npt.NDArray[np.complex128],
        npt.NDArray[np.complex128],
        npt.NDArray[np.complex128],
        npt.NDArray[np.complex128],
        npt.NDArray[np.complex128],
        npt.NDArray[np.complex128],
        int,
        int
    ]:

    # number of energy points
    ne = energy.shape[0]

    # number of blocks
    nb = hamiltionian_obj.Bmin.shape[0]
    # start and end index of each block in python indexing
    bmax = hamiltionian_obj.Bmax - 1
    bmin = hamiltionian_obj.Bmin - 1

    # fix nbc to 2 for the given solution
    # todo calculate it
    nbc = 2

    # block sizes after matrix multiplication
    bmax_ref = bmax[nbc-1:nb:nbc]
    bmin_ref = bmin[0:nb:nbc]
    # number of blocks after matrix multiplication
    nb_mm = bmax_ref.size
    # larges block length after matrix multiplication
    lb_max_mm = np.max(bmax_ref - bmin_ref + 1)

    # create empty buffer for screened interaction
    # in block formatg
    wr_diag, wr_upper, wl_diag, wl_upper, wg_diag, wg_upper = matrix_creation.initialize_block_G(ne, nb_mm, lb_max_mm)
    xr_diag = np.zeros((ne, nb_mm, lb_max_mm, lb_max_mm), dtype = np.complex128)
    # set number of mkl threads
    mkl.set_num_threads(mkl_threads)

    # todo remove this
    # not used inside rgf_W
    index_e = np.arange(ne)

    # Create a process pool with 4 workers
    with concurrent.futures.ThreadPoolExecutor(max_workers=worker_num) as executor:
        # Use the map function to apply the inv_matrices function to each pair of matrices in parallel
        executor.map(
                    rgf_W.rgf_W,
                    repeat(vh),
                    pg, pl, pr,
                    repeat(bmax), repeat(bmin),
                    wg_diag, wg_upper,
                    wl_diag, wl_upper,
                    wr_diag, wr_upper,
                    xr_diag, repeat(nbc),
                    index_e, factor
                     )

    return wg_diag, wg_upper, wl_diag, wl_upper, wr_diag, wr_upper, nb_mm, lb_max_mm

def p2w_mpi_cpu(
    hamiltionian_obj: object,
    energy: npt.NDArray[np.float64],
    pg: npt.NDArray[np.complex128],
    pl: npt.NDArray[np.complex128],
    pr: npt.NDArray[np.complex128],
    vh: npt.NDArray[np.complex128],
    factor: npt.NDArray[np.float64],
    mkl_threads: int = 1
) -> typing.Tuple[
        npt.NDArray[np.complex128],
        npt.NDArray[np.complex128],
        npt.NDArray[np.complex128],
        npt.NDArray[np.complex128],
        npt.NDArray[np.complex128],
        npt.NDArray[np.complex128],
        int,
        int
    ]:

    # number of energy points
    ne = energy.shape[0]

    # number of blocks
    nb = hamiltionian_obj.Bmin.shape[0]
    # start and end index of each block in python indexing
    bmax = hamiltionian_obj.Bmax - 1
    bmin = hamiltionian_obj.Bmin - 1

    # fix nbc to 2 for the given solution
    # todo calculate it
    nbc = 2

    # block sizes after matrix multiplication
    bmax_ref = bmax[nbc-1:nb:nbc]
    bmin_ref = bmin[0:nb:nbc]
    # number of blocks after matrix multiplication
    nb_mm = bmax_ref.size
    # larges block length after matrix multiplication
    lb_max_mm = np.max(bmax_ref - bmin_ref + 1)

    # create empty buffer for screened interaction
    # in block formatg
    wr_diag, wr_upper, wl_diag, wl_upper, wg_diag, wg_upper = matrix_creation.initialize_block_G(ne, nb_mm, lb_max_mm)
    xr_diag = np.zeros((ne, nb_mm, lb_max_mm, lb_max_mm), dtype = np.complex128)
    # set number of mkl threads
    mkl.set_num_threads(mkl_threads)

    # todo remove this
    # not used inside rgf_W
    index_e = np.arange(ne)

    # create buffer for every energy point
    wg_diag_buf = np.zeros((nb_mm, lb_max_mm, lb_max_mm), dtype = np.complex128)
    wg_upper_buf = np.zeros((nb_mm-1, lb_max_mm, lb_max_mm), dtype = np.complex128)
    wl_diag_buf = np.zeros((nb_mm, lb_max_mm, lb_max_mm), dtype = np.complex128)
    wl_upper_buf = np.zeros((nb_mm-1, lb_max_mm, lb_max_mm), dtype = np.complex128)
    wr_diag_buf = np.zeros((nb_mm, lb_max_mm, lb_max_mm), dtype = np.complex128)
    wr_upper_buf = np.zeros((nb_mm-1, lb_max_mm, lb_max_mm), dtype = np.complex128)
    xr_diag_buf = np.zeros((nb_mm, lb_max_mm, lb_max_mm), dtype = np.complex128)
    
    for ie in range(ne):
        rgf_W.rgf_W(
                vh,
                pg[ie], pl[ie], pr[ie],
                bmax, bmin,
                wg_diag_buf, wg_upper_buf,
                wl_diag_buf, wl_upper_buf,
                wr_diag_buf, wr_upper_buf,
                xr_diag_buf, nbc,
                index_e[ie], factor[ie]
        )
        # copy buffer to output
        wg_diag[ie,:,:] = wg_diag_buf
        wg_upper[ie,:,:] = wg_upper_buf
        wl_diag[ie,:,:] = wl_diag_buf
        wl_upper[ie,:,:] = wl_upper_buf
        wr_diag[ie,:,:] = wr_diag_buf
        wr_upper[ie,:,:] = wr_upper_buf
        xr_diag[ie,:,:] = xr_diag_buf

    return wg_diag, wg_upper, wl_diag, wl_upper, wr_diag, wr_upper, nb_mm, lb_max_mm
