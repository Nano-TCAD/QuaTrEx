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
from scipy import sparse
from utils import matrix_creation
from utils import change_format
from block_tri_solvers import rgf_W
from block_tri_solvers import matrix_inversion_w
from OBC import obc_w
import time

def p2w_pool_mpi_cpu(
    hamiltionian_obj: object,
    energy: npt.NDArray[np.float64],
    pg: npt.NDArray[np.complex128],
    pl: npt.NDArray[np.complex128],
    pr: npt.NDArray[np.complex128],
    vh: npt.NDArray[np.complex128],
    dosw: npt.NDArray[np.complex128],
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
    """Calculates the screened interaction on the cpu.
    Uses mkl threading and pool threads.

    Args:
        hamiltionian_obj (object): Class containing the hamiltonian information
        energy (npt.NDArray[np.float64]): energy points
        pg (npt.NDArray[np.complex128]): Greater polarization, vector of sparse matrices
        pl (npt.NDArray[np.complex128]): Lesser polarization, vector of sparse matrices
        pr (npt.NDArray[np.complex128]): Retarded polarization, vector of sparse matrices
        vh (npt.NDArray[np.complex128]): Vh sparse matrix
        factor (npt.NDArray[np.float64]): Smoothing factor
        mkl_threads (int, optional): Number of mkl threads used. Defaults to 1.
        worker_num(int, optional): Number of pool workers used. Defaults to 1.

    Returns:
        typing.Tuple[npt.NDArray[np.complex128],
                    npt.NDArray[np.complex128],
                    npt.NDArray[np.complex128],
                    npt.NDArray[np.complex128],
                    npt.NDArray[np.complex128],
                    npt.NDArray[np.complex128],
                    int, int ]:
    Diagonal/Upper block tensor (#blocks, blocksize, blocksize) of greater, lesser, retarded screened interaction.
    Number of blocks and block size after matrix multiplication
    """
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
    bmax_mm = bmax[nbc-1:nb:nbc]
    bmin_mm = bmin[0:nb:nbc]
    # number of blocks after matrix multiplication
    nb_mm = bmax_mm.size
    # larges block length after matrix multiplication
    lb_max_mm = np.max(bmax_mm - bmin_mm + 1)

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
                    xr_diag, dosw, repeat(nbc),
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
    dosw: npt.NDArray[np.complex128],
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
    """Calculates the screened interaction on the cpu.
    Uses only mkl threading.

    Args:
        hamiltionian_obj (object): Class containing the hamiltonian information
        energy (npt.NDArray[np.float64]): energy points
        pg (npt.NDArray[np.complex128]): Greater polarization, vector of sparse matrices
        pl (npt.NDArray[np.complex128]): Lesser polarization, vector of sparse matrices
        pr (npt.NDArray[np.complex128]): Retarded polarization, vector of sparse matrices
        vh (npt.NDArray[np.complex128]): Vh sparse matrix
        factor (npt.NDArray[np.float64]): Smoothing factor
        mkl_threads (int, optional): Number of mkl threads used. Defaults to 1.

    Returns:
        typing.Tuple[npt.NDArray[np.complex128],
                    npt.NDArray[np.complex128],
                    npt.NDArray[np.complex128],
                    npt.NDArray[np.complex128],
                    npt.NDArray[np.complex128],
                    npt.NDArray[np.complex128],
                    int, int ]:
    Diagonal/Upper block tensor (#blocks, blocksize, blocksize) of greater, lesser, retarded screened interaction.
    Number of blocks and block size after matrix multiplication
    """

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
    bmax_mm = bmax[nbc-1:nb:nbc]
    bmin_mm = bmin[0:nb:nbc]
    # number of blocks after matrix multiplication
    nb_mm = bmax_mm.size
    # larges block length after matrix multiplication
    lb_max_mm = np.max(bmax_mm - bmin_mm + 1)

    # create empty buffer for screened interaction
    # in block format
    wr_diag, wr_upper, wl_diag, wl_upper, wg_diag, wg_upper = matrix_creation.initialize_block_G(ne, nb_mm, lb_max_mm)
    xr_diag = np.zeros((ne, nb_mm, lb_max_mm, lb_max_mm), dtype = np.complex128)
    # set number of mkl threads
    mkl.set_num_threads(mkl_threads)

    # todo remove this
    # not used inside rgf_W
    index_e = np.arange(ne)

    for ie in range(ne):
        rgf_W.rgf_W(
                vh,
                pg[ie],
                pl[ie],
                pr[ie],
                bmax, bmin,
                wg_diag[ie], wg_upper[ie],
                wl_diag[ie], wl_upper[ie],
                wr_diag[ie], wr_upper[ie],
                xr_diag[ie], dosw[ie], nbc,
                index_e[ie], factor[ie]
        )
        
    return wg_diag, wg_upper, wl_diag, wl_upper, wr_diag, wr_upper, nb_mm, lb_max_mm


def w2s_l(
    vh: npt.NDArray[np.complex128],
    pg_vec: npt.NDArray[np.complex128],
    pl_vec: npt.NDArray[np.complex128],
    pr_vec: npt.NDArray[np.complex128]
) -> typing.Tuple[npt.NDArray[np.complex128],
                  npt.NDArray[np.complex128],
                  npt.NDArray[np.complex128]]:
    """
    Calculates for all the energy points the additional helper variables.

    Args:
        vh (npt.NDArray[np.complex128]): Effective interaction
        pg_vec (npt.NDArray[np.complex128]): Greater Polarization, vector of sparse matrices
        pl_vec (npt.NDArray[np.complex128]): Lesser Polarization, vector of sparse matrices
        pr_vec (npt.NDArray[np.complex128]): Retarded Polarization, vector of sparse matrices

    Returns:
        typing.Tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128], npt.NDArray[np.complex128]]: S^{r}\left(E\right), L^{>}\left(E\right), L^{<}\left(E\right)
    """

    # number of energy points
    ne = pg_vec.shape[0]
    # create output vector
    sr_vec = np.ndarray((ne,), dtype=object)
    lg_vec = np.ndarray((ne,), dtype=object)
    ll_vec = np.ndarray((ne,), dtype=object)
    vh_ct = vh.conjugate().transpose()

    for i in range(ne):
        # calculate S^{r}\left(E\right)
        sr_vec[i] = vh @ pr_vec[i]

        # calculate L^{\lessgtr}\left(E\right)
        lg_vec[i] = vh @ pg_vec[i] @ vh_ct
        ll_vec[i] = vh @ pl_vec[i] @ vh_ct

    return sr_vec, lg_vec, ll_vec

def p2w_mpi_cpu_alt(
    hamiltionian_obj: object,
    ij2ji: npt.NDArray[np.int32],
    rows: npt.NDArray[np.int32],
    columns: npt.NDArray[np.int32],
    pg: npt.NDArray[np.complex128],
    pl: npt.NDArray[np.complex128],
    pr: npt.NDArray[np.complex128],
    vh: npt.NDArray[np.complex128],
    factors: npt.NDArray[np.float64],
    map_diag_mm: npt.NDArray[np.int32],
    map_upper_mm: npt.NDArray[np.int32],
    map_lower_mm: npt.NDArray[np.int32],
    mkl_threads: int = 1
) -> typing.Tuple[
        npt.NDArray[np.complex128],
        npt.NDArray[np.complex128],
        npt.NDArray[np.complex128]
    ]:

    # todo try to not additionally symmetrize
    # Anti-Hermitian symmetrizing of pl and pg
    pg = 1j * np.imag(pg)
    pg = (pg - pg[:,ij2ji].conjugate()) / 2
    pl = 1j * np.imag(pl)
    pl = (pl - pl[:,ij2ji].conjugate()) / 2


    # pr has to be derived from pl and pg and then has to be symmetrized
    pr = 1j * np.imag(pg - pl) / 2
    pr = (pr + pr[:,ij2ji]) / 2


    # number of energy points and nonzero elements
    ne = pg.shape[0]
    no = pg.shape[1]
    nao = np.max(hamiltionian_obj.Bmax)

    # number of blocks
    nb = hamiltionian_obj.Bmin.shape[0]
    # start and end index of each block in python indexing
    bmax = hamiltionian_obj.Bmax - 1
    bmin = hamiltionian_obj.Bmin - 1

    # fix nbc to 2 for the given solution
    # todo calculate it
    nbc = 2
    # block sizes after matrix multiplication
    bmax_mm = bmax[nbc-1:nb:nbc]
    bmin_mm = bmin[0:nb:nbc]
    # number of blocks after matrix multiplication
    nb_mm = bmax_mm.size
    # larges block length after matrix multiplication
    lb_max_mm = np.max(bmax_mm - bmin_mm + 1)

    # create empty buffer for screened interaction
    # diagonal blocks
    xr_diag  = np.zeros((ne, nb_mm, lb_max_mm, lb_max_mm), dtype = np.complex128)
    wg_diag  = np.zeros((ne, nb_mm, lb_max_mm, lb_max_mm), dtype = np.complex128)
    wl_diag  = np.zeros((ne, nb_mm, lb_max_mm, lb_max_mm), dtype = np.complex128)
    wr_diag  = np.zeros((ne, nb_mm, lb_max_mm, lb_max_mm), dtype = np.complex128)
    # upper diagonal blocks
    wg_upper = np.zeros((ne, nb_mm-1, lb_max_mm, lb_max_mm), dtype = np.complex128)
    wl_upper = np.zeros((ne, nb_mm-1, lb_max_mm, lb_max_mm), dtype = np.complex128)
    wr_upper = np.zeros((ne, nb_mm-1, lb_max_mm, lb_max_mm), dtype = np.complex128)

    # set number of mkl threads
    mkl.set_num_threads(mkl_threads)

    # copy vh to overwrite it
    time_vec = -time.perf_counter()
    vh_sparse = sparse.coo_array((vh, (rows, columns)),
                          shape=(nao, nao), dtype = np.complex128).tocsr()



    # transform from 2D format to list/vector of sparse arrays format
    pg_vec = change_format.sparse2vecsparse_v2(pg, rows, columns, nao)
    pl_vec = change_format.sparse2vecsparse_v2(pl, rows, columns, nao)
    pr_vec = change_format.sparse2vecsparse_v2(pr, rows, columns, nao)
    time_vec += time.perf_counter()
    print("Time vec: ", time_vec)

    # compute helper arrays
    # sr is not the self energy, but helper variable
    time_w2s = -time.perf_counter()
    sr_vec, lg_vec, ll_vec = w2s_l(vh_sparse, pg_vec, pl_vec, pr_vec)
    mr_vec = np.ndarray((ne,), dtype=object)
    time_w2s += time.perf_counter()
    print("Time w2s: ", time_w2s)

    # boundary conditions
    time_bc = -time.perf_counter()
    vh_vec = np.array([vh_sparse.copy() for i in range(ne)])
    compute_flag = np.array([False for i in range(ne)])
    for i in range(ne):
        compute_flag[i], mr_vec[i] = obc_w.obc_w(
                                pg_vec[i],
                                pl_vec[i],
                                pr_vec[i],
                                vh_vec[i],
                                sr_vec[i],
                                lg_vec[i],
                                ll_vec[i],
                                bmax,
                                bmin,
                                nbc)
    time_bc += time.perf_counter()
    print("Time boundary conditions: ", time_bc)

    time_inv = -time.perf_counter()
    # calculate the inversion for every energy point
    for i in range(ne):
        if compute_flag[i]:
            matrix_inversion_w.rgf(
                bmax_mm,
                bmin_mm,
                vh_vec[i],
                mr_vec[i],
                lg_vec[i],
                ll_vec[i],
                factors[i],
                wg_diag[i],
                wg_upper[i],
                wl_diag[i],
                wl_upper[i],
                wr_diag[i],
                wr_upper[i],
                xr_diag[i]
            )
    time_inv += time.perf_counter()
    print("Time inversion: ", time_inv)

    time_block = -time.perf_counter()
    # lower blocks from identity
    wg_lower = -wg_upper.conjugate().transpose((0,1,3,2))
    wl_lower = -wl_upper.conjugate().transpose((0,1,3,2))
    wr_lower = wr_upper.transpose((0,1,3,2))

    # tranform to 2D format
    wg = change_format.block2sparse_energy_alt(map_diag_mm, map_upper_mm,
                                                    map_lower_mm, wg_diag, wg_upper,
                                                    wg_lower, no, ne,
                                                    energy_contiguous=False)
    wl = change_format.block2sparse_energy_alt(map_diag_mm, map_upper_mm,
                                                    map_lower_mm, wl_diag, wl_upper,
                                                    wl_lower, no, ne,
                                                    energy_contiguous=False)
    wr = change_format.block2sparse_energy_alt(map_diag_mm, map_upper_mm,
                                                    map_lower_mm, wr_diag, wr_upper,
                                                    wr_lower, no, ne,
                                                    energy_contiguous=False)
    time_block += time.perf_counter()
    print("Time block: ", time_block)
    return wg, wl, wr
