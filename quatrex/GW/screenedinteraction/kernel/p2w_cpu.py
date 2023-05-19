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
from OBC import obc_w_cpu
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

    # create timing vector
    times = np.zeros((10))

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
        times += rgf_W.rgf_w_opt(
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
    print("Time symmetrize: ", times[0])
    print("Time sr,lg,ll arrays: ", times[1])
    print("Time scattering obc: ", times[2])
    print("Time beyn obc: ", times[3])
    print("Time dl obc: ", times[4])
    print("Time inversion: ", times[5])

    return wg_diag, wg_upper, wl_diag, wl_upper, wr_diag, wr_upper, nb_mm, lb_max_mm

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
    map_diag_mm2m: npt.NDArray[np.int32],
    map_upper_mm2m: npt.NDArray[np.int32],
    map_lower_mm2m: npt.NDArray[np.int32],
    mkl_threads: int = 1
) -> typing.Tuple[
        npt.NDArray[np.complex128],
        npt.NDArray[np.complex128],
        npt.NDArray[np.complex128]
    ]:
    """
    Calculates the screened interaction on the cpu.
    Uses only mkl threading.
    Splits up the screened interaction calculations into six parts:
    - Symmetrization of the polarization.
    - Change of the format into vectors of sparse matrices.
    - Calculation of helper variables.
    - Application of the scattering boundary conditions.
    - Application of the beyn boundary conditions.
    - Inversion

    Args:
        hamiltionian_obj (object): Class containing the hamiltonian information
        ij2ji (npt.NDArray[np.int32]): Vector to transpose in orbitals
        rows (npt.NDArray[np.int32]): Non zero row indices of input matrices
        columns (npt.NDArray[np.int32]): Non zero column indices of input matrices
        pg (npt.NDArray[np.complex128]): Greater polarization, (#energy, #nnz)
        pl (npt.NDArray[np.complex128]): Greater polarization, (#energy, #nnz)
        pr (npt.NDArray[np.complex128]): Greater polarization, (#energy, #nnz)
        vh (npt.NDArray[np.complex128]): Effective interaction, (#nnz)
        factors (npt.NDArray[np.float64]): Factors to smooth in energy, multiplied at the end, (#energy)
        map_diag_mm2m (npt.NDArray[np.int32]): Map from dense block diagonal to 2D
        map_upper_mm2m (npt.NDArray[np.int32]): Map from dense block upper to 2D
        map_upper_mm2m (npt.NDArray[np.int32]): Map from dense block lower to 2D
        mkl_threads (int, optional): Number of mkl threads to use. Defaults to 1.

    Returns:
        typing.Tuple[ npt.NDArray[np.complex128], npt.NDArray[np.complex128], npt.NDArray[np.complex128] ]: Greater, lesser, retarded screened interaction
    """

    # create timing vector
    times = np.zeros((10))

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



    times[0] = -time.perf_counter()
    # todo try to not additionally symmetrize
    # Anti-Hermitian symmetrizing of pl and pg
    pg = 1j * np.imag(pg)
    pg = (pg - pg[:,ij2ji].conjugate()) / 2
    pl = 1j * np.imag(pl)
    pl = (pl - pl[:,ij2ji].conjugate()) / 2
    # pr has to be derived from pl and pg and then has to be symmetrized
    pr = 1j * np.imag(pg - pl) / 2
    pr = (pr + pr[:,ij2ji]) / 2
    times[0] += time.perf_counter()


    # copy vh to overwrite it
    times[1] = -time.perf_counter()
    # todo possible to merge with later computations
    vh_sparse = sparse.csr_array((vh, (rows, columns)),
                          shape=(nao, nao), dtype = np.complex128)
    # transform from 2D format to list/vector of sparse arrays format
    pg_vec = change_format.sparse2vecsparse_v2(pg, rows, columns, nao)
    pl_vec = change_format.sparse2vecsparse_v2(pl, rows, columns, nao)
    pr_vec = change_format.sparse2vecsparse_v2(pr, rows, columns, nao)
    times[1] += time.perf_counter()
    

    # compute helper arrays
    # sr is not the self energy, but a helper variable
    times[2] = -time.perf_counter()


    vh_vec = np.array([vh_sparse.copy() for i in range(ne)])
    mr_vec = np.ndarray((ne,), dtype=object)
    sr_vec = np.ndarray((ne,), dtype=object)
    lg_vec = np.ndarray((ne,), dtype=object)
    ll_vec = np.ndarray((ne,), dtype=object)
    for i in range(ne):
        sr_vec[i], lg_vec[i], ll_vec[i] = obc_w_cpu.obc_w_sl(vh_vec[i], pg_vec[i], pl_vec[i], pr_vec[i])

    times[2] += time.perf_counter()



    # boundary conditions
    times[3] = -time.perf_counter()
    cond_l = np.zeros((ne), dtype=np.float64)
    cond_r = np.zeros((ne), dtype=np.float64)

    for i in range(ne):
        mr_vec[i] = obc_w_cpu.obc_w_sc(
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
    times[3] += time.perf_counter()


    times[4] = -time.perf_counter()
    dxr_sd = np.ndarray((ne,), dtype=object)
    dxr_ed = np.ndarray((ne,), dtype=object)
    dmr_sd = np.ndarray((ne,), dtype=object)
    dmr_ed = np.ndarray((ne,), dtype=object)
    dvh_sd = np.ndarray((ne,), dtype=object)
    dvh_ed = np.ndarray((ne,), dtype=object)
    for i in range(ne):
        cond_r[i], cond_l[i], dxr_sd[i], dxr_ed[i], dmr_sd[i], dmr_ed[i], dvh_sd[i], dvh_ed[i] = obc_w_cpu.obc_w_beyn(
                        pr_vec[i],
                        vh_vec[i],
                        bmax,
                        bmin,
                        nbc)
        
    times[4] += time.perf_counter()

    times[5] = -time.perf_counter()
    for i in range(ne):
        cond_r[i], cond_l[i] = obc_w_cpu.obc_w_dl(
                                pg_vec[i],
                                pl_vec[i],
                                pr_vec[i],
                                vh_vec[i],
                                mr_vec[i],
                                lg_vec[i],
                                ll_vec[i],
                                dxr_sd[i],
                                dxr_ed[i],
                                dmr_sd[i],
                                dmr_ed[i],
                                dvh_sd[i],
                                dvh_ed[i],
                                bmax,
                                bmin,
                                nbc,
                                cond_l[i],
                                cond_r[i])
    times[5] += time.perf_counter()


    times[6] = -time.perf_counter()
    # calculate the inversion for every energy point
    for i in range(ne):
        if not np.isnan(cond_r[i]) and not np.isnan(cond_l[i]):
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
    times[6] += time.perf_counter()
    

    times[7] = -time.perf_counter()
    # lower blocks from identity
    wg_lower = -wg_upper.conjugate().transpose((0,1,3,2))
    wl_lower = -wl_upper.conjugate().transpose((0,1,3,2))
    wr_lower = wr_upper.transpose((0,1,3,2))

    # tranform to 2D format
    wg = change_format.block2sparse_energy_alt(map_diag_mm2m, map_upper_mm2m,
                                                    map_lower_mm2m, wg_diag, wg_upper,
                                                    wg_lower, no, ne,
                                                    energy_contiguous=False)
    wl = change_format.block2sparse_energy_alt(map_diag_mm2m, map_upper_mm2m,
                                                    map_lower_mm2m, wl_diag, wl_upper,
                                                    wl_lower, no, ne,
                                                    energy_contiguous=False)
    wr = change_format.block2sparse_energy_alt(map_diag_mm2m, map_upper_mm2m,
                                                    map_lower_mm2m, wr_diag, wr_upper,
                                                    wr_lower, no, ne,
                                                    energy_contiguous=False)
    times[7] += time.perf_counter()


    print("Time symmetrize: ", times[0])
    print("Time to list: ", times[1])
    print("Time sr,lg,ll arrays: ", times[2])
    print("Time scattering obc: ", times[3])
    print("Time beyn obc: ", times[4])
    print("Time dl obc: ", times[5])
    print("Time inversion: ", times[6])
    print("Time block: ", times[7])
    return wg, wl, wr


def p2w_mpi_cpu_block(
    hamiltionian_obj: object,
    ij2ji: npt.NDArray[np.int32],
    rows: npt.NDArray[np.int32],
    columns: npt.NDArray[np.int32],
    pg: npt.NDArray[np.complex128],
    pl: npt.NDArray[np.complex128],
    pr: npt.NDArray[np.complex128],
    vh: npt.NDArray[np.complex128],
    factors: npt.NDArray[np.float64],
    map_diag_mm2m: npt.NDArray[np.int32],
    map_upper_mm2m: npt.NDArray[np.int32],
    map_lower_mm2m: npt.NDArray[np.int32],
    mkl_threads: int = 1
) -> typing.Tuple[
        npt.NDArray[np.complex128],
        npt.NDArray[np.complex128],
        npt.NDArray[np.complex128]
    ]:
    """
    Calculates the screened interaction on the cpu.
    Uses only mkl threading.
    Splits up the screened interaction calculations into six parts:
    - Symmetrization of the polarization.
    - Change of the format into vectors of sparse matrices.
    - Calculation of helper variables.
    - Application of the scattering boundary conditions.
    - Application of the beyn boundary conditions.
    - Inversion

    Args:
        hamiltionian_obj (object): Class containing the hamiltonian information
        ij2ji (npt.NDArray[np.int32]): Vector to transpose in orbitals
        rows (npt.NDArray[np.int32]): Non zero row indices of input matrices
        columns (npt.NDArray[np.int32]): Non zero column indices of input matrices
        pg (npt.NDArray[np.complex128]): Greater polarization, (#energy, #nnz)
        pl (npt.NDArray[np.complex128]): Greater polarization, (#energy, #nnz)
        pr (npt.NDArray[np.complex128]): Greater polarization, (#energy, #nnz)
        vh (npt.NDArray[np.complex128]): Effective interaction, (#nnz)
        factors (npt.NDArray[np.float64]): Factors to smooth in energy, multiplied at the end, (#energy)
        map_diag_mm2mm (npt.NDArray[np.int32]): Map from 2D after mm to dense block diagonal
        map_upper_mm2mm (npt.NDArray[np.int32]): Map from 2D after mm to dense block upper
        map_upper_mm2mm (npt.NDArray[np.int32]): Map from 2D after mm to dense block lower
        map_diag_mm2m (npt.NDArray[np.int32]): Map from dense block diagonal to 2D
        map_upper_mm2m (npt.NDArray[np.int32]): Map from dense block upper to 2D
        map_upper_mm2m (npt.NDArray[np.int32]): Map from dense block lower to 2D
        mkl_threads (int, optional): Number of mkl threads to use. Defaults to 1.

    Returns:
        typing.Tuple[ npt.NDArray[np.complex128], npt.NDArray[np.complex128], npt.NDArray[np.complex128] ]: Greater, lesser, retarded screened interaction
    """
    # create timing vector
    times = np.zeros((10))

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
    mr_diag  = np.zeros((ne, nb_mm, lb_max_mm, lb_max_mm), dtype = np.complex128)
    sr_diag  = np.zeros((ne, nb_mm, lb_max_mm, lb_max_mm), dtype = np.complex128)
    lg_diag  = np.zeros((ne, nb_mm, lb_max_mm, lb_max_mm), dtype = np.complex128)
    ll_diag  = np.zeros((ne, nb_mm, lb_max_mm, lb_max_mm), dtype = np.complex128)

    # upper diagonal blocks and lower blocks
    wg_upper = np.zeros((ne, nb_mm-1, lb_max_mm, lb_max_mm), dtype = np.complex128)
    wl_upper = np.zeros((ne, nb_mm-1, lb_max_mm, lb_max_mm), dtype = np.complex128)
    wr_upper = np.zeros((ne, nb_mm-1, lb_max_mm, lb_max_mm), dtype = np.complex128)
    mr_upper = np.zeros((ne, nb_mm-1, lb_max_mm, lb_max_mm), dtype = np.complex128)
    sr_upper = np.zeros((ne, nb_mm-1, lb_max_mm, lb_max_mm), dtype = np.complex128)
    lg_upper = np.zeros((ne, nb_mm-1, lb_max_mm, lb_max_mm), dtype = np.complex128)
    ll_upper = np.zeros((ne, nb_mm-1, lb_max_mm, lb_max_mm), dtype = np.complex128)
    mr_lower = np.zeros((ne, nb_mm-1, lb_max_mm, lb_max_mm), dtype = np.complex128)
    sr_lower = np.zeros((ne, nb_mm-1, lb_max_mm, lb_max_mm), dtype = np.complex128)
    lg_lower = np.zeros((ne, nb_mm-1, lb_max_mm, lb_max_mm), dtype = np.complex128)
    ll_lower = np.zeros((ne, nb_mm-1, lb_max_mm, lb_max_mm), dtype = np.complex128)

    # set number of mkl threads
    mkl.set_num_threads(mkl_threads)



    times[0] = -time.perf_counter()
    # todo try to not additionally symmetrize
    # Anti-Hermitian symmetrizing of pl and pg
    pg = 1j * np.imag(pg)
    pg = (pg - pg[:,ij2ji].conjugate()) / 2
    pl = 1j * np.imag(pl)
    pl = (pl - pl[:,ij2ji].conjugate()) / 2
    # pr has to be derived from pl and pg and then has to be symmetrized
    pr = 1j * np.imag(pg - pl) / 2
    pr = (pr + pr[:,ij2ji]) / 2
    times[0] += time.perf_counter()


    # copy vh to overwrite it
    times[1] = -time.perf_counter()
    # todo possible to merge with later computations
    vh_sparse = sparse.csr_array((vh, (rows, columns)),
                          shape=(nao, nao), dtype = np.complex128)
    # transform from 2D format to list/vector of sparse arrays format
    pg_vec = change_format.sparse2vecsparse_v2(pg, rows, columns, nao)
    pl_vec = change_format.sparse2vecsparse_v2(pl, rows, columns, nao)
    pr_vec = change_format.sparse2vecsparse_v2(pr, rows, columns, nao)
    times[1] += time.perf_counter()
    

    # compute helper arrays
    # sr is not the self energy, but a helper variable
    times[2] = -time.perf_counter()



    for i in range(ne):
        pg_e = sparse.bsr_matrix((pg[i,:], (rows, columns)))
        pl_e = sparse.bsr_matrix((pl[i,:], (rows, columns)))
        pr_e = sparse.bsr_matrix((pr[i,:], (rows, columns)))
        sr_e, lg_e, ll_e = obc_w_cpu.obc_w_sl(vh_sparse, pg_e, pl_e, pr_e)
        change_format.sparse2block_no_map(sr_e, sr_diag[i], sr_upper[i], sr_lower[i], bmax_mm, bmin_mm)
        change_format.sparse2block_no_map(lg_e, lg_diag[i], lg_upper[i], lg_lower[i], bmax_mm, bmin_mm)
        change_format.sparse2block_no_map(ll_e, ll_diag[i], ll_upper[i], ll_lower[i], bmax_mm, bmin_mm)

    times[2] += time.perf_counter()



    # boundary conditions
    times[3] = -time.perf_counter()
    cond_l = np.zeros((ne), dtype=np.float64)
    cond_r = np.zeros((ne), dtype=np.float64)

    for i in range(ne):
        mr_vec[i] = obc_w_cpu.obc_w_sc(
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
    times[3] += time.perf_counter()


    times[4] = -time.perf_counter()
    dxr_sd = np.ndarray((ne,), dtype=object)
    dxr_ed = np.ndarray((ne,), dtype=object)
    dmr_sd = np.ndarray((ne,), dtype=object)
    dmr_ed = np.ndarray((ne,), dtype=object)
    dvh_sd = np.ndarray((ne,), dtype=object)
    dvh_ed = np.ndarray((ne,), dtype=object)
    for i in range(ne):
        cond_r[i], cond_l[i], dxr_sd[i], dxr_ed[i], dmr_sd[i], dmr_ed[i], dvh_sd[i], dvh_ed[i] = obc_w_cpu.obc_w_beyn(
                        pr_vec[i],
                        vh_vec[i],
                        bmax,
                        bmin,
                        nbc)
        
    times[4] += time.perf_counter()

    times[5] = -time.perf_counter()
    for i in range(ne):
        cond_r[i], cond_l[i] = obc_w_cpu.obc_w_dl(
                                pg_vec[i],
                                pl_vec[i],
                                pr_vec[i],
                                vh_vec[i],
                                mr_vec[i],
                                lg_vec[i],
                                ll_vec[i],
                                dxr_sd[i],
                                dxr_ed[i],
                                dmr_sd[i],
                                dmr_ed[i],
                                dvh_sd[i],
                                dvh_ed[i],
                                bmax,
                                bmin,
                                nbc,
                                cond_l[i],
                                cond_r[i])
    times[5] += time.perf_counter()


    times[6] = -time.perf_counter()
    # calculate the inversion for every energy point
    for i in range(ne):
        if not np.isnan(cond_r[i]) and not np.isnan(cond_l[i]):
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
    times[6] += time.perf_counter()
    

    times[7] = -time.perf_counter()
    # lower blocks from identity
    wg_lower = -wg_upper.conjugate().transpose((0,1,3,2))
    wl_lower = -wl_upper.conjugate().transpose((0,1,3,2))
    wr_lower = wr_upper.transpose((0,1,3,2))

    # tranform to 2D format
    wg = change_format.block2sparse_energy_alt(map_diag_mm2m, map_upper_mm2m,
                                                    map_lower_mm2m, wg_diag, wg_upper,
                                                    wg_lower, no, ne,
                                                    energy_contiguous=False)
    wl = change_format.block2sparse_energy_alt(map_diag_mm2m, map_upper_mm2m,
                                                    map_lower_mm2m, wl_diag, wl_upper,
                                                    wl_lower, no, ne,
                                                    energy_contiguous=False)
    wr = change_format.block2sparse_energy_alt(map_diag_mm2m, map_upper_mm2m,
                                                    map_lower_mm2m, wr_diag, wr_upper,
                                                    wr_lower, no, ne,
                                                    energy_contiguous=False)
    times[7] += time.perf_counter()


    print("Time symmetrize: ", times[0])
    print("Time to list: ", times[1])
    print("Time sr,lg,ll arrays: ", times[2])
    print("Time scattering obc: ", times[3])
    print("Time beyn obc: ", times[4])
    print("Time dl obc: ", times[5])
    print("Time inversion: ", times[6])
    print("Time block: ", times[7])
    return wg, wl, wr

