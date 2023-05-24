"""
Functions to calculate the screened interaction on the gpu
See README.md for more information. 
"""
import numpy as np
import numpy.typing as npt
import cupy as cp
from scipy import sparse
from cupyx.scipy import sparse as cusparse
import mkl
import typing
from utils import change_format
from block_tri_solvers import matrix_inversion_w
from OBC import obc_w_gpu
from OBC import obc_w_cpu
import time

def p2w_mpi_gpu(
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
    map_lower_mm2m: npt.NDArray[np.int32]
) -> typing.Tuple[
        npt.NDArray[np.complex128],
        npt.NDArray[np.complex128],
        npt.NDArray[np.complex128]
    ]:
    """
    Calculates the screened interaction on the cpu.
    Uses the gpu for most of the calculations.
    Splits up the screened interaction calculations into six parts:
    - Symmetrization of the polarization.
    - Change of the format into vectors of sparse matrices
    + Calculation of helper variables.
    + calculation of the blocks for boundary conditions.
    - calculation of the beyn boundary conditions.
    - calculation of the dl boundary conditions.
    - Inversion
    - back transformation into 2D format

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
    # block lengths
    lb = bmax - bmin + 1
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

    # create buffer for boundary conditions
    mr_sf = np.zeros((ne, 3, nbc*lb[0], nbc*lb[0]), dtype = np.complex128)
    mr_ef = np.zeros((ne, 3, nbc*lb[0], nbc*lb[0]), dtype = np.complex128)
    lg_sf = np.zeros((ne, 3, nbc*lb[0], nbc*lb[0]), dtype = np.complex128)
    lg_ef = np.zeros((ne, 3, nbc*lb[0], nbc*lb[0]), dtype = np.complex128)
    ll_sf = np.zeros((ne, 3, nbc*lb[0], nbc*lb[0]), dtype = np.complex128)
    ll_ef = np.zeros((ne, 3, nbc*lb[0], nbc*lb[0]), dtype = np.complex128)
    dxr_sf = np.zeros((ne, 1, nbc*lb[0], nbc*lb[0]), dtype = np.complex128)
    dxr_ef = np.zeros((ne, 1, nbc*lb[0], nbc*lb[0]), dtype = np.complex128)    
    dmr_sf = np.zeros((ne, 1, nbc*lb[0], nbc*lb[0]), dtype = np.complex128)
    dmr_ef = np.zeros((ne, 1, nbc*lb[0], nbc*lb[0]), dtype = np.complex128)
    dlg_sf = np.zeros((ne, 1, nbc*lb[0], nbc*lb[0]), dtype = np.complex128)
    dlg_ef = np.zeros((ne, 1, nbc*lb[0], nbc*lb[0]), dtype = np.complex128)
    dll_sf = np.zeros((ne, 1, nbc*lb[0], nbc*lb[0]), dtype = np.complex128)
    dll_ef = np.zeros((ne, 1, nbc*lb[0], nbc*lb[0]), dtype = np.complex128)
    vh_sf = np.zeros((ne, 1, nbc*lb[0], nbc*lb[0]), dtype = np.complex128)
    vh_ef = np.zeros((ne, 1, nbc*lb[0], nbc*lb[0]), dtype = np.complex128)
    dvh_sf = np.zeros((ne, 1, nbc*lb[0], nbc*lb[0]), dtype = np.complex128)
    dvh_ef = np.zeros((ne, 1, nbc*lb[0], nbc*lb[0]), dtype = np.complex128)

    # vh = np.copy(vh)
    pg = np.copy(pg)
    pl = np.copy(pl)
    pr = np.copy(pr)
    vh_gpu = cp.empty_like(vh)
    pg_gpu = cp.empty_like(pg)
    pl_gpu = cp.empty_like(pl)
    pr_gpu = cp.empty_like(pr)
    ij2ji_gpu = cp.empty_like(ij2ji)
    rows_gpu = cp.empty_like(rows)
    columns_gpu = cp.empty_like(columns)
    mr_gpu_sf = cp.empty_like(mr_sf)
    mr_gpu_ef = cp.empty_like(mr_ef)
    lg_gpu_sf = cp.empty_like(lg_sf)
    lg_gpu_ef = cp.empty_like(lg_ef)
    ll_gpu_sf = cp.empty_like(ll_sf)
    ll_gpu_ef = cp.empty_like(ll_ef)
    dmr_gpu_sf = cp.empty_like(dmr_sf)
    dmr_gpu_ef = cp.empty_like(dmr_ef)
    dlg_gpu_sf = cp.empty_like(dlg_sf)
    dlg_gpu_ef = cp.empty_like(dlg_ef)
    dll_gpu_sf = cp.empty_like(dll_sf)
    dll_gpu_ef = cp.empty_like(dll_ef)
    vh_gpu_sf = cp.empty_like(vh_sf)
    vh_gpu_ef = cp.empty_like(vh_ef)


    times[0] = -time.perf_counter()

    # load data to the gpu
    vh_gpu.set(vh)
    pg_gpu.set(pg)
    pl_gpu.set(pl)
    pr_gpu.set(pr)
    ij2ji_gpu.set(ij2ji)
    rows_gpu.set(rows)
    columns_gpu.set(columns)

    # todo try to not additionally symmetrize
    # Anti-Hermitian symmetrizing of pl and pg
    pg_gpu = 1j * cp.imag(pg_gpu)
    pg_gpu = (pg_gpu - pg_gpu[:,ij2ji_gpu].conjugate()) / 2
    pl_gpu = 1j * cp.imag(pl_gpu)
    pl_gpu = (pl_gpu - pl_gpu[:,ij2ji_gpu].conjugate()) / 2
    # pr has to be derived from pl and pg and then has to be symmetrized
    pr_gpu = 1j * cp.imag(pg_gpu - pl_gpu) / 2
    pr_gpu = (pr_gpu + pr_gpu[:,ij2ji_gpu]) / 2

    # unload
    # pg_gpu.get(out=pg)
    # pl_gpu.get(out=pl)
    # pr_gpu.get(out=pr)

    times[0] += time.perf_counter()

    # compute helper arrays
    times[1] = -time.perf_counter()
    vh_sparse = cusparse.csr_matrix((vh_gpu, (rows_gpu, columns_gpu)),
                          shape=(nao, nao), dtype = cp.complex128)
    mr_vec = []
    lg_vec = []
    ll_vec = []
    for i in range(ne):
        pg_s = cusparse.csr_matrix((pg_gpu[i,:,], (rows_gpu, columns_gpu)),
                            shape=(nao, nao), dtype = cp.complex128)
        pl_s = cusparse.csr_matrix((pl_gpu[i,:,], (rows_gpu, columns_gpu)),
                            shape=(nao, nao), dtype = cp.complex128)
        pr_s = cusparse.csr_matrix((pr_gpu[i,:,], (rows_gpu, columns_gpu)),
                            shape=(nao, nao), dtype = cp.complex128)
        mr, lg, ll = obc_w_gpu.obc_w_sl(vh_sparse, pg_s, pl_s, pr_s, nao)
        mr_vec.append(mr)
        lg_vec.append(lg)
        ll_vec.append(ll)
        obc_w_gpu.obc_w_sc(
                    pg_s,
                    pl_s,
                    pr_s,
                    vh_sparse,
                    mr_gpu_sf[i],
                    mr_gpu_ef[i],
                    lg_gpu_sf[i],
                    lg_gpu_ef[i],
                    ll_gpu_sf[i],
                    ll_gpu_ef[i],
                    dmr_gpu_sf[i],
                    dmr_gpu_ef[i],
                    dlg_gpu_sf[i],
                    dlg_gpu_ef[i],
                    dll_gpu_sf[i],
                    dll_gpu_ef[i],
                    vh_gpu_sf[i],
                    vh_gpu_ef[i],
                    bmax,
                    bmin,
                    nbc)
    # unload
    mr_gpu_sf.get(out=mr_sf)
    mr_gpu_ef.get(out=mr_ef)
    lg_gpu_sf.get(out=lg_sf)
    lg_gpu_ef.get(out=lg_ef)
    ll_gpu_sf.get(out=ll_sf)
    ll_gpu_ef.get(out=ll_ef)
    dmr_gpu_sf.get(out=dmr_sf)
    dmr_gpu_ef.get(out=dmr_ef)
    dlg_gpu_sf.get(out=dlg_sf)
    dlg_gpu_ef.get(out=dlg_ef)
    dll_gpu_sf.get(out=dll_sf)
    dll_gpu_ef.get(out=dll_ef)
    vh_gpu_sf.get(out=vh_sf)
    vh_gpu_ef.get(out=vh_ef)

    times[1] += time.perf_counter()


    times[2] = -time.perf_counter()
    cond_l = np.zeros((ne), dtype=np.float64)
    cond_r = np.zeros((ne), dtype=np.float64)
    for i in range(ne):
        cond_r[i], cond_l[i] = obc_w_cpu.obc_w_beyn(
                        dxr_sf[i],
                        dxr_ef[i],
                        mr_sf[i],
                        mr_ef[i],
                        vh_sf[i],
                        vh_ef[i],
                        dmr_sf[i],
                        dmr_ef[i],
                        dvh_sf[i],
                        dvh_ef[i])
        
    times[2] += time.perf_counter()

    times[3] = -time.perf_counter()
    for i in range(ne):
        if not np.isnan(cond_r[i]) and not np.isnan(cond_l[i]):
            obc_w_cpu.obc_w_dl(
                    dxr_sf[i],
                    dxr_ef[i],
                    lg_sf[i],
                    lg_ef[i],
                    ll_sf[i],
                    ll_ef[i],
                    mr_sf[i],
                    mr_ef[i],
                    dlg_sf[i],
                    dll_sf[i],
                    dlg_ef[i],
                    dll_ef[i])
    times[3] += time.perf_counter()


    times[4] = -time.perf_counter()
    # calculate the inversion for every energy point
    for i in range(ne):
        if not np.isnan(cond_r[i]) and not np.isnan(cond_l[i]):
            matrix_inversion_w.rgf(
                bmax_mm,
                bmin_mm,
                vh_sparse.get(),
                mr_vec[i].get(),
                lg_vec[i].get(),
                ll_vec[i].get(),
                factors[i],
                wg_diag[i],
                wg_upper[i],
                wl_diag[i],
                wl_upper[i],
                wr_diag[i],
                wr_upper[i],
                xr_diag[i],
                dmr_ef[i,0],
                dlg_ef[i,0],
                dll_ef[i,0],
                dvh_ef[i,0],
                dmr_sf[i,0],
                dlg_sf[i,0],
                dll_sf[i,0],
                dvh_sf[i,0]
            )
    times[4] += time.perf_counter()
    

    times[5] = -time.perf_counter()
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
    times[5] += time.perf_counter()


    print("Time symmetrize: ", times[0])
    print("Time to list + sr,lg,ll arrays + scattering obc: ", times[1])
    print("Time beyn obc: ", times[2])
    print("Time dl obc: ", times[3])
    print("Time inversion: ", times[4])
    print("Time block: ", times[5])
    return wg, wl, wr
