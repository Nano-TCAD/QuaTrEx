"""
Functions to calculate the screened interaction on the gpu
See README.md for more information. 
"""
import numpy as np
import numpy.typing as npt
import cupy as cp
from cupyx.scipy import sparse
import mkl
import typing
from utils import change_format
from block_tri_solvers import matrix_inversion_w
from OBC import obc_w_gpu
import time

class Slice_w2p():
    """
    Slice the w2p matrix into the different blocks.
    """
    def __init__(
            self,
            bmax: npt.NDArray[np.int32],
            bmin: npt.NDArray[np.int32],
            nao: int,
            nbc: int
    ):
        """
        Initialize the slice object.

        Args:
            bmax (npt.NDArray[np.int32]): End index of the blocks
            bmin (npt.NDArray[np.int32]): Start index of the blocks
            nao (int): Number of orbitals
            nbc (int): How block size changes after matrix multiplication (in number of current block size)
        """
        # limit for beyn
        self.imag_lim = 1e-4
        # todo find out what rr/R is
        # (R only is against the style guide)
        self.rr = 1e12

        self.bmax = bmax
        self.bmin = bmin
        self.nao = nao
        self.nbc = nbc

        # number of blocks
        self.nb = bmin.size
        # number of total orbitals (number of atomic orbitals)
        self.nao = nao

        # vector of block lengths
        self.lb_vec = bmax - bmin + 1
        # length of first and last block
        self.lb_start = self.lb_vec[0]
        self.lb_end = self.lb_vec[self.nb-1]

        # slice block start diagonal and slice block start off diagonal
        self.slb_sd = slice(0,self.lb_start)
        self.slb_so = slice(self.lb_start,2*self.lb_start)
        # slice block end diagonal and slice block end off diagonal
        self.slb_ed = slice(nao-self.lb_end,nao)
        self.slb_eo = slice(nao-2*self.lb_end,nao-self.lb_end)

        # slice block start diagonal
        # for block after matrix multiplication
        self.slb_sd_mm = slice(bmin[0], bmax[self.nbc-1] + 1)
        # slice block end diagonal
        # for block after matrix multiplication
        self.slb_ed_mm = slice(bmin[self.nb - self.nbc], nao)

        # block sizes after matrix multiplication
        self.bmax_mm = bmax[nbc-1:self.nb:nbc]
        self.bmin_mm = bmin[0:self.nb:nbc]
        # number of blocks after matrix multiplication
        self.nb_mm = self.bmax_mm.size
        # larges block length after matrix multiplication
        self.lb_max_mm = np.max(self.bmax_mm - self.bmin_mm + 1)

def w2s_l(
    vh: cp.ndarray,
    pg: cp.ndarray,
    pl: cp.ndarray,
    pr: cp.ndarray,
    rows: cp.ndarray,
    columns: cp.ndarray
) -> typing.Tuple[sparse.csr_matrix,
                  sparse.csr_matrix,
                  sparse.csr_matrix,
                  sparse.csr_matrix,
                  sparse.csr_matrix,
                  sparse.csr_matrix,
                  sparse.csr_matrix]:
    """
    Calculates the additional helper variables.

    Args:
        vh (cp.ndarray): Effective interaction, data vector
        pg (cp.ndarray): Greater Polarization, data vector
        pl (cp.ndarray): Lesser Polarization, data vector
        pr (cp.ndarray): Retarded Polarization, data vector
        rows (cp.ndarray): row indices
        columns (cp.ndarray): column indices

    Returns:
        typing.Tuple[
            sparse.csr_matrix,
            sparse.csr_matrix,
            sparse.csr_matrix,
            sparse.csr_matrix,
            sparse.csr_matrix,
            sparse.csr_matrix,
            sparse.csr_matrix]: S^{r}\left(E\right), L^{>}\left(E\right), L^{<}\left(E\right)
            and the vh/pg/pl/pr as sparse csr
    """

    # create csr
    vh_csr = sparse.csr_matrix((vh, (rows, columns)))
    pg_csr = sparse.csr_matrix((pg, (rows, columns)))
    pl_csr = sparse.csr_matrix((pl, (rows, columns)))
    pr_csr = sparse.csr_matrix((pr, (rows, columns)))
    # conjugate transpose
    vh_ct = vh_csr.conjugate().transpose()

    # calculate S^{r}\left(E\right)
    sr_csr = vh_csr @ pr_csr

    # calculate L^{\lessgtr}\left(E\right)
    lg_csr = vh_csr @ pg_csr @ vh_ct
    ll_csr = vh_csr @ pl_csr @ vh_ct

    return sr_csr, lg_csr, ll_csr, vh_csr, pg_csr, pl_csr, pr_csr

def p2w_mpi_gpu(
    slicing_obj: object,
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

    # number of energy points and nonzero elements
    ne = pg.shape[0]
    no = pg.shape[1]

    # set number of mkl threads
    mkl.set_num_threads(mkl_threads)

    # load data to the gpu
    pg_gpu = cp.empty_like(pg)
    pl_gpu = cp.empty_like(pl)
    pr_gpu = cp.empty_like(pr)
    vh_gpu = cp.empty_like(vh)
    ij2ji_gpu = cp.empty_like(ij2ji)
    rows_gpu = cp.empty_like(rows)
    columns_gpu = cp.empty_like(columns)
    factors_gpu = cp.empty_like(factors)

    pg_gpu.set(pg)
    pl_gpu.set(pl)
    pr_gpu.set(pr)
    vh_gpu.set(vh)
    ij2ji_gpu.set(ij2ji)
    rows_gpu.set(rows)
    columns_gpu.set(columns)
    factors_gpu.set(factors)

    # todo try to not additionally symmetrize
    # Anti-Hermitian symmetrizing of pl and pg
    pg_gpu = 1j * cp.imag(pg_gpu)
    pg_gpu = (pg_gpu - pg_gpu[:,ij2ji].conjugate()) / 2
    pl_gpu = 1j * cp.imag(pl_gpu)
    pl_gpu = (pl_gpu - pl_gpu[:,ij2ji].conjugate()) / 2
    # pr has to be derived from pl and pg and then has to be symmetrized
    pr_gpu = 1j * cp.imag(pg_gpu - pl_gpu) / 2
    pr_gpu = (pr_gpu + pr_gpu[:,ij2ji]) / 2


    # create empty buffer for screened interaction
    # diagonal blocks
    xr_diag  = np.zeros((ne, slicing_obj.nb_mm, slicing_obj.lb_max_mm, slicing_obj.lb_max_mm), dtype = np.complex128)
    wg_diag  = np.zeros((ne, slicing_obj.nb_mm, slicing_obj.lb_max_mm, slicing_obj.lb_max_mm), dtype = np.complex128)
    wl_diag  = np.zeros((ne, slicing_obj.nb_mm, slicing_obj.lb_max_mm, slicing_obj.lb_max_mm), dtype = np.complex128)
    wr_diag  = np.zeros((ne, slicing_obj.nb_mm, slicing_obj.lb_max_mm, slicing_obj.lb_max_mm), dtype = np.complex128)
    # upper diagonal blocks
    wg_upper = np.zeros((ne, slicing_obj.nb_mm-1, slicing_obj.lb_max_mm, slicing_obj.lb_max_mm), dtype = np.complex128)
    wl_upper = np.zeros((ne, slicing_obj.nb_mm-1, slicing_obj.lb_max_mm, slicing_obj.lb_max_mm), dtype = np.complex128)
    wr_upper = np.zeros((ne, slicing_obj.nb_mm-1, slicing_obj.lb_max_mm, slicing_obj.lb_max_mm), dtype = np.complex128)


    time_loop = -time.perf_counter()
    for i in range(ne):
        
        # calculate helper variables and transform to csr
        sr_csr, lg_csr, ll_csr, vh_csr, pg_csr, pl_csr, pr_csr = w2s_l(vh_gpu, pg_gpu[i,:], pl_gpu[i,:], pr_gpu[i,:], rows_gpu, columns_gpu)


        vh_cp = vh_csr.copy()
        compute_flag, mr_csr = obc_w_gpu.obc_w(
                                pg_csr,
                                pl_csr,
                                pr_csr,
                                vh_cp,
                                sr_csr,
                                lg_csr,
                                ll_csr,
                                slicing_obj)
        sr_csr = sr_csr.get()
        lg_csr = lg_csr.get()
        ll_csr = ll_csr.get()
        vh_cp = vh_cp.get()
        pg_csr = pg_csr.get()
        pl_csr = pl_csr.get()
        pr_csr = pr_csr.get()
        mr_csr = mr_csr.get()
        if compute_flag:
            matrix_inversion_w.rgf(
                slicing_obj.bmax_mm,
                slicing_obj.bmin_mm,
                vh_cp,
                mr_csr,
                lg_csr,
                ll_csr,
                factors[i],
                wg_diag[i],
                wg_upper[i],
                wl_diag[i],
                wl_upper[i],
                wr_diag[i],
                wr_upper[i],
                xr_diag[i]
            )
    time_loop += time.perf_counter()
    print("Time loop: ", time_loop)

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
