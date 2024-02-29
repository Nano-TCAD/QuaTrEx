import cupy as cp
import numpy as np
import numpy.typing as npt
from scipy import sparse
import typing

def rgf_w_opt_standalone_batched_gpu(
    vh_diag: npt.NDArray[np.complex128],
    vh_upper: npt.NDArray[np.complex128],
    vh_lower: npt.NDArray[np.complex128],
    lg_diag: npt.NDArray[np.complex128],
    lg_upper: npt.NDArray[np.complex128],
    lg_lower: npt.NDArray[np.complex128],
    ll_diag: npt.NDArray[np.complex128],
    ll_upper: npt.NDArray[np.complex128],
    ll_lower: npt.NDArray[np.complex128],
    mr_diag: npt.NDArray[np.complex128],
    mr_upper: npt.NDArray[np.complex128],
    mr_lower: npt.NDArray[np.complex128],
    bmax: npt.NDArray[np.int32],
    bmin: npt.NDArray[np.int32],
    wg_diag: npt.NDArray[np.complex128],
    wg_upper: npt.NDArray[np.complex128],
    wl_diag: npt.NDArray[np.complex128],
    wl_upper: npt.NDArray[np.complex128],
    wr_diag: npt.NDArray[np.complex128],
    wr_upper: npt.NDArray[np.complex128],
    xr_diag: npt.NDArray[np.complex128],
    dosw: npt.NDArray[np.complex128],
    nEw: npt.NDArray[np.complex128],
    nPw: npt.NDArray[np.complex128],
    nbc: np.int64,
    solve: bool = True,
    input_stream: cp.cuda.Stream = None,
    output_stream: cp.cuda.Stream = None
) -> typing.Tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128], npt.NDArray[np.complex128],
                  npt.NDArray[np.complex128]]:
    """Calculates the step from the polarization to the screened interaction.
    Beyn open boundary conditions are used by default.
    The outputs (w and x) are inputs which are changed inplace.
    See the start of this file for more informations.

    Args:
        vh_diag (npt.NDArray[np.complex128]): dense matrix of size (#blocks_mm, energy_batchsize, maxblocklength_mm, maxblocklength_mm)
        vh_upper (npt.NDArray[np.complex128]): dense matrix of size (#blocks_mm-1, maxblocklength_mm, maxblocklength_mm)
        vh_lower (npt.NDArray[np.complex128]): dense matrix of size (#blocks_mm-1, maxblocklength_mm, maxblocklength_mm)
        lg_diag (npt.NDArray[np.complex128]): dense matrix of size (#blocks_mm, energy_batchsize, maxblocklength_mm, maxblocklength_mm)
        .
        .
        .

        bmax (npt.NDArray[np.int32]): end idx of the blocks, vector of size number of blocks
        bmin (npt.NDArray[np.int32]): start idx of the blocks, vector of size number of blocks
        wg_diag (npt.NDArray[np.complex128]): dense matrix of size (#blocks_mm, energy_batchsize, maxblocklength_mm, maxblocklength_mm)
        wg_upper (npt.NDArray[np.complex128]): dense matrix of size (#blocks_mm-1, energy_batchsize, maxblocklength_mm, maxblocklength_mm)
        wl_diag (npt.NDArray[np.complex128]): dense matrix of size (#blocks_mm, energy_batchsize, maxblocklength_mm, maxblocklength_mm)
        wl_upper (npt.NDArray[np.complex128]): dense matrix of size (#blocks_mm-1, energy_batchsize, maxblocklength_mm, maxblocklength_mm)
        wr_diag (npt.NDArray[np.complex128]): dense matrix of size (#blocks_mm, energy_batchsize, maxblocklength_mm, maxblocklength_mm)
        wr_upper (npt.NDArray[np.complex128]): dense matrix of size (#blocks_mm-1, energy_batchsize, maxblocklength_mm, maxblocklength_mm)
        xr_diag (npt.NDArray[np.complex128]): dense matrix of size (#blocks_mm, energy_batchsize, maxblocklength_mm, maxblocklength_mm)
        dosw (npt.NDArray[np.complex128]): density of states, vector of size number of blocks
        nbc (np.int64): how block size changes after matrix multiplication
        ie (np.int32): energy index (not used)
        factor (np.float64, optional): factor to multiply the result with. Defaults to 1.0.
        ref_flag (bool, optional): If reference solution to rgf made by np.linalg.inv should be returned
        sancho_flag (bool, optional): If sancho or beyn should be used. Defaults to False.
    
    Returns:
        typing.Tuple[npt.NDArray[np.complex128], xr from inv
                  npt.NDArray[np.complex128],    wg from inv
                  npt.NDArray[np.complex128],    wl from inv
                  npt.NDArray[np.complex128]     wr from inv
                ] warning all dense arrays, only returned if ref_flag is True
    """
    # number of blocks
    nb = bmin.size
    energy_batchsize = lg_diag.shape[1]

    # block sizes after matrix multiplication
    bmax_mm = bmax[nbc - 1:nb:nbc]
    bmin_mm = bmin[0:nb:nbc]
    # number of blocks after matrix multiplication
    nb_mm = bmin_mm.size
    # vector of block lengths after matrix multiplication
    lb_vec_mm = bmax_mm - bmin_mm + 1
    # max block size after matrix multiplication
    lb_max_mm = np.max(lb_vec_mm)

    vh_diag_gpu = cp.empty((2, energy_batchsize, lb_max_mm, lb_max_mm), dtype=vh_diag.dtype)

    vh_upper_gpu = cp.empty((2, lb_max_mm, lb_max_mm), dtype=vh_upper.dtype)
    vh_lower_gpu = cp.empty((2, lb_max_mm, lb_max_mm), dtype=vh_lower.dtype)

    lg_diag_gpu = cp.empty((2, energy_batchsize, lb_max_mm, lb_max_mm), dtype=lg_diag.dtype)
    lg_upper_gpu = cp.empty((2, energy_batchsize, lb_max_mm, lb_max_mm), dtype=lg_upper.dtype)
    lg_lower_gpu = cp.empty((2, energy_batchsize, lb_max_mm, lb_max_mm), dtype=lg_lower.dtype)
    ll_diag_gpu = cp.empty((2, energy_batchsize, lb_max_mm, lb_max_mm), dtype=ll_diag.dtype)
    ll_upper_gpu = cp.empty((2, energy_batchsize, lb_max_mm, lb_max_mm), dtype=ll_upper.dtype)
    ll_lower_gpu = cp.empty((2, energy_batchsize, lb_max_mm, lb_max_mm), dtype=ll_lower.dtype)
    mr_diag_gpu = cp.empty((2, energy_batchsize, lb_max_mm, lb_max_mm), dtype=mr_diag.dtype)
    mr_upper_gpu = cp.empty((2, energy_batchsize, lb_max_mm, lb_max_mm), dtype=mr_upper.dtype)
    mr_lower_gpu = cp.empty((2, energy_batchsize, lb_max_mm, lb_max_mm), dtype=mr_lower.dtype)

    computation_stream = cp.cuda.Stream.null
    input_stream = input_stream or cp.cuda.Stream(non_blocking=True)
    output_stream = output_stream or cp.cuda.Stream(non_blocking=True)
    input_events = [cp.cuda.Event() for _ in range(2)]
    computation_event = cp.cuda.Event()

    # not true inverse, but build up inverses from either corner
    xr_diag_rgf = cp.empty((nb_mm, energy_batchsize, lb_max_mm, lb_max_mm), dtype=xr_diag.dtype)
    wg_diag_rgf = cp.empty((nb_mm, energy_batchsize, lb_max_mm, lb_max_mm), dtype=xr_diag.dtype)
    wl_diag_rgf = cp.empty((nb_mm, energy_batchsize, lb_max_mm, lb_max_mm), dtype=xr_diag.dtype)
    wr_diag_rgf = cp.empty((nb_mm, energy_batchsize, lb_max_mm, lb_max_mm), dtype=xr_diag.dtype)

    #output observables
    dosw_gpu = cp.empty((energy_batchsize, nb_mm), dtype=dosw.dtype)
    nEw_gpu = cp.empty((energy_batchsize, nb_mm), dtype=nEw.dtype)
    nPw_gpu = cp.empty((energy_batchsize, nb_mm), dtype=nPw.dtype)

    # "True" quantities of selected inverse
    xr_diag_gpu = cp.empty((2, energy_batchsize, lb_max_mm, lb_max_mm), dtype=xr_diag.dtype)
    xr_upper_gpu = cp.empty((2, energy_batchsize, lb_max_mm, lb_max_mm), dtype=xr_diag.dtype)
    xr_lower_gpu = cp.empty((2, energy_batchsize, lb_max_mm, lb_max_mm), dtype=xr_diag.dtype)
    wg_diag_gpu = cp.empty((2, energy_batchsize, lb_max_mm, lb_max_mm), dtype=xr_diag.dtype)
    wg_upper_gpu = cp.empty((2, energy_batchsize, lb_max_mm, lb_max_mm), dtype=xr_diag.dtype)
    wl_diag_gpu = cp.empty((2, energy_batchsize, lb_max_mm, lb_max_mm), dtype=xr_diag.dtype)
    wl_upper_gpu = cp.empty((2, energy_batchsize, lb_max_mm, lb_max_mm), dtype=xr_diag.dtype)
    wr_diag_gpu = cp.empty((2, energy_batchsize, lb_max_mm, lb_max_mm), dtype=xr_diag.dtype)
    wr_upper_gpu = cp.empty((2, energy_batchsize, lb_max_mm, lb_max_mm), dtype=xr_diag.dtype)

    desc = """
    Meaning of variables:
    _lb: last block
    _c:  current or center
    _p:  previous
    _n:  next
    _r:  right
    _l:  left
    _d:  down
    _u:  up
    """

    # Backward pass

    # First iteration
    IB = nb_mm - 1
    nIB = IB - 1
    idx = IB % 2
    ndix = nIB % 2
    NN = lb_vec_mm[IB]

    # input quantities for this iteration
    md = mr_diag_gpu[idx]
    lgd = lg_diag_gpu[idx]
    lld = ll_diag_gpu[idx]
    vhd = vh_diag_gpu[idx]

    # output quantities for this iteration
    xr_d_rgf = xr_diag_rgf[IB]
    wg_d_rgf = wg_diag_rgf[IB]
    wl_d_rgf = wl_diag_rgf[IB]
    wr_d_rgf = wr_diag_rgf[IB]

    # Need to upload input quantities of first iteration synchronously
    md.set(mr_diag[IB])
    lgd.set(lg_diag[IB])
    lld.set(ll_diag[IB])
    vhd.set(vh_diag[IB])

    # Asynchronous upload of input quantities of next iteration
    if nIB >=0:
        with input_stream:
            nmd = mr_diag_gpu[ndix]
            nmu = mr_upper_gpu[ndix]
            nml = mr_lower_gpu[ndix]
            nlgd = lg_diag_gpu[ndix]
            nlgu = lg_upper_gpu[ndix]
            nlgd = lg_lower_gpu[ndix]
            nlld = ll_diag_gpu[ndix]
            nllu = ll_upper_gpu[ndix]
            nlll = ll_lower_gpu[ndix]
            nvhd = vh_diag_gpu[ndix]
            nvhu = vh_upper_gpu[ndix]
            nvhl = vh_lower_gpu[ndix]

            nmd.set(mr_diag[nIB])
            nmu.set(mr_upper[nIB])
            nml.set(mr_lower[nIB])
            nlgd.set(lg_diag[nIB])
            nlgu.set(lg_upper[nIB])
            nlgd.set(lg_lower[nIB])
            nlld.set(ll_diag[nIB])
            nllu.set(ll_upper[nIB])
            nlll.set(ll_lower[nIB])
            nvhd.set(vh_diag[nIB])
            nvhu.set(vh_upper[nIB])
            nvhl.set(vh_lower[nIB])
            input_events[idx].record(input_stream)
    if solve:
        gpu_identity = cp.identity(NN, dtype=md.dtype)
        gpu_identity_batch = cp.repeat(gpu_identity[cp.newaxis, :, :], energy_batchsize, axis=0)
        xr_d_rgf[:, 0:NN, 0:NN] = cp.linalg.solve(md, gpu_identity_batch)
        computation_stream.synchronize()
    else:
        xr_d_rgf[:, 0:NN, 0:NN] = cp.linalg.inv(md[:, 0:NN, 0:NN])

    xr_d_rgf_h = cp.conjugate(xr_d_rgf[:, 0:NN, 0:NN].transpose((0, 2, 1)))
    cp.matmul(xr_d_rgf[:, 0:NN, 0:NN], vhd[:, 0:NN, 0:NN], out=wr_d_rgf[:, 0:NN, 0:NN])
    cp.matmul(xr_d_rgf[:, 0:NN, 0:NN] @ lld[:, 0:NN, 0:NN], xr_d_rgf_h[:, 0:NN, 0:NN], out=wl_d_rgf[:, 0:NN, 0:NN])
    cp.matmul(xr_d_rgf_h[:, 0:NN, 0:NN] @ lgd[:, 0:NN, 0:NN], xr_d_rgf[:, 0:NN, 0:NN], out=wg_d_rgf[:, 0:NN, 0:NN])

    # Rest of iterations
    for IB in range(nb_mm - 2, -1, -1):
        nIB = IB - 1
        idx = IB % 2
        ndix = nIB % 2
        NI = lb_vec_mm[IB]
        NP = lb_vec_mm[IB + 1]

        # input quantities for this iteration
        md = mr_diag_gpu[idx]
        mu = mr_upper_gpu[idx]
        ml = mr_lower_gpu[idx]
        lgd = lg_diag_gpu[idx]
        lgu = lg_upper_gpu[idx]
        lld = ll_diag_gpu[idx]
        llu = ll_upper_gpu[idx]
        ll = ll_lower_gpu[idx]
        vhd = vh_diag_gpu[idx]
        vhu = vh_upper_gpu[idx]
        vhl = vh_lower_gpu[idx]

        # output quantities for this iteration
        xr_d_rgf = xr_diag_rgf[IB]
        pxr_d_rgf = xr_diag_rgf[IB + 1]
        wg_d_rgf = wg_diag_rgf[IB]
        pwg_d_rgf = wg_diag_rgf[IB + 1]
        wl_d_rgf = wl_diag_rgf[IB]
        pwl_d_rgf = wl_diag_rgf[IB + 1]
        wr_d_rgf = wr_diag_rgf[IB]
        pwr_d_rgf = wr_diag_rgf[IB + 1]

        # Asynchronous upload of input quantities of next iteration
        if nIB >=0:
            with input_stream:
                nmd = mr_diag_gpu[ndix]
                nmu = mr_upper_gpu[ndix]
                nml = mr_lower_gpu[ndix]
                nlgd = lg_diag_gpu[ndix]
                nlgu = lg_upper_gpu[ndix]
                nlgd = lg_lower_gpu[ndix]
                nlld = ll_diag_gpu[ndix]
                nllu = ll_upper_gpu[ndix]
                nlll = ll_lower_gpu[ndix]
                nvhd = vh_diag_gpu[ndix]
                nvhu = vh_upper_gpu[ndix]
                nvhl = vh_lower_gpu[ndix]

                nmd.set(mr_diag[nIB])





