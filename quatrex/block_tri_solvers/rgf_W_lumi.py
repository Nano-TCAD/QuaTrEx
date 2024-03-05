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
        pidx = (IB + 1) % 2
        NI = lb_vec_mm[IB]
        NP = lb_vec_mm[IB + 1]

        # input quantities for this iteration
        md = mr_diag_gpu[idx]
        mu = mr_upper_gpu[idx]
        ml = mr_lower_gpu[idx]
        lgd = lg_diag_gpu[idx]
        lgl = lg_lower_gpu[idx]
        lld = ll_diag_gpu[idx]
        lll = ll_lower_gpu[idx]
        vhd = vh_diag_gpu[idx]
        vhl = vh_lower_gpu[idx]

        # output quantities for this iteration
        xr_d_rgf = xr_diag_rgf[IB]
        pxr_d_rgf = xr_diag_rgf[IB + 1]
        wg_d_rgf = wg_diag_rgf[IB]
        pwg_d_rgf = wg_diag_rgf[IB + 1]
        wl_d_rgf = wl_diag_rgf[IB]
        pwl_d_rgf = wl_diag_rgf[IB + 1]
        wr_d_rgf = wr_diag_rgf[IB]

        if IB == 0:
            xr_d_rgf = xr_diag_gpu[0]
            #wr_d_rgf = wr_diag_gpu[0] # Comment this has different formula for first element.
            wg_d_rgf = wg_diag_gpu[0]
            wl_d_rgf = wl_diag_gpu[0]

        computation_stream.synchronize()
        # Asynchronous upload of input quantities of next iteration
        with input_stream:
            if nIB >=0:
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

            else:
                # To-Do: think what to extra upload for last block 
                nlgu = lg_upper_gpu[idx]
                nllu = ll_upper_gpu[idx]
                nvhu = vh_upper_gpu[idx]

                nlgu.set(lg_upper[IB])
                nllu.set(ll_upper[IB])
                nvhu.set(vh_upper[IB])

            input_events[idx].record(input_stream)
        
        computation_stream.wait_event(event = input_events[pidx])
        mu_pxr = mu[:, 0:NI, 0:NP] @ pxr_d_rgf[:, 0:NP, 0:NP]
        muh = mu.conjugate().transpose((0, 2, 1))

        ag = mu_pxr @ lgl[:, 0:NP, 0:NI]
        al = mu_pxr @ lll[:, 0:NP, 0:NI]
        ag_diff = ag - ag.conjugate().transpose((0, 2, 1))
        al_diff = al - al.conjugate().transpose((0, 2, 1))

        inv_arg = md[:, 0:NI, 0:NI] - mu_pxr @ ml[:, 0:NP, 0:NI]
        if solve:
            gpu_identity = cp.identity(NN, dtype=md.dtype)
            gpu_identity_batch = cp.repeat(gpu_identity[cp.newaxis, :, :], energy_batchsize, axis=0)
            xr_d_rgf[:, 0:NI, 0:NI] = cp.linalg.solve(inv_arg, gpu_identity_batch)
            computation_stream.synchronize()
        else:
            xr_d_rgf[:, 0:NI, 0:NI] = cp.linalg.inv(inv_arg)

        xr_d_rgf_h = cp.conjugate(xr_d_rgf[:, 0:NI, 0:NI].transpose((0, 2, 1)))
        mu_pwg_muh = mu @ pwg_d_rgf[:, 0:NP, 0:NP] @ muh - ag_diff
        mu_pwl_muh = mu @ pwl_d_rgf[:, 0:NP, 0:NP] @ muh - al_diff
        mu_pxr_vhl = mu_pxr @ cp.repeat(vhl[cp.newaxis, 0:NP, 0:NI], energy_batchsize, axis=0)

        cp.multiply(xr_d_rgf[:, 0:NI, 0:NI] @ (lgd[:, 0:NI, 0:NI] + mu_pwg_muh), xr_d_rgf_h[:, 0:NI, 0:NI], out=wg_d_rgf)
        cp.multiply(xr_d_rgf[:, 0:NI, 0:NI] @ (lld[:, 0:NI, 0:NI] + mu_pwl_muh), xr_d_rgf_h[:, 0:NI, 0:NI], out=wl_d_rgf)
        cp.multiply(xr_d_rgf[:, 0:NI, 0:NI], vhd[:, 0:NI, 0:NI] - mu_pxr_vhl, out=wr_d_rgf)

        if IB == 0:
            computation_stream.synchronize()
            computation_event.record(computation_stream)
            # To-Do first round of download to the host.
            with output_stream:
                output_stream.wait_event(event=computation_event)
                xr_diag_gpu[0].get(out=xr_diag[0])
                wr_diag_gpu[0].get(out=wr_diag[0])
                wg_diag_gpu[0].get(out=wg_diag[0])
                wl_diag_gpu[0].get(out=wl_diag[0])

            dosw_gpu[:, 0] = 1j * cp.trace(wr_d_rgf[:, 0:NI, 0:NI] - wr_d_rgf[:, 0:NI, 0:NI].conjugate().transpose((0,2,1)), axis1=1, axis2=2)
            nEw_gpu[:, 0] = -1j * cp.trace(wl_d_rgf[:, 0:NI, 0:NI], axis1=1, axis2=2)
            nPw_gpu[:, 0] = 1j * cp.trace(wg_d_rgf, axis1=1, axis2=2)


    # Forward pass
            
    # First iteration
    IB = 0
    nIB = IB + 1
    idx = IB % 2
    ndix = nIB % 2

    mu = mr_upper_gpu[idx]
    ml = mr_lower_gpu[idx]
    vhd = vh_diag_gpu[idx]
    vhl = vh_lower_gpu[idx]
    vhu = vh_upper_gpu[idx]
    lgu = lg_upper_gpu[idx]
    llu = ll_upper_gpu[idx]

    XR = xr_diag_gpu[IB]
    XRu = xr_upper_gpu[idx]
    XRl = xr_lower_gpu[idx]
    WR = wr_diag_gpu[IB]
    WRu = wr_upper_gpu[idx]
    WG = wg_diag_gpu[IB]
    WGu = wg_upper_gpu[idx]
    WL = wl_diag_gpu[IB]
    WLu = wl_upper_gpu[idx]

    # NOTE: These were written directly to output in the last iteration of the backward pass
    xr_d_rgf = xr_diag_gpu[IB]
    wr_d_rgf = wr_diag_gpu[IB] # Comment this has different formula for first element and was thus not written yet.
    wg_d_rgf = wg_diag_gpu[IB]
    wl_d_rgf = wl_diag_gpu[IB]

    pxr_d_rgf = xr_diag_rgf[nIB]
    pwg_d_rgf = wg_diag_rgf[nIB]
    pwl_d_rgf = wl_diag_rgf[nIB]

    computation_stream.synchronize()
    with input_stream:
        if nIB < nb_mm:
            npmu = mr_upper_gpu[ndix]
            npml = mr_lower_gpu[ndix]
            nplgl = lg_lower_gpu[ndix]
            nplll = ll_lower_gpu[ndix]

            npmu.set(mr_upper[IB])
            npml.set(mr_lower[IB])
            nplgl.set(lg_lower[IB])
            nplll.set(ll_lower[IB])

            if nIB < nb_mm - 1:
                nmu = mr_diag_gpu[ndix] # use diag here because upper is already used.
                nml = ll_diag_gpu[ndix] # use diag here because lower is already used.
                npvhu = vh_upper_gpu[ndix]
                nvhl = vh_lower_gpu[ndix]
                nvhd = vh_diag_gpu[ndix]
                nlgu = lg_upper_gpu[ndix]
                nllu = ll_upper_gpu[ndix]

                nmu.set(mr_upper[nIB])
                nml.set(mr_lower[nIB])
                npvhu.set(vh_upper[IB])
                nvhl.set(vh_lower[nIB])
                nvhd.set(vh_diag[nIB])
                nlgu.set(lg_upper[nIB])
                nllu.set(ll_upper[nIB])


        input_events[ndix].record(stream = input_stream)

    computation_stream.wait_event(event = input_events[idx])

    pxr_ml = pxr_d_rgf[:, 0:NP, 0:NP] @ ml[:, 0:NP, 0:NI]
    pxr_ml_h = pxr_ml.conjugate().transpose((0, 2, 1))

    cp.multiply(-xr_d_rgf[:, 0:NI, 0:NI] @ mu[:, 0:NI, 0:NP], pxr_d_rgf[:, 0:NP, 0:NP], out=XRu[:, 0:NI, 0:NP])
    cp.multiply(-pxr_d_rgf[:, 0:NP, 0:NP] @ ml[:, 0:NP, 0:NI], xr_d_rgf[:, 0:NI, 0:NI], out=XRl[:, 0:NP, 0:NI])

    xrd_vhd = xr_d_rgf[:, 0:NI, 0:NI] @ vhd[:, 0:NI, 0:NI]
    xru_vhl = XRu[:, 0:NI, 0:NP] @ cp.repeat(vhl[cp.newaxis, 0:NP, 0:NI], energy_batchsize, axis=0)

    cp.add(xrd_vhd, xru_vhl, out=WR[:, 0:NI, 0:NI])

    wrd_mrlt = WR[:, 0:NI, 0:NI] @ ml[:, 0:NP, 0:NI].transpose((0,2,1))
    cp.multiply(cp.repeat(vhu[cp.newaxis, 0:NI, 0:NP], energy_batchsize, axis=0) - wrd_mrlt[:, 0:NI, 0:NP], pxr_d_rgf[:, 0:NP, 0:NP].transpose((0,2,1)), out=WRu[:, 0:NI, 0:NP])

    pxrh = pxr_d_rgf[:, 0:NP, 0:NP].conjugate().transpose((0, 2, 1))
    xrd_lgu_pxrh = xr_d_rgf[:, 0:NI, 0:NI] @ lgu[:, 0:NI, 0:NP] @ pxrh[:, 0:NP, 0:NP]
    xrd_llu_pxrh = xr_d_rgf[:, 0:NI, 0:NI] @ llu[:, 0:NI, 0:NP] @ pxrh[:, 0:NP, 0:NP]

    xrd_mru_pwgd = xr_d_rgf[:, 0:NI, 0:NI] @ mu[:, 0:NI, 0:NP] @ pwg_d_rgf[:, 0:NP, 0:NP]
    xrd_mru_pwld = xr_d_rgf[:, 0:NI, 0:NI] @ mu[:, 0:NI, 0:NP] @ pwl_d_rgf[:, 0:NP, 0:NP]

    wgd_pxr_ml_h = wg_d_rgf[:, 0:NI, 0:NI] @ pxr_ml_h[:, 0:NI, 0:NP]
    wld_pxr_ml_h = wl_d_rgf[:, 0:NI, 0:NI] @ pxr_ml_h[:, 0:NI, 0:NP]

    cp.add(xrd_lgu_pxrh - xrd_mru_pwgd,  wgd_pxr_ml_h,out=WGu[:, 0:NI, 0:NP])
    cp.add(xrd_llu_pxrh - xrd_mru_pwld,  wld_pxr_ml_h,out=WLu[:, 0:NI, 0:NP])

    computation_stream.synchronize()
    computation_event.record(stream = computation_stream)
    with output_stream:
        output_stream.wait_event(event = computation_event)
        WRu.get(out=wr_upper[0])
        WLu.get(out=wl_upper[0])
        WGu.get(out=wg_upper[0])

    # Rest iterations
    for IB in range(1, nb_mm):

        pIB = IB - 1
        nIB = IB + 1
        idx = IB % 2
        pidx = pIB % 2
        ndix = nIB % 2
        NI = lb_vec_mm[IB]
        NP = lb_vec_mm[IB - 1]

        pmu = mr_upper_gpu[pidx]
        pml = mr_lower_gpu[pidx]
        plgl = lg_lower_gpu[pidx]
        plll = ll_lower_gpu[pidx]

        XR = xr_diag_gpu[idx]
        pXR = xr_diag_gpu[pidx]
        XRu = xr_upper_gpu[idx]
        XRl = xr_lower_gpu[idx]
        pXRl = xr_lower_gpu[pidx]
        WR = wr_diag_gpu[idx]
        WRu = wr_upper_gpu[idx]
        WG = wg_diag_gpu[idx]
        pWG = wg_diag_gpu[pidx]
        WGu = wg_upper_gpu[idx]
        WL = wl_diag_gpu[idx]
        pWL = wl_diag_gpu[pidx]
        WLu = wl_upper_gpu[idx]

        # output quantities for this iteration
        xr_d_rgf = xr_diag_rgf[IB]
        wg_d_rgf = wg_diag_rgf[IB]
        wl_d_rgf = wl_diag_rgf[IB]
        wr_d_rgf = wr_diag_rgf[IB]

        computation_stream.synchronize()
        with input_stream:
            npmu = mr_upper_gpu[idx]
            npml = mr_lower_gpu[idx]
            nplgl = lg_lower_gpu[idx]
            nplll = ll_lower_gpu[idx]

            npmu.set(mr_upper[IB])
            npml.set(mr_lower[IB])
            nplgl.set(lg_lower[IB])
            nplll.set(ll_lower[IB])

            if nIB < nb_mm - 1:
                nmu = mr_diag_gpu[ndix] # use diag here because upper is already used.
                nml = ll_diag_gpu[ndix] # use diag here because lower is already used.
                npvhu = vh_upper_gpu[ndix]
                nvhl = vh_lower_gpu[ndix]
                nvhd = vh_diag_gpu[ndix]
                nlgu = lg_upper_gpu[ndix]
                nllu = ll_upper_gpu[ndix]

                nmu.set(mr_upper[nIB])
                nml.set(mr_lower[nIB])
                npvhu.set(vh_upper[IB])
                nvhl.set(vh_lower[nIB])
                nvhd.set(vh_diag[nIB])
                nlgu.set(lg_upper[nIB])
                nllu.set(ll_upper[nIB])
            #uploads
            input_events[ndix].record(stream = input_stream)

        computation_stream.wait_event(event = input_events[idx])

        #calculations
        xrd_pml = xr_d_rgf[:, 0:NI, 0:NI] @ pml[:, 0:NI, 0:NP] # called xr_mr in rgf_W_GPU.py
        xrd_pml_h = xrd_pml.conjugate().transpose((0, 2, 1))
        xrd_pml_pXRd = xrd_pml @ pXR[:, 0:NP, 0:NP]
        xrd_pml_pXRd_pmu = xrd_pml_pXRd @ pmu[:, 0:NP, 0:NI]
        xrd_pml_pXRd_h = xrd_pml_pXRd.conjugate().transpose((0, 2, 1))


        # XR_E_kk = xR_E_kk + (xR_E_kk * M_E_kk-1 * XR_E_k-1k-1 * M_E_k-1k * xR_E_kk)
        cp.add(xr_d_rgf[:, 0:NI, 0:NI],  xrd_pml_pXRd_pmu[:, 0:NI, 0:NI] @ xr_d_rgf[:, 0:NI, 0:NI], out=XR[:, 0:NI, 0:NI])

        # A^{\lessgtr} = xR_E_kk * L^{\lessgtr}_E_kk-1 * XR_E_k-1k-1.H * (xR_E_kk * M_E_kk-1).H
        ag = xr_d_rgf[:, 0:NI, 0:NI] @ plgl[:, 0:NI, 0:NP] @ xrd_pml_pXRd_h[:, 0:NP, 0:NI]
        al = xr_d_rgf[:, 0:NI, 0:NI] @ plll[:, 0:NI, 0:NP] @ xrd_pml_pXRd_h[:, 0:NP, 0:NI]
        ag_diff = ag - ag.conjugate().transpose((0, 2, 1))
        al_diff = al - al.conjugate().transpose((0, 2, 1))

        # B^{\lessgtr} = xR_E_kk * M_E_kk-1 * XR_E_k-1k-1 * M_E_k-1k * w^{\lessgtr}_E_kk
        bg = xrd_pml_pXRd_pmu[:, 0:NI, 0:NI] @ wg_d_rgf[:, 0:NI, 0:NI]
        bl = xrd_pml_pXRd_pmu[:, 0:NI, 0:NI] @ wl_d_rgf[:, 0:NI, 0:NI]
        bf_diff = bg - bg.conjugate().transpose((0, 2, 1))
        bl_diff = bl - bl.conjugate().transpose((0, 2, 1))

        #W^{\lessgtr}_E_kk = w^{\lessgtr}_E_kk + xR_E_kk*M_E_kk-1*W^{\lessgtr}_E_k-1k-1*(xR_E_kk*M_E_kk-1).H - (A^{\lessgtr}-A^{\lessgtr}.H) + (B^{\lessgtr}-B^{\lessgtr}.H)
        xrd_pml_pWG_xrd_pml_h = xrd_pml[:, 0:NI, 0:NP] @ pWG[:, 0:NP, 0:NP] @ xrd_pml_h[:, 0:NP, 0:NI]
        xrd_pml_pWL_xrd_pml_h = xrd_pml[:, 0:NI, 0:NP] @ pWL[:, 0:NP, 0:NP] @ xrd_pml_h[:, 0:NP, 0:NI]
        cp.add(wg_d_rgf[:, 0:NI, 0:NI], xrd_pml_pWG_xrd_pml_h - ag_diff + bf_diff, out=WG[:, 0:NI, 0:NI])
        cp.add(wl_d_rgf[:, 0:NI, 0:NI], xrd_pml_pWL_xrd_pml_h - al_diff + bl_diff, out=WL[:, 0:NI, 0:NI])

        if IB < nb_mm - 1:
            NP = lb_vec_mm[IB + 1]

            mu = mr_diag_gpu[idx] # use diag here because lower is already used.
            ml = ll_diag_gpu[idx] # use diag here because lower is already used.
            llu = ll_upper_gpu[idx]
            lgu = lg_upper_gpu[idx]
            vhd = vh_diag_gpu[idx]
            pvhu = vh_upper_gpu[idx]
            vhl = vh_lower_gpu[idx]

            nxr_d_rgf = xr_diag_gpu[nIB]
            nwg_d_rgf = wg_diag_gpu[nIB]
            nwl_d_rgf = wl_diag_gpu[nIB]

            nxr_d_rgf_h = nxr_d_rgf[:, 0:NP, 0:NP].conjugate().transpose((0, 2, 1))

            nxrd_ml = nxr_d_rgf[:, 0:NI, 0:NI] @ ml[:, 0:NI, 0:NP]
            nxrd_ml_h = nxrd_ml.conjugate().transpose((0, 2, 1))

            cp.multiply(-XR[:, 0:NI, 0:NI] @ mu[:, 0:NI, 0:NP], nxr_d_rgf[:, 0:NP, 0:NP], out=XRu[:, 0:NI, 0:NP])
            cp.multiply(-nxr_d_rgf[:, 0:NP, 0:NP] @ ml[:, 0:NP, 0:NI], XR[:, 0:NI, 0:NI], out=XRl[:, 0:NP, 0:NI])

            pXRl_pvhu = pXRl[:, 0:NI, 0:NP] @ cp.repeat(pvhu[cp.newaxis, 0:NP, 0:NI], energy_batchsize, axis=0)
            XR_vhd = XR[:, 0:NI, 0:NI] @ vhd[:, 0:NI, 0:NI]
            XR_u_vhl = XRu[:, 0:NI, 0:NP] @ cp.repeat(vhl[cp.newaxis, 0:NP, 0:NI], energy_batchsize, axis=0)

            cp.add(pXRl_pvhu, XR_vhd + XR_u_vhl, out=WR[:, 0:NI, 0:NI])

            # WR_E_kk+1 = (V_k+1k.T - WR_E_kk*M_E_k+1k.T) * xR_E_k+1k+1.T
            vhl_t_nxr_d_rgf_t = cp.repeat(vhl[cp.newaxis, 0:NP, 0:NI], energy_batchsize, axis=0).transpose((0, 2, 1)) @ nxr_d_rgf[:, 0:NP, 0:NP].transpose((0, 2, 1))
            WR_nxrd_t = WR[:, 0:NI, 0:NI] @ nxr_d_rgf[:, 0:NP, 0:NI].transpose((0, 2, 1))
            cp.subtract(vhl_t_nxr_d_rgf_t, WR_nxrd_t, out=WRu[:, 0:NI, 0:NP])

            # W^{\lessgtr}_E_kk+1 = XR_E_kk*(L^{\lessgtr}_E_kk+1*xR_E_k+1k+1.H - M_E_kk+1*w^{\lessgtr}_E_k+1k+1) - W^{\lessgtr}_E_kk*M_E_k+1k.H*xxR_E_k+1k+1.H
            inner_wgupper = lgu[:, 0:NI, 0:NP] @ nxr_d_rgf_h - mu[:, 0:NI, 0:NP] @ nwg_d_rgf[:, 0:NP, 0:NP]
            inner_wlupper = llu[:, 0:NI, 0:NP] @ nxr_d_rgf_h - mu[:, 0:NI, 0:NP] @ nwl_d_rgf[:, 0:NP, 0:NP]
            cp.subtract(XR[:, 0:NI, 0:NI] @ inner_wgupper, WG[:, 0:NI, 0:NI] @ nxrd_ml_h, out=WGu[:, 0:NI, 0:NP])
            cp.subtract(XR[:, 0:NI, 0:NI] @ inner_wlupper, WL[:, 0:NI, 0:NI] @ nxrd_ml_h, out=WLu[:, 0:NI, 0:NP])

            computation_stream.synchronize()
            computation_event.record(stream = computation_stream)
            with output_stream:
                output_stream.wait_event(event = computation_event)
                XR.get(out=xr_diag[IB])
                WR.get(out=wr_diag[IB])
                WRu.get(out=wr_upper[IB])
                WL.get(out=wl_diag[IB])
                WLu.get(out=wl_upper[IB])
                WG.get(out=wg_diag[IB])
                WGu.get(out=wg_upper[IB])

        
        else:
            cp.add(pXRl_pvhu, XR_vhd, out=WR[:, 0:NI, 0:NI])
            computation_stream.synchronize()
            computation_event.record(stream = computation_stream)
            with output_stream:
                output_stream.wait_event(event = computation_event)
                XR.get(out=xr_diag[IB])
                WR.get(out=wr_diag[IB])
                WG.get(out=wg_diag[IB])
                WL.get(out=wl_diag[IB])
        
        dosw_gpu[:, IB] = 1j * cp.trace(WR[:, 0:NI, 0:NI] - WR[:, 0:NI, 0:NI].conjugate().transpose((0,2,1)), axis1=1, axis2=2)
        nEw_gpu[:, IB] = -1j * cp.trace(WL[:, 0:NI, 0:NI], axis1=1, axis2=2)
        nPw_gpu[:, IB] = 1j * cp.trace(WG[:, 0:NI, 0:NI], axis1=1, axis2=2)

    dosw_gpu.get(out=dosw)
    nEw_gpu.get(out=nEw)
    nPw_gpu.get(out=nPw)    
    input_stream.synchronize()
    output_stream.synchronize()
    computation_stream.synchronize()

            
    








