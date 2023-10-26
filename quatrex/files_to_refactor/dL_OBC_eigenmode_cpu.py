# Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.

import numpy as np

def get_dl_obc_alt(
        xr_d: np.ndarray,
        lg_d: np.ndarray,
        lg_o: np.ndarray,
        ll_d: np.ndarray,
        ll_o: np.ndarray,
        mr_x: np.ndarray,
        blk: str
) -> (np.ndarray, np.ndarray):
    """
    Calculates open boundary corrections for lg and ll.
    Assumes that input blocks are dense.

    Args:
        xr_d (np.ndarray):
        ll_d (np.ndarray):
        ll_o (np.ndarray):
        lg_d (np.ndarray):
        lg_o (np.ndarray):
        mr_x (np.ndarray):
        blk (str): either "R" or "L" depending on which side to correct

    Raises:
        ValueError: if blk is not "R" or "L"

    Returns:
        typing.Tuple[np.ndarray, np.ndarray]: greater and lesser correction
    """
    # Number of iterations for the refinement
    ref_iteration = 1

    # length of block
    lb = mr_x.shape[0]

    # non zero indexes of mr_x
    #rows, cols = mr_x.nonzero()

    # non zero indexes of mr_x
    mr_x_max = np.max(np.abs(mr_x))
    rows, cols = np.where(np.abs(mr_x) > mr_x_max / 1e8)

    if (not rows.size):
        return np.nan, np.nan

    # conjugate transpose of mr/xr
    mr_x_ct = mr_x.conjugate().T
    xr_d_ct = xr_d.conjugate().T

    mrxr_xd = mr_x @ xr_d
    ag = mrxr_xd @ lg_o
    al = mrxr_xd @ ll_o

    # only difference between ax and ax^H is needed
    ag_diff = ag - ag.conjugate().T
    al_diff = al - al.conjugate().T

    fg = xr_d @ (lg_d - ag_diff) @ xr_d_ct
    fl = xr_d @ (ll_d - al_diff) @ xr_d_ct

    # case for the left/right (start/end) block
    # differentiates between which block to look at
    if blk == "L":
        #idx_max = np.max([np.max(rows), lb - np.min(cols)])
        idx_max = np.max([np.max(rows) + 1, lb - np.min(cols)])
        ip = lb - idx_max
        sl_x = slice(ip, lb)
        #sl_y = slice(0, idx_max + 1)
        sl_y = slice(0, idx_max)
    elif blk == "R":
        #idx_max = np.max([np.max(cols), lb - np.min(rows)])
        idx_max = np.max([np.max(cols) + 1, lb - np.min(rows)])
        ip = lb - idx_max
        #sl_x = slice(0, idx_max + 1)
        sl_x = slice(0, idx_max)
        sl_y = slice(ip, lb)
    else:
        raise ValueError("Argument error, type input not possible")

    ar = xr_d[sl_x, sl_y] @ mr_x[sl_y, sl_x]
    # add imaginary part to stabilize
    #ar = ar + np.identity(ar.shape[0])*1j*1e-4

    # eigen values and eigen vectors
    eival, eivec = np.linalg.eig(ar)

    # conjugate/transpose/abs square
    eivec_ct = eivec.conjugate().T
    ieivec = np.linalg.inv(eivec)
    ieivec_ct = ieivec.conjugate().T
    eival_sq = np.diag(eival) @ np.diag(eival).conjugate()

    # greater component
    yg_d = np.divide(ieivec @ fg[sl_x, sl_x] @ ieivec_ct, 1 - eival_sq)
    wg_d = eivec @ yg_d @ eivec_ct
    xrmr_dx_s = xr_d[sl_x, :] @ mr_x[:, sl_x]
    mrxr_ct_xd_s = mr_x_ct[sl_x, :] @ xr_d_ct[:, sl_x]
    for _ in range(ref_iteration):
        wg_d = fg[sl_x, sl_x] + xrmr_dx_s @ wg_d @ mrxr_ct_xd_s

    dlg_d = mr_x[:, sl_x] @ wg_d @ mr_x_ct[sl_x, :] - ag_diff

    # lesser component
    yl_d = np.divide(ieivec @ fl[sl_x, sl_x] @ ieivec_ct, 1 - eival_sq)
    wl_d = eivec @ yl_d @ eivec_ct
    for _ in range(ref_iteration):
        wl_d = fl[sl_x, sl_x] + xrmr_dx_s @ wl_d @ mrxr_ct_xd_s

    dll_d = mr_x[:, sl_x] @ wl_d @ mr_x_ct[sl_x, :] - al_diff

    return dlg_d, dll_d
