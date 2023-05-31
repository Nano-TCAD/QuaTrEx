"""
Contains functions for calculating the different correction terms
for the calculation of the screened interaction.
Naming is correction_ + from where the correction term is read from.

Has the same function as get_OBC_blocks.m, but split up for readability

"""
import numpy as np
import numpy.typing as npt
from scipy import sparse
import typing
import numba

def stack_three(
    a: sparse.csr_matrix,
    b: sparse.csr_matrix,
    c: sparse.csr_matrix
) -> sparse.csr_matrix:
    """stacks the inputs in the following way:
    [a;b,c]

    Args:
        a (sparse.csr_matrix):
        b (sparse.csr_matrix):
        c (sparse.csr_matrix):

    Returns:
        sparse.csr_matrix: stacked sparse matrix
    """
    tmp1 = sparse.hstack([b, c])
    out = sparse.vstack([a, tmp1])
    return out

def stack_four(
    a: sparse.csr_matrix,
    b: sparse.csr_matrix,
    c: sparse.csr_matrix,
    d: sparse.csr_matrix
) -> sparse.csr_matrix:
    """stacks the inputs in the following way:
    [a,b;c,d]

    Args:
        a (sparse.csr_matrix):
        b (sparse.csr_matrix):
        c (sparse.csr_matrix):
        d (sparse.csr_matrix):

    Returns:
        sparse.csr_matrix: stacked sparse matrix
    """
    tmp1 = sparse.hstack([a, b])
    tmp2 = sparse.hstack([c, d])
    out = sparse.vstack([tmp1, tmp2])
    return out

def stack_nine(
    a: sparse.csr_matrix,
    b: sparse.csr_matrix,
    c: sparse.csr_matrix,
    d: sparse.csr_matrix,
    e: sparse.csr_matrix,
    f: sparse.csr_matrix,
    g: sparse.csr_matrix,
    h: sparse.csr_matrix,
    i: sparse.csr_matrix,
) -> sparse.csr_matrix:
    """stacks the inputs in the following way:
    [a,b,c;d,e,f;g,h,i]

    Args:
        a (sparse.csr_matrix):
        b (sparse.csr_matrix):
        c (sparse.csr_matrix):
        d (sparse.csr_matrix):
        e (sparse.csr_matrix):
        f (sparse.csr_matrix):
        g (sparse.csr_matrix):
        h (sparse.csr_matrix):
        i (sparse.csr_matrix):

    Returns:
        sparse.csr_matrix: stacked sparse matrix
    """
    tmp1 = sparse.hstack([a, b, c])
    tmp2 = sparse.hstack([d, e, f])
    tmp3 = sparse.hstack([g, h, i])
    out = sparse.vstack([tmp1, tmp2, tmp3])
    return out

def stack_vh(
    vh_1: sparse.csr_matrix,
    vh_2: sparse.csr_matrix,
    nbc: np.int64
) -> typing.Tuple[sparse.csr_matrix,
                  sparse.csr_matrix,
                  sparse.csr_matrix]:
    """Creates three from vh derived sparse csr matrices
        Reasons why they have there from todo

    Args:
        vh_1 (sparse.csr_matrix): diag lesser polarization (nao,nao)
        vh_2 (sparse.csr_matrix): off lesser polarization (nao,nao)
        nbc (np.int64): parameter to determine low large block size is after mm

    Returns:
        typing.Tuple[sparse.csr_matrix, derived diagonal block
                    sparse.csr_matrix,  derived upper block
                    sparse.csr_matrix   derived lower block
                    ]
        
    """
    lb = vh_1.shape[0]
    vh_2_ct = vh_2.conjugate().transpose()
    if nbc == 1:
        vh_d = vh_1
        vh_u = vh_2
        vh_l = vh_u.conjugate().transpose()

    elif nbc == 2:
        zs_1 = sparse.csr_array((lb,lb), dtype=np.complex128)
        zs_2 = sparse.csr_array((lb,2*lb), dtype=np.complex128)
        vh_d = stack_four(vh_1,vh_2,
                          vh_2_ct,vh_1)
        vh_u = stack_three(zs_2,
                           vh_2,zs_1)
        vh_l = vh_u.conjugate().transpose()

    elif nbc == 3:
        # create empty sparse arrays to stack
        zs_1 = sparse.csr_array((lb,lb), dtype=np.complex128)
        zs_2 = sparse.csr_array((lb,2*lb), dtype=np.complex128)
        zs_3 = sparse.csr_array((2*lb,3*lb), dtype=np.complex128)

        # diagonal, upper, lower blocks
        vh_d = stack_nine(vh_1,vh_2,zs_1,
                        vh_2_ct,vh_1,vh_2,
                        zs_1,vh_2_ct,vh_1
                        )
        vh_u = stack_three(zs_3,
                        vh_2,zs_2)
        vh_l = vh_u.conjugate().transpose()

    else:
        raise ValueError(
        "Argument error, type input not possible")
    return vh_d, vh_u, vh_l


def stack_pr(
    pr_1: sparse.csr_matrix,
    pr_2: sparse.csr_matrix,
    nbc: np.int64
) -> typing.Tuple[sparse.csr_matrix,
                  sparse.csr_matrix,
                  sparse.csr_matrix]:
    """Creates three from pr derived sparse csr matrices
        Reasons why they have there from todo

    Args:
        pr_1 (sparse.csr_matrix): diag lesser polarization (nao,nao)
        pr_2 (sparse.csr_matrix): off lesser polarization (nao,nao)
        nbc (np.int64): parameter to determine low large block size is after mm

    Returns:
        typing.Tuple[sparse.csr_matrix, derived diagonal block
                    sparse.csr_matrix,  derived upper block
                    sparse.csr_matrix   derived lower block
                    ]
        
    """
    # read out slices to stack
    lb = pr_1.shape[0]
    pr_2_t = pr_2.transpose()
    if nbc == 1:
        pr_d = pr_1
        pr_u = pr_2
        pr_l = pr_u.transpose()

    elif nbc == 2:
        zs_1 = sparse.csr_array((lb,lb), dtype=np.complex128)
        zs_2 = sparse.csr_array((lb,2*lb), dtype=np.complex128)
        pr_d = stack_four(pr_1,pr_2,
                          pr_2_t,pr_1)
        pr_u = stack_three(zs_2,
                           pr_2,zs_1)
        pr_l = pr_u.transpose()

    elif nbc == 3:
        # create empty sparse arrays to stack
        zs_1 = sparse.csr_array((lb,lb), dtype=np.complex128)
        zs_2 = sparse.csr_array((lb,2*lb), dtype=np.complex128)
        zs_3 = sparse.csr_array((2*lb,3*lb), dtype=np.complex128)
        # diagonal, upper, lower blocks
        pr_d = stack_nine(pr_1,pr_2,zs_1,
                        pr_2_t,pr_1,pr_2,
                        zs_1,pr_2_t,pr_1
                        )
        pr_u = stack_three(zs_3,
                        pr_2,zs_2)
        pr_l = pr_u.transpose()
    else:
        raise ValueError(
        "Argument error, type input not possible")
    return pr_d, pr_u, pr_l

def stack_px(
    px_1: sparse.csr_matrix,
    px_2: sparse.csr_matrix,
    nbc: np.int64
) -> typing.Tuple[sparse.csr_matrix,
                  sparse.csr_matrix,
                  sparse.csr_matrix]:
    """Creates three from pl/pr derived sparse csr matrices
        Reasons why they have there from todo

    Args:
        px_1 (sparse.csr_matrix): diag lesser polarization (nao,nao)
        px_2 (sparse.csr_matrix): off lesser polarization (nao,nao)
        nbc (np.int64): parameter to determine low large block size is after mm

    Returns:
        typing.Tuple[sparse.csr_matrix, derived diagonal block
                    sparse.csr_matrix,  derived upper block
                    sparse.csr_matrix   derived lower block
                    ]
        
    """
    # read out slices to stack
    lb = px_1.shape[0]
    px_2_ct = px_2.conjugate().transpose()
    if nbc == 1:
        px_d = px_1
        px_u = px_2
        px_l = -px_u.conjugate().transpose()

    elif nbc == 2:
        zs_1 = sparse.csr_array((lb,lb), dtype=np.complex128)
        zs_2 = sparse.csr_array((lb,2*lb), dtype=np.complex128)
        px_d = stack_four(px_1,px_2,
                          -px_2_ct,px_1)
        px_u = stack_three(zs_2,
                           px_2,zs_1)
        px_l = -px_u.conjugate().transpose()

    elif nbc == 3:
        # create empty sparse arrays to stack
        zs_1 = sparse.csr_array((lb,lb), dtype=np.complex128)
        zs_2 = sparse.csr_array((lb,2*lb), dtype=np.complex128)
        zs_3 = sparse.csr_array((2*lb,3*lb), dtype=np.complex128)
        # diagonal, upper, lower blocks
        px_d = stack_nine(px_1,px_2,zs_1,
                        -px_2_ct,px_1,px_2,
                        zs_1,-px_2_ct,px_1
                        )
        px_u = stack_three(zs_3,
                        px_2,zs_2)
        px_l = -px_u.conjugate().transpose()
    else:
        raise ValueError(
        "Argument error, type input not possible")
    return px_d, px_u, px_l

def stack_mr(
    vh_d: sparse.csr_matrix,
    vh_u: sparse.csr_matrix,
    vh_l: sparse.csr_matrix,
    pr_d: sparse.csr_matrix,
    pr_u: sparse.csr_matrix,
    pr_l: sparse.csr_matrix
) -> typing.Tuple[sparse.csr_matrix,
                  sparse.csr_matrix,
                  sparse.csr_matrix]:
    """Generate the derived values from mr.
    Todo find out why they are how they are

    Args:
        vh_d (sparse.csr_matrix): (3x,3x) of the initial block size
        vh_u (sparse.csr_matrix): (3x,3x) of the initial block size
        vh_l (sparse.csr_matrix): (3x,3x) of the initial block size
        pr_d (sparse.csr_matrix): (3x,3x) of the initial block size
        pr_u (sparse.csr_matrix): (3x,3x) of the initial block size
        pr_l (sparse.csr_matrix): (3x,3x) of the initial block size

    Returns:
        typing.Tuple[sparse.csr_matrix, derived diagonal block
                     sparse.csr_matrix, derived upper block
                     sparse.csr_matrix  derived lower block
                    ]
        
    """
    lb = vh_d.shape[0]
    mr_d = sparse.identity(lb, format="csr",dtype=np.complex128) * (1+1j*1e-10)
    mr_d = mr_d - vh_l @ pr_u - vh_d @ pr_d - vh_u @ pr_l
    mr_u = -vh_d @ pr_u -vh_u @ pr_d
    mr_l = -vh_l @ pr_d -vh_d @ pr_l
    return mr_d, mr_u, mr_l


def stack_lx(
    vh_d: sparse.csr_matrix,
    vh_u: sparse.csr_matrix,
    vh_l: sparse.csr_matrix,
    px_d: sparse.csr_matrix,
    px_u: sparse.csr_matrix,
    px_l: sparse.csr_matrix
) -> typing.Tuple[sparse.csr_matrix,
                  sparse.csr_matrix,
                  sparse.csr_matrix]:
    """Generate the derived values from ll/lg.
    Todo find out why they are how they are

    Args:
        vh_d (sparse.csr_matrix): (3x,3x) of the initial block size
        vh_u (sparse.csr_matrix): (3x,3x) of the initial block size
        vh_l (sparse.csr_matrix): (3x,3x) of the initial block size
        px_d (sparse.csr_matrix): (3x,3x) of the initial block size
        px_u (sparse.csr_matrix): (3x,3x) of the initial block size
        px_l (sparse.csr_matrix): (3x,3x) of the initial block size

    Returns:
        typing.Tuple[sparse.csr_matrix, derived diagonal block
                     sparse.csr_matrix, derived upper block
                     sparse.csr_matrix  derived lower block
                    ]
        
    """
    lx_d = (vh_l @ px_d @ vh_u +
            vh_l @ px_u @ vh_d +
            vh_d @ px_l @ vh_u +
            vh_d @ px_d @ vh_d +
            vh_d @ px_u @ vh_l +
            vh_u @ px_l @ vh_d +
            vh_u @ px_d @ vh_l)

    lx_u = (vh_l @ px_u @ vh_u +
            vh_d @ px_d @ vh_u +
            vh_d @ px_u @ vh_d +
            vh_u @ px_l @ vh_u +
            vh_u @ px_d @ vh_d)

    lx_l = -lx_u.conjugate().transpose()
    return lx_d, lx_u, lx_l


def get_mm_obc(
        vh_1: sparse.csr_matrix,
        vh_2: sparse.csr_matrix,
        pg_1: sparse.csr_matrix,
        pg_2: sparse.csr_matrix,
        pl_1: sparse.csr_matrix,
        pl_2: sparse.csr_matrix,
        pr_1: sparse.csr_matrix,
        pr_2: sparse.csr_matrix,
        nbc: int
) -> typing.Tuple[
    typing.Tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128], npt.NDArray[np.complex128]],
    typing.Tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128], npt.NDArray[np.complex128]],
    typing.Tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128], npt.NDArray[np.complex128]],
    typing.Tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128]],
    typing.Tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128]],
    typing.Tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128]],
    typing.Tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128]]
]:
    """
    Calculates the different correction terms for the scattering OBCs for the screened interaction calculations.
    In this version the blocks are not stacked, but the matrices are multiplied directly and inserted afterwards.

    Args:
        vh_1 (sparse.csr_matrix): Diagonal block of effective interaction
        vh_2 (sparse.csr_matrix): Off diagonal block of effective interaction
        pg_1 (sparse.csr_matrix): Diagonal block of greater polarization
        pg_2 (sparse.csr_matrix): Off diagonal block of greater polarization
        pl_1 (sparse.csr_matrix): Diagonal block of lesser polarization
        pl_2 (sparse.csr_matrix): Off diagonal block of lesser polarization
        pr_1 (sparse.csr_matrix): Diagonal block of retarded polarization
        pr_2 (sparse.csr_matrix): Off diagonal block of retarded polarization
        nbc (int): How block size changes after matrix multiplication

    Returns:
        typing.Tuple[
            typing.Tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128], npt.NDArray[np.complex128]],
            typing.Tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128], npt.NDArray[np.complex128]],
            typing.Tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128], npt.NDArray[np.complex128]],
            typing.Tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128]],
            typing.Tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128]],
            typing.Tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128]],
            typing.Tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128]]
            ]: mr_d/u/l, lg_d/u/l, ll_d/u/l, dg_lu/ul, dl_lu/ul, vh_u/l
    """
    # block size
    lb = vh_1.shape[0]
    # block size after mm
    lb_mm = nbc * lb
    # define the right blocks
    vh_d1 = vh_1
    vh_u1 = vh_2
    vh_l1 = vh_u1.conjugate().transpose()
    pg_d1 = pg_1
    pg_u1 = pg_2
    pg_l1 = -pg_u1.conjugate().transpose()
    pl_d1 = pl_1
    pl_u1 = pl_2
    pl_l1 = -pl_u1.conjugate().transpose()
    pr_d1 = pr_1
    pr_u1 = pr_2
    pr_l1 = pr_u1.transpose()

    # output matrices
    mr_d2 = np.empty((lb_mm,lb_mm), dtype=np.complex128)
    mr_u2 = np.zeros((lb_mm,lb_mm), dtype=np.complex128)
    mr_l2 = np.zeros((lb_mm,lb_mm), dtype=np.complex128)
    lg_d2 = np.empty((lb_mm,lb_mm), dtype=np.complex128)
    lg_u2 = np.zeros((lb_mm,lb_mm), dtype=np.complex128)
    lg_l2 = np.empty((lb_mm,lb_mm), dtype=np.complex128)
    ll_d2 = np.empty((lb_mm,lb_mm), dtype=np.complex128)
    ll_u2 = np.zeros((lb_mm,lb_mm), dtype=np.complex128)
    ll_l2 = np.empty((lb_mm,lb_mm), dtype=np.complex128)
    dmr_lu = np.zeros((lb_mm,lb_mm), dtype=np.complex128)
    dmr_ul = np.zeros((lb_mm,lb_mm), dtype=np.complex128)
    dlg_lu = np.zeros((lb_mm,lb_mm), dtype=np.complex128)
    dlg_ul = np.zeros((lb_mm,lb_mm), dtype=np.complex128)
    dll_lu = np.zeros((lb_mm,lb_mm), dtype=np.complex128)
    dll_ul = np.zeros((lb_mm,lb_mm), dtype=np.complex128)
    vh_u = np.zeros((lb_mm,lb_mm), dtype=np.complex128)
    vh_l = np.zeros((lb_mm,lb_mm), dtype=np.complex128)
    if nbc == 1:
        # fill output matrices
        # M^{r}\left(E\right)
        mr_d2[:,:] = -(vh_d1 @ pr_d1).toarray() - (vh_u1 @ pr_l1).toarray() -(vh_l1 @ pr_u1).toarray()
        mr_u2[:,:] = -(vh_d1 @ pr_u1).toarray() - (vh_u1 @ pr_d1).toarray()
        mr_l2[:,:] = -(vh_d1 @ pr_l1).toarray() - (vh_l1 @ pr_d1).toarray()

        # from L^{\lessgtr}\left(E\right)
        lg_d2[:,:] = ((vh_l1 @ pg_d1 @ vh_u1).toarray() +
                (vh_l1 @ pg_u1 @ vh_d1).toarray() +
                (vh_d1 @ pg_l1 @ vh_u1).toarray() +
                (vh_d1 @ pg_d1 @ vh_d1).toarray() +
                (vh_d1 @ pg_u1 @ vh_l1).toarray() +
                (vh_u1 @ pg_l1 @ vh_d1).toarray() +
                (vh_u1 @ pg_d1 @ vh_l1).toarray())
        lg_u2[:,:] = ((vh_l1 @ pg_u1 @ vh_u1).toarray() +
                (vh_d1 @ pg_d1 @ vh_u1).toarray() +
                (vh_d1 @ pg_u1 @ vh_d1).toarray() +
                (vh_u1 @ pg_l1 @ vh_u1).toarray() +
                (vh_u1 @ pg_d1 @ vh_d1).toarray())
        ll_d2[:,:] = ((vh_l1 @ pl_d1 @ vh_u1).toarray() +
                (vh_l1 @ pl_u1 @ vh_d1).toarray() +
                (vh_d1 @ pl_l1 @ vh_u1).toarray() +
                (vh_d1 @ pl_d1 @ vh_d1).toarray() +
                (vh_d1 @ pl_u1 @ vh_l1).toarray() +
                (vh_u1 @ pl_l1 @ vh_d1).toarray() +
                (vh_u1 @ pl_d1 @ vh_l1).toarray())
        ll_u2[:,:] = ((vh_l1 @ pl_u1 @ vh_u1).toarray() +
                (vh_d1 @ pl_d1 @ vh_u1).toarray() +
                (vh_d1 @ pl_u1 @ vh_d1).toarray() +
                (vh_u1 @ pl_l1 @ vh_u1).toarray() +
                (vh_u1 @ pl_d1 @ vh_d1).toarray())
        dmr_lu[:,:] = -(vh_l1 @ pr_u1).toarray()
        dmr_ul[:,:] = -(vh_u1 @ pr_l1).toarray()
        dlg_lu[:,:] = (vh_l1 @ pg_d1 @ vh_u1).toarray() + (vh_l1 @ pg_u1 @ vh_d1).toarray() + (vh_d1 @ pg_l1 @ vh_u1).toarray()
        dll_lu[:,:] = (vh_l1 @ pl_d1 @ vh_u1).toarray() + (vh_l1 @ pl_u1 @ vh_d1).toarray() + (vh_d1 @ pl_l1 @ vh_u1).toarray()
        dlg_ul[:,:] = (vh_u1 @ pg_d1 @ vh_l1).toarray() + (vh_u1 @ pg_l1 @ vh_d1).toarray() + (vh_d1 @ pg_u1 @ vh_l1).toarray()
        dll_ul[:,:] = (vh_u1 @ pl_d1 @ vh_l1).toarray() + (vh_u1 @ pl_l1 @ vh_d1).toarray() + (vh_d1 @ pl_u1 @ vh_l1).toarray()
        vh_u[:,:] = vh_u1.toarray()
        vh_l[:,:] = vh_l1.toarray()
    elif nbc == 2:
        # compute multiplications
        vhpr_d1d1 = -(vh_d1 @ pr_d1).toarray()
        vhpr_d1u1 = -(vh_d1 @ pr_u1).toarray()
        vhpr_d1l1 = -(vh_d1 @ pr_l1).toarray()
        vhpr_u1d1 = -(vh_u1 @ pr_d1).toarray()
        vhpr_u1u1 = -(vh_u1 @ pr_u1).toarray()
        vhpr_u1l1 = -(vh_u1 @ pr_l1).toarray()
        vhpr_l1d1 = -(vh_l1 @ pr_d1).toarray()
        vhpr_l1u1 = -(vh_l1 @ pr_u1).toarray()
        vhpr_l1l1 = -(vh_l1 @ pr_l1).toarray()

        vhpg_d1d1 = vh_d1 @ pg_d1
        vhpg_d1u1 = vh_d1 @ pg_u1
        vhpg_d1l1 = vh_d1 @ pg_l1
        vhpg_u1d1 = vh_u1 @ pg_d1
        vhpg_u1u1 = vh_u1 @ pg_u1
        vhpg_u1l1 = vh_u1 @ pg_l1
        vhpg_l1d1 = vh_l1 @ pg_d1
        vhpg_l1u1 = vh_l1 @ pg_u1
        vhpg_l1l1 = vh_l1 @ pg_l1
        vhpgvh_d1d1d1 = (vhpg_d1d1 @ vh_d1).toarray()
        vhpgvh_d1d1u1 = (vhpg_d1d1 @ vh_u1).toarray()
        vhpgvh_d1d1l1 = (vhpg_d1d1 @ vh_l1).toarray()
        vhpgvh_d1u1d1 = (vhpg_d1u1 @ vh_d1).toarray()
        vhpgvh_d1u1u1 = (vhpg_d1u1 @ vh_u1).toarray()
        vhpgvh_d1u1l1 = (vhpg_d1u1 @ vh_l1).toarray()
        vhpgvh_d1l1d1 = (vhpg_d1l1 @ vh_d1).toarray()
        vhpgvh_d1l1u1 = (vhpg_d1l1 @ vh_u1).toarray()
        vhpgvh_u1d1d1 = (vhpg_u1d1 @ vh_d1).toarray()
        vhpgvh_u1d1u1 = (vhpg_u1d1 @ vh_u1).toarray()
        vhpgvh_u1d1l1 = (vhpg_u1d1 @ vh_l1).toarray()
        vhpgvh_u1u1d1 = (vhpg_u1u1 @ vh_d1).toarray()
        vhpgvh_u1u1u1 = (vhpg_u1u1 @ vh_u1).toarray()
        vhpgvh_u1u1l1 = (vhpg_u1u1 @ vh_l1).toarray()
        vhpgvh_u1l1d1 = (vhpg_u1l1 @ vh_d1).toarray()
        vhpgvh_u1l1u1 = (vhpg_u1l1 @ vh_u1).toarray()
        vhpgvh_u1l1l1 = (vhpg_u1l1 @ vh_l1).toarray()
        vhpgvh_l1d1d1 = (vhpg_l1d1 @ vh_d1).toarray()
        vhpgvh_l1d1u1 = (vhpg_l1d1 @ vh_u1).toarray()
        vhpgvh_l1u1d1 = (vhpg_l1u1 @ vh_d1).toarray()
        vhpgvh_l1u1u1 = (vhpg_l1u1 @ vh_u1).toarray()
        vhpgvh_l1u1l1 = (vhpg_l1u1 @ vh_l1).toarray()
        vhpgvh_l1l1u1 = (vhpg_l1l1 @ vh_u1).toarray()
        vhpl_d1d1 = vh_d1 @ pl_d1
        vhpl_d1u1 = vh_d1 @ pl_u1
        vhpl_d1l1 = vh_d1 @ pl_l1
        vhpl_u1d1 = vh_u1 @ pl_d1
        vhpl_u1u1 = vh_u1 @ pl_u1
        vhpl_u1l1 = vh_u1 @ pl_l1
        vhpl_l1d1 = vh_l1 @ pl_d1
        vhpl_l1u1 = vh_l1 @ pl_u1
        vhpl_l1l1 = vh_l1 @ pl_l1
        vhplvh_d1d1d1 = (vhpl_d1d1 @ vh_d1).toarray()
        vhplvh_d1d1u1 = (vhpl_d1d1 @ vh_u1).toarray()
        vhplvh_d1d1l1 = (vhpl_d1d1 @ vh_l1).toarray()
        vhplvh_d1u1d1 = (vhpl_d1u1 @ vh_d1).toarray()
        vhplvh_d1u1u1 = (vhpl_d1u1 @ vh_u1).toarray()
        vhplvh_d1u1l1 = (vhpl_d1u1 @ vh_l1).toarray()
        vhplvh_d1l1d1 = (vhpl_d1l1 @ vh_d1).toarray()
        vhplvh_d1l1u1 = (vhpl_d1l1 @ vh_u1).toarray()
        vhplvh_u1d1d1 = (vhpl_u1d1 @ vh_d1).toarray()
        vhplvh_u1d1u1 = (vhpl_u1d1 @ vh_u1).toarray()
        vhplvh_u1d1l1 = (vhpl_u1d1 @ vh_l1).toarray()
        vhplvh_u1u1d1 = (vhpl_u1u1 @ vh_d1).toarray()
        vhplvh_u1u1u1 = (vhpl_u1u1 @ vh_u1).toarray()
        vhplvh_u1u1l1 = (vhpl_u1u1 @ vh_l1).toarray()
        vhplvh_u1l1d1 = (vhpl_u1l1 @ vh_d1).toarray()
        vhplvh_u1l1u1 = (vhpl_u1l1 @ vh_u1).toarray()
        vhplvh_u1l1l1 = (vhpl_u1l1 @ vh_l1).toarray()
        vhplvh_l1d1d1 = (vhpl_l1d1 @ vh_d1).toarray()
        vhplvh_l1d1u1 = (vhpl_l1d1 @ vh_u1).toarray()
        vhplvh_l1u1d1 = (vhpl_l1u1 @ vh_d1).toarray()
        vhplvh_l1u1u1 = (vhpl_l1u1 @ vh_u1).toarray()
        vhplvh_l1u1l1 = (vhpl_l1u1 @ vh_l1).toarray()
        vhplvh_l1l1u1 = (vhpl_l1l1 @ vh_u1).toarray()

        # fill output matrices
        # M^{r}\left(E\right)
        mr_d2[:lb,:lb] = vhpr_d1d1 + vhpr_u1l1 + vhpr_l1u1
        mr_d2[:lb,lb:] = vhpr_d1u1 + vhpr_u1d1
        mr_d2[lb:,:lb] = vhpr_d1l1 + vhpr_l1d1
        mr_d2[lb:,lb:] = vhpr_d1d1 + vhpr_l1u1 + vhpr_u1l1

        mr_u2[:lb,:lb] = vhpr_u1u1
        mr_u2[lb:,:lb] = vhpr_d1u1 + vhpr_u1d1
        mr_u2[lb:,lb:] = vhpr_u1u1

        mr_l2[:lb,:lb] = vhpr_l1l1
        mr_l2[:lb,lb:] = vhpr_d1l1 + vhpr_l1d1
        mr_l2[lb:,lb:] = vhpr_l1l1

        # L^{\lessgtr}\left(E\right)
        lg_d2[:lb,:lb] = vhpgvh_d1l1u1 + vhpgvh_l1d1u1 + vhpgvh_l1u1d1 + vhpgvh_d1d1d1 + vhpgvh_u1l1d1 + vhpgvh_d1u1l1 + vhpgvh_u1d1l1
        lg_d2[:lb,lb:] = vhpgvh_l1u1u1 + vhpgvh_u1u1l1 + vhpgvh_d1d1u1 + vhpgvh_u1l1u1 + vhpgvh_d1u1d1 + vhpgvh_u1d1d1
        lg_d2[lb:,:lb] = vhpgvh_l1l1u1 + vhpgvh_u1l1l1 + vhpgvh_d1d1l1 + vhpgvh_l1u1l1 + vhpgvh_d1l1d1 + vhpgvh_l1d1d1
        lg_d2[lb:,lb:] = vhpgvh_d1u1l1 + vhpgvh_u1d1l1 + vhpgvh_u1l1d1 + vhpgvh_d1d1d1 + vhpgvh_l1u1d1 + vhpgvh_d1l1u1 + vhpgvh_l1d1u1

        lg_u2[:lb,:lb] = vhpgvh_u1u1d1 + vhpgvh_d1u1u1 + vhpgvh_u1d1u1
        lg_u2[:lb,lb:] = vhpgvh_u1u1u1
        lg_u2[lb:,:lb] = vhpgvh_d1u1d1 + vhpgvh_u1d1d1 + vhpgvh_u1l1u1 + vhpgvh_u1u1l1 + vhpgvh_d1d1u1 + vhpgvh_l1u1u1
        lg_u2[lb:,lb:] = vhpgvh_d1u1u1 + vhpgvh_u1d1u1 + vhpgvh_u1u1d1

        ll_d2[:lb,:lb] = vhplvh_d1l1u1 + vhplvh_l1d1u1 + vhplvh_l1u1d1 + vhplvh_d1d1d1 + vhplvh_u1l1d1 + vhplvh_d1u1l1 + vhplvh_u1d1l1
        ll_d2[:lb,lb:] = vhplvh_l1u1u1 + vhplvh_u1u1l1 + vhplvh_d1d1u1 + vhplvh_u1l1u1 + vhplvh_d1u1d1 + vhplvh_u1d1d1
        ll_d2[lb:,:lb] = vhplvh_l1l1u1 + vhplvh_u1l1l1 + vhplvh_d1d1l1 + vhplvh_l1u1l1 + vhplvh_d1l1d1 + vhplvh_l1d1d1
        ll_d2[lb:,lb:] = vhplvh_d1u1l1 + vhplvh_u1d1l1 + vhplvh_u1l1d1 + vhplvh_d1d1d1 + vhplvh_l1u1d1 + vhplvh_d1l1u1 + vhplvh_l1d1u1

        ll_u2[:lb,:lb] = vhplvh_u1u1d1 + vhplvh_d1u1u1 + vhplvh_u1d1u1
        ll_u2[:lb,lb:] = vhplvh_u1u1u1
        ll_u2[lb:,:lb] = vhplvh_d1u1d1 + vhplvh_u1d1d1 + vhplvh_u1l1u1 + vhplvh_u1u1l1 + vhplvh_d1d1u1 + vhplvh_l1u1u1
        ll_u2[lb:,lb:] = vhplvh_d1u1u1 + vhplvh_u1d1u1 + vhplvh_u1u1d1

        dmr_lu[:lb,:lb] = vhpr_l1u1
        dmr_ul[lb:,lb:] = vhpr_u1l1
    
        dlg_lu[:lb,:lb] = vhpgvh_d1l1u1 + vhpgvh_l1d1u1 + vhpgvh_l1u1d1
        dlg_lu[:lb,lb:] = vhpgvh_l1u1u1
        dlg_lu[lb:,:lb] = vhpgvh_l1l1u1

        dlg_ul[:lb,lb:] = vhpgvh_u1u1l1
        dlg_ul[lb:,:lb] = vhpgvh_u1l1l1
        dlg_ul[lb:,lb:] = vhpgvh_d1u1l1 + vhpgvh_u1d1l1 + vhpgvh_u1l1d1

        dll_lu[:lb,:lb] = vhplvh_d1l1u1 + vhplvh_l1d1u1 + vhplvh_l1u1d1
        dll_lu[:lb,lb:] = vhplvh_l1u1u1
        dll_lu[lb:,:lb] = vhplvh_l1l1u1

        dll_ul[:lb,lb:] = vhplvh_u1u1l1
        dll_ul[lb:,:lb] = vhplvh_u1l1l1
        dll_ul[lb:,lb:] = vhplvh_d1u1l1 + vhplvh_u1d1l1 + vhplvh_u1l1d1

        vh_u[lb:,:lb] = vh_u1.toarray()
        vh_l[:lb,lb:] = vh_l1.toarray()
    elif nbc == 3:
        # compute multiplications
        vhpr_d1d1 = -(vh_d1 @ pr_d1).toarray()
        vhpr_d1u1 = -(vh_d1 @ pr_u1).toarray()
        vhpr_d1l1 = -(vh_d1 @ pr_l1).toarray()
        vhpr_u1d1 = -(vh_u1 @ pr_d1).toarray()
        vhpr_u1u1 = -(vh_u1 @ pr_u1).toarray()
        vhpr_u1l1 = -(vh_u1 @ pr_l1).toarray()
        vhpr_l1d1 = -(vh_l1 @ pr_d1).toarray()
        vhpr_l1u1 = -(vh_l1 @ pr_u1).toarray()
        vhpr_l1l1 = -(vh_l1 @ pr_l1).toarray()

        vhpg_d1d1 = vh_d1 @ pg_d1
        vhpg_d1u1 = vh_d1 @ pg_u1
        vhpg_d1l1 = vh_d1 @ pg_l1
        vhpg_u1d1 = vh_u1 @ pg_d1
        vhpg_u1u1 = vh_u1 @ pg_u1
        vhpg_u1l1 = vh_u1 @ pg_l1
        vhpg_l1d1 = vh_l1 @ pg_d1
        vhpg_l1u1 = vh_l1 @ pg_u1
        vhpg_l1l1 = vh_l1 @ pg_l1
        vhpgvh_d1d1d1 = (vhpl_d1d1 @ vh_d1).toarray()
        vhpgvh_d1d1u1 = (vhpl_d1d1 @ vh_u1).toarray()
        vhpgvh_d1d1l1 = (vhpl_d1d1 @ vh_l1).toarray()
        vhpgvh_d1u1d1 = (vhpl_d1u1 @ vh_d1).toarray()
        vhpgvh_d1u1u1 = (vhpl_d1u1 @ vh_u1).toarray()
        vhpgvh_d1u1l1 = (vhpl_d1u1 @ vh_l1).toarray()
        vhpgvh_d1l1d1 = (vhpl_d1l1 @ vh_d1).toarray()
        vhpgvh_d1l1u1 = (vhpl_d1l1 @ vh_u1).toarray()
        vhpgvh_d1l1l1 = (vhpl_d1l1 @ vh_l1).toarray()
        vhpgvh_u1d1d1 = (vhpl_u1d1 @ vh_d1).toarray()
        vhpgvh_u1d1u1 = (vhpl_u1d1 @ vh_u1).toarray()
        vhpgvh_u1d1l1 = (vhpl_u1d1 @ vh_l1).toarray()
        vhpgvh_u1u1d1 = (vhpl_u1u1 @ vh_d1).toarray()
        vhpgvh_u1u1u1 = (vhpl_u1u1 @ vh_u1).toarray()
        vhpgvh_u1u1l1 = (vhpl_u1u1 @ vh_l1).toarray()
        vhpgvh_u1l1d1 = (vhpl_u1l1 @ vh_d1).toarray()
        vhpgvh_u1l1u1 = (vhpl_u1l1 @ vh_u1).toarray()
        vhpgvh_u1l1l1 = (vhpl_u1l1 @ vh_l1).toarray()
        vhpgvh_l1d1d1 = (vhpl_l1d1 @ vh_d1).toarray()
        vhpgvh_l1d1u1 = (vhpl_l1d1 @ vh_u1).toarray()
        vhpgvh_l1d1l1 = (vhpl_l1d1 @ vh_l1).toarray()
        vhpgvh_l1u1d1 = (vhpl_l1u1 @ vh_d1).toarray()
        vhpgvh_l1u1u1 = (vhpl_l1u1 @ vh_u1).toarray()
        vhpgvh_l1u1l1 = (vhpl_l1u1 @ vh_l1).toarray()
        vhpgvh_l1l1d1 = (vhpl_l1l1 @ vh_d1).toarray()
        vhpgvh_l1l1u1 = (vhpl_l1l1 @ vh_u1).toarray()
        vhpl_d1d1 = vh_d1 @ pl_d1
        vhpl_d1u1 = vh_d1 @ pl_u1
        vhpl_d1l1 = vh_d1 @ pl_l1
        vhpl_u1d1 = vh_u1 @ pl_d1
        vhpl_u1u1 = vh_u1 @ pl_u1
        vhpl_u1l1 = vh_u1 @ pl_l1
        vhpl_l1d1 = vh_l1 @ pl_d1
        vhpl_l1u1 = vh_l1 @ pl_u1
        vhpl_l1l1 = vh_l1 @ pl_l1
        vhplvh_d1d1d1 = (vhpl_d1d1 @ vh_d1).toarray()
        vhplvh_d1d1u1 = (vhpl_d1d1 @ vh_u1).toarray()
        vhplvh_d1d1l1 = (vhpl_d1d1 @ vh_l1).toarray()
        vhplvh_d1u1d1 = (vhpl_d1u1 @ vh_d1).toarray()
        vhplvh_d1u1u1 = (vhpl_d1u1 @ vh_u1).toarray()
        vhplvh_d1u1l1 = (vhpl_d1u1 @ vh_l1).toarray()
        vhplvh_d1l1d1 = (vhpl_d1l1 @ vh_d1).toarray()
        vhplvh_d1l1u1 = (vhpl_d1l1 @ vh_u1).toarray()
        vhplvh_d1l1l1 = (vhpl_d1l1 @ vh_l1).toarray()
        vhplvh_u1d1d1 = (vhpl_u1d1 @ vh_d1).toarray()
        vhplvh_u1d1u1 = (vhpl_u1d1 @ vh_u1).toarray()
        vhplvh_u1d1l1 = (vhpl_u1d1 @ vh_l1).toarray()
        vhplvh_u1u1d1 = (vhpl_u1u1 @ vh_d1).toarray()
        vhplvh_u1u1u1 = (vhpl_u1u1 @ vh_u1).toarray()
        vhplvh_u1u1l1 = (vhpl_u1u1 @ vh_l1).toarray()
        vhplvh_u1l1d1 = (vhpl_u1l1 @ vh_d1).toarray()
        vhplvh_u1l1u1 = (vhpl_u1l1 @ vh_u1).toarray()
        vhplvh_u1l1l1 = (vhpl_u1l1 @ vh_l1).toarray()
        vhplvh_l1d1d1 = (vhpl_l1d1 @ vh_d1).toarray()
        vhplvh_l1d1u1 = (vhpl_l1d1 @ vh_u1).toarray()
        vhplvh_l1d1l1 = (vhpl_l1d1 @ vh_l1).toarray()
        vhplvh_l1u1d1 = (vhpl_l1u1 @ vh_d1).toarray()
        vhplvh_l1u1u1 = (vhpl_l1u1 @ vh_u1).toarray()
        vhplvh_l1u1l1 = (vhpl_l1u1 @ vh_l1).toarray()
        vhplvh_l1l1d1 = (vhpl_l1l1 @ vh_d1).toarray()
        vhplvh_l1l1u1 = (vhpl_l1l1 @ vh_u1).toarray()

        # fill output matrices
        # M^{r}\left(E\right)
        mr_d2[:lb,:lb] = vhpr_d1d1 + vhpr_l1u1 + vhpr_u1l1
        mr_d2[:lb,lb:2*lb] = vhpr_d1u1 + vhpr_u1d1
        mr_d2[:lb,2*lb:] = vhpr_u1u1
        mr_d2[lb:2*lb,:lb] = vhpr_d1l1 + vhpr_l1d1
        mr_d2[lb:2*lb,lb:2*lb] = vhpr_d1d1 + vhpr_l1u1 + vhpr_u1l1
        mr_d2[lb:2*lb,2*lb:] = vhpr_d1u1 + vhpr_u1d1
        mr_d2[2*lb:,:lb] = vhpr_l1l1
        mr_d2[2*lb:,lb:2*lb] = vhpr_d1l1 + vhpr_l1d1
        mr_d2[2*lb:,2*lb:] = vhpr_d1d1 + vhpr_l1u1 + vhpr_u1l1

        mr_u2[lb:2*lb,:lb] = vhpr_u1u1
        mr_u2[2*lb:,:lb] = vhpr_d1u1 + vhpr_u1d1
        mr_u2[2*lb:,lb:2*lb] = vhpr_u1u1

        mr_l2[:lb,lb:2*lb] = vhpr_l1l1
        mr_l2[:lb,2*lb:] = vhpr_d1l1 + vhpr_l1d1
        mr_l2[lb:2*lb,2*lb:] = vhpr_l1l1

        lg_d2[:lb,:lb] = vhpgvh_d1l1u1 + vhpgvh_l1d1u1 + vhpgvh_l1u1d1 + vhpgvh_d1d1d1 + vhpgvh_u1l1d1 + vhpgvh_d1u1l1 + vhpgvh_u1d1l1
        lg_d2[:lb,lb:2*lb] = vhpgvh_l1u1u1 + vhpgvh_u1u1l1 + vhpgvh_d1d1u1 + vhpgvh_u1l1u1 + vhpgvh_d1u1d1 + vhpgvh_u1d1d1
        lg_d2[:lb,2*lb:] = vhpgvh_u1u1d1 + vhpgvh_d1u1d1 + vhpgvh_u1d1u1
        lg_d2[lb:2*lb,:lb] = vhpgvh_l1l1u1 + vhpgvh_d1l1d1 + vhpgvh_l1d1d1 + vhpgvh_d1d1l1 + vhpgvh_l1u1l1 + vhpgvh_u1l1l1
        lg_d2[lb:2*lb,lb:2*lb] = vhpgvh_d1l1u1 + vhpgvh_l1d1u1 + vhpgvh_d1u1l1 + vhpgvh_u1d1l1 + vhpgvh_d1d1d1 + vhpgvh_l1u1d1 + vhpgvh_u1l1d1
        lg_d2[lb:2*lb,2*lb:] = vhpgvh_u1u1l1 + vhpgvh_d1u1d1 + vhpgvh_u1d1d1 + vhpgvh_d1d1u1 + vhpgvh_l1u1u1 + vhpgvh_u1l1u1
        lg_d2[2*lb:,:lb] = vhpgvh_l1l1d1 + vhpgvh_d1l1l1 + vhpgvh_l1d1l1
        lg_d2[2*lb:,lb:2*lb] = vhpgvh_l1l1u1 + vhpgvh_u1l1l1 + vhpgvh_d1d1l1 + vhpgvh_l1u1l1 + vhpgvh_d1l1d1 + vhpgvh_l1d1d1
        lg_d2[2*lb:,2*lb:] = vhpgvh_d1u1l1 + vhpgvh_u1d1l1 + vhpgvh_u1l1d1 + vhpgvh_d1d1d1 + vhpgvh_l1u1d1 + vhpgvh_d1l1u1 + vhpgvh_l1d1u1

        lg_u2[:lb,:lb] = vhpgvh_u1u1u1
        lg_u2[lb:2*lb,:lb] = vhpgvh_u1u1d1 + vhpgvh_d1u1d1 + vhpgvh_u1d1u1
        lg_u2[lb:2*lb,lb:2*lb] = vhpgvh_u1u1u1
        lg_u2[2*lb:,:lb] = vhpgvh_d1u1d1 + vhpgvh_u1d1d1 + vhpgvh_u1l1u1 + vhpgvh_u1u1l1 + vhpgvh_d1d1u1 + vhpgvh_l1u1u1
        lg_u2[2*lb:,lb:2*lb] = vhpgvh_d1u1d1 + vhpgvh_u1d1u1 + vhpgvh_u1u1d1
        lg_u2[2*lb:,2*lb:] = vhpgvh_u1u1u1

        ll_d2[:lb,:lb] = vhplvh_d1l1u1 + vhplvh_l1d1u1 + vhplvh_l1u1d1 + vhplvh_d1d1d1 + vhplvh_u1l1d1 + vhplvh_d1u1l1 + vhplvh_u1d1l1
        ll_d2[:lb,lb:2*lb] = vhplvh_l1u1u1 + vhplvh_u1u1l1 + vhplvh_d1d1u1 + vhplvh_u1l1u1 + vhplvh_d1u1d1 + vhplvh_u1d1d1
        ll_d2[:lb,2*lb:] = vhplvh_u1u1d1 + vhplvh_d1u1u1 + vhplvh_u1d1u1
        ll_d2[lb:2*lb,:lb] = vhplvh_l1l1u1 + vhplvh_d1l1d1 + vhplvh_l1d1d1 + vhplvh_d1d1l1 + vhplvh_l1u1l1 + vhplvh_u1l1l1
        ll_d2[lb:2*lb,lb:2*lb] = vhplvh_d1l1u1 + vhplvh_l1d1u1 + vhplvh_d1u1l1 + vhplvh_u1d1l1 + vhplvh_d1d1d1 + vhplvh_l1u1d1 + vhplvh_u1l1d1
        ll_d2[lb:2*lb,2*lb:] = vhplvh_u1u1l1 + vhplvh_d1u1d1 + vhplvh_u1d1d1 + vhplvh_d1d1u1 + vhplvh_l1u1u1 + vhplvh_u1l1u1
        ll_d2[2*lb:,:lb] = vhplvh_l1l1d1 + vhplvh_d1l1l1 + vhplvh_l1d1l1
        ll_d2[2*lb:,lb:2*lb] = vhplvh_l1l1u1 + vhplvh_u1l1l1 + vhplvh_d1d1l1 + vhplvh_l1u1l1 + vhplvh_d1l1d1 + vhplvh_l1d1d1
        ll_d2[2*lb:,2*lb:] = vhplvh_d1u1l1 + vhplvh_u1d1l1 + vhplvh_u1l1d1 + vhplvh_d1d1d1 + vhplvh_l1u1d1 + vhplvh_d1l1u1 + vhplvh_l1d1u1

        ll_u2[:lb,:lb] = vhplvh_u1u1u1
        ll_u2[lb:2*lb,:lb] = vhplvh_u1u1d1 + vhplvh_d1u1u1 + vhplvh_u1d1u1
        ll_u2[lb:2*lb,lb:2*lb] = vhplvh_u1u1u1
        ll_u2[2*lb:,:lb] = vhplvh_d1u1d1 + vhplvh_u1d1d1 + vhplvh_u1l1u1 + vhplvh_u1u1l1 + vhplvh_d1d1u1 + vhplvh_l1u1u1
        ll_u2[2*lb:,lb:2*lb] = vhplvh_d1u1u1 + vhplvh_u1d1u1 + vhplvh_u1u1d1
        ll_u2[2*lb:,2*lb:] = vhplvh_u1u1u1

        dmr_lu[:lb,:lb] = vhpr_l1u1

        dmr_ul[2*lb:,2*lb:] = vhpr_u1l1

        dlg_lu[:lb,:lb] = vhpgvh_d1l1u1 + vhpgvh_l1d1u1 + vhpgvh_l1u1d1
        dlg_lu[:lb,lb:2*lb] = vhpgvh_l1u1u1
        dlg_lu[lb:2*lb,:lb] = vhpgvh_l1l1u1

        dlg_ul[lb:2*lb,2*lb:] = vhpgvh_u1u1l1
        dlg_ul[2*lb:,lb:2*lb] = vhpgvh_u1l1l1
        dlg_ul[2*lb:,2*lb:] = vhpgvh_d1u1l1 + vhpgvh_u1d1l1 + vhpgvh_u1l1d1

        dll_lu[:lb,:lb] = vhplvh_d1l1u1 + vhplvh_l1d1u1 + vhplvh_l1u1d1
        dll_lu[:lb,lb:2*lb] = vhplvh_l1u1u1
        dll_lu[lb:2*lb,:lb] = vhplvh_l1l1u1

        dll_ul[lb:2*lb,2*lb:] = vhplvh_u1u1l1
        dll_ul[2*lb:,lb:2*lb] = vhplvh_u1l1l1
        dll_ul[2*lb:,2*lb:] = vhplvh_d1u1l1 + vhplvh_u1d1l1 + vhplvh_u1l1d1

        vh_u[2*lb:,:lb] = vh_u

        vh_l[:lb,2*lb:] = vh_l

    lg_l2[:,:] = -lg_u2.conjugate().transpose()
    ll_l2[:,:] = -ll_u2.conjugate().transpose()
    mr_d2 += np.identity(lb_mm, dtype=np.complex128) * (1+1j*1e-10)

    return ((mr_d2, mr_u2, mr_l2), (lg_d2, lg_u2, lg_l2), (ll_d2, ll_u2, ll_l2), (dmr_lu, dmr_ul), (dlg_lu, dlg_ul), (dll_lu, dll_ul), (vh_u, vh_l))

# @numba.njit("(c16[:,:], c16[:,:], c16[:,:], c16[:,:], c16[:,:], c16[:,:], c16[:,:], c16[:,:], i8)",
#             parallel=True,
#             cache=True,
#             nogil=True,
#             error_model="numpy")
def get_mm_obc_dense(
        vh_1: npt.NDArray[np.complex128],
        vh_2: npt.NDArray[np.complex128],
        pg_1: npt.NDArray[np.complex128],
        pg_2: npt.NDArray[np.complex128],
        pl_1: npt.NDArray[np.complex128],
        pl_2: npt.NDArray[np.complex128],
        pr_1: npt.NDArray[np.complex128],
        pr_2: npt.NDArray[np.complex128],
        nbc: int
) -> typing.Tuple[
    typing.Tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128], npt.NDArray[np.complex128]],
    typing.Tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128], npt.NDArray[np.complex128]],
    typing.Tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128], npt.NDArray[np.complex128]],
    typing.Tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128]],
    typing.Tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128]],
    typing.Tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128]],
    typing.Tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128]]
]:
    """
    Calculates the different correction terms for the scattering OBCs for the screened interaction calculations.
    In this version the blocks are not stacked, but the matrices are multiplied directly and inserted afterwards.

    Args:
        vh_1 (npt.NDArray[np.complex128]): Diagonal block of effective interaction
        vh_2 (npt.NDArray[np.complex128]): Off diagonal block of effective interaction
        pg_1 (npt.NDArray[np.complex128]): Diagonal block of greater polarization
        pg_2 (npt.NDArray[np.complex128]): Off diagonal block of greater polarization
        pl_1 (npt.NDArray[np.complex128]): Diagonal block of lesser polarization
        pl_2 (npt.NDArray[np.complex128]): Off diagonal block of lesser polarization
        pr_1 (npt.NDArray[np.complex128]): Diagonal block of retarded polarization
        pr_2 (npt.NDArray[np.complex128]): Off diagonal block of retarded polarization
        nbc (int): How block size changes after matrix multiplication

    Returns:
        typing.Tuple[
            typing.Tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128], npt.NDArray[np.complex128]],
            typing.Tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128], npt.NDArray[np.complex128]],
            typing.Tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128], npt.NDArray[np.complex128]],
            typing.Tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128]],
            typing.Tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128]],
            typing.Tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128]],
            typing.Tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128]]
            ]: mr_d/u/l, lg_d/u/l, ll_d/u/l, dg_lu/ul, dl_lu/ul, vh_u/l
    """
    # block size
    lb = vh_1.shape[0]
    # block size after mm
    lb_mm = nbc * lb
    # define the right blocks
    vh_d1 = vh_1
    vh_u1 = vh_2
    vh_l1 = vh_u1.conjugate().transpose()
    pg_d1 = pg_1
    pg_u1 = pg_2
    pg_l1 = -pg_u1.conjugate().transpose()
    pl_d1 = pl_1
    pl_u1 = pl_2
    pl_l1 = -pl_u1.conjugate().transpose()
    pr_d1 = pr_1
    pr_u1 = pr_2
    pr_l1 = pr_u1.transpose()

    # output matrices
    mr_d2 = np.empty((lb_mm,lb_mm), dtype=np.complex128)
    mr_u2 = np.zeros((lb_mm,lb_mm), dtype=np.complex128)
    mr_l2 = np.zeros((lb_mm,lb_mm), dtype=np.complex128)
    lg_d2 = np.empty((lb_mm,lb_mm), dtype=np.complex128)
    lg_u2 = np.zeros((lb_mm,lb_mm), dtype=np.complex128)
    lg_l2 = np.empty((lb_mm,lb_mm), dtype=np.complex128)
    ll_d2 = np.empty((lb_mm,lb_mm), dtype=np.complex128)
    ll_u2 = np.zeros((lb_mm,lb_mm), dtype=np.complex128)
    ll_l2 = np.empty((lb_mm,lb_mm), dtype=np.complex128)
    dmr_lu = np.zeros((lb_mm,lb_mm), dtype=np.complex128)
    dmr_ul = np.zeros((lb_mm,lb_mm), dtype=np.complex128)
    dlg_lu = np.zeros((lb_mm,lb_mm), dtype=np.complex128)
    dlg_ul = np.zeros((lb_mm,lb_mm), dtype=np.complex128)
    dll_lu = np.zeros((lb_mm,lb_mm), dtype=np.complex128)
    dll_ul = np.zeros((lb_mm,lb_mm), dtype=np.complex128)
    vh_u = np.zeros((lb_mm,lb_mm), dtype=np.complex128)
    vh_l = np.zeros((lb_mm,lb_mm), dtype=np.complex128)
    if nbc == 1:
        # fill output matrices
        # M^{r}\left(E\right)
        mr_d2[:,:] = -vh_d1 @ pr_d1 - vh_u1 @ pr_l1 -vh_l1 @ pr_u1
        mr_u2[:,:] = -vh_d1 @ pr_u1 - vh_u1 @ pr_d1
        mr_l2[:,:] = -vh_d1 @ pr_l1 - vh_l1 @ pr_d1

        # from L^{\lessgtr}\left(E\right)
        lg_d2[:,:] = (vh_l1 @ pg_d1 @ vh_u1 +
                vh_l1 @ pg_u1 @ vh_d1 +
                vh_d1 @ pg_l1 @ vh_u1 +
                vh_d1 @ pg_d1 @ vh_d1 +
                vh_d1 @ pg_u1 @ vh_l1 +
                vh_u1 @ pg_l1 @ vh_d1 +
                vh_u1 @ pg_d1 @ vh_l1)
        lg_u2[:,:] = (vh_l1 @ pg_u1 @ vh_u1 +
                vh_d1 @ pg_d1 @ vh_u1 +
                vh_d1 @ pg_u1 @ vh_d1 +
                vh_u1 @ pg_l1 @ vh_u1 +
                vh_u1 @ pg_d1 @ vh_d1)
        ll_d2[:,:] = (vh_l1 @ pl_d1 @ vh_u1 +
                vh_l1 @ pl_u1 @ vh_d1 +
                vh_d1 @ pl_l1 @ vh_u1 +
                vh_d1 @ pl_d1 @ vh_d1 +
                vh_d1 @ pl_u1 @ vh_l1 +
                vh_u1 @ pl_l1 @ vh_d1 +
                vh_u1 @ pl_d1 @ vh_l1)
        ll_u2[:,:] = (vh_l1 @ pl_u1 @ vh_u1 +
                vh_d1 @ pl_d1 @ vh_u1 +
                vh_d1 @ pl_u1 @ vh_d1 +
                vh_u1 @ pl_l1 @ vh_u1 +
                vh_u1 @ pl_d1 @ vh_d1)
        dmr_lu[:,:] = -vh_l1 @ pr_u1
        dmr_ul[:,:] = -vh_u1 @ pr_l1
        dlg_lu[:,:] = vh_l1 @ pg_d1 @ vh_u1 + vh_l1 @ pg_u1 @ vh_d1 + vh_d1 @ pg_l1 @ vh_u1
        dll_lu[:,:] = vh_l1 @ pl_d1 @ vh_u1 + vh_l1 @ pl_u1 @ vh_d1 + vh_d1 @ pl_l1 @ vh_u1
        dlg_ul[:,:] = vh_u1 @ pg_d1 @ vh_l1 + vh_u1 @ pg_l1 @ vh_d1 + vh_d1 @ pg_u1 @ vh_l1
        dll_ul[:,:] = vh_u1 @ pl_d1 @ vh_l1 + vh_u1 @ pl_l1 @ vh_d1 + vh_d1 @ pl_u1 @ vh_l1
        vh_u[:,:] = vh_u1
        vh_l[:,:] = vh_l1
    elif nbc == 2:
        # compute multiplications
        vhpr_d1d1 = -vh_d1 @ pr_d1
        vhpr_d1u1 = -vh_d1 @ pr_u1
        vhpr_d1l1 = -vh_d1 @ pr_l1
        vhpr_u1d1 = -vh_u1 @ pr_d1
        vhpr_u1u1 = -vh_u1 @ pr_u1
        vhpr_u1l1 = -vh_u1 @ pr_l1
        vhpr_l1d1 = -vh_l1 @ pr_d1
        vhpr_l1u1 = -vh_l1 @ pr_u1
        vhpr_l1l1 = -vh_l1 @ pr_l1

        vhpg_d1d1 = vh_d1 @ pg_d1
        vhpg_d1u1 = vh_d1 @ pg_u1
        vhpg_d1l1 = vh_d1 @ pg_l1
        vhpg_u1d1 = vh_u1 @ pg_d1
        vhpg_u1u1 = vh_u1 @ pg_u1
        vhpg_u1l1 = vh_u1 @ pg_l1
        vhpg_l1d1 = vh_l1 @ pg_d1
        vhpg_l1u1 = vh_l1 @ pg_u1
        vhpg_l1l1 = vh_l1 @ pg_l1
        vhpgvh_d1d1d1 = vhpg_d1d1 @ vh_d1
        vhpgvh_d1d1u1 = vhpg_d1d1 @ vh_u1
        vhpgvh_d1d1l1 = vhpg_d1d1 @ vh_l1
        vhpgvh_d1u1d1 = vhpg_d1u1 @ vh_d1
        vhpgvh_d1u1u1 = vhpg_d1u1 @ vh_u1
        vhpgvh_d1u1l1 = vhpg_d1u1 @ vh_l1
        vhpgvh_d1l1d1 = vhpg_d1l1 @ vh_d1
        vhpgvh_d1l1u1 = vhpg_d1l1 @ vh_u1
        vhpgvh_u1d1d1 = vhpg_u1d1 @ vh_d1
        vhpgvh_u1d1u1 = vhpg_u1d1 @ vh_u1
        vhpgvh_u1d1l1 = vhpg_u1d1 @ vh_l1
        vhpgvh_u1u1d1 = vhpg_u1u1 @ vh_d1
        vhpgvh_u1u1u1 = vhpg_u1u1 @ vh_u1
        vhpgvh_u1u1l1 = vhpg_u1u1 @ vh_l1
        vhpgvh_u1l1d1 = vhpg_u1l1 @ vh_d1
        vhpgvh_u1l1u1 = vhpg_u1l1 @ vh_u1
        vhpgvh_u1l1l1 = vhpg_u1l1 @ vh_l1
        vhpgvh_l1d1d1 = vhpg_l1d1 @ vh_d1
        vhpgvh_l1d1u1 = vhpg_l1d1 @ vh_u1
        vhpgvh_l1u1d1 = vhpg_l1u1 @ vh_d1
        vhpgvh_l1u1u1 = vhpg_l1u1 @ vh_u1
        vhpgvh_l1u1l1 = vhpg_l1u1 @ vh_l1
        vhpgvh_l1l1u1 = vhpg_l1l1 @ vh_u1
        vhpl_d1d1 = vh_d1 @ pl_d1
        vhpl_d1u1 = vh_d1 @ pl_u1
        vhpl_d1l1 = vh_d1 @ pl_l1
        vhpl_u1d1 = vh_u1 @ pl_d1
        vhpl_u1u1 = vh_u1 @ pl_u1
        vhpl_u1l1 = vh_u1 @ pl_l1
        vhpl_l1d1 = vh_l1 @ pl_d1
        vhpl_l1u1 = vh_l1 @ pl_u1
        vhpl_l1l1 = vh_l1 @ pl_l1
        vhplvh_d1d1d1 = vhpl_d1d1 @ vh_d1
        vhplvh_d1d1u1 = vhpl_d1d1 @ vh_u1
        vhplvh_d1d1l1 = vhpl_d1d1 @ vh_l1
        vhplvh_d1u1d1 = vhpl_d1u1 @ vh_d1
        vhplvh_d1u1u1 = vhpl_d1u1 @ vh_u1
        vhplvh_d1u1l1 = vhpl_d1u1 @ vh_l1
        vhplvh_d1l1d1 = vhpl_d1l1 @ vh_d1
        vhplvh_d1l1u1 = vhpl_d1l1 @ vh_u1
        vhplvh_u1d1d1 = vhpl_u1d1 @ vh_d1
        vhplvh_u1d1u1 = vhpl_u1d1 @ vh_u1
        vhplvh_u1d1l1 = vhpl_u1d1 @ vh_l1
        vhplvh_u1u1d1 = vhpl_u1u1 @ vh_d1
        vhplvh_u1u1u1 = vhpl_u1u1 @ vh_u1
        vhplvh_u1u1l1 = vhpl_u1u1 @ vh_l1
        vhplvh_u1l1d1 = vhpl_u1l1 @ vh_d1
        vhplvh_u1l1u1 = vhpl_u1l1 @ vh_u1
        vhplvh_u1l1l1 = vhpl_u1l1 @ vh_l1
        vhplvh_l1d1d1 = vhpl_l1d1 @ vh_d1
        vhplvh_l1d1u1 = vhpl_l1d1 @ vh_u1
        vhplvh_l1u1d1 = vhpl_l1u1 @ vh_d1
        vhplvh_l1u1u1 = vhpl_l1u1 @ vh_u1
        vhplvh_l1u1l1 = vhpl_l1u1 @ vh_l1
        vhplvh_l1l1u1 = vhpl_l1l1 @ vh_u1

        # fill output matrices
        # M^{r}\left(E\right)
        mr_d2[:lb,:lb] = vhpr_d1d1 + vhpr_u1l1 + vhpr_l1u1
        mr_d2[:lb,lb:] = vhpr_d1u1 + vhpr_u1d1
        mr_d2[lb:,:lb] = vhpr_d1l1 + vhpr_l1d1
        mr_d2[lb:,lb:] = vhpr_d1d1 + vhpr_l1u1 + vhpr_u1l1

        mr_u2[:lb,:lb] = vhpr_u1u1
        mr_u2[lb:,:lb] = vhpr_d1u1 + vhpr_u1d1
        mr_u2[lb:,lb:] = vhpr_u1u1

        mr_l2[:lb,:lb] = vhpr_l1l1
        mr_l2[:lb,lb:] = vhpr_d1l1 + vhpr_l1d1
        mr_l2[lb:,lb:] = vhpr_l1l1

        # L^{\lessgtr}\left(E\right)
        lg_d2[:lb,:lb] = vhpgvh_d1l1u1 + vhpgvh_l1d1u1 + vhpgvh_l1u1d1 + vhpgvh_d1d1d1 + vhpgvh_u1l1d1 + vhpgvh_d1u1l1 + vhpgvh_u1d1l1
        lg_d2[:lb,lb:] = vhpgvh_l1u1u1 + vhpgvh_u1u1l1 + vhpgvh_d1d1u1 + vhpgvh_u1l1u1 + vhpgvh_d1u1d1 + vhpgvh_u1d1d1
        lg_d2[lb:,:lb] = vhpgvh_l1l1u1 + vhpgvh_u1l1l1 + vhpgvh_d1d1l1 + vhpgvh_l1u1l1 + vhpgvh_d1l1d1 + vhpgvh_l1d1d1
        lg_d2[lb:,lb:] = vhpgvh_d1u1l1 + vhpgvh_u1d1l1 + vhpgvh_u1l1d1 + vhpgvh_d1d1d1 + vhpgvh_l1u1d1 + vhpgvh_d1l1u1 + vhpgvh_l1d1u1

        lg_u2[:lb,:lb] = vhpgvh_u1u1d1 + vhpgvh_d1u1u1 + vhpgvh_u1d1u1
        lg_u2[:lb,lb:] = vhpgvh_u1u1u1
        lg_u2[lb:,:lb] = vhpgvh_d1u1d1 + vhpgvh_u1d1d1 + vhpgvh_u1l1u1 + vhpgvh_u1u1l1 + vhpgvh_d1d1u1 + vhpgvh_l1u1u1
        lg_u2[lb:,lb:] = vhpgvh_d1u1u1 + vhpgvh_u1d1u1 + vhpgvh_u1u1d1

        ll_d2[:lb,:lb] = vhplvh_d1l1u1 + vhplvh_l1d1u1 + vhplvh_l1u1d1 + vhplvh_d1d1d1 + vhplvh_u1l1d1 + vhplvh_d1u1l1 + vhplvh_u1d1l1
        ll_d2[:lb,lb:] = vhplvh_l1u1u1 + vhplvh_u1u1l1 + vhplvh_d1d1u1 + vhplvh_u1l1u1 + vhplvh_d1u1d1 + vhplvh_u1d1d1
        ll_d2[lb:,:lb] = vhplvh_l1l1u1 + vhplvh_u1l1l1 + vhplvh_d1d1l1 + vhplvh_l1u1l1 + vhplvh_d1l1d1 + vhplvh_l1d1d1
        ll_d2[lb:,lb:] = vhplvh_d1u1l1 + vhplvh_u1d1l1 + vhplvh_u1l1d1 + vhplvh_d1d1d1 + vhplvh_l1u1d1 + vhplvh_d1l1u1 + vhplvh_l1d1u1

        ll_u2[:lb,:lb] = vhplvh_u1u1d1 + vhplvh_d1u1u1 + vhplvh_u1d1u1
        ll_u2[:lb,lb:] = vhplvh_u1u1u1
        ll_u2[lb:,:lb] = vhplvh_d1u1d1 + vhplvh_u1d1d1 + vhplvh_u1l1u1 + vhplvh_u1u1l1 + vhplvh_d1d1u1 + vhplvh_l1u1u1
        ll_u2[lb:,lb:] = vhplvh_d1u1u1 + vhplvh_u1d1u1 + vhplvh_u1u1d1

        dmr_lu[:lb,:lb] = vhpr_l1u1
        dmr_ul[lb:,lb:] = vhpr_u1l1
    
        dlg_lu[:lb,:lb] = vhpgvh_d1l1u1 + vhpgvh_l1d1u1 + vhpgvh_l1u1d1
        dlg_lu[:lb,lb:] = vhpgvh_l1u1u1
        dlg_lu[lb:,:lb] = vhpgvh_l1l1u1

        dlg_ul[:lb,lb:] = vhpgvh_u1u1l1
        dlg_ul[lb:,:lb] = vhpgvh_u1l1l1
        dlg_ul[lb:,lb:] = vhpgvh_d1u1l1 + vhpgvh_u1d1l1 + vhpgvh_u1l1d1

        dll_lu[:lb,:lb] = vhplvh_d1l1u1 + vhplvh_l1d1u1 + vhplvh_l1u1d1
        dll_lu[:lb,lb:] = vhplvh_l1u1u1
        dll_lu[lb:,:lb] = vhplvh_l1l1u1

        dll_ul[:lb,lb:] = vhplvh_u1u1l1
        dll_ul[lb:,:lb] = vhplvh_u1l1l1
        dll_ul[lb:,lb:] = vhplvh_d1u1l1 + vhplvh_u1d1l1 + vhplvh_u1l1d1

        vh_u[lb:,:lb] = vh_u1
        vh_l[:lb,lb:] = vh_l1
    elif nbc == 3:
        # compute multiplications
        vhpr_d1d1 = -vh_d1 @ pr_d1
        vhpr_d1u1 = -vh_d1 @ pr_u1
        vhpr_d1l1 = -vh_d1 @ pr_l1
        vhpr_u1d1 = -vh_u1 @ pr_d1
        vhpr_u1u1 = -vh_u1 @ pr_u1
        vhpr_u1l1 = -vh_u1 @ pr_l1
        vhpr_l1d1 = -vh_l1 @ pr_d1
        vhpr_l1u1 = -vh_l1 @ pr_u1
        vhpr_l1l1 = -vh_l1 @ pr_l1

        vhpg_d1d1 = vh_d1 @ pg_d1
        vhpg_d1u1 = vh_d1 @ pg_u1
        vhpg_d1l1 = vh_d1 @ pg_l1
        vhpg_u1d1 = vh_u1 @ pg_d1
        vhpg_u1u1 = vh_u1 @ pg_u1
        vhpg_u1l1 = vh_u1 @ pg_l1
        vhpg_l1d1 = vh_l1 @ pg_d1
        vhpg_l1u1 = vh_l1 @ pg_u1
        vhpg_l1l1 = vh_l1 @ pg_l1
        vhpgvh_d1d1d1 = vhpl_d1d1 @ vh_d1
        vhpgvh_d1d1u1 = vhpl_d1d1 @ vh_u1
        vhpgvh_d1d1l1 = vhpl_d1d1 @ vh_l1
        vhpgvh_d1u1d1 = vhpl_d1u1 @ vh_d1
        vhpgvh_d1u1u1 = vhpl_d1u1 @ vh_u1
        vhpgvh_d1u1l1 = vhpl_d1u1 @ vh_l1
        vhpgvh_d1l1d1 = vhpl_d1l1 @ vh_d1
        vhpgvh_d1l1u1 = vhpl_d1l1 @ vh_u1
        vhpgvh_d1l1l1 = vhpl_d1l1 @ vh_l1
        vhpgvh_u1d1d1 = vhpl_u1d1 @ vh_d1
        vhpgvh_u1d1u1 = vhpl_u1d1 @ vh_u1
        vhpgvh_u1d1l1 = vhpl_u1d1 @ vh_l1
        vhpgvh_u1u1d1 = vhpl_u1u1 @ vh_d1
        vhpgvh_u1u1u1 = vhpl_u1u1 @ vh_u1
        vhpgvh_u1u1l1 = vhpl_u1u1 @ vh_l1
        vhpgvh_u1l1d1 = vhpl_u1l1 @ vh_d1
        vhpgvh_u1l1u1 = vhpl_u1l1 @ vh_u1
        vhpgvh_u1l1l1 = vhpl_u1l1 @ vh_l1
        vhpgvh_l1d1d1 = vhpl_l1d1 @ vh_d1
        vhpgvh_l1d1u1 = vhpl_l1d1 @ vh_u1
        vhpgvh_l1d1l1 = vhpl_l1d1 @ vh_l1
        vhpgvh_l1u1d1 = vhpl_l1u1 @ vh_d1
        vhpgvh_l1u1u1 = vhpl_l1u1 @ vh_u1
        vhpgvh_l1u1l1 = vhpl_l1u1 @ vh_l1
        vhpgvh_l1l1d1 = vhpl_l1l1 @ vh_d1
        vhpgvh_l1l1u1 = vhpl_l1l1 @ vh_u1
        vhpl_d1d1 = vh_d1 @ pl_d1
        vhpl_d1u1 = vh_d1 @ pl_u1
        vhpl_d1l1 = vh_d1 @ pl_l1
        vhpl_u1d1 = vh_u1 @ pl_d1
        vhpl_u1u1 = vh_u1 @ pl_u1
        vhpl_u1l1 = vh_u1 @ pl_l1
        vhpl_l1d1 = vh_l1 @ pl_d1
        vhpl_l1u1 = vh_l1 @ pl_u1
        vhpl_l1l1 = vh_l1 @ pl_l1
        vhplvh_d1d1d1 = vhpl_d1d1 @ vh_d1
        vhplvh_d1d1u1 = vhpl_d1d1 @ vh_u1
        vhplvh_d1d1l1 = vhpl_d1d1 @ vh_l1
        vhplvh_d1u1d1 = vhpl_d1u1 @ vh_d1
        vhplvh_d1u1u1 = vhpl_d1u1 @ vh_u1
        vhplvh_d1u1l1 = vhpl_d1u1 @ vh_l1
        vhplvh_d1l1d1 = vhpl_d1l1 @ vh_d1
        vhplvh_d1l1u1 = vhpl_d1l1 @ vh_u1
        vhplvh_d1l1l1 = vhpl_d1l1 @ vh_l1
        vhplvh_u1d1d1 = vhpl_u1d1 @ vh_d1
        vhplvh_u1d1u1 = vhpl_u1d1 @ vh_u1
        vhplvh_u1d1l1 = vhpl_u1d1 @ vh_l1
        vhplvh_u1u1d1 = vhpl_u1u1 @ vh_d1
        vhplvh_u1u1u1 = vhpl_u1u1 @ vh_u1
        vhplvh_u1u1l1 = vhpl_u1u1 @ vh_l1
        vhplvh_u1l1d1 = vhpl_u1l1 @ vh_d1
        vhplvh_u1l1u1 = vhpl_u1l1 @ vh_u1
        vhplvh_u1l1l1 = vhpl_u1l1 @ vh_l1
        vhplvh_l1d1d1 = vhpl_l1d1 @ vh_d1
        vhplvh_l1d1u1 = vhpl_l1d1 @ vh_u1
        vhplvh_l1d1l1 = vhpl_l1d1 @ vh_l1
        vhplvh_l1u1d1 = vhpl_l1u1 @ vh_d1
        vhplvh_l1u1u1 = vhpl_l1u1 @ vh_u1
        vhplvh_l1u1l1 = vhpl_l1u1 @ vh_l1
        vhplvh_l1l1d1 = vhpl_l1l1 @ vh_d1
        vhplvh_l1l1u1 = vhpl_l1l1 @ vh_u1

        # fill output matrices
        # M^{r}\left(E\right)
        mr_d2[:lb,:lb] = vhpr_d1d1 + vhpr_l1u1 + vhpr_u1l1
        mr_d2[:lb,lb:2*lb] = vhpr_d1u1 + vhpr_u1d1
        mr_d2[:lb,2*lb:] = vhpr_u1u1
        mr_d2[lb:2*lb,:lb] = vhpr_d1l1 + vhpr_l1d1
        mr_d2[lb:2*lb,lb:2*lb] = vhpr_d1d1 + vhpr_l1u1 + vhpr_u1l1
        mr_d2[lb:2*lb,2*lb:] = vhpr_d1u1 + vhpr_u1d1
        mr_d2[2*lb:,:lb] = vhpr_l1l1
        mr_d2[2*lb:,lb:2*lb] = vhpr_d1l1 + vhpr_l1d1
        mr_d2[2*lb:,2*lb:] = vhpr_d1d1 + vhpr_l1u1 + vhpr_u1l1

        mr_u2[lb:2*lb,:lb] = vhpr_u1u1
        mr_u2[2*lb:,:lb] = vhpr_d1u1 + vhpr_u1d1
        mr_u2[2*lb:,lb:2*lb] = vhpr_u1u1

        mr_l2[:lb,lb:2*lb] = vhpr_l1l1
        mr_l2[:lb,2*lb:] = vhpr_d1l1 + vhpr_l1d1
        mr_l2[lb:2*lb,2*lb:] = vhpr_l1l1

        lg_d2[:lb,:lb] = vhpgvh_d1l1u1 + vhpgvh_l1d1u1 + vhpgvh_l1u1d1 + vhpgvh_d1d1d1 + vhpgvh_u1l1d1 + vhpgvh_d1u1l1 + vhpgvh_u1d1l1
        lg_d2[:lb,lb:2*lb] = vhpgvh_l1u1u1 + vhpgvh_u1u1l1 + vhpgvh_d1d1u1 + vhpgvh_u1l1u1 + vhpgvh_d1u1d1 + vhpgvh_u1d1d1
        lg_d2[:lb,2*lb:] = vhpgvh_u1u1d1 + vhpgvh_d1u1d1 + vhpgvh_u1d1u1
        lg_d2[lb:2*lb,:lb] = vhpgvh_l1l1u1 + vhpgvh_d1l1d1 + vhpgvh_l1d1d1 + vhpgvh_d1d1l1 + vhpgvh_l1u1l1 + vhpgvh_u1l1l1
        lg_d2[lb:2*lb,lb:2*lb] = vhpgvh_d1l1u1 + vhpgvh_l1d1u1 + vhpgvh_d1u1l1 + vhpgvh_u1d1l1 + vhpgvh_d1d1d1 + vhpgvh_l1u1d1 + vhpgvh_u1l1d1
        lg_d2[lb:2*lb,2*lb:] = vhpgvh_u1u1l1 + vhpgvh_d1u1d1 + vhpgvh_u1d1d1 + vhpgvh_d1d1u1 + vhpgvh_l1u1u1 + vhpgvh_u1l1u1
        lg_d2[2*lb:,:lb] = vhpgvh_l1l1d1 + vhpgvh_d1l1l1 + vhpgvh_l1d1l1
        lg_d2[2*lb:,lb:2*lb] = vhpgvh_l1l1u1 + vhpgvh_u1l1l1 + vhpgvh_d1d1l1 + vhpgvh_l1u1l1 + vhpgvh_d1l1d1 + vhpgvh_l1d1d1
        lg_d2[2*lb:,2*lb:] = vhpgvh_d1u1l1 + vhpgvh_u1d1l1 + vhpgvh_u1l1d1 + vhpgvh_d1d1d1 + vhpgvh_l1u1d1 + vhpgvh_d1l1u1 + vhpgvh_l1d1u1

        lg_u2[:lb,:lb] = vhpgvh_u1u1u1
        lg_u2[lb:2*lb,:lb] = vhpgvh_u1u1d1 + vhpgvh_d1u1d1 + vhpgvh_u1d1u1
        lg_u2[lb:2*lb,lb:2*lb] = vhpgvh_u1u1u1
        lg_u2[2*lb:,:lb] = vhpgvh_d1u1d1 + vhpgvh_u1d1d1 + vhpgvh_u1l1u1 + vhpgvh_u1u1l1 + vhpgvh_d1d1u1 + vhpgvh_l1u1u1
        lg_u2[2*lb:,lb:2*lb] = vhpgvh_d1u1d1 + vhpgvh_u1d1u1 + vhpgvh_u1u1d1
        lg_u2[2*lb:,2*lb:] = vhpgvh_u1u1u1

        ll_d2[:lb,:lb] = vhplvh_d1l1u1 + vhplvh_l1d1u1 + vhplvh_l1u1d1 + vhplvh_d1d1d1 + vhplvh_u1l1d1 + vhplvh_d1u1l1 + vhplvh_u1d1l1
        ll_d2[:lb,lb:2*lb] = vhplvh_l1u1u1 + vhplvh_u1u1l1 + vhplvh_d1d1u1 + vhplvh_u1l1u1 + vhplvh_d1u1d1 + vhplvh_u1d1d1
        ll_d2[:lb,2*lb:] = vhplvh_u1u1d1 + vhplvh_d1u1u1 + vhplvh_u1d1u1
        ll_d2[lb:2*lb,:lb] = vhplvh_l1l1u1 + vhplvh_d1l1d1 + vhplvh_l1d1d1 + vhplvh_d1d1l1 + vhplvh_l1u1l1 + vhplvh_u1l1l1
        ll_d2[lb:2*lb,lb:2*lb] = vhplvh_d1l1u1 + vhplvh_l1d1u1 + vhplvh_d1u1l1 + vhplvh_u1d1l1 + vhplvh_d1d1d1 + vhplvh_l1u1d1 + vhplvh_u1l1d1
        ll_d2[lb:2*lb,2*lb:] = vhplvh_u1u1l1 + vhplvh_d1u1d1 + vhplvh_u1d1d1 + vhplvh_d1d1u1 + vhplvh_l1u1u1 + vhplvh_u1l1u1
        ll_d2[2*lb:,:lb] = vhplvh_l1l1d1 + vhplvh_d1l1l1 + vhplvh_l1d1l1
        ll_d2[2*lb:,lb:2*lb] = vhplvh_l1l1u1 + vhplvh_u1l1l1 + vhplvh_d1d1l1 + vhplvh_l1u1l1 + vhplvh_d1l1d1 + vhplvh_l1d1d1
        ll_d2[2*lb:,2*lb:] = vhplvh_d1u1l1 + vhplvh_u1d1l1 + vhplvh_u1l1d1 + vhplvh_d1d1d1 + vhplvh_l1u1d1 + vhplvh_d1l1u1 + vhplvh_l1d1u1

        ll_u2[:lb,:lb] = vhplvh_u1u1u1
        ll_u2[lb:2*lb,:lb] = vhplvh_u1u1d1 + vhplvh_d1u1u1 + vhplvh_u1d1u1
        ll_u2[lb:2*lb,lb:2*lb] = vhplvh_u1u1u1
        ll_u2[2*lb:,:lb] = vhplvh_d1u1d1 + vhplvh_u1d1d1 + vhplvh_u1l1u1 + vhplvh_u1u1l1 + vhplvh_d1d1u1 + vhplvh_l1u1u1
        ll_u2[2*lb:,lb:2*lb] = vhplvh_d1u1u1 + vhplvh_u1d1u1 + vhplvh_u1u1d1
        ll_u2[2*lb:,2*lb:] = vhplvh_u1u1u1

        dmr_lu[:lb,:lb] = vhpr_l1u1

        dmr_ul[2*lb:,2*lb:] = vhpr_u1l1

        dlg_lu[:lb,:lb] = vhpgvh_d1l1u1 + vhpgvh_l1d1u1 + vhpgvh_l1u1d1
        dlg_lu[:lb,lb:2*lb] = vhpgvh_l1u1u1
        dlg_lu[lb:2*lb,:lb] = vhpgvh_l1l1u1

        dlg_ul[lb:2*lb,2*lb:] = vhpgvh_u1u1l1
        dlg_ul[2*lb:,lb:2*lb] = vhpgvh_u1l1l1
        dlg_ul[2*lb:,2*lb:] = vhpgvh_d1u1l1 + vhpgvh_u1d1l1 + vhpgvh_u1l1d1

        dll_lu[:lb,:lb] = vhplvh_d1l1u1 + vhplvh_l1d1u1 + vhplvh_l1u1d1
        dll_lu[:lb,lb:2*lb] = vhplvh_l1u1u1
        dll_lu[lb:2*lb,:lb] = vhplvh_l1l1u1

        dll_ul[lb:2*lb,2*lb:] = vhplvh_u1u1l1
        dll_ul[2*lb:,lb:2*lb] = vhplvh_u1l1l1
        dll_ul[2*lb:,2*lb:] = vhplvh_d1u1l1 + vhplvh_u1d1l1 + vhplvh_u1l1d1

        vh_u[2*lb:,:lb] = vh_u

        vh_l[:lb,2*lb:] = vh_l

    lg_l2[:,:] = -lg_u2.conjugate().transpose()
    ll_l2[:,:] = -ll_u2.conjugate().transpose()
    mr_d2 = mr_d2 + np.identity(lb_mm, dtype=np.complex128) * (1+1j*1e-10)

    return ((mr_d2, mr_u2, mr_l2), (lg_d2, lg_u2, lg_l2), (ll_d2, ll_u2, ll_l2), (dmr_lu, dmr_ul), (dlg_lu, dlg_ul), (dll_lu, dll_ul), (vh_u, vh_l))


def get_dl_obc_start(
    mr_x: np.ndarray,
    xr_d: np.ndarray,
    xr_d_ct: np.ndarray,
    lx_d: np.ndarray,
    lx_o: np.ndarray
) -> typing.Tuple[np.ndarray,
                  np.ndarray]:
    """Helper function to replace symmetric computations 
    for both greater and lesser obc corrections.

    Args:
        mr_x (np.ndarray):
        xr_d (np.ndarray):
        xr_d_ct (np.ndarray):
        lx_d (np.ndarray): different between lesser/greater
        lx_o (np.ndarray): different between lesser/greater

    Returns:
        typing.Tuple[np.ndarray, np.ndarray]: fx and ax_diff
    """
    # honestly no clue why
    # this function does what it does
    ax = mr_x @ xr_d @ lx_o

    # only difference between ax and ax^H is needed
    ax_diff = ax - ax.conjugate().transpose()

    fx = xr_d @ (lx_d - ax_diff) @ xr_d_ct
    return ax_diff, fx


def get_dl_obc_end(
    eivec: np.ndarray,
    eivec_ct: np.ndarray,
    ieivec: np.ndarray,
    ieivec_ct: np.ndarray,
    eival_sq: np.ndarray,
    sl_x: slice,
    fx: np.ndarray,
    mr_x: np.ndarray,
    mr_x_ct: np.ndarray,
    xr_d: np.ndarray,
    xr_d_ct: np.ndarray,
    ax_diff: np.ndarray,
    ref_iteration: int
) -> np.ndarray:
    """Helper function to replace symmetric computations 
    for both greater and lesser obc corrections.

    Args:
        eivec (np.ndarray):
        eivec_ct (np.ndarray):
        ieivec (np.ndarray):
        ieivec_ct (np.ndarray):
        eival_sq (np.ndarray):
        sl_x (slice):
        fx (np.ndarray): different between lesser/greater
        mr_x (np.ndarray):
        mr_x_ct (np.ndarray):
        xr_d (np.ndarray):
        xr_d_ct (np.ndarray):
        ax_diff (np.ndarray): different between lesser/greater
        ref_iteration (int): number of iterations for the refinement

    Returns:
        np.ndarray:
    """
    yx_d = np.divide(ieivec @ fx[sl_x,sl_x] @ ieivec_ct, 1 - eival_sq)
    wx_d = eivec @ yx_d @ eivec_ct
    for i in range(ref_iteration):
        wx_d = fx[sl_x,sl_x] + xr_d[sl_x,:] @ mr_x[:,sl_x] @ wx_d @ mr_x_ct[sl_x,:] @ xr_d_ct[:,sl_x]

    dlx_d = mr_x[:,sl_x] @ wx_d @ mr_x_ct[sl_x,:] - ax_diff
    return dlx_d

def get_dl_obc(
    xr_d: np.ndarray,
    lg_d: np.ndarray,
    lg_o: np.ndarray,
    ll_d: np.ndarray,
    ll_o: np.ndarray,
    mr_x: np.ndarray,
    blk:  str
) -> typing.Tuple[np.ndarray,
                  np.ndarray]:
    """Calculates open boundary corrections for lg and ll
    Todo find out why it does what it does

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
    mr_x_max = np.max(np.abs(mr_x))
    rows, cols = np.where(np.abs(mr_x) > mr_x_max / 1e8)
    #rows, cols = mr_x.nonzero()
    if(not rows.size):
        return np.nan, np.nan
    # conjugate transpose of mr/xr
    mr_x_ct = mr_x.conjugate().transpose()
    xr_d_ct = xr_d.conjugate().transpose()

    ag = mr_x @ xr_d @ lg_o
    al = mr_x @ xr_d @ ll_o

    # only difference between ax and ax^H is needed
    ag_diff = ag - ag.conjugate().transpose()
    al_diff = al - al.conjugate().transpose()

    fg = xr_d @ (lg_d - ag_diff) @ xr_d_ct
    fl = xr_d @ (ll_d - al_diff) @ xr_d_ct
    ag_diff, fg = get_dl_obc_start(
                                mr_x,
                                xr_d,
                                xr_d_ct,
                                lg_d,
                                lg_o
                                )
    al_diff, fl = get_dl_obc_start(
                                mr_x,
                                xr_d,
                                xr_d_ct,
                                ll_d,
                                ll_o
                                )

    # case for the left/right (start/end) block
    # differentiates between which block to look at
    if blk == "L":
        idx_max = np.max([np.max(rows), lb - np.min(cols)])
        ip = lb - idx_max
        sl_x = slice(ip,lb)
        sl_y = slice(0,idx_max+1)
    elif blk == "R":
        idx_max = np.max([np.max(cols), lb - np.min(rows)])
        ip = lb - idx_max
        sl_x = slice(0,idx_max+1)
        sl_y = slice(ip,lb)
    else:
        raise ValueError(
        "Argument error, type input not possible")

    ar = xr_d[sl_x,sl_y] @ mr_x[sl_y,sl_x]
    # add imaginary part to stabilize
    # ar = ar + np.identity(ar.shape[0])*1j*1e-4

    # eigen values and eigen vectors
    eival, eivec = np.linalg.eig(ar)

    # conjugate/transpose/abs square
    eivec_ct = eivec.conjugate().transpose()
    ieivec = np.linalg.inv(eivec)
    ieivec_ct = ieivec.conjugate().transpose()
    eival_sq = np.diag(eival) @ np.diag(eival).conjugate()

    # greater component
    # yg_d = np.divide(ieivec @ fg[sl_x,sl_x] @ ieivec_ct, 1 - eival_sq)
    # wg_d = eivec @ yg_d @ eivec_ct
    # wg_d = fg[sl_x,sl_x] + xr_d[sl_x,:] @ mr_x[:,sl_x] @ wg_d @ mr_x_ct[sl_x,:] @ xr_d_ct[:,sl_x]

    # dlg_d = mr_x[:,sl_x] @ wg_d @ mr_x_ct[sl_x,:] - ag_diff
    dlg_d = get_dl_obc_end(eivec,
                        eivec_ct,
                        ieivec,
                        ieivec_ct,
                        eival_sq,
                        sl_x,
                        fg,
                        mr_x,
                        mr_x_ct,
                        xr_d,
                        xr_d_ct,
                        ag_diff,
                        ref_iteration)

    # lesser component
    # yl_d = np.divide(ieivec @ fl[sl_x,sl_x] @ ieivec_ct, 1 - eival_sq)
    # wl_d = eivec @ yl_d @ eivec_ct
    # wl_d = fl[sl_x,sl_x] + xr_d[sl_x,:] @ mr_x[:,sl_x] @ wl_d @ mr_x_ct[sl_x,:] @ xr_d_ct[:,sl_x]

    # dll_d = mr_x[:,sl_x] @ wl_d @ mr_x_ct[sl_x,:] - al_diff
    dll_d = get_dl_obc_end(eivec,
                        eivec_ct,
                        ieivec,
                        ieivec_ct,
                        eival_sq,
                        sl_x,
                        fl,
                        mr_x,
                        mr_x_ct,
                        xr_d,
                        xr_d_ct,
                        al_diff,
                        ref_iteration)

    return dlg_d, dll_d

def get_dl_obc_alt(
    xr_d: np.ndarray,
    lg_d: np.ndarray,
    lg_o: np.ndarray,
    ll_d: np.ndarray,
    ll_o: np.ndarray,
    mr_x: np.ndarray,
    blk:  str
) -> typing.Tuple[np.ndarray,
                  np.ndarray]:
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
        idx_max = np.max([np.max(rows), lb - np.min(cols)])
        ip = lb - idx_max
        sl_x = slice(ip,lb)
        sl_y = slice(0,idx_max+1)
    elif blk == "R":
        idx_max = np.max([np.max(cols), lb - np.min(rows)])
        ip = lb - idx_max
        sl_x = slice(0,idx_max+1)
        sl_y = slice(ip,lb)
    else:
        raise ValueError(
        "Argument error, type input not possible")

    ar = xr_d[sl_x,sl_y] @ mr_x[sl_y,sl_x]
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
    yg_d = np.divide(ieivec @ fg[sl_x,sl_x] @ ieivec_ct, 1 - eival_sq)
    wg_d = eivec @ yg_d @ eivec_ct
    xrmr_dx_s = xr_d[sl_x,:] @ mr_x[:,sl_x]
    mrxr_ct_xd_s = mr_x_ct[sl_x,:] @ xr_d_ct[:,sl_x]
    for i in range(ref_iteration):
        wg_d = fg[sl_x,sl_x] + xrmr_dx_s @ wg_d @ mrxr_ct_xd_s

    dlg_d = mr_x[:,sl_x] @ wg_d @ mr_x_ct[sl_x,:] - ag_diff

    # lesser component
    yl_d = np.divide(ieivec @ fl[sl_x,sl_x] @ ieivec_ct, 1 - eival_sq)
    wl_d = eivec @ yl_d @ eivec_ct
    for i in range(ref_iteration):
        wl_d = fl[sl_x,sl_x] + xrmr_dx_s @ wl_d @ mrxr_ct_xd_s

    dll_d = mr_x[:,sl_x] @ wl_d @ mr_x_ct[sl_x,:] - al_diff

    return dlg_d, dll_d
