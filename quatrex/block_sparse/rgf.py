from typing import Union

import numpy as np
import numpy.linalg as npla

from bsr import bsr
from vbsr import vbsr


def _init_bsr(M: bsr) -> tuple[bsr, int]:
    """Initializes G for a VBSR matrix."""
    if M.blocksize[0] != M.blocksize[1]:
        raise ValueError("Blocks must be square.")

    num_blocks = M.shape[0] // M.blocksize[0]
    G = bsr.diag(
        [np.ones(M.blocksize, M.dtype) for __ in range(num_blocks)],
        blocksize=M.blocksize,
    )

    return G, num_blocks


def _init_vbsr(M: vbsr) -> tuple[vbsr, int]:
    """Initializes G for a VBSR matrix."""
    G = vbsr.diag([np.ones((s, s), M.dtype) for s in M.blocksizes])
    return G, M.num_blocks


def rgf(M: Union[bsr, vbsr]) -> Union[bsr, vbsr]:
    """Applies the recursive Green's function method."""
    if isinstance(M, bsr):
        G, num_blocks = _init_bsr(M)
    elif isinstance(M, vbsr):
        G, num_blocks = _init_vbsr(M)
    else:
        raise TypeError("Matrix must be BSR or VBSR.")

    G.set_block(0, 0, npla.inv(M.get_block(0, 0)))

    # Forwards sweep.
    for q in range(num_blocks - 1):
        M_pp, M_pq, M_qp = M.get_blocks((q + 1, q + 1), (q + 1, q), (q, q + 1))
        g_qq = G.get_block(q, q)
        G.set_block(q + 1, q + 1, npla.inv(M_pp - M_pq @ g_qq @ M_qp))

    # Backwards sweep.
    for q in range(num_blocks - 2, -1, -1):
        M_qp, M_pq = M.get_blocks((q, q + 1), (q + 1, q))
        g_qq, G_pp = G.get_blocks((q, q), (q + 1, q + 1))
        G.set_block(q, q, g_qq + g_qq @ (M_qp @ G_pp @ M_pq) @ g_qq)

    return G
