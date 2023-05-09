import numpy as np
import numpy.linalg as npla

from bsr import bsr


def rgf(M: bsr, off_diagonal=False) -> bsr:
    """Applies the recursive Green's function method to a BSR matrix."""
    if off_diagonal:
        raise NotImplementedError("Off-diagonal RGF not implemented yet.")

    if M.blocksize[0] != M.blocksize[1]:
        raise ValueError("Blocks must be square.")

    num_blocks = M.shape[0] // M.blocksize[0]

    diag_blocks = np.repeat(np.ones(M.blocksize, M.dtype), num_blocks)
    G = bsr.diag(*diag_blocks.reshape(-1, *M.blocksize).tolist(), blocksize=M.blocksize)

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
