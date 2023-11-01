# Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.

import numpy as np
from numpy.typing import ArrayLike


def sancho_rubio(
    M_ii: ArrayLike,
    M_ij: ArrayLike,
    M_ji: ArrayLike = None,
    max_iterations: int = 5000,
    max_delta: float = 1e-8,
) -> np.ndarray:
    """Calculates the surface Green's function iteratively.

    This function generalizes the iterative scheme for the calculation
    of surface Green's functions given in [1]_ in the sense that it can
    be applied to arbitrary periodic system matrices.

    Parameters
    ----------
    M_ii : array_like
        On-diagonal block of the system matrix.
    M_ij : array_like
        Off-diagonal block of the system matrix.
    M_ji : array_like, optional
        Off-diagonal block of the system matrix, by default None. If
        None, the off-diagonal block is assumed to be the hermitian
        adjoint of M_ij.
    max_iterations : int, optional
        Maximum number of iterations, by default 5000.
    max_delta : float, optional
        Maximum relative change in the surface greens function, by
        default 1e-8.

    Returns
    -------
    x_ii : np.ndarray
        The surface Green's function.

    References
    ----------
    .. [1] M.P. López-Sancho, Jose Lopez Sancho, Jessy Rubio. (2000).
       Highly convergent schemes for the calculation of bulk and surface
       Green-Functions. Journal of Physics F: Metal Physics. 15. 851.

    """
    M_ii = np.asarray(M_ii)
    M_ij = np.asarray(M_ij)
    if M_ji is None:
        M_ji = M_ij.conj().T

    epsilon = M_ii.copy()
    epsilon_s = M_ii.copy()
    alpha = M_ji.copy()
    beta = M_ij.copy()

    delta = float("inf")

    for _ in range(max_iterations):
        inverse = np.linalg.inv(epsilon)

        epsilon = epsilon - alpha @ inverse @ beta - beta @ inverse @ alpha
        epsilon_s = epsilon_s - alpha @ inverse @ beta

        alpha = alpha @ inverse @ alpha
        beta = beta @ inverse @ beta

        delta = np.sum(np.abs(alpha) + np.abs(beta)) / 2

        if delta < max_delta:
            break

    else:  # Did not break, i.e. max_iterations reached.
        raise RuntimeError("Surface Green's function did not converge.")

    return np.linalg.inv(epsilon_s)
