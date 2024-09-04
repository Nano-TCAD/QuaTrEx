"""
Script for aserting current conservation of the self-energy
"""
import numpy as np


def current_conservation(seg, sel, gfg, gfl):
    """
    Assert current conservation of the self-energy.

    The current conservation of the self-energy is given by::
        \int dE dk Tr [\Sigma^<(E,k) G^>(E,k) - \Sigma^>(E,k) G^<(E,k)] = 0
    
    Parameters
    ----------
    seg : ndarray, shape (num_kpts*ne, nnz)
        The greater self-energy.
    sel : ndarray, shape (num_kpts*ne, nnz)
        The lesser self-energy.
    gfg : ndarray, shape (num_kpts*ne, nnz)
        The greater Green's function.
    gfl : ndarray, shape (num_kpts*ne, nnz)
        The lesser Green's function.
    
    Returns
    -------
    current1 : float
        Current conservation of the first term.
    current2 : float
        Current conservation of the second term.
    """

    # Assert same shape
    assert seg.shape == sel.shape == gfg.shape == gfl.shape

    # Assert current conservation
    resolved_current_term1 = np.conjugate(gfl) * seg
    resolved_current_term2 = sel * np.conjugate(gfg)
    current1 = resolved_current_term1.sum()
    current2 = resolved_current_term2.sum()
    return np.array([current1, current2])
    