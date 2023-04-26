"""Implements the calculation of the Polarization for dense G/P"""
# Copyright Runsheng Ouyang, 2023
import numpy as np
from utils.linalg import correlate_3D


def g2p_dense(
    gg: np.ndarray, gl: np.ndarray, gr: np.ndarray,
    energy: np.ndarray, workers=64
):
    '''
    P^<>_ij(E') = -i*denergy/2pi \sum_{E} (G^<>_ij(E) G^><_ji(E-E'))
    P^r_ij(E') = -i*denergy/2pi \sum_{E} (G^<_ij(E) G^a_ji(E-E')+G^r_ij(E) G^<_ji(E-E'))
    Changed naming to comply with google style guide
    
    Args
        gg, gl, gr : np.ndarray
            greater, lesser, and retarded green's function
            shape = (ne, n_orb, n_orb)
        energy : np.ndarray
            energy = np.linspace(E_min, E_max, ne) has to be evenly spaced!
            shape = (ne,)

    Returns
        pg, pl, pr : np.ndarray
            greater, lesser, and retarded polarization
            shape = (nep, n_orb, n_orb)
        ep : np.ndarray
            energy list for P and W
            should be symmetric
            E'~[-(ne-1)*denergy, ..., 0, ..., +(ne-1)*denergy], denergy=(E_max-E_min)/(ne-1)=E[1]-E[0]
            shape = (nep,)
    '''

    assert len(gr.shape) == 3
    assert len(energy.shape) == 1
    assert energy.shape[0] == gr.shape[0]
    assert gr.shape[1] == gr.shape[2]
    assert gr.shape == gl.shape
    assert gr.shape == gg.shape
    assert np.allclose(np.diff(energy), np.diff(energy)[0])  # evenly spaced

    ne = energy.shape[0]
    denergy = energy[1]-energy[0]

    nep = 2*(ne-1)+1  # mode='full'
    ep = np.linspace(-(ne-1)*denergy, +(ne-1)*denergy, nep)

    # P^<>_ij(E') = -i*denergy/2pi \sum_{E} (G^<>_ij(E) G^><_ji(E-E'))
    pl = correlate_3D(
        gl, gg, mode='full', b_index='ji', method='fft', n_worker=workers
    )
    assert pl.shape[0] == nep

    pg = correlate_3D(
        gg, gl, mode='full', b_index='ji', method='fft', n_worker=workers
    )

    # P^r_ij(E') = -i*denergy/2pi \sum_{E} (G^<_ij(E) G^a_ji(E-E')+G^r_ij(E) G^<_ji(E-E'))
    # G^a=(G^r)^dagger
    ga = np.conjugate(np.einsum('ijk->ikj', gr, optimize='optimal'))
    pr = correlate_3D(
        gl, ga, mode='full', b_index='ji', method='fft', n_worker=workers
    )
    del ga
    pr += correlate_3D(
        gr, gl, mode='full', b_index='ji', method='fft', n_worker=workers
    )
    # times factor 2 to match gold solution, changed minus to plus
    pre_factor = -2*1.0j*denergy/(2.0*np.pi)
    pr *= pre_factor
    pl *= pre_factor
    pg *= pre_factor

    return  pg, pl, pr, ep
