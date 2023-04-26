"""Implements functions for generating initial values for retarded, greater, lesser green's functions.
"""
import numpy as np
import numpy.typing as npt
import typing
from scipy import sparse


def init_dense(
    ne: np.int32, no: np.int32, seed: np.int32
) -> typing.Tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128],
                  npt.NDArray[np.complex128]]:
    """
    Initializes retarded, lesser, greater as dense tensors
    with random values, but the right symmetry

    Args:
        ne (np.int32): number of energy points
        no (np.int32): number of orbitals
        seed    (int): seed for random number generation

    Returns:
        tuple(npt.NDArray[np.complex128], retarded Green's function
              npt.NDArray[np.complex128], lesser Green's function
              npt.NDArray[np.complex128]) greater Green's function
    """

    rng = np.random.default_rng(seed)

    # assumption symmetric retarded green's function:
    # \underline{\underline{G}}^{R} \equiv \left( \underline{\underline{G}}^{R}\right)^{T}
    gr: npt.NDArray[np.complex128] = rng.uniform(
        size=(ne, no, no)) + 1j * rng.uniform(size=(ne, no, no))
    gr = 0.5 * (gr.transpose([0, 2, 1]) + gr)

    # \underline{\underline{G}}^{\lessgtr} \equiv - \left( \underline{\underline{G}}^{\lessgtr}\right)^{H}
    gl: npt.NDArray[np.complex128] = rng.uniform(
        size=(ne, no, no)) + 1j * rng.uniform(size=(ne, no, no))
    gl = 0.5 * (gl - gl.conjugate().transpose([0, 2, 1]))

    # \underline{\underline{G}}^{A} \equiv \left( \underline{\underline{G}}^{R}\right)^{H}
    # \underline{\underline{G}}^{R} - \underline{\underline{G}}^{A} = \underline{\underline{G}}^{>} - \underline{\underline{G}}^{<}
    # under above assumption to show that identity for gg is full filled:
    # \underline{\underline{G}}^{>} = \underline{\underline{G}}^{<} + \underline{\underline{G}}^{R} - \left(\underline{\underline{G}}^{R}\right)^{*}
    # -\left( \underline{\underline{G}}^{>} \right)^{H} = -\left( \underline{\underline{G}}^{>} \right)^{H} -  \left( \underline{\underline{G}}^{R} \right)^{H} + \left( \underline{\underline{G}}^{R} \right)^{T}
    # = \underline{\underline{G}}^{>} - \underline{\underline{G}}^{<} + \underline{\underline{G}}^{>}
    gg: npt.NDArray[np.complex128] = gl + gr - gr.conjugate().transpose([0, 2, 1])

    # assert all physical identities
    assert np.allclose(gr, gr.transpose([0, 2, 1]))
    assert np.allclose(gl, -gl.conjugate().transpose([0, 2, 1]))
    assert np.allclose(gg, -gg.conjugate().transpose([0, 2, 1]))
    assert np.allclose(gg - gl, gr - gr.conjugate().transpose([0, 2, 1]))

    return (gr, gl, gg)


def init_sparse(ne: np.int32, nao: np.int32, seed: np.int32
                ) -> typing.Tuple[npt.NDArray[np.double],
                                  npt.NDArray[np.int32],
                                  npt.NDArray[np.int32],
                                  npt.NDArray[np.complex128],
                                  npt.NDArray[np.complex128],
                                  npt.NDArray[np.complex128]]:
    """Create random input values for the calculation of the polarization

    Args:
        ne   (np.int32): number of energy points
        nao  (np.int32): number of orbitals
        seed (np.int32): random seed

    Returns:
        typing.Tuple[npt.NDArray[np.double],    energy
                    npt.NDArray[np.int32],      rows
                    npt.NDArray[np.int32],      columns
                    npt.NDArray[np.complex128], gg
                    npt.NDArray[np.complex128], gl
                    npt.NDArray[np.complex128]  gl
                    ]
    """
    # sanity check
    rng = np.random.default_rng(seed)

    # random energy, not thought out
    energy: npt.NDArray[np.double] = np.arange(ne) + rng.uniform(size=1)[0]
    # create random rows and columns
    rows:       npt.NDArray[np.int32]
    columns:    npt.NDArray[np.int32]

    gr_s = sparse.random(nao, nao, random_state=rng) + \
        1j*sparse.random(nao, nao, random_state=rng)
    gr_s = 0.5 * (gr_s + gr_s.transpose())

    rows = gr_s.tocoo().row.astype(np.int32)
    columns = gr_s.tocoo().col.astype(np.int32)
    no = gr_s.nnz

    gl_s = sparse.coo_matrix(
        (rng.uniform(size=no) + 1j*rng.uniform(size=no), (rows, columns)))
    gl_s = 0.5 * (gl_s - gl_s.conjugate().transpose())
    gg_s = gl_s + gr_s - gr_s.conjugate().transpose()

    # sanity checks
    abstol = 1e-14
    reltol = 1e-6
    assert np.max(np.abs(gr_s - gr_s.transpose())) < abstol + \
        reltol*np.max(np.abs(gr_s))
    assert np.max(np.abs(gl_s + gl_s.conjugate().transpose())
                  ) < abstol + reltol*np.max(np.abs(gl_s))
    assert np.max(np.abs(gg_s + gg_s.conjugate().transpose())
                  ) < abstol + reltol*np.max(np.abs(gg_s))
    assert np.max(np.abs(gg_s - gl_s - gr_s + gr_s.conjugate().transpose())
                  ) < abstol + reltol*np.max(np.abs(gg_s))

    # expand data to ne energy points
    expand = (np.arange(ne) + rng.uniform(size=1)[0]).reshape(1, -1)
    gg: npt.NDArray[np.complex128] = gg_s.tocoo().data.reshape(-1, 1) @ expand
    gl: npt.NDArray[np.complex128] = gl_s.tocoo().data.reshape(-1, 1) @ expand
    gr: npt.NDArray[np.complex128] = gr_s.tocoo().data.reshape(-1, 1) @ expand

    energy:     npt.NDArray[np.double]  = np.squeeze(energy)
    rows:       npt.NDArray[np.int32]   = np.squeeze(rows)
    columns:    npt.NDArray[np.int32]   = np.squeeze(columns)
    return (energy, rows, columns, gg, gl, gr)
