# Copyright 2023-2024 ETH Zurich and the QuaTrEx authors. All rights reserved.

import numpy as np
import numpy.typing as npt

from dataclasses import dataclass, field
from functools import partial
from typing import Callable, Optional


def wannierHam_generator_1d(wannier_hr: npt.NDArray[np.complexfloating],
                            potential: npt.NDArray[np.floating],
                            nb: int,
                            ns: int,
                            row_ind: int,
                            col_ind: int,
                            iblock: int,
                            idiag:int) -> np.complexfloating:
    r1 = idiag*ns + (col_ind - row_ind) // nb 
    m = row_ind % nb 
    n = col_ind % nb 
    r2 = 0
    r3 = 0
    if (row_ind == col_ind):
        pot_shift = potential(row_ind + iblock*ns*nb) 
    else:
        pot_shift = 0.0
    return wannier_hr[r1,r2,r3,m,n] + pot_shift


def wannierHam_generator_3d(wannier_hr: npt.NDArray[np.complexfloating],
                            potential: npt.NDArray[np.floating],
                            nb: int,
                            ns: int,
                            kvec: npt.NDArray[np.floating],
                            cell: npt.NDArray[np.complexfloating],
                            row_ind: int,
                            col_ind: int,
                            iblock: int,
                            idiag:int) -> np.complexfloating:
    a1=cell[:,0]
    a2=cell[:,1]
    a3=cell[:,2]
    r1 = idiag*ns + (col_ind - row_ind) // nb 
    m = row_ind % nb 
    n = col_ind % nb 
    ny= wannier_hr.shape[1]
    nz= wannier_hr.shape[2]
    h = 0.0+1j*0.0
    for r2 in range(ny):
        for r3 in range(nz):
            rt = (r2 - ny//2) * a2 + (r3 - nz//2) * a3 
            phi = np.exp( - 1j * kvec.dot(rt) )
            h += wannier_hr[r1,r2,r3,m,n] * phi
    
    if (row_ind == col_ind):
        pot_shift = potential(row_ind + iblock*ns*nb) 
    else:
        pot_shift = 0.0
    return h + pot_shift


wannier_map = {1: wannierHam_generator_1d,
               3: wannierHam_generator_3d}


@dataclass(frozen=True)
class WannierHam:

    key: int
    wannier_hr: npt.NDArray[np.complexfloating]
    potential: npt.NDArray[np.floating]
    nb: int
    ns: int
    kvec: Optional[npt.NDArray[np.floating]] = None
    cell: Optional[npt.NDArray[np.complexfloating]] = None

    def generator(self) -> Callable:
        if self.key == 1:
            func = wannierHam_generator_1d
            return partial(func, self.wannier_hr, self.potential, self.nb, self.ns)
        elif self.key == 3:
            func = wannierHam_generator_3d
            return partial(func, self.wannier_hr, self.potential, self.nb, self.ns, self.kvec, self.cell)
        else:
            raise ValueError("Invalid key for WannierHam")


if __name__ == "__main__":
    
    rng = np.random.default_rng(0)

    num_rows = 10
    num_cols = 10
    num_blocks = 5
    num_diags = 3

    # TODO: Select proper values for the dimensions
    wannier_hr = rng.random((5, 5, 5, 5, 5)) + 1j * rng.random((5, 5, 5, 5, 5))
    potential = rng.random(50)
    nb = 5
    ns = 2
    kvec = np.array([0.1, 0.2, 0.3])
    cell = rng.random((3, 3)) + 1j * rng.random((3, 3))

    row_ind = 1
    col_ind = 2
    iblock = 1
    idiag = 1

    ref_1d = wannierHam_generator_1d(wannier_hr, potential, nb, ns, row_ind, col_ind, iblock, idiag)
    ref_3d = wannierHam_generator_3d(wannier_hr, potential, nb, ns, kvec, cell, row_ind, col_ind, iblock, idiag)

    wannier_1d = WannierHam(1, wannier_hr, potential, nb, ns).generator()
    wannier_3d = WannierHam(3, wannier_hr, potential, nb, ns, kvec, cell).generator()
    val_1d = wannier_1d(row_ind, col_ind, iblock, idiag)
    val_3d = wannier_3d(row_ind, col_ind, iblock, idiag)

    assert np.allclose(ref_1d, val_1d)
    assert np.allclose(ref_3d, val_3d)
