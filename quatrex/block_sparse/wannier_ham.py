# Copyright 2023-2024 ETH Zurich and the QuaTrEx authors. All rights reserved.

import numpy as np
import numpy.typing as npt
import time
from dataclasses import dataclass, field
from functools import partial
from typing import Callable, Optional

from bcsr_matrix import bcsr_find_sparsity_pattern,get_block_from_bcsr

def wannierHam_generator_1d(wannier_hr: npt.NDArray[np.complexfloating],
                            potential: npt.NDArray[np.floating],
                            nb: int,
                            ns: int,
                            row_ind: int,
                            col_ind: int,
                            iblock: int,
                            idiag:int) -> np.complexfloating:
    '''return the specific matrix element of the upscaled device matrix for the Wannier-stype 
    model with the potential applied to the diagonal elements. This is a simplified version 
    for quasi-1D materials, with less computation needed than the more general 3D version.

    Parameters
    ----------  
    wannier_hr: 
        a ndarray of Wannier-style periodic matrix elements 
        [R1,R2,R3,m,n] where R=(R1,R2,R3) and |nR> refers to function `n` in unit cell `R`        
        < m0 | H | nR >  is the matrix element of matrix H
        see [https://wannier.org/support/] for details on Wannier functions

    potential:
        on-site (Hartree) potential 

    nb: 
        number of bands / wannier functions
    
    ns:
        number of unit cells in the transport super cell

    Returns
    -------
    value of device matrix at a specific position
    '''
    r1 = idiag*ns + (col_ind - row_ind) // nb 
    m = row_ind % nb 
    n = col_ind % nb 
    r2 = 0
    r3 = 0
    if (row_ind == col_ind):
        pot_shift = potential[row_ind + iblock*ns*nb]
    else:
        pot_shift = 0.0
    if (r1<wannier_hr.shape[0]):
        h=wannier_hr[r1,r2,r3,m,n]    
    else:
        h=0.0
    return h + pot_shift


def wannierHam_generator_3d(wannier_hr: npt.NDArray[np.complexfloating],
                            potential: npt.NDArray[np.floating],
                            nb: int,
                            ns: int,
                            row_ind: int,
                            col_ind: int,
                            iblock: int,
                            idiag:int,
                            kvec: npt.NDArray[np.floating] = None,
                            cell: npt.NDArray[np.complexfloating] = None) -> np.complexfloating:
    '''return the specific matrix element of the upscaled device matrix at a transverse k, 
    for the Wannier-stype periodic matrix with potential applied to the 
    diagonal elements. 

    Parameters
    ----------  
    wannier_hr: 
        the Wannier-style periodic matrix elements 
        [R1,R2,R3,m,n] where R=(R1,R2,R3) and |nR> refers to function `n` in unit cell `R`        
        < m0 | H | nR >  is the matrix element of matrix H
        see [https://wannier.org/support/] for details on Wannier functions

    potential:
        on-site (Hartree) potential 

    nb: 
        number of bands / wannier functions
    
    ns:
        number of unit cells in the transport super cell

    kvec:
        transverse k vector size of [3]      

    cell:
        unit cell size of [3,3]

    Returns
    -------
    value of device matrix at a specific position
    '''
    a1=cell[:,0]
    a2=cell[:,1]
    a3=cell[:,2]
    r1 = idiag*ns + (col_ind - row_ind) // nb 
    m = row_ind % nb 
    n = col_ind % nb 
    ny= wannier_hr.shape[1]
    nz= wannier_hr.shape[2]
    h = 0.0+1j*0.0
    if (r1<wannier_hr.shape[0]):
        for r2 in range(ny):
            for r3 in range(nz):
                rt = (r2 - ny//2) * a2 + (r3 - nz//2) * a3 
                phi = np.exp( - 1j * kvec.dot(rt) )
                h += wannier_hr[r1,r2,r3,m,n] * phi
    
    if (row_ind == col_ind):
        pot_shift = potential[row_ind + iblock*ns*nb]
    else:
        pot_shift = 0.0
    return h + pot_shift

@dataclass(frozen=True)
class WannierHam:

    func: Callable
    wannier_hr: npt.NDArray[np.complexfloating]
    potential: npt.NDArray[np.floating]
    nb: int
    ns: int
    kwargs: Optional[dict] = None    

    def generator(self) -> Callable:
        kwargs = self.kwargs or dict()
        return partial(self.func, self.wannier_hr, self.potential, self.nb, self.ns, **kwargs)        


if __name__ == "__main__":
    
    rng = np.random.default_rng(0)

    num_rows = 10
    num_cols = 10
    num_blocks = 5
    num_diags = 3

    # TODO: Select proper values for the dimensions
    wannier_hr = rng.random((5, 5, 5, 500, 500)) + 1j * rng.random((5, 5, 5, 500, 500))
    num_blocks = 10
    nb = 500
    ns = 4
    block_sizes = np.ones(num_blocks,dtype=int) * nb*ns
    potential = rng.random(num_blocks*nb*ns)
    
    kvec = np.array([0.1, 0.2, 0.3])
    cell = rng.random((3, 3)) + 1j * rng.random((3, 3))

    row_ind = 1
    col_ind = 2
    iblock = 1
    idiag = 1

    ref_1d = wannierHam_generator_1d(wannier_hr, potential, nb, ns, row_ind, col_ind, iblock, idiag)
    ref_3d = wannierHam_generator_3d(wannier_hr, potential, nb, ns, row_ind, col_ind, iblock, idiag, kvec, cell)

    wannier_1d = WannierHam(wannierHam_generator_1d, wannier_hr, potential, nb, ns).generator()
    wannier_3d = WannierHam(wannierHam_generator_3d, wannier_hr, potential, nb, ns, {'kvec':kvec, 'cell':cell}).generator()
    val_1d = wannier_1d(row_ind, col_ind, iblock, idiag)
    val_3d = wannier_3d(row_ind, col_ind, iblock, idiag)

    assert np.allclose(ref_1d, val_1d)
    assert np.allclose(ref_3d, val_3d)


    col_index, ind_ptr, nnz = bcsr_find_sparsity_pattern(wannier_1d,num_blocks=num_blocks,
                                                         num_diag=3,
                                                         block_sizes=block_sizes,threshold=0.1)
    start_time = time.time()
    mat = get_block_from_bcsr(wannier_1d,col_index=col_index,
                              ind_ptr=ind_ptr,block_sizes=block_sizes,
                              iblock=1,idiag=0,dtype='complex')
    end_time = time.time()
    print('densify a block take seconds =',end_time - start_time)
    print('dense matrix block size=',mat.shape)