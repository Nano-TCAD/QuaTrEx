# Copyright 2023-2024 ETH Zurich and the QuaTrEx authors. All rights reserved.

import numpy as np
import numpy.typing as npt
import time
from dataclasses import dataclass, field
from functools import partial
from typing import Callable, Optional

from bcsr_matrix import bcsr_find_sparsity_pattern,get_block_from_bcsr



def wannierCoulomb_generator_1d(wannier_hr: npt.NDArray[np.complexfloating],
                            nb: int,
                            ns: int,
                            xmin: int,               
                            ymin: int,
                            zmin: int,             
                            wannier_centers: npt.NDArray[np.floating],
                            cell: npt.NDArray[np.floating],
                            eps_screen: float,
                            r0: float,
                            ldiag: bool,
                            row_ind: int,
                            col_ind: int,
                            iblock: int,
                            idiag:int) -> np.complexfloating:
    '''return the specific matrix element of the upscaled Coulomb matrix for the Wannier-stype 
    model with the potential applied to the diagonal elements. This is a simplified version 
    for quasi-1D materials, with less computation needed than the more general 3D version.

    Parameters
    ----------  
    wannier_hr: 
        a ndarray of Wannier-style periodic matrix elements 
        [R1,R2,R3,m,n] where R=(R1,R2,R3) and |nR> refers to function `n` in unit cell `R`        
        < m0 | H | nR >  is the matrix element of matrix H        
    nb: 
        number of bands / wannier functions    
    ns:
        number of unit cells in the transport super cell
    xmin:
        min of R1
    ymin:
        min of R2    
    zmin:
        min of R3        
    cell:
        unit cell size of [3,3]    
    wannier_centers:
        positions of wannier centers size of [nb][3] in angstrom 
    eps_screen:
        screening epsilon
    r0:
        screening length    

    Returns
    -------
    value of Coulomb matrix at a specific position
    '''
    r1 = idiag * ns + int((col_ind - row_ind) / nb) 
    m = row_ind % nb 
    n = col_ind % nb 
    r2 = 0
    r3 = 0
    a1=cell[:,0]
    a2=cell[:,1]
    a3=cell[:,2]
    r = r1*a1 + r2*a2 + r3*a3 + wannier_centers[m] - wannier_centers[n]
    normr = np.linalg.norm(r)
    e=1.60217663e-19
    epsilon0=8.854e-12
    if (normr >0.0):
        v = (e)/(4.0*np.pi*epsilon0*eps_screen*normr*1.0e-10) * np.tanh(normr/r0)  # in eV
    else:
        if (ldiag):
            v = (e)/(4.0*np.pi*epsilon0*eps_screen*1.0e-10) * (1.0/r0) # self-interaction
        else:
            v = 0.0           
    return v


def wannierHam_generator_1d(wannier_hr: npt.NDArray[np.complexfloating],
                            potential: npt.NDArray[np.floating],
                            nb: int,
                            ns: int,
                            xmin: int,               
                            ymin: int,
                            zmin: int,             
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
    xmin:
        min of R1
    ymin:
        min of R2    
    zmin:
        min of R3        

    Returns
    -------
    value of device matrix at a specific position
    '''
    r1 = idiag * ns + int((col_ind - row_ind) / nb) - xmin      
    m = row_ind % nb 
    n = col_ind % nb 
    r2 = -ymin
    r3 = -zmin
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
                            xmin: int,
                            ymin: int,
                            zmin: int,
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
    xmin: 
        min of R1
    ymin:
        min of R2    
    zmin:
        min of R3    
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
    r1 = idiag * ns + int((col_ind - row_ind) / nb) - xmin      
    m = row_ind % nb 
    n = col_ind % nb 
    ny= wannier_hr.shape[1]
    nz= wannier_hr.shape[2]
    h = 0.0+1j*0.0
    if (r1<wannier_hr.shape[0]):
        for r2 in range(ny):
            for r3 in range(nz):
                rt = (r2 + ymin) * a2 + (r3 + zmin) * a3 
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
    xmin: int
    ymin: int 
    zmin: int
    kwargs: Optional[dict] = None    

    def generator(self) -> Callable:
        kwargs = self.kwargs or dict()
        return partial(self.func, self.wannier_hr, self.potential, self.nb, self.ns, self.xmin, self.ymin, self.zmin, **kwargs)        


if __name__ == "__main__":
    
    rng = np.random.default_rng(0)

    hamdat = np.loadtxt('ham_dat')
    xmin = int(np.min(hamdat[:,0]))
    xmax = int(np.max(hamdat[:,0]))
    ymin = int(np.min(hamdat[:,1]))
    ymax = int(np.max(hamdat[:,1]))
    zmin = int(np.min(hamdat[:,2]))
    zmax = int(np.max(hamdat[:,2]))
    nb = int(np.max(hamdat[:,3])) 
    nx=xmax-xmin+1
    ny=ymax-ymin+1
    nz=zmax-zmin+1
    print(nx,ny,nz,nb)
    print(xmin,xmax)
    wannier_hr = np.zeros((nx,ny,nz,nb,nb),dtype='complex')
    for i in range(hamdat.shape[0]):
        wannier_hr[int(hamdat[i,0]-xmin),
                   int(hamdat[i,1]-ymin),
                   int(hamdat[i,2]-zmin),
                   int(hamdat[i,3])-1,
                   int(hamdat[i,4])-1] = hamdat[i,5] + hamdat[i,6] * 1j 

    
    num_diags = 1   # number of off-diagonals 
    num_blocks = 50    
    ns = 5
    block_sizes = np.ones(num_blocks,dtype=int) * nb*ns
    potential = rng.random(num_blocks*nb*ns) 
    
    kvec = np.array([0.1, 0.2, 0.3])
    cell = rng.random((3, 3)) + 1j * rng.random((3, 3))

    row_ind = 1
    col_ind = 0
    iblock = 0
    idiag = 0

    ref_1d = wannierHam_generator_1d(wannier_hr, potential, nb, ns, xmin, ymin, zmin, row_ind, col_ind, iblock, idiag)
    ref_3d = wannierHam_generator_3d(wannier_hr, potential, nb, ns, xmin, ymin, zmin, row_ind, col_ind, iblock, idiag, kvec, cell)

    wannier_1d = WannierHam(wannierHam_generator_1d, wannier_hr, potential, nb, ns, xmin, ymin, zmin).generator()
    wannier_3d = WannierHam(wannierHam_generator_3d, wannier_hr, potential, nb, ns, xmin, ymin, zmin, {'kvec':kvec, 'cell':cell}).generator()
    val_1d = wannier_1d(row_ind, col_ind, iblock, idiag)
    val_3d = wannier_3d(row_ind, col_ind, iblock, idiag)

    assert np.allclose(ref_1d, val_1d)
    assert np.allclose(ref_3d, val_3d)

    start_time = time.time()
    col_index, ind_ptr, nnz = bcsr_find_sparsity_pattern(wannier_1d,num_blocks=num_blocks,
                                                         num_diag=num_diags,
                                                         block_sizes=block_sizes,threshold=1e-6)
    end_time = time.time()
    print('nnz=',nnz)
    print('sparsity take seconds =',end_time - start_time)
    
    print('--- block diag ---')
    start_time = time.time()
    mat = get_block_from_bcsr(wannier_1d,col_index=col_index,
                              ind_ptr=ind_ptr,block_sizes=block_sizes,
                              iblock=iblock,idiag=0,dtype='complex')
    end_time = time.time()
    print('densify a block take seconds =',end_time - start_time)
    print('dense matrix block size=',mat.shape)
    print('block sparsity ratio=',(ind_ptr[-1,iblock,0]-ind_ptr[0,iblock,0])/(mat.shape[0]*mat.shape[1]))
    H00 = mat
    
    print('--- block 1st offdiag ---')
    start_time = time.time()
    mat = get_block_from_bcsr(wannier_1d,col_index=col_index,
                              ind_ptr=ind_ptr,block_sizes=block_sizes,
                              iblock=iblock,idiag=1,dtype='complex')
    end_time = time.time()
    print('densify a block take seconds =',end_time - start_time)
    print('dense matrix block size=',mat.shape)
    print('block sparsity ratio=',(ind_ptr[-1,iblock,1]-ind_ptr[0,iblock,1])/(mat.shape[0]*mat.shape[1]))
    H01 = mat

    print('--- block -1st offdiag ---')
    start_time = time.time()
    mat = get_block_from_bcsr(wannier_1d,col_index=col_index,
                              ind_ptr=ind_ptr,block_sizes=block_sizes,
                              iblock=iblock,idiag=-1,dtype='complex')
    end_time = time.time()
    print('densify a block take seconds =',end_time - start_time)
    print('dense matrix block size=',mat.shape)
    print('block sparsity ratio=',(ind_ptr[-1,iblock,-1]-ind_ptr[0,iblock,-1])/(mat.shape[0]*mat.shape[1]))
    H10 = mat

    # np.savetxt('H00.dat',H00)
    # np.savetxt('H10.dat',H10)
    # np.savetxt('H01.dat',H01)
    # np.savetxt('wannier_hr.dat',wannier_hr[-xmin,0,0,:,:])
    # np.savetxt('col.dat',col_index[ind_ptr[:,iblock,0]])
    # np.savetxt('indptr1.dat',ind_ptr[:,iblock,-1])
    # np.savetxt('indptr2.dat',ind_ptr[:,iblock,1])

    assert np.allclose(H00, H00.conj().T)
    assert np.allclose(H10, H01.conj().T)
    