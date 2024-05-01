import numpy as np    
import numpy.typing as npt
from typing import Callable, Optional
from rgf_sparse import get_block_from_bcsr


def calc_fork(G_lesser,
            G_greater,
            D_lesser,
            D_greater,
            col_index,
            ind_ptr,
            nnz,
            nen,
            iblock,
            idiag,
            block_sizes,
            iw):
    '''return a Fork self energy that is described by 
    $G_ij D_ij$

    '''
    sigma = np.zeros((block_sizes[iblock],block_sizes[iblock]),dtype=np.complexfloating)

    return sigma 



def calc_fork_with_coupling_matrix(G_lesser,
                                   G_greater,
                                   M,
                                   D_lesser,
                                   D_greater,
                                   col_index,
                                   ind_ptr,
                                   nnz,
                                   nen,
                                   iblock,
                                   idiag,
                                   block_sizes,
                                   iw):
    '''return a Fork self energy that is described by 
    $ M_ij G_jk M_kl D_il +
      M_ij G_jk M_kl D_jk +
      M_ij G_jk M_kl D_ik +
      M_ij G_jk M_kl D_jl +$

    '''
    sigma = np.zeros((block_sizes[iblock],block_sizes[iblock]),dtype=np.complexfloating)

    return sigma 





def calc_bubble_four_byElement(G_lesser,
                             G_greater,
                             mapping,iw,nen,
                             element):
    '''return a bubble self-energy with four indices for one (i,j,k,l) element
    $ P^>(iw; i,j,k,l) = \sum_ie G^>(ie;j,l) * G^<(ie-iw;k,i) $
    $ P^<(iw; i,j,k,l) = \sum_ie G^<(ie;j,l) * G^>(ie-iw;k,i) $
    
    Parameters
    ----------   

    Returns
    -------

    '''
    P_greater = 0.0+0.0*1j
    P_lesser = 0.0+0.0*1j
    ki,jl = mapping(element)
    for ie in range(nen):
        P_greater += G_greater[ie,jl] * G_lesser[ie-iw,ki]
        P_lesser += G_lesser[ie,jl] * G_greater[ie-iw,ki]
    return P_greater, P_lesser


def calc_bubble_four_byBlock(G_lesser,
                        G_greater,
                        col_index, 
                        ind_ptr, 
                        nnz, 
                        nen,
                        iblock,
                        idiag,
                        block_sizes,
                        iw):
    '''return a bubble self-energy with four indices for i-th block
    $ P^>(iw; i,j,k,l) = \sum_ie G^>(ie;j,l) * G^<(ie-iw;k,i) $
    $ P^<(iw; i,j,k,l) = \sum_ie G^<(ie;j,l) * G^>(ie-iw;k,i) $

    Parameters
    ----------   

    Returns
    -------

    '''
    nm = (block_sizes[iblock],block_sizes[iblock],
          block_sizes[iblock+idiag],block_sizes[iblock+idiag])
    P_greater = np.zeros(nm,dtype=np.complexfloating)
    P_lesser = np.zeros(nm,dtype=np.complexfloating)
    for ie in range(nen):
        g_l = get_block_from_bcsr(G_lesser[ie,:],col_index=col_index,
                                  ind_ptr=ind_ptr,block_sizes=block_sizes,
                            iblock=iblock,idiag=idiag,nnz=nnz)
        g_g = get_block_from_bcsr(G_greater[ie,:],col_index=col_index,
                                  ind_ptr=ind_ptr,block_sizes=block_sizes,
                            iblock=iblock,idiag=idiag,nnz=nnz)
        g_l_down = get_block_from_bcsr(G_lesser[ie-iw,:],col_index=col_index,
                                  ind_ptr=ind_ptr,block_sizes=block_sizes,
                            iblock=iblock,idiag=idiag,nnz=nnz)
        g_g_down = get_block_from_bcsr(G_greater[ie-iw,:],col_index=col_index,
                                  ind_ptr=ind_ptr,block_sizes=block_sizes,
                            iblock=iblock,idiag=idiag,nnz=nnz)
        for i in range(nm[0]):
            for j in range(nm[1]):
                for k in range(nm[2]):
                    for l in range(nm[3]):
                        P_greater[i,j,k,l] += g_g[j,l] * g_l_down[k,i]
                        P_lesser[i,j,k,l] += g_l[j,l] * g_g_down[k,i]
    
    return P_greater,P_lesser



def calc_bubble_two(G_lesser,
                     G_greater,
                     transpose_index,
                     nnz, 
                     nen,
                     iw):
    '''return a bubble self-energy with two indices for i-th block
    $ P^>(iw; i,j) = \sum_ie G^>(ie;i,j) * G^<(ie-iw;j,i) $
    $ P^<(iw; i,j) = \sum_ie G^<(ie;i,j) * G^>(ie-iw;j,i) $

    '''    
    P_greater = np.zeros(nnz,dtype=np.complexfloating)
    P_lesser = np.zeros(nnz,dtype=np.complexfloating)
    for ie in range(nen):
        g_l = G_lesser[ie,:]
        g_g = G_greater[ie,:]
        g_l_down = G_lesser[ie-iw,:]
        g_g_down = G_greater[ie-iw,:]
        
        P_greater += g_g * g_l_down[transpose_index]
        P_lesser += g_l * g_g_down[transpose_index]

    return P_greater,P_lesser