import numpy as np    
import numpy.typing as npt
from typing import Callable, Optional
from bcsr_matrix import get_block_from_bcsr
from bcsr_tensor import get_block_from_bcsr_tensor


def calc_fork(G_lesser,
            G_greater,
            D_lesser,
            D_greater,            
            nnz,
            ie,
            nw,
            map_ew_pair):
    '''return a Fork self energy that is described by 
    $ Sig^{<>}_{ij}(E) = \sum_w 
                        G{<>}_{ij}(E-w) D^{<>}_{ij}(w) + 
                        G{<>}_{ij}(E+w) D^{><}_{ij}(w) $    
    '''
    sig_lesser = np.zeros(nnz,dtype=np.complexfloating)
    sig_greater = np.zeros(nnz,dtype=np.complexfloating)
    for iw in range(nw):
        ie_up   = map_ew_pair(ie,iw,'up')
        ie_down = map_ew_pair(ie,iw,'down')
        gl_down = G_lesser[ie_down,:]
        gl_up   = G_lesser[ie_up,:]
        gg_down = G_greater[ie_down,:]
        gg_up   = G_greater[ie_up,:]
        dl      = D_lesser[iw,:]
        dg      = D_greater[iw,:]

        sig_lesser  += gl_down * dl + gl_up * dg 
        sig_greater += gg_down * dg + gg_up * dl 

    return sig_lesser,sig_greater 



def calc_fork_with_coupling_matrix(G_lesser,
                                G_greater,
                                M,
                                D_lesser,
                                D_greater,
                                col_index,
                                ind_ptr,
                                nnz,
                                ie,
                                nw,
                                map_ew_pair,
                                signature,
                                iblock,
                                idiag,
                                block_sizes,
                                num_diag,
                                num_blocks,
                                obc):
    '''return a Fork self energy that is described by 
    $ Sig^{<>}_{ij}(E) = \sum_w
                        M_{ij} G^{<>}_{jk}(E-w) M_{kl} D^{<>}_{il}(w) +/-
                        M_{ij} G^{<>}_{jk}(E-w) M_{kl} D^{<>}_{jk}(w) +/-
                        M_{ij} G^{<>}_{jk}(E-w) M_{kl} D^{<>}_{ik}(w) +/-
                        M_{ij} G^{<>}_{jk}(E-w) M_{kl} D^{<>}_{jl}(w) + 
                        M_{ij} G^{<>}_{jk}(E+w) M_{kl} D^{><}_{il}(w) +/-
                        M_{ij} G^{<>}_{jk}(E+w) M_{kl} D^{><}_{jk}(w) +/-
                        M_{ij} G^{<>}_{jk}(E+w) M_{kl} D^{><}_{ik}(w) +/-
                        M_{ij} G^{<>}_{jk}(E+w) M_{kl} D^{><}_{jl}(w) $
    return a dense block at iblock and idiag
    '''
    sig_lesser = np.zeros((block_sizes[iblock],block_sizes[iblock+idiag]),dtype=np.complexfloating)
    sig_greater = np.zeros((block_sizes[iblock],block_sizes[iblock+idiag]),dtype=np.complexfloating)
    for iw in range(nw):
        # refer iblock as index 0, after 3 jumps will be on diagonal l=step1+2+3
        i=0
        for step1 in range(-num_diag,num_diag+1):  
            j=i+step1      
            rowB=iblock+j
            M1 = get_block_from_bcsr(M,col_index=col_index,ind_ptr=ind_ptr,block_sizes=block_sizes,
                                        iblock=iblock,idiag=step1,dtype='complex',nnz=nnz)  
            for step2 in range(-num_diag,num_diag+1):
                k=j+step2 
                rowC=iblock+k 
                in_range = ((rowB>=0) and (rowB<num_blocks) and 
                            (rowC>=0) and (rowC<num_blocks))
                if (in_range or obc):
                    # case `rowB` and `rowC` be inside the matrix 
                    # or case opposite, but to correct boundary effect
                    if (not in_range): 
                        rowB = max(0, min(rowB, num_blocks-1))
                        rowC = max(0, min(rowC, num_blocks-1))
                          
                    ie_up   = map_ew_pair(ie,iw,'up')
                    ie_down = map_ew_pair(ie,iw,'down')

                    gl_down = get_block_from_bcsr(G_lesser[ie_down,:],col_index=col_index,ind_ptr=ind_ptr,block_sizes=block_sizes,
                                                    iblock=rowB,idiag=step2,dtype='complex',nnz=nnz)
                    gl_up   = get_block_from_bcsr(G_lesser[ie_up,:],col_index=col_index,ind_ptr=ind_ptr,block_sizes=block_sizes,
                                                    iblock=rowB,idiag=step2,dtype='complex',nnz=nnz)
                    gg_down = get_block_from_bcsr(G_greater[ie_down,:],col_index=col_index,ind_ptr=ind_ptr,block_sizes=block_sizes,
                                                    iblock=rowB,idiag=step2,dtype='complex',nnz=nnz)
                    gg_up   = get_block_from_bcsr(G_greater[ie_up,:],col_index=col_index,ind_ptr=ind_ptr,block_sizes=block_sizes,
                                                    iblock=rowB,idiag=step2,dtype='complex',nnz=nnz)
                    
                    dl      = get_block_from_bcsr(D_lesser[iw,:],col_index=col_index,ind_ptr=ind_ptr,block_sizes=block_sizes,
                                                    iblock=rowB,idiag=step2,dtype='complex',nnz=nnz)
                    dg      = get_block_from_bcsr(D_greater[iw,:],col_index=col_index,ind_ptr=ind_ptr,block_sizes=block_sizes,
                                                    iblock=rowB,idiag=step2,dtype='complex',nnz=nnz)
                    
                    GldDl = gl_down * dl # G^{<>}_{jk}(E-w) D^{<>}_{jk}(w) 
                    GgdDg = gg_down * dg
                    GluDg = gl_up * dg # G^{<>}_{jk}(E+w) D^{><}_{jk}(w) 
                    GguDl = gg_up * dl

                    M_GldDl = M1 @ GldDl #  M_{ij} G^{<>}_{jk}(E-w) D^{<>}_{jk}(w)
                    M_GgdDg = M1 @ GgdDg
                    M_GluDg = M1 @ GluDg
                    M_GguDl = M1 @ GguDl
                                    
                    MGld = M1 @ gl_down # M_{ij} G^{<>}_{jk}(E-w)
                    MGgd = M1 @ gg_down
                    MGlu = M1 @ gl_up # M_{ij} G^{<>}_{jk}(E+w)
                    MGgu = M1 @ gg_up

                    if ((k>=-num_diag) and (k<=num_diag)): 
                        dl = get_block_from_bcsr(D_lesser[iw,:],col_index=col_index,ind_ptr=ind_ptr,block_sizes=block_sizes,
                                                    iblock=iblock,idiag=k,dtype='complex',nnz=nnz)
                        dg = get_block_from_bcsr(D_greater[iw,:],col_index=col_index,ind_ptr=ind_ptr,block_sizes=block_sizes,
                                                    iblock=iblock,idiag=k,dtype='complex',nnz=nnz)
                        
                        MGldDl = MGld * dl # M_{ij} G^{<>}_{jk}(E-w) D^{<>}_{ik}(w)
                        MGgdDg = MGgd * dg 
                        MGluDg = MGlu * dg # M_{ij} G^{<>}_{jk}(E+w) D^{><}_{ik}(w)
                        MGguDl = MGgu * dl 

                    for step3 in range(-num_diag,num_diag+1):
                        l=k+step3
                        if ((l>=-num_diag)and(l<=num_diag)):
                            M2 = get_block_from_bcsr(M,col_index=col_index,ind_ptr=ind_ptr,block_sizes=block_sizes,
                                                        iblock=rowC,idiag=step3,dtype='complex',nnz=nnz)  
                            
                            GldM =  gl_down @ M2
                            GgdM =  gg_down @ M2
                            GluM =  gl_up   @ M2
                            GguM =  gg_up   @ M2

                            dl = get_block_from_bcsr(D_lesser[iw,:],col_index=col_index,ind_ptr=ind_ptr,block_sizes=block_sizes,
                                                        iblock=rowB,idiag=step2+step3,dtype='complex',nnz=nnz)
                            dg = get_block_from_bcsr(D_greater[iw,:],col_index=col_index,ind_ptr=ind_ptr,block_sizes=block_sizes,
                                                        iblock=rowB,idiag=step2+step3,dtype='complex',nnz=nnz)
                            
                            GldMDl = GldM * dl
                            GgdMDg = GgdM * dg
                            GluMDg = GluM * dg
                            GguMDl = GguM * dl

                            M_GldMDl = M1 @ GldMDl        
                            M_GgdMDg = M1 @ GgdMDg
                            M_GluMDg = M1 @ GluMDg
                            M_GguMDl = M1 @ GguMDl

                            sig_lesser  += signature[3] * (M_GldMDl + M_GluMDg)
                            sig_greater += signature[3] * (M_GgdMDg + M_GguMDl)
                            
                            M_GldDl_M = M_GldDl @ M2 #  M_{ij} G^{<>}_{jk}(E-w) D^{<>}_{jk}(w)  M_{kl}
                            M_GgdDg_M = M_GgdDg @ M2
                            M_GluDg_M = M_GluDg @ M2
                            M_GguDl_M = M_GguDl @ M2

                            sig_lesser  += signature[1] * (M_GldDl_M + M_GluDg_M)
                            sig_greater += signature[1] * (M_GgdDg_M + M_GguDl_M)

                            MGldM = MGld @ M2
                            MGgdM = MGgd @ M2
                            MGluM = MGlu @ M2
                            MGguM = MGgu @ M2

                            dl = get_block_from_bcsr(D_lesser[iw,:],col_index=col_index,ind_ptr=ind_ptr,block_sizes=block_sizes,
                                                        iblock=iblock,idiag=l,dtype='complex',nnz=nnz)
                            dg = get_block_from_bcsr(D_greater[iw,:],col_index=col_index,ind_ptr=ind_ptr,block_sizes=block_sizes,
                                                        iblock=iblock,idiag=l,dtype='complex',nnz=nnz)
                            
                            MGldM_Dl = MGldM * dl # M_{ij} G^{<>}_{jk}(E-w) M_{kl} D^{<>}_{il}(w)
                            MGgdM_Dg = MGgdM * dg
                            MGluM_Dg = MGluM * dg
                            MGguM_Dl = MGguM * dl
                            
                            sig_lesser  += signature[0] * (MGldM_Dl + MGluM_Dg)
                            sig_greater += signature[0] * (MGgdM_Dg + MGguM_Dl)

                            if ((k>=-num_diag) and (k<=num_diag)): 
                                MGldDlM = MGldDl @ M2 # M_{ij} G^{<>}_{jk}(E-w) D^{<>}_{ik}(w) M_{kl} 
                                MGgdDgM = MGgdDg @ M2
                                MGluDgM = MGluDg @ M2
                                MGguDlM = MGguDl @ M2 
                                sig_lesser  += signature[2] * (MGldDlM + MGluDgM)
                                sig_greater += signature[2] * (MGgdDgM + MGguDlM)
    return sig_lesser,sig_greater 



def calc_fork_with_coupling_tensor(G_lesser,
                                G_greater,
                                M,
                                D_lesser,
                                D_greater,
                                col_index,
                                ind_ptr,
                                nnz,
                                ie,
                                nw,
                                map_ew_pair,                                   
                                iblock,
                                idiag,
                                block_sizes):
    '''return a Fork self energy that is described by 
    $ Sig^{<>}_{ij}(E) = \sum_w  
                        M_{ij,p} G^{<>}_{jk}(E-w) M_{kl,q} D^{<>}_{pq}(w) +
                        M_{ij,p} G^{<>}_{jk}(E+w) M_{kl,q} D^{><}_{pq}(w) $
    return a dense block at iblock and idiag
    '''
    sig_lesser = np.zeros((block_sizes[iblock],block_sizes[iblock+idiag]),dtype=np.complexfloating)
    sig_greater = np.zeros((block_sizes[iblock],block_sizes[iblock+idiag]),dtype=np.complexfloating)
    for iw in range(nw):
        ie_up   = map_ew_pair(ie,iw,'up')
        ie_down = map_ew_pair(ie,iw,'down')

    return sig_lesser,sig_greater 




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