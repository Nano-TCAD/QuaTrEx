import numpy as np    
from typing import Callable

from numpy.linalg import inv
from scipy.sparse import coo_matrix

#  block sparse CSR matrix:
#     simply stack CSR of each block continuously, with an easy access to the i-th block on i-th diagonal [iblock,idiag]
#     the diagonal block sizes are defined by `block_sizes`
#     can consider using a generator function instead of an array of values for matrix-free representation 

def get_block_from_bcsr(v,col_index:np.ndarray,ind_ptr:np.ndarray,block_sizes:np.ndarray,
                        iblock:int,idiag:int,dtype='complex',nnz:int=0,num_blocks:int=0,
                        num_diag:int=0,offset:int=0) -> np.ndarray:
    """return a dense matrix of size (block_size x block_size) of the block 
    [iblock,idiag] filled with values from `v`.    
     
    Parameters
    ----------   
    v: 
        values
        NOTE: `v` can be a function to return the matrix value of element at 
            corresponding position,
            or an array of values of size [nnz ( // comm_size)].
    col_index:  
        like CSR column indices array but with column index within block matrix of size [nnz ( // comm_size)]
    ind_ptr: 
        like a CSR row pointer array but for each block of size [max_block_size+1,num_blocks,num_diag]
    iblock: 
        block index
    idiag: 
        off-diagonal index of the wanted block     
    offset: 
        offset of pointer index of this MPI-rank 
    block_sizes:
        block sizes of the diagonal blocks
    num_diag:
        number of off-diagonals (same for upper and lower)
    num_blocks:
        number of blocks on diagonal
    nnz:
        number of nonzero/specified elements

    Returns
    -------
    mat : np.ndarray
        Dense matrix of the wanted block
    """
    mat = np.zeros((block_sizes[iblock],block_sizes[iblock+idiag]),dtype=dtype)
    if (isinstance(v, Callable)):
        for i in range(block_sizes[iblock]):
            # get ind_ptr for the block row i
            ptr1 = ind_ptr[i,  iblock, idiag]
            ptr2 = ind_ptr[i+1,iblock, idiag]
            for j in range(ptr1,ptr2):
                col=col_index[j] # no need to substract offset because each MPI-rank keep the entire col_index
                mat[i, col] = v(i,col,iblock,idiag)   
    else: 
        for i in range(block_sizes[iblock]):
            # get ind_ptr for the block row i
            ptr1 = ind_ptr[i,  iblock, idiag]
            ptr2 = ind_ptr[i+1,iblock, idiag]
            for j in range(ptr1,ptr2):
                col=col_index[j] # no need to substract offset because each MPI-rank keep the entire col_index
                mat[i, col] = v[j-offset]   
    return mat        


# put a dense matrix values into the corresponding position of value array `v` 
#    NOTE: `v` is an array
def put_block_to_bcsr(v,col_index:np.ndarray,ind_ptr:np.ndarray,block_sizes:np.ndarray,
                        iblock:int,idiag:int,mat,nnz:int=0,offset:int=0,num_blocks:int=0,
                        num_diag:int=0,num_dim:int=0):
    for i in range(block_sizes[iblock]):
        # get ind_ptr for the block row i
        ptr1 = ind_ptr[i,  iblock, idiag]
        ptr2 = ind_ptr[i+1,iblock, idiag]
        for j in range(ptr1,ptr2):
            v[j-offset] = mat[i, col_index[j]] # no need to substract offset because each MPI-rank keep the entire col_index
    return


def coo_to_bcsr(v_coo:np.ndarray,row:np.ndarray,col:np.ndarray,block_sizes:np.ndarray,
                nnz:int,num_blocks:int,num_diag:int):
    ind_ptr = np.zeros((np.max(block_sizes),num_blocks,num_diag),dtype=int)
    col_index = np.zeros(nnz,dtype=int)
    v_bcsr = np.zeros(nnz,dtype=v_coo.dtype)
    block_startidx = np.zeros(num_blocks+1,dtype=int)
    ind = np.zeros((nnz//num_blocks*2,num_blocks,num_diag),dtype=int) # 2 is to leave some space
    nn = np.zeros((num_blocks,num_diag),dtype=int) 
    for i in range(num_blocks):
        block_startidx[i+1] = block_startidx[i]+block_sizes[i]                
    for i in range(nnz):
        row_block  = np.searchsorted(block_startidx , row[i]) - 1
        col_block  = np.searchsorted(block_startidx , col[i]) - 1
        idiag = col_block - row_block
        iblock= row_block        
        ind[nn[iblock,idiag],iblock,idiag] = i
        nn[iblock,idiag] += 1
    bcsr_nnz=0    
    for iblock in range(num_blocks):
        for idiag in range(num_diag):
            block_csr = coo_matrix( ( v_coo[ind[0:nn[iblock,idiag],iblock,idiag]], 
                                    (row[ind[0:nn[iblock,idiag],iblock,idiag]], col[ind[0:nn[iblock,idiag],iblock,idiag]]) ),
                                    shape=(block_sizes[iblock], block_sizes[iblock+idiag]) ).tocsr()
            block_nnz = block_csr.nnz                
            v_bcsr[bcsr_nnz:bcsr_nnz+block_nnz] = block_csr.data
            col_index[bcsr_nnz:bcsr_nnz+block_nnz] = block_csr.indices
            ind_ptr[0:block_sizes[iblock]+1,iblock,idiag] = block_csr.indptr
            bcsr_nnz += block_nnz                
    return v_bcsr,col_index,ind_ptr,bcsr_nnz


def bcsr_to_coo(col_index:np.ndarray,ind_ptr:np.ndarray,nnz:int,num_blocks:int,
                num_diag:int,block_sizes:np.ndarray):
    col = np.zeros(nnz,dtype=int)
    row = np.zeros(nnz,dtype=int)
    block_startidx = np.zeros(num_blocks+1,dtype=int)
    for i in range(num_blocks):
        block_startidx[i+1] = block_startidx[i]+block_sizes[i]    
    for iblock in range(num_blocks):
        for idiag in range(num_diag):
             for i in range(block_sizes[iblock]):
                # get ind_ptr for the i-th block on i-th diagonal
                ptr1 = ind_ptr[i,  iblock, idiag]
                ptr2 = ind_ptr[i+1,iblock, idiag]
                for j in range(ptr1,ptr2):
                    col[j] = col_index[j] + block_startidx[iblock+idiag]
                    row[j] = i + block_startidx[iblock]                    
    return (row,col)



def bcsr_find_sparsity_pattern(operator,num_blocks:int,num_diag:int,
                               block_sizes:np.ndarray,threshold=1e-6,
                               return_values=False):
    nnz=0
    col_index=[]
    v=[]
    block_startidx = np.zeros(num_blocks+1, dtype=int)
    max_blocksize = np.max(block_sizes)
    ind_ptr = np.zeros((max_blocksize+1,num_blocks,num_diag*2+1), dtype = int)
    for i in range(num_blocks):
        block_startidx[i+1] = block_startidx[i]+block_sizes[i]   
    nnz = 0    
    for iblock in range(num_blocks):
        for idiag in range(-num_diag,num_diag+1):            
            for i in range(block_sizes[iblock]):
                ind_ptr[i,iblock,idiag] = nnz
                jblock = max(min(iblock+idiag,num_blocks-1),0)
                for j in range(block_sizes[jblock]):
                    Hij = operator(i,j,iblock,idiag)
                    if (np.abs(Hij) > threshold):
                        col_index.append(j)
                        if (return_values):
                            v.append(Hij)
                        nnz += 1 
            ind_ptr[block_sizes[iblock], iblock,idiag] = nnz
    col_index=np.array(col_index,dtype=int)
    if (return_values):
        return col_index, ind_ptr, nnz, v 
    else:      
        return col_index, ind_ptr, nnz



# compute V'@P@V for iblock in a range of off-diagonal and return the dense blocks for several diagonals
def trimul_bcsr(V,P,col_index,ind_ptr,nnz,block_size,
                        num_blocks,num_diag,dtype,iblock,outndiag1,outndiag2,obc):
    mat = np.zeros((outndiag1+outndiag2+1,block_size,block_size),dtype=dtype)
    # refer iblock as index 0, after 3 jumps will be on diagonal step1+2+3
    i=0
    for step1 in range(-num_diag,num_diag+1):  
        j=i+step1      
        rowB=iblock+j
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
                Vblock = get_block_from_bcsr(V,col_index,ind_ptr,nnz,block_size,
                                    num_blocks,num_diag,iblock,step1)
                Pblock = get_block_from_bcsr(P,col_index,ind_ptr,nnz,block_size,
                                    num_blocks,num_diag,rowB,step2)
                tmp = Vblock.H @ Pblock
                for step3 in range(-num_diag,num_diag+1):
                    l=k+step3
                    if ((l>=-outndiag1)and(l<=outndiag2)):
                        Vblock = get_block_from_bcsr(V,col_index,ind_ptr,nnz,block_size,
                                    num_blocks,num_diag,rowC,step3)
                        mat[l,:,:]+=tmp @ Vblock

    return mat

# compute M@G for iblock on idiag-th diagonal and return the dense block
def matmul_bcsr(M,G,col_index,ind_ptr,nnz,block_size,
                        num_blocks,num_diag,dtype,iblock,outndiag1,outndiag2,obc):
    
    mat = np.zeros((outndiag1+outndiag2+1,block_size,block_size),dtype=dtype)
    # refer iblock as index 0, after 2 jumps will be on diagonal step1+2
    i=0
    for step1 in range(-num_diag,num_diag+1):  
        j=i+step1      
        rowB=iblock+j
        for step2 in range(-num_diag,num_diag+1):
            k=j+step2 
            rowC=iblock+k 
            in_range = ((rowB>=0) and (rowB<num_blocks))
            if (in_range or obc):
                # case `rowB` be inside the matrix 
                # or case opposite, but to correct boundary effect
                if (not in_range): 
                    rowB = max(0, min(rowB, num_blocks-1))                    
                Mblock = get_block_from_bcsr(M,col_index,ind_ptr,nnz,block_size,
                                    num_blocks,num_diag,iblock,step1)
                Gblock = get_block_from_bcsr(G,col_index,ind_ptr,nnz,block_size,
                                    num_blocks,num_diag,rowB,step2)
                if ((k>=-outndiag1)and(k<=outndiag2)):
                    mat[k,:,:] += Mblock @ Gblock

    return mat
 