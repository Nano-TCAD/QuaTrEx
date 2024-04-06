import numpy as np    
import types

#  block sparse CSR matrix:
#     same as CSR format, but with an easier access to the ind_ptr in block [i,j]
#     assume each block to be of same size `block_size`
#     consider also using function instead of array for matrix-free representation

#  return a dense block matrix of size (block_size x block_size) filled with values from 
#    `v` , the `col_index` is like CSR index array but with column index within block matrix
#    the `ind_ptr` is like a CSR `ind_ptr` but for each block
#    the `iblock` is block index, `idiag` is off-diagonal index of the wanted block 
#    NOTE: `v` can be a function to return the matrix value of element at corresponding position,
#          or an array of values.
def get_block_from_bcsr(v,col_index,ind_ptr,nnz,block_size,
                        num_blocks,num_diag,dtype,iblock,idiag):
    mat = np.zeros((block_size,block_size),dtype=dtype)
    if (type(v) == types.functionType):
        for i in range(block_size):
            # get ind_ptr for the block row i
            ptr1 = ind_ptr[i,  iblock, idiag]
            ptr2 = ind_ptr[i+1,iblock, idiag]
            for j in range(ptr1,ptr2):
                mat[i, col_index[j]] = v(i,col_index[j],iblock,idiag)   
    else: 
        for i in range(block_size):
            # get ind_ptr for the block row i
            ptr1 = ind_ptr[i,  iblock, idiag]
            ptr2 = ind_ptr[i+1,iblock, idiag]
            for j in range(ptr1,ptr2):
                mat[i, col_index[j]] = v[j]   
    return mat        

# put a dense matrix values into the corresponding position of value array `v` 
#    NOTE: `v` is an array
def put_block_to_bcsr(v,col_index,ind_ptr,nnz,block_size,
                      num_blocks,num_diag,iblock,idiag,mat):
    for i in range(block_size):
        # get ind_ptr for the block row i
        ptr1 = ind_ptr[i,  iblock, idiag]
        ptr2 = ind_ptr[i+1,iblock, idiag]
        for j in range(ptr1,ptr2):
            v[j] = mat[i, col_index[j]]
    return

# compute V@P@V for iblock in a range of off-diagonal and return the dense blocks for several diagonals
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
                tmp = Vblock @ Pblock
                for step3 in range(-num_diag,num_diag+1):
                    l=k+step3
                    if ((l>=-outndiag1)and(l<=outndiag2)):
                        Vblock = get_block_from_bcsr(V,col_index,ind_ptr,nnz,block_size,
                                    num_blocks,num_diag,rowC,step3)
                        mat[l,:,:]=tmp @ Vblock

    return mat

# compute M@G for iblock on idiag-th diagonal and return the dense block
def matmul_bcsr(M,G,col_index,ind_ptr,nnz,block_size,
                        num_blocks,num_diag,dtype,iblock,idiag):
    
    mat = np.zeros((block_size,block_size),dtype=dtype)

    Mblock = get_block_from_bcsr(M,col_index,ind_ptr,nnz,block_size,
                        num_blocks,num_diag,iblock,idiag)
    Gblock = get_block_from_bcsr(G,col_index,ind_ptr,nnz,block_size,
                        num_blocks,num_diag,iblock,idiag)
    
    return mat


