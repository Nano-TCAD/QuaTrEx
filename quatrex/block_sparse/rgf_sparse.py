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
 
# forward pass of RGF
#   solve the left-connected equation system: M@G^r = I and G^< = G^r @ Sig^< @ (G^r)^H
#   G^> can be implicitly known from the identity: G^r - (G^r)^H = G^> - G^<
#   several forward passes can be chained together to solve a big system in small partitions,
#   just need to pass the `sigma_out` and `sigma_out_lesser` into the next function call as
#   `sigma_in` and `sigma_in_lesser` 
#
#   `M` is system matrix, `sigma_scat` and `sigma_scat_lesser` are scattering self-energies
#   `sigma_in` and `sigma_in_lesser` are boundary incoming self-energies from the first site
#   `sigma_out` and `sigma_out_lesser` are boundary outgoing self-energies from the last site
#   `gl` and `gl_lesser` are left-connected Green functions
#   `start_iblock` and `end_iblock` are starting and ending block index for the partition
#   `inc` is the incremental direction of transport axis, + for going from 0 to L and - for opposite
def rgf_forward_pass(M,sigma_scat,sigma_scat_lesser,sigma_in,sigma_in_lesser,
                     start_iblock,end_iblock,num_blocks,block_size):
    length = abs(start_iblock - end_iblock)
    inc = np.sign(end_iblock-start_iblock)
    gl = np.zeros((length,block_size,block_size),dtype='complex')
    gl_lesser = np.zeros((length,block_size,block_size),dtype='complex')
    sigma_out = np.zeros((block_size,block_size),dtype='complex')
    sigma_out_lesser = np.zeros((block_size,block_size),dtype='complex')

    return gl,gl_lesser,sigma_out,sigma_out_lesser

# backward pass of RGF
#   solve the fully-connected equation system: M@G^r = I and G^< = G^r @ Sig^< @ (G^r)^H
def rgf_backward():
    return
