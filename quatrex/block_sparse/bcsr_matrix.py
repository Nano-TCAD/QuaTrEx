import numpy as np    
from typing import Callable

from mpi4py import MPI
from mpi4py.MPI import COMM_WORLD as comm

# from numpy.linalg import inv
from scipy.sparse import coo_matrix
from scipy.sparse import csr_matrix

#  blocked sparse CSR matrix:
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
            cols=col_index[ptr1:ptr2] # no need to substract offset because each MPI-rank keep the entire col_index
            mat[i, cols] = v[ptr1-offset : ptr2-offset]   
    return mat        


def get_csr_from_bcsr(v,col_index:np.ndarray,ind_ptr:np.ndarray,block_sizes:np.ndarray,
                        iblock:int,idiag:int,dtype='complex',nnz:int=0,num_blocks:int=0,
                        num_diag:int=0,offset:int=0) -> np.ndarray:    
    bnrow = block_sizes[iblock] # number of rows in the block
    ptr0 = ind_ptr[0,  iblock, idiag]
    ptrN = ind_ptr[bnrow+1, iblock, idiag]
    if (isinstance(v, Callable)):
        # if v is a function, we need to compute the required values first
        data=np.zeros((ptrN-ptr0),dtype=dtype)
        for i in range(bnrow):
            # get ind_ptr for the block row i
            ptr1 = ind_ptr[i,  iblock, idiag]
            ptr2 = ind_ptr[i+1,iblock, idiag]
            for j in range(ptr1,ptr2):
                col = col_index[j] 
                data[j-ptr0] = v(i,col,iblock,idiag)   
        return csr_matrix((data,
                           col_index[ptr0:ptrN],
                           ind_ptr[0 : bnrow+1, iblock, idiag]),
                           shape=(block_sizes[iblock],block_sizes[iblock+idiag]))        
    else: 
        # if v is an array, we can immediately return because the data is contiguous in memory
        return csr_matrix((v[ptr0-offset : ptrN-offset], # remove the offset due to a possible MPI distribution over ijs
                           col_index[ptr0:ptrN],
                           ind_ptr[0 : bnrow+1, iblock, idiag]),
                           shape=(block_sizes[iblock],block_sizes[iblock+idiag]))
        


# put a dense matrix values into the corresponding position of value array `v` 
#    NOTE: `v` is an array
def put_block_to_bcsr(v,col_index:np.ndarray,ind_ptr:np.ndarray,block_sizes:np.ndarray,
                        iblock:int,idiag:int,mat,nnz:int=0,offset:int=0,num_blocks:int=0,
                        num_diag:int=0):
    for i in range(block_sizes[iblock]):
        # get ind_ptr for the block row i
        ptr1 = ind_ptr[i,  iblock, idiag]
        ptr2 = ind_ptr[i+1,iblock, idiag]        
        v[ptr1-offset : ptr2-offset] = mat[i, col_index[ptr1:ptr2]] 
        # no need to substract offset for col_index because we assume that each MPI-rank keep the entire col_index
    return



# put a dense matrix values > a specified threshold into the value array `v` and generate sparsity information 
#    NOTE: `v` is an array
#    NOTE: it puts the CSR data into the `v` and `col_index` arrays from the `start_pos` location, so be careful about this 
def put_block_to_bcsr_variable_sparsity(v,col_index:np.ndarray,ind_ptr:np.ndarray,block_sizes:np.ndarray,
                        iblock:int,idiag:int,start_pos:int,mat,nnz:int=0,threshold:float=1e-6):    
    mask = np.abs(mat) > threshold
    block_csr=coo_matrix(mat[mask]).tocsr()
    block_nnz = block_csr.nnz                
    v[start_pos : start_pos+block_nnz] = block_csr.data
    col_index[start_pos : start_pos+block_nnz] = block_csr.indices
    ind_ptr[0:block_sizes[iblock]+1, iblock, idiag] = block_csr.indptr + start_pos
    nnz += block_nnz
    return

def bcsr_identity_operator(row,col,iblock,idiag):
    '''an identity operator for BCSR
    '''
    if ((idiag==0) and (row==col)):
        v=1.0
    else:
        v=0.0
    return v

def bcsr_get_diagonal_ptrs(col_index:np.ndarray,
                         ind_ptr:np.ndarray,
                         block_sizes:np.ndarray,
                         nnz:int,
                         num_blocks:int)->np.ndarray:
    '''return the data pointers for the diagonal elements 
    '''    
    H_size = np.sum(block_sizes)
    diag_ptrs = np.zeros(H_size,int)
    block_offset=0
    for iblock in range(num_blocks):
        for i in range(block_sizes[iblock]):
            # get ind_ptr for the diagonal block at row i
            idiag=0
            ptr1 = ind_ptr[i,  iblock, idiag]
            ptr2 = ind_ptr[i+1,iblock, idiag]
            cols = col_index[ptr1:ptr2]            
            diag = np.where(cols == i)[0]
            if (diag.size > 0):
                diag_ptrs[block_offset + i] = diag[0] + ptr1
            else:
                diag_ptrs[block_offset + i] = -1
        block_offset += block_sizes[iblock]
    return diag_ptrs


def bcsr_get_transpose_ptrs(col_index:np.ndarray,
                         ind_ptr:np.ndarray,
                         block_sizes:np.ndarray,
                         nnz:int,
                         num_blocks:int,
                         num_diag:int)->np.ndarray:
    '''return the data pointers for the transposed matrix 
    '''        
    transpose_ptrs = np.zeros(nnz,int)
    for iblock in range(num_blocks):
        for idiag in range(-num_diag,num_diag+1):
            for i in range(block_sizes[iblock]):
                i_ptr1 = ind_ptr[i,  iblock, idiag]
                i_ptr2 = ind_ptr[i+1,iblock, idiag]      
                for iptr in range(i_ptr1,i_ptr2):  
                    i_col = col_index[iptr]
                    # find its transposed element, we know easily which block                     
                    jdiag = -idiag
                    jblock = iblock + idiag
                    # we also know j_row == i_col
                    j = i_col                    
                    j_ptr1 = ind_ptr[j,  jblock, jdiag]
                    j_ptr2 = ind_ptr[j+1,jblock, jdiag]  
                    j_cols = col_index[j_ptr1:j_ptr2]
                    transpose_ptr = np.where(j_cols == i)[0]   
                    if (transpose_ptr.size > 0):
                        transpose_ptrs[iptr] = transpose_ptr[0] + j_ptr1
                    else:
                        # can't find the transpose element
                        transpose_ptrs[iptr] = -1    

    return transpose_ptrs


def create_twobody_space(col_index:np.ndarray,
                         ind_ptr:np.ndarray,
                         block_sizes:np.ndarray,
                         nnz:int,
                         num_blocks:int,
                         num_diag:int,
                         interaction_distance:int=0,
                         ):
    '''create the sparsity pattern of a two-body operator $M_{ij;kl}$ expanded from an
    one-body operator $H_{ij}$, following the definition $M_{ij;kl} = H_{jl} H_{ki}$.
    The sparsity pattern of $M_{ij;kl}$ is then stored in the BCSR format. The mapping 
    function from {ij;kl} to {ki} and {jl} is also generated.  This information  
    allows to generate any block of $M_{ij;kl}$ from $H_{ij}$ whenever needed, and to use 
    all the functionalities of BCSR in principle.

    Inputs are the `col_index`, `ind_ptr`, `block_sizes`, `nnz` of $H_{ij}$
    Outputs are the `M_col_index`, `M_ind_ptr`, `M_block_sizes`, `mapping_two_to_onebody` of $M_{ij;kl}$
    '''
    ijkl_list = np.zeros((nnz*nnz,2),dtype=int) # keep a track of (ij;kl)
    M_size = 0
    ij_list = np.zeros((nnz,3),dtype=int) # keep a track of (ij)  
    # generate the list of ij to form the row/col basis 
    #   put i == j first 
    block_startidx = np.zeros(num_blocks+1, dtype=int)
    max_blocksize = np.max(block_sizes)
    H_size = np.sum(block_sizes)
    for i in range(num_blocks):
        block_startidx[i+1] = block_startidx[i]+block_sizes[i]   
    for iblock in range(num_blocks):
        for i in range(block_sizes[iblock]):
            # get ind_ptr for the block row i
            idiag=0
            ptr1 = ind_ptr[i,  iblock, idiag]
            ptr2 = ind_ptr[i+1,iblock, idiag]
            cols=col_index[ptr1:ptr2]            
            diag_ptr = np.where(cols == i)[0]
            if (diag_ptr.size > 0):
                # diagonal elements i==j
                col = i + block_startidx[iblock]
                ij_list[M_size,:] = [ptr1 + diag_ptr[0], col, col]
                M_size += 1  
            
    tip_blocksize = M_size                    
    #   then abs(i-j) <= interaction_distance
    for iblock in range(num_blocks):
        for idiag in range(-num_diag,num_diag+1):
            for i in range(block_sizes[iblock]):
                # get ind_ptr for the block row i   
                ptr1 = ind_ptr[i,  iblock, idiag]
                ptr2 = ind_ptr[i+1,iblock, idiag]
                for j in range(ptr1,ptr2):
                    col = col_index[j] + block_startidx[iblock+idiag] 
                    row = i + block_startidx[iblock] 
                    if ((row != col) and (abs(row - col) <= interaction_distance)):                        
                        ij_list[M_size,:]=[j, row, col]
                        M_size += 1
    #   the interactions outside range are approximated to zero
        
    # Find sparsity pattern of $M_{ij;kl}$    
    M_nnz=0
    for row in range(M_size): 
        for col in range(M_size):    
            i = ij_list[row,1]
            j = ij_list[row,2]
            k = ij_list[col,1]
            l = ij_list[col,2]            
            if ((abs(i-k)<=interaction_distance) and (abs(j-l)<=interaction_distance) and 
                (abs(j-k)<=interaction_distance) and 
                (abs(i-l)<=interaction_distance) and (abs(i-j)<=interaction_distance) and 
                (abs(k-l)<=interaction_distance)):
                ijkl_list[M_nnz,:] = [row,col]
                M_nnz += 1
    # Extract the sparsity information   
    M_col_index = np.zeros((M_nnz),dtype=int)          
    M_bandwidth = 0
    for i in range(M_nnz):
        row = ijkl_list[i,0]
        col = ijkl_list[i,1]
        if ((row >= tip_blocksize) and (col >= tip_blocksize)):
            diag = col-row
            if (abs(diag) > M_bandwidth):
                M_bandwidth = abs(diag)
    M_block_size = M_bandwidth                 
    M_num_blocks = - ( - (M_nnz - tip_blocksize) // M_block_size ) + 1 # round up and plus one for tip block
    M_block_sizes = np.zeros((M_num_blocks),dtype=int) 
    M_block_sizes[0:-1] = M_block_size
    M_block_sizes[-1] = tip_blocksize 
    M_num_diag = 1 + 1 # 1 off-diagonal block, plus 1 for upper/lower arrow, upper as the 2nd diagonal, lower transposed as the -2nd diagonal              
    M_ind_ptr = np.zeros((M_block_size+1,M_num_blocks,M_num_diag*2+1),dtype=int)
    # Find which element belongs to which block    
    ind = np.zeros((M_nnz//M_num_blocks,M_num_blocks,M_num_diag*2+1),dtype=int) # register the {ijkl} belonging to a block
    nn = np.zeros((M_num_blocks,M_num_diag*2+1),dtype=int) # number of nonzeros in each block    
    for i in range(M_nnz):
        row = ijkl_list[i,0]
        col = ijkl_list[i,1]
        if ((row < tip_blocksize) and (col < tip_blocksize)):
            # tip block
            iblock = 0
            idiag = 0            
        elif ((row < tip_blocksize) and (col >= tip_blocksize)):
            # upper arrow
            iblock = (col - tip_blocksize) // M_block_size + 1
            idiag = 2
            if (iblock>=M_num_blocks):
                raise Exception(f"Problem! iblock=",iblock)
        elif ((row >= tip_blocksize) and (col < tip_blocksize)):    
            # lower arrow
            iblock = (row - tip_blocksize) // M_block_size + 1
            idiag = -2
            if (iblock>=M_num_blocks):
                raise Exception(f"Problem! iblock=",iblock)
        if ((row > tip_blocksize) and (col > tip_blocksize)):
            # remaining
            row_block = (col - tip_blocksize) // M_block_size + 1
            col_block = (col - tip_blocksize) // M_block_size + 1
            iblock = row_block
            idiag = col_block - row_block
            if ((abs(idiag)>1) or (iblock>=M_num_blocks)):
                raise Exception(f"Problem! idiag=",idiag,'iblock=',iblock)

        ind[nn[iblock,idiag],iblock,idiag] = i
        nn[iblock,idiag] += 1
    
    data = np.ones((M_nnz),dtype=int)      
    bcsr_nnz = 0    
    for iblock in range(M_num_blocks):
        for idiag in range(-M_num_diag,M_num_diag+1):
            if (idiag==-2 or idiag==2):
                jblock =  0
            else:
                jblock =  iblock + idiag
            block_csr = coo_matrix( ( data[ind[0:nn[iblock,idiag],iblock,idiag]], 
                                (ijkl_list[ind[0:nn[iblock,idiag],iblock,idiag],0], 
                                 ijkl_list[ind[0:nn[iblock,idiag],iblock,idiag],1]) ),
                                    shape=(block_sizes[iblock], block_sizes[jblock]) ).tocsr()
            block_nnz = block_csr.nnz                            
            M_col_index[bcsr_nnz:bcsr_nnz+block_nnz] = block_csr.indices
            M_ind_ptr[0:block_sizes[iblock]+1,iblock,idiag] = block_csr.indptr + bcsr_nnz
            bcsr_nnz += block_nnz                  

    def mapping_two_to_onebody(ijkl:int):
        ki:int 
        jl:int
        row = ijkl_list[ijkl,0]
        col = ijkl_list[ijkl,1]
        i = ij_list[row,1]
        j = ij_list[row,2]
        k = ij_list[col,1]
        l = ij_list[col,2]    
        ki = bcsr_map_ij_to_ptr(col_index,
                                ind_ptr,
                                block_sizes,
                                nnz,
                                num_blocks,
                                num_diag,
                                k,i)
        jl = bcsr_map_ij_to_ptr(col_index,
                                ind_ptr,
                                block_sizes,
                                nnz,
                                num_blocks,
                                num_diag,
                                j,l)
        return ki,jl
    
    return M_col_index, M_ind_ptr, M_block_sizes, M_size, M_num_blocks, M_num_diag, M_nnz, mapping_two_to_onebody



def bcsr_map_ij_to_ptr(col_index:np.ndarray,
                    ind_ptr:np.ndarray,
                    block_sizes:np.ndarray,
                    nnz:int,
                    num_blocks:int,
                    num_diag:int,
                    row:int,col:int)->int:
    '''find the data pointer from the row and col of a BCSR matrix
    '''
    # locate the block
    block_startidx = np.zeros(num_blocks+1,dtype=int)
    for i in range(num_blocks):
        block_startidx[i+1] = block_startidx[i]+block_sizes[i]          
    row_block  = np.searchsorted(block_startidx , row) - 1
    col_block  = np.searchsorted(block_startidx , col) - 1
    idiag = col_block - row_block
    iblock= row_block    
    col_in_block = col - block_startidx[iblock+idiag]
    row_in_block = row - block_startidx[iblock]
    ptr = -1
    # find ptr for the row and col in the block 
    i = row_in_block
    ptr1 = ind_ptr[i,  iblock, idiag]
    ptr2 = ind_ptr[i+1,iblock, idiag]
    for j in range(ptr1,ptr2):
        if (col_in_block == col_index[j]):
            ptr = j           
    if (ptr == -1):
        raise Exception(f"Problem! cannot find element in the BCSR, (row,col)=",row,col,'iblock=',iblock,'idiag=',idiag)        
    return ptr


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
                                       (row[ind[0:nn[iblock,idiag],iblock,idiag]], 
                                        col[ind[0:nn[iblock,idiag],iblock,idiag]]) ),
                                    shape=(block_sizes[iblock], block_sizes[iblock+idiag]) ).tocsr()
            block_nnz = block_csr.nnz                
            v_bcsr[bcsr_nnz:bcsr_nnz+block_nnz] = block_csr.data
            col_index[bcsr_nnz:bcsr_nnz+block_nnz] = block_csr.indices
            ind_ptr[0:block_sizes[iblock]+1,iblock,idiag] = block_csr.indptr + bcsr_nnz
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
                col[ptr1:ptr2] = col_index[ptr1:ptr2] + block_startidx[iblock+idiag]
                row[ptr1:ptr2] = i + block_startidx[iblock]                    
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
        return col_index, ind_ptr, nnz, np.array(v) 
    else:      
        return col_index, ind_ptr, nnz



# compute U@P@V for iblock and idiag-th diagonal and return the dense block 
def trimul_bcsr(U,V,P,col_index,ind_ptr,block_sizes,
                        num_blocks,num_diag,dtype,iblock,idiag,obc):    
    mat = np.zeros((block_sizes[iblock],block_sizes[iblock+idiag]),dtype=dtype)
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
                Ublock = get_block_from_bcsr(U,col_index,ind_ptr,block_sizes,iblock,step1)         
                Pblock = get_block_from_bcsr(P,col_index,ind_ptr,block_sizes,rowB,step2)
                tmp = Ublock @ Pblock
                for step3 in range(-num_diag,num_diag+1):
                    l=k+step3
                    if (l==idiag):
                        Vblock = get_block_from_bcsr(V,col_index,ind_ptr,block_sizes,rowC,step3)
                        mat += tmp @ Vblock

    return mat

# compute M@G for iblock on idiag-th diagonal and return the dense block
def matmul_bcsr(M,G,col_index,ind_ptr,block_sizes,
                        num_blocks,num_diag,dtype,iblock,idiag,obc):
    
    mat = np.zeros((block_sizes[iblock],block_sizes[iblock+idiag]),dtype=dtype)
    # refer iblock as index 0, after 2 jumps will be on diagonal step1+2
    i=0
    for step1 in range(-num_diag,num_diag+1):  
        j=i+step1      
        rowB=iblock+j
        for step2 in range(-num_diag,num_diag+1):
            k=j+step2 
            in_range = ((rowB>=0) and (rowB<num_blocks))
            if (in_range or obc):
                # case `rowB` be inside the matrix 
                # or case opposite, but to correct boundary effect
                if (not in_range): 
                    rowB = max(0, min(rowB, num_blocks-1))                    
                Mblock = get_block_from_bcsr(M,col_index,ind_ptr,block_sizes,iblock,step1)
                Gblock = get_block_from_bcsr(G,col_index,ind_ptr,block_sizes,rowB,step2)
                if (k==idiag):
                    mat += Mblock @ Gblock

    return mat
 

def bcsr_transpose_over_MPI(data:np.ndarray,local_nnz:int,local_nek:int):
    for i in range(local_nek):
        comm.Alltoall(MPI.IN_PLACE,data[i,:])
        buf=np.transpose(np.reshape(data[i,:],(comm.size,local_nnz)))
        data[i,:] = buf.flatten()
    data = np.reshape(np.transpose(data) , (local_nnz,local_nek*comm.size))   

    return data

 
if __name__ == "__main__":
    
    rank = comm.Get_rank()

    num_energies = 15
    nnz = 10

    local_nnz = nnz//comm.size
    local_nek = num_energies//comm.size
    
    data = (1.0+1.0j) * (np.arange(local_nek * local_nnz * comm.size).reshape(local_nek, local_nnz*comm.size) + rank*local_nek*local_nnz*comm.size)    

    ref_data=data.copy()
    
    print("process %s data \n %s " % (rank,data))
    comm.barrier()
    print("transpose ")
    data = bcsr_transpose_over_MPI(data,local_nnz,local_nek)
    print("process %s data \n %s " % (rank,data))
    comm.barrier()
    print("transpose back")    
    data = bcsr_transpose_over_MPI(data,local_nek,local_nnz)
    print("process %s data \n %s " % (rank,data))

    assert np.allclose(ref_data, data)