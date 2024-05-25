import numpy as np    
import numpy.typing as npt
from typing import Callable, Optional

# from numpy.linalg import inv
# from scipy.sparse import coo_matrix

#  block CSR tensor of dimension `num_dim`, generalization of block CSR to higher dimensional data:
#     simply append CSR tensor of each block continuously, with a direct access to the i-th block on i-th diagonal [iblock,idiag]
#     the diagonal tensor block sizes are defined by `block_sizes`
#     can consider using a generator function instead of an array of values for matrix-free representation 

def get_block_from_bcsr_tensor(v,
                            col_index:np.ndarray,
                            ind_ptr:np.ndarray,
                            block_sizes:np.ndarray,
                            iblock:int,
                            idiag:np.ndarray,
                            dtype='complex',
                            nnz:int=0,
                            num_blocks:int=0,
                            num_diag:int=0,
                            num_dim:int=2,
                            offset:int=0) -> np.ndarray:
    """return a densified tensor of size block_size and dimension num_dim of the wanted block 
    [iblock,idiag] filled with values from `v`.    
     
    Parameters
    ----------   
    v: 
        values 
        NOTE: `v` can be a function to return the tensor value of element at 
            corresponding position,
            or an array of values, of size [nnz ( // comm_size)].
    col_index:  
        extend CSR column indices for tensor, and with column index within the block, of size [nnz ( // comm_size), num_dim-1]
    ind_ptr: 
        like a CSR row pointer, for each block, of size [max_block_size+1, num_blocks, num_diag, num_diag, ...]
    iblock: 
        block index
    idiag: 
        off-diagonal index of the wanted block, of size [num_dim-1]     
    offset: 
        offset of pointer index of this MPI-rank 
    block_sizes:
        block sizes of the diagonal blocks

    Returns
    -------
    tensor : np.ndarray
        Dense tensor of the wanted block
    """
    shape = np.concatenate( [ [block_sizes[iblock]], block_sizes[iblock+idiag[:]] ] )
    shape = tuple(shape)
    tensor = np.zeros(shape,dtype=dtype)
    if (isinstance(v, Callable)):
        for i in range(block_sizes[iblock]):
            # get ind_ptr for the block row i
            ptr1 = ind_ptr[i,  iblock, tuple(idiag)]
            ptr2 = ind_ptr[i+1,iblock, tuple(idiag)]
            for j in range(ptr1,ptr2):
                col = col_index[j,:] # no need to substract offset because each MPI-rank keep the entire col_index
                tensor[i, tuple(col)] = v(i,col,iblock,idiag)   
    else: 
        for i in range(block_sizes[iblock]):
            # get ind_ptr for the block row i
            ptr1 = ind_ptr[i,  iblock, tuple(idiag)]
            ptr2 = ind_ptr[i+1,iblock, tuple(idiag)]
            for j in range(ptr1,ptr2):
                col=col_index[j,:] # no need to substract offset because each MPI-rank keep the entire col_index
                tensor[i, tuple(col)] = v[j-offset]   
    return tensor        


# put a dense tensor values at specific positions into the corresponding positions of value array `v` 
#    NOTE: `v` is an array
def put_block_to_bcsr_tensor(v,
                        col_index:np.ndarray,
                        ind_ptr:np.ndarray,
                        block_sizes:np.ndarray,
                        iblock:int,
                        idiag:np.ndarray,
                        tensor:np.ndarray,
                        nnz:int=0,
                        offset:int=0,
                        num_blocks:int=0,
                        num_diag:int=0,
                        num_dim:int=2):
    for i in range(block_sizes[iblock]):
        # get ind_ptr for the block row i
        ptr1 = ind_ptr[i,  iblock, tuple(idiag)]
        ptr2 = ind_ptr[i+1,iblock, tuple(idiag)]
        for j in range(ptr1,ptr2):
            col=col_index[j,:] # no need to substract offset because each MPI-rank keep the entire col_index
            v[j-offset] = tensor[i, tuple(col)]
    return
