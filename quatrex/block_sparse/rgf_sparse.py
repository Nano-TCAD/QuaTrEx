import numpy as np    
import types
from numpy.linalg import inv
from scipy.sparse import coo_array,csr_array

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
        like a CSR row pointer array but for each block of size [max_block_size+1,num_blocks,num_diag,num_dim]
    iblock: 
        block index
    idiag: 
        off-diagonal index of the wanted block     
    offset: 
        offset of pointer index of this MPI-rank 
    block_sizes:
        block sizes of the diagonal blocks

    Returns
    -------
    mat : np.ndarray
        Dense matrix of the wanted block
    """
    mat = np.zeros((block_sizes[iblock],block_sizes[iblock+idiag]),dtype=dtype)
    if (type(v) == types.functionType):
        for i in range(block_sizes[iblock]):
            # get ind_ptr for the block row i
            ptr1 = ind_ptr[i,  iblock, idiag]
            ptr2 = ind_ptr[i+1,iblock, idiag]
            for j in range(ptr1,ptr2):
                col=col_index[j-offset]
                mat[i, col] = v(i,col,iblock,idiag)   
    else: 
        for i in range(block_sizes[iblock]):
            # get ind_ptr for the block row i
            ptr1 = ind_ptr[i,  iblock, idiag]
            ptr2 = ind_ptr[i+1,iblock, idiag]
            for j in range(ptr1,ptr2):
                col=col_index[j-offset]
                mat[i, col] = v[j-offset]   
    return mat        


# put a dense matrix values into the corresponding position of value array `v` 
#    NOTE: `v` is an array
def put_block_to_bcsr(v,col_index:np.ndarray,ind_ptr:np.ndarray,block_sizes:np.ndarray,
                        iblock:int,idiag:int,mat,nnz:int=0,offset:int=0,num_blocks:int=0,
                        num_diag:int=0,num_dim:int=0):
    for i in range(block_sizes[iblock,idiag]):
        # get ind_ptr for the block row i
        ptr1 = ind_ptr[i,  iblock, idiag]
        ptr2 = ind_ptr[i+1,iblock, idiag]
        for j in range(ptr1,ptr2):
            v[j-offset] = mat[i, col_index[j-offset]]
    return


def coo_to_bcsr(v_coo:np.ndarray,row:np.ndarray,col:np.ndarray,block_sizes:np.ndarray,
                nnz:int,num_blocks:int,num_diag:int):
    ind_ptr = np.zeros((np.max(block_sizes),num_blocks,num_diag),dtype=int)
    col_index = np.zeros(nnz,dtype=int)
    v_bcsr = np.zeros(nnz,dtype=v_coo.dtype)
    block_startidx = np.zeros(num_blocks+1)
    ind = np.zeros(nnz//num_blocks*2,num_blocks,num_diag,dtype=int) # 2 is to leave some space
    nn = np.zeros(num_blocks,num_diag,dtype=int) 
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
            block_csr = coo_array( ( v_coo[ind[0:nn[iblock,idiag]-1,iblock,idiag]], 
                                    (row[ind[0:nn[iblock,idiag]-1,iblock,idiag]], col[ind[0:nn[iblock,idiag]-1,iblock,idiag]]) ),
                                    shape=(block_sizes[iblock], block_sizes[iblock+idiag]) ).tocsr()
            block_nnz = block_csr.nnz                
            v_bcsr[bcsr_nnz:bcsr_nnz+block_nnz] = block_csr.data
            col_index[bcsr_nnz:bcsr_nnz+block_nnz] = block_csr.indices
            ind_ptr[:,iblock,idiag] = block_csr.ind_ptr
            bcsr_nnz += block_nnz                
    return v_bcsr,col_index,ind_ptr,bcsr_nnz



def generate_wannierHam_generator_1d(wannier_hr:np.ndarray, 
                                    potential:np.ndarray,
                                    nb:int, ns:int) :
    '''return a generator function of the upscaled matrix for the Wannier-stype periodic matrix with potential applied to the 
    diagonal elements. This is a simplified 1D version.

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

    Returns
    -------
    wannierHam_generator: function
        generator function of the upscaled matrix 
    '''
    def wannierHam_generator(row_ind:int,col_ind:int,iblock:int,idiag:int) -> np.complex128:
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
    return wannierHam_generator
        


def generate_wannierHam_generator_3d(wannier_hr:np.ndarray, 
                                    potential:np.ndarray,
                                    nb:int, ns:int, kvec:np.ndarray, cell:np.array) :
    '''return a generator function of the upscaled matrix at a transverse k, 
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
        transverse k vector size of [2]      

    cell:
        unit cell size of [3,3]

    Returns
    -------
    wannierHam_generator: function
        generator function of the upscaled matrix 
    '''
    a1=cell[:,0]
    a2=cell[:,1]
    a3=cell[:,2]
    def wannierHam_generator(row_ind:int,col_ind:int,iblock:int,idiag:int) -> np.complex128:
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
    
    return wannierHam_generator


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
#   solve the left-connected equation system: (ES-M-Sig^r@S)@G^r = I and G^< = G^r @ Sig^< @ (G^r)^H
#   G^> can be implicitly known from the identity: G^r - (G^r)^H = G^> - G^<
#   several forward passes can be chained together to solve a big system in small partitions,
#   just need to pass the `sigma_out` and `sigma_out_lesser` into the next function call as
#   `sigma_in` and `sigma_in_lesser` 
#
#   `E` is energy, `M` is system matrix, `S` is overlap matrix, 
#   `sigma_scat` and `sigma_scat_lesser` are scattering self-energies
#   `sigma_in` and `sigma_in_lesser` are boundary incoming self-energies from the first site
#   `sigma_out` and `sigma_out_lesser` are boundary outgoing self-energies from the last site
#   `gl` and `gl_lesser` are left-connected Green functions
#   `start_iblock` and `end_iblock` are starting and ending block index of the partition
#   `inc` is the incremental direction of transport axis, + for going from 0 to L and - for opposite
def rgf_forward_pass(E,M,S,sigma_scat,sigma_scat_lesser,sigma_in,sigma_in_lesser,
                     start_iblock,end_iblock,num_blocks,block_size,
                     col_index,ind_ptr,nnz,num_diag):
    length = abs(end_iblock - start_iblock) - 1 
    inc = np.sign(end_iblock-start_iblock)
    gl = np.zeros((length,block_size,block_size),dtype='complex')
    gl_lesser = np.zeros((length,block_size,block_size),dtype='complex')
    sigma_out = np.zeros((block_size,block_size),dtype='complex')
    sigma_out_lesser = np.zeros((block_size,block_size),dtype='complex')
    z=E+0.0j
    H00=np.zeros((block_size,block_size),dtype='complex')    
    A=np.zeros((block_size,block_size),dtype='complex')
    B=np.zeros((block_size,block_size),dtype='complex')
    for ix in range(start_iblock,end_iblock+inc,inc):
        sig_ph_r = get_block_from_bcsr(sigma_scat,col_index,ind_ptr,nnz,block_size,
                            num_blocks,num_diag,dtype='complex',iblock=ix,idiag=0)
        sig_ph_l = get_block_from_bcsr(sigma_scat_lesser,col_index,ind_ptr,nnz,block_size,
                            num_blocks,num_diag,dtype='complex',iblock=ix,idiag=0)
        Hii = get_block_from_bcsr(M,col_index,ind_ptr,nnz,block_size,
                            num_blocks,num_diag,dtype='complex',iblock=ix,idiag=0)
        H1i = get_block_from_bcsr(M,col_index,ind_ptr,nnz,block_size,
                            num_blocks,num_diag,dtype='complex',iblock=ix,idiag=-inc)
        Sii = get_block_from_bcsr(S,col_index,ind_ptr,nnz,block_size,
                            num_blocks,num_diag,dtype='complex',iblock=ix,idiag=0)
        B = sig_ph_r @ Sii 
        H00 = Hii + B
        # $$H00 = H(i,i) + \Sigma_{ph}(i) * S(i,i)$$
        # $$Gl(i) = [E*S(i,i) - H00 - H(i,i-1) * Gl(i-1) * H(i-1,i)]^{-1}$$
        if (ix != start_iblock):
            B = H1i @ gl[ix-1,:,:] @ H1i.conj().T     
        else:
            B = sigma_in   

        if (ix == end_iblock):
            sigma_out = B
        else:
            A = z*Sii - H00 - B        
            gl[ix,:,:] = inv(A)   

        # $$Gln(i) = Gl(i) * [\Sigma_{ph}^<(i)*S(i,i) + H(i,i+1)*Gln(i+1)*H(i+1,i)] * Gl(i)^\dagger$$
        if (ix != start_iblock):
            B = H1i @ gl_lesser[ix-1,:,:] @ H1i.conj().T
        else:
            B = sigma_in_lesser

        if (ix == end_iblock):
            sigma_out_lesser = B            
        else:
            A = sig_ph_l @ Sii
            B = B + A
            gl_lesser[ix,:,:] = gl[ix,:,:] @ B @ gl[ix,:,:].conj().T

    return gl,gl_lesser,sigma_out,sigma_out_lesser


# backward pass of RGF
#   solve the fully-connected equation system: (ES-M-Sig^r@S)@G^r = I and G^< = G^r @ Sig^< @ (G^r)^H
#   
#   `gl` and `gl_lesser` are left-connected Green functions obtained from forward pass
#   `M` is system matrix in BCSR form
#   `G_l_prev` and `G_r_prev` are the fully-connected Green functions of the previous block of `start_iblock`
#   `start_iblock` and `end_iblock` are starting and ending block index of the partition
#   `G_retarded` and `G_lesser` and `G_greater` are fully-connected Green functions in BCSR form
def rgf_backward_pass(gl,gl_lesser,G_r_prev,G_l_prev,M,start_iblock,end_iblock,num_blocks,block_size,
                     col_index,ind_ptr,nnz,num_diag,
                     G_retarded,G_lesser,G_greater,cur):    
    inc = np.sign(end_iblock-start_iblock)
    A=np.zeros((block_size,block_size),dtype='complex')
    B=np.zeros((block_size,block_size),dtype='complex')
    GN0=np.zeros((block_size,block_size),dtype='complex')    
    G_l = G_l_prev
    G_r = G_r_prev
    G_g = np.zeros((block_size,block_size),dtype='complex')
    G_l_new = np.zeros((block_size,block_size),dtype='complex')
    G_r_new = np.zeros((block_size,block_size),dtype='complex')

    for ix in range(start_iblock,end_iblock+inc,inc):
        H1i = get_block_from_bcsr(M,col_index,ind_ptr,nnz,block_size,
                            num_blocks,num_diag,dtype='complex',iblock=ix-inc,idiag=inc)
        Hi1 = get_block_from_bcsr(M,col_index,ind_ptr,nnz,block_size,
                            num_blocks,num_diag,dtype='complex',iblock=ix,idiag=-inc)
        # $$A = G^<(i+1) * H(i+1,i) * Gl(i)^\dagger + G(i+1) * H(i+1,i) * Gln(i)$$        
        A = G_l @ H1i @ gl[ix,:,:].conj().T
        B = G_r @ H1i @ gl_lesser[ix,:,:]
        A += B        
        # $$B = H(i,i+1) * A$$
        # $$Jdens(i) = -2 * B$$
        B = Hi1 @ A
        cur[ix,:,:] = -2.0*B        
        # $$GN0 = Gl(i) * H(i,i+1) * G(i+1)$$
        # $$G(i) = Gl(i) + GN0 * H(i+1,i) * Gl(i)$$        
        B = gl[ix,:,:] @ Hi1
        GN0 = B @ G_r 
        A = GN0 @ H1i @ gl[ix,:,:]                
        G_r_new = gl[ix,:,:] + A
        
        # $$G^<(i) = Gln(i) + Gl(i) * H(i,i+1) * G^<(i+1) * H(i+1,i) *Gl(i)^\dagger$$
        B = gl[ix,:,:] @ Hi1 
        C = B @ G_l 
        A = C @ H1i 
        C = A @ gl[ix,:,:].conj().T
        G_l_new = gl_lesser[ix,:,:] + C            
        # $$G^<(i) = G^<(i) + GN0 * H(i+1,i) * Gln(i)$$
        B = GN0 @ H1i 
        C = B @ gl_lesser[ix,:,:]
        G_l_new +=  C            
        # $$G^<(i) = G^<(i) + Gln(i) * H(i,i+1) * GN0$$
        B = gl_lesser[ix,:,:] @ Hi1 
        C = B @ GN0.conj().T
        G_l_new +=  C
        
        # $$G^>(i) = G^<(i) + [G(i) - G(i)^\dagger]$$
        G_g = G_l + (G_r - G_r.conj().T)
        G_l = G_l_new
        G_r = G_r_new

        put_block_to_bcsr(G_retarded,col_index,ind_ptr,nnz,block_size,
                          num_blocks,num_diag,iblock=ix,idiag=0,mat=G_r)
        put_block_to_bcsr(G_lesser,col_index,ind_ptr,nnz,block_size,
                          num_blocks,num_diag,iblock=ix,idiag=0,mat=G_l)
        put_block_to_bcsr(G_greater,col_index,ind_ptr,nnz,block_size,
                          num_blocks,num_diag,iblock=ix,idiag=0,mat=G_g)
      
    return G_r,G_l


# RGF
#   driver for solving the Green functions: (ES-M-Sig^r@S)@G^r = I and G^< = G^r @ Sig^< @ (G^r)^H
#   
#   `E` is energy, `M` is system matrix, `S` is overlap matrix, 
#   `sigma_scat` and `sigma_scat_lesser` are scattering self-energies
#   `flavor` is the order to solve the RGF equation
#   `mu` and `temp` are arrays of chemical potentials and temperatures of the left and right contacts
#   `fd` is the statistical distribution function with respect to energy at a certain temperature and chemical potential, ie. Fermi-Dirac or Bose-Einstein or ...
#   `surface_green_function` is the surface Green's function solver
#   `num_blocks`, `block_size`, `col_index`, `ind_ptr`, `nnz`, `num_diag` are the BCSR parameters
#   
#   `G_retarded` and `G_lesser` and `G_greater` are fully-connected Green functions in BCSR form
#   `cur` is the current density flowing between adjacent blocks
def rgf(E,M,S,sigma_scat,sigma_scat_lesser,flavor,mu,temp,fd,surface_green_function,
        G_retarded,G_lesser,G_greater,cur,num_blocks,block_size,col_index,ind_ptr,nnz,num_diag):
    z = E + 0.0*1j
    # left boundary selfenergy
    ix=0
    inc=-1
    H00 = get_block_from_bcsr(M,col_index,ind_ptr,nnz,block_size,
                        num_blocks,num_diag,dtype='complex',iblock=ix,idiag=0)
    H10 = get_block_from_bcsr(M,col_index,ind_ptr,nnz,block_size,
                        num_blocks,num_diag,dtype='complex',iblock=ix,idiag=inc)
    S00 = get_block_from_bcsr(M,col_index,ind_ptr,nnz,block_size,
                                  num_blocks,num_diag,dtype='complex',iblock=ix,idiag=0)
    sig_ph_r = get_block_from_bcsr(sigma_scat,col_index,ind_ptr,nnz,block_size,
                                       num_blocks,num_diag,dtype='complex',iblock=ix,idiag=0)
    sig_ph_l = get_block_from_bcsr(sigma_scat_lesser,col_index,ind_ptr,nnz,block_size,
                                    num_blocks,num_diag,dtype='complex',iblock=ix,idiag=0)
    H00 += sig_ph_r @ S00
    g00 = surface_green_function(z,H00,H10,S00)
    sig_r_left = H10 @ g00 @ H10.conj().T
    sig_l_left = - (sig_r_left - sig_r_left.conjg().T) * fd(mu[0],temp[0]) # fluctuation–dissipation relation 
    # right boundary selfenergy
    ix=num_diag
    inc=1
    H00 = get_block_from_bcsr(M,col_index,ind_ptr,nnz,block_size,
                        num_blocks,num_diag,dtype='complex',iblock=ix,idiag=0)
    H10 = get_block_from_bcsr(M,col_index,ind_ptr,nnz,block_size,
                        num_blocks,num_diag,dtype='complex',iblock=ix,idiag=inc)
    S00 = get_block_from_bcsr(M,col_index,ind_ptr,nnz,block_size,
                                  num_blocks,num_diag,dtype='complex',iblock=ix,idiag=0)
    sig_ph_r = get_block_from_bcsr(sigma_scat,col_index,ind_ptr,nnz,block_size,
                                       num_blocks,num_diag,dtype='complex',iblock=ix,idiag=0)
    sig_ph_l = get_block_from_bcsr(sigma_scat_lesser,col_index,ind_ptr,nnz,block_size,
                                    num_blocks,num_diag,dtype='complex',iblock=ix,idiag=0)
    H00 += sig_ph_r @ S00
    g00 = surface_green_function(z,H00,H10,S00)
    sig_r_right = H10 @ g00 @ H10.conj().T
    sig_l_right = - (sig_r_right - sig_r_right.conjg().T) * fd(mu[1],temp[1]) # fluctuation–dissipation relation 
    if (flavor=='lrl'):
        # left-right-left         
        gl,gl_lesser,sigma_out,sigma_out_lesser = rgf_forward_pass(E,M,S,sigma_scat,sigma_scat_lesser,
                                                                   sigma_in=sig_r_left,sigma_in_lesser=sig_l_left,
                                                                   start_iblock=0,end_iblock=num_blocks,num_blocks=num_blocks,block_size=block_size,
                                                                   col_index=col_index,ind_ptr=ind_ptr,nnz=nnz,num_diag=num_diag)        
        # solve fully-connected GF for the ix block with full connections to left and right sides
        ix=num_blocks
        H00 = get_block_from_bcsr(M,col_index,ind_ptr,nnz,block_size,
                        num_blocks,num_diag,dtype='complex',iblock=ix,idiag=0)    
        S00 = get_block_from_bcsr(M,col_index,ind_ptr,nnz,block_size,
                                  num_blocks,num_diag,dtype='complex',iblock=ix,idiag=0)
        sig_ph_r = get_block_from_bcsr(sigma_scat,col_index,ind_ptr,nnz,block_size,
                                       num_blocks,num_diag,dtype='complex',iblock=ix,idiag=0)
        sig_ph_l = get_block_from_bcsr(sigma_scat_lesser,col_index,ind_ptr,nnz,block_size,
                                    num_blocks,num_diag,dtype='complex',iblock=ix,idiag=0)
        A = z*S00 - H00 - sigma_out - sig_r_right - sig_ph_r @ S00
        G_r = inv(A)
        sig = sigma_out_lesser
        sig += sig_l_right
        sig += sig_ph_l @ S00 
        G_l = G_r @ sig @ G_r.conjg().T
        G_g = G_l + (G_r - G_r.conj().T)        
        put_block_to_bcsr(G_retarded,col_index,ind_ptr,nnz,block_size,
                          num_blocks,num_diag,iblock=ix,idiag=0,mat=G_r)
        put_block_to_bcsr(G_lesser,col_index,ind_ptr,nnz,block_size,
                          num_blocks,num_diag,iblock=ix,idiag=0,mat=G_l)
        put_block_to_bcsr(G_greater,col_index,ind_ptr,nnz,block_size,
                          num_blocks,num_diag,iblock=ix,idiag=0,mat=G_g)
        # backward pass
        G_r,G_l = rgf_backward_pass(gl,gl_lesser,G_r,G_l,M,start_iblock=num_blocks-1,end_iblock=0,num_blocks=num_blocks,
                                    block_size=block_size,col_index=col_index,ind_ptr=ind_ptr,nnz=nnz,num_diag=num_diag,
                                    G_retarded=G_retarded,G_lesser=G_lesser,G_greater=G_greater,cur=cur)
    elif (flavor=='rlr'):
        # right-left-right        
        gl,gl_lesser,sigma_out,sigma_out_lesser = rgf_forward_pass(E,M,S,sigma_scat,sigma_scat_lesser,
                                                                   sigma_in=sig_r_right,sigma_in_lesser=sig_l_right,
                                                                   start_iblock=num_blocks,end_iblock=0,num_blocks=num_blocks,
                                                                   block_size=block_size,col_index=col_index,ind_ptr=ind_ptr,nnz=nnz,
                                                                   num_diag=num_diag)                
        # solve fully-connected GF for the ix block with full connections to left and right sides
        ix=0
        H00 = get_block_from_bcsr(M,col_index,ind_ptr,nnz,block_size,
                        num_blocks,num_diag,dtype='complex',iblock=ix,idiag=0)    
        S00 = get_block_from_bcsr(M,col_index,ind_ptr,nnz,block_size,
                                  num_blocks,num_diag,dtype='complex',iblock=ix,idiag=0)
        sig_ph_r = get_block_from_bcsr(sigma_scat,col_index,ind_ptr,nnz,block_size,
                                       num_blocks,num_diag,dtype='complex',iblock=ix,idiag=0)
        sig_ph_l = get_block_from_bcsr(sigma_scat_lesser,col_index,ind_ptr,nnz,block_size,
                                    num_blocks,num_diag,dtype='complex',iblock=ix,idiag=0)
        A = z*S00 - H00 - sigma_out - sig_r_left - sig_ph_r @ S00        
        G_r = inv(A)
        sig = sigma_out_lesser
        sig += sig_l_left
        sig += sig_ph_l @ S00 
        G_l = G_r @ sig @ G_r.conjg().T
        G_g = G_l + (G_r - G_r.conj().T)        
        put_block_to_bcsr(G_retarded,col_index,ind_ptr,nnz,block_size,
                          num_blocks,num_diag,iblock=ix,idiag=0,mat=G_r)
        put_block_to_bcsr(G_lesser,col_index,ind_ptr,nnz,block_size,
                          num_blocks,num_diag,iblock=ix,idiag=0,mat=G_l)
        put_block_to_bcsr(G_greater,col_index,ind_ptr,nnz,block_size,
                          num_blocks,num_diag,iblock=ix,idiag=0,mat=G_g)
        # backward pass
        G_r,G_l = rgf_backward_pass(gl,gl_lesser,G_r,G_l,M,start_iblock=1,end_iblock=num_blocks,
                                    num_blocks=num_blocks,block_size=block_size,col_index=col_index,
                                    ind_ptr=ind_ptr,nnz=nnz,num_diag=num_diag,
                                    G_retarded=G_retarded,G_lesser=G_lesser,G_greater=G_greater,cur=cur)
    elif (flavor=='2sided'):
        # 2-sided
        ix = num_blocks // 2
        gl1,gl_lesser1,sigma_out1,sigma_out_lesser1 = rgf_forward_pass(E,M,S,sigma_scat,sigma_scat_lesser,
                                                                   sigma_in=sig_r_left,sigma_in_lesser=sig_l_left,
                                                                   start_iblock=0,end_iblock=ix,num_blocks=num_blocks,
                                                                   block_size=block_size,col_index=col_index,ind_ptr=ind_ptr,nnz=nnz,
                                                                   num_diag=num_diag)    
        gl2,gl_lesser2,sigma_out2,sigma_out_lesser2 = rgf_forward_pass(E,M,S,sigma_scat,sigma_scat_lesser,
                                                                   sigma_in=sig_r_right,sigma_in_lesser=sig_l_right,
                                                                   start_iblock=num_blocks,end_iblock=ix,num_blocks=num_blocks,
                                                                   block_size=block_size,col_index=col_index,ind_ptr=ind_ptr,nnz=nnz,
                                                                   num_diag=num_diag)    
        # solve fully-connected GF for the ix block with full connections to left and right sides
        H00 = get_block_from_bcsr(M,col_index,ind_ptr,nnz,block_size,
                        num_blocks,num_diag,dtype='complex',iblock=ix,idiag=0)    
        S00 = get_block_from_bcsr(M,col_index,ind_ptr,nnz,block_size,
                                  num_blocks,num_diag,dtype='complex',iblock=ix,idiag=0)
        sig_ph_r = get_block_from_bcsr(sigma_scat,col_index,ind_ptr,nnz,block_size,
                                       num_blocks,num_diag,dtype='complex',iblock=ix,idiag=0)
        sig_ph_l = get_block_from_bcsr(sigma_scat_lesser,col_index,ind_ptr,nnz,block_size,
                                    num_blocks,num_diag,dtype='complex',iblock=ix,idiag=0)
        A = z*S00 - H00 - sigma_out1 - sigma_out2 - sig_ph_r @ S00        
        G_r = inv(A)
        sig = sigma_out_lesser1 + sigma_out_lesser2
        sig += sig_ph_l @ S00 
        G_l = G_r @ sig @ G_r.conjg().T
        G_g = G_l + (G_r - G_r.conj().T)        
        put_block_to_bcsr(G_retarded,col_index,ind_ptr,nnz,block_size,
                          num_blocks,num_diag,iblock=ix,idiag=0,mat=G_r)
        put_block_to_bcsr(G_lesser,col_index,ind_ptr,nnz,block_size,
                          num_blocks,num_diag,iblock=ix,idiag=0,mat=G_l)
        put_block_to_bcsr(G_greater,col_index,ind_ptr,nnz,block_size,
                          num_blocks,num_diag,iblock=ix,idiag=0,mat=G_g)
        # backward pass
        G_r1,G_l1 = rgf_backward_pass(gl1,gl_lesser1,G_r,G_l,M,start_iblock=ix-1,end_iblock=0,num_blocks=num_blocks,
                                    block_size=block_size,col_index=col_index,ind_ptr=ind_ptr,nnz=nnz,num_diag=num_diag,
                                    G_retarded=G_retarded,G_lesser=G_lesser,G_greater=G_greater,cur=cur)
        G_r2,G_l2 = rgf_backward_pass(gl2,gl_lesser2,G_r,G_l,M,start_iblock=ix+1,end_iblock=num_blocks,num_blocks=num_blocks,
                                    block_size=block_size,col_index=col_index,ind_ptr=ind_ptr,nnz=nnz,num_diag=num_diag,
                                    G_retarded=G_retarded,G_lesser=G_lesser,G_greater=G_greater,cur=cur)
    return 
    