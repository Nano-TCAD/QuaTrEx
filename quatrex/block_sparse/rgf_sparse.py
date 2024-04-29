import numpy as np    
from typing import Callable

from numpy.linalg import inv
from bcsr_matrix import get_block_from_bcsr,put_block_to_bcsr
 
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
                     start_iblock,end_iblock,num_blocks,block_sizes,
                     col_index,ind_ptr,nnz,num_diag):
    length = abs(end_iblock - start_iblock) - 1 
    inc = np.sign(end_iblock-start_iblock)
    max_blocksize=np.max(block_sizes)
    gl = np.zeros((length,max_blocksize,max_blocksize),dtype='complex')
    gl_lesser = np.zeros((length,max_blocksize,max_blocksize),dtype='complex')
    sigma_out = np.zeros((max_blocksize,max_blocksize),dtype='complex')
    sigma_out_lesser = np.zeros((max_blocksize,max_blocksize),dtype='complex')
    z=E+0.0j
    H00=np.zeros((max_blocksize,max_blocksize),dtype='complex')    
    A=np.zeros((max_blocksize,max_blocksize),dtype='complex')
    B=np.zeros((max_blocksize,max_blocksize),dtype='complex')
    for ix in range(start_iblock,end_iblock+inc,inc):
        sig_ph_r = get_block_from_bcsr(sigma_scat,col_index,ind_ptr,nnz,block_sizes,
                            num_blocks,num_diag,dtype='complex',iblock=ix,idiag=0)
        sig_ph_l = get_block_from_bcsr(sigma_scat_lesser,col_index,ind_ptr,nnz,block_sizes,
                            num_blocks,num_diag,dtype='complex',iblock=ix,idiag=0)
        Hii = get_block_from_bcsr(M,col_index,ind_ptr,nnz,block_sizes,
                            num_blocks,num_diag,dtype='complex',iblock=ix,idiag=0)
        H1i = get_block_from_bcsr(M,col_index,ind_ptr,nnz,block_sizes,
                            num_blocks,num_diag,dtype='complex',iblock=ix,idiag=-inc)
        Sii = get_block_from_bcsr(S,col_index,ind_ptr,nnz,block_sizes,
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
def rgf_backward_pass(gl,gl_lesser,G_r_prev,G_l_prev,M,start_iblock,end_iblock,num_blocks,block_sizes,
                     col_index,ind_ptr,nnz,num_diag,
                     G_retarded,G_lesser,G_greater,cur):    
    inc = np.sign(end_iblock-start_iblock)
    max_blocksize = np.max(block_sizes)
    # A=np.zeros((max_blocksize,max_blocksize),dtype='complex')
    # B=np.zeros((max_blocksize,max_blocksize),dtype='complex')
    # GN0=np.zeros((max_blocksize,max_blocksize),dtype='complex')    
    G_l = G_l_prev
    G_r = G_r_prev
    G_g = np.zeros((max_blocksize,max_blocksize),dtype='complex')
    G_l_new = np.zeros((max_blocksize,max_blocksize),dtype='complex')
    G_r_new = np.zeros((max_blocksize,max_blocksize),dtype='complex')

    for ix in range(start_iblock,end_iblock+inc,inc):
        H1i = get_block_from_bcsr(M,col_index,ind_ptr,nnz,block_sizes,
                            num_blocks,num_diag,dtype='complex',iblock=ix-inc,idiag=inc)
        Hi1 = get_block_from_bcsr(M,col_index,ind_ptr,nnz,block_sizes,
                            num_blocks,num_diag,dtype='complex',iblock=ix,idiag=-inc)
        # $$A = G^<(i+1) * H(i+1,i) * Gl(i)^\dagger + G(i+1) * H(i+1,i) * Gln(i)$$        
        A = G_l @ H1i @ gl[ix,:,:].conj().T
        B = G_r @ H1i @ gl_lesser[ix,:,:]
        A += B        
        # $$B = H(i,i+1) * A$$
        # $$Jdens(i) = -2 * B$$
        B = Hi1 @ A
        cur[ix] = -2.0*np.trace(B)        
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

        put_block_to_bcsr(G_retarded,col_index,ind_ptr,nnz,block_sizes,
                          num_blocks,num_diag,iblock=ix,idiag=0,mat=G_r)
        put_block_to_bcsr(G_lesser,col_index,ind_ptr,nnz,block_sizes,
                          num_blocks,num_diag,iblock=ix,idiag=0,mat=G_l)
        put_block_to_bcsr(G_greater,col_index,ind_ptr,nnz,block_sizes,
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
def rgf(E,M,S,sigma_scat,sigma_scat_lesser,flavor,sigma_boundary_retarded,sigma_boundary_lesser,
        G_retarded,G_lesser,G_greater,cur,num_blocks,block_sizes,col_index,ind_ptr,nnz,num_diag,do_backward_pass):
    z = E + 0.0*1j
    # left boundary selfenergy    
    sig_r_left = sigma_boundary_retarded[:,:,0]
    sig_l_left = sigma_boundary_lesser[:,:,0]
    # right boundary selfenergy    
    sig_r_right = sigma_boundary_retarded[:,:,-1]
    sig_l_right = sigma_boundary_lesser[:,:,-1] 
    if (flavor=='lrl'):
        # left-right-left         
        gl,gl_lesser,sigma_out,sigma_out_lesser = rgf_forward_pass(E,M,S,sigma_scat,sigma_scat_lesser,
                                                                   sigma_in=sig_r_left,sigma_in_lesser=sig_l_left,
                                                                   start_iblock=0,end_iblock=num_blocks,num_blocks=num_blocks,block_size=block_size,
                                                                   col_index=col_index,ind_ptr=ind_ptr,nnz=nnz,num_diag=num_diag)        
        # solve fully-connected GF for the ix block with full connections to left and right sides
        ix=num_blocks
        H00 = get_block_from_bcsr(M,col_index,ind_ptr,nnz,block_sizes,
                        num_blocks,num_diag,dtype='complex',iblock=ix,idiag=0)    
        S00 = get_block_from_bcsr(M,col_index,ind_ptr,nnz,block_sizes,
                                  num_blocks,num_diag,dtype='complex',iblock=ix,idiag=0)
        sig_ph_r = get_block_from_bcsr(sigma_scat,col_index,ind_ptr,nnz,block_sizes,
                                       num_blocks,num_diag,dtype='complex',iblock=ix,idiag=0)
        sig_ph_l = get_block_from_bcsr(sigma_scat_lesser,col_index,ind_ptr,nnz,block_sizes,
                                    num_blocks,num_diag,dtype='complex',iblock=ix,idiag=0)
        A = z*S00 - H00 - sigma_out - sig_r_right - sig_ph_r @ S00
        G_r = inv(A)
        sig = sigma_out_lesser
        sig += sig_l_right
        sig += sig_ph_l @ S00 
        G_l = G_r @ sig @ G_r.conjg().T
        G_g = G_l + (G_r - G_r.conj().T)        
        put_block_to_bcsr(G_retarded,col_index,ind_ptr,nnz,block_sizes,
                          num_blocks,num_diag,iblock=ix,idiag=0,mat=G_r)
        put_block_to_bcsr(G_lesser,col_index,ind_ptr,nnz,block_sizes,
                          num_blocks,num_diag,iblock=ix,idiag=0,mat=G_l)
        put_block_to_bcsr(G_greater,col_index,ind_ptr,nnz,block_sizes,
                          num_blocks,num_diag,iblock=ix,idiag=0,mat=G_g)
        # backward pass
        if (do_backward_pass):
            G_r,G_l = rgf_backward_pass(gl,gl_lesser,G_r,G_l,M,start_iblock=num_blocks-1,end_iblock=0,num_blocks=num_blocks,
                                    block_sizes=block_sizes,col_index=col_index,ind_ptr=ind_ptr,nnz=nnz,num_diag=num_diag,
                                    G_retarded=G_retarded,G_lesser=G_lesser,G_greater=G_greater,cur=cur)
    elif (flavor=='rlr'):
        # right-left-right        
        gl,gl_lesser,sigma_out,sigma_out_lesser = rgf_forward_pass(E,M,S,sigma_scat,sigma_scat_lesser,
                                                                   sigma_in=sig_r_right,sigma_in_lesser=sig_l_right,
                                                                   start_iblock=num_blocks,end_iblock=0,num_blocks=num_blocks,
                                                                   block_sizes=block_sizes,col_index=col_index,ind_ptr=ind_ptr,nnz=nnz,
                                                                   num_diag=num_diag)                
        # solve fully-connected GF for the ix block with full connections to left and right sides
        ix=0
        H00 = get_block_from_bcsr(M,col_index,ind_ptr,nnz,block_sizes,
                        num_blocks,num_diag,dtype='complex',iblock=ix,idiag=0)    
        S00 = get_block_from_bcsr(M,col_index,ind_ptr,nnz,block_sizes,
                                  num_blocks,num_diag,dtype='complex',iblock=ix,idiag=0)
        sig_ph_r = get_block_from_bcsr(sigma_scat,col_index,ind_ptr,nnz,block_sizes,
                                       num_blocks,num_diag,dtype='complex',iblock=ix,idiag=0)
        sig_ph_l = get_block_from_bcsr(sigma_scat_lesser,col_index,ind_ptr,nnz,block_sizes,
                                    num_blocks,num_diag,dtype='complex',iblock=ix,idiag=0)
        A = z*S00 - H00 - sigma_out - sig_r_left - sig_ph_r @ S00        
        G_r = inv(A)
        sig = sigma_out_lesser
        sig += sig_l_left
        sig += sig_ph_l @ S00 
        G_l = G_r @ sig @ G_r.conjg().T
        G_g = G_l + (G_r - G_r.conj().T)        
        put_block_to_bcsr(G_retarded,col_index,ind_ptr,nnz,block_sizes,
                          num_blocks,num_diag,iblock=ix,idiag=0,mat=G_r)
        put_block_to_bcsr(G_lesser,col_index,ind_ptr,nnz,block_sizes,
                          num_blocks,num_diag,iblock=ix,idiag=0,mat=G_l)
        put_block_to_bcsr(G_greater,col_index,ind_ptr,nnz,block_sizes,
                          num_blocks,num_diag,iblock=ix,idiag=0,mat=G_g)
        # backward pass
        if (do_backward_pass):
            G_r,G_l = rgf_backward_pass(gl,gl_lesser,G_r,G_l,M,start_iblock=1,end_iblock=num_blocks,
                                    num_blocks=num_blocks,block_sizes=block_sizes,col_index=col_index,
                                    ind_ptr=ind_ptr,nnz=nnz,num_diag=num_diag,
                                    G_retarded=G_retarded,G_lesser=G_lesser,G_greater=G_greater,cur=cur)
    elif (flavor=='2sided'):
        # 2-sided
        ix = num_blocks // 2
        gl1,gl_lesser1,sigma_out1,sigma_out_lesser1 = rgf_forward_pass(E,M,S,sigma_scat,sigma_scat_lesser,
                                                                   sigma_in=sig_r_left,sigma_in_lesser=sig_l_left,
                                                                   start_iblock=0,end_iblock=ix,num_blocks=num_blocks,
                                                                   block_sizes=block_sizes,col_index=col_index,ind_ptr=ind_ptr,nnz=nnz,
                                                                   num_diag=num_diag)    
        gl2,gl_lesser2,sigma_out2,sigma_out_lesser2 = rgf_forward_pass(E,M,S,sigma_scat,sigma_scat_lesser,
                                                                   sigma_in=sig_r_right,sigma_in_lesser=sig_l_right,
                                                                   start_iblock=num_blocks,end_iblock=ix,num_blocks=num_blocks,
                                                                   block_sizes=block_sizes,col_index=col_index,ind_ptr=ind_ptr,nnz=nnz,
                                                                   num_diag=num_diag)    
        # solve fully-connected GF for the ix block with full connections to left and right sides
        H00 = get_block_from_bcsr(M,col_index,ind_ptr,nnz,block_sizes,
                        num_blocks,num_diag,dtype='complex',iblock=ix,idiag=0)    
        S00 = get_block_from_bcsr(M,col_index,ind_ptr,nnz,block_sizes,
                                  num_blocks,num_diag,dtype='complex',iblock=ix,idiag=0)
        sig_ph_r = get_block_from_bcsr(sigma_scat,col_index,ind_ptr,nnz,block_sizes,
                                       num_blocks,num_diag,dtype='complex',iblock=ix,idiag=0)
        sig_ph_l = get_block_from_bcsr(sigma_scat_lesser,col_index,ind_ptr,nnz,block_sizes,
                                    num_blocks,num_diag,dtype='complex',iblock=ix,idiag=0)
        A = z*S00 - H00 - sigma_out1 - sigma_out2 - sig_ph_r @ S00        
        G_r = inv(A)
        sig = sigma_out_lesser1 + sigma_out_lesser2
        sig += sig_ph_l @ S00 
        G_l = G_r @ sig @ G_r.conjg().T
        G_g = G_l + (G_r - G_r.conj().T)        
        put_block_to_bcsr(G_retarded,col_index,ind_ptr,nnz,block_sizes,
                          num_blocks,num_diag,iblock=ix,idiag=0,mat=G_r)
        put_block_to_bcsr(G_lesser,col_index,ind_ptr,nnz,block_sizes,
                          num_blocks,num_diag,iblock=ix,idiag=0,mat=G_l)
        put_block_to_bcsr(G_greater,col_index,ind_ptr,nnz,block_sizes,
                          num_blocks,num_diag,iblock=ix,idiag=0,mat=G_g)
        # backward pass
        if (do_backward_pass):
            G_r1,G_l1 = rgf_backward_pass(gl1,gl_lesser1,G_r,G_l,M,start_iblock=ix-1,end_iblock=0,num_blocks=num_blocks,
                                    block_sizes=block_sizes,col_index=col_index,ind_ptr=ind_ptr,nnz=nnz,num_diag=num_diag,
                                    G_retarded=G_retarded,G_lesser=G_lesser,G_greater=G_greater,cur=cur)
            G_r2,G_l2 = rgf_backward_pass(gl2,gl_lesser2,G_r,G_l,M,start_iblock=ix+1,end_iblock=num_blocks,num_blocks=num_blocks,
                                    block_sizes=block_sizes,col_index=col_index,ind_ptr=ind_ptr,nnz=nnz,num_diag=num_diag,
                                    G_retarded=G_retarded,G_lesser=G_lesser,G_greater=G_greater,cur=cur)
    return 
    