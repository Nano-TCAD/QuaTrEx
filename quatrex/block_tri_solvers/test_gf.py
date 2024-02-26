import cupy as cp
import cupyx as cpx
import numpy as np
import scipy.sparse as sp

from quatrex.utils.change_format import (sparse2block_energyhamgen_no_map, sparse2block_energy_forbatchedblockwise,
                                         block2sparse_energy_alt)
from quatrex.utils.matrix_creation import initialize_block_sigma_batched, initialize_block_G_batched


def random_complex(shape, rng: np.random.Generator):
    return rng.random(shape) + 1j * rng.random(shape)


def map_to_mapping(map, num_blocks):
    mapping = [None for _ in range(num_blocks)]
    for block_id in range(num_blocks):
        block_indices = np.nonzero(map[0] == block_id)[0]
        block_map = np.empty((3, len(block_indices)), dtype=np.int32)
        block_map[0] = map[3][block_indices]  # data indices
        block_map[1] = map[1][block_indices]  # rows
        block_map[2] = map[2][block_indices]  # cols
        mapping[block_id] = block_map
    return mapping


def canonicalize_csr(csr: sp.csr_matrix):
    result = csr
    if not isinstance(csr, sp.csr_matrix):
        result = csr.tocsr()
    result.eliminate_zeros()
    result.sum_duplicates()
    result.sort_indices()
    result.has_canonical_format = True
    return result


def csr_to_block_tridiagonal_csr(csr: sp.csr_matrix, bmin, bmax):
    num_blocks = len(bmin)
    block_diag = [None for _ in range(num_blocks)]
    block_upper = [None for _ in range(num_blocks - 1)]
    block_lower = [None for _ in range(num_blocks - 1)]

    for block_id in range(num_blocks):
        diag_slice = slice(bmin[block_id], bmax[block_id])
        block_diag[block_id] = canonicalize_csr(csr[diag_slice, diag_slice])
        if block_id < num_blocks - 1:
            upper_slice = lower_slice = slice(bmin[block_id + 1], bmax[block_id + 1])
            block_upper[block_id] = canonicalize_csr(csr[diag_slice, upper_slice])
            block_lower[block_id] = canonicalize_csr(csr[lower_slice, diag_slice])

    return block_diag, block_upper, block_lower


def validate_hamiltonian(energies,
                         hamiltonian_diag, hamiltonian_upper, hamiltonian_lower,
                         overlap_diag, overlap_upper, overlap_lower,
                         hamiltonian_diag_dense, hamiltonian_upper_dense, hamiltonian_lower_dense):
    for ie, energy in enumerate(energies):
        for block_id in range(len(hamiltonian_diag)):
            block_diag = (energy + 1j * 1e-12) * overlap_diag[block_id] - hamiltonian_diag[block_id]
            assert np.allclose(block_diag.toarray(), hamiltonian_diag_dense[block_id, ie])
            if block_id < len(hamiltonian_upper):
                block_upper = (energy + 1j * 1e-12) * overlap_upper[block_id] - hamiltonian_upper[block_id]
                assert np.allclose(block_upper.toarray(), hamiltonian_upper_dense[block_id, ie])
                block_lower = (energy + 1j * 1e-12) * overlap_lower[block_id] - hamiltonian_lower[block_id]
                assert np.allclose(block_lower.toarray(), hamiltonian_lower_dense[block_id, ie])
    
    print("Hamiltonian validated")


def validate_self_energies(energies,
                           self_energies_retarded, self_energies_lesser, self_energies_greater,
                           sigma_retarded_boundary_left, sigma_retarded_boundary_right,
                           sigma_lesser_boundary_left, sigma_lesser_boundary_right,
                           sigma_greater_boundary_left, sigma_greater_boundary_right,
                           mapping_diag, mapping_upper, mapping_lower,
                           sr_blco_diag, sr_blco_upper, sr_blco_lower,
                           sl_blco_diag, sl_blco_upper, sl_blco_lower,
                           sg_blco_diag, sg_blco_upper, sg_blco_lower):
    num_blocks, num_energies, block_size, _ = sr_blco_diag.shape
    sigma_dense = np.empty((num_energies, block_size, block_size), dtype=np.complex128)


    for block_id in range(num_blocks):

        vals = mapping_diag[block_id][0]
        rows = mapping_diag[block_id][1]
        cols = mapping_diag[block_id][2]

        for sigma, bc_left, bc_right, ref in (
                (self_energies_retarded, sigma_retarded_boundary_left, sigma_retarded_boundary_right, sr_blco_diag),
                (self_energies_lesser, sigma_lesser_boundary_left, sigma_lesser_boundary_right, sl_blco_diag),
                (self_energies_greater, sigma_greater_boundary_left, sigma_greater_boundary_right, sg_blco_diag)):
            sigma_dense[:] = 0
            sigma_dense[:, rows, cols] = sigma[:, vals]
            if block_id == 0:
                sigma_dense += bc_left
            elif block_id == num_blocks - 1:
                sigma_dense += bc_right
            assert np.allclose(sigma_dense, ref[block_id])

        if block_id < num_blocks - 1:

            vals = mapping_upper[block_id][0]
            rows = mapping_upper[block_id][1]
            cols = mapping_upper[block_id][2]
            for sigma, ref in ((self_energies_retarded, sr_blco_upper),
                               (self_energies_lesser, sl_blco_upper),
                               (self_energies_greater, sg_blco_upper)):
                sigma_dense[:] = 0
                sigma_dense[:, rows, cols] = sigma[:, vals]
                assert np.allclose(sigma_dense, ref[block_id])

            vals = mapping_lower[block_id][0]
            rows = mapping_lower[block_id][1]
            cols = mapping_lower[block_id][2]
            for sigma, ref in ((self_energies_retarded, sr_blco_lower),
                               (self_energies_lesser, sl_blco_lower),
                               (self_energies_greater, sg_blco_lower)):
                sigma_dense[:] = 0
                sigma_dense[:, rows, cols] = sigma[:, vals]
                assert np.allclose(sigma_dense, ref[block_id])

    print("Self-energies validated")

def validate_system_matrix(energies,
                           hamiltonian_diag, hamiltonian_upper, hamiltonian_lower,
                           overlap_diag, overlap_upper, overlap_lower,
                           self_energies_retarded, sigma_retarded_boundary_left, sigma_retarded_boundary_right,
                           mapping_diag, mapping_upper, mapping_lower,
                           system_matrix_diag_dense, system_matrix_upper_dense, system_matrix_lower_dense):
    num_blocks, num_energies, block_size, _ = system_matrix_diag_dense.shape
    sigma_dense = np.empty((num_energies, block_size, block_size), dtype=np.complex128)

    for block_id in range(num_blocks):

        H = hamiltonian_diag[block_id].toarray()
        S = overlap_diag[block_id].toarray()
        block_diag = (energies + 1j * 1e-12).reshape(num_energies, 1, 1) * S.reshape(1, *S.shape) - H
        vals = mapping_diag[block_id][0]
        rows = mapping_diag[block_id][1]
        cols = mapping_diag[block_id][2]
        sigma_dense[:] = 0
        sigma_dense[:, rows, cols] = self_energies_retarded[:, vals]
        if block_id == 0:
            sigma_dense += sigma_retarded_boundary_left
        elif block_id == num_blocks - 1:
            sigma_dense += sigma_retarded_boundary_right
        block_diag -= sigma_dense
        assert np.allclose(block_diag, system_matrix_diag_dense[block_id])

        if block_id < num_blocks - 1:

            H = hamiltonian_upper[block_id].toarray()
            S = overlap_upper[block_id].toarray()
            block_upper = (energies + 1j * 1e-12).reshape(num_energies, 1, 1) * S.reshape(1, *S.shape) - H
            vals = mapping_upper[block_id][0]
            rows = mapping_upper[block_id][1]
            cols = mapping_upper[block_id][2]
            sigma_dense[:] = 0
            sigma_dense[:, rows, cols] = self_energies_retarded[:, vals]
            block_upper -= sigma_dense
            assert np.allclose(block_upper, system_matrix_upper_dense[block_id])

            H = hamiltonian_lower[block_id].toarray()
            S = overlap_lower[block_id].toarray()
            block_lower = (energies + 1j * 1e-12).reshape(num_energies, 1, 1) * S.reshape(1, *S.shape) - H
            vals = mapping_lower[block_id][0]
            rows = mapping_lower[block_id][1]
            cols = mapping_lower[block_id][2]
            sigma_dense[:] = 0
            sigma_dense[:, rows, cols] = self_energies_retarded[:, vals]
            block_lower -= sigma_dense
            assert np.allclose(block_lower, system_matrix_lower_dense[block_id])

    print("System matrix validated")


def self_energy_preprocess_2d(sl, sg, sr, sl_phn, sg_phn, sr_phn, rows, columns, ij2ji):
    sl[:] = (sl - sl[:, ij2ji].conj()) / 2
    sg[:] = (sg - sg[:, ij2ji].conj()) / 2
    sr[:] = np.real(sr) + (sg - sl) / 2

    sl[:, rows == columns] += sl_phn
    sg[:, rows == columns] += sg_phn
    sr[:, rows == columns] += sr_phn


def rgf_standaloneGF_batched_GPU(
           ham_diag,
           ham_upper,
           ham_lower,
           sg_diag,
           sg_upper,
           sg_lower,
           sl_diag,
           sl_upper,
           sl_lower,
           SigGBR,
           SigLBR,
           GR,
           GRnn1,
           GL,
           GLnn1,
           GG,
           GGnn1,
           DOS,
           nE,
           nP,
           idE,
           Bmin_fi,
           Bmax_fi
):
    # rgf_GF(DH, E, EfL, EfR, Temp) This could be the function call considering Leo's code
    '''
    Working!
    
    '''
    Bmax = Bmax_fi - 1
    Bmin = Bmin_fi - 1
    Bsize = max(Bmax - Bmin + 1)  # Used for declaration of variables
    NB = len(Bmin)
    NT = Bmax[NB - 1] + 1  # Not used in this fcn

    LBsize = Bmax[0] - Bmin[0] + 1
    RBsize = Bmax[NB - 1] - Bmin[NB - 1] + 1

    energy_batchsize = ham_diag.shape[1]

    # # Upload to GPU
    # ham_diag_gpu = cp.asarray(ham_diag)
    # ham_upper_gpu = cp.asarray(ham_upper)
    # ham_lower_gpu = cp.asarray(ham_lower)

    # sg_diag_gpu = cp.asarray(sg_diag)
    # sg_upper_gpu = cp.asarray(sg_upper)
    # sg_lower_gpu = cp.asarray(sg_lower)

    # sl_diag_gpu = cp.asarray(sl_diag)
    # sl_upper_gpu = cp.asarray(sl_upper)
    # sl_lower_gpu = cp.asarray(sl_lower)

    ham_diag_gpu = cp.empty_like(ham_diag)
    ham_upper_gpu = cp.empty_like(ham_upper)
    ham_lower_gpu = cp.empty_like(ham_lower)

    sg_diag_gpu = cp.empty_like(sg_diag)
    sg_upper_gpu = cp.empty_like(sg_upper)
    sg_lower_gpu = cp.empty_like(sg_lower)

    sl_diag_gpu = cp.empty_like(sl_diag)
    sl_upper_gpu = cp.empty_like(sl_upper)
    sl_lower_gpu = cp.empty_like(sl_lower)

    for i in range(len(ham_diag)):
        if i < len(ham_diag) - 1:
            ham_upper_gpu[i].set(ham_upper[i])
            ham_lower_gpu[i].set(ham_lower[i])

            sl_upper_gpu[i].set(sl_upper[i])
            sl_lower_gpu[i].set(sl_lower[i])

            sg_upper_gpu[i].set(sg_upper[i])
            sg_lower_gpu[i].set(sg_lower[i])
            
        sl_diag_gpu[i].set(sl_diag[i])            
        ham_diag_gpu[i].set(ham_diag[i])
        sg_diag_gpu[i].set(sg_diag[i])
    
    SigGBR_gpu = cp.asarray(SigGBR)
    SigLBR_gpu = cp.asarray(SigLBR)

    gR_gpu = cp.zeros((NB, energy_batchsize, Bsize, Bsize), dtype=cp.cfloat)  # Retarded (right)
    gL_gpu = cp.zeros((NB, energy_batchsize, Bsize, Bsize), dtype=cp.cfloat)  # Lesser (right)
    gG_gpu = cp.zeros((NB, energy_batchsize, Bsize, Bsize), dtype=cp.cfloat)  # Greater (right)
    SigLB_gpu = cp.zeros((NB - 1, energy_batchsize, Bsize, Bsize), dtype=cp.cfloat)  # Lesser boundary self-energy
    SigGB_gpu = cp.zeros((NB - 1, energy_batchsize, Bsize, Bsize), dtype=cp.cfloat)  # Greater boundary self-energy

    # IdE = np.zeros((NB, energy_batchsize))
    # n = np.zeros((NB, energy_batchsize))
    # p = np.zeros((NB, energy_batchsize))

    # First step of iteration
    NN = Bmax[-1] - Bmin[-1] + 1
    gpu_identity = cp.identity(NN, dtype = cp.cfloat)
    gpu_identity_batch = cp.repeat(gpu_identity[cp.newaxis, :, :], energy_batchsize, axis = 0)
    print(ham_diag_gpu.shape)
    print(gpu_identity_batch.shape)
    gR_gpu[-1, :, 0:NN, 0:NN] = cp.linalg.solve(ham_diag_gpu[-1, :, 0:NN, 0:NN], gpu_identity_batch)
    #gR_gpu[-1, :, 0:NN, 0:NN] = cp.linalg.inv(ham_diag_gpu[-1, :, 0:NN, 0:NN])
    gL_gpu[-1, :, 0:NN, 0:NN] = gR_gpu[-1, :, 0:NN, 0:NN] @ (sl_diag_gpu[-1, :, 0:NN, 0:NN]) @ gR_gpu[-1, :, 0:NN, 0:NN].conjugate().transpose((0,2,1))
    gG_gpu[-1, :, 0:NN, 0:NN] = gR_gpu[-1, :, 0:NN, 0:NN] @ (sg_diag_gpu[-1, :, 0:NN, 0:NN]) @ gR_gpu[-1, :, 0:NN, 0:NN].conjugate().transpose((0,2,1))
    
    for IB in range(NB - 2, -1, -1):
        NI = Bmax[IB] - Bmin[IB] + 1
        NP = Bmax[IB + 1] - Bmin[IB + 1] + 1

        # # Extracting diagonal Hamiltonian block
        # if(IB == 0):
        #     M_c = M[Bmin[IB]:Bmax[IB] + 1, Bmin[IB]:Bmax[IB] + 1].toarray() - SigRBL
        #     SigL_c = SigL[Bmin[IB]:Bmax[IB] + 1, Bmin[IB]:Bmax[IB] + 1].toarray() + SigLBL
        #     SigG_c = SigG[Bmin[IB]:Bmax[IB] + 1, Bmin[IB]:Bmax[IB] + 1].toarray() + SigGBL
        # else: 
        #     M_c = M[Bmin[IB]:Bmax[IB] + 1, Bmin[IB]:Bmax[IB] + 1].toarray()
        #     SigL_c = SigL[Bmin[IB]:Bmax[IB] + 1, Bmin[IB]:Bmax[IB] + 1].toarray()
        #     SigG_c = SigG[Bmin[IB]:Bmax[IB] + 1, Bmin[IB]:Bmax[IB] + 1].toarray()
        # #M_c = M[Bmin[IB]:Bmax[IB] + 1, Bmin[IB]:Bmax[IB] + 1].toarray()

        # # Extracting off-diagonal Hamiltonian block (right)
        # M_r = M[Bmin[IB]:Bmax[IB] + 1, Bmin[IB + 1]:Bmax[IB + 1] + 1].toarray()

        # # Extracting off-diagonal Hamiltonian block (lower)
        # M_d = M[Bmin[IB + 1]:Bmax[IB + 1] + 1, Bmin[IB]:Bmax[IB] + 1].toarray()

        # # Extracting diagonal lesser Self-energy block
        # #SigL_c = SigL[Bmin[IB]:Bmax[IB] + 1, Bmin[IB]:Bmax[IB] + 1].toarray()

        # # Extracting off-diagonal lesser Self-energy block (lower)
        # SigL_l = SigL[Bmin[IB + 1]:Bmax[IB + 1] + 1, Bmin[IB]:Bmax[IB] + 1].toarray()

        # # Extracting diagonal greater Self-energy block
        # #SigG_c = SigG[Bmin[IB]:Bmax[IB] + 1, Bmin[IB]:Bmax[IB] + 1].toarray()

        # # Extracting off-diagonal greater Self-energy block (lower)
        # SigG_l = SigG[Bmin[IB + 1]:Bmax[IB + 1] + 1, Bmin[IB]:Bmax[IB] + 1].toarray()

        gpu_identity = cp.identity(NI, dtype = cp.cfloat)
        gpu_identity_batch = cp.repeat(gpu_identity[cp.newaxis, :, :], energy_batchsize, axis = 0)
       # gR_gpu[IB, :, 0:NI, 0:NI] = cp.linalg.inv(ham_diag_gpu[IB, :, 0:NN, 0:NN] \
       #                                           - ham_upper_gpu[IB, :, 0:NI, 0:NP] \
       #                                           @ gR_gpu[IB+1, :, 0:NP, 0:NP] \
       #                                           @ ham_lower_gpu[IB, :, :NP, 0:NI])
        gR_gpu[IB, :, 0:NI, 0:NI] = cp.linalg.solve(ham_diag_gpu[IB, :, 0:NN, 0:NN] \
                             - ham_upper_gpu[IB, :, 0:NI, 0:NP] \
                             @ gR_gpu[IB+1, :, 0:NP, 0:NP] \
                         @ ham_lower_gpu[IB, :, :NP, 0:NI], gpu_identity_batch)#######
        # AL, What is this? Handling off-diagonal sigma elements?
        AL = ham_upper_gpu[IB, :, 0:NI, 0:NP] \
            @ gR_gpu[IB+1, :, 0:NP, 0:NP] \
            @ sl_lower_gpu[IB, :, 0:NP, 0:NI]
        
        SigLB_gpu[IB, :, 0:NI, 0:NI] = ham_upper_gpu[IB, :, 0:NI, 0:NP] \
                            @ gL_gpu[IB+1, :, 0:NP, 0:NP] \
                            @ ham_upper_gpu[IB, :, 0:NI, 0:NP].transpose((0,2,1)).conj() \
                            - (AL - AL.transpose((0,2,1)).conj())

        # gL[IB, 0:NI, 0:NI] = gR[IB, 0:NI, 0:NI] \
        #                     @ (SigL_c \
        #                     + M_r \
        #                     @ gL[IB+1, 0:NP, 0:NP] \
        #                     @ M_r.T.conj() \
        #                     - (AL - AL.T.conj()))  \
        #                     @ gR[IB, 0:NI, 0:NI].T.conj() # Confused about the AL

        gL_gpu[IB, :, 0:NI, 0:NI] = gR_gpu[IB, :, 0:NI, 0:NI] \
                            @ (sl_diag_gpu[IB, :, 0:NI, 0:NI] \
                            + SigLB_gpu[IB, :, 0:NI, 0:NI])  \
                            @ gR_gpu[IB, :, 0:NI, 0:NI].transpose((0, 2, 1)).conj() # Confused about the AL

        ### What is this?
        AG = ham_upper_gpu[IB, :, 0:NI, 0:NP] \
            @ gR_gpu[IB+1, :, 0:NP, 0:NP] \
            @ sg_lower_gpu[IB, :, 0:NP, 0:NI]     # Handling off-diagonal sigma elements? Prob. need to check

        # gG[IB, 0:NI, 0:NI] = gR[IB, 0:NI, 0:NI] \
        #                     @ (SigG_c \
        #                     + M_r \
        #                     @ gG[IB+1, 0:NP, 0:NP] \
        #                     @ M_r.T.conj() \
        #                     - (AG - AG.T.conj())) \
        #                     @ gR[IB, 0:NI, 0:NI].T.conj() # Confused about the AG. 
        SigGB_gpu[IB, :, 0:NI, 0:NI] = ham_upper_gpu[IB, :, 0:NI, 0:NP] \
                            @ gG_gpu[IB+1, :,  0:NP, 0:NP] \
                            @ ham_upper_gpu[IB, :, 0:NI, 0:NP].transpose((0,2,1)).conj() \
                            - (AG - AG.transpose((0,2,1)).conj())

        gG_gpu[IB, :, 0:NI, 0:NI] = gR_gpu[IB, :, 0:NI, 0:NI] \
                            @ (sg_diag_gpu[IB, :, 0:NI, 0:NI] \
                                + SigGB_gpu[IB, :, 0:NI, 0:NI]) \
                                @ gR_gpu[IB, :, 0:NI, 0:NI].transpose((0,2,1)).conj() # Confused about the AG. 
    
    #Second step of iteration
    GR_gpu = cp.zeros_like(gR_gpu)
    GRnn1_gpu = cp.zeros_like(SigLB_gpu)
    GL_gpu = cp.zeros_like(gL_gpu)
    GLnn1_gpu = cp.zeros_like(SigLB_gpu)
    GG_gpu = cp.zeros_like(gG_gpu)
    GGnn1_gpu = cp.zeros_like(SigLB_gpu)



    GR_gpu[0, :,  :NI, :NI] = gR_gpu[0, :, :NI, :NI]
    GRnn1_gpu[0, :,  :NI, :NP] = -GR_gpu[0, :, :NI, :NI] @ ham_upper_gpu[0, :, :NI, :NP] @ gR_gpu[1, :, :NP, :NP]

    GL_gpu[0, :, :NI, :NI] = gL_gpu[0, :, :NI, :NI]
    GLnn1_gpu[0, :, :NI, :NP] = GR_gpu[0, :, :NI, :NI] @ sl_upper_gpu[0, :, :NI, :NP] @ gR_gpu[1, :, :NP, :NP].transpose((0,2,1)).conj() \
                - GR_gpu[0,:, :NI,:NI] @ ham_upper_gpu[0, :, :NI, :NP] @ gL_gpu[1,:, :NP, :NP] \
                - GL_gpu[0,:, :NI,:NI] @ ham_lower_gpu[0, :, :NP, :NI].transpose((0,2,1)).conj() @ gR_gpu[1,:, :NP, :NP].transpose((0,2,1)).conj()

    GG_gpu[0, :, :NI, :NI] = gG_gpu[0, :, :NI, :NI]
    GGnn1_gpu[0, :, :NI, :NP] = GR_gpu[0, :,:NI, :NI] @ sg_upper_gpu[0, :, :NI, :NP] @ gR_gpu[1, :,:NP, :NP].transpose((0,2,1)).conj() \
                - GR_gpu[0,:, :NI,:NI] @ ham_upper_gpu[0, :, :NI, :NP] @ gG_gpu[1, :, :NP, :NP] \
                - GG_gpu[0,:,:NI,:NI] @ ham_lower_gpu[0, :, :NP, :NI].transpose((0,2,1)).conj() @ gR_gpu[1, :, :NP, :NP].transpose((0,2,1)).conj() 
    
    idE[:, 0] = cp.real(cp.trace(SigGB_gpu[0, :, :NI, :NI] @ GL_gpu[0, :, :NI, :NI] - GG_gpu[0, :, :NI, :NI] @ SigLB_gpu[0, :, :NI, :NI], axis1 = 1, axis2 = 2)).get()
    
    for IB in range(1, NB):

        NM = Bmax[IB - 1] - Bmin[IB - 1] + 1
        NI = Bmax[IB] - Bmin[IB] + 1

        # # Extracting off-diagonal Hamiltonian block (upper)
        # M_u = M[Bmin[IB - 1]:Bmax[IB - 1] + 1, Bmin[IB]:Bmax[IB] + 1].toarray()

        # # # Extracting off-diagonal Hamiltonian block (left)
        # M_l = M[Bmin[IB]:Bmax[IB] + 1, Bmin[IB - 1]:Bmax[IB - 1] + 1].toarray()

        # # Extracting off-diagonal lesser Self-energy block (left)
        # SigL_l = SigL[Bmin[IB]:Bmax[IB] + 1, Bmin[IB - 1]:Bmax[IB - 1] + 1].toarray()

        # # Extracting off-diagonal greater Self-energy block (left)
        # SigG_l = SigG[Bmin[IB]:Bmax[IB] + 1, Bmin[IB - 1]:Bmax[IB - 1] + 1].toarray()

        GR_gpu[IB, :, :NI, :NI] = gR_gpu[IB, :, :NI, :NI] + gR_gpu[IB, :,  :NI, :NI] \
                        @ ham_lower_gpu[IB-1, :, :NI, :NM] \
                        @ GR_gpu[IB-1, :, 0:NM, 0:NM] \
                        @ ham_upper_gpu[IB-1, :, :NM, :NI] \
                        @ gR_gpu[IB, :,  :NI, :NI]
        # What is this? Handling off-diagonal elements?
        AL = gR_gpu[IB, :, :NI, :NI] \
            @ sl_lower_gpu[IB-1, :, :NI, :NM] \
            @ GR_gpu[IB-1, :, 0:NM, 0:NM].transpose((0,2,1)).conj() \
            @ ham_lower_gpu[IB-1, :, :NI, :NM].transpose((0,2,1)).conj() \
            @ gR_gpu[IB,:, 0:NI, 0:NI].transpose((0,2,1)).conj()
        # What is this?
        BL = gR_gpu[IB, :, :NI, :NI] \
            @ ham_lower_gpu[IB-1, :, :NI, :NM] \
            @ GR_gpu[IB-1, :, 0:NM, 0:NM] \
            @ ham_upper_gpu[IB-1, :, :NM, :NI] \
            @ gL_gpu[IB, :, :NI, :NI]

        GL_gpu[IB, :, 0:NI, 0:NI] = gL_gpu[IB, :, :NI, :NI] \
                            + gR_gpu[IB, :, 0:NI, 0:NI] \
                            @ ham_lower_gpu[IB-1, :, :NI, :NM] \
                            @ GL_gpu[IB-1, :, 0:NM, 0:NM] \
                            @ ham_lower_gpu[IB-1, :, :NI, :NM].transpose((0,2,1)).conj() \
                            @ gR_gpu[IB, :, :NI, :NI].transpose((0,2,1)).conj() \
                            - (AL - AL.transpose((0,2,1)).conj()) + (BL - BL.transpose((0,2,1)).conj())


        AG = gR_gpu[IB, :, 0:NI, 0:NI] \
            @ sg_lower_gpu[IB-1, :, :NI, :NM] \
            @ GR_gpu[IB-1, :, 0:NM, 0:NM].transpose((0,2,1)).conj() \
            @ ham_lower_gpu[IB-1, :, :NI, :NM].transpose((0,2,1)).conj() \
            @ gR_gpu[IB, :, :NI, :NI].transpose((0,2,1)).conj()

        BG = gR_gpu[IB, :, 0:NI, 0:NI] \
            @ ham_lower_gpu[IB-1, :, :NI, :NM] \
            @ GR_gpu[IB-1, :, 0:NM, 0:NM] \
            @ ham_upper_gpu[IB-1, :, :NM, :NI] \
            @ gG_gpu[IB, :, 0:NI, 0:NI]

        GG_gpu[IB, :, 0:NI, 0:NI] = gG_gpu[IB, :, 0:NI, 0:NI] \
                            + gR_gpu[IB, :, 0:NI, 0:NI] \
                            @ ham_lower_gpu[IB-1, :, :NI, :NM] \
                            @ GG_gpu[IB-1, :, 0:NM, 0:NM] \
                            @ ham_lower_gpu[IB-1, :, :NI, :NM].transpose((0,2,1)).conj() \
                            @ gR_gpu[IB, :, 0:NI, 0:NI].transpose((0,2,1)).conj() \
                            - (AG - AG.transpose((0,2,1)).conj()) + (BG - BG.transpose((0,2,1)).conj()) #

        if IB < NB - 1:  #Off-diagonal are only interesting for IdE!

            # # Extracting off-diagonal Hamiltonian block (right)
            # M_r = M[Bmin[IB]:Bmax[IB] + 1, Bmin[IB + 1]:Bmax[IB + 1] + 1].toarray()

            # # # Extracting off-diagonal Hamiltonian block (lower)
            # M_d = M[Bmin[IB + 1]:Bmax[IB + 1] + 1, Bmin[IB]:Bmax[IB] + 1].toarray()

            # # Extracting off-diagonal lesser Self-energy block (right)
            # SigL_r = SigL[Bmin[IB]:Bmax[IB] + 1, Bmin[IB + 1]:Bmax[IB + 1] + 1].toarray()

            # # Extracting off-diagonal greater Self-energy block (right)
            # SigG_r = SigG[Bmin[IB]:Bmax[IB] + 1, Bmin[IB + 1]:Bmax[IB + 1] + 1].toarray()

            NP = Bmax[IB + 1] - Bmin[IB + 1] + 1

            GRnn1_gpu[IB, :, 0:NI, 0:NP] = - GR_gpu[IB, :,  0:NI, 0:NI] \
                                    @ ham_upper_gpu[IB, :, :NM, :NI] \
                                    @ gR_gpu[IB+1, :, 0:NP, 0:NP]

            GLnn1_gpu[IB, :, 0:NI, 0:NP] = GR_gpu[IB, :, 0:NI, 0:NI] \
                                    @ sl_upper_gpu[IB, :, :NM, :NI] \
                                    @ gR_gpu[IB+1, :, 0:NP, 0:NP].transpose((0,2,1)).conj() \
                                    - GR_gpu[IB, :, 0:NI, 0:NI] \
                                    @ ham_upper_gpu[IB, :, :NM, :NI] \
                                    @ gL_gpu[IB+1, :, 0:NP, 0:NP] \
                                    - GL_gpu[IB, :, :NI, :NI] \
                                    @ ham_lower_gpu[IB, :, :NI, :NM].transpose((0,2,1)).conj() \
                                    @ gR_gpu[IB+1, :, 0:NP, 0:NP].transpose((0,2,1)).conj()
            GGnn1_gpu[IB, :, 0:NI, 0:NP] = GR_gpu[IB, :, 0:NI, 0:NI] \
                                    @ sg_upper_gpu[IB, :, :NM, :NI] \
                                    @ gR_gpu[IB+1, :, 0:NP, 0:NP].transpose((0,2,1)).conj() \
                                    - GR_gpu[IB, :, 0:NI, 0:NI] \
                                    @ ham_upper_gpu[IB, :, :NM, :NI]  \
                                    @ gG_gpu[IB+1, :, 0:NP, 0:NP] \
                                    - GG_gpu[IB, :, 0:NI, 0:NI] \
                                    @ ham_lower_gpu[IB, :, :NI, :NM].transpose((0,2,1)).conj() \
                                    @ gR_gpu[IB+1, :, 0:NP, 0:NP].transpose((0,2,1)).conj()
            idE[:, IB] = cp.real(cp.trace(SigGB_gpu[IB, :NI, :NI] @ GL_gpu[IB, :NI, :NI] - GG_gpu[IB, :NI, :NI] @ SigLB_gpu[IB, :NI, :NI], axis1 = 1, axis2 = 2)).get() 
    for IB in range(NB):
        
        NI = Bmax[IB] - Bmin[IB] + 1
        # GR[IB, :, :, :] *= factor
        # GL[IB, :, :, :] *= factor
        # GG[IB, :, :, :] *= factor
        DOS[:, IB] = 1j * cp.trace(GR_gpu[IB, :, :, :] - GR_gpu[IB, :, :, :].transpose((0,2,1)).conj(), axis1= 1, axis2 = 2).get()
        nE[:, IB] = -1j * cp.trace(GL_gpu[IB, :, :, :], axis1= 1, axis2 = 2).get()
        nP[:, IB] = 1j * cp.trace(GG_gpu[IB, :, :, :], axis1= 1, axis2 = 2).get()

        # if IB < NB-1:
        #     NP = Bmax[IB+1] - Bmin[IB+1] + 1
        #     #idE[IB] = -2 * np.trace(np.real(H[Bmin[IB+1]:Bmax[IB+1]+1, Bmin[IB]:Bmax[IB]+1].toarray() @ GLnn1[IB, 0:NI, 0:NP]))
        #     # GRnn1[IB, :, :, :] *= factor
        #     # GLnn1[IB, :, :, :] *= factor
        #     # GGnn1[IB, :, :, :] *= factor

    
    #idE[NB - 1] = idE[NB - 2]
    idE[:, NB-1] = cp.real(cp.trace(SigGBR_gpu[:, :NI, :NI] @ GL_gpu[NB-1, :, :NI, :NI] - GG_gpu[NB-1, :, :NI, :NI] @ SigLBR_gpu[:, :NI, :NI], axis1 = 1, axis2 = 2)).get()

    # #Final Data Transfer
    # #GR[:, :, :, :] = GR_gpu.get()
    # GL[:, :, :, :] = GL_gpu.get()
    # GG[:, :, :, :] = GG_gpu.get()
    # #GRnn1[:, :, :, :] = GRnn1_gpu.get()
    # GLnn1[:, :, :, :] = GLnn1_gpu.get()
    # GGnn1[:, :, :, :] = GGnn1_gpu.get()

    for i in range(len(GL)):
        GR_gpu[i].get(out=GR[i])
        GL_gpu[i].get(out=GL[i])
        GG_gpu[i].get(out=GG[i])
    for i in range(len(GLnn1)):
        GRnn1_gpu[i].get(out=GRnn1[i])
        GLnn1_gpu[i].get(out=GLnn1[i])
        GGnn1_gpu[i].get(out=GGnn1[i])
    cp.cuda.Stream.null.synchronize()


def _copy_csr_to_gpu(csr):
    tmp = cp.sparse.csr_matrix((cp.asarray(csr.data), cp.asarray(csr.indices), cp.asarray(csr.indptr)),
                               shape=csr.shape, dtype=csr.dtype)
    tmp.has_canonical_format = True
    return tmp


def _get_dense_block_batch(compressed_data,  # Input data, (NE, NNZ) format
                           mapping,  # Mapping (NE, NNZ) format to (NB, NE, BS, BS) format
                           block_idx, # Block index
                           uncompressed_data,  # Output data, (NE, BS, BS) format
                           add_block=None  # Additional block to add
                           ):
    block_data_indices = mapping[block_idx][0]
    block_rows = mapping[block_idx][1]
    block_cols = mapping[block_idx][2]
    uncompressed_data[:] = 0
    uncompressed_data[:, block_rows, block_cols] = compressed_data[:, block_data_indices]
    if add_block is not None:
        uncompressed_data += add_block


def _store_compressed(mapping_diag, mapping_upper, mapping_lower,
                      uncompressed_diag, uncompressed_upper,
                      block_idx,
                      compressed_data):
    block_data_indices = mapping_diag[block_idx][0]
    block_rows = mapping_diag[block_idx][1]
    block_cols = mapping_diag[block_idx][2]
    # compressed_data[:, block_data_indices] = uncompressed_diag[:, block_rows, block_cols]
    compressed_data[:, block_data_indices] = cp.asnumpy(uncompressed_diag[:, block_rows, block_cols])
    # uncompressed_diag[:, block_rows, block_cols].get(out=compressed_data[:, block_data_indices])

    if uncompressed_upper is not None:
        block_data_indices = mapping_upper[block_idx][0]
        block_rows = mapping_upper[block_idx][1]
        block_cols = mapping_upper[block_idx][2]
        # compressed_data[:, block_data_indices] = uncompressed_upper[:, block_rows, block_cols]
        compressed_data[:, block_data_indices] = cp.asnumpy(uncompressed_upper[:, block_rows, block_cols])
        # uncompressed_upper[:, block_rows, block_cols].get(out=compressed_data[:, block_data_indices])
        block_data_indices = mapping_lower[block_idx][0]
        block_rows = mapping_lower[block_idx][1]
        block_cols = mapping_lower[block_idx][2]
        # compressed_data[:, block_data_indices] =  -uncompressed_upper[:, block_cols, block_rows].conj()
        compressed_data[:, block_data_indices] = cp.asnumpy(-uncompressed_upper[:, block_cols, block_rows].conj())
        # (-uncompressed_upper[:, block_cols, block_rows].conj()).get(out=compressed_data[:, block_data_indices])
        # compressed_data[:, block_data_indices] =  -uncompressed_upper.transpose((0, 2, 1)).conj()[:, block_rows, block_cols]


@cpx.jit.rawkernel()
def _get_system_matrix(energies,
                       H_data, H_indices, H_indptr,
                       S_data, S_indices, S_indptr,
                       SR,
                       out,
                       batch_size, block_size):
    
    tid = cpx.jit.threadIdx.x
    if tid < block_size:

        num_threads = cpx.jit.blockDim.x
        bid = cpx.jit.blockIdx.x
        ie = bid // block_size
        ir = bid % block_size

        energy = energies[ie]
         
        buf = cpx.jit.shared_memory(cp.complex128, 416)
        for i in range(tid, block_size, num_threads):
            buf[i] = 0
        cpx.jit.syncthreads()

        start = S_indptr[ir] 
        end = S_indptr[ir + 1]
        i = start + tid
        while i < end:
            j = S_indices[i]
            buf[j] += energy * S_data[i]
            i += num_threads
        cpx.jit.syncthreads()

        start = H_indptr[ir]
        end = H_indptr[ir + 1]
        i = start + tid
        while i < end:
            j = H_indices[i]
            buf[j] -= H_data[i]
            i += num_threads
        cpx.jit.syncthreads()

        for i in range(tid, block_size, num_threads):
            out[ie, ir, i] = buf[i] - SR[ie, ir, i]



def _get_system_matrix_block_batch(energies,  # Energy vector, dense format
                                   H,  # Hamiltonian block, CSR format,
                                   S,  # Overlap block, CSR format
                                   SR,  # Retarded self-energy block-batch, dense format
                                   batch_size, block_size,  # Sizes
                                   ):
    # (E + 1j * 1e-12) * S - H - SRB
    S_dense = S.toarray().reshape(1, block_size, block_size)
    H_dense = H.toarray().reshape(1, block_size, block_size)
    return (energies + 1j * 1e-12).reshape(batch_size, 1, 1) * S_dense - H_dense - SR


def rgf_batched_GPU(energies,  # Energy vector, dense format
                    
                    map_diag, map_upper, map_lower,  # Mapping (NE, NNZ) format to (NB, NE, BS, BS) format
                    
                    H_diag_host, H_upper_host, H_lower_host,  # Hamiltonian matrix, CSR format
                    S_diag_host, S_upper_host, S_lower_host,  # Overlap matrix, CSR format
                    SR_host, SL_host, SG_host,  # Retarded, Lesser, Greater self-energy, (NE, NNZ) format
                    SigRB_left_host, SigRB_right_host,  # Retarded boundary conditions, dense format
                    SigLB_left_host, SigLB_right_host,  # Lesser boundary conditions, dense format
                    SigGB_left_host, SigGB_right_host,  # Greater boundary conditions, dense format

                    # SR_diag_host, SR_upper_host, SR_lower_host,  # Retarded self-energy, CSR format
                    # SL_diag_host, SL_upper_host, SL_lower_host,  # Lesser self-energy, CSR format
                    # SG_diag_host, SG_upper_host, SG_lower_host,  # Greater self-energy, CSR format

                                #  GR_host, GRnn1_host,  # Output Retarded Green's Functions (unused)
                                #  GL_host, GLnn1_host,  # Output Lesser Green's Functions
                                #  GG_host, GGnn1_host,  # Output Greater Green's Functions
                                GR_host, GL_host, GG_host,  # Output Green's Functions
                                 DOS, nE, nP, idE,  # Output Observables
                                 Bmin_fi, Bmax_fi,  # Indices
                                 solve: bool = True,
                                 input_stream: cp.cuda.Stream = None,
                                 output_stream: cp.cuda.Stream = None,
                                ):

    # Sizes
    # Why are subtracing by 1 every time? Fix 0-based indexing
    Bmax = Bmax_fi - 1
    Bmin = Bmin_fi - 1
    batch_size = len(energies)
    num_blocks = len(H_diag_host)
    block_size = max(Bmax - Bmin + 1)
    dtype = np.complex128

    num_threads = min(1024, block_size)
    num_thread_blocks = batch_size * block_size
    md = cp.empty((batch_size, block_size, block_size), dtype=dtype)

    # (energy[k]+ 1j * 1e-12) * S - H

    # H_diag_buffer = cp.empty((2, batch_size, block_size, block_size), dtype=H_diag_host.dtype)
    # H_upper_buffer = cp.empty((2, batch_size, block_size, block_size), dtype=H_diag_host.dtype)
    # H_lower_buffer = cp.empty((2, batch_size, block_size, block_size), dtype=H_diag_host.dtype)

    H_diag_buffer = [None, None]
    H_upper_buffer = [None, None]
    H_lower_buffer = [None, None]
    S_diag_buffer = [None, None]
    S_upper_buffer = [None, None]
    S_lower_buffer = [None, None]

    prev_H_upper_buffer = [None, None]
    prev_H_lower_buffer = [None, None]
    prev_S_upper_buffer = [None, None]
    prev_S_lower_buffer = [None, None]

    SR_diag_buffer = cp.empty((2, batch_size, block_size, block_size), dtype=dtype)
    SR_upper_buffer = cp.empty((2, batch_size, block_size, block_size), dtype=dtype)
    SR_lower_buffer = cp.empty((2, batch_size, block_size, block_size), dtype=dtype)
    SL_diag_buffer = cp.empty((2, batch_size, block_size, block_size), dtype=dtype)
    SL_upper_buffer = cp.empty((2, batch_size, block_size, block_size), dtype=dtype)
    SL_lower_buffer = cp.empty((2, batch_size, block_size, block_size), dtype=dtype)
    SG_diag_buffer = cp.empty((2, batch_size, block_size, block_size), dtype=dtype)
    SG_upper_buffer = cp.empty((2, batch_size, block_size, block_size), dtype=dtype)
    SG_lower_buffer = cp.empty((2, batch_size, block_size, block_size), dtype=dtype)

    computation_stream = cp.cuda.Stream.null
    input_stream = input_stream or cp.cuda.Stream(non_blocking=True)
    output_stream = output_stream or cp.cuda.Stream(non_blocking=True)
    input_events = [cp.cuda.Event() for _ in range(2)]
    computation_event = cp.cuda.Event()

    gR_gpu = cp.empty((num_blocks, batch_size, block_size, block_size), dtype=dtype)  # Retarded (right)
    gL_gpu = cp.empty((num_blocks, batch_size, block_size, block_size), dtype=dtype)  # Lesser (right)
    gG_gpu = cp.empty((num_blocks, batch_size, block_size, block_size), dtype=dtype)  # Greater (right)
    SigLB_gpu = cp.empty((num_blocks-1, batch_size, block_size, block_size), dtype=dtype)  # Lesser boundary self-energy
    SigGB_gpu = cp.empty((num_blocks-1, batch_size, block_size, block_size), dtype=dtype)  # Greater boundary self-energy
    DOS_gpu = cp.empty((batch_size, num_blocks), dtype=dtype)
    nE_gpu = cp.empty((batch_size, num_blocks), dtype=dtype)
    nP_gpu = cp.empty((batch_size, num_blocks), dtype=dtype)
    idE_gpu = cp.empty((batch_size, num_blocks), dtype=idE.dtype)

    GR_gpu = cp.empty((2, batch_size, block_size, block_size), dtype=dtype)  # Retarded (right)
    GRnn1_gpu = cp.empty((2, batch_size, block_size, block_size), dtype=dtype)  # Retarded (right)
    GL_gpu = cp.empty((2, batch_size, block_size, block_size), dtype=dtype)  # Lesser (right)
    GLnn1_gpu = cp.empty((2, batch_size, block_size, block_size), dtype=dtype)  # Lesser (right)
    GG_gpu = cp.empty((2, batch_size, block_size, block_size), dtype=dtype)  # Greater (right)
    GGnn1_gpu = cp.empty((2, batch_size, block_size, block_size), dtype=dtype)  # Greater (right)

    SR_dev, SL_dev, SG_dev = None, None, None
    SigRB_dev = [None for _ in range(num_blocks)]
    SigLB_dev = [None for _ in range(num_blocks)]
    SigGB_dev = [None for _ in range(num_blocks)]
    
    # Backward pass IB \in {NB - 1, ..., 0}

    # First iteration IB = NB - 1
    IB = num_blocks - 1
    nIB = IB - 1
    idx = IB % 2
    nidx = nIB % 2
    NN = Bmax[-1] - Bmin[-1] + 1

    with input_stream:

        map_diag_dev = cp.asarray(map_diag)
        map_upper_dev = cp.asarray(map_upper)
        map_lower_dev = cp.asarray(map_lower)
        H_diag_buffer[idx] = _copy_csr_to_gpu(H_diag_host[IB])
        S_diag_buffer[idx] = _copy_csr_to_gpu(S_diag_host[IB])
        SR_dev = cp.asarray(SR_host)
        SL_dev = cp.asarray(SL_host)
        SG_dev = cp.asarray(SG_host)
        input_events[idx].record(stream=input_stream)

        if num_blocks > 1:

            nsrd = SR_diag_buffer[nidx]
            nsru = SR_upper_buffer[nidx]
            nsrl = SR_lower_buffer[nidx]
            nsld = SL_diag_buffer[nidx]
            nsll = SL_lower_buffer[nidx]
            nsgd = SG_diag_buffer[nidx]
            nsgl = SG_lower_buffer[nidx]

            H_diag_buffer[nidx] = _copy_csr_to_gpu(H_diag_host[nIB])
            H_upper_buffer[nidx] = _copy_csr_to_gpu(H_upper_host[nIB])
            H_lower_buffer[nidx] = _copy_csr_to_gpu(H_lower_host[nIB])
            S_diag_buffer[nidx] = _copy_csr_to_gpu(S_diag_host[nIB])
            S_upper_buffer[nidx] = _copy_csr_to_gpu(S_upper_host[nIB])
            S_lower_buffer[nidx] = _copy_csr_to_gpu(S_lower_host[nIB])

            # NOTE: We are assuming here that the next block is not in the boundary.
            _get_dense_block_batch(SR_dev, map_diag_dev, nIB, nsrd)
            _get_dense_block_batch(SR_dev, map_upper_dev, nIB, nsru)
            _get_dense_block_batch(SR_dev, map_lower_dev, nIB, nsrl)
            _get_dense_block_batch(SL_dev, map_diag_dev, nIB, nsld)
            _get_dense_block_batch(SL_dev, map_lower_dev, nIB, nsll)
            _get_dense_block_batch(SG_dev, map_diag_dev, nIB, nsgd)
            _get_dense_block_batch(SG_dev, map_lower_dev, nIB, nsgl)

            input_events[nidx].record(stream=input_stream)
    
    energies_dev = cp.asarray(energies)
    SigRB_dev[0] = cp.asarray(SigRB_left_host)
    SigRB_dev[-1] = cp.asarray(SigRB_right_host)
    SigLB_dev[0] = cp.asarray(SigLB_left_host)
    SigLB_dev[-1] = cp.asarray(SigLB_right_host)
    SigGB_dev[0] = cp.asarray(SigGB_left_host)
    SigGB_dev[-1] = cp.asarray(SigGB_right_host)
    computation_stream.wait_event(event=input_events[idx])
    computation_stream.synchronize()

    hd = H_diag_buffer[idx]
    sd = S_diag_buffer[idx]
    srd = SR_diag_buffer[idx]
    sld = SL_diag_buffer[idx]
    sgd = SG_diag_buffer[idx]

    gr = gR_gpu[IB]
    gl = gL_gpu[IB]
    gg = gG_gpu[IB]

    srb = SigRB_dev[IB]
    slb = SigLB_dev[IB]
    sgb = SigGB_dev[IB]

    _get_dense_block_batch(SR_dev, map_diag_dev, IB, srd, srb)
    _get_dense_block_batch(SL_dev, map_diag_dev, IB, sld, slb)
    _get_dense_block_batch(SG_dev, map_diag_dev, IB, sgd, sgb)
    # md =_get_system_matrix_block_batch(energies_dev, hd, sd, srd, batch_size, block_size)
    _get_system_matrix[num_thread_blocks, num_threads](energies_dev,
                                                       hd.data, hd.indices, hd.indptr,
                                                       sd.data, sd.indices, sd.indptr,
                                                       srd, md, batch_size, block_size)
    
    if solve:
        gpu_identity = cp.identity(NN, dtype=md.dtype)
        gpu_identity_batch = cp.repeat(gpu_identity[cp.newaxis, :, :], batch_size, axis=0)
        gr[:, 0:NN, 0:NN] = cp.linalg.solve(md[:, 0:NN, 0:NN], gpu_identity_batch)
        computation_stream.synchronize()
    else:
        gr[:, 0:NN, 0:NN] = cp.linalg.inv(md[:, 0:NN, 0:NN])
    gr_h = cp.conjugate(gr[:, 0:NN, 0:NN].transpose((0,2,1)))
    cp.matmul(gr[:, 0:NN, 0:NN] @ sld[:, 0:NN, 0:NN], gr_h[:, 0:NN, 0:NN], out=gl[:, 0:NN, 0:NN])
    cp.matmul(gr[:, 0:NN, 0:NN] @ sgd[:, 0:NN, 0:NN], gr_h[:, 0:NN, 0:NN], out=gg[:, 0:NN, 0:NN])

    # Rest iterations IB \in {NB - 2, ..., 0}
    for IB in range(num_blocks - 2, -1, -1):

        pIB = IB + 1
        nIB = IB - 1
        idx = IB % 2
        nidx = nIB % 2
        pidx = (IB + 1) % 2
        NI = Bmax[IB] - Bmin[IB] + 1
        NP = Bmax[IB + 1] - Bmin[IB + 1] + 1

        computation_stream.synchronize()
        with input_stream:
                
            if nIB >= 0:

                nsrd = SR_diag_buffer[nidx]
                nsru = SR_upper_buffer[nidx]
                nsrl = SR_lower_buffer[nidx]
                nsld = SL_diag_buffer[nidx]
                nsll = SL_lower_buffer[nidx]
                nsgd = SG_diag_buffer[nidx]
                nsgl = SG_lower_buffer[nidx]

                H_diag_buffer[nidx] = _copy_csr_to_gpu(H_diag_host[nIB])
                H_upper_buffer[nidx] = _copy_csr_to_gpu(H_upper_host[nIB])
                H_lower_buffer[nidx] = _copy_csr_to_gpu(H_lower_host[nIB])
                S_diag_buffer[nidx] = _copy_csr_to_gpu(S_diag_host[nIB])
                S_upper_buffer[nidx] = _copy_csr_to_gpu(S_upper_host[nIB])
                S_lower_buffer[nidx] = _copy_csr_to_gpu(S_lower_host[nIB])

                nsrb = SigRB_dev[nIB]
                nslb = SigLB_dev[nIB]
                nsgb = SigGB_dev[nIB]

                _get_dense_block_batch(SR_dev, map_diag_dev, nIB, nsrd, nsrb)
                _get_dense_block_batch(SR_dev, map_upper_dev, nIB, nsru)
                _get_dense_block_batch(SR_dev, map_lower_dev, nIB, nsrl)
                _get_dense_block_batch(SL_dev, map_diag_dev, nIB, nsld, nslb)
                _get_dense_block_batch(SL_dev, map_lower_dev, nIB, nsll)
                _get_dense_block_batch(SG_dev, map_diag_dev, nIB, nsgd, nsgb)
                _get_dense_block_batch(SG_dev, map_lower_dev, nIB, nsgl)
        
            else:  # nIB < 0

                nslu = SL_upper_buffer[idx]
                nsgu = SG_upper_buffer[idx]

                _get_dense_block_batch(SL_dev, map_upper_dev, IB, nslu)
                _get_dense_block_batch(SG_dev, map_upper_dev, IB, nsgu)

            input_events[nidx].record(stream=input_stream)
        
        hd = H_diag_buffer[idx]
        hu = H_upper_buffer[idx]
        hl = H_lower_buffer[idx]
        sd = S_diag_buffer[idx]
        su = S_upper_buffer[idx]
        sl = S_lower_buffer[idx]
        srd = SR_diag_buffer[idx]
        sru = SR_upper_buffer[idx]
        srl = SR_lower_buffer[idx]
        sld = SL_diag_buffer[idx]
        sll = SL_lower_buffer[idx]
        sgd = SG_diag_buffer[idx]
        sgl = SG_lower_buffer[idx]

        gr = gR_gpu[IB]
        pgr = gR_gpu[IB + 1]
        gl = gL_gpu[IB]
        pgl = gL_gpu[IB + 1]
        gg = gG_gpu[IB]
        pgg = gG_gpu[IB + 1]
        srb = SigRB_dev[IB]
        slb = SigLB_gpu[IB]
        sgb = SigGB_gpu[IB]

        if IB == 0:
            gr = GR_gpu[0]
            gl = GL_gpu[0]
            gg = GG_gpu[0]

        computation_stream.wait_event(event=input_events[idx])
        md = _get_system_matrix_block_batch(energies_dev, hd, sd, srd, batch_size, block_size)
        mu = _get_system_matrix_block_batch(energies_dev, hu, su, sru, batch_size, block_size)
        ml = _get_system_matrix_block_batch(energies_dev, hl, sl, srl, batch_size, block_size)
        mu_h = cp.conjugate(mu[:, 0:NI, 0:NP].transpose((0,2,1)))
        mu_x_pgr = mu[:, 0:NI, 0:NP] @ pgr[:, 0:NP, 0:NP]
        al = mu_x_pgr @ sll[:, 0:NP, 0:NI]
        ag = mu_x_pgr @ sgl[:, 0:NP, 0:NI]
        inv_arg = md[:, 0:NI, 0:NI] - mu_x_pgr @ ml[:, :NP, 0:NI]
        if solve:
            gpu_identity = cp.identity(NI, dtype=inv_arg.dtype)
            gpu_identity_batch = cp.repeat(gpu_identity[cp.newaxis, :, :], batch_size, axis=0)
            gr[:, 0:NI, 0:NI] = cp.linalg.solve(inv_arg, gpu_identity_batch)
            computation_stream.synchronize()
        else:
            gr[:, 0:NI, 0:NI] = cp.linalg.inv(inv_arg)
        gr_h = cp.conjugate(gr[:, 0:NI, 0:NI].transpose((0,2,1)))
        cp.subtract(mu[:, 0:NI, 0:NP] @ pgl[:, 0:NP, 0:NP] @ mu_h[:, 0:NP, 0:NI],
                    al - cp.conjugate(al.transpose((0,2,1))), out=slb[:, 0:NI, 0:NI])  # SLB must change
        cp.subtract(mu[:, 0:NI, 0:NP] @ pgg[:, 0:NP, 0:NP] @ mu_h[:, 0:NP, 0:NI],
                    ag - cp.conjugate(ag.transpose((0,2,1))), out=sgb[:, 0:NI, 0:NI])  # SGB must change
        cp.matmul(gr[:, 0:NI, 0:NI] @ (sld[:, 0:NI, 0:NI] + slb[:, 0:NI, 0:NI]), gr_h[:, 0:NI, 0:NI], out=gl[:, 0:NI, 0:NI])
        cp.matmul(gr[:, 0:NI, 0:NI] @ (sgd[:, 0:NI, 0:NI] + sgb[:, 0:NI, 0:NI]), gr_h[:, 0:NI, 0:NI], out=gg[:, 0:NI, 0:NI])

        if IB == 0:

            computation_stream.synchronize()
            computation_event.record(stream=computation_stream)

            # with output_stream:
            #     output_stream.wait_event(event=computation_event)
            #     # GR_gpu[0].get(out=GR_host[0])
            #     # GL_gpu[0].get(out=GL_host[0])
            #     # GG_gpu[0].get(out=GG_host[0])
            #     _store_compressed(mapping_diag, mapping_upper, mapping_lower, GR2[IB], GRnn12[IB], IB, GR_host)
            #     _store_compressed(mapping_diag, mapping_upper, mapping_lower, GL2[IB], GLnn12[IB], IB, GL_host)
            #     _store_compressed(mapping_diag, mapping_upper, mapping_lower, GG2[IB], GGnn12[IB], IB, GG_host)

            DOS_gpu[:, 0] = 1j * cp.trace(gr[:, 0:NI, 0:NI] - gr_h[:, 0:NI, 0:NI], axis1=1, axis2=2)
            nE_gpu[:, 0] = -1j * cp.trace(gl[:, 0:NI, 0:NI], axis1=1, axis2=2)
            nP_gpu[:, 0] = 1j * cp.trace(gg[:, 0:NI, 0:NI], axis1=1, axis2=2)
            idE_gpu[:, 0] = cp.real(cp.trace(sgb[:, 0:NI, 0:NI] @ gl[:, 0:NI, 0:NI] -
                                             gg[:, 0:NI, 0:NI] @ slb[:, 0:NI, 0:NI], axis1=1, axis2=2))
    
    # Forward pass

    # First iteration
    IB = 0
    nIB = IB + 1
    idx = IB % 2
    nidx = nIB % 2

    computation_stream.synchronize()
    with input_stream:
            
        if nIB < num_blocks:

            npsll = SL_lower_buffer[nidx]
            npsgl = SG_lower_buffer[nidx]

            prev_H_upper_buffer[nidx] = _copy_csr_to_gpu(H_upper_host[IB])
            prev_H_lower_buffer[nidx] = _copy_csr_to_gpu(H_lower_host[IB])
            prev_S_upper_buffer[nidx] = _copy_csr_to_gpu(S_upper_host[IB])
            prev_S_lower_buffer[nidx] = _copy_csr_to_gpu(S_lower_host[IB])

            _get_dense_block_batch(SL_dev, map_lower_dev, IB, npsll)
            _get_dense_block_batch(SG_dev, map_lower_dev, IB, npsgl)

            if nIB < num_blocks - 1:

                nsru = SR_upper_buffer[nidx]
                nsrl = SR_lower_buffer[nidx]
                nslu = SL_upper_buffer[nidx]
                nsgu = SG_upper_buffer[nidx]

                H_upper_buffer[nidx] = _copy_csr_to_gpu(H_upper_host[nIB])
                H_lower_buffer[nidx] = _copy_csr_to_gpu(H_lower_host[nIB])
                S_upper_buffer[nidx] = _copy_csr_to_gpu(S_upper_host[nIB])
                S_lower_buffer[nidx] = _copy_csr_to_gpu(S_lower_host[nIB])

                _get_dense_block_batch(SR_dev, map_upper_dev, nIB, nsru)
                _get_dense_block_batch(SR_dev, map_lower_dev, nIB, nsrl)
                _get_dense_block_batch(SL_dev, map_upper_dev, nIB, nslu)
                _get_dense_block_batch(SG_dev, map_upper_dev, nIB, nsgu)

            input_events[idx].record(stream=input_stream)
    
    hu = H_upper_buffer[idx]
    su = S_upper_buffer[idx]
    hl = H_lower_buffer[idx]
    sl = S_lower_buffer[idx]
    slu = SL_upper_buffer[idx]
    sgu = SG_upper_buffer[idx]

    GR = GR_gpu[IB]
    GRnn1 = GRnn1_gpu[idx]
    GL = GL_gpu[IB]
    GLnn1 = GLnn1_gpu[idx]
    GG = GG_gpu[IB]
    GGnn1 = GGnn1_gpu[idx]

    # NOTE: These were written directly to output in the last iteration of the backward pass
    gr = GR_gpu[IB]
    gl = GL_gpu[IB]
    gg = GG_gpu[IB]

    pgr = gR_gpu[nIB]
    pgl = gL_gpu[nIB]
    pgg = gG_gpu[nIB]

    computation_stream.wait_event(event=input_events[nidx])
    mu = _get_system_matrix_block_batch(energies_dev, hu, su, sru, batch_size, block_size)
    ml = _get_system_matrix_block_batch(energies_dev, hl, sl, srl, batch_size, block_size)
    gr_h = cp.conjugate(pgr[:, 0:NP, 0:NP].transpose((0,2,1)))
    ml_h = cp.conjugate(ml[:, 0:NP, 0:NI].transpose((0,2,1)))
    gr_x_mu = gr[:, 0:NI, 0:NI] @ mu[:, 0:NI, 0:NP]
    ml_h_x_gr_h = ml_h @ gr_h[:, 0:NP, 0:NP]
    # NOTE: These were written in the last iteration of the backward pass
    # GR[:] = gr
    # GL[:] = gl
    # GG[:] = gg
    cp.negative(gr_x_mu @ pgr[:, 0:NP, 0:NP], out=GRnn1[:, 0:NI, 0:NP])
    cp.subtract(gr[:, 0:NI, 0:NI] @ slu[:, 0:NI, 0:NP] @ gr_h[:, 0:NP, 0:NP] - gr_x_mu @ pgl[:, 0:NP, 0:NP], gl[:, 0:NI, 0:NI] @ ml_h_x_gr_h, out=GLnn1[:, 0:NI, 0:NP])
    cp.subtract(gr[:, 0:NI, 0:NI] @ sgu[:, 0:NI, 0:NP] @ gr_h[:, 0:NP, 0:NP] - gr_x_mu @ pgg[:, 0:NP, 0:NP], gg[:, 0:NI, 0:NI] @ ml_h_x_gr_h, out=GGnn1[:, 0:NI, 0:NP])

    computation_stream.synchronize()
    computation_event.record(stream=computation_stream)
    # with output_stream:
    #     output_stream.wait_event(event=computation_event)
    #     GRnn1.get(out=GRnn1_host[0])
    #     GLnn1.get(out=GLnn1_host[0])
    #     GGnn1.get(out=GGnn1_host[0])
    with output_stream:
        output_stream.wait_event(event=computation_event)
        _store_compressed(mapping_diag, mapping_upper, mapping_lower, GR, GRnn1, IB, GR_host)
        _store_compressed(mapping_diag, mapping_upper, mapping_lower, GL, GLnn1, IB, GL_host)
        _store_compressed(mapping_diag, mapping_upper, mapping_lower, GG, GGnn1, IB, GG_host)

    # Rest iterations
    for IB in range(1, num_blocks):

        pIB = IB - 1
        nIB = IB + 1
        idx = IB % 2
        pidx = pIB % 2
        nidx = nIB % 2
        NI = Bmax[IB] - Bmin[IB] + 1
        NM = Bmax[IB - 1] - Bmin[IB - 1] + 1

        computation_stream.synchronize()
        with input_stream:
                
            if nIB < num_blocks:

                npsll = SL_lower_buffer[idx]
                npsgl = SG_lower_buffer[idx]

                _get_dense_block_batch(SL_dev, map_lower_dev, IB, npsll)
                _get_dense_block_batch(SG_dev, map_lower_dev, IB, npsgl)


                if nIB < num_blocks - 1:

                    nsru = SR_upper_buffer[nidx]
                    nsrl = SR_lower_buffer[nidx]
                    nslu = SL_upper_buffer[nidx]
                    nsgu = SG_upper_buffer[nidx]

                    H_upper_buffer[nidx] = _copy_csr_to_gpu(H_upper_host[nIB])
                    H_lower_buffer[nidx] = _copy_csr_to_gpu(H_lower_host[nIB])
                    S_upper_buffer[nidx] = _copy_csr_to_gpu(S_upper_host[nIB])
                    S_lower_buffer[nidx] = _copy_csr_to_gpu(S_lower_host[nIB])

                    _get_dense_block_batch(SR_dev, map_upper_dev, nIB, nsru)
                    _get_dense_block_batch(SR_dev, map_lower_dev, nIB, nsrl)
                    _get_dense_block_batch(SL_dev, map_upper_dev, nIB, nslu)
                    _get_dense_block_batch(SG_dev, map_upper_dev, nIB, nsgu)

            input_events[idx].record(stream=input_stream)
        
        # phu = prev_H_upper_buffer[pidx]
        # phl = prev_H_lower_buffer[pidx]
        # psu = prev_S_upper_buffer[pidx]
        # psl = prev_S_lower_buffer[pidx]
        pmu = mu
        pml = ml
        psll = SL_lower_buffer[pidx]
        psgl = SG_lower_buffer[pidx]

        GR = GR_gpu[idx]
        pGR = GR_gpu[pidx]
        GL = GL_gpu[idx]
        pGL = GL_gpu[pidx]
        GG = GG_gpu[idx]
        pGG = GG_gpu[pidx]

        gr = gR_gpu[IB]
        gl = gL_gpu[IB]
        gg = gG_gpu[IB]

        computation_stream.wait_event(event=input_events[pidx])
        pGRh = cp.conjugate(pGR[:, 0:NM, 0:NM].transpose((0,2,1)))
        phlh = cp.conjugate(pml[:, 0:NI, 0:NM].transpose((0,2,1)))
        grh = cp.conjugate(gr[:, 0:NI, 0:NI].transpose((0,2,1)))
        grphl = gr[:, 0:NI, 0:NI] @ pml[:, 0:NI, 0:NM]
        grphlpGRphu = grphl @ pGR[:, 0:NM, 0:NM] @ pmu[:, 0:NM, 0:NI]
        phlhgrh = phlh @ grh
        pGRhphlhgrh = pGRh @ phlhgrh
        al = gr[:, 0:NI, 0:NI] @ psll[:, 0:NI, 0:NI] @ pGRhphlhgrh
        bl = grphlpGRphu @ gl[:, 0:NI, 0:NI]
        ag = gr[:, 0:NI, 0:NI] @ psgl[:, 0:NI, 0:NM] @ pGRhphlhgrh
        bg = grphlpGRphu @ gg[:, 0:NI, 0:NI]
        cp.add(gr[:, 0:NI, 0:NI], grphlpGRphu @ gr[:, 0:NI, 0:NI], out=GR[:, 0:NI, 0:NI])
        cp.subtract(gl[:, 0:NI, 0:NI] + grphl @ pGL[:, 0:NM, 0:NM] @ phlhgrh,
                    al - cp.conjugate(al.transpose((0,2,1))) - bl + cp.conjugate(bl.transpose((0,2,1))),
                    out=GL[:, 0:NI, 0:NI])
        cp.subtract(gg[:, 0:NI, 0:NI] + grphl @ pGG[:, 0:NM, 0:NM] @ phlhgrh,
                    ag - cp.conjugate(ag.transpose((0,2,1))) - bg + cp.conjugate(bg.transpose((0,2,1))),
                    out=GG[:, 0:NI, 0:NI])
    
        if IB < num_blocks - 1:

            NP = Bmax[IB + 1] - Bmin[IB + 1] + 1

            GRnn1 = GRnn1_gpu[idx]
            GLnn1 = GLnn1_gpu[idx]
            GGnn1 = GGnn1_gpu[idx]

            ngr = gR_gpu[nIB]
            ngl = gL_gpu[nIB]
            ngg = gG_gpu[nIB]

            hu = H_upper_buffer[idx]
            hl = H_lower_buffer[idx]
            su = S_upper_buffer[idx]
            sl = S_lower_buffer[idx]
            sru = SR_upper_buffer[idx]
            srl = SR_lower_buffer[idx]
            slu = SL_upper_buffer[idx]
            sgu = SG_upper_buffer[idx]

            mu = _get_system_matrix_block_batch(energies_dev, hu, su, sru, batch_size, block_size)
            ml = _get_system_matrix_block_batch(energies_dev, hl, sl, srl, batch_size, block_size)

            ngrh = cp.conjugate(ngr[:, 0:NP, 0:NP].transpose((0,2,1)))
            hlh = cp.conjugate(ml[:, 0:NP, 0:NI].transpose((0,2,1)))
            GRhu = GR[:, 0:NI, 0:NI] @ mu[:, 0:NI, 0:NP]
            hlhngrh = hlh @ ngrh
            cp.negative(GR[:, 0:NI, 0:NI] @ mu[:, 0:NI, 0:NP] @ ngr[:, 0:NP, 0:NP], out=GRnn1[:, 0:NI, 0:NP])
            cp.subtract(GR[:, 0:NI, 0:NI] @ slu[:, 0:NI, 0:NP] @ ngrh,
                        GRhu @ ngl[:, 0:NP, 0:NP] + GL[:, 0:NI, 0:NI] @ hlhngrh, out=GLnn1[:, 0:NI, 0:NP])
            cp.subtract(GR[:, 0:NI, 0:NI] @ sgu[:, 0:NI, 0:NP] @ ngrh,
                        GRhu @ ngg[:, 0:NP, 0:NP] + GG[:, 0:NI, 0:NI] @ hlhngrh, out=GGnn1[:, 0:NI, 0:NP])
        
            computation_stream.synchronize()
            computation_event.record(stream=computation_stream)
            # with output_stream:
            #     output_stream.wait_event(event=computation_event)
            #     GR.get(out=GR_host[IB])
            #     GL.get(out=GL_host[IB])
            #     GG.get(out=GG_host[IB])
            #     GRnn1.get(out=GRnn1_host[IB])
            #     GLnn1.get(out=GLnn1_host[IB])
            #     GGnn1.get(out=GGnn1_host[IB])
            with output_stream:
                output_stream.wait_event(event=computation_event)
                _store_compressed(mapping_diag, mapping_upper, mapping_lower, GR, GRnn1, IB, GR_host)
                _store_compressed(mapping_diag, mapping_upper, mapping_lower, GL, GLnn1, IB, GL_host)
                _store_compressed(mapping_diag, mapping_upper, mapping_lower, GG, GGnn1, IB, GG_host)

            slb = SigLB_gpu[IB]
            sgb = SigGB_gpu[IB]
            
            idE_gpu[:, IB] = cp.real(cp.trace(sgb[:, 0:NI, 0:NI] @ GL[:, 0:NI, 0:NI] -
                                              GG[:, 0:NI, 0:NI] @ slb[:, 0:NI, 0:NI], axis1=1, axis2=2))
        
        DOS_gpu[:, IB] = 1j * cp.trace(GR[:, 0:NI, 0:NI] - cp.conjugate(GR[:, 0:NI, 0:NI].transpose(0,2,1)), axis1=1, axis2=2)
        nE_gpu[:, IB] = -1j * cp.trace(GL[:, 0:NI, 0:NI], axis1=1, axis2=2)
        nP_gpu[:, IB] = 1j * cp.trace(GG[:, 0:NI, 0:NI], axis1=1, axis2=2)

    slb = SigLB_dev[-1]
    sgb = SigGB_dev[-1]
    idE_gpu[:, num_blocks - 1] = cp.real(cp.trace(sgb[:, 0:NI, 0:NI] @ GL[:, 0:NI, 0:NI] -
                                          GG[:, 0:NI, 0:NI] @ slb[:, 0:NI, 0:NI], axis1=1, axis2=2))
    

    # GR.get(out=GR_host[-1])
    # GL.get(out=GL_host[-1])
    # GG.get(out=GG_host[-1])
    _store_compressed(mapping_diag, mapping_upper, mapping_lower, GR, None, num_blocks - 1, GR_host)
    _store_compressed(mapping_diag, mapping_upper, mapping_lower, GL, None, num_blocks - 1, GL_host)
    _store_compressed(mapping_diag, mapping_upper, mapping_lower, GG, None, num_blocks - 1, GG_host)
    DOS_gpu.get(out=DOS)
    nE_gpu.get(out=nE)
    nP_gpu.get(out=nP)
    idE_gpu.get(out=idE)
    input_stream.synchronize()
    output_stream.synchronize()
    computation_stream.synchronize()


if __name__ == "__main__":

    validate = False

    # Data sizes
    num_energies = 10
    num_blocks = 13
    block_size = 416
    full_size = num_blocks * block_size
    nnz = 491040

    # Mappings
    map_diag = np.load("map_diag.npy")
    map_lower = np.load("map_lower.npy")
    map_upper = np.load("map_upper.npy")
    rows = np.load("rows.npy")
    cols = np.load("columns.npy")
    ij2ji = np.load("ij2ji.npy")
    bmin = np.arange(num_blocks) * block_size
    bmax = np.arange(1, num_blocks + 1) * block_size
    mapping_diag = map_to_mapping(map_diag, num_blocks)
    mapping_upper = map_to_mapping(map_upper, num_blocks - 1)
    mapping_lower = map_to_mapping(map_lower, num_blocks - 1)

    # RNG
    rng = np.random.default_rng(42)

    # Energies
    energies = rng.random(num_energies)

    # Hamiltonian and overlap matrices
    hamiltonian = sp.random(full_size, full_size, density=0.1, format="csr", random_state=rng)
    overlap = sp.random(full_size, full_size, density=0.1, format="csr", random_state=rng)
    hamiltonian_diag_dense = np.zeros((num_blocks, num_energies, block_size, block_size), dtype=np.complex128)
    hamiltonian_upper_dense = np.zeros((num_blocks - 1, num_energies, block_size, block_size), dtype=np.complex128)
    hamiltonian_lower_dense = np.zeros((num_blocks - 1, num_energies, block_size, block_size), dtype=np.complex128)
    sparse2block_energyhamgen_no_map(hamiltonian, overlap,
                                     hamiltonian_diag_dense, hamiltonian_upper_dense, hamiltonian_lower_dense,
                                     bmax - 1, bmin, energies)
    hamiltonian_diag, hamiltonian_upper, hamiltonian_lower = csr_to_block_tridiagonal_csr(hamiltonian, bmin, bmax)
    overlap_diag, overlap_upper, overlap_lower = csr_to_block_tridiagonal_csr(overlap, bmin, bmax)
    if validate:
        validate_hamiltonian(energies,
                             hamiltonian_diag, hamiltonian_upper, hamiltonian_lower,
                             overlap_diag, overlap_upper, overlap_lower,
                             hamiltonian_diag_dense, hamiltonian_upper_dense, hamiltonian_lower_dense)

    # Self-energies
    self_energies_retarded = random_complex((num_energies, nnz), rng)
    self_energies_lesser = random_complex((num_energies, nnz), rng)
    self_energies_greater = random_complex((num_energies, nnz), rng)
    self_energies_retarded_phonon = random_complex((num_energies, full_size), rng)
    self_energies_lesser_phonon = random_complex((num_energies, full_size), rng)
    self_energies_greater_phonon = random_complex((num_energies, full_size), rng)
    self_energy_preprocess_2d(self_energies_lesser, self_energies_greater, self_energies_retarded,
                              self_energies_lesser_phonon, self_energies_greater_phonon, self_energies_retarded_phonon,
                              rows, cols, ij2ji)
    
    # Boundary conditions
    sigma_retarded_boundary_left = random_complex((num_energies, block_size, block_size), rng)
    sigma_retarded_boundary_right = random_complex((num_energies, block_size, block_size), rng)
    sigma_lesser_boundary_left = random_complex((num_energies, block_size, block_size), rng)
    sigma_lesser_boundary_right = random_complex((num_energies, block_size, block_size), rng)
    sigma_greater_boundary_left = random_complex((num_energies, block_size, block_size), rng)
    sigma_greater_boundary_right = random_complex((num_energies, block_size, block_size), rng)

    # Self-energies treatment
    (sr_blco_diag, sr_blco_upper, sr_blco_lower,
     sl_blco_diag, sl_blco_upper, sl_blco_lower,
     sg_blco_diag, sg_blco_upper, sg_blco_lower) = initialize_block_sigma_batched(num_energies, num_blocks, block_size)
    sparse2block_energy_forbatchedblockwise(map_diag, map_upper, map_lower, self_energies_retarded,
                                            sr_blco_diag, sr_blco_upper, sr_blco_lower, bmax, bmin)
    sparse2block_energy_forbatchedblockwise(map_diag, map_upper, map_lower, self_energies_lesser,
                                            sl_blco_diag, sl_blco_upper, sl_blco_lower, bmax, bmin)
    sparse2block_energy_forbatchedblockwise(map_diag, map_upper, map_lower, self_energies_greater,
                                            sg_blco_diag, sg_blco_upper, sg_blco_lower, bmax, bmin)
    sr_blco_diag[0] += sigma_retarded_boundary_left
    sr_blco_diag[-1] += sigma_retarded_boundary_right
    sl_blco_diag[0] += sigma_lesser_boundary_left
    sl_blco_diag[-1] += sigma_lesser_boundary_right
    sg_blco_diag[0] += sigma_greater_boundary_left
    sg_blco_diag[-1] += sigma_greater_boundary_right
    if validate:
        validate_self_energies(energies,
                               self_energies_retarded, self_energies_lesser, self_energies_greater,
                               sigma_retarded_boundary_left, sigma_retarded_boundary_right,
                               sigma_lesser_boundary_left, sigma_lesser_boundary_right,
                               sigma_greater_boundary_left, sigma_greater_boundary_right,
                               mapping_diag, mapping_upper, mapping_lower,
                               sr_blco_diag, sr_blco_upper, sr_blco_lower,
                               sl_blco_diag, sl_blco_upper, sl_blco_lower,
                               sg_blco_diag, sg_blco_upper, sg_blco_lower)

    # System matrix (hamiltonian with self-energies and boundary conditions)
    system_matrix_diag_dense = hamiltonian_diag_dense - sr_blco_diag
    system_matrix_upper_dense = hamiltonian_upper_dense - sr_blco_upper
    system_matrix_lower_dense = hamiltonian_lower_dense - sr_blco_lower
    if validate:
        validate_system_matrix(energies,
                               hamiltonian_diag, hamiltonian_upper, hamiltonian_lower,
                               overlap_diag, overlap_upper, overlap_lower,
                               self_energies_retarded, sigma_retarded_boundary_left, sigma_retarded_boundary_right,
                               mapping_diag, mapping_upper, mapping_lower,
                               system_matrix_diag_dense, system_matrix_upper_dense, system_matrix_lower_dense)

    # Green's functions
    GR, GRnn1, GL, GLnn1, GG, GGnn1 = initialize_block_G_batched(num_energies, num_blocks, block_size)
    GR2, GRnn12, GL2, GLnn12, GG2, GGnn12 = initialize_block_G_batched(num_energies, num_blocks, block_size)
    greens_function_retarded = np.empty((num_blocks, num_energies, block_size, block_size), dtype=np.complex128)
    greens_function_lesser = np.empty((num_blocks, num_energies, block_size, block_size), dtype=np.complex128)
    greens_function_greater = np.empty((num_blocks, num_energies, block_size, block_size), dtype=np.complex128)

    # Observables
    DOS = cpx.zeros_pinned((num_energies, num_blocks), dtype=np.complex128)
    nE = cpx.zeros_pinned((num_energies, num_blocks), dtype=np.complex128)
    nP = cpx.zeros_pinned((num_energies, num_blocks), dtype=np.complex128)
    idE = cpx.zeros_pinned((num_energies, num_blocks), dtype=np.complex128)
    DOS2 = cpx.zeros_pinned((num_energies, num_blocks), dtype=np.complex128)
    nE2 = cpx.zeros_pinned((num_energies, num_blocks), dtype=np.complex128)
    nP2 = cpx.zeros_pinned((num_energies, num_blocks), dtype=np.complex128)
    idE2 = cpx.zeros_pinned((num_energies, num_blocks), dtype=np.complex128)

    # Reference RGF
    rgf_standaloneGF_batched_GPU(system_matrix_diag_dense, system_matrix_upper_dense, system_matrix_lower_dense,
                                 sg_blco_diag, sg_blco_upper, sg_blco_lower,
                                 sl_blco_diag, sl_blco_upper, sl_blco_lower,
                                 sigma_greater_boundary_right, sigma_lesser_boundary_right,
                                 GR, GRnn1, GL, GLnn1, GG, GGnn1,
                                 DOS, nE, nP, idE, bmin + 1, bmax)
    gr_diag = GR.transpose((1, 0, 2, 3))
    gr_upper = GRnn1.transpose((1, 0, 2, 3))
    gr_lower = - gr_upper.transpose((0, 1, 3, 2)).conjugate()
    gl_diag = GL.transpose((1, 0, 2, 3))
    gl_upper = GLnn1.transpose((1, 0, 2, 3))
    gl_lower = - gl_upper.transpose((0, 1, 3, 2)).conjugate()
    gg_diag = GG.transpose((1, 0, 2, 3))
    gg_upper = GGnn1.transpose((1, 0, 2, 3))
    gg_lower = - gg_upper.transpose((0, 1, 3, 2)).conjugate()
    gr_h2g = block2sparse_energy_alt(map_diag, map_upper, map_lower, gr_diag, gr_upper, gr_lower, nnz, num_energies, False)
    gl_h2g = block2sparse_energy_alt(map_diag, map_upper, map_lower, gl_diag, gl_upper, gl_lower, nnz, num_energies, False)
    gg_h2g = block2sparse_energy_alt(map_diag, map_upper, map_lower, gg_diag, gg_upper, gg_lower, nnz, num_energies, False)
                                     
    # New RGF
    gr_h2g_2 = cpx.empty_pinned((num_energies, nnz), dtype=np.complex128)
    gl_h2g_2 = cpx.empty_pinned((num_energies, nnz), dtype=np.complex128)
    gg_h2g_2 = cpx.empty_pinned((num_energies, nnz), dtype=np.complex128)
    rgf_batched_GPU(energies,
                    mapping_diag, mapping_upper, mapping_lower,
                    hamiltonian_diag, hamiltonian_upper, hamiltonian_lower,
                    overlap_diag, overlap_upper, overlap_lower,
                    self_energies_retarded, self_energies_lesser, self_energies_greater,
                    sigma_retarded_boundary_left, sigma_retarded_boundary_right,
                    sigma_lesser_boundary_left, sigma_lesser_boundary_right,
                    sigma_greater_boundary_left, sigma_greater_boundary_right,
                    # GR2, GRnn12, GL2, GLnn12, GG2, GGnn12,
                    gr_h2g_2, gl_h2g_2, gg_h2g_2,
                    DOS2, nE2, nP2, idE2, bmin + 1, bmax, solve=True)
    # for i in range(num_blocks - 1):
    #     _store_compressed(mapping_diag, mapping_upper, mapping_lower, GR2[i], GRnn12[i], i, gr_h2g_2)
    #     _store_compressed(mapping_diag, mapping_upper, mapping_lower, GL2[i], GLnn12[i], i, gl_h2g_2)
    #     _store_compressed(mapping_diag, mapping_upper, mapping_lower, GG2[i], GGnn12[i], i, gg_h2g_2)
    # _store_compressed(mapping_diag, mapping_upper, mapping_lower, GR2[num_blocks - 1], None, num_blocks - 1, gr_h2g_2)
    # _store_compressed(mapping_diag, mapping_upper, mapping_lower, GL2[num_blocks - 1], None, num_blocks - 1, gl_h2g_2)
    # _store_compressed(mapping_diag, mapping_upper, mapping_lower, GG2[num_blocks - 1], None, num_blocks - 1, gg_h2g_2)
    
    assert np.allclose(gr_h2g, gr_h2g_2)
    assert np.allclose(gl_h2g, gl_h2g_2)
    assert np.allclose(gg_h2g, gg_h2g_2)

    # # Compare
    # assert np.allclose(GR, GR2)
    # assert np.allclose(GL, GL2)
    # assert np.allclose(GG, GG2)
    # assert np.allclose(GRnn1, GRnn12)
    # assert np.allclose(GLnn1, GLnn12)
    # assert np.allclose(GGnn1, GGnn12)
    assert np.allclose(DOS, DOS2)
    assert np.allclose(nE, nE2)
    assert np.allclose(nP, nP2)
    assert np.allclose(idE, idE2)
    print("Success!")
