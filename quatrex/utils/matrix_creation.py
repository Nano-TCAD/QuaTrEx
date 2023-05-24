import numpy as np
from scipy import sparse
from json import JSONEncoder
import json

import numpy.typing as npt
import typing

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

def create_matrices_W(n, n_blocks, sparsity=1):
    V00 = sparse.random(n, n, 1, dtype = np.cfloat)
    V00 = (V00 + V00.conj())/2
    V00 = (V00 + V00.T)/2
    V01 = sparse.random(n,n,1)
    
    PR00 = sparse.random(n,n,sparsity) + sparse.random(n,n,1)*1j
    PR00 = (PR00 + PR00.T)/2
    PR01 = sparse.random(n,n,sparsity/2) + sparse.random(n,n,1)*1j
    
    PL00 = sparse.random(n,n,sparsity) + sparse.random(n,n,1)*1j
    PL00 = (PL00 - PL00.conj().T)/2
    PL01 = sparse.random(n,n,sparsity/2) + sparse.random(n,n,1)*1j
    
    n_temp = n
    
    while n_temp < n*n_blocks:
        V00 = sparse.vstack([sparse.hstack([V00, V01]), 
                             sparse.hstack([V01.T, V00])
                            ])
        V01 = sparse.vstack([sparse.csr_matrix((n_temp, 2*n_temp)), 
                             sparse.hstack([V01, sparse.csc_matrix((n_temp,n_temp))])
                            ])

        PR00 = sparse.vstack([sparse.hstack([PR00, PR01]), 
                              sparse.hstack([PR01.T, PR00])
                             ])
        PR01 = sparse.vstack([sparse.csr_matrix((n_temp, 2*n_temp)), 
                              sparse.hstack([PR01, sparse.csc_matrix((n_temp,n_temp))])
                             ])

        PL00 = sparse.vstack([sparse.hstack([PL00, PL01]), 
                              sparse.hstack([-PL01.conj().T, PL00])
                             ])
        PL01 = sparse.vstack([sparse.csr_matrix((n_temp, 2*n_temp)), 
                              sparse.hstack([PL01, sparse.csc_matrix((n_temp,n_temp))])
                             ])
        
        n_temp *= 2

    V = V00.tocsr()[:n*n_blocks,:n*n_blocks]
    PR = PR00.tocsr()[:n*n_blocks,:n*n_blocks]
    PL = PL00.tocsr()[:n*n_blocks,:n*n_blocks]
    
    
    return V, PR, PL

def create_matrices_H(n, n_blocks, sparsity=1):
    """
    From Leo's code.
    Modified.
    Creates a Sparse block-tridiagonal matrix.
    
    Lesser/Greater self-energies should be purely imaginary??
    """
    H00 = sparse.random(n, n, 1, dtype = np.float64)
    H00 = (H00 + H00.T)/2                 # Make it symmetric
    H01 = sparse.random(n,n,1)
    
    SL00 = 1j*sparse.random(n, n, 1, dtype = np.float64)
    SL00 = (SL00 - SL00.T.conj())         #### Make it satisfy lesser/greater hc condition
    #SL00 = (SL00 + SL00.T)
    #SL01 = sparse.csc_matrix((n, n))
    SL01 = 1j*sparse.random(n, n, 1, dtype = np.float64)
    
    SG00 = 1j*sparse.random(n, n, 1, dtype = np.float64)
    SG00 = (SG00 - SG00.T.conj())         # Make it satisfy lesser/greater hc condition
    #SG00 = (SG00 + SG00.T)
    #SG01 = sparse.csc_matrix((n, n))        
    SG01 = 1j*sparse.random(n, n, 1, dtype = np.float64)
    
    
    n_temp = n
    
    while n_temp < n*n_blocks:
        H00 = sparse.vstack([sparse.hstack([H00, H01]), 
                             sparse.hstack([H01.T, H00])
                            ])
        H01 = sparse.vstack([sparse.csr_matrix((n_temp, 2*n_temp)), 
                             sparse.hstack([H01, sparse.csc_matrix((n_temp,n_temp))])
                            ])
        
        SL00 = sparse.vstack([sparse.hstack([SL00, SL01]), 
                             sparse.hstack([-SL01.T.conj(), SL00])
                            ])
        SL01 = sparse.vstack([sparse.csr_matrix((n_temp, 2*n_temp)), 
                             sparse.hstack([SL01, sparse.csc_matrix((n_temp,n_temp))])
                            ])
        
        SG00 = sparse.vstack([sparse.hstack([SG00, SG01]), 
                             sparse.hstack([-SG01.T.conj(), SG00])
                            ])
        SG01 = sparse.vstack([sparse.csr_matrix((n_temp, 2*n_temp)), 
                             sparse.hstack([SG01, sparse.csc_matrix((n_temp,n_temp))])
                            ])
        
        n_temp *= 2
    
    H = H00.tocsr()[:n*n_blocks,:n*n_blocks]
    SL = SL00.tocsr()[:n*n_blocks,:n*n_blocks]
    SG = SG00.tocsr()[:n*n_blocks,:n*n_blocks]
    
    return H, SL, SG

def get_matrices(fileName):
    with open(fileName, "r") as read_file:
        data = json.load(read_file)
        
        n = data['n']
        nBlocks = data['nBlocks']
        V = sparse.csr_matrix(data['V'], dtype = np.cfloat)
        PR = sparse.csr_matrix(data['PR_r']) + sparse.csr_matrix(data['PR_i'])*1j
        PL = sparse.csr_matrix(data['PL_r']) + sparse.csr_matrix(data['PL_i'])*1j

    return n, nBlocks, V, PR, PL

def save_matrices(n,nBlocks,fileName):
    # write v pl and pr to file    
    V, PR, PL = create_matrices_W(n,nBlocks)
    matrices = {'n': n, 'nBlocks': nBlocks, 'V': V.todense(), 'PR_r': PR.real.todense(),'PR_i': PR.imag.todense(), 'PL_r': PL.real.todense(), 'PL_i': PL.imag.todense()}

    with open(fileName, "w") as write_file:
        json.dump(matrices, write_file, cls=NumpyArrayEncoder)
    return

def mat_assembly_fullG(G_block, Gnn1_block, Bmin_fi, Bmax_fi, format = 'sparse', type = 'R'):
    
    Bmax = Bmax_fi -  1
    Bmin = Bmin_fi -  1
    NB = len(Bmin)
    NT = Bmax[-1] + 1
    
    if format == 'sparse':
        G = sparse.lil_matrix((NT,NT), dtype=np.cfloat)
    elif format == 'dense':
        G = np.zeros((NT,NT), dtype=np.cfloat)
    
    if type == 'R':
        for IB in range(NB):
            if IB == 0:
                G[Bmin[IB]:Bmax[IB]+1,Bmin[IB]:Bmax[IB+1]+1] = np.hstack([G_block[IB],Gnn1_block[IB]])
            elif IB == NB-1:
                G[Bmin[IB]:Bmax[IB]+1,Bmin[IB-1]:Bmax[IB]+1] = np.hstack([Gnn1_block[IB-1].T,G_block[IB]])
            else:
                G[Bmin[IB]:Bmax[IB]+1,Bmin[IB-1]:Bmax[IB+1]+1] = np.hstack([Gnn1_block[IB-1].T,G_block[IB],Gnn1_block[IB]])
    elif type == 'L' or type == 'G':
        for IB in range(NB):
            if IB == 0:
                G[Bmin[IB]:Bmax[IB]+1,Bmin[IB]:Bmax[IB+1]+1] = np.hstack([G_block[IB],Gnn1_block[IB]])
            elif IB == NB-1:
                G[Bmin[IB]:Bmax[IB]+1,Bmin[IB-1]:Bmax[IB]+1] = np.hstack([-Gnn1_block[IB-1].T.conj(),G_block[IB]])
            else:
                G[Bmin[IB]:Bmax[IB]+1,Bmin[IB-1]:Bmax[IB+1]+1] = np.hstack([-Gnn1_block[IB-1].T.conj(),G_block[IB],Gnn1_block[IB]])

    
    if format == 'sparse':
        G = G.tocsr()
    
    return G

def initialize_block_G(NE, NB, Bsize):
    GR_3D_E = np.zeros((NE, NB, Bsize, Bsize), dtype = np.complex128)
    GRnn1_3D_E = np.zeros((NE, NB - 1, Bsize, Bsize), dtype = np.complex128)
    GL_3D_E = np.zeros((NE, NB, Bsize, Bsize), dtype = np.complex128)
    GLnn1_3D_E = np.zeros((NE, NB - 1, Bsize, Bsize), dtype = np.complex128)
    GG_3D_E = np.zeros((NE, NB, Bsize, Bsize), dtype = np.complex128)
    GGnn1_3D_E = np.zeros((NE, NB - 1, Bsize, Bsize), dtype = np.complex128)

    return(GR_3D_E, GRnn1_3D_E, GL_3D_E, GLnn1_3D_E, GG_3D_E, GGnn1_3D_E)


def negative_hermitian_transpose(A : npt.NDArray[np.complex128]) -> npt.NDArray[np.complex128]:
   """Return the negative hermitian transpose of a matrix.
    Args:
         A: A matrix.
    Returns:
        The negative hermitian transpose of A.
    
   
   """
   return -A.T.conj()