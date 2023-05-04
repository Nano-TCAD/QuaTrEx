import numpy as np
from scipy import sparse
from mpi4py import MPI

def create_matrices(n, n_blocks, sparsity=1):
    """
    From Leo's code.
    Modified.
    Creates a Sparse block-tridiagonal matrix.
    
    Lesser/Greater self-energies should be purely imaginary??
    """
    H00 = sparse.random(n, n, 1, dtype = np.float64)
    H00 = (H00 + H00.T)/2                 # Make it symmetric
    H01 = sparse.random(n,n,1)
    
    SigL00 = 1j*sparse.random(n, n, 1, dtype = np.float64)
    SigL00 = (SigL00 - SigL00.T.conj())         #### Make it satisfy lesser/greater hc condition
    #SigL01 = sparse.csc_matrix((n, n))
    SigL01 = 1j*sparse.random(n, n, 1, dtype = np.float64)
    
    SigG00 = 1j*sparse.random(n, n, 1, dtype = np.float64)
    SigG00 = (SigG00 - SigG00.T.conj())         # Make it satisfy lesser/greater hc condition
    #SigG01 = sparse.csc_matrix((n, n))        
    SigG01 = 1j*sparse.random(n, n, 1, dtype = np.float64)
    
    
    n_temp = n
    
    while n_temp < n*n_blocks:
        H00 = sparse.vstack([sparse.hstack([H00, H01]), 
                             sparse.hstack([H01.T, H00])
                            ])
        H01 = sparse.vstack([sparse.csr_matrix((n_temp, 2*n_temp)), 
                             sparse.hstack([H01, sparse.csc_matrix((n_temp,n_temp))])
                            ])
        
        SigL00 = sparse.vstack([sparse.hstack([SigL00, SigL01]), 
                             sparse.hstack([-SigL01.T.conj(), SigL00])
                            ])
        SigL01 = sparse.vstack([sparse.csr_matrix((n_temp, 2*n_temp)), 
                             sparse.hstack([SigL01, sparse.csc_matrix((n_temp,n_temp))])
                            ])
        
        SigG00 = sparse.vstack([sparse.hstack([SigG00, SigG01]), 
                             sparse.hstack([-SigG01.T.conj(), SigG00])
                            ])
        SigG01 = sparse.vstack([sparse.csr_matrix((n_temp, 2*n_temp)), 
                             sparse.hstack([SigG01, sparse.csc_matrix((n_temp,n_temp))])
                            ])
        
        n_temp *= 2
    
    H = H00.tocsr()[:n*n_blocks,:n*n_blocks]
    SigL = SigL00.tocsr()[:n*n_blocks,:n*n_blocks]
    SigG = SigG00.tocsr()[:n*n_blocks,:n*n_blocks]
    
    return H, SigL, SigG

def step1_GF_rs(M, SigL, SigG, grR, grL, grG, Bmin, Bmax):
    """
    Right sided step-1 of RGF method.
    """

    NB = len(Bmin)

    # First step of iteration
    NN = Bmax[-1] - Bmin[-1] + 1
    
    grR[-1, 0:NN, 0:NN] = np.linalg.inv(M[Bmin[-1]:Bmax[-1]+1, Bmin[-1]:Bmax[-1]+1].toarray())
    grL[-1, 0:NN, 0:NN] = grR[-1, 0:NN, 0:NN] @ SigL[Bmin[-1]:Bmax[-1]+1, Bmin[-1]:Bmax[-1]+1] @ grR[-1, 0:NN, 0:NN].T.conj()
    grG[-1, 0:NN, 0:NN] = grR[-1, 0:NN, 0:NN] @ SigG[Bmin[-1]:Bmax[-1]+1, Bmin[-1]:Bmax[-1]+1] @ grR[-1, 0:NN, 0:NN].T.conj()

    for IB in range(NB-2, -1, -1):
    
        NI = Bmax[IB] - Bmin[IB] + 1
        NP = Bmax[IB+1] - Bmin[IB+1] + 1
       
        grR[IB, 0:NI, 0:NI] = np.linalg.inv(M[Bmin[IB]:Bmax[IB]+1, Bmin[IB]:Bmax[IB]+1].toarray() \
                             - M[Bmin[IB]:Bmax[IB]+1, Bmin[IB+1]:Bmax[IB+1]+1] \
                             @ grR[IB+1, 0:NP, 0:NP] \
                             @ M[Bmin[IB+1]:Bmax[IB+1]+1, Bmin[IB]:Bmax[IB]+1])
        
        #--------------------------------------------------------------------------------
        
        AL = M[Bmin[IB]:Bmax[IB]+1, Bmin[IB+1]:Bmax[IB+1]+1] \
             @ grR[IB+1, 0:NP, 0:NP] \
             @ SigL[Bmin[IB+1]:Bmax[IB+1]+1, Bmin[IB]:Bmax[IB]+1]

        grL[IB, 0:NI, 0:NI] = grR[IB, 0:NI, 0:NI] \
                             @ (SigL[Bmin[IB]:Bmax[IB]+1, Bmin[IB]:Bmax[IB]+1] \
                             + M[Bmin[IB]:Bmax[IB]+1, Bmin[IB+1]:Bmax[IB+1]+1] \
                             @ grL[IB+1, 0:NP, 0:NP] \
                             @ M[Bmin[IB]:Bmax[IB]+1, Bmin[IB+1]:Bmax[IB+1]+1].T.conj() \
                             - (AL - AL.T.conj()))  \
                             @ grR[IB, 0:NI, 0:NI].T.conj() 
    
        AG = M[Bmin[IB]:Bmax[IB]+1, Bmin[IB+1]:Bmax[IB+1]+1] \
             @ grR[IB+1, 0:NP, 0:NP] \
             @ SigG[Bmin[IB+1]:Bmax[IB+1]+1, Bmin[IB]:Bmax[IB]+1]     # Handling off-diagonal sigma elements? Prob. need to check

        grG[IB, 0:NI, 0:NI] = grR[IB, 0:NI, 0:NI] \
                             @ (SigG[Bmin[IB]:Bmax[IB]+1, Bmin[IB]:Bmax[IB]+1] \
                             + M[Bmin[IB]:Bmax[IB]+1, Bmin[IB+1]:Bmax[IB+1]+1] \
                             @ grG[IB+1, 0:NP, 0:NP] \
                             @ M[Bmin[IB]:Bmax[IB]+1, Bmin[IB+1]:Bmax[IB+1]+1].T.conj() \
                             - (AG - AG.T.conj())) \
                             @ grR[IB, 0:NI, 0:NI].T.conj() # Confused about the AG. 
     
    return

def step1_GF_ls(M, SigL, SigG, glR, glL, glG, Bmin, Bmax):
    """
    Left sided step-1 RGF method.
    """

    NB = len(Bmin)

    NF = Bmax[0] - Bmin[0] + 1
    glR[0, 0:NF, 0:NF] = np.linalg.inv(M[Bmin[0]:Bmax[0]+1, Bmin[0]:Bmax[0]+1].toarray())
    glL[0, 0:NF, 0:NF] = glR[0, 0:NF, 0:NF] @ SigL[Bmin[0]:Bmax[0]+1, Bmin[0]:Bmax[0]+1] @ glR[0, 0:NF, 0:NF].T.conj()
    glG[0, 0:NF, 0:NF] = glR[0, 0:NF, 0:NF] @ SigG[Bmin[0]:Bmax[0]+1, Bmin[0]:Bmax[0]+1] @ glR[0, 0:NF, 0:NF].T.conj()

    for IB in range(1, NB):
        
        NI = Bmax[IB] - Bmin[IB] + 1
        NM = Bmax[IB-1] - Bmin[IB-1] + 1
        
        glR[IB, 0:NI, 0:NI] = np.linalg.inv(M[Bmin[IB]:Bmax[IB]+1, Bmin[IB]:Bmax[IB]+1].toarray() \
                             - M[Bmin[IB]:Bmax[IB]+1, Bmin[IB-1]:Bmax[IB-1]+1] \
                             @ glR[IB-1, 0:NM, 0:NM] \
                             @ M[Bmin[IB-1]:Bmax[IB-1]+1, Bmin[IB]:Bmax[IB]+1])
           
                
        AL = M[Bmin[IB]:Bmax[IB]+1, Bmin[IB-1]:Bmax[IB-1]+1] \
             @ glR[IB-1, 0:NM, 0:NM] \
             @ SigL[Bmin[IB-1]:Bmax[IB-1]+1, Bmin[IB]:Bmax[IB]+1]

        glL[IB, 0:NI, 0:NI] = glR[IB, 0:NI, 0:NI] \
                             @ (SigL[Bmin[IB]:Bmax[IB]+1, Bmin[IB]:Bmax[IB]+1] \
                             + M[Bmin[IB]:Bmax[IB]+1, Bmin[IB-1]:Bmax[IB-1]+1] \
                             @ glL[IB-1, 0:NM, 0:NM] \
                             @ M[Bmin[IB]:Bmax[IB]+1, Bmin[IB-1]:Bmax[IB-1]+1].T.conj() \
                             - (AL - AL.T.conj()))  \
                             @ glR[IB, 0:NI, 0:NI].T.conj()
        
        
        AG = M[Bmin[IB]:Bmax[IB]+1, Bmin[IB-1]:Bmax[IB-1]+1] \
             @ glR[IB-1, 0:NM, 0:NM] \
             @ SigG[Bmin[IB-1]:Bmax[IB-1]+1, Bmin[IB]:Bmax[IB]+1]     # Handling off-diagonal sigma elements? Prob. need to check

        glG[IB, 0:NI, 0:NI] = glR[IB, 0:NI, 0:NI] \
                             @ (SigG[Bmin[IB]:Bmax[IB]+1, Bmin[IB]:Bmax[IB]+1] \
                             + M[Bmin[IB]:Bmax[IB]+1, Bmin[IB-1]:Bmax[IB-1]+1] \
                             @ glG[IB-1, 0:NM, 0:NM] \
                             @ M[Bmin[IB]:Bmax[IB]+1, Bmin[IB-1]:Bmax[IB-1]+1].T.conj() \
                             - (AG - AG.T.conj())) \
                             @ glR[IB, 0:NI, 0:NI].T.conj()  
    
    return


def step2_GF_rs(M, SigL, SigG, grR, grL, grG, GR, GRnn1, 
                GL, GLnn1, GG, GGnn1, Bmin, Bmax):
    """
    Right sided step-2 of RGF method.
    """

    NB = len(Bmin)

    for IB in range(1, NB):
        
        NM = Bmax[IB-1] - Bmin[IB-1] + 1
        NI = Bmax[IB] - Bmin[IB] + 1

        GR[IB, :NI, :NI] = grR[IB, :NI, :NI] + grR[IB, :NI, :NI] \
                           @ M[Bmin[IB]:Bmax[IB]+1, Bmin[IB-1]:Bmax[IB-1]+1] \
                           @ GR[IB-1,0:NM, 0:NM] \
                           @ M[Bmin[IB-1]:Bmax[IB-1]+1, Bmin[IB]:Bmax[IB]+1] \
                           @ grR[IB, :NI, :NI]
        
        AL = grR[IB, :NI, :NI] \
             @ SigL[Bmin[IB]:Bmax[IB]+1, Bmin[IB-1]:Bmax[IB-1]+1] \
             @ GR[IB-1, 0:NM, 0:NM].T.conj() \
             @ M[Bmin[IB]:Bmax[IB]+1, Bmin[IB-1]:Bmax[IB-1]+1].T.conj() \
             @ grR[IB, 0:NI, 0:NI].T.conj() 
        
        BL = grR[IB, :NI, :NI] \
             @ M[Bmin[IB]:Bmax[IB]+1, Bmin[IB-1]:Bmax[IB-1]+1] \
             @ GR[IB-1, 0:NM, 0:NM] \
             @ M[Bmin[IB-1]:Bmax[IB-1]+1, Bmin[IB]:Bmax[IB]+1] \
             @ grL[IB, :NI, :NI] 

        GL[IB][0:NI, 0:NI] = grL[IB, :NI, :NI] \
                             + grR[IB, 0:NI, 0:NI] \
                             @ M[Bmin[IB]:Bmax[IB]+1, Bmin[IB-1]:Bmax[IB-1]+1] \
                             @ GL[IB-1, 0:NM, 0:NM] \
                             @ M[Bmin[IB]:Bmax[IB]+1, Bmin[IB-1]:Bmax[IB-1]+1].T.conj() \
                             @ grR[IB, :NI, :NI].T.conj() \
                             - (AL - AL.T.conj()) + (BL - BL.T.conj()) 


        AG = grR[IB, 0:NI, 0:NI] \
             @ SigG[Bmin[IB]:Bmax[IB]+1, Bmin[IB-1]:Bmax[IB-1]+1] \
             @ GR[IB-1, 0:NM, 0:NM].T.conj() \
             @ M[Bmin[IB]:Bmax[IB]+1, Bmin[IB-1]:Bmax[IB-1]+1].T.conj() \
             @ grR[IB, 0:NI, 0:NI].T.conj() 

        BG = grR[IB, 0:NI, 0:NI] \
             @ M[Bmin[IB]:Bmax[IB]+1, Bmin[IB-1]:Bmax[IB-1]+1] \
             @ GR[IB-1, 0:NM, 0:NM] \
             @ M[Bmin[IB-1]:Bmax[IB-1]+1, Bmin[IB]:Bmax[IB]+1] \
             @ grG[IB, 0:NI, 0:NI]

        GG[IB, 0:NI, 0:NI] = grG[IB, 0:NI, 0:NI] \
                             + grR[IB, 0:NI, 0:NI] \
                             @ M[Bmin[IB]:Bmax[IB]+1, Bmin[IB-1]:Bmax[IB-1]+1] \
                             @ GG[IB-1, 0:NM, 0:NM] \
                             @ M[Bmin[IB]:Bmax[IB]+1, Bmin[IB-1]:Bmax[IB-1]+1].T.conj() \
                             @ grR[IB, 0:NI, 0:NI].T.conj() \
                             - (AG - AG.T.conj()) + (BG - BG.T.conj()) # 

        if IB < NB-1:  #Off-diagonal are only interesting for IdE!
            NP = Bmax[IB+1] - Bmin[IB+1] + 1

            GRnn1[IB, 0:NI, 0:NP] = - GR[IB, 0:NI, 0:NI] \
                                    @ M[Bmin[IB]:Bmax[IB]+1, Bmin[IB+1]:Bmax[IB+1]+1] \
                                    @ grR[IB+1, 0:NP, 0:NP]

            GLnn1[IB, 0:NI, 0:NP] = GR[IB, 0:NI, 0:NI] \
                                    @ SigL[Bmin[IB]:Bmax[IB]+1, Bmin[IB+1]:Bmax[IB+1]+1] \
                                    @ grR[IB+1, 0:NP, 0:NP].T.conj() \
                                    - GR[IB, 0:NI, 0:NI] \
                                    @ M[Bmin[IB]:Bmax[IB]+1, Bmin[IB+1]:Bmax[IB+1]+1] \
                                    @ grL[IB+1, 0:NP, 0:NP] \
                                    - GL[IB, :NI, :NI] \
                                    @ M[Bmin[IB+1]:Bmax[IB+1]+1, Bmin[IB]:Bmax[IB]+1].T.conj() \
                                    @ grR[IB+1, 0:NP, 0:NP].T.conj()   
            GGnn1[IB, 0:NI, 0:NP] = GR[IB, 0:NI, 0:NI] \
                                    @ SigG[Bmin[IB]:Bmax[IB]+1, Bmin[IB+1]:Bmax[IB+1]+1] \
                                    @ grR[IB+1, 0:NP, 0:NP].T.conj() \
                                    - GR[IB, 0:NI, 0:NI] \
                                    @ M[Bmin[IB]:Bmax[IB]+1, Bmin[IB+1]:Bmax[IB+1]+1] \
                                    @ grG[IB+1, 0:NP, 0:NP] \
                                    - GG[IB, 0:NI, 0:NI] \
                                    @ M[Bmin[IB+1]:Bmax[IB+1]+1, Bmin[IB]:Bmax[IB]+1].T.conj() \
                                    @ grR[IB+1, 0:NP, 0:NP].T.conj()
    return

def step2_GF_ls(M, SigL, SigG, glR, glL, glG, GR, GRnn1, 
                GL, GLnn1, GG, GGnn1, Bmin, Bmax):
    """
    Left sided step-2 RGF method.
    """

    NB = len(Bmin)

    for IB in range(NB-2, -1, -1):
        
        NP = Bmax[IB+1] - Bmin[IB+1] + 1
        NI = Bmax[IB] - Bmin[IB] + 1

        GR[IB, :NI, :NI] = glR[IB, :NI, :NI] + glR[IB, :NI, :NI] \
                           @ M[Bmin[IB]:Bmax[IB]+1, Bmin[IB+1]:Bmax[IB+1]+1] \
                           @ GR[IB+1,0:NP, 0:NP] \
                           @ M[Bmin[IB+1]:Bmax[IB+1]+1, Bmin[IB]:Bmax[IB]+1] \
                           @ glR[IB, :NI, :NI]
  
        
        AL = glR[IB, :NI, :NI] \
             @ SigL[Bmin[IB]:Bmax[IB]+1, Bmin[IB+1]:Bmax[IB+1]+1] \
             @ GR[IB+1, 0:NP, 0:NP].T.conj() \
             @ M[Bmin[IB]:Bmax[IB]+1, Bmin[IB+1]:Bmax[IB+1]+1].T.conj() \
             @ glR[IB, 0:NI, 0:NI].T.conj() 

        BL = glR[IB, :NI, :NI] \
             @ M[Bmin[IB]:Bmax[IB]+1, Bmin[IB+1]:Bmax[IB+1]+1] \
             @ GR[IB+1, 0:NP, 0:NP] \
             @ M[Bmin[IB+1]:Bmax[IB+1]+1, Bmin[IB]:Bmax[IB]+1] \
             @ glL[IB, :NI, :NI] 

        GL[IB][0:NI, 0:NI] = glL[IB, :NI, :NI] \
                             + glR[IB, 0:NI, 0:NI] \
                             @ M[Bmin[IB]:Bmax[IB]+1, Bmin[IB+1]:Bmax[IB+1]+1] \
                             @ GL[IB+1, 0:NP, 0:NP] \
                             @ M[Bmin[IB]:Bmax[IB]+1, Bmin[IB+1]:Bmax[IB+1]+1].T.conj() \
                             @ glR[IB, :NI, :NI].T.conj() \
                             - (AL - AL.T.conj()) + (BL - BL.T.conj()) 


        AG = glR[IB, 0:NI, 0:NI] \
             @ SigG[Bmin[IB]:Bmax[IB]+1, Bmin[IB+1]:Bmax[IB+1]+1] \
             @ GR[IB+1, 0:NP, 0:NP].T.conj() \
             @ M[Bmin[IB]:Bmax[IB]+1, Bmin[IB+1]:Bmax[IB+1]+1].T.conj() \
             @ glR[IB, 0:NI, 0:NI].T.conj() 

        BG = glR[IB, 0:NI, 0:NI] \
             @ M[Bmin[IB]:Bmax[IB]+1, Bmin[IB+1]:Bmax[IB+1]+1] \
             @ GR[IB+1, 0:NP, 0:NP] \
             @ M[Bmin[IB+1]:Bmax[IB+1]+1, Bmin[IB]:Bmax[IB]+1] \
             @ glG[IB, 0:NI, 0:NI]

        GG[IB, 0:NI, 0:NI] = glG[IB, 0:NI, 0:NI] \
                             + glR[IB, 0:NI, 0:NI] \
                             @ M[Bmin[IB]:Bmax[IB]+1, Bmin[IB+1]:Bmax[IB+1]+1] \
                             @ GG[IB+1, 0:NP, 0:NP] \
                             @ M[Bmin[IB]:Bmax[IB]+1, Bmin[IB+1]:Bmax[IB+1]+1].T.conj() \
                             @ glR[IB, 0:NI, 0:NI].T.conj() \
                             - (AG - AG.T.conj()) + (BG - BG.T.conj()) # 
        
        #if IB > 0:  #Off-diagonal are only interesting for IdE! Can probably use these results also.
        #    NM = Bmax[IB-1] - Bmin[IB-1] + 1

        GRnn1[IB, 0:NI, 0:NP] = - glR[IB, 0:NI, 0:NI] \
                            @ M[Bmin[IB]:Bmax[IB]+1, Bmin[IB+1]:Bmax[IB+1]+1] \
                            @ GR[IB+1, 0:NP, 0:NP]

        GLnn1[IB, 0:NI, 0:NP] = - glR[IB, 0:NI, 0:NI] \
                            @ M[Bmin[IB]:Bmax[IB]+1, Bmin[IB+1]:Bmax[IB+1]+1] \
                            @ GL[IB+1, :NP, :NP] \
                            - glL[IB, 0:NI, 0:NI] \
                            @ M[Bmin[IB+1]:Bmax[IB+1]+1, Bmin[IB]:Bmax[IB]+1].T.conj() \
                            @ GR[IB+1, 0:NP, 0:NP].T.conj() \
                            + glR[IB, 0:NI, 0:NI] \
                            @ SigL[Bmin[IB]:Bmax[IB]+1, Bmin[IB+1]:Bmax[IB+1]+1] \
                            @ GR[IB+1, 0:NP, 0:NP].T.conj()   
                            
        GGnn1[IB, 0:NI, 0:NP] = - glR[IB, 0:NI, 0:NI] \
                            @ M[Bmin[IB]:Bmax[IB]+1, Bmin[IB+1]:Bmax[IB+1]+1] \
                            @ GG[IB+1, :NP, :NP] \
                            - glG[IB, 0:NI, 0:NI] \
                            @ M[Bmin[IB+1]:Bmax[IB+1]+1, Bmin[IB]:Bmax[IB]+1].T.conj() \
                            @ GR[IB+1, 0:NP, 0:NP].T.conj() \
                            + glR[IB, 0:NI, 0:NI] \
                            @ SigG[Bmin[IB]:Bmax[IB]+1, Bmin[IB+1]:Bmax[IB+1]+1] \
                            @ GR[IB+1, 0:NP, 0:NP].T.conj()
    return

def rgf_GF_2S_V0(M, SigL, SigG, Bmin, Bmax):
    # rgf_GF(DH, E, EfL, EfR, Temp) This could be the function call considering Leo's code
    '''
    2-sided rgf
    
    Working!

    Sequential.
    '''

    Bsize = max(Bmax - Bmin + 1) # Used for declaration of variables
    NB = len(Bmin)

    grR = np.zeros((NB, Bsize, Bsize), dtype=np.cfloat) # Retarded (right)
    grL = np.zeros((NB, Bsize, Bsize), dtype=np.cfloat) # Lesser (right)
    grG = np.zeros((NB, Bsize, Bsize), dtype=np.cfloat) # Greater (right)
    glR = np.zeros((NB, Bsize, Bsize), dtype=np.cfloat) # Retarded (left)
    glL = np.zeros((NB, Bsize, Bsize), dtype=np.cfloat) # Lesser (left)
    glG = np.zeros((NB, Bsize, Bsize), dtype=np.cfloat) # Greater (left)
    
    GR = np.zeros((NB, Bsize, Bsize), dtype=np.cfloat) # Retarded GF
    GRnn1 = np.zeros((NB - 1, Bsize, Bsize), dtype=np.cfloat) #Off-diagonal GR
    GL = np.zeros((NB, Bsize, Bsize), dtype=np.cfloat) # Lesser GF
    GLnn1 = np.zeros((NB - 1, Bsize, Bsize), dtype=np.cfloat) # Off-diagonal GL
    GG = np.zeros((NB, Bsize, Bsize), dtype=np.cfloat) # Greater GF
    GGnn1 = np.zeros((NB - 1, Bsize, Bsize), dtype=np.cfloat) # Off-diagonal GG

    # First step of iteration
    NN = Bmax[-1] - Bmin[-1] + 1
    grR[-1, 0:NN, 0:NN] = np.linalg.inv(M[Bmin[-1]:Bmax[-1]+1, Bmin[-1]:Bmax[-1]+1].toarray())
    grL[-1, 0:NN, 0:NN] = grR[-1, 0:NN, 0:NN] @ SigL[Bmin[-1]:Bmax[-1]+1, Bmin[-1]:Bmax[-1]+1] @ grR[-1, 0:NN, 0:NN].T.conj()
    grG[-1, 0:NN, 0:NN] = grR[-1, 0:NN, 0:NN] @ SigG[Bmin[-1]:Bmax[-1]+1, Bmin[-1]:Bmax[-1]+1] @ grR[-1, 0:NN, 0:NN].T.conj()
    
    NF = Bmax[0] - Bmin[0] + 1
    glR[0, 0:NF, 0:NF] = np.linalg.inv(M[Bmin[0]:Bmax[0]+1, Bmin[0]:Bmax[0]+1].toarray())
    glL[0, 0:NF, 0:NF] = glR[0, 0:NF, 0:NF] @ SigL[Bmin[0]:Bmax[0]+1, Bmin[0]:Bmax[0]+1] @ glR[0, 0:NF, 0:NF].T.conj()
    glG[0, 0:NF, 0:NF] = glR[0, 0:NF, 0:NF] @ SigG[Bmin[0]:Bmax[0]+1, Bmin[0]:Bmax[0]+1] @ glR[0, 0:NF, 0:NF].T.conj()

    for IB in range(NB-2, -1, -1):
        IIB = NB - 1 - IB # For left
        
        NI = Bmax[IB] - Bmin[IB] + 1
        NP = Bmax[IB+1] - Bmin[IB+1] + 1
        
        NNM = Bmax[IIB-1] - Bmin[IIB-1] + 1
        NNI = Bmax[IIB] - Bmin[IIB] + 1
        
       
        grR[IB, 0:NI, 0:NI] = np.linalg.inv(M[Bmin[IB]:Bmax[IB]+1, Bmin[IB]:Bmax[IB]+1].toarray() \
                             - M[Bmin[IB]:Bmax[IB]+1, Bmin[IB+1]:Bmax[IB+1]+1] \
                             @ grR[IB+1, 0:NP, 0:NP] \
                             @ M[Bmin[IB+1]:Bmax[IB+1]+1, Bmin[IB]:Bmax[IB]+1])
        
        glR[IIB, 0:NNI, 0:NNI] = np.linalg.inv(M[Bmin[IIB]:Bmax[IIB]+1, Bmin[IIB]:Bmax[IIB]+1].toarray() \
                             - M[Bmin[IIB]:Bmax[IIB]+1, Bmin[IIB-1]:Bmax[IIB-1]+1] \
                             @ glR[IIB-1, 0:NNM, 0:NNM] \
                             @ M[Bmin[IIB-1]:Bmax[IIB-1]+1, Bmin[IIB]:Bmax[IIB]+1])
        
        #--------------------------------------------------------------------------------
        
        # AL, What is this? Handling off-diagonal sigma elements
        AL = M[Bmin[IB]:Bmax[IB]+1, Bmin[IB+1]:Bmax[IB+1]+1] \
             @ grR[IB+1, 0:NP, 0:NP] \
             @ SigL[Bmin[IB+1]:Bmax[IB+1]+1, Bmin[IB]:Bmax[IB]+1]

        grL[IB, 0:NI, 0:NI] = grR[IB, 0:NI, 0:NI] \
                             @ (SigL[Bmin[IB]:Bmax[IB]+1, Bmin[IB]:Bmax[IB]+1] \
                             + M[Bmin[IB]:Bmax[IB]+1, Bmin[IB+1]:Bmax[IB+1]+1] \
                             @ grL[IB+1, 0:NP, 0:NP] \
                             @ M[Bmin[IB]:Bmax[IB]+1, Bmin[IB+1]:Bmax[IB+1]+1].T.conj() \
                             - (AL - AL.T.conj()))  \
                             @ grR[IB, 0:NI, 0:NI].T.conj() 
        ### What is this?
        AG = M[Bmin[IB]:Bmax[IB]+1, Bmin[IB+1]:Bmax[IB+1]+1] \
             @ grR[IB+1, 0:NP, 0:NP] \
             @ SigG[Bmin[IB+1]:Bmax[IB+1]+1, Bmin[IB]:Bmax[IB]+1]     # Handling off-diagonal sigma elements? Prob. need to check

        grG[IB, 0:NI, 0:NI] = grR[IB, 0:NI, 0:NI] \
                             @ (SigG[Bmin[IB]:Bmax[IB]+1, Bmin[IB]:Bmax[IB]+1] \
                             + M[Bmin[IB]:Bmax[IB]+1, Bmin[IB+1]:Bmax[IB+1]+1] \
                             @ grG[IB+1, 0:NP, 0:NP] \
                             @ M[Bmin[IB]:Bmax[IB]+1, Bmin[IB+1]:Bmax[IB+1]+1].T.conj() \
                             - (AG - AG.T.conj())) \
                             @ grR[IB, 0:NI, 0:NI].T.conj() # Confused about the AG. 
     
                
        #--------------------------------------------------------------------------------        
                
        AL = M[Bmin[IIB]:Bmax[IIB]+1, Bmin[IIB-1]:Bmax[IIB-1]+1] \
             @ glR[IIB-1, 0:NNM, 0:NNM] \
             @ SigL[Bmin[IIB-1]:Bmax[IIB-1]+1, Bmin[IIB]:Bmax[IIB]+1]

        glL[IIB, 0:NNI, 0:NNI] = glR[IIB, 0:NNI, 0:NNI] \
                             @ (SigL[Bmin[IIB]:Bmax[IIB]+1, Bmin[IIB]:Bmax[IIB]+1] \
                             + M[Bmin[IIB]:Bmax[IIB]+1, Bmin[IIB-1]:Bmax[IIB-1]+1] \
                             @ glL[IIB-1, 0:NNM, 0:NNM] \
                             @ M[Bmin[IIB]:Bmax[IIB]+1, Bmin[IIB-1]:Bmax[IIB-1]+1].T.conj() \
                             - (AL - AL.T.conj()))  \
                             @ glR[IIB, 0:NNI, 0:NNI].T.conj()
        
        ### What is this?
        AG = M[Bmin[IIB]:Bmax[IIB]+1, Bmin[IIB-1]:Bmax[IIB-1]+1] \
             @ glR[IB-1, 0:NNM, 0:NNM] \
             @ SigG[Bmin[IIB-1]:Bmax[IIB-1]+1, Bmin[IIB]:Bmax[IIB]+1]     # Handling off-diagonal sigma elements? Prob. need to check

        glG[IIB, 0:NNI, 0:NNI] = glR[IIB, 0:NNI, 0:NNI] \
                             @ (SigG[Bmin[IIB]:Bmax[IIB]+1, Bmin[IIB]:Bmax[IIB]+1] \
                             + M[Bmin[IIB]:Bmax[IIB]+1, Bmin[IIB-1]:Bmax[IIB-1]+1] \
                             @ glG[IIB-1, 0:NNM, 0:NNM] \
                             @ M[Bmin[IIB]:Bmax[IIB]+1, Bmin[IIB-1]:Bmax[IIB-1]+1].T.conj() \
                             - (AG - AG.T.conj())) \
                             @ glR[IIB, 0:NNI, 0:NNI].T.conj() # Confused about the AG. 
    
        #--------------------------------------------------------------------------------
    
    EI = int(NB/2) # Exchange index
    
    NM = Bmax[EI-1] - Bmin[EI-1] + 1
    NI = Bmax[EI] - Bmin[EI] + 1
    NP = Bmax[EI+1] - Bmin[EI+1] + 1

    #Second step of iteration
    
    GR[EI, :NI, :NI] = np.linalg.inv(M[Bmin[EI]:Bmax[EI]+1, Bmin[EI]:Bmax[EI]+1]
                                   - M[Bmin[EI]:Bmax[EI]+1, Bmin[EI-1]:Bmax[EI-1]+1]
                                   @ glR[EI-1, :NM, :NM]
                                   @ M[Bmin[EI-1]:Bmax[EI-1]+1, Bmin[EI]:Bmax[EI]+1]
                                   - M[Bmin[EI]:Bmax[EI]+1, Bmin[EI+1]:Bmax[EI+1]+1]
                                   @ grR[EI+1, :NP, :NP]
                                   @ M[Bmin[EI+1]:Bmax[EI+1]+1, Bmin[EI]:Bmax[EI]+1])
    GRnn1[EI, :NI, :NI] = - GR[EI, 0:NI, 0:NI] \
                            @ M[Bmin[EI]:Bmax[EI]+1, Bmin[EI+1]:Bmax[EI+1]+1] \
                            @ grR[EI+1, 0:NP, 0:NP] # Could also use left 
    
    GRnn1[EI-1, :NI, :NI] = - glR[EI-1, 0:NM, 0:NM] \
                            @ M[Bmin[EI-1]:Bmax[EI-1]+1, Bmin[EI]:Bmax[EI]+1] \
                            @ GR[EI, 0:NI, 0:NI]
            
    CL = M[Bmin[EI]:Bmax[EI]+1, Bmin[EI-1]:Bmax[EI-1]+1] \
         @ glR[EI-1, 0:NM, 0:NM] \
         @ SigL[Bmin[EI-1]:Bmax[EI-1]+1, Bmin[EI]:Bmax[EI]+1] \
            
    DL = M[Bmin[EI]:Bmax[EI]+1, Bmin[EI+1]:Bmax[EI+1]+1] \
         @ grR[EI+1, 0:NP, 0:NP] \
         @ SigL[Bmin[EI+1]:Bmax[EI+1]+1, Bmin[EI]:Bmax[EI]+1] \
    
    GL[EI, :NI, :NI] = GR[EI, :NI, :NI] @ (
                        M[Bmin[EI]:Bmax[EI]+1, Bmin[EI-1]:Bmax[EI-1]+1]
                        @ glL[EI-1, :NM, :NM]
                        @ M[Bmin[EI]:Bmax[EI]+1, Bmin[EI-1]:Bmax[EI-1]+1].T.conj()
                        + M[Bmin[EI]:Bmax[EI]+1, Bmin[EI+1]:Bmax[EI+1]+1]
                        @ grL[EI+1, :NP, :NP]
                        @ M[Bmin[EI]:Bmax[EI]+1, Bmin[EI+1]:Bmax[EI+1]+1].T.conj()
                        + SigL[Bmin[EI]:Bmax[EI]+1, Bmin[EI]:Bmax[EI]+1]
                        - (CL-CL.T.conj())
                        - (DL-DL.T.conj())) @ GR[EI, :NI, :NI].T.conj()
    
    GLnn1[EI, :NI, :NI] = GR[EI, 0:NI, 0:NI] \
                        @ SigL[Bmin[EI]:Bmax[EI]+1, Bmin[EI+1]:Bmax[EI+1]+1] \
                        @ grR[EI+1, 0:NP, 0:NP].T.conj() \
                        - GR[EI, 0:NI, 0:NI] \
                        @ M[Bmin[EI]:Bmax[EI]+1, Bmin[EI+1]:Bmax[EI+1]+1] \
                        @ grL[EI+1, 0:NP, 0:NP] \
                        - GL[EI, :NI, :NI] \
                        @ M[Bmin[EI+1]:Bmax[EI+1]+1, Bmin[EI]:Bmax[EI]+1].T.conj() \
                        @ grR[EI+1, 0:NP, 0:NP].T.conj() # Could also have used the left
                        
    GLnn1[EI-1, :NI, :NI] = - glR[EI-1, 0:NM, 0:NM] \
                        @ M[Bmin[EI-1]:Bmax[EI-1]+1, Bmin[EI]:Bmax[EI]+1] \
                        @ GL[EI, :NI, :NI] \
                        - glL[EI-1, 0:NM, 0:NM] \
                        @ M[Bmin[EI]:Bmax[EI]+1, Bmin[EI-1]:Bmax[EI-1]+1].T.conj() \
                        @ GR[EI, 0:NI, 0:NI].T.conj() \
                        + glR[EI-1, 0:NM, 0:NM] \
                        @ SigL[Bmin[EI-1]:Bmax[EI-1]+1, Bmin[EI]:Bmax[EI]+1] \
                        @ GR[EI, 0:NI, 0:NI].T.conj()

    CG = M[Bmin[EI]:Bmax[EI]+1, Bmin[EI-1]:Bmax[EI-1]+1] \
         @ glR[EI-1, 0:NM, 0:NM] \
         @ SigG[Bmin[EI-1]:Bmax[EI-1]+1, Bmin[EI]:Bmax[EI]+1]
            
    DG = M[Bmin[EI]:Bmax[EI]+1, Bmin[EI+1]:Bmax[EI+1]+1] \
         @ grR[EI+1, 0:NP, 0:NP] \
         @ SigG[Bmin[EI+1]:Bmax[EI+1]+1, Bmin[EI]:Bmax[EI]+1]                   
                        
    GG[EI, :NI, :NI] = GR[EI, :NI, :NI] @ (
                        M[Bmin[EI]:Bmax[EI]+1, Bmin[EI-1]:Bmax[EI-1]+1]
                        @ glG[EI-1, :NM, :NM]
                        @ M[Bmin[EI]:Bmax[EI]+1, Bmin[EI-1]:Bmax[EI-1]+1].T.conj()
                        + M[Bmin[EI]:Bmax[EI]+1, Bmin[EI+1]:Bmax[EI+1]+1]
                        @ grG[EI+1, :NP, :NP]
                        @ M[Bmin[EI]:Bmax[EI]+1, Bmin[EI+1]:Bmax[EI+1]+1].T.conj()
                        + SigG[Bmin[EI]:Bmax[EI]+1, Bmin[EI]:Bmax[EI]+1]
                        - (CG-CG.T.conj())
                        - (DG-DG.T.conj())) @ GR[EI, :NI, :NI].T.conj()
    
    GGnn1[EI, :NI, :NI] = GR[EI, 0:NI, 0:NI] \
                        @ SigG[Bmin[EI]:Bmax[EI]+1, Bmin[EI+1]:Bmax[EI+1]+1] \
                        @ grR[EI+1, 0:NP, 0:NP].T.conj() \
                        - GR[EI, 0:NI, 0:NI] \
                        @ M[Bmin[EI]:Bmax[EI]+1, Bmin[EI+1]:Bmax[EI+1]+1] \
                        @ grG[EI+1, 0:NP, 0:NP] \
                        - GG[EI, :NI, :NI] \
                        @ M[Bmin[EI+1]:Bmax[EI+1]+1, Bmin[EI]:Bmax[EI]+1].T.conj() \
                        @ grR[EI+1, 0:NP, 0:NP].T.conj() # Could also have used the left      
                        
    GGnn1[EI-1, :NI, :NI] = - glR[EI-1, 0:NM, 0:NM] \
                        @ M[Bmin[EI-1]:Bmax[EI-1]+1, Bmin[EI]:Bmax[EI]+1] \
                        @ GG[EI, :NI, :NI] \
                        - glG[EI-1, 0:NM, 0:NM] \
                        @ M[Bmin[EI]:Bmax[EI]+1, Bmin[EI-1]:Bmax[EI-1]+1].T.conj() \
                        @ GR[EI, 0:NI, 0:NI].T.conj() \
                        + glR[EI-1, 0:NM, 0:NM] \
                        @ SigG[Bmin[EI-1]:Bmax[EI-1]+1, Bmin[EI]:Bmax[EI]+1] \
                        @ GR[EI, 0:NI, 0:NI].T.conj()


    # -------------------------------------------------------------------------------------------------------
    
    # First loop
    for IB in range(EI+1, NB):
        
        NM = Bmax[IB-1] - Bmin[IB-1] + 1
        NI = Bmax[IB] - Bmin[IB] + 1

        GR[IB, :NI, :NI] = grR[IB, :NI, :NI] + grR[IB, :NI, :NI] \
                           @ M[Bmin[IB]:Bmax[IB]+1, Bmin[IB-1]:Bmax[IB-1]+1] \
                           @ GR[IB-1,0:NM, 0:NM] \
                           @ M[Bmin[IB-1]:Bmax[IB-1]+1, Bmin[IB]:Bmax[IB]+1] \
                           @ grR[IB, :NI, :NI]
        # What is this? Handling off-diagonal elements?
        AL = grR[IB, :NI, :NI] \
             @ SigL[Bmin[IB]:Bmax[IB]+1, Bmin[IB-1]:Bmax[IB-1]+1] \
             @ GR[IB-1, 0:NM, 0:NM].T.conj() \
             @ M[Bmin[IB]:Bmax[IB]+1, Bmin[IB-1]:Bmax[IB-1]+1].T.conj() \
             @ grR[IB, 0:NI, 0:NI].T.conj() 
        # What is this?
        BL = grR[IB, :NI, :NI] \
             @ M[Bmin[IB]:Bmax[IB]+1, Bmin[IB-1]:Bmax[IB-1]+1] \
             @ GR[IB-1, 0:NM, 0:NM] \
             @ M[Bmin[IB-1]:Bmax[IB-1]+1, Bmin[IB]:Bmax[IB]+1] \
             @ grL[IB, :NI, :NI] 

        GL[IB][0:NI, 0:NI] = grL[IB, :NI, :NI] \
                             + grR[IB, 0:NI, 0:NI] \
                             @ M[Bmin[IB]:Bmax[IB]+1, Bmin[IB-1]:Bmax[IB-1]+1] \
                             @ GL[IB-1, 0:NM, 0:NM] \
                             @ M[Bmin[IB]:Bmax[IB]+1, Bmin[IB-1]:Bmax[IB-1]+1].T.conj() \
                             @ grR[IB, :NI, :NI].T.conj() \
                             - (AL - AL.T.conj()) + (BL - BL.T.conj()) 


        AG = grR[IB, 0:NI, 0:NI] \
             @ SigG[Bmin[IB]:Bmax[IB]+1, Bmin[IB-1]:Bmax[IB-1]+1] \
             @ GR[IB-1, 0:NM, 0:NM].T.conj() \
             @ M[Bmin[IB]:Bmax[IB]+1, Bmin[IB-1]:Bmax[IB-1]+1].T.conj() \
             @ grR[IB, 0:NI, 0:NI].T.conj() 

        BG = grR[IB, 0:NI, 0:NI] \
             @ M[Bmin[IB]:Bmax[IB]+1, Bmin[IB-1]:Bmax[IB-1]+1] \
             @ GR[IB-1, 0:NM, 0:NM] \
             @ M[Bmin[IB-1]:Bmax[IB-1]+1, Bmin[IB]:Bmax[IB]+1] \
             @ grG[IB, 0:NI, 0:NI]

        GG[IB, 0:NI, 0:NI] = grG[IB, 0:NI, 0:NI] \
                             + grR[IB, 0:NI, 0:NI] \
                             @ M[Bmin[IB]:Bmax[IB]+1, Bmin[IB-1]:Bmax[IB-1]+1] \
                             @ GG[IB-1, 0:NM, 0:NM] \
                             @ M[Bmin[IB]:Bmax[IB]+1, Bmin[IB-1]:Bmax[IB-1]+1].T.conj() \
                             @ grR[IB, 0:NI, 0:NI].T.conj() \
                             - (AG - AG.T.conj()) + (BG - BG.T.conj()) # 

        if IB < NB-1:  #Off-diagonal are only interesting for IdE!
            NP = Bmax[IB+1] - Bmin[IB+1] + 1

            GRnn1[IB, 0:NI, 0:NP] = - GR[IB, 0:NI, 0:NI] \
                                    @ M[Bmin[IB]:Bmax[IB]+1, Bmin[IB+1]:Bmax[IB+1]+1] \
                                    @ grR[IB+1, 0:NP, 0:NP]

            GLnn1[IB, 0:NI, 0:NP] = GR[IB, 0:NI, 0:NI] \
                                    @ SigL[Bmin[IB]:Bmax[IB]+1, Bmin[IB+1]:Bmax[IB+1]+1] \
                                    @ grR[IB+1, 0:NP, 0:NP].T.conj() \
                                    - GR[IB, 0:NI, 0:NI] \
                                    @ M[Bmin[IB]:Bmax[IB]+1, Bmin[IB+1]:Bmax[IB+1]+1] \
                                    @ grL[IB+1, 0:NP, 0:NP] \
                                    - GL[IB, :NI, :NI] \
                                    @ M[Bmin[IB+1]:Bmax[IB+1]+1, Bmin[IB]:Bmax[IB]+1].T.conj() \
                                    @ grR[IB+1, 0:NP, 0:NP].T.conj()   
            GGnn1[IB, 0:NI, 0:NP] = GR[IB, 0:NI, 0:NI] \
                                    @ SigG[Bmin[IB]:Bmax[IB]+1, Bmin[IB+1]:Bmax[IB+1]+1] \
                                    @ grR[IB+1, 0:NP, 0:NP].T.conj() \
                                    - GR[IB, 0:NI, 0:NI] \
                                    @ M[Bmin[IB]:Bmax[IB]+1, Bmin[IB+1]:Bmax[IB+1]+1] \
                                    @ grG[IB+1, 0:NP, 0:NP] \
                                    - GG[IB, 0:NI, 0:NI] \
                                    @ M[Bmin[IB+1]:Bmax[IB+1]+1, Bmin[IB]:Bmax[IB]+1].T.conj() \
                                    @ grR[IB+1, 0:NP, 0:NP].T.conj()
    
    # Second loop
    for IB in range(EI-1, -1, -1):
        
        NP = Bmax[IB+1] - Bmin[IB+1] + 1
        NI = Bmax[IB] - Bmin[IB] + 1

        GR[IB, :NI, :NI] = glR[IB, :NI, :NI] + glR[IB, :NI, :NI] \
                           @ M[Bmin[IB]:Bmax[IB]+1, Bmin[IB+1]:Bmax[IB+1]+1] \
                           @ GR[IB+1,0:NP, 0:NP] \
                           @ M[Bmin[IB+1]:Bmax[IB+1]+1, Bmin[IB]:Bmax[IB]+1] \
                           @ glR[IB, :NI, :NI]
        # What is this? Handling off-diagonal elements?
        AL = glR[IB, :NI, :NI] \
             @ SigL[Bmin[IB]:Bmax[IB]+1, Bmin[IB+1]:Bmax[IB+1]+1] \
             @ GR[IB+1, 0:NP, 0:NP].T.conj() \
             @ M[Bmin[IB]:Bmax[IB]+1, Bmin[IB+1]:Bmax[IB+1]+1].T.conj() \
             @ glR[IB, 0:NI, 0:NI].T.conj() 

        BL = glR[IB, :NI, :NI] \
             @ M[Bmin[IB]:Bmax[IB]+1, Bmin[IB+1]:Bmax[IB+1]+1] \
             @ GR[IB+1, 0:NP, 0:NP] \
             @ M[Bmin[IB+1]:Bmax[IB+1]+1, Bmin[IB]:Bmax[IB]+1] \
             @ glL[IB, :NI, :NI] 

        GL[IB][0:NI, 0:NI] = glL[IB, :NI, :NI] \
                             + glR[IB, 0:NI, 0:NI] \
                             @ M[Bmin[IB]:Bmax[IB]+1, Bmin[IB+1]:Bmax[IB+1]+1] \
                             @ GL[IB+1, 0:NP, 0:NP] \
                             @ M[Bmin[IB]:Bmax[IB]+1, Bmin[IB+1]:Bmax[IB+1]+1].T.conj() \
                             @ glR[IB, :NI, :NI].T.conj() \
                             - (AL - AL.T.conj()) + (BL - BL.T.conj()) 


        AG = glR[IB, 0:NI, 0:NI] \
             @ SigG[Bmin[IB]:Bmax[IB]+1, Bmin[IB+1]:Bmax[IB+1]+1] \
             @ GR[IB+1, 0:NP, 0:NP].T.conj() \
             @ M[Bmin[IB]:Bmax[IB]+1, Bmin[IB+1]:Bmax[IB+1]+1].T.conj() \
             @ glR[IB, 0:NI, 0:NI].T.conj() 

        BG = glR[IB, 0:NI, 0:NI] \
             @ M[Bmin[IB]:Bmax[IB]+1, Bmin[IB+1]:Bmax[IB+1]+1] \
             @ GR[IB+1, 0:NP, 0:NP] \
             @ M[Bmin[IB+1]:Bmax[IB+1]+1, Bmin[IB]:Bmax[IB]+1] \
             @ glG[IB, 0:NI, 0:NI]

        GG[IB, 0:NI, 0:NI] = glG[IB, 0:NI, 0:NI] \
                             + glR[IB, 0:NI, 0:NI] \
                             @ M[Bmin[IB]:Bmax[IB]+1, Bmin[IB+1]:Bmax[IB+1]+1] \
                             @ GG[IB+1, 0:NP, 0:NP] \
                             @ M[Bmin[IB]:Bmax[IB]+1, Bmin[IB+1]:Bmax[IB+1]+1].T.conj() \
                             @ glR[IB, 0:NI, 0:NI].T.conj() \
                             - (AG - AG.T.conj()) + (BG - BG.T.conj()) # 
        
        if IB > 0:  #Off-diagonal are only interesting for IdE! Can probably use these results also.
            NM = Bmax[IB-1] - Bmin[IB-1] + 1

            GRnn1[IB-1, 0:NI, 0:NP] = - glR[IB-1, 0:NM, 0:NM] \
                            @ M[Bmin[IB-1]:Bmax[IB-1]+1, Bmin[IB]:Bmax[IB]+1] \
                            @ GR[IB, 0:NI, 0:NI]

            GLnn1[IB-1, 0:NI, 0:NP] = - glR[IB-1, 0:NM, 0:NM] \
                            @ M[Bmin[IB-1]:Bmax[IB-1]+1, Bmin[IB]:Bmax[IB]+1] \
                            @ GL[IB, :NI, :NI] \
                            - glL[IB-1, 0:NM, 0:NM] \
                            @ M[Bmin[IB]:Bmax[IB]+1, Bmin[IB-1]:Bmax[IB-1]+1].T.conj() \
                            @ GR[IB, 0:NI, 0:NI].T.conj() \
                            + glR[IB-1, 0:NM, 0:NM] \
                            @ SigL[Bmin[IB-1]:Bmax[IB-1]+1, Bmin[IB]:Bmax[IB]+1] \
                            @ GR[IB, 0:NI, 0:NI].T.conj()   
                            
            GGnn1[IB-1, 0:NI, 0:NP] = - glR[IB-1, 0:NM, 0:NM] \
                            @ M[Bmin[IB-1]:Bmax[IB-1]+1, Bmin[IB]:Bmax[IB]+1] \
                            @ GG[IB, :NI, :NI] \
                            - glG[IB-1, 0:NM, 0:NM] \
                            @ M[Bmin[IB]:Bmax[IB]+1, Bmin[IB-1]:Bmax[IB-1]+1].T.conj() \
                            @ GR[IB, 0:NI, 0:NI].T.conj() \
                            + glR[IB-1, 0:NM, 0:NM] \
                            @ SigG[Bmin[IB-1]:Bmax[IB-1]+1, Bmin[IB]:Bmax[IB]+1] \
                            @ GR[IB, 0:NI, 0:NI].T.conj()

                                                                                                    
    return GR, GRnn1, GL, GLnn1, GG, GGnn1

def rgf_GF_2S_V1(M, SigL, SigG, Bmin, Bmax):
    '''
    2-sided rgf
    
    Working!

    Sequential.
    '''
    Bsize = max(Bmax - Bmin + 1) # Used for declaration of variables
    NB = len(Bmin)

    EI = int(NB/2) # Exchange index

    grR = np.zeros((NB-EI, Bsize, Bsize), dtype=np.cfloat) # Retarded (right)
    grL = np.zeros((NB-EI, Bsize, Bsize), dtype=np.cfloat) # Lesser (right)
    grG = np.zeros((NB-EI, Bsize, Bsize), dtype=np.cfloat) # Greater (right)
    
    glR = np.zeros((EI, Bsize, Bsize), dtype=np.cfloat) # Retarded (left)
    glL = np.zeros((EI, Bsize, Bsize), dtype=np.cfloat) # Lesser (left)
    glG = np.zeros((EI, Bsize, Bsize), dtype=np.cfloat) # Greater (left)
    
    GR = np.zeros((NB, Bsize, Bsize), dtype=np.cfloat) # Retarded GF
    GRnn1 = np.zeros((NB - 1, Bsize, Bsize), dtype=np.cfloat) #Off-diagonal GR
    GL = np.zeros((NB, Bsize, Bsize), dtype=np.cfloat) # Lesser GF
    GLnn1 = np.zeros((NB - 1, Bsize, Bsize), dtype=np.cfloat) # Off-diagonal GL
    GG = np.zeros((NB, Bsize, Bsize), dtype=np.cfloat) # Greater GF
    GGnn1 = np.zeros((NB - 1, Bsize, Bsize), dtype=np.cfloat) # Off-diagonal GG

    # Step 1:
    # Valid?
    step1_GF_rs(M[Bmin[EI]:, Bmin[EI]:], SigL[Bmin[EI]:,Bmin[EI]:], SigG[Bmin[EI]:,Bmin[EI]:], 
                grR, grL, grG, Bmin[EI:]-Bmin[EI], Bmax[EI:]-Bmin[EI])
    # Valid?
    step1_GF_ls(M[:Bmin[EI], :Bmin[EI]], SigL[:Bmin[EI], :Bmin[EI]], SigG[:Bmin[EI], :Bmin[EI]], 
                glR, glL, glG, Bmin[:EI], Bmax[:EI])
    
    # Exchange:

    NM = Bmax[EI-1] - Bmin[EI-1] + 1
    NI = Bmax[EI] - Bmin[EI] + 1
    NP = Bmax[EI+1] - Bmin[EI+1] + 1
    
    GR[EI, :NI, :NI] = np.linalg.inv(M[Bmin[EI]:Bmax[EI]+1, Bmin[EI]:Bmax[EI]+1]
                                   - M[Bmin[EI]:Bmax[EI]+1, Bmin[EI-1]:Bmax[EI-1]+1]
                                   @ glR[EI-1, :NM, :NM]
                                   @ M[Bmin[EI-1]:Bmax[EI-1]+1, Bmin[EI]:Bmax[EI]+1]
                                   - M[Bmin[EI]:Bmax[EI]+1, Bmin[EI+1]:Bmax[EI+1]+1]
                                   @ grR[1, :NP, :NP]
                                   @ M[Bmin[EI+1]:Bmax[EI+1]+1, Bmin[EI]:Bmax[EI]+1])
    
    GRnn1[EI, :NI, :NI] = - GR[EI, :NI, :NI] \
                          @ M[Bmin[EI]:Bmax[EI]+1, Bmin[EI+1]:Bmax[EI+1]+1] \
                          @ grR[1, :NP, :NP] # Could also use left 
    
    GRnn1[EI-1, :NI, :NI] = - glR[EI-1, :NM, :NM] \
                            @ M[Bmin[EI-1]:Bmax[EI-1]+1, Bmin[EI]:Bmax[EI]+1] \
                            @ GR[EI, :NI, :NI]
            
    CL = M[Bmin[EI]:Bmax[EI]+1, Bmin[EI-1]:Bmax[EI-1]+1] \
         @ glR[EI-1, :NM, :NM] \
         @ SigL[Bmin[EI-1]:Bmax[EI-1]+1, Bmin[EI]:Bmax[EI]+1] \
            
    DL = M[Bmin[EI]:Bmax[EI]+1, Bmin[EI+1]:Bmax[EI+1]+1] \
         @ grR[1, :NP, :NP] \
         @ SigL[Bmin[EI+1]:Bmax[EI+1]+1, Bmin[EI]:Bmax[EI]+1] \
    
    GL[EI, :NI, :NI] = GR[EI, :NI, :NI] @ (
                        M[Bmin[EI]:Bmax[EI]+1, Bmin[EI-1]:Bmax[EI-1]+1]
                        @ glL[EI-1, :NM, :NM]
                        @ M[Bmin[EI]:Bmax[EI]+1, Bmin[EI-1]:Bmax[EI-1]+1].T.conj()
                        + M[Bmin[EI]:Bmax[EI]+1, Bmin[EI+1]:Bmax[EI+1]+1]
                        @ grL[1, :NP, :NP]
                        @ M[Bmin[EI]:Bmax[EI]+1, Bmin[EI+1]:Bmax[EI+1]+1].T.conj()
                        + SigL[Bmin[EI]:Bmax[EI]+1, Bmin[EI]:Bmax[EI]+1]
                        - (CL-CL.T.conj())
                        - (DL-DL.T.conj())) @ GR[EI, :NI, :NI].T.conj()
    
    GLnn1[EI, :NI, :NI] = GR[EI, :NI, :NI] \
                        @ SigL[Bmin[EI]:Bmax[EI]+1, Bmin[EI+1]:Bmax[EI+1]+1] \
                        @ grR[1, :NP, :NP].T.conj() \
                        - GR[EI, :NI, :NI] \
                        @ M[Bmin[EI]:Bmax[EI]+1, Bmin[EI+1]:Bmax[EI+1]+1] \
                        @ grL[1, :NP, :NP] \
                        - GL[EI, :NI, :NI] \
                        @ M[Bmin[EI+1]:Bmax[EI+1]+1, Bmin[EI]:Bmax[EI]+1].T.conj() \
                        @ grR[1, :NP, :NP].T.conj() # Could also have used the left
                        
    GLnn1[EI-1, :NI, :NI] = - glR[EI-1, :NM, :NM] \
                        @ M[Bmin[EI-1]:Bmax[EI-1]+1, Bmin[EI]:Bmax[EI]+1] \
                        @ GL[EI, :NI, :NI] \
                        - glL[EI-1, :NM, :NM] \
                        @ M[Bmin[EI]:Bmax[EI]+1, Bmin[EI-1]:Bmax[EI-1]+1].T.conj() \
                        @ GR[EI, :NI, :NI].T.conj() \
                        + glR[EI-1, :NM, :NM] \
                        @ SigL[Bmin[EI-1]:Bmax[EI-1]+1, Bmin[EI]:Bmax[EI]+1] \
                        @ GR[EI, :NI, :NI].T.conj()


    CG = M[Bmin[EI]:Bmax[EI]+1, Bmin[EI-1]:Bmax[EI-1]+1] \
         @ glR[EI-1, :NM, :NM] \
         @ SigG[Bmin[EI-1]:Bmax[EI-1]+1, Bmin[EI]:Bmax[EI]+1]
            
    DG = M[Bmin[EI]:Bmax[EI]+1, Bmin[EI+1]:Bmax[EI+1]+1] \
         @ grR[1, :NP, :NP] \
         @ SigG[Bmin[EI+1]:Bmax[EI+1]+1, Bmin[EI]:Bmax[EI]+1]                   
                        
    GG[EI, :NI, :NI] = GR[EI, :NI, :NI] @ (
                        M[Bmin[EI]:Bmax[EI]+1, Bmin[EI-1]:Bmax[EI-1]+1]
                        @ glG[EI-1, :NM, :NM]
                        @ M[Bmin[EI]:Bmax[EI]+1, Bmin[EI-1]:Bmax[EI-1]+1].T.conj()
                        + M[Bmin[EI]:Bmax[EI]+1, Bmin[EI+1]:Bmax[EI+1]+1]
                        @ grG[1, :NP, :NP]
                        @ M[Bmin[EI]:Bmax[EI]+1, Bmin[EI+1]:Bmax[EI+1]+1].T.conj()
                        + SigG[Bmin[EI]:Bmax[EI]+1, Bmin[EI]:Bmax[EI]+1]
                        - (CG-CG.T.conj())
                        - (DG-DG.T.conj())) @ GR[EI, :NI, :NI].T.conj()
    
    GGnn1[EI, :NI, :NI] = GR[EI, :NI, :NI] \
                        @ SigG[Bmin[EI]:Bmax[EI]+1, Bmin[EI+1]:Bmax[EI+1]+1] \
                        @ grR[1, :NP, :NP].T.conj() \
                        - GR[EI, :NI, :NI] \
                        @ M[Bmin[EI]:Bmax[EI]+1, Bmin[EI+1]:Bmax[EI+1]+1] \
                        @ grG[1, :NP, :NP] \
                        - GG[EI, :NI, :NI] \
                        @ M[Bmin[EI+1]:Bmax[EI+1]+1, Bmin[EI]:Bmax[EI]+1].T.conj() \
                        @ grR[1, :NP, :NP].T.conj() # Could also have used the left      
                        
    GGnn1[EI-1, :NI, :NI] = - glR[EI-1, 0:NM, 0:NM] \
                        @ M[Bmin[EI-1]:Bmax[EI-1]+1, Bmin[EI]:Bmax[EI]+1] \
                        @ GG[EI, :NI, :NI] \
                        - glG[EI-1, 0:NM, 0:NM] \
                        @ M[Bmin[EI]:Bmax[EI]+1, Bmin[EI-1]:Bmax[EI-1]+1].T.conj() \
                        @ GR[EI, 0:NI, 0:NI].T.conj() \
                        + glR[EI-1, 0:NM, 0:NM] \
                        @ SigG[Bmin[EI-1]:Bmax[EI-1]+1, Bmin[EI]:Bmax[EI]+1] \
                        @ GR[EI, 0:NI, 0:NI].T.conj()

    # Step 2:

    step2_GF_rs(M[Bmin[EI]:, Bmin[EI]:], SigL[Bmin[EI]:, Bmin[EI]:], SigG[Bmin[EI]:, Bmin[EI]:], grR, grL, grG, 
                GR[EI:], GRnn1[EI:], GL[EI:], GLnn1[EI:], GG[EI:], GGnn1[EI:], Bmin[EI:]-Bmin[EI], Bmax[EI:]-Bmin[EI])
    
    step2_GF_ls(M[:Bmin[EI+1], :Bmin[EI+1]], SigL[:Bmin[EI+1], :Bmin[EI+1]], SigG[:Bmin[EI+1], :Bmin[EI+1]], glR, glL, glG, 
                GR[:EI+1], GRnn1[:EI+1], GL[:EI+1], GLnn1[:EI+1], GG[:EI+1], GGnn1[:EI+1], Bmin[:EI+1], Bmax[:EI+1])
    
    return GR, GRnn1, GL, GLnn1, GG, GGnn1

if __name__ == '__main__':
    
    """
    MPI implementation in this script.
    """


    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    n = 2 # Block size
    nBlocks = 10 # Blocks

    EI = int(nBlocks/2) # Exchange index

    if rank == 0:
        H, SigL, SigG = create_matrices(n,nBlocks)

        """
        "Open-boundary conditions"-like matrices
        """
        SBL = sparse.random(n,n,1,dtype=np.cfloat)
        SBL = (SBL+SBL.T.conj())/2
        SBR = sparse.random(n,n,1,dtype=np.cfloat)
        SBR = (SBR+SBR.T.conj())/2
        SB = sparse.block_diag((SBL, sparse.csc_matrix((n*(nBlocks-2),n*(nBlocks-2))), SBR))
        M=H+SB

        nt = n*nBlocks
        
        Bmin = np.array(range(0, nt, n), dtype=np.int32)   
        Bmax = np.array(range(n-1, nt, n), dtype=np.int32)

        if len(Bmax) != len(Bmin):
            raise Exception("specified number of block starts does not match with specified block ends!")
        
        if size == 1:

            GR, GRnn1, GL, GLnn1, GG, GGnn1 = rgf_GF_2S_V1(M,SigL,SigG,Bmin,Bmax)

            TGR, TGRnn1, TGL, TGLnn1, TGG, TGGnn1 = rgf_GF_2S_V0(M,SigL,SigG,Bmin,Bmax)    

        else:
            
            Bsize = max(Bmax - Bmin + 1) # Used for declaration of variables
            NB = len(Bmin)

            comm.Send(Bmin[:EI+1], dest=1, tag=1)
            comm.Send(Bmax[:EI+1], dest=1, tag=2)
            #MPI.Request.Waitall([req1, req2])
            
            # Sending sparse matrices need special threatment.
            M_part = M[:Bmin[EI+1], :Bmin[EI+1]]
            SigL_part = SigL[:Bmin[EI+1], :Bmin[EI+1]]
            SigG_part = SigG[:Bmin[EI+1], :Bmin[EI+1]]
            
            size_buffer_M = np.array([M_part.shape[0], M_part.shape[1], M_part.nnz], dtype=np.int32)
            size_buffer_SigL = np.array([SigL_part.shape[0], SigL_part.shape[1], SigL_part.nnz], dtype=np.int32)
            size_buffer_SigG = np.array([SigG_part.shape[0], SigG_part.shape[1], SigG_part.nnz], dtype=np.int32)
            
            comm.Send(size_buffer_M, dest=1, tag=3)
            comm.Send(M_part.indptr, dest=1, tag=4)
            comm.Send(M_part.indices, dest=1, tag=5)
            comm.Send(M_part.data, dest=1, tag=6)

            comm.Send(size_buffer_SigL, dest=1, tag=7)
            comm.Send(SigL_part.indptr, dest=1, tag=8)
            comm.Send(SigL_part.indices, dest=1, tag=9)
            comm.Send(SigL_part.data, dest=1, tag=10)

            comm.Send(size_buffer_SigG, dest=1, tag=11)
            comm.Send(SigG_part.indptr, dest=1, tag=12)
            comm.Send(SigG_part.indices, dest=1, tag=13)
            comm.Send(SigG_part.data, dest=1, tag=14)

            #req3 = comm.Isend(M[:Bmin[EI+1], :Bmin[EI+1]], dest=1, tag=3)
            #req4 = comm.Isend(SigL[:Bmin[EI+1], :Bmin[EI+1]], dest=1, tag=4)
            #req5 = comm.Isend(SigG[:Bmin[EI+1], :Bmin[EI+1]], dest=1, tag=5)
            #MPI.Request.Waitall([req1, req2, req3, req4, req5])

            grR = np.zeros((NB-EI, Bsize, Bsize), dtype=np.cfloat) # Retarded (right)
            grL = np.zeros((NB-EI, Bsize, Bsize), dtype=np.cfloat) # Lesser (right)
            grG = np.zeros((NB-EI, Bsize, Bsize), dtype=np.cfloat) # Greater (right)

            # Step 1:
            step1_GF_rs(M[Bmin[EI]:, Bmin[EI]:], SigL[Bmin[EI]:,Bmin[EI]:], SigG[Bmin[EI]:,Bmin[EI]:], 
                grR, grL, grG, Bmin[EI:]-Bmin[EI], Bmax[EI:]-Bmin[EI])
            
            # Exchange:

            glR_ex = np.zeros((Bmax[EI-1] - Bmin[EI-1] + 1, Bmax[EI-1] - Bmin[EI-1] + 1), 
                            dtype=np.cfloat)
            glL_ex = np.zeros((Bmax[EI-1] - Bmin[EI-1] + 1, Bmax[EI-1] - Bmin[EI-1] + 1), 
                            dtype=np.cfloat)
            glG_ex = np.zeros((Bmax[EI-1] - Bmin[EI-1] + 1, Bmax[EI-1] - Bmin[EI-1] + 1), 
                            dtype=np.cfloat)
            
            comm.Recv(glR_ex, source=1, tag=21)
            comm.Recv(glL_ex, source=1, tag=22)
            comm.Recv(glG_ex, source=1, tag=23)

            comm.Send(grR[0], dest=1, tag=24)
            comm.Send(grL[0], dest=1, tag=25)
            comm.Send(grG[0], dest=1, tag=26)
            #MPI.Request.Waitall([req21, req22, req23,
            #                    req24, req25, req26])

            GR = np.zeros((NB, Bsize, Bsize), dtype=np.cfloat) # Retarded GF
            GRnn1 = np.zeros((NB - 1, Bsize, Bsize), dtype=np.cfloat) #Off-diagonal GR
            GL = np.zeros((NB, Bsize, Bsize), dtype=np.cfloat) # Lesser GF
            GLnn1 = np.zeros((NB - 1, Bsize, Bsize), dtype=np.cfloat) # Off-diagonal GL
            GG = np.zeros((NB, Bsize, Bsize), dtype=np.cfloat) # Greater GF
            GGnn1 = np.zeros((NB - 1, Bsize, Bsize), dtype=np.cfloat) # Off-diagonal GG

            NM = Bmax[EI-1] - Bmin[EI-1] + 1
            NI = Bmax[EI] - Bmin[EI] + 1
            NP = Bmax[EI+1] - Bmin[EI+1] + 1
            
            GR[EI, :NI, :NI] = np.linalg.inv(M[Bmin[EI]:Bmax[EI]+1, Bmin[EI]:Bmax[EI]+1]
                                        - M[Bmin[EI]:Bmax[EI]+1, Bmin[EI-1]:Bmax[EI-1]+1]
                                        @ glR_ex
                                        @ M[Bmin[EI-1]:Bmax[EI-1]+1, Bmin[EI]:Bmax[EI]+1]
                                        - M[Bmin[EI]:Bmax[EI]+1, Bmin[EI+1]:Bmax[EI+1]+1]
                                        @ grR[1, :NP, :NP]
                                        @ M[Bmin[EI+1]:Bmax[EI+1]+1, Bmin[EI]:Bmax[EI]+1])
            
            GRnn1[EI, :NI, :NI] = - GR[EI, :NI, :NI] \
                                @ M[Bmin[EI]:Bmax[EI]+1, Bmin[EI+1]:Bmax[EI+1]+1] \
                                @ grR[1, :NP, :NP] # Could also use left 
                    
            CL = M[Bmin[EI]:Bmax[EI]+1, Bmin[EI-1]:Bmax[EI-1]+1] \
                @ glR_ex \
                @ SigL[Bmin[EI-1]:Bmax[EI-1]+1, Bmin[EI]:Bmax[EI]+1] \
                    
            DL = M[Bmin[EI]:Bmax[EI]+1, Bmin[EI+1]:Bmax[EI+1]+1] \
                @ grR[1, :NP, :NP] \
                @ SigL[Bmin[EI+1]:Bmax[EI+1]+1, Bmin[EI]:Bmax[EI]+1] \
            
            GL[EI, :NI, :NI] = GR[EI, :NI, :NI] @ (
                                M[Bmin[EI]:Bmax[EI]+1, Bmin[EI-1]:Bmax[EI-1]+1]
                                @ glL_ex
                                @ M[Bmin[EI]:Bmax[EI]+1, Bmin[EI-1]:Bmax[EI-1]+1].T.conj()
                                + M[Bmin[EI]:Bmax[EI]+1, Bmin[EI+1]:Bmax[EI+1]+1]
                                @ grL[1, :NP, :NP]
                                @ M[Bmin[EI]:Bmax[EI]+1, Bmin[EI+1]:Bmax[EI+1]+1].T.conj()
                                + SigL[Bmin[EI]:Bmax[EI]+1, Bmin[EI]:Bmax[EI]+1]
                                - (CL-CL.T.conj())
                                - (DL-DL.T.conj())) @ GR[EI, :NI, :NI].T.conj()
            
            GLnn1[EI, :NI, :NI] = GR[EI, :NI, :NI] \
                                @ SigL[Bmin[EI]:Bmax[EI]+1, Bmin[EI+1]:Bmax[EI+1]+1] \
                                @ grR[1, :NP, :NP].T.conj() \
                                - GR[EI, :NI, :NI] \
                                @ M[Bmin[EI]:Bmax[EI]+1, Bmin[EI+1]:Bmax[EI+1]+1] \
                                @ grL[1, :NP, :NP] \
                                - GL[EI, :NI, :NI] \
                                @ M[Bmin[EI+1]:Bmax[EI+1]+1, Bmin[EI]:Bmax[EI]+1].T.conj() \
                                @ grR[1, :NP, :NP].T.conj() # Could also have used the left


            CG = M[Bmin[EI]:Bmax[EI]+1, Bmin[EI-1]:Bmax[EI-1]+1] \
                @ glR_ex \
                @ SigG[Bmin[EI-1]:Bmax[EI-1]+1, Bmin[EI]:Bmax[EI]+1]
                    
            DG = M[Bmin[EI]:Bmax[EI]+1, Bmin[EI+1]:Bmax[EI+1]+1] \
                @ grR[1, :NP, :NP] \
                @ SigG[Bmin[EI+1]:Bmax[EI+1]+1, Bmin[EI]:Bmax[EI]+1]                   
                                
            GG[EI, :NI, :NI] = GR[EI, :NI, :NI] @ (
                                M[Bmin[EI]:Bmax[EI]+1, Bmin[EI-1]:Bmax[EI-1]+1]
                                @ glG_ex
                                @ M[Bmin[EI]:Bmax[EI]+1, Bmin[EI-1]:Bmax[EI-1]+1].T.conj()
                                + M[Bmin[EI]:Bmax[EI]+1, Bmin[EI+1]:Bmax[EI+1]+1]
                                @ grG[1, :NP, :NP]
                                @ M[Bmin[EI]:Bmax[EI]+1, Bmin[EI+1]:Bmax[EI+1]+1].T.conj()
                                + SigG[Bmin[EI]:Bmax[EI]+1, Bmin[EI]:Bmax[EI]+1]
                                - (CG-CG.T.conj())
                                - (DG-DG.T.conj())) @ GR[EI, :NI, :NI].T.conj()
            
            GGnn1[EI, :NI, :NI] = GR[EI, :NI, :NI] \
                                @ SigG[Bmin[EI]:Bmax[EI]+1, Bmin[EI+1]:Bmax[EI+1]+1] \
                                @ grR[1, :NP, :NP].T.conj() \
                                - GR[EI, :NI, :NI] \
                                @ M[Bmin[EI]:Bmax[EI]+1, Bmin[EI+1]:Bmax[EI+1]+1] \
                                @ grG[1, :NP, :NP] \
                                - GG[EI, :NI, :NI] \
                                @ M[Bmin[EI+1]:Bmax[EI+1]+1, Bmin[EI]:Bmax[EI]+1].T.conj() \
                                @ grR[1, :NP, :NP].T.conj() # Could also have used the left  

            # Step 2
            
            step2_GF_rs(M[Bmin[EI]:, Bmin[EI]:], SigL[Bmin[EI]:, Bmin[EI]:], SigG[Bmin[EI]:, Bmin[EI]:], grR, grL, grG, 
                        GR[EI:], GRnn1[EI:], GL[EI:], GLnn1[EI:], GG[EI:], GGnn1[EI:], Bmin[EI:]-Bmin[EI], Bmax[EI:]-Bmin[EI])
            
            # Retreave all results

            comm.Recv(GR[:EI], source=1, tag=31)
            comm.Recv(GRnn1[:EI], source=1, tag=32)
            comm.Recv(GL[:EI], source=1, tag=33)
            comm.Recv(GLnn1[:EI], source=1, tag=34)
            comm.Recv(GG[:EI], source=1, tag=35)
            comm.Recv(GGnn1[:EI], source=1, tag=36)
            #MPI.Request.Waitall([req31, req32, req33, req34, req35, req36])

        # Checking results

        GR_full = np.linalg.inv(M.toarray())
        GL_full = GR_full @ SigL @ GR_full.T.conj()
        GG_full = GR_full @ SigG @ GR_full.T.conj()

        #IT = int(nBlocks/2)-1
        #print(GR[IT])
        #print(GR_full[Bmin[IT]:Bmax[IT]+1, Bmin[IT]:Bmax[IT]+1])

        for IB in range(len(Bmin)-1,-1,-1):
            assert np.allclose(GR[IB], GR_full[Bmin[IB]:Bmax[IB]+1, Bmin[IB]:Bmax[IB]+1])
            assert np.allclose(GL[IB], GL_full[Bmin[IB]:Bmax[IB]+1, Bmin[IB]:Bmax[IB]+1])
            assert np.allclose(GG[IB], GG_full[Bmin[IB]:Bmax[IB]+1, Bmin[IB]:Bmax[IB]+1])
            if IB < nBlocks-1:
                assert np.allclose(GRnn1[IB], GR_full[Bmin[IB]:Bmax[IB]+1, Bmin[IB+1]:Bmax[IB+1]+1])
                assert np.allclose(GLnn1[IB], GL_full[Bmin[IB]:Bmax[IB]+1, Bmin[IB+1]:Bmax[IB+1]+1])
                assert np.allclose(GGnn1[IB], GG_full[Bmin[IB]:Bmax[IB]+1, Bmin[IB+1]:Bmax[IB+1]+1])
        print("Implementation is correct.")
    

    if rank == 1:

        Bmin = np.zeros(EI+1, dtype=np.int32)
        Bmax = np.zeros(EI+1, dtype=np.int32)

        comm.Recv(Bmin, source=0, tag=1)
        comm.Recv(Bmax, source=0, tag=2)
        #MPI.Request.Waitall([req1, req2])

        Bsize = max(Bmax - Bmin + 1) # Used for declaration of variables
        NB = len(Bmin)-1 # Here, NB = EI and not NB = nBlocks!
        
        # Recieve part of sparse matrix. For-loop?
        size_buffer_M = np.empty(3, dtype=np.int32)
        size_buffer_SigL = np.empty(3, dtype=np.int32)
        size_buffer_SigG = np.empty(3, dtype=np.int32)
        
        comm.Recv(size_buffer_M, source=0, tag=3)
        lNI, lNK, lNNZ = size_buffer_M
        indptr = np.empty(lNI + 1, dtype=np.int32)
        indices = np.empty(lNNZ, dtype=np.int32)
        data = np.empty(lNNZ, dtype=np.cfloat)
        comm.Recv(indptr, source=0, tag=4)
        comm.Recv(indices, source=0, tag=5)
        comm.Recv(data, source=0, tag=6)
        M_part = sparse.csr_matrix((data, indices, indptr), shape=(lNI, lNK), dtype=np.cfloat)
        
        comm.Recv(size_buffer_SigL, source=0, tag=7)
        lNI, lNK, lNNZ = size_buffer_SigL
        indptr = np.empty(lNI + 1, dtype=np.int32)
        indices = np.empty(lNNZ, dtype=np.int32)
        data = np.empty(lNNZ, dtype=np.cfloat)
        comm.Recv(indptr, source=0, tag=8)
        comm.Recv(indices, source=0, tag=9)
        comm.Recv(data, source=0, tag=10)
        SigL_part = sparse.csr_matrix((data, indices, indptr), shape=(lNI, lNK), dtype=np.cfloat)

        comm.Recv(size_buffer_SigG, source=0, tag=11)
        lNI, lNK, lNNZ = size_buffer_SigG
        indptr = np.empty(lNI + 1, dtype=np.int32)
        indices = np.empty(lNNZ, dtype=np.int32)
        data = np.empty(lNNZ, dtype=np.cfloat)
        comm.Recv(indptr, source=0, tag=12)
        comm.Recv(indices, source=0, tag=13)
        comm.Recv(data, source=0, tag=14)
        SigG_part = sparse.csr_matrix((data, indices, indptr), shape=(lNI, lNK), dtype=np.cfloat)
        
        glR = np.zeros((NB, Bsize, Bsize), dtype=np.cfloat) # Retarded (left)
        glL = np.zeros((NB, Bsize, Bsize), dtype=np.cfloat) # Lesser (left)
        glG = np.zeros((NB, Bsize, Bsize), dtype=np.cfloat) # Greater (left)

        step1_GF_ls(M_part[:Bmin[EI], :Bmin[EI]], SigL_part[:Bmin[EI], :Bmin[EI]], 
                    SigG_part[:Bmin[EI], :Bmin[EI]], glR, glL, glG, Bmin[:EI], Bmax[:EI]) # EI or EI+1??

        # Exchange:
        
        grR_ex = np.zeros((Bmax[-1] - Bmin[-1] + 1, Bmax[-1] - Bmin[-1] + 1), 
                         dtype=np.cfloat)
        grL_ex = np.zeros((Bmax[-1] - Bmin[-1] + 1, Bmax[-1] - Bmin[-1] + 1), 
                         dtype=np.cfloat)
        grG_ex = np.zeros((Bmax[-1] - Bmin[-1] + 1, Bmax[-1] - Bmin[-1] + 1), 
                         dtype=np.cfloat)

        comm.Send(glR[-1], dest=0, tag=21)
        comm.Send(glL[-1], dest=0, tag=22)
        comm.Send(glG[-1], dest=0, tag=23)

        comm.Recv(grR_ex, source=0, tag=24)
        comm.Recv(grL_ex, source=0, tag=25)
        comm.Recv(grG_ex, source=0, tag=26)
        #MPI.Request.Waitall([req21, req22, req23,
        #                     req24, req25, req26])

        GR = np.zeros((NB, Bsize, Bsize), dtype=np.cfloat) # Retarded GF
        GRnn1 = np.zeros((NB, Bsize, Bsize), dtype=np.cfloat) #Off-diagonal GR
        GL = np.zeros((NB, Bsize, Bsize), dtype=np.cfloat) # Lesser GF
        GLnn1 = np.zeros((NB, Bsize, Bsize), dtype=np.cfloat) # Off-diagonal GL
        GG = np.zeros((NB, Bsize, Bsize), dtype=np.cfloat) # Greater GF
        GGnn1 = np.zeros((NB, Bsize, Bsize), dtype=np.cfloat) # Off-diagonal GG

        NM = Bmax[NB-2] - Bmin[NB-2] + 1
        NI = Bmax[NB-1] - Bmin[NB-1] + 1
        NP = Bmax[NB] - Bmin[NB] + 1

        GR[EI-1, :NI, :NI] = np.linalg.inv(M_part[Bmin[EI-1]:Bmax[EI-1]+1, Bmin[EI-1]:Bmax[EI-1]+1]
                                    - M_part[Bmin[EI-1]:Bmax[EI-1]+1, Bmin[EI-2]:Bmax[EI-2]+1]
                                    @ glR[EI-2, :NM, :NM]
                                    @ M_part[Bmin[EI-2]:Bmax[EI-2]+1, Bmin[EI-1]:Bmax[EI-1]+1]
                                    - M_part[Bmin[EI-1]:Bmax[EI-1]+1, Bmin[EI]:Bmax[EI]+1]
                                    @ grR_ex
                                    @ M_part[Bmin[EI]:Bmax[EI]+1, Bmin[EI-1]:Bmax[EI-1]+1])
        
        GRnn1[EI-1, :NI, :NI] = - GR[EI-1, :NI, :NI] \
                            @ M_part[Bmin[EI-1]:Bmax[EI-1]+1, Bmin[EI]:Bmax[EI]+1] \
                            @ grR_ex # Could also use left. Unsure about this
                
        CL = M_part[Bmin[EI-1]:Bmax[EI-1]+1, Bmin[EI-2]:Bmax[EI-2]+1] \
            @ glR[EI-2, :NM, :NM] \
            @ SigL_part[Bmin[EI-2]:Bmax[EI-2]+1, Bmin[EI-1]:Bmax[EI-1]+1] \
                
        DL = M_part[Bmin[EI-1]:Bmax[EI-1]+1, Bmin[EI]:Bmax[EI]+1] \
            @ grR_ex \
            @ SigL_part[Bmin[EI]:Bmax[EI]+1, Bmin[EI-1]:Bmax[EI-1]+1] \
        
        GL[EI-1, :NI, :NI] = GR[EI-1, :NI, :NI] @ (
                            M_part[Bmin[EI-1]:Bmax[EI-1]+1, Bmin[EI-2]:Bmax[EI-2]+1]
                            @ glL[EI-2, :NM, :NM]
                            @ M_part[Bmin[EI-1]:Bmax[EI-1]+1, Bmin[EI-2]:Bmax[EI-2]+1].T.conj()
                            + M_part[Bmin[EI-1]:Bmax[EI-1]+1, Bmin[EI]:Bmax[EI]+1]
                            @ grL_ex
                            @ M_part[Bmin[EI-1]:Bmax[EI-1]+1, Bmin[EI]:Bmax[EI]+1].T.conj()
                            + SigL_part[Bmin[EI-1]:Bmax[EI-1]+1, Bmin[EI-1]:Bmax[EI-1]+1]
                            - (CL-CL.T.conj())
                            - (DL-DL.T.conj())) @ GR[EI-1, :NI, :NI].T.conj()
        
        GLnn1[EI-1, :NI, :NI] = GR[EI-1, :NI, :NI] \
                            @ SigL_part[Bmin[EI-1]:Bmax[EI-1]+1, Bmin[EI]:Bmax[EI]+1] \
                            @ grR_ex.T.conj() \
                            - GR[EI-1, :NI, :NI] \
                            @ M_part[Bmin[EI-1]:Bmax[EI-1]+1, Bmin[EI]:Bmax[EI]+1] \
                            @ grL_ex \
                            - GL[EI-1, :NI, :NI] \
                            @ M_part[Bmin[EI]:Bmax[EI]+1, Bmin[EI-1]:Bmax[EI-1]+1].T.conj() \
                            @ grR_ex.T.conj() # Could also have used the left. Unsure about this


        CG = M_part[Bmin[EI-1]:Bmax[EI-1]+1, Bmin[EI-2]:Bmax[EI-2]+1] \
            @ glR[EI-2, :NM, :NM] \
            @ SigG_part[Bmin[EI-2]:Bmax[EI-2]+1, Bmin[EI-1]:Bmax[EI-1]+1]
                
        DG = M_part[Bmin[EI-1]:Bmax[EI-1]+1, Bmin[EI]:Bmax[EI]+1] \
            @ grR_ex \
            @ SigG_part[Bmin[EI]:Bmax[EI]+1, Bmin[EI-1]:Bmax[EI-1]+1]                   
                            
        GG[EI-1, :NI, :NI] = GR[EI-1, :NI, :NI] @ (
                            M_part[Bmin[EI-1]:Bmax[EI-1]+1, Bmin[EI-2]:Bmax[EI-2]+1]
                            @ glG[EI-2, :NM, :NM]
                            @ M_part[Bmin[EI-1]:Bmax[EI-1]+1, Bmin[EI-2]:Bmax[EI-2]+1].T.conj()
                            + M_part[Bmin[EI-1]:Bmax[EI-1]+1, Bmin[EI]:Bmax[EI]+1]
                            @ grG_ex
                            @ M_part[Bmin[EI-1]:Bmax[EI-1]+1, Bmin[EI]:Bmax[EI]+1].T.conj()
                            + SigG_part[Bmin[EI-1]:Bmax[EI-1]+1, Bmin[EI-1]:Bmax[EI-1]+1]
                            - (CG-CG.T.conj())
                            - (DG-DG.T.conj())) @ GR[EI-1, :NI, :NI].T.conj()
        
        GGnn1[EI-1, :NI, :NI] = GR[EI-1, :NI, :NI] \
                            @ SigG_part[Bmin[EI-1]:Bmax[EI-1]+1, Bmin[EI]:Bmax[EI]+1] \
                            @ grR_ex.T.conj() \
                            - GR[EI-1, :NI, :NI] \
                            @ M_part[Bmin[EI-1]:Bmax[EI-1]+1, Bmin[EI]:Bmax[EI]+1] \
                            @ grG_ex \
                            - GG[EI-1, :NI, :NI] \
                            @ M_part[Bmin[EI]:Bmax[EI]+1, Bmin[EI-1]:Bmax[EI-1]+1].T.conj() \
                            @ grR_ex.T.conj() # Could also have used the left
        
        step2_GF_ls(M_part[:Bmin[EI], :Bmin[EI]], SigL_part[:Bmin[EI], :Bmin[EI]], SigG_part[:Bmin[EI], :Bmin[EI]], glR, glL, glG, 
                    GR[:EI], GRnn1[:EI], GL[:EI], GLnn1[:EI], GG[:EI], GGnn1[:EI], Bmin[:EI], Bmax[:EI])
        
        # Sending all results to master node (rank 0)

        comm.Send(GR[:EI], dest=0, tag=31)
        comm.Send(GRnn1[:EI], dest=0, tag=32)
        comm.Send(GL[:EI], dest=0, tag=33)
        comm.Send(GLnn1[:EI], dest=0, tag=34)
        comm.Send(GG[:EI], dest=0, tag=35)
        comm.Send(GGnn1[:EI], dest=0, tag=36)
        #MPI.Request.Waitall([req21, req22, req23, req24, req25, req26])
