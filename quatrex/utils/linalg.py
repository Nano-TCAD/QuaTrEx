"""
Todo refactore
"""

import numpy as np
import copy, logging, time
import scipy.signal
import scipy.fft

def invert(M:np.ndarray, method='solve'):
    if method=='solve':
        assert M.ndim>=2
        assert M.shape[-2]==M.shape[-1] # last two indices are square matrix
        n = M.shape[-1]
        s = tuple(list(M.shape[:-2])+[1,1]) # copy the identity matrix to this shape
        return np.linalg.solve(M, np.tile(np.eye(n, dtype=complex), s))
    elif method=='inv':
        return np.linalg.inv(M)
    else:
        raise ValueError("method not found!")

def sancho(
    E:float, H00:np.ndarray, H01:np.ndarray,
    n_orb:int, eta=1e-3, itermax=100, tol=1e-50
    ):
    '''
    Sancho-Rubio algorithm for the surface green function of a semi-infinite chain of 
        layers. At step n it will effectively get 2^n layers.
    python version of subroutine surfgreen_1985() in
        https://github.com/quanshengwu/wannier_tools/blob/master/src/surfgreen.f90
    [Ref] J.Phys.F.Met.Phys.15(1985)851-858

    Args
        E : float
            energy
        H00 : np.ndarray
            Hamiltonian of one layer
            shape = (n_orb, n_orb)
            H00 is Hermitian
        H01 : np.ndarray
            Hamiltonian between two neighboring layers
            shape = (n_orb, n_orb)
        n_orb : int
            number of total (spin-)orbitals in a layer
        eta : float
            imaginary part of energy to avoid singularity
        itermax : int
            max iteration steps
        tol : float
            tolerance of the norm of alpha

    Returns
        GLL : np.ndarray 
            left surface Green function
            shape = (n_orb, n_orb)
        GBB : np.ndarray 
            bulk Green function
            shape = (n_orb, n_orb)
        GRR : np.ndarray 
            dual surface Green function
            shape = (n_orb, n_orb)
    '''

    assert H00.shape==H01.shape 
    assert H00.shape==(n_orb, n_orb)

    # eq.(9), deepcopy for safety
    eps_l = copy.deepcopy(H00) # left surface, or \epsilon^s_i in the paper
    eps_r = copy.deepcopy(H00) # dual surface, or \tilde{\epsilon}^s_i in the paper
    eps_b = copy.deepcopy(H00) # bulk, or \epsilon_i in the paper
    alpha = copy.deepcopy(H01) # effective hopping
    beta = copy.deepcopy(np.conjugate(np.transpose(H01)))

    # avoid singularity in (EI-H)^(-1)
    E_c = E + 1.0j*eta
    Ed = np.eye(n_orb, dtype=complex)*E_c

    for i in range(itermax):
        # eq.(11)
        g0 = invert(Ed - eps_b)
        g_a = np.matmul(g0, alpha)
        g_b = np.matmul(g0, beta)
        a_g_b = np.matmul(alpha, g_b)
        b_g_a = np.matmul(beta, g_a)
        eps_l = eps_l + a_g_b
        eps_r = eps_r + b_g_a # eq.(17)
        eps_b = eps_b + a_g_b + b_g_a
        alpha = np.matmul(alpha, g_a)
        beta = np.matmul(beta, g_b)

        # the iteration is to be repeated until alpha and beta vanishes
        if np.linalg.norm(alpha)+np.linalg.norm(beta) < tol:
            break
    
    if i==itermax-1:
        logging.warning('E = {:.3f}'.format(E))
        np.savez_compressed(
            '/home/msc22h9/NEGF/test/methods/sanchoerr.npz', 
            E=E, H00=H00, H01=H01, n_orb=n_orb, eta=eta, itermax=itermax, tol=tol
        )
        raise ValueError("sancho reached max iteration!")
    
    # eq.(14) left surface green function
    GLL = invert(Ed-eps_l)

    # eq.(15) bulk green function
    GBB = invert(Ed-eps_b)

    # eq.(16) dual surface green function
    GRR = invert(Ed-eps_r)

    return GLL, GBB, GRR

def surface_inversion(
    M00:np.ndarray, M01:np.ndarray, M10:np.ndarray,
    itermax=100, tol=1e-16
    ):
    '''
    upper left block of the infinite tridiagonal matrix M^{-1}
    similar to surface_function() in https://gitlab.ethz.ch/dleonard/NEGF_GW/-/blob/main/GW_Python.py
        but definition of M01 and M10 are different

        M00 M01
        M10 M00 M01
    M =     M10 M00 M01
                M10 M00 ...
                    ... ...

    Args
        M00, M01, M10 : np.ndarray
            square matrix blocks
        itermax : int
            max iteration steps
        tol : float
            tolerance of the norm of alpha
    
    Returns
        G00 : np.ndarray
            upper left block of the full infinite matrix G=M^{-1}
    '''
    assert len(M00.shape)==2
    assert M00.shape[0]==M00.shape[1]
    assert M00.shape==M01.shape
    assert M00.shape==M10.shape

    alpha = copy.deepcopy(M01)
    beta = copy.deepcopy(M10)
    eps_s = copy.deepcopy(M00) # factor for G(0, 0)
    eps_t = copy.deepcopy(M00) # factor for G(2^n, 0)
    i = 0
    for i in range(itermax):
        g = invert(eps_t)
        g_a = np.matmul(g, alpha)
        g_b = np.matmul(g, beta)
        a_g_b = np.matmul(alpha, g_b)
        b_g_a = np.matmul(beta, g_a)
        eps_s -= a_g_b
        eps_t -= (a_g_b + b_g_a)
        alpha = np.matmul(alpha, g_a)
        beta = np.matmul(beta, g_b)
        # the iteration is to be repeated until alpha vanishes
        cond = np.linalg.norm(alpha)
        if cond < tol:
            break
        
    logging.debug("surface_inversion total step = {0:d}".format(i))

    G00 = invert(eps_s)
    return G00

def convolve_3D_einsum(a:np.ndarray, b:np.ndarray, mode='valid', b_index='ij'):
    assert len(a.shape)==3
    assert len(b.shape)==3
    assert a.shape[-2]==a.shape[-1]
    assert a.shape[-2:]==b.shape[-2:]
    assert mode in ('full', 'valid')
    assert b_index in ('ij', 'ji')

    # make sure a is longer equal than b
    if a.shape[0]<b.shape[0]:
        a, b = b, a
        if b_index=='ij':
            path = 'eij,eij->ij'
        elif b_index=='ji':
            path = 'eji,eij->ij'
        else:
            raise ValueError("b_index not found")
        inverted = True
    else:
        if b_index=='ij':
            path = 'eij,eij->ij'
        elif b_index=='ji':
            path = 'eij,eji->ij'
        else:
            raise ValueError("b_index not found")
        inverted = False
    
    n_a = a.shape[0]
    n_b = b.shape[0]
    n_orb = a.shape[1]

    if mode=='full':
        n_left = n_b-1
        n_right = n_b-1
        n_c = n_a+n_b-1
    elif mode=='valid':
        n_left = 0
        n_right = 0
        n_c = n_a-n_b+1
    else:
        raise ValueError('mode not found')
    c = np.zeros((n_c, n_orb, n_orb), dtype=complex)

    b = np.flip(b, axis=0)
    i_c = 0
    # some part of b is on the left of a
    for i in range(n_left):
        c[i_c] = np.einsum(path, a[:i+1], b[-(i+1):], optimize='optimal')
        i_c += 1
    # b is inside a
    for i in range(n_a-n_b+1):
        c[i_c] = np.einsum(path, a[i:i+n_b], b, optimize='optimal')
        i_c += 1
    # some part of b is on the right of a
    for i in range(n_right):
        c[i_c] = np.einsum(path, a[-(n_b-1-i):], b[:n_b-1-i], optimize='optimal')
        i_c += 1
    assert i_c==n_c
    b = np.flip(b, axis=0)

    if inverted is True:
        a, b = b, a
    
    return c

def convolve_3D_loop(a:np.ndarray, b:np.ndarray, mode='valid', b_index='ij'):
    assert len(a.shape)==3
    assert len(b.shape)==3
    assert a.shape[-2]==a.shape[-1]
    assert a.shape[-2:]==b.shape[-2:]
    assert mode in ('full', 'valid')
    assert b_index in ('ij', 'ji')


    n_a = a.shape[0]
    n_b = b.shape[0]
    n_orb = a.shape[1]

    if mode=='full':
        n_c = n_a+n_b-1
    elif mode=='valid':
        n_c = max(n_a, n_b)-min(n_a, n_b)+1
    else:
        raise ValueError('mode not found')
    c = np.zeros((n_c, n_orb, n_orb), dtype=complex)
    
    if b_index=='ij':
        for i in range(n_orb):
            for j in range(n_orb):
                # c[:,i,j]=scipy.signal.convolve(a[:,i,j],b[:,i,j],mode=mode,method='auto')
                c[:,i,j] = np.convolve(a[:,i,j], b[:,i,j], mode=mode)
    elif b_index=='ji':
        for i in range(n_orb):
            for j in range(n_orb):
                # c[:,i,j]=scipy.signal.convolve(a[:,i,j],b[:,i,j],mode=mode,method='auto')
                c[:,i,j] = np.convolve(a[:,i,j], b[:,j,i], mode=mode)
    else:
        raise ValueError('b_index not found')

    return c

def convolve_3D_fft(a:np.ndarray, b:np.ndarray, mode='valid', b_index='ij', n_worker=16): 
    assert a.ndim==3
    assert b.ndim==3
    assert a.shape[-2]==a.shape[-1]
    assert a.shape[-2:]==b.shape[-2:]
    assert mode in ('full', 'valid')
    assert b_index in ('ij', 'ji')
    assert isinstance(n_worker, int) and n_worker>0

    if b_index=='ji':
        b = np.einsum('ijk->ikj', b)

    with scipy.fft.set_workers(n_worker):
        c = scipy.signal.fftconvolve(a, b, mode=mode, axes=0)

    if b_index=='ji':
        b = np.einsum('ijk->ikj', b)

    return c

def convolve_3D(
    a:np.ndarray, b:np.ndarray, mode='valid', b_index='ij', method='fft', n_worker=32
    ):
    '''
    c[n] = \sum_{m} g(a[n-m], b[m]) = \sum_{m} g(a[m], b[n-m])
    g(a, b) = a*b

    Args
        a, b : np.ndarray
            array of matrices to corrlate
            shape = (n_a/n_b, n_orb, n_orb)
        mode : str
            https://numpy.org/doc/stable/reference/generated/numpy.convolve.html
            c.shape[0] =
                'full'  (n_a+n_b-1)
                'valid' (max(n_a,n_b) - min(n_a,n_b) + 1)
        b_index : str
            whether b is transposed
            'ij' : b        fij=aij*bij
            'ji' : b.T      fij=aij*bji
        method : str
            'loop' uses np.convolve (not fft) and loop over all orbitals
            'einsum' uses np.einsum for each energy point
            'fft' uses scipy.signal.convolve
        n_worker : int
            number of workers used in scipy.fft
    
    Returns
        c : np.ndarray
            convolved matrices
            shape = (n_a+n_b-1, n_orb, n_orb)
    '''
    assert method in ('einsum', 'loop', 'fft')
    if method=='einsum':
        return convolve_3D_einsum(a, b, mode, b_index)
    elif method=='loop':
        return convolve_3D_loop(a, b, mode, b_index)
    elif method=='fft':
        return convolve_3D_fft(a, b, mode, b_index, n_worker)
    else:
        raise ValueError('method not found!')

def correlate_3D_einsum(a:np.ndarray, b:np.ndarray, mode='full', b_index='ji'):
    assert len(a.shape)==3
    assert len(b.shape)==3
    assert a.shape[-2]==a.shape[-1]
    assert a.shape[-2:]==b.shape[-2:]
    assert mode in ('full', 'valid')
    assert b_index in ('ij', 'ji')

    # make sure a is longer equal than b
    if a.shape[0]<b.shape[0]:
        a, b = b, a
        if b_index=='ij':
            path = 'eij,eij->ij'
        elif b_index=='ji':
            path = 'eji,eij->ij'
        else:
            raise ValueError("b_index not found")
        inverted = True
    else:
        if b_index=='ij':
            path = 'eij,eij->ij'
        elif b_index=='ji':
            path = 'eij,eji->ij'
        else:
            raise ValueError("b_index not found")
        inverted = False
    
    n_a = a.shape[0]
    n_b = b.shape[0]
    n_orb = a.shape[1]

    if mode=='full':
        n_left = n_b-1
        n_right = n_b-1
        n_c = n_a+n_b-1
    elif mode=='valid':
        n_left = 0
        n_right = 0
        n_c = n_a-n_b+1
    else:
        raise ValueError('mode not found')
    c = np.zeros((n_c, n_orb, n_orb), dtype=complex)

    i_c = 0
    # some part of b is on the left of a
    for i in range(n_left):
        c[i_c] = np.einsum(path, a[:i+1], b[-(i+1):], optimize='optimal')
        i_c += 1
    # b is inside a
    for i in range(n_a-n_b+1):
        c[i_c] = np.einsum(path, a[i:i+n_b], b, optimize='optimal')
        i_c += 1
    # some part of b is on the right of a
    for i in range(n_right):
        c[i_c] = np.einsum(path, a[-(n_b-1-i):], b[:n_b-1-i], optimize='optimal')
        i_c += 1
    assert i_c==n_c

    if inverted is True:
        a, b = b, a
        c = np.flip(c, axis=0)
    
    return c

def correlate_3D_loop(a:np.ndarray, b:np.ndarray, mode='full', b_index='ji'):
    assert len(a.shape)==3
    assert len(b.shape)==3
    assert a.shape[-2]==a.shape[-1]
    assert a.shape[-2:]==b.shape[-2:]
    assert mode in ('full', 'valid')
    assert b_index in ('ij', 'ji')

    n_a = a.shape[0]
    n_b = b.shape[0]
    n_orb = a.shape[1]

    if mode=='full':
        n_c = n_a+n_b-1
    elif mode=='valid':
        n_c = max(n_a, n_b)-min(n_a, n_b)+1
    else:
        raise ValueError('mode not found')
    c = np.zeros((n_c, n_orb, n_orb), dtype=complex)

    b = np.conj(b) # numpy and scipy correlation is a*conj(b)
    if b_index=='ij':
        for i in range(n_orb):
            for j in range(n_orb): 
                # c[:,i,j]=scipy.signal.correlate(a[:,i,j],b[:,i,j],mode=mode,method='auto')
                c[:,i,j] = np.correlate(a[:,i,j], b[:,i,j], mode=mode)
    elif b_index=='ji':
        for i in range(n_orb):
            for j in range(n_orb):
                # c[:,i,j]=scipy.signal.correlate(a[:,i,j],b[:,j,i],mode=mode,method='auto')
                c[:,i,j] = np.correlate(a[:,i,j], b[:,j,i], mode=mode)
    else:
        raise ValueError('b_index not found')
    b = np.conj(b)

    return c

def correlate_3D_fft(a:np.ndarray, b:np.ndarray, mode='full', b_index='ji', n_worker=16):
    b = np.flip(b, axis=0)
    c = convolve_3D_fft(a, b, mode, b_index, n_worker)
    b = np.flip(b, axis=0)
    return c

def correlate_3D(
    a:np.ndarray, b:np.ndarray, mode='full', b_index='ji', method='fft', n_worker=32
    ):
    '''
    c[n] = \sum_{m} f(a[m], b[m-n])

    Args
        a, b : np.ndarray
            array of matrices to corrlate
            shape = (n_a/n_b, n_orb, n_orb)
        mode : str
            https://numpy.org/doc/stable/reference/generated/numpy.convolve.html
            c.shape[0] =
                'full'  (n_a+n_b-1)
                'valid' (max(n_a,n_b) - min(n_a,n_b) + 1)
        b_index : str
            whether b is transposed
            'ij' : b        fij=aij*bij
            'ji' : b.T      fij=aij*bji
        method : str
            'loop' uses np.convolve (not fft) and loop over all orbitals
            'einsum' uses np.einsum for each energy point
            'fft' uses scipy.signal.convolve
        n_worker : int
            number of workers used in scipy.fft

    Returns
        c : np.ndarray
            corrlated matrices
            shape = (n_a+n_b-1, n_orb, n_orb)
    '''
    assert method in ('einsum', 'loop', 'fft')
    if method=='einsum':
        return correlate_3D_einsum(a, b, mode, b_index)
    elif method=='loop':
        return correlate_3D_loop(a, b, mode, b_index)
    elif method=='fft':
        return correlate_3D_fft(a, b, mode, b_index, n_worker)
    else:
        raise ValueError('method not found!')

if __name__ == "__main__":
    n_a = 1000
    n_b = 1000
    n_orb = 1000
    mode='full'
    b_index='ji'
    print(__file__)

    a = np.random.rand(n_a, n_orb, n_orb)+np.random.rand(n_a, n_orb, n_orb)*1.0j
    b = np.random.rand(n_b, n_orb, n_orb)+np.random.rand(n_b, n_orb, n_orb)*1.0j
    test_a = copy.deepcopy(a)
    test_b = copy.deepcopy(b)
    assert np.allclose(a, test_a) and np.allclose(b, test_b)
    
    # correlate
    tic = time.perf_counter()
    e = correlate_3D(a, b, mode, b_index, 'fft')
    print("correlate_3D_fft {:.3f}".format(time.perf_counter()-tic))
    assert np.allclose(a, test_a) and np.allclose(b, test_b)

    # tic = time.perf_counter()
    # d = correlate_3D(a, b, mode, b_index, 'loop')
    # print("correlate_3D_loop {:.3f}".format(time.perf_counter()-tic))
    # assert np.allclose(a, test_a) and np.allclose(b, test_b)
    # assert np.allclose(e, d)

    # tic = time.perf_counter()
    # c = correlate_3D(a, b, mode, b_index, 'einsum')
    # print("correlate_3D_einsum {:.3f}".format(time.perf_counter()-tic))
    # assert np.allclose(a, test_a) and np.allclose(b, test_b)
    # assert np.allclose(c, d)

    # convolve
    tic = time.perf_counter()
    e = convolve_3D(a, b, mode, b_index, 'fft')
    print("convolve_3D_fft {:.3f}".format(time.perf_counter()-tic))
    assert np.allclose(a, test_a) and np.allclose(b, test_b)

    # tic = time.perf_counter()
    # d = convolve_3D(a, b, mode, b_index, 'loop')
    # print("convolve_3D_loop {:.3f}".format(time.perf_counter()-tic))
    # assert np.allclose(a, test_a) and np.allclose(b, test_b)
    # assert np.allclose(e, d)

    # tic = time.perf_counter()
    # c = convolve_3D(a, b, mode, b_index, 'einsum')
    # print("convolve_3D_einsum {:.3f}".format(time.perf_counter()-tic))
    # assert np.allclose(a, test_a) and np.allclose(b, test_b)
    # assert np.allclose(c, d)

