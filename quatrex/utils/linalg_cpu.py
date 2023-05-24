"""
Few line helper functions used in g2p and gw2sigma
Most compiled with numba to run parallel on the cpu
"""
import numpy as np
import numpy.typing as npt
import typing
import numba
from numpy import fft
import scipy

@numba.njit("(c16[:,:],)",
            parallel=True,
            cache=True,
            nogil=True,
            error_model="numpy")
def reversal(g1: npt.NDArray[np.complex128]) -> npt.NDArray[np.complex128]:
    """Reverses data in time
        Single core alternative (Can not be compiled with numba):
        out = np.roll(np.flip(g1, axis=1), 1, axis=1)

    Args:
        g1 (npt.NDArray[np.complex128]): 2D array: orbital * energy

    Returns:
        npt.NDArray[np.complex128]: np.roll(
            np.flip(g1, axis=1), 1, axis=1)[ij2ji, :]
    """
    out: npt.NDArray[np.complex128] = np.empty_like(g1)
    no:  np.int32                   = np.shape(g1)[0]
    ne:  np.int32                   = np.shape(g1)[1]
    for i in numba.prange(no):
        for j in range(ne):
            out[i, j] = g1[i, -j]
    return out

@numba.njit("(c16[:,:], i4[:])",
            parallel=True,
            cache=True,
            nogil=True,
            error_model="numpy")
def reversal_transpose(g1: npt.NDArray[np.complex128], ij2ji: npt.NDArray[np.int32]) -> npt.NDArray[np.complex128]:
    """Reverses data in time and transposes in orbital space
        Scaling: Halves with 4 numba threads

        Single core alternative (Can not be compiled with numba):
        out = np.roll(np.flip(g1, axis=1), 1, axis=1)[ij2ji, :]

    Args:
        g1 (npt.NDArray[np.complex128]): 2D array: orbital * energy
        ij2ji (npt.NDArray[np.int32]): Mapping to transposed

    Returns:
        npt.NDArray[np.complex128]: np.roll(
            np.flip(g1, axis=1), 1, axis=1)[ij2ji, :]
    """
    out: npt.NDArray[np.complex128] = np.empty_like(g1)
    tmp: npt.NDArray[np.complex128] = np.empty_like(g1)
    no:  np.int32                   = np.shape(g1)[0]
    ne:  np.int32                   = np.shape(g1)[1]
    for i in numba.prange(no):
        for j in range(ne):
            tmp[i, j] = g1[i, -j]
    for i in numba.prange(no):
        out[ij2ji[i], :] = tmp[i, :]
    return out

@numba.njit("(c16[:,:], c16)",
            parallel=True,
            cache=True,
            nogil=True,
            error_model="numpy")
def scalarmul(g1: npt.NDArray[np.complex128],
              fac: complex) -> npt.NDArray[np.complex128]:
    """Multiplies elementwise g1 with fac
        Scales multicore with numba, but not ideal scaling
        Did not parallelize with openBLAS 
        Alternative multicore:
        out = np.empty_like(g1)
        for i in numba.prange(g1.shape[0]):
            for j in numba.prange(g1.shape[1]):
                out[i, j] = g1[i, j] * fac
    Args:
        g1 (npt.NDArray[np.complex128]): 2D matrix
        fac (np.complex128): complex double

    Returns:
        npt.NDArray[np.complex128]: fac*g1
    """
    out = np.multiply(g1, fac)
    return out

@numba.njit("(c16[:,:], c16[:,:])",
            parallel=True,
            cache=True,
            nogil=True,
            error_model="numpy")
def elementmul(g1: npt.NDArray[np.complex128],
               g2: npt.NDArray[np.complex128]) -> npt.NDArray[np.complex128]:
    """Multiplies elementwise g1 and g2, both with same size
        Scales multicore with numba, but not ideal scaling
        Did not parallelize with openBLAS 
        Alternative multicore:
        out = np.empty_like(g1)
        for i in numba.prange(g1.shape[0]):
            for j in numba.prange(g1.shape[1]):
                out[i, j] = g1[i, j] * g2[i, j]
    Args:
        g1 (npt.NDArray[np.complex128]): 2D Matrix
        g2 (npt.NDArray[np.complex128]): 2D Matrix

    Returns:
        npt.NDArray[np.complex128]: g1*g2,
    """
    out: npt.NDArray[np.complex128] = np.multiply(g1, g2)
    return out

@numba.njit("(c16[:,:], i8, i8)",
            parallel=True,
            cache=True,
            nogil=True,
            error_model="numpy")
def fft_numba(g1: npt.NDArray[np.complex128], ne2: np.int64, no: np.int64) -> npt.NDArray[np.complex128]:
    """Tries to compile scipy fft with numba

    Args:
        g1 (npt.NDArray[np.complex128]): 2D matrix
        ne (np.int32): number energy points
        workers (np.int32): #workers for ifft

    Returns:
        npt.NDArray[np.complex128]: fft(g1) with padding ne, ne2 = 2*ne
    """
    out: npt.NDArray[np.complex128] = np.empty((no,ne2), dtype=np.complex128)
    for i in numba.prange(no):
        out[i,:] = fft.fft(g1[i,:], n=ne2)
    return out

@numba.njit("(c16[:,:], c16, i8, i8)",
            parallel=True,
            cache=True,
            nogil=True,
            error_model="numpy")
def scalarmul_ifft_cutoff(g1: npt.NDArray[np.complex128], fac: np.complex128, ne: np.int64, no: np.int64) -> npt.NDArray[np.complex128]:
    """IFFT, crops, and multiplies elementwise g1 with fac
        Needs RocketFFT! https://github.com/styfenschaer/rocket-fft
        Numba scalarmul and fft separately takes more or less the same time
        Alternative multicore:
        g1 = fft.ifft(g1, axis=1, workers=workers)[:, :ne]
        out = np.multiply(g1, fac)
    Args:
        g1 (npt.NDArray[np.complex128]): 2D matrix
        fac (np.complex128): complex double
        ne (np.int32): #energy points
        workers(np.int32): #workers for ifft

    Returns:
        npt.NDArray[np.complex128]: fac*ifft(g1)[:,:ne]
    """
    out = np.empty((no,ne), dtype=np.complex128)
    for i in numba.prange(no):
        out[i,:] = fft.ifft(g1[i,:])[:ne]
    for i in numba.prange(no):
        for j in numba.prange(ne):
            out[i, j] = out[i, j] * fac
    return out

@numba.njit("(c16[:,:], c16, i8, i8)",
            parallel=True,
            cache=True,
            nogil=True,
            error_model="numpy")
def scalarmul_ifft(g1: npt.NDArray[np.complex128], fac: np.complex128, ne: np.int64, no: np.int64) -> npt.NDArray[np.complex128]:
    """IFFT, crops, and multiplies elementwise g1 with fac
        Needs RocketFFT! https://github.com/styfenschaer/rocket-fft
        Numba scalarmul and fft separately takes more or less the same time
        Alternative multicore:
        g1 = fft.ifft(g1, axis=1, workers=workers)
        out = np.multiply(g1, fac)
    Args:
        g1 (npt.NDArray[np.complex128]): 2D matrix
        fac (np.complex128): complex double
        ne (np.int32): #energy points
        workers(np.int32): #workers for ifft

    Returns:
        npt.NDArray[np.complex128]: fac*ifft(g1)[:,:ne]
    """
    out = np.empty((no,2*ne), dtype=np.complex128)
    for i in numba.prange(no):
        out[i,:] = fft.ifft(g1[i,:])
    for i in numba.prange(no):
        for j in numba.prange(2*ne):
            out[i, j] = out[i, j] * fac
    return out

@numba.njit("(i4, i4, c16[:,:], c16[:,:])",
            parallel=True,
            cache=True,
            nogil=True,
            error_model="numpy")
def elementmult_chunk(start: np.int32, end: np.int32, g1: npt.NDArray[np.complex128], g2: npt.NDArray[np.complex128]) -> npt.NDArray[np.complex128]:
    """Function to elementwise multiply chunks of g1 and g2
        Todo not tested if njit + parallel true works

    Args:
        g1 (npt.NDArray[np.complex128]): First array to multiply
        g2 (npt.NDArray[np.complex128]): Second array to multiply

    Returns:
        npt.NDArray[np.complex128]: g1[chunk,:] * g2[chunk,:]
    """
    out: npt.NDArray[np.complex128] = np.multiply(g1[start: end, :], g2[start: end, :])
    return out

def correlate_3D_einsum(a:np.ndarray, b:np.ndarray, mode="full", b_index="ji"):
    assert len(a.shape)==3
    assert len(b.shape)==3
    assert a.shape[-2]==a.shape[-1]
    assert a.shape[-2:]==b.shape[-2:]
    assert mode in ("full", "valid")
    assert b_index in ("ij", "ji")

    # make sure a is longer equal than b
    if a.shape[0]<b.shape[0]:
        a, b = b, a
        if b_index=="ij":
            path = "eij,eij->ij"
        elif b_index=="ji":
            path = "eji,eij->ij"
        else:
            raise ValueError("b_index not found")
        inverted = True
    else:
        if b_index=="ij":
            path = "eij,eij->ij"
        elif b_index=="ji":
            path = "eij,eji->ij"
        else:
            raise ValueError("b_index not found")
        inverted = False
    
    n_a = a.shape[0]
    n_b = b.shape[0]
    n_orb = a.shape[1]

    if mode=="full":
        n_left = n_b-1
        n_right = n_b-1
        n_c = n_a+n_b-1
    elif mode=="valid":
        n_left = 0
        n_right = 0
        n_c = n_a-n_b+1
    else:
        raise ValueError("mode not found")
    c = np.zeros((n_c, n_orb, n_orb), dtype=complex)

    i_c = 0
    # some part of b is on the left of a
    for i in range(n_left):
        c[i_c] = np.einsum(path, a[:i+1], b[-(i+1):], optimize="optimal")
        i_c += 1
    # b is inside a
    for i in range(n_a-n_b+1):
        c[i_c] = np.einsum(path, a[i:i+n_b], b, optimize="optimal")
        i_c += 1
    # some part of b is on the right of a
    for i in range(n_right):
        c[i_c] = np.einsum(path, a[-(n_b-1-i):], b[:n_b-1-i], optimize="optimal")
        i_c += 1
    assert i_c==n_c

    if inverted is True:
        a, b = b, a
        c = np.flip(c, axis=0)
    
    return c

def correlate_3D_loop(a:np.ndarray, b:np.ndarray, mode="full", b_index="ji"):
    assert len(a.shape)==3
    assert len(b.shape)==3
    assert a.shape[-2]==a.shape[-1]
    assert a.shape[-2:]==b.shape[-2:]
    assert mode in ("full", "valid")
    assert b_index in ("ij", "ji")

    n_a = a.shape[0]
    n_b = b.shape[0]
    n_orb = a.shape[1]

    if mode=="full":
        n_c = n_a+n_b-1
    elif mode=="valid":
        n_c = max(n_a, n_b)-min(n_a, n_b)+1
    else:
        raise ValueError("mode not found")
    c = np.zeros((n_c, n_orb, n_orb), dtype=complex)

    b = np.conj(b) # numpy and scipy correlation is a*conj(b)
    if b_index=="ij":
        for i in range(n_orb):
            for j in range(n_orb): 
                # c[:,i,j]=scipy.signal.correlate(a[:,i,j],b[:,i,j],mode=mode,method="auto")
                c[:,i,j] = np.correlate(a[:,i,j], b[:,i,j], mode=mode)
    elif b_index=="ji":
        for i in range(n_orb):
            for j in range(n_orb):
                # c[:,i,j]=scipy.signal.correlate(a[:,i,j],b[:,j,i],mode=mode,method="auto")
                c[:,i,j] = np.correlate(a[:,i,j], b[:,j,i], mode=mode)
    else:
        raise ValueError("b_index not found")
    b = np.conj(b)

    return c

def correlate_3D_fft(a:np.ndarray, b:np.ndarray, mode="full", b_index="ji", n_worker=16):
    b = np.flip(b, axis=0)
    c = convolve_3D_fft(a, b, mode, b_index, n_worker)
    b = np.flip(b, axis=0)
    return c

def correlate_3D(
    a:np.ndarray, b:np.ndarray, mode="full", b_index="ji", method="fft", n_worker=32
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
                "full"  (n_a+n_b-1)
                "valid" (max(n_a,n_b) - min(n_a,n_b) + 1)
        b_index : str
            whether b is transposed
            "ij" : b        fij=aij*bij
            "ji" : b.T      fij=aij*bji
        method : str
            "loop" uses np.convolve (not fft) and loop over all orbitals
            "einsum" uses np.einsum for each energy point
            "fft" uses scipy.signal.convolve
        n_worker : int
            number of workers used in scipy.fft

    Returns
        c : np.ndarray
            corrlated matrices
            shape = (n_a+n_b-1, n_orb, n_orb)
    '''
    assert method in ("einsum", "loop", "fft")
    if method=="einsum":
        return correlate_3D_einsum(a, b, mode, b_index)
    elif method=="loop":
        return correlate_3D_loop(a, b, mode, b_index)
    elif method=="fft":
        return correlate_3D_fft(a, b, mode, b_index, n_worker)
    else:
        raise ValueError("method not found!")

def convolve_3D_einsum(a:np.ndarray, b:np.ndarray, mode="valid", b_index="ij"):
    assert len(a.shape)==3
    assert len(b.shape)==3
    assert a.shape[-2]==a.shape[-1]
    assert a.shape[-2:]==b.shape[-2:]
    assert mode in ("full", "valid")
    assert b_index in ("ij", "ji")

    # make sure a is longer equal than b
    if a.shape[0]<b.shape[0]:
        a, b = b, a
        if b_index=="ij":
            path = "eij,eij->ij"
        elif b_index=="ji":
            path = "eji,eij->ij"
        else:
            raise ValueError("b_index not found")
        inverted = True
    else:
        if b_index=="ij":
            path = "eij,eij->ij"
        elif b_index=="ji":
            path = "eij,eji->ij"
        else:
            raise ValueError("b_index not found")
        inverted = False
    
    n_a = a.shape[0]
    n_b = b.shape[0]
    n_orb = a.shape[1]

    if mode=="full":
        n_left = n_b-1
        n_right = n_b-1
        n_c = n_a+n_b-1
    elif mode=="valid":
        n_left = 0
        n_right = 0
        n_c = n_a-n_b+1
    else:
        raise ValueError("mode not found")
    c = np.zeros((n_c, n_orb, n_orb), dtype=complex)

    b = np.flip(b, axis=0)
    i_c = 0
    # some part of b is on the left of a
    for i in range(n_left):
        c[i_c] = np.einsum(path, a[:i+1], b[-(i+1):], optimize="optimal")
        i_c += 1
    # b is inside a
    for i in range(n_a-n_b+1):
        c[i_c] = np.einsum(path, a[i:i+n_b], b, optimize="optimal")
        i_c += 1
    # some part of b is on the right of a
    for i in range(n_right):
        c[i_c] = np.einsum(path, a[-(n_b-1-i):], b[:n_b-1-i], optimize="optimal")
        i_c += 1
    assert i_c==n_c
    b = np.flip(b, axis=0)

    if inverted is True:
        a, b = b, a

    return c

def convolve_3D_loop(a:np.ndarray, b:np.ndarray, mode="valid", b_index="ij"):
    assert len(a.shape)==3
    assert len(b.shape)==3
    assert a.shape[-2]==a.shape[-1]
    assert a.shape[-2:]==b.shape[-2:]
    assert mode in ("full", "valid")
    assert b_index in ("ij", "ji")


    n_a = a.shape[0]
    n_b = b.shape[0]
    n_orb = a.shape[1]

    if mode=="full":
        n_c = n_a+n_b-1
    elif mode=="valid":
        n_c = max(n_a, n_b)-min(n_a, n_b)+1
    else:
        raise ValueError("mode not found")
    c = np.zeros((n_c, n_orb, n_orb), dtype=complex)
    
    if b_index=="ij":
        for i in range(n_orb):
            for j in range(n_orb):
                # c[:,i,j]=scipy.signal.convolve(a[:,i,j],b[:,i,j],mode=mode,method="auto")
                c[:,i,j] = np.convolve(a[:,i,j], b[:,i,j], mode=mode)
    elif b_index=="ji":
        for i in range(n_orb):
            for j in range(n_orb):
                # c[:,i,j]=scipy.signal.convolve(a[:,i,j],b[:,i,j],mode=mode,method="auto")
                c[:,i,j] = np.convolve(a[:,i,j], b[:,j,i], mode=mode)
    else:
        raise ValueError("b_index not found")

    return c

def convolve_3D_fft(a:np.ndarray, b:np.ndarray, mode="valid", b_index="ij", n_worker=16): 
    assert a.ndim==3
    assert b.ndim==3
    assert a.shape[-2]==a.shape[-1]
    assert a.shape[-2:]==b.shape[-2:]
    assert mode in ("full", "valid")
    assert b_index in ("ij", "ji")
    assert isinstance(n_worker, int) and n_worker>0

    if b_index=="ji":
        b = np.einsum("ijk->ikj", b)

    with scipy.fft.set_workers(n_worker):
        c = scipy.signal.fftconvolve(a, b, mode=mode, axes=0)

    if b_index=="ji":
        b = np.einsum("ijk->ikj", b)

    return c

def convolve_3D(
    a:np.ndarray, b:np.ndarray, mode="valid", b_index="ij", method="fft", n_worker=32
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
                "full"  (n_a+n_b-1)
                "valid" (max(n_a,n_b) - min(n_a,n_b) + 1)
        b_index : str
            whether b is transposed
            "ij" : b        fij=aij*bij
            "ji" : b.T      fij=aij*bji
        method : str
            "loop" uses np.convolve (not fft) and loop over all orbitals
            "einsum" uses np.einsum for each energy point
            "fft" uses scipy.signal.convolve
        n_worker : int
            number of workers used in scipy.fft
    
    Returns
        c : np.ndarray
            convolved matrices
            shape = (n_a+n_b-1, n_orb, n_orb)
    '''
    assert method in ("einsum", "loop", "fft")
    if method=="einsum":
        return convolve_3D_einsum(a, b, mode, b_index)
    elif method=="loop":
        return convolve_3D_loop(a, b, mode, b_index)
    elif method=="fft":
        return convolve_3D_fft(a, b, mode, b_index, n_worker)
    else:
        raise ValueError("method not found!")
