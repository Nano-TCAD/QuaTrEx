import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import inv
import time

def fullInversion(A):
    """
        Invert a matrix using:
            numpy dense matrix: numpy.linalg.inv
            scipy CSC matrix: scipy.sparse.linalg.inv
    """

    if type(A) == np.ndarray:
        tic = time.perf_counter()
        A_inv = np.linalg.inv(A)
        toc = time.perf_counter()

        print(f"Numpy: Full inversion took {toc - tic:0.4f} seconds")
        
        return A_inv

    elif type(A) == csc_matrix:
        tic = time.perf_counter()
        A_inv = inv(A)
        toc = time.perf_counter()

        print(f"Scipy CSC: Full inversion took {toc - tic:0.4f} seconds")
        
        return A_inv
    
    

def rgf(A, bandwidth):
    """
        Block-diagonal selected inversion using RGF algorithm.
    """

    



