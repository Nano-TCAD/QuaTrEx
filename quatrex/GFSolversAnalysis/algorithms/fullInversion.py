import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import inv
import time
from mpi4py import MPI


def fullInversion(A, label=""):
    """
        Invert a matrix using:
            numpy dense matrix: numpy.linalg.inv
            scipy CSC matrix: scipy.sparse.linalg.inv
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if type(A) == np.ndarray:
        tic = time.perf_counter()
        A_inv = np.linalg.inv(A)
        toc = time.perf_counter()

        if rank == 0:
            print(f"Numpy: {label} Full inversion took {toc - tic:0.4f} seconds")
        
        return A_inv

    elif type(A) == csc_matrix:
        tic = time.perf_counter()
        A_inv = inv(A)
        toc = time.perf_counter()

        if rank == 0:
            print(f"Scipy CSC: {label} Full inversion took {toc - tic:0.4f} seconds")
        
        return A_inv