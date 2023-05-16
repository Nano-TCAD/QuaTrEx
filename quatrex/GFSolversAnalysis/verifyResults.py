import numpy as np
from scipy.sparse import csc_matrix

def verifResults(A_ref, A):
    """
        Compare the results of two matrices. Either dense or sparse.
    """

    if type(A_ref) == np.ndarray:
        if type(A) == np.ndarray:
            if np.allclose(A_ref, A):
                return True
            else:
                return False

        elif type(A) == csc_matrix:
            if np.allclose(A_ref, A.toarray()):
                return True
            else:
                return False

    if type(A_ref) == csc_matrix:
        if type(A) == np.ndarray:
            if np.allclose(A_ref.toarray(), A):
                return True
            else:
                return False
        
        elif type(A) == csc_matrix:
            if np.allclose(A_ref.data, A.data) and A_ref.indices == A.indices and A_ref.indptr == A.indptr:
                return True
            else:
                return False


def verifResultsBlocksTri(A_ref_diag, A_ref_upper, A_ref_lower, A_diag, A_upper, A_lower):
    """
        Check the correctness of the inverting of the block tridiagonal matrix.
    """

    if np.allclose(A_ref_diag, A_diag) and np.allclose(A_ref_upper, A_upper) and np.allclose(A_ref_lower, A_lower):
        return True
    else:
        return False

