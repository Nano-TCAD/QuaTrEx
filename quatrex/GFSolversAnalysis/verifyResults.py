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


def verifResultsDiag(A_ref, A):
    """
        Compare the diagonal of two matrices.
    """
    
    if np.allclose(np.diag(A_ref), np.diag(A)):
        return True
    else:
        return False



def extractDiagonalBandedElements(A, bandwidth):
    """
        Extract the elements of the diagonal and the bandwidth of a matrix and store
        them into a new matrix of the same type than the one parsed.
    """

    if type(A) == np.ndarray:
        A_band = np.zeros(A.shape)
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                if i - j > bandwidth or j - i > bandwidth:
                    A_band[i, j] = 0
                else:
                    A_band[i, j] = A[i, j]

                """ if i - j <= bandwidth and j - i <= bandwidth:
                    A_band[i, j] = A[i, j] """

        return A_band
    
    elif type(A) == csc_matrix:
        data : np.ndarray = np.array([])
        row : np.ndarray = np.array([])
        col : np.ndarray = np.array([])

        A_coo = A.tocoo()
        for d, r, c in zip(A_coo.data, A_coo.row, A_coo.col):
            if r - c <= bandwidth and c - r <= bandwidth:
                data = np.append(data, d)
                row = np.append(row, r)
                col = np.append(col, c)

        return csc_matrix((data, (row, col)), shape=A.shape)
