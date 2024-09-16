import numpy as np
from scipy import sparse


def inv_E(A_E):
    #option 1
    return np.linalg.inv(A_E)

    #option 2
    #s = tuple(list(A_E.shape[:-2])+[1,1]) # copy the identity matrix to this shape
    #return np.linalg.solve(A_E, np.tile(np.eye(A_E.shape[-1], dtype=complex), s))

    #option 3
    #B = np.zeros(A_E.shape, dtype=np.complex128)
    #for Ept, A in enumerate(A_E):
    #    B[Ept] = np.linalg.inv(A)

    #option4
    B = np.zeros(A_E.shape, dtype=np.complex128)
    for Ept, A in enumerate(A_E):
        B[Ept] = np.linalg.solve(A, np.eye(len(A), dtype=complex))

    return B


def read_E(A, b_starts, b_length, nt, nEpts, b_hight=None):

    if not b_hight:
        b_hight = b_length
    B = [np.ndarray((nEpts, b_length, b_hight), dtype=np.complex128) for i in range(len(b_starts))]
    for Ept, block in enumerate(A):
        for i, b_start in enumerate(b_starts):
            s_r = b_start[0]
            s_c = b_start[1]
            e_r = s_r + b_length
            e_c = s_c + b_hight
            if e_r == 0:
                e_r = nt + 1
            if e_c == 0:
                e_c = nt + 1

            B[i][Ept] = A[Ept][s_r:e_r, s_c:e_c].toarray()
    return B
