from matplotlib import pyplot as plt

def vizualiseDenseMatrixFlat(mat):
    """
        Visualise a dense matrix in a 2D plot. Third dimension is represented by color.
    """
    plt.imshow(mat, cmap='hot', interpolation='nearest')
    plt.show()


def vizualiseCSRMatrixFlat(mat):
    """
        Visualise a CSR matrix in a 2D plot. Third dimension is represented by color.
    """
    plt.imshow(mat.toarray(), cmap='hot', interpolation='nearest')
    plt.show()