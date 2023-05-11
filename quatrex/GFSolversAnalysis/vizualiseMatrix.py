from matplotlib import pyplot as plt

def vizualiseDenseMatrixFlat(mat, legend=""):
    """
        Visualise a dense matrix in a 2D plot. Third dimension is represented by color.
    """
    plt.title('Dense matrix hotmat: ' + legend)
    plt.imshow(abs(mat), cmap='hot', interpolation='nearest')
    plt.show()


def vizualiseCSCMatrixFlat(mat, legend=""):
    """
        Visualise a CSC matrix in a 2D plot. Third dimension is represented by color.
    """
    # add a title and axis labels
    plt.title('CSC matrix hotmat: '+ legend)
    plt.imshow(abs(mat.toarray()), cmap='hot', interpolation='nearest')
    plt.show()