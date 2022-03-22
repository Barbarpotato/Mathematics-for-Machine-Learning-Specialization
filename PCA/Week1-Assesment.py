import numpy as np

# this is the example matrix to calculate
A = np.array([[3, 4], [9, 6]])


def mean(matrix):
    """returns the means of some matrix"""
    row, column, sum, mean = (
        matrix.shape[0],
        matrix.shape[1],
        0,
        np.zeros((matrix.shape[0],)),
    )
    for a in range(row):
        for b in range(column):
            sum += matrix[b, a]
        mean[a], sum = sum / matrix.shape[0], 0
    return mean


def cov_matrix(matrix):
    """min variable is used to hold the substraction from matrix and the means value"""
    N, D = matrix.shape
    cov, min, mean_matrix = np.zeros((D, D)), np.zeros((N, D)), mean(matrix)
    for i in range(N):
        min[i] = matrix[i, :] - mean_matrix
    for i in range(D):
        for j in range(D):
            cov[i, j] = cov[i, j] + min[:, i] @ min[:, j]
    return cov / N
