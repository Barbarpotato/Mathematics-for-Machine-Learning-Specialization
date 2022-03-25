import math
import numpy as np


def distance(x0, x1):
    """Compute distance between two vectors x0, x1 using the dot product
    for a bigger problem, like a tons of array, we can use np.dot"""
    """distance = np.dot(x1-x0,x1-x0)**(0.5)
       return distance"""
    distance = (x0 - x1).T @ (x0 - x1)
    return float(distance ** (1 / 2))


def angle(x0, x1):
    """Compute the angle between two vectors x0, x1 using the dot product
    for a bigger problem, like a tons of array we can use np.dot"""
    """angle = np.arccos((np.dot(x0,x1)/(np.dot(x0,x0)*np.dot(x1,x1))**(0.5)))
       return angle"""
    angle = (x0).T @ x1 / round(((x0 @ x0) ** (1 / 2)) * ((x1 @ x1) ** (1 / 2)), 3)
    return math.acos(angle)


def most_similiar_images(distances):
    """Search the most similiar image at first image,
    which need distances as a argument.distances is a list
    that contains distances between each images."""
    similiar = 0
    index = 0
    for idx, val in enumerate(distances[1:499]):
        if idx == 0:
            similiar = val
        if val <= similiar:
            similiar = val
            index = idx + 1
    return index


def pairwise_distance_matrix(X, Y):
    """Compute the pairwise distance between rows of X and rows of Y

    Arguments
    ----------
    X: ndarray of size (N, D)
    Y: ndarray of size (M, D)

    Returns
    --------
    distance_matrix: matrix of shape (N, M), each entry distance_matrix[i,j] is the distance between
    ith row of X and the jth row of Y (we use the dot product to compute the distance).
    """
    N, D = X.shape
    M, _ = Y.shape
    distance_matrix = np.zeros(
        (N, M)
    )  # <-- EDIT THIS to compute the correct distance matrix
    for i in range(N):
        for j in range(M):
            distance_matrix[i][j] = np.linalg.norm(X[i] - Y[j])
    return distance_matrix
