import numpy as np
from numpy.testing import assert_allclose


def projection_matrix_1d(b):
    """Compute the projection matrix onto the space spanned by `b`
    Args:
        b: ndarray of dimension (D,), the basis for the subspace

    Returns:
        p: the projection matrix
    """
    (D,) = b.shape
    p = np.outer(b, b)
    len_b_arr = np.square(b)
    len_b = np.sum(len_b_arr)
    p = np.true_divide(p, len_b)
    return p


def project_1d(x, b):
    """Compute the projection matrix onto the space spanned by `b`
    Args:
        x: the vector to be projected
        b: ndarray of dimension (D,), the basis for the subspace

    Returns:
        y: ndarray of shape (D,) projection of x in space spanned by b
    """
    p = np.zeros((3,))
    len_b_arr = np.square(b)
    len_b = np.sum(len_b_arr)
    p_matrix = x.T @ b
    multiplied = p_matrix * b
    p = np.true_divide(multiplied, len_b)
    return p


def projection_matrix_general(B):
    """Compute the projection matrix onto the space spanned by the columns of `B`
    Args:
        B: ndarray of dimension (D, M), the basis for the subspace

    Returns:
        P: the projection matrix
    """
    p = B @ np.linalg.inv(B.T @ B) @ B.T
    return p


def project_general(x, B):
    """Compute the projection matrix onto the space spanned by the columns of `B`
    Args:
        x: ndarray of dimension (D, 1), the vector to be projected
        B: ndarray of dimension (D, M), the basis for the subspace

    Returns:
        p: projection of x onto the subspac spanned by the columns of B; size (D, 1)
    """
    p_matrix = np.linalg.inv(B.T @ B) @ B.T @ x
    p = 0
    for a in range(B.shape[1]):
        scalar = p_matrix[a]
        proj = B[:, a] * scalar
        p += proj
    return p.reshape(-1, 1)


# Testing projection_matrix_1d function
assert_allclose(
    projection_matrix_1d(np.array([1, 2, 2])),
    np.array([[1, 2, 2], [2, 4, 4], [2, 4, 4]]) / 9,
)

# Testing project_1d function
assert_allclose(project_1d(np.ones(3), np.array([1, 2, 2])), np.array([5, 10, 10]) / 9)

# Testing projection_matrix_general function
B = np.array([[1, 0], [1, 1], [1, 2]])

assert_allclose(
    projection_matrix_general(B), np.array([[5, 2, -1], [2, 2, 2], [-1, 2, 5]]) / 6
)

# Testing project_general function
assert_allclose(
    project_general(np.array([6, 0, 0]).reshape(-1, 1), B),
    np.array([5, 2, -1]).reshape(-1, 1),
)
