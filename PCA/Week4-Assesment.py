import numpy as np


def normalize(X):
    """Normalize the given dataset X
    Args:
        X: ndarray, dataset

    Returns:
        (Xbar, mean, std): tuple of ndarray, Xbar is the normalized dataset
        with mean 0 and standard deviation 1; mean and std are the
        mean and standard deviation respectively.
    """
    mu = X.mean(0)
    Xbar = X - mu
    return Xbar, mu


def eig(S):
    """Compute the eigenvalues and corresponding eigenvectors
        for the covariance matrix S.
    Args:
        S: ndarray, covariance matrix

    Returns:
        (eigvals, eigvecs): ndarray, the eigenvalues and eigenvectors
    """
    eigvals, eigvecs = np.linalg.eig(S)
    sort_indices = np.argsort(eigvals)[::-1]
    return eigvals[sort_indices], eigvecs[:, sort_indices]


def projection_matrix(B):
    """Compute the projection matrix onto the space spanned by `B`
    Args:
        B: ndarray of dimension (D, M), the basis for the subspace

    Returns:
        P: the projection matrix
    """
    return B @ (np.linalg.inv(B.T @ B)) @ B.T


def PCA(X, num_components):
    """
    Args:
        X: ndarray of size (N, D), where D is the dimension of the data,
           and N is the number of datapoints
        num_components: the number of principal components to use.
    Returns:
        X_reconstruct: ndarray of the reconstruction
        of X from the first `num_components` principal components.
    """
    N, D = X.shape
    X_normalized, mean = normalize(X)
    S = np.cov(X_normalized / N, rowvar=False, bias=True)
    eig_vals, eig_vecs = eig(S)
    principal_vals = eig_vals[:num_components]
    principal_components = eig_vecs[:, :num_components]
    P = projection_matrix(principal_components)  # projection matrix
    reconst = (P @ X_normalized.T).T + mean
    return reconst.astype(float), mean, principal_vals, principal_components


def PCA_high_dim(X, num_components):
    """Compute PCA for small sample size but high-dimensional features.
    Args:
        X: ndarray of size (N, D), where D is the dimension of the sample,
           and N is the number of samples
        num_components: the number of principal components to use.
    Returns:
        X_reconstruct: (N, D) ndarray. the reconstruction
        of X from the first `num_components` pricipal components.
    """
    N, D = X.shape
    X_normalized, mean = normalize(X)
    M = np.dot(X_normalized, X_normalized.T) / N
    eig_vals, eig_vecs = eig(M)
    principal_values = eig_vals[:num_components]
    principal_components = eig_vecs[:, :num_components]
    P = projection_matrix(principal_components)
    reconst = (P @ X_normalized) + mean
    return reconst, mean, principal_values, principal_components
