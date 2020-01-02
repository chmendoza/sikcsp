"""
Numpy ans scikit-learn wrappers
"""
import numpy as np

from sklearn.metrics.pairwise import euclidean_distances

from shift_kmeans.datasets import utils

def shift_invariant_euclidean_distances(kernels, X, X_norm_squared, squared=False):
    """
    Shift-invariant wrapper of euclidean_distances()

    Rows of `kernels` are shorter than rows of X. Rows of `kernels` are
    zero-padded an rolled to compute the distances to the rows of `X` at each
    shift.

    Parameters
    ----------
    kernels (numpy.ndarray):
        kernels[i] is a kernel with length `kernel_length`.
    X (numpy.ndarray):
        X[i] is a sample with length `n_features`. 
        `kernel_length` < `n_features`.
    X_norm_squared (numpy.ndarray):
        Precomputed squared euclidean norm of rows of `X`.
    squared (bool):
        If True, the euclidean distance is squared.

    Returns
    -------
    distances (numpy.ndarray):
        distances[i][j][k] is the euclidean distance from kernels[i], shifted
        `j` positions, to sample X[k]. 
    """

    n_kernels, kernel_length = kernels.shape
    n_samples, n_features = X.shape
    n_shifts = n_features - kernel_length + 1

    # Pad and roll the kernels
    kernels_padded = np.pad(kernels, [(0, 0), (0, n_shifts-1)], mode='constant')
    shift = np.tile(np.arange(n_shifts), (n_kernels, 1))
    kernels_rolled = utils.roll_rows(kernels_padded, shift)

    distances = np.empty((n_kernels, n_shifts, n_samples))

    # Use euclidean_distances() to find the distance from all the rolled
    # versions of each kernel to all the samples in X
    for kernel in np.arange(n_kernels):
        distances[kernel] = euclidean_distances(kernels_rolled[kernel], X,
                                                X_norm_squared, squared=True)

    return distances
