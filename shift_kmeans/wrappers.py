"""
Numpy ans scikit-learn wrappers
"""
import numpy as np

from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import pairwise_distances_argmin_min

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

def shift_invariant_pairwise_distances_argmin_min(X, kernels, x_squared_norms):
    """
    Shift-invariant wrapper of http://bit.ly/argmin_min_sklearn
    """

   # euclidean_distances() requires 2D
    if x_squared_norms.ndim == 1:
        x_squared_norms = x_squared_norms.reshape(1,-1)

    n_samples, sample_length = X.shape
    n_kernels, kernel_length = kernels.shape
    n_shifts = sample_length - kernel_length + 1

    metric_kwargs = {'squared': True, 'X_norm_squared': x_squared_norms}

    # Assume n_shifts <= 2**16 = 65536
    best_shifts = np.empty((n_kernels, n_samples), dtype=np.uint16)
    best_distances = np.empty((n_kernels, n_samples))

    # Pad and roll the kernels
    padded_kernels = np.pad(kernels, [(0, 0), (0, n_shifts-1)],
                            mode='constant')
    shifts = np.tile(np.arange(n_shifts, dtype=np.int), (n_kernels, 1))
    rolled_kernels = utils.roll_rows(padded_kernels, shifts)

    for kernel_id, rolled_kernel in enumerate(rolled_kernels):
        best_shifts[kernel_id], best_distances[kernel_id] = \
            pairwise_distances_argmin_min(
                X=X, Y=rolled_kernel, metric_kwargs=metric_kwargs)

    # For each sample, find closest shifted kernel
    labels = np.argmin(best_distances, axis=0)
    best_shifts = best_shifts[labels, np.arange(n_samples)]
    best_distances = np.min(best_distances, axis=0)

    return labels, best_shifts, best_distances
