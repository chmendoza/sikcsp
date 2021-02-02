"""
Extensions to numpy functions
"""

import numpy as np

from shift_kmeans.utils import check_rng

def roll_rows(array, shift):
    """
    Roll each row of `array` independently

    Parameters
    ----------
    array (numpy.ndarray):
        2D matrix whose rows are to be rolled
    shift (numpy.ndarray):
        shift[i] has the offset(s) to be applied independently to a[i].

    Returns
    -------
    rolled_array (numpy.ndarray):
        Array with rolled rows according to shift

    Shapes
    ------
    array:          (n_rows, n_cols)
    shift:          (n_rows, n_shifts)
    rolled_array:   (n_rows, n_shifts, n_cols)

    See https://stackoverflow.com/a/20361561/4292705
    """

    n_rows, n_cols = array.shape

    row_indices, column_indices = np.ogrid[:array.shape[0], :array.shape[1]]

    # Use always a negative shift, so that column_indices are valid.
    # (could also use module operation)
    shift[shift < 0] += n_cols

    # Check array shape
    assert array.ndim == 2, 'The array to be rolled must have two dimensions'

    # Checl shift shape
    assert shift.ndim == 2, 'The shift array must have two dimensions'
    assert shift.shape[0] == n_rows,\
        '`array` and `shifts` must have the same number of rows'

    # Add third dimension to broadcast properly the indexes.
    shift = np.expand_dims(shift, -1)

    # Compute column indexes and broadcast
    column_indices = column_indices - shift

    # Add third dimension to row indexes for proper broadcasting
    row_indices = np.expand_dims(row_indices, -1)

    return array[row_indices, column_indices]

def roll_rows_of_3D_matrix(a, shift):
    """
    Roll each row of each 2D matrix independently

    a[i] is a 2D matrix. Each of its rows is rolled independently, according to
    shift[i].

    See https://stackoverflow.com/a/44146327/4292705

    It uses broadcasting to create a 3D cube of indices to reorder (or roll)
    `a`.
    """

    n_2D_matrices = a.shape[0]
    n_rows = a.shape[1]
    n_cols = a.shape[2]
    idx = np.reshape(np.arange(n_2D_matrices), [-1, 1, 1])
    idy = np.reshape(np.arange(n_rows), [1, -1, 1])
    shift = shift[:, :, None]
    idz = np.arange(n_cols)
    idz = (idz - shift) % n_cols

    return a[idx, idy, idz]


def add_noise(X, snr, rng=None):
    """
    Add Gaussian noise to `X` to achieve the given `snr`

    It adds Gaussian noise to `X` so that the ratio between the linear power of
    the unperturbed `X` and the noise is equal to `snr`.

    Args:
        X (numpy.ndarray):
            Unperturbed data matrix, with samples in its rows.
        snr (float):
            Signal to noise ratio
        rng (int, Generator instance, None):
            Random generator
    """

    rng = check_rng(rng)

    noise = rng.normal(0, 1, X.shape)
    noise_norm = np.linalg.norm(noise, axis=1, keepdims=True)
    X_norm = np.linalg.norm(X, axis=1, keepdims=True)
    noise = noise * X_norm / (np.sqrt(snr) * noise_norm)
    X = X + noise

    return X
