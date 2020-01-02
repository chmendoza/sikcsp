"""
Extensions to numpy functions
"""

import numpy as np

def roll_rows(a, shift):
    """
    Roll each row of `a` independently

    shift[i] has the offsets to be applied independently to a[i]. `shift` must
    have the same number of rows than `a`.

    See https://stackoverflow.com/a/20361561/4292705
    """

    row_indices, column_indices = np.ogrid[:a.shape[0], :a.shape[1]]

    # Use always a negative shift, so that column_indices are valid.
    # (could also use module operation)
    shift[shift < 0] += a.shape[1]

    if shift.shape[0] == 1:
        shift = shift.T

    shift = np.expand_dims(shift, -1)
    column_indices = column_indices - shift

    if column_indices.ndim == 3:
        row_indices = np.expand_dims(row_indices, -1)

    return a[row_indices, column_indices]

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
    idx = np.reshape(np.arange(n_2D_matrices),[-1,1,1])
    idy = np.reshape(np.arange(n_rows),[1,-1,1])
    shift = shift[:,:,None]
    idz = np.arange(n_cols)
    idz = (idz - shift) % n_cols

    return a[idx,idy,idz]


def add_noise(X, snr):
    """
    Add Gaussian noise to `X` to achieve the given `snr`

    It adds Gaussian noise to `X` so that the ratio between the linear power of
    the unperturbed `X` and the noise is equal to `snr`.

    Args:
        X (numpy.ndarray):  Unperturbed data matrix, with samples in its rows.
        snr (float):        Signal to noise ratio
    """

    noise = np.random.normal(0, 1, X.shape)
    noise_norm = np.linalg.norm(noise, axis=1, keepdims=True)
    X_norm = np.linalg.norm(X, axis=1, keepdims=True)
    noise = noise * X_norm / (np.sqrt(snr) * noise_norm)
    X = X + noise

    return X



