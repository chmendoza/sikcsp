"""
Utils functions
"""
import numpy as np
from numpy.lib.stride_tricks import as_strided

from sklearn.utils import check_random_state

def _pick_windows_from_strided(X, n_windows, window_length, random_state):
    """
    See https://stackoverflow.com/a/47983009/4292705
    """

    s0, s1 = X.strides
    n_samples, n_features = X.shape
    n_offsets = n_features - window_length + 1

    # Create `n_windows` unique random indexes for each sample
    windows_ids = np.tile(np.arange(n_offsets), (n_samples, 1))
    windows_ids = np.apply_along_axis(random_state.choice, 1, windows_ids,
                                      size=n_windows, replace=False)

    # Find all possible `n_offsets` windows for each sample
    windows = as_strided(X,
                         shape=(n_samples, n_offsets, window_length),
                         strides=(s0, s1, s1))

    # Choose `n_windows` per sample
    windows = windows[np.arange(n_samples)[:, None, None],
                      windows_ids[:, :, None],
                      np.arange(window_length)[None, None, :]]
    return windows


def _pick_windows_with_integer_indexing(X, n_windows, window_length, random_state):
    """
    See https://stackoverflow.com/a/47982961/4292705
    """

    n_samples, n_features = X.shape
    n_offsets = n_features - window_length + 1

    # Create offsets, the start index of each window
    offsets = np.tile(np.arange(n_offsets), (n_samples, 1))
    chosen_offsets = np.apply_along_axis(random_state.choice, 1, offsets,
                                         size=n_windows, replace=False)

    # Build index of columns according to offsets
    col_ids = np.arange(window_length).reshape(1, 1, window_length)
    col_ids = chosen_offsets[:, :, None] + col_ids

    # Pick the windows using advanced indexing
    windows = X[np.arange(n_samples)[:, None, None], col_ids]

    return windows

def pick_random_windows(X, n_windows, window_length, random_state=None):
    """
    It picks windows randomly from each row of `X`.

    Parameters
    ----------
    X (numpy.ndarray):
        Matrix with data samples in its rows.
    n_windows (int):
        Number of unique windows per sample. Windows are taken randomly, but
        each window is different.
    window_length (int):
        Lenght of the window
    random_state (int, RandomState instance, or None):
        Determines random number generation for picking the windows. Use an
        int to make the randomness deterministic.

    Returns
    -------
    Y (numpy.ndarray):
        Windows of rows of `X` selected randomly

    Shapes
    ------
    X : (`n_samples`, `n_features`)
    Y : (`n_samples`, `n_windows`, `window_length`)
    """

    if X.ndim == 1:
        X = X[np.newaxis, :]

    n_samples, n_features = X.shape
    n_bytes_per_element = X.dtype.itemsize
    n_offsets = n_features - window_length + 1
    peak_memory_use = n_samples * n_offsets * window_length \
                      * n_bytes_per_element

    random_state = check_random_state(random_state)

    if peak_memory_use < 1 * 1e9: # 2 GB
        return _pick_windows_from_strided(X, n_windows, window_length,
                                          random_state)
    else:
        return _pick_windows_with_integer_indexing(X, n_windows, window_length,
                                                   random_state)
