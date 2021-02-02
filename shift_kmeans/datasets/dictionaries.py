"""
Functions to generate dictionaries
"""

import numpy as np

from sklearn.utils.extmath import row_norms
from shift_kmeans.utils import check_rng
from sklearn.metrics.pairwise import euclidean_distances
from shift_kmeans.datasets.utils import roll_rows


def _orthogonal_morlet(D, max_freq, min_freq, max_std, min_std, rng):
    """
    Greedily orthogonalize D
    """

    n_atoms, n_features = D.shape
    times = np.arange(n_features)
    shift = np.tile(np.arange(n_features), (n_atoms, 1))
    discarded_ids = None
    D_row_norms = row_norms(D)[None, :]

    while discarded_ids is None or discarded_ids.size > 0:
        rolled_D = roll_rows(D, shift)
        rolled_D = rolled_D.swapaxes(0, 1)

        for rollD in rolled_D:
            dist = euclidean_distances(
                D, rollD, squared='True', X_norm_squared=D_row_norms)
            np.fill_diagonal(dist, 2)
            dist = np.triu(dist)
            dist = np.where(dist == 0, 2, dist)
            row_ids, col_ids = np.where(np.abs(dist - 2) > 5e-3)
            row_ids = np.unique(row_ids)
            col_ids = np.unique(col_ids)
            discarded_ids = row_ids if row_ids.size <= col_ids.size else col_ids

            for j in discarded_ids:
                freq = rng.rand()\
                    * (max_freq - min_freq)\
                    + min_freq
                phase = rng.rand() * 2 * np.pi
                std = rng.rand() * (max_std - min_std) + min_std
                sinu = np.sin(2 * np.pi * freq * times + phase)
                gauss = np.exp(-((times - n_features/2) ** 2) / std ** 2)
                D[j] = sinu * gauss
                D[j] = D[j] / np.linalg.norm(D[j])

            if discarded_ids.size > 0:
                break
    return D


def real_morlet(n_atoms, n_features, freq_range=(0.1, 1),
                gauss_std_range=(0.01, 0.5), rng=None,
                mode='continuous'):
    """
    Real Morlet wavelet dictionary

    Parameters
    ----------
    n_atoms (int):      Number of atoms
    n_features (int):   Atom's length
    dtype (dtype):      Dictionary's data type

    Returns
    -------
    D (ndarray):    The dictionary

    Shapes:
    -------
    D: (n_atoms, n_features)
    """

    rng = check_rng(rng)

    min_freq, max_freq = freq_range
    min_std, max_std = gauss_std_range
    min_std *= n_features
    max_std *= n_features

    frequencies = np.empty((n_atoms, 1))
    stds = np.empty((n_atoms, 1))

    if isinstance(mode, str) and mode == 'disjoint':
        freq_range_length = (max_freq - min_freq) / n_atoms
        freq_win_std = freq_range_length / 8
        std_range_length = (max_std - min_std) / n_atoms
        std_win_std = std_range_length / 8

        for i in range(n_atoms):

            max_freq = min_freq + freq_range_length
            mid_freq = (max_freq + min_freq) / 2
            frequencies[i] = rng.normal(
                loc=mid_freq, scale=freq_win_std)
            min_freq += freq_range_length

            max_std = min_std + std_range_length
            mid_std = (max_std + min_std) / 2
            stds[i] = rng.normal(loc=mid_std, scale=std_win_std)
            min_std += std_range_length

        rng.shuffle(stds)

    elif isinstance(mode, str) and mode == 'continuous':
        frequencies = rng.random(size=(n_atoms, 1))\
            * (max_freq - min_freq) + min_freq
        stds = rng.random(size=(n_atoms, 1)) * (max_std - min_std) + min_std

    phases = rng.random(size=(n_atoms, 1)) * 2 * np.pi
    times = np.arange(n_features)

    D = np.sin(2 * np.pi * frequencies * times + phases)

    gaussians = np.exp(-((times - n_features/2) ** 2) / stds ** 2)

    D = D * gaussians
    D = D / np.linalg.norm(D, axis=1, keepdims=True)

    return D


def sindict(n_atoms, n_features, rng=None, dtype=np.float64):
    '''
    Discrete Sine Transform (DST-I) dictionary

    Args:
        n_atoms (int):
            Number of atoms
        rng (int, Generator instance, or None):
            Random generator. Use int for deterministic randomness.
        n_features (int):
            length of each atom in the dictionary

    Returns:
        D (ndarray): Dictionary

    Shape:
        D: (n_atoms, n_features)



    Ref: https://en.wikipedia.org/wiki/Discrete_sine_transform#DST-I
    '''

    rng = check_rng(rng)

    n = np.arange(n_features)[np.newaxis, :]  # time
    k = n.transpose()  # frequency
    time_freq_matrix = (k+1)*(n+1)
    D = np.sin(np.pi*time_freq_matrix/(n_features+1), dtype=dtype)
    ind = rng.permutation(n_atoms)
    ind = ind[:n_atoms]
    D = D[ind]

    D_norm = np.linalg.norm(D, ord=2, axis=1)[:, np.newaxis]
    D = D / D_norm

    return D
