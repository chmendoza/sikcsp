"""
Generate synthetic datasets
"""
import itertools

import numpy as np
from scipy import special

from shift_kmeans.utils import check_rng

from ..datasets.dictionaries import sindict
from ..datasets.utils import roll_rows_of_3D_matrix, add_noise


def make_data(sample_length, sparsity, n_atoms, n_samples_per_comb, snr,
              data_path, name_prefix):

    n_nonzeros = n_atoms * sparsity
    data_size = np.int(special.comb(n_atoms, n_nonzeros))*n_samples_per_comb
    num_training_samples = np.int(0.7 * data_size)
    num_validation_samples = np.int(0.2 * data_size)

    # Generates the dictionary
    D = sindict(sample_length, n_atoms, np.float32)

    # Create list and dictionary of IDs for data samples
    IDlist, IDdict = sampleIDs(name_prefix, data_size,
                               num_training_samples, num_validation_samples)

    # Generate data samples and save them in .npz files
    make_samples(D, n_atoms, n_nonzeros, n_samples_per_comb,
                 snr, data_path, IDlist)

    return D, IDlist, IDdict


def make_samples(dictionary, n_atoms, n_nonzeros, n_samples_per_comb,
                 snr, data_path, IDlist):
    '''
    Generates samples from dictionary and saves them in a folder

    It generates samples by sparsely combining `n_nonzeros` out of `n_atoms`
    atoms in a `dictionary`. For each combination of atoms, it generates
    `n_samples_per_comb` samples with different set of coefficient amplitudes
    drawn from a normal distribution.

    Gaussian noise is added to the samples to achieve the desired `snr`.

    Each generated samples its named according to the list of IDs in `IDlist`,
    and saved in a *.npz file in `data_path`.

    Args:
        dictionary (2D matrix): Dictionary
        n_atoms (int): Number of atoms to be sparsely combined
        n_nonzeros (int): Num of atoms with non-zero coefficients
        n_samples_per_comb (int): Num of samples per sparse combination
        snr (float): Signal to Noise power ratio in linear scale
        data_path (int): Full path to folder where data will be saved
        IDlist (string list): Names for each sample

    '''

    # Amplitudes of coefficients are ~Normal(mu, sigma**2)
    sigma = 0.3; mu = 1

    rows = np.arange(n_atoms)
    itercomb = itertools.combinations(rows, n_nonzeros)
    sample_cnt = 0
    sample_length = dictionary.shape[0]
    for comb in itercomb:
        Z = np.zeros((n_samples_per_comb,n_atoms), dtype=np.float32)
        atom_indices = np.asarray(comb)
        amplitudes = sigma * np.random.randn(n_nonzeros,n_samples_per_comb) + mu
        Z[:,atom_indices] = amplitudes
        X = np.matmul(Z,dictionary)
        noise = np.random.normal(0, 1, X.shape)
        noise_norm = np.linalg.norm(noise, axis=1)
        noise = noise * np.linalg.norm(X,axis=1)/(np.sqrt(snr)*noise_norm)
        X = X + noise
        for sample in range(n_samples_per_comb):
            np.savez(data_path + IDlist[sample_cnt] + '.npz', x=X[sample],
                     trueZ=Z[sample])
            sample_cnt += 1



def make_samples_from_shifted_centroids(n_samples, sample_length, dictionary,
                                      n_centroids_per_sample, scale=True,
                                      sigma=0.2, mu=1, snr=10,
                                      rng=None):
    """
    Make samples by shifting centroids

    It makes `n_samples` samples of length `sample_length` by sparsely
    combining `n_centroids_per_sample` centroids from the `dictionary`. The
    generated samples, `X`, and a list of tuples (nu,tau,alpha), are returned,
    where nu, tau, and alpha are the index, time shift, and amplitudes of the
    centroids picked in the sparse combination. Gaussian noise is added to the
    samples to achieve the given linear `snr`.

    Args:
        n_samples (int):
            Number of samples to be generated
        sample_length (int):
            Length of each sample
        dictionary (numpy.ndarray):
            The dictionary of centroids used to generate the samples.
        n_centroids_per_sample (int):
            Number of centroids used to generate a sample.
        scale (bool):
            If True, scale the centroids with amplitudes drawn from
            Y ~ Normal(mu, sigma^2). If False, don't scale the centroids.
        sigma (float):
            Standard deviation of Y
        mu (float):
            Mean of Y
        snr (float):
            Linear signal to noise power ratio.
        rng (int, Generator instance, or None):
            Random generator

    Returns:
        X (numpy.ndarray):              Matrix with a sample in each row.
        centroid_indexes (numpy.ndarray): Indexes of the centroids.
        time_shifts (numpy.ndarray):    Time shifts of the centroids.
        amplitudes (numpy.ndarray):     Amplitudes to scale the centroids.

    Shapes:
        dictionary:     (`n_centroids`, `centroid_length`)
        X:              (`n_samples`, `sample_length`)
        centroid_indexes: (`n_samples`, `n_centroids_per_sample`)
        time_shifts:    (`n_samples`, `n_centroids_per_sample`)
        amplitudes:     (`n_samples`, `n_centroids_per_sample`)
    """

    rng = check_rng(rng)

    n_centroids, centroid_length = dictionary.shape
    centroid_indexes = rng.integers(
        0, n_centroids, (n_samples, n_centroids_per_sample))
    time_shifts = rng.integers(
        0, sample_length-centroid_length+1, (n_samples, n_centroids_per_sample))

    if scale:
        amplitudes = sigma * \
            rng.standard_normal(size=(n_samples, n_centroids_per_sample)) + mu
    else:
        amplitudes = np.ones((n_samples, n_centroids_per_sample))

    padded_dictionary = np.pad(
        dictionary, [(0, 0), (0, sample_length-centroid_length)],
        mode='constant')
    chosen_atoms = padded_dictionary[centroid_indexes]

    rolled_centroids = roll_rows_of_3D_matrix(chosen_atoms, time_shifts)
    X = np.einsum('ij,ijk->ik', amplitudes, rolled_centroids)
    X = add_noise(X, snr)

    return X, centroid_indexes, time_shifts, amplitudes


def sampleIDs(name_prefix, data_size, num_training_samples,
        num_validation_samples):

    test_start = num_training_samples + num_validation_samples - 1

    IDlist = [name_prefix + str(i) for i in range(data_size)]
    IDdict = dict(
        train = IDlist[:num_training_samples],
        validate = IDlist[num_training_samples:test_start],
        test = IDlist[test_start:]
    )

    return IDlist, IDdict

