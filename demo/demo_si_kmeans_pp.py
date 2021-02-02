"""
Numerical test of the shift-invariant k-means++ algorithm.
"""

from shift_kmeans.wrappers import si_pairwise_distances_argmin_min, si_row_norms
import shift_kmeans.shift_kmeans as sikmeans
import shift_kmeans.datasets.generator as generator
import shift_kmeans.datasets.dictionaries as dictionaries
from sklearn.utils import check_random_state
from sklearn.utils.extmath import row_norms
import numpy as np
import os
import sys

# Add path to package directory to access main module using absolute import
# This is needed because this is meant to be executed as an script
# See https://stackoverflow.com/a/11537218/4292705
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))


n_runs = 10
n_samples, n_features = 300, 1000
n_centroids, centroid_length = 5, 800
n_centroids_per_sample = 1
n_shifts = n_features - centroid_length + 1

random_state = 13
random_state = check_random_state(random_state)

# Discrete-time Sine transform dictionary
D = dictionaries.sindict(n_centroids, centroid_length)

# Get samples from the dictionary
X, true_centroid_index, true_time_shift, _ = \
    generator.make_samples_from_shifted_centroids(
        n_samples, n_features, D, n_centroids_per_sample, scale=False, snr=10)

# Precompute squared norms of rows of X for efficient computation of
# euclidean distances between centroids and samples.
x_squared_norms = si_row_norms(X, centroid_length, squared=True)

kmeanspp_pot, rand_pot = 0, 0

for run in np.arange(n_runs):
    # Run shift-invariant kmeans++
    kmeanspp_centroids = sikmeans.init_centroids(
        X, n_centroids, centroid_length, 'k-means++', x_squared_norms,
        random_state)

    # Pick centroids randomly from X for comparison
    rand_centroids = sikmeans.init_centroids(
        X, n_centroids, centroid_length, 'random', random_state)

    # Compute potential with seed centroids
    kmeanspp_labels, _, kmeanspp_mindist =\
        si_pairwise_distances_argmin_min(
            X, kmeanspp_centroids, x_squared_norms)
    kmeanspp_pot += kmeanspp_mindist.sum()

    # Compute potential with random centroids
    rand_labels, _, rand_mindist =\
        si_pairwise_distances_argmin_min(
            X, rand_centroids, x_squared_norms)
    rand_pot += rand_mindist.sum()

print('-------------------------------------')
print(f'Average potential after {n_runs} runs')
print('-------------------------------------\n')
print(f'With centroids computed using k-means++: {kmeanspp_pot / n_runs}')
print(f'With random centroids: {rand_pot / n_runs}')
