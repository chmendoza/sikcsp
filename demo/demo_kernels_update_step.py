"""
Numerical test of shift_kmeans.shift_kmeans._centroids_update_step
"""

import os
import sys

# Add path to package directory to access main module using absolute import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

from sklearn.utils.extmath import row_norms
from sklearn.utils import check_random_state
from sklearn.metrics.pairwise import pairwise_distances_argmin_min

import shift_kmeans
import shift_kmeans.datasets.dictionaries as dictionaries
import shift_kmeans.datasets.generator as generator
import shift_kmeans.datasets.utils as utils
import shift_kmeans.shift_kmeans as sikmeans

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
X, true_labels, true_shifts, _ =\
    generator.make_samples_from_shifted_centroids(
        n_samples, n_features, D, n_centroids_per_sample, scale=False, snr=10)

# Squeeze singleton dimension to get 1D array
true_labels = true_labels.squeeze()
true_shifts = true_shifts.squeeze()

# subtract of mean of X for more accurate distance computations
X_mean = X.mean(axis=0)
X_centered = X - X_mean

# Center the centroids
D_mean = D.mean(axis=0)
D_centered = D - D_mean

# centroids update step
D_hat = sikmeans._centroids_update_step(X_centered, centroid_length, n_centroids,
                                      true_labels, true_shifts, 1)

# Compare with true dictionary
print(f'D_hat.shape = {D_hat.shape}')
print(f'D_centered.shape = {D_centered.shape}')
dist_to_true_D = np.sum((D_centered - D_hat)**2)
print(f'Squared distance to true dictionary: {dist_to_true_D}')
