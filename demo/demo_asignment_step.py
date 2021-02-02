"""
Numerical test of the asignment step in the shift-invariant k-means
algorithm
"""

import shift_kmeans.shift_kmeans as sikmeans
import shift_kmeans.datasets.utils as utils
import shift_kmeans.datasets.generator as generator
import shift_kmeans.datasets.dictionaries as dictionaries
import shift_kmeans
from sklearn.metrics.pairwise import pairwise_distances_argmin_min
from sklearn.utils import check_random_state
from sklearn.utils.extmath import row_norms
import numpy as np
import os
import sys

# Add path to package directory to access main module using absolute import
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
X, true_labels, true_shifts, _ = \
    generator.make_samples_from_shifted_centroids(
        n_samples, n_features, D, n_centroids_per_sample, scale=False,
        snr=10)

# Squeeze singleton dimension to get 1D array
true_labels = true_labels.squeeze()
true_shifts = true_shifts.squeeze()


# subtract of mean of X for more accurate distance computations
X_mean = X.mean(axis=0)
X_centered = X - X_mean

# Precompute squared norms of rows of X for efficient computation of
# euclidean distances between centroids and samples.
x_squared_norms = np.empty((n_shifts, n_samples))
for shift in range(n_shifts):
    x_squared_norms[shift] = row_norms(
        X_centered[:, shift:shift+centroid_length], squared=True)

# Center the centroids
D_mean = D.mean(axis=0)
D_centered = D-D_mean

# Asignment step of shift-invariant k-means
labels, shifts, distances = sikmeans._asignment_step(
    X_centered, D_centered, x_squared_norms)

# Compare labels with ground-thruth
dist_to_true_labels = np.sum((true_labels - labels)**2)
print(f'Squared distance to true labels: {dist_to_true_labels}')

# Compare shifts
dist_to_true_shifts = np.sum((true_shifts - shifts)**2)
print(f'Squared distance to true shifts: {dist_to_true_shifts}')
