"""
Numerical test of _k_init(), the shift-invariant k-means++ algorithm.
"""

import os
import sys

# Add path to package directory to access main module using absolute import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt

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
n_kernels, kernel_length = 5, 800
n_kernels_per_sample = 1
n_shifts = n_features - kernel_length + 1

random_state = 13
random_state = check_random_state(random_state)

# Discrete-time Sine transform dictionary
D = dictionaries.sindict(n_kernels, n_features)

# Get samples from the dictionary
X, true_kernel_index, true_time_shift, _ = \
    generator.make_samples_from_shifted_kernels(
        n_samples, n_features, D, n_kernels_per_sample, scale=False, snr=10)

# subtract of mean of X for more accurate distance computations
X_mean = X.mean(axis=0)
X_centered = X - X_mean

# Precompute squared norms of rows of X for efficient computation of
# Euclidean distances between kernels and samples.
x_squared_norms = row_norms(X_centered, squared=True)

init_pot, rand_pot = 0, 0

for run in np.arange(n_runs):
    # Run shift-invariant kmeans++
    init_kernels, init_shifts =  sikmeans._k_init(
        X_centered, n_kernels, kernel_length, x_squared_norms, random_state)

    # Padd and shift kernels
    init_kernels = np.pad(init_kernels, [(0, 0), (0, n_shifts-1)], mode='constant')
    init_kernels = utils.roll_rows(init_kernels, init_shifts)

    # Add back the mean to the kernels
    init_kernels += X_mean

    # Pick kernels randomly from X for comparison
    unique_ids = random_state.choice(n_samples, size=n_kernels, replace=False)
    rand_kernels = X[unique_ids]

    # Compute potential with seed kernels
    init_labels, init_mindist = pairwise_distances_argmin_min(
        X=X_centered, Y=init_kernels, metric='euclidean',
        metric_kwargs={'squared': True})
    init_pot += init_mindist.sum()

    # Compute potential with random kernels
    rand_labels, rand_mindist = pairwise_distances_argmin_min(
        X=X_centered, Y=rand_kernels, metric='euclidean',
        metric_kwargs={'squared': True})
    rand_pot += rand_mindist.sum()

print(f'Average potential after {n_runs} runs\n')
print('-------------------------------------')
print(f'With kernels computed using kmeans++: {init_pot / n_runs}\n')
print(f'With random kernels: {rand_pot / n_runs}\n')
