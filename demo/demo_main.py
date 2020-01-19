"""
Numerical test of the shift-invariant k-means algorithm.
"""

import os
import sys

# Add path to package directory to access main module using absolute import
sys.path.insert(0,
                os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             '..')))

import numpy as np

import matplotlib
import matplotlib.pyplot as plt

from sklearn.utils.extmath import row_norms
from sklearn.metrics.pairwise import pairwise_distances_argmin_min

import shift_kmeans.datasets.dictionaries as dictionaries
import shift_kmeans.datasets.generator as generator
import shift_kmeans.datasets.utils as utils
import shift_kmeans.shift_kmeans as sikmeans


###############################################################################
# Parameters for the experiment

n_runs = 10
sampling_freq = 250  # samples/Hz
min_freq, max_freq = 1, 12  # hZ
sample_length, kernel_length = 1, 0.8  # seconds
n_samples = np.arange(start=300, stop=2000, step=200)
n_kernels = np.arange(start=6, stop=11)

min_freq = min_freq / sampling_freq
max_freq = max_freq / sampling_freq
sample_length = np.int(sample_length * sampling_freq)
kernel_length = np.int(kernel_length * sampling_freq)
n_kernels_per_sample = 1

np.random.seed(13)

###############################################################################
# Start the experiment: run shift-invariant k-means for different number of
# samples and kernels

estimation_snr = [[None] * n_samples.size for _ in range(n_kernels.size)]
D = [[None] * n_samples.size for _ in range(n_kernels.size)]
D_hat = [[None] * n_samples.size for _ in range(n_kernels.size)]

for i in range(n_kernels.size):
    for j in range(n_samples.size):

        # Build the dictionary
        D[i][j] = dictionaries.real_morlet(
            n_kernels[i], kernel_length, mode='disjoint',
            freq_range=(min_freq, max_freq))

        # Get samples from the dictionary
        X, _, _, _ = \
            generator.make_samples_from_shifted_kernels(
                n_samples[j], sample_length, D[i][j], n_kernels_per_sample,
                scale=False, snr=10)

        # Call the main algorithm
        D_hat[i][j], _, _, _, _ =\
            sikmeans.shift_invariant_k_means(
                X, n_kernels[i], kernel_length, n_init=n_runs)

        # Normalize the kernels
        D_hat[i][j] = D_hat[i][j]\
            / np.linalg.norm(D_hat[i][j], axis=1, keepdims=True)

        # Plot true and learned kernels
        n_rows = np.int(np.round(np.sqrt(n_kernels[i])))
        n_cols = np.int(np.ceil(n_kernels[i] / n_rows))

        fig1, axes1 = plt.subplots(n_rows, n_cols)
        fig2, axes2 = plt.subplots(n_rows, n_cols)
        axes1 = axes1.flatten()
        axes2 = axes2.flatten()

        for k in range(n_kernels[i]):
            axes1[k].plot(D_hat[i][j][k])
            axes2[k].plot(D[i][j][k])

        fig1.suptitle(r'$\hat{D}$ (learned kernels)')
        fig2.suptitle(r'D (true kernels)')

        plt.show(block=False)

        # Get shift-invariant min distance
        D_rows_norm = row_norms(D[i][j], squared=True)[np.newaxis, :]
        shift = np.tile(np.arange(kernel_length), (n_kernels[i], 1))
        rolled_D_hat = utils.roll_rows(D_hat[i][j], shift)
        rolled_D_hat = rolled_D_hat.swapaxes(0, 1)
        D_label = np.empty((kernel_length, n_kernels[i]), dtype=np.int)
        D_min_dist = np.empty((kernel_length, n_kernels[i]))

        for k in range(kernel_length):
            D_label[k], D_min_dist[k] =\
                pairwise_distances_argmin_min(
                    X=D[i][j], Y=rolled_D_hat[k],
                    metric_kwargs={'squared': True,
                                   'X_norm_squared': D_rows_norm})

        best_shifts = np.argmin(D_min_dist, axis=0)
        D_label = D_label[best_shifts, np.arange(n_kernels[i])]
        D_min_dist = np.min(D_min_dist, axis=0)

        # SNR of the estimation. Kernels are unit norm.
        estimation_snr[i][j] = 10 * np.log10(1/D_min_dist)

        # Print SNR table
        print('Signal to noise ration (SNR) of the kernel estimation for '
              f'{n_samples[j]} samples and {n_kernels[i]} clusters:\n')
        headers = ['Kernel #', 'SNR [dB]']
        width = [len(h) + 5 for h in headers]
        title = ['{: >{width}}'.format(h, width=len(h) + 5) for h in headers]
        bar = ['{: >{width}}'.format('-' * len(h), width=len(h) + 5)
               for h in headers]

        print(''.join(title))
        print(''.join(bar))

        for k in range(n_kernels[i]):
            print('{: >{wtk}}{: >{wsnr}.3{ftype}}'.format(
                k, estimation_snr[i][j][k], wtk=width[0],
                wsnr=width[1], ftype='g'))

        # Check if there are learned kernels that were not assigned
        # to a true kernel
        if len(set(D_label)) != n_kernels[i]:
            print('Two or more learned kernels are too close to each other')


input('Press ENTER to exit')
