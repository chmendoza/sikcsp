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
import shift_kmeans.extutils as extutils
import shift_kmeans.shift_kmeans as sikmeans


###############################################################################
# Parameters for the experiment

images_path =\
    '/home/cmendoza/MEGA/Research/Shift_Invariant_kmeans/report02/images'
ext = 'png'
n_runs = 10
sampling_freq = 250  # samples/Hz
min_freq, max_freq = 1, 12  # hZ
sample_length, centroid_length = 1, 0.8  # seconds
n_samples = np.arange(start=600, stop=2100, step=200)
n_centroids = np.arange(start=6, stop=11)
snr = 1e12
morlet_mode = 'disjoint'
gauss_std_range = (0.001, 0.20)

min_freq = min_freq / sampling_freq
max_freq = max_freq / sampling_freq
sample_length = np.int(sample_length * sampling_freq)
centroid_length = np.int(centroid_length * sampling_freq)
n_centroids_per_sample = 1

np.random.seed(13)

###############################################################################
# Start the experiment: run shift-invariant k-means for different number of
# samples and centroids

estimation_snr = [[None] * n_samples.size for _ in range(n_centroids.size)]
average_snr = np.empty((n_centroids.size, n_samples.size))
D = [[None] * n_samples.size for _ in range(n_centroids.size)]
D_hat = [[None] * n_samples.size for _ in range(n_centroids.size)]

for i in range(n_centroids.size):
    for j in range(n_samples.size):

        # Build the dictionary
        D[i][j] = dictionaries.real_morlet(
            n_centroids[i], centroid_length, mode=morlet_mode,
            freq_range=(min_freq, max_freq), gauss_std_range=gauss_std_range)

        # Get samples from the dictionary
        X, _, _, _ = \
            generator.make_samples_from_shifted_centroids(
                n_samples[j], sample_length, D[i][j], n_centroids_per_sample,
                scale=False, snr=snr)

        # Call the main algorithm
        D_hat[i][j], _, _, _, _ =\
            sikmeans.shift_invariant_k_means(
                X, n_centroids[i], centroid_length, n_init=n_runs)

        # Normalize the centroids
        D_hat[i][j] = D_hat[i][j]\
            / np.linalg.norm(D_hat[i][j], axis=1, keepdims=True)

        # Plot true and learned centroids
        n_rows = np.int(np.round(np.sqrt(n_centroids[i])))
        n_cols = np.int(np.ceil(n_centroids[i] / n_rows))

        fig1, axes1 = plt.subplots(n_rows, n_cols)
        fig2, axes2 = plt.subplots(n_rows, n_cols)
        axes1 = axes1.flatten()
        axes2 = axes2.flatten()

        for k in range(n_centroids[i]):
            axes1[k].plot(D_hat[i][j][k])
            axes2[k].plot(D[i][j][k])

        fig1.suptitle(r'$\hat{D}$ (learned centroids)')
        fig2.suptitle(r'D (true centroids)')

        fig1.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig2.tight_layout(rect=[0, 0.03, 1, 0.95])

        fig1.savefig(
            os.path.join(
                images_path,
                f'D_hat_{n_centroids[i]}_clusters_{n_samples[j]}_samples.{ext}'),
            bbox_inches='tight')
        fig2.savefig(
            os.path.join(
                images_path,
                f'D_{n_centroids[i]}_clusters_{n_samples[j]}_samples.{ext}'),
            bbox_inches='tight')
        plt.close('all')

        # Get shift-invariant min distance. Roll the learned centroids and
        # find, for each sample, the closest rolled centroid.
        D_rows_norm = row_norms(D[i][j], squared=True)[np.newaxis, :]
        shift = np.tile(np.arange(centroid_length), (n_centroids[i], 1))
        rolled_D_hat = utils.roll_rows(D_hat[i][j], shift)
        rolled_D_hat = rolled_D_hat.swapaxes(0, 1)
        D_label = np.empty((centroid_length, n_centroids[i]), dtype=np.int)
        D_min_dist = np.empty((centroid_length, n_centroids[i]))

        # Iterate over all possible shifts.
        for k in range(centroid_length):
            D_label[k], D_min_dist[k] =\
                pairwise_distances_argmin_min(
                    X=D[i][j], Y=rolled_D_hat[k],
                    metric_kwargs={'squared': True,
                                   'X_norm_squared': D_rows_norm})

        best_shifts = np.argmin(D_min_dist, axis=0)
        D_label = D_label[best_shifts, np.arange(n_centroids[i])]
        D_min_dist = np.min(D_min_dist, axis=0)

        # SNR of the estimation. centroids are unit norm.
        estimation_snr[i][j] = 1/D_min_dist
        average_snr[i][j] = np.sum(estimation_snr[i][j]) / n_centroids[i]

        # Print SNR table
        print('Signal to noise ration (SNR) of the centroid estimation for '
              f'{n_samples[j]} samples and {n_centroids[i]} clusters:\n')
        headers = ['centroid #', 'SNR [dB]']
        extutils.print_table_header(headers)
        width = [len(h) + 5 for h in headers]

        for k in range(n_centroids[i]):
            print('{: >{wtk}}{: >{wsnr}.3{ftype}}'.format(
                k, 10 * np.log10(estimation_snr[i][j][k]), wtk=width[0],
                wsnr=width[1], ftype='g'))

        # Check if there are learned centroids that were not assigned
        # to a true centroid
        if len(set(D_label)) != n_centroids[i]:
            print('Two or more learned centroids are too close to each other')

# Print average SNR table
print('Average SNR...')
headers = ['# of samples', '# of clusters', 'Sample density',
           'Average SNR [dB]']
extutils.print_table_header(headers)
width = [len(h) + 5 for h in headers]

for i in range(n_centroids.size):
    for j in range(n_samples.size):
        print('{: >{wns}}{: >{wnc}}{: >{wsd}.3g}{: >{wsnr}.3g}'.format(
            n_samples[j], n_centroids[i], n_samples[j] / n_centroids[i],
            10 * np.log10(average_snr[i][j]), wns=width[0], wnc=width[1],
            wsd=width[2], wsnr=width[3]))

# Plot average SNR
sample_density = n_samples[None, :] / n_centroids[:, None]
unique_sample_density, sd_ids, sd_counts = np.unique(
    sample_density, return_index=True, return_counts=True)

if unique_sample_density.size < sample_density.size:
    print('The following combinations of number of samples and clusters '
          'produce the same sample density: ')
    repeated_ids = np.asarray(sd_counts > 1).nonzero()[0]
    repeated_densities = unique_sample_density[repeated_ids]
    for rd in repeated_densities:
        nk_id, ns_id = np.asarray(sample_density == rd).nonzero()
        for i, j in zip(nk_id, ns_id):
            print(f'# of samples: {n_samples[j]}, # of clusters: '
                  f'{n_centroids[i]}, '
                  f'average SNR [dB]: {10 * np.log10(average_snr[i][j])}')
        print('\n')


average_snr = average_snr.flatten()
average_snr = average_snr[sd_ids]

plt.figure()
plt.plot(unique_sample_density, 10 * np.log10(average_snr))
plt.title('Average SNR as a function of sample density')
plt.xlabel('Number of samples / Number of clusters')
plt.ylabel('Average SNR [dB]')
plt.show(block=False)

input('Press ENTER to exit')
