"""
Use shift-invariant k-means on top k CSP waves
"""
# %%
import numpy as np
import matplotlib.pyplot as plt

import os
import sys
import subprocess
from argparse import ArgumentParser


# Add path to package directory to access main module using absolute import
# This is needed because this is meant to be executed as an script
# See https://stackoverflow.com/a/11537218/4292705
from numpy import matrix
from numpy.ma.core import right_shift

sys.path.insert(0,
                os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from shift_kmeans.wrappers import si_pairwise_distances_argmin_min, si_row_norms
import shift_kmeans.shift_kmeans as sikmeans
from kcsp import bayes, utils, pltaux

#%%
parser = ArgumentParser()
parser.add_argument("-i", "--init", dest="init",
                    help="k-means init method", default='k-means++')
# args = parser.parse_args()
# https://stackoverflow.com/a/55329134/4292705:
args, unknown = parser.parse_known_args()
# %%
np.random.seed(13)
patient = ['HUP070', 'HUP078']
patient = patient[1]
method = ['regular', 'betadiv', 'max-sliced-Bures']
method = method[0]
dmonte, i_band = 7, 1

data_dir = '/home/cmendoza/Research/sikmeans/LKini2019/results/NER21/'
images_path = '/home/cmendoza/MEGA/Research/mySecondPaper/results/' + patient
data_dir = data_dir + patient
fname = 'results_dmonte%d_trainTest_band%d_%s.mat' % (dmonte, i_band, method)
fname = os.path.join(data_dir, fname)

X = utils.load_data(fname, 'X')
X1_train = X[0][0]  # Preictal (positive class). shape=(n_csp, winlen, k)
X2_train = X[1][0]  # Interictal
X1_test = X[0][1]
X2_test = X[1][1]
X1_train = X1_train[0].T  # first CSP filter. Rows are windows. Columns are time points.
X2_train = X2_train[1].T  # second CSP filter.
X1_test = X1_test[0].T
X2_test = X2_test[1].T
sample_length = X1_train.shape[1]  # number of time points

# %% Run shift-invariant k-means
metric = 'cosine'
init = args.init
srate = 512  # Hz
n_runs = 3
n_centroids = 5
centroid_length = np.int(np.round(250e-3 * srate))  # seconds
#%% Training
C1, nu1, tau1, d1, sumd1, n_iter1 =\
    sikmeans.shift_invariant_k_means(X1_train, n_centroids, centroid_length,\
        metric=metric, init=init, n_init=n_runs, verbose=True)

C2, nu2, tau2, d2, sumd2, n_iter2 =\
    sikmeans.shift_invariant_k_means(X2_train, n_centroids, centroid_length,
                                     metric=metric, init=init, n_init=n_runs, verbose=True)

#%% Compute posterior
k1, k2 = n_centroids, n_centroids
N1, N2 = X1_train.shape[0], X2_train.shape[0]
nu = bayes.cluster_assignment(X1_train, X2_train, C1, C2, metric=metric)
p_J = bayes.likelihood(nu, N=(N1, N2), k=(k1, k2))  # (2,k1,k2)
p_S = np.zeros((2,1,1))
p_S[0] =  N1/(N1+N2)
p_S[1] = N2/(N1+N2)
posterior = p_S * p_J  # ~ P(S=s | J1 = j, J2 = l). (2,k1,k2).

#%% Compute MAP
N1, N2 = X1_test.shape[0], X2_test.shape[0]
nu = bayes.cluster_assignment(X1_test, X2_test, C1, C2, metric=metric)
# Posterior when data (cluster assignments) come from class s=1
# nu[0] are cluster assignments using C1
# nu[1] are cluster assignments using C2
pp1 = posterior[:, nu[0], nu[1]]  # (2, N1, N1).
pp2 = posterior[:, nu[2], nu[3]]
# for each pair assingment pair (j,l), find the value of s (row index) with max MAP. Row 0 is class 1.
I1 = np.argmax(pp1, axis=0)
I2 = np.argmax(pp2, axis=0)
misclass = np.sum(I1 != 0)
misclass += np.sum(I2 != 1)
misclass /= (N1 + N2)

# %% Plot the centroids
n_rows = np.int(np.round(np.sqrt(n_centroids)))
n_cols = np.int(np.ceil(n_centroids / n_rows))
fig1, axes1 = plt.subplots(n_rows, n_cols)
axes1 = axes1.flatten()
for k in range(n_centroids):
    axes1[k].plot(C1[k])

fig1.tight_layout(rect=[0, 0.03, 1, 0.95])
# %% Find and plot 5 closest and furthers waveforms to each centroid
for ic in range(n_centroids):
    cluster = X1_train[nu1 == ic]
    
    cshifts = tau1[nu1 == ic]
    cdistances = d1[nu1 == ic]

    isort = np.argsort(cdistances)
    cdistances = cdistances[isort]
    cluster = cluster[isort]
    cshifts = cshifts[isort]

    fig2 = plt.figure(figsize=(20, 9))
    gs = fig2.add_gridspec(4, 12)
    f2_ax1 = fig2.add_subplot(gs[1:3, :2])
    t = np.arange(start=0, stop=centroid_length / srate, step=1 / srate)
    f2_ax1.plot(t, C1[ic])
    plt.xlabel('Time [s]')
    f2_ax1.set_title('centroid %d' % ic)
    t = np.arange(start=0, stop=sample_length / srate, step=1 / srate)
    for col in range(5):
        f2_ax2 = fig2.add_subplot(gs[:2, 2 * col + 2:2 * col + 4])
        f2_ax2.plot(t, cluster[col])
        wave = pltaux.wave(cluster[col], cshifts[col], centroid_length)
        rline = f2_ax2.plot(t, wave, c='red', label="%.3f" % cdistances[col])
        plt.legend(handles=rline)
        f2_ax2 = fig2.add_subplot(gs[2:, 2 * col + 2:2 * col + 4])
        col2 = -1 - col
        f2_ax2.plot(t, cluster[col2])
        wave = pltaux.wave(cluster[col2], cshifts[col2], centroid_length)
        rline = f2_ax2.plot(t, wave, c='red', label="%.3f" % cdistances[col2])
        plt.legend(handles=rline)

    plt.suptitle('First row: 5 closest waves. Second row: 5 farthest waves.')
    fig2.tight_layout()
    fname = 'centroid%d_5waves_%dms_%s_%s.pdf'\
        % (ic, 1e3 * centroid_length / srate, metric, init)
    fig2.savefig(os.path.join(images_path, fname), bbox_inches='tight')

pat = 'centroid*_5waves_%dms_%s_%s.pdf'\
    % (1e3 * centroid_length / srate, metric, init)
pat = os.path.join(images_path, pat)
fname='centroid_5waves_%dms_%s_%s.pdf'\
    % (1e3 * centroid_length / srate, metric, init)
fname = os.path.join(images_path, fname)
subprocess.call("pdfunite " + pat + " " + fname, shell=True)

# %%
