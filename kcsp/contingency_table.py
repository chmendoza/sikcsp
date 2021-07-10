"""
Use shift-invariant k-means on top k CSP waves
"""
# %%
import numpy as np

import os
import sys
import yaml
from argparse import ArgumentParser

# Add path to package directory to access main module using absolute import
# This is needed because this is meant to be executed as an script
# See https://stackoverflow.com/a/11537218/4292705
sys.path.insert(0,
                os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from shift_kmeans.wrappers import si_pairwise_distances_argmin_min, si_row_norms
from kcsp import utils

def minusone(x): return x - 1  # Matlab index starts at 1, Python at 0


# Parse command-line arguments
parser = ArgumentParser()
parser.add_argument("-f", "--config-file", dest="confpath",
                    help="YAML configuration file")

parser.add_argument("-n", "--n-cpus", dest="n_cpus", type=int,
                    default=1, help="Number of SLURM CPUS")

args = parser.parse_args()
n_cpus = args.n_cpus

os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

#%% Read paramaters
confpath = args.confpath

with open(confpath, 'r') as yamlfile:
    params = yaml.load(yamlfile, Loader=yaml.FullLoader)

print('=========== Configuration =============')
print(yaml.dump(params, sort_keys=False))
print('=======================================')
patient_dir = params['Patient dir']
wfname = params['Filenames']['CSP filters']
dfname = params['Filenames']['Data indices']
cfname = params['Filenames']['Codebooks']
rfname = params['Filenames']['Results']
winlen = params['Data']['Window length']
seglen = params['Data']['Segment length']
i_csp = params['Data']['Index of CSP filters']


#%% Get the CSP filters
wpath = os.path.join(patient_dir, wfname)
W = utils.loadmat73(wpath, 'W')
i_csp = np.array(i_csp)
W = W[:, i_csp]

i_start = [0]*2
dfnames = [0]*2

#%% Extract data and apply CSP filter
conditions = ['preictal', 'interictal']
X = [0]*2  # 1x2 list
for i_condition, condition in enumerate(conditions):
    # file names and start indices of preictal segments
    dirpath = os.path.join(patient_dir, condition)
    fpath = os.path.join(dirpath, dfname)
    i_start = utils.loadmat73(fpath, 'test_indices')
    i_start = utils.apply2list(i_start, np.squeeze)
    i_start = utils.apply2list(i_start, minusone)

    dfnames = utils.loadmat73(fpath, 'test_names')

    # Extract data and apply CSP filter
    X[i_condition] = utils.getCSPdata(
            dirpath, dfnames, i_start, seglen, W, n_cpus=n_cpus)


# Concatenate preictal and interictal test data filtered with CSP-1
X1test = np.concatenate((X[0][0], X[1][0]), axis=0)
# Concatenate preictal and interictal test data filtered with CSP-C
X2test = np.concatenate((X[0][1], X[1][1]), axis=0)

# Split test segments into smaller windows and stack them vertically
X1test = utils.splitdata(X1test, winlen)
X2test = utils.splitdata(X2test, winlen)

# Load learned codebooks
cpath = os.path.join(patient_dir, cfname)
data = np.load(cpath)
C1, C2 = data['C1'], data['C2']

n_centroids = C1.shape[0]  # Assume C1 and C2 have same number of centroids
test_counts = np.zeros((n_centroids, 2))  # counts of centroids per codebook

metric = 'cosine'
XX = None
nu_1, _, _ = si_pairwise_distances_argmin_min(
    X1test, C1, metric, XX)
nu_2, _, _ = si_pairwise_distances_argmin_min(
    X2test, C2, metric, XX)
uniq_1, cnts_1 = np.unique(nu_1, return_counts=True)
uniq_2, cnts_2 = np.unique(nu_2, return_counts=True)

# Number of test samples assigned to centroids from each codebook
test_counts[uniq_1, 0] = cnts_1
test_counts[uniq_2, 1] = cnts_2

# Split test segments into smaller windows
X1test = utils.splitdata(X[0][0], winlen)  # preictal, filtered with CSP-1
X2test = utils.splitdata(X[1][1], winlen)  # interictal, filtered with CSP-C

n_samples = np.zeros(2)
n_samples[0], n_samples[1] = X1test.shape[0], X2test.shape[0]
tot_samples = np.sum(n_samples)

# Concatenate learned codebooks
C = np.concatenate((C1,C2), axis=0)
n_centroids = C.shape[0]
master_counts = np.zeros((n_centroids, 2))  # counts of centroids per condition

nu_1, _, _ = si_pairwise_distances_argmin_min(
    X1test, C, metric, XX)
nu_2, _, _ = si_pairwise_distances_argmin_min(
    X2test, C, metric, XX)
uniq_1, cnts_1 = np.unique(nu_1, return_counts=True)
uniq_2, cnts_2 = np.unique(nu_2, return_counts=True)

master_counts[uniq_1, 0] = cnts_1
master_counts[uniq_2, 1] = cnts_2

# Build contingency table and compute chi-squared test
chi_squared = np.zeros(n_centroids)
for i_centroid in range(n_centroids):
    contable = np.zeros((2,2))
    contable[0, 0] = master_counts[i_centroid, 0] # centroid present, preictal
    contable[0, 1] = master_counts[i_centroid, 1] # present, interictal
    contable[1, 0] = n_samples[0] - contable[0, 0] # absent, preictal
    contable[1, 1] = n_samples[1] - contable[0, 1]
   
    p_i = np.sum(contable, axis=1) / tot_samples
    p_j = np.sum(contable, axis=0) / tot_samples
    p_ip_j = p_i.reshape(-1,1) * p_j.reshape(1,-1)

    chi_squared[i_centroid] = tot_samples * np.sum(\
        p_ip_j * ((contable/tot_samples - p_ip_j)/p_ip_j)**2)

rpath = os.path.join(patient_dir, rfname)
with open(rpath, 'wb') as f:
    np.savez(f, test_counts=test_counts,\
        master_counts=master_counts, chi_squared=chi_squared)