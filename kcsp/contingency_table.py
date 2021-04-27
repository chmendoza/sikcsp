"""
Use shift-invariant k-means on top k CSP waves
"""
# %%
import numpy as np
import matplotlib.pyplot as plt
from  matplotlib.backends.backend_pdf import PdfPages

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


#%% Get the CSP filters
wpath = os.path.join(patient_dir, wfname)
W = utils.loadmat73(wpath, 'W')

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
            dirpath, dfnames, i_start, seglen, W[:, i_condition], n_cpus=n_cpus)

# Split training segments into smaller windows
Xtest = [0]*2
n_samples = np.zeros(2)
Xtest[0] = utils.splitdata(X[0], winlen)  # preictal
Xtest[1] = utils.splitdata(X[1], winlen)  # interictal
n_samples[0] = Xtest[0].shape[0]
n_samples[1] = Xtest[1].shape[0]

# Load and concatenate learned codebooks
cpath = os.path.join(patient_dir, cfname)
data = np.load(cpath)
C1, C2 = data['C1'], data['C2']
C = np.concatenate((C1,C2), axis=0)
n_centroids = C.shape[0]
counts = np.zeros((n_centroids, 2))  # counts of centroids per condition

metric = 'cosine'
XX = None
nu_1, _, _ = si_pairwise_distances_argmin_min(
    Xtest[0], C, metric, XX)
nu_2, _, _ = si_pairwise_distances_argmin_min(
    Xtest[1], C, metric, XX)
uniq_1, cnts_1 = np.unique(nu_1, return_counts=True)
uniq_2, cnts_2 = np.unique(nu_2, return_counts=True)

counts[uniq_1, 0] = cnts_1
counts[uniq_2, 1] = cnts_2

#np.savetxt(os.path.join(patient_dir, 'centroid_counts.csv'),
#           counts, delimiter=",")

rpath = os.path.join(patient_dir, rfname)
with open(rpath, 'wb') as f:
    np.savez(f, n_samples=n_samples, counts=counts)