#%%
import os, sys, re
import numpy as np
import time
import yaml
from argparse import ArgumentParser

# Add path to package directory to access main module using absolute import
sys.path.insert(0,
                os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             '..')))

import shift_kmeans.shift_kmeans as sikmeans
from kcsp import utils, bayes

def minusone(x): return x - 1  # Matlab index starts at 1, Python at 0


# Parse command-line arguments
parser = ArgumentParser()
parser.add_argument("-d", "--dir", dest="data_dir",
                    help="Path to folder with cross-validation result files")
parser.add_argument("-b", "--band", dest="band", type=int,
                    help="Spectral band chosen")

args = parser.parse_args()

data_dir = args.data_dir
band = args.band

n_clusters = [4, 8, 16, 32, 64, 128]
centroid_lengths = [30, 40, 60, 120, 200, 350]
method = 'regular'
n_folds = 10

MCC = np.zeros((len(n_clusters), len(centroid_lengths)))
for i_k, k in enumerate(n_clusters):
    for i_P, P in enumerate(centroid_lengths):        
        foldname = 'band%d_%s_k%d-%d_P%d-%d' % (
            band, method, k, k, P, P)
        for i_fold in range(n_folds):
            rfname = 'crossval_fold%d_band%d_%s_k%d-%d_P%d-%d.npz' % (
                i_fold, band, method, k, k, P, P)
            fpath = os.path.join(data_dir, foldname, rfname)
            with np.load(fpath) as data:
                MCC[i_k, i_P] += data['MCC'][1] # Maximum likelihood-based score
        MCC[i_k, i_P] /= n_folds
        print('Average MCC for k = %d and P = %d: %.3f' \
            % (k, P, MCC[i_k, i_P]))

with np.printoptions(precision=3, floatmode='fixed'):            
    print(MCC)

np.savetxt(os.path.join(data_dir, 'averageMCC.csv'), MCC, delimiter=",")

