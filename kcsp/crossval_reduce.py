"""
Print the cross-validation results for a given subject and spectral band

Print the results from all the binary classifiers where experiments were run, and print which were the best hyper-parameters.
"""

#%%
import os
import numpy as np
from argparse import ArgumentParser
import glob
import re

# Parse command-line arguments
parser = ArgumentParser()
parser.add_argument("-d", "--dir", dest="data_dir",
                    help="Path to folder with cross-validation result files")
parser.add_argument("-b", "--band", dest="band", type=int,
                    help="Spectral band chosen")


args = parser.parse_args()

data_dir = args.data_dir
band = args.band

os.chdir(data_dir)

n_clusters = np.r_[4, 8, 16, 32, 64, 128]
centroid_length = np.r_[30, 40, 60, 120, 200, 350]
MCC = np.zeros((6, 6))
best_MCC_max = 0
best_MCC = np.zeros((6, 6))
best_clf, best_C, best_k, best_P = None, None, None, None

filepat = 'averageMCC_band\d_clf(?P<clf>\d+)(_C(?P<C>.+))*[.]npy'
filepat = re.compile(filepat)

filelistpat = f'averageMCC_band{band}*.npy'
filelist = glob.glob(filelistpat)
filelist.sort()

for file in filelist:    
    match = filepat.match(file)
    patdict = match.groupdict()
    C = patdict['C']
    clf = patdict['clf']

    MCC = np.load(file)
    imax = np.unravel_index(np.argmax(MCC), MCC.shape)
    MCC_max = MCC[imax]
    k = n_clusters[imax[0]]
    P = centroid_length[imax[1]]

    if C:
        print(f'Classifier={clf}, C={C}, '
              f'best (k,P,MCC)={k},{P},{MCC_max}')
    else:
        print(f'Classifier={clf}, '
              f'best (k,P,MCC)={k},{P},{MCC_max}')

    if MCC_max > best_MCC_max:
        best_MCC_max = MCC_max
        best_k = k
        best_P = P
        best_MCC = MCC
        best_clf = clf
        if C:
            best_C = C
        else:
            best_C = None

print(f'Best overall (k,P,MCC): {best_k},{best_P}, {best_MCC_max}')
print(f'Best classifier: {best_clf}')
if best_C:
    print(f'Best hyperparameter value: {best_C}')
print(f'Cross-validated average MCC on k-P hyperparameter grid:')
with np.printoptions(precision=3, floatmode='fixed'):
        print(best_MCC)
