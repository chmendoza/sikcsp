"""
Train and test the model

Learn codebooks using best hyperparameters (see crossval.py) and test perfomance on test data.
"""

#%%
import os
import sys
import numpy as np
from sklearn.metrics import matthews_corrcoef, precision_recall_fscore_support

import time
import yaml
from argparse import ArgumentParser

# Add path to package directory to access main module using absolute import
sys.path.insert(0,
                os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             '..')))


import shift_kmeans.shift_kmeans as sikmeans
from kcsp import utils, configtrain, classifier

def minusone(x): return x - 1  # Matlab index starts at 1, Python at 0


# Parse command-line arguments
parser = ArgumentParser()
parser.add_argument("-k", "--nclusters", dest="n_clusters",
                    type=int, help="Number of clusters")
parser.add_argument("-P", "--centroid-length", type=int,
                    dest="centroid_length", help="Length of cluster centroids")
parser.add_argument("--patient", dest="patient", help="Patient label")
parser.add_argument("--band", dest="band",
                    type=int, help="Spectral band id")
parser.add_argument("--classifier", dest="clf", help="Classifier id")
parser.add_argument("-C", type=float, dest="regfactor", default=1,
                    help="Regularization factor for SVM and logistic\
                        regression classifiers")
parser.add_argument("-n", "--n-cpus", dest="n_cpus", type=int,
                    default=1, help="Number of SLURM CPUS")

args = parser.parse_args()
n_cpus = args.n_cpus
band = args.band

os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

#%% Read paramaters
params = configtrain.configure_experiment(args)

print('=========== Configuration =============')
print(yaml.dump(params, sort_keys=False))
print('=======================================')
patient_dir = params['Patient dir']
wfname = params['Filenames']['CSP filters']
dfname = params['Filenames']['Data indices']
rfname = params['Filenames']['Results']
winlen = params['Data']['Window length']
seglen = params['Data']['Segment length']
i_csp = params['Data']['Index of CSP filters']
i_csp = list(map(int, i_csp.split()))  # str->list of ints
metric = params['Algorithm']['metric']
init = params['Algorithm']['init']
n_runs = params['Algorithm']['n_runs']
k = params['Algorithm']['Num. of clusters']
P = params['Algorithm']['Centroid length']
clfdict = params['Classifier']

#%% Random generator
# entropy.txt has the entropy used to make the
# crossvalidation splits (folds), which are the same used
# to compute the intermediate crossval codebooks and to do
# run the full crossval with a given classifier.
kP_dir = os.path.join(patient_dir, f'band{band}_k{k}-{k}_P{P}-{P}')
rng_path = os.path.join(kP_dir, 'entropy.txt')
with open(rng_path, 'r') as f:
    seed = int(f.read().strip())
    seed = np.random.SeedSequence(entropy=seed)
    print('Initial random seed: %s' % seed.entropy)

rng = np.random.default_rng(seed)

#%% Get the CSP filters
wpath = os.path.join(patient_dir, wfname)
W = utils.loadmat73(wpath, 'W')
i_csp = np.array(i_csp)
W = W[:, i_csp]

#%% Extract data and apply CSP filter
conditions = ['preictal', 'interictal']
X = [[0]*2 for x in range(2)]  # 2x2 list
tic = time.perf_counter()
set_labels = ['train', 'test']
for i_condition, condition in enumerate(conditions):    
    # file names and start indices of preictal segments
    dirpath = os.path.join(patient_dir, condition)
    fpath = os.path.join(dirpath, dfname)
    for i_set, set_label in enumerate(set_labels):         
        i_start = utils.loadmat73(fpath, '%s_indices' % set_label)
        i_start = utils.apply2list(i_start, np.squeeze)
        i_start = utils.apply2list(i_start, minusone)    

        dfnames = utils.loadmat73(fpath, '%s_names' % set_label)

        # Extract data and apply CSP filter
        X[i_condition][i_set] = utils.getCSPdata(
            dirpath, dfnames, i_start, seglen, W, n_cpus=n_cpus)

toc = time.perf_counter()
print("Data gathered and filtered after %0.4f seconds" % (toc - tic))

# Split training segments into smaller windows
Xtrain = [0] * 2
Xtrain[0] = utils.splitdata(X[0][0][0], winlen) # preictal,training,CSP-1
Xtrain[1] = utils.splitdata(X[1][0][1], winlen) # interictal,training,CSP-C

# Training -- Build codebooks
C1, nu1, tau1, d1, sumd1, n_iter1 = sikmeans.shift_invariant_k_means(
    Xtrain[0], k, P, metric=metric, init=init, n_init=n_runs, rng=rng, verbose=True)
C2, nu2, tau2, d2, sumd2, n_iter2 = sikmeans.shift_invariant_k_means(
    Xtrain[1], k, P, metric=metric, init=init, n_init=n_runs, rng=rng, verbose=True)

clfname = clfdict['name']
if 'params' in clfdict:
    clfparams = clfdict['params']
else:
    clfparams = dict()

# Reshape data to extract features for classification. Concatenate preictal and
# interictal data along the codebook axis (0). Xtrain[0] is the data that was
# filtered with CSP-1 and is to be clustered using C1, the preictal codebook.
Xtrain = utils.concatenate(X[0][0], X[1][0], winlen)

# Training data for classifier
train_sample = classifier.extract_features(\
    Xtrain, (C1,C2), metric=metric, clfname=clfname)

# True label vector
s_true = np.r_[np.ones(X[0][0].shape[1], dtype=int),
               2 * np.ones(X[1][0].shape[1], dtype=int)]

# Instantiate and train classifier
clf = classifier.fit(train_sample, s_true,
                     clfname=clfname, clfparams=clfparams)

# ==== Training ends, Testing begins ====

# Prepare data
Xtest = utils.concatenate(X[0][1], X[1][1], winlen)
# Test data for classifier
test_sample = classifier.extract_features(
         Xtest, (C1,C2), metric=metric, clfname=clfname)

# Predict
s_hat = clf.predict(test_sample)
# True label vector
s_true = np.r_[np.ones(X[0][1].shape[1], dtype=int),
               2 * np.ones(X[1][1].shape[1], dtype=int)]

MCC = matthews_corrcoef(s_true, s_hat)
precision, recall, F1, _ = precision_recall_fscore_support(s_true, s_hat)
confmat = utils.confusion_matrix(s_true, s_hat)

rpath = os.path.join(patient_dir, rfname)
with open(rpath, 'wb') as f:
    np.savez(f, C1=C1, C2=C2, MCC=MCC, F1=F1, precision=precision, recall=recall, confmat=confmat)
