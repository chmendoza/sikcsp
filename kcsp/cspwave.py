#%%
import os
import sys
import numpy as np
import time
from numpy.core.fromnumeric import trace
import yaml
from argparse import ArgumentParser
import tracemalloc

# Add path to package directory to access main module using absolute import
sys.path.insert(0,
                os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             '..')))

from kcsp import utils, bayes
import shift_kmeans.shift_kmeans as sikmeans

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
# confdir = '/home/cmendoza/MEGA/Research/software/shift_kmeans/kcsp/config'
# confname = 'crossval_kcsp_dmonte7_band2_regular_k8-64_P30-40.yaml'
# confpath = os.path.join(confdir, 'HUP078', confname)
confpath = args.confpath

with open(confpath, 'r') as yamlfile:
    params = yaml.load(yamlfile, Loader=yaml.FullLoader)

print('=========== Configuration =============')
print(yaml.dump(params, sort_keys=False))
print('=======================================')
patient_dir = params['Patient dir']
foldname = params['Fold indices dir']
ffname = params['Filenames']['Fold indices']
wfname = params['Filenames']['CSP filters']
dfname = params['Filenames']['Data indices']
rfname = params['Filenames']['Results']
n_folds = params['Crossvalidation']['n_folds']
i_fold = params['Crossvalidation']['i_fold']
winlen = params['Data']['Window length']
seglen = params['Data']['Segment length']
metric = params['Algorithm']['metric']
init = params['Algorithm']['init']
n_runs = params['Algorithm']['n_runs']
k1, k2 = params['Algorithm']['n_clusters']
P1, P2 = params['Algorithm']['centroid_length']
init_seed = params['Algorithm']['rng_seed']

tracemalloc.start() # start memory tracing
#%% Get the CSP filters
wpath = os.path.join(patient_dir, wfname)
W = utils.loadmat73(wpath, 'W')

#%% Extract data and apply CSP filter
i_set = 0  # 0: training, 1: testing
conditions = ['preictal', 'interictal']
X = [0]*2
tic = time.perf_counter()
for i_condition, condition in enumerate(conditions):
    # file names and start indices of preictal segments
    dirpath = os.path.join(patient_dir, condition)
    fpath = os.path.join(dirpath, dfname)
    i_start = utils.loadmat73(fpath, 'train_indices')
    i_start = utils.apply2list(i_start, np.squeeze)
    i_start = utils.apply2list(i_start, minusone)    
    dfnames = utils.loadmat73(fpath, 'train_names')

    # Extract data and apply CSP filter
    X[i_condition] = utils.getCSPdata(
        dirpath, dfnames, i_start, seglen, W[:, i_condition], n_cpus=n_cpus)

toc = time.perf_counter()
print("Data gathered and filtered after %0.4f seconds" % (toc - tic))

## Get indices of one cross-validation fold
ffpath = os.path.join(patient_dir, foldname, ffname)
with np.load(ffpath) as indices:
    train1, test1 = indices['train1'], indices['test1']
    train2, test2 = indices['train2'], indices['test2']

# Split (croosval) training segments into smaller windows
X1train = utils.splitdata(X[0][train1], winlen) # preictal
X2train = utils.splitdata(X[1][train2], winlen)  # interictal

#%% Random generator
seed = np.random.SeedSequence(init_seed)
print('Initial random seed: %s' % seed.entropy)
rng = np.random.default_rng(seed)
if init_seed is None:
    print('Saving initial seed to disk...')
    params['init_seed'] = seed.entropy
    with open(confpath, 'w') as yamfile:
        yaml.dump(params, yamfile, sort_keys=False)

tic = time.perf_counter()
print('Fold %d out of %d' % (i_fold+1, n_folds))
# Training
C1, nu1, tau1, d1, sumd1, n_iter1 = sikmeans.shift_invariant_k_means(
    X1train, k1, P1, metric=metric, init=init, n_init=n_runs, rng=rng, verbose=True)
C2, nu2, tau2, d2, sumd2, n_iter2 = sikmeans.shift_invariant_k_means(
    X2train, k2, P2, metric=metric, init=init, n_init=n_runs, rng=rng, verbose=True)

# Estimate posterior
nu_11, nu_12 = bayes.cluster_assignment(X1train, C1, C2, metric=metric)
nu_21, nu_22 = bayes.cluster_assignment(X2train, C1, C2, metric=metric)
nu = nu_11, nu_12, nu_21, nu_22
N1, N2 = X[0].shape[0], X[1].shape[0] # Number of windows
p_C = bayes.likelihood(nu, N=(N1, N2), k=(k1, k2))  # (2,k1,k2)
p_S = np.zeros(2)
p_S[0] = N1/(N1+N2)
p_S[1] = N2/(N1+N2)

# Split (croosval) test segments into smaller windows
X1test = utils.splitdata(X[0][test1], winlen, keep_dims=False)  # preictal
X2test = utils.splitdata(X[1][test2], winlen, keep_dims=False)  # interictal
del X  # Free memory used by X

# Concatenate segments from both classes
Xtest = np.concatenate((X1test, X2test), axis=0)
M_bar1, M_bar2 = X1test.shape[0], X2test.shape[0]  # Number of segments
del X1test, X2test

# True label vector
s = np.r_[np.ones(M_bar1, dtype=int), 2 * np.ones(M_bar2, dtype=int)]
M_bar = M_bar1 + M_bar2 # Total number of test segments

# Initialize estimated label, s_hat
s_hat = np.zeros(M_bar, dtype=int)

# Predict class label using MAP
for i_segment in np.arange(M_bar):
    nu_1, nu_2 = bayes.cluster_assignment(
        Xtest[i_segment], C1, C2, metric=metric)        
    M = Xtest[i_segment].shape[0] # Num. of windows on a segment
    # Log-posterior:
    logpost = M*np.log(p_S) + \
            np.sum(np.log(p_C[:, nu_1, nu_2]), axis=1)
    # add 1 to convert array element index to class label
    s_hat[i_segment] = np.argmax(logpost) + 1

# Compute Matthews correlation coefficient (MCC)
confmat = utils.confusion_matrix(s, s_hat)
tp, fn, fp, tn = confmat.flatten()
MCC = (tp * tn - fp * fn)/\
    np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))

toc = time.perf_counter()
print("Fold processed after %0.4f seconds" % (toc - tic))
print('MCC: %.3f' % MCC)

rpath = os.path.join(patient_dir, foldname, rfname)
with open(rpath, 'wb') as f:
    np.save(f, MCC)

snapshot = tracemalloc.take_snapshot()
utils.display_top(snapshot)
