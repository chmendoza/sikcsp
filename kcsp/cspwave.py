#%%
import os
import sys
import numpy as np
import time
import yaml
from argparse import ArgumentParser

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

N1, N2 = X[0].shape[0], X[1].shape[0]
misclass = 0

## Get indices of one cross-validation fold
ffpath = os.path.join(patient_dir, foldname, ffname)
with np.load(ffpath) as indices:
    train1, test1 = indices['train1'], indices['test1']
    train2, test2 = indices['train2'], indices['test2']

# Split (croosval) training segments into smaller windows and shuffle at random
X1train = X[0][train1]


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
    X[0][train1], k1, P1, metric=metric, init=init, n_init=n_runs, rng=rng, verbose=True)
C2, nu2, tau2, d2, sumd2, n_iter2 = sikmeans.shift_invariant_k_means(
    X[1][train2], k2, P2, metric=metric, init=init, n_init=n_runs, rng=rng, verbose=True)

# Estimate posterior
nu = bayes.cluster_assignment(
    X[0][train1], X[1][train2], C1, C2, metric=metric)
p_J = bayes.likelihood(nu, N=(N1, N2), k=(k1, k2))  # (2,k1,k2)
p_S = np.zeros((2, 1, 1))
p_S[0] = N1/(N1+N2)
p_S[1] = N2/(N1+N2)
posterior = p_S * p_J  # ~ P(S=s | J1 = j, J2 = l)

# Compute misclassification rate using MAP
N1, N2 = X[0][test1].shape[0], X[1][test2].shape[0]
nu = bayes.cluster_assignment(
    X[0][test1], X[1][test2], C1, C2, metric=metric)
# Posterior when data (cluster assignments) come from class s=1
# nu[0] are cluster assignments using C1
# nu[1] are cluster assignments using C2
pp1 = posterior[:, nu[0], nu[1]]  # (2, N1).
pp2 = posterior[:, nu[2], nu[3]]
# for each pair assignment pair (j,l), find the value of s (row index) with AP. Row 0 is class 1.
I1 = np.argmax(pp1, axis=0)
I2 = np.argmax(pp2, axis=0)
misclass = np.sum(I1 != 0)
misclass += np.sum(I2 != 1)
misclass /= (N1 + N2)


toc = time.perf_counter()
print("Fold processed after %0.4f seconds" % (toc - tic))
print('Misclassification: %.3f' % misclass)

rpath = os.path.join(patient_dir, rfname)
with open(rpath, 'wb') as f:
    np.save(f, misclass)
