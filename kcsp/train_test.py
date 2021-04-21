#%%
import os
import sys
import numpy as np
import time
from numpy.core.fromnumeric import trace
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
rfname = params['Filenames']['Results']
winlen = params['Data']['Window length']
seglen = params['Data']['Segment length']
metric = params['Algorithm']['metric']
init = params['Algorithm']['init']
n_runs = params['Algorithm']['n_runs']
k = params['Algorithm']['n_clusters']
P = params['Algorithm']['centroid_length']
init_seed = params['Algorithm']['rng_seed']

#%% Get the CSP filters
wpath = os.path.join(patient_dir, wfname)
W = utils.loadmat73(wpath, 'W')

i_start = [0]*2
dfnames = [0]*2

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
            dirpath, dfnames, i_start, seglen, W[:, i_condition], n_cpus=n_cpus)

toc = time.perf_counter()
print("Data gathered and filtered after %0.4f seconds" % (toc - tic))

# Split training segments into smaller windows
X1train = utils.splitdata(X[0][0], winlen)  # preictal
X2train = utils.splitdata(X[1][0], winlen)  # interictal

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

# Training
C1, nu1, tau1, d1, sumd1, n_iter1 = sikmeans.shift_invariant_k_means(
    X1train, k, P, metric=metric, init=init, n_init=n_runs, rng=rng, verbose=True)
C2, nu2, tau2, d2, sumd2, n_iter2 = sikmeans.shift_invariant_k_means(
    X2train, k, P, metric=metric, init=init, n_init=n_runs, rng=rng, verbose=True)

# Syntax for cluster assignments: nu_rs. r is the index of the codebook (C1 or C2). s is the index of the window (or segment) class, preictal (s=1) or interictal (s=2).
nu_11, nu_21 = bayes.cluster_assignment(X1train, C1, C2, metric=metric)
nu_12, nu_22 = bayes.cluster_assignment(X2train, C1, C2, metric=metric)
nu = nu_11, nu_21, nu_12, nu_22
N1, N2 = X1train.shape[0], X2train.shape[0]  # Number of windows
p_C = bayes.likelihood(nu, N=(N1, N2), k=(k, k))  # (2,k1,k2)
p_S = np.zeros(2)
p_S[0] = N1/(N1+N2)
p_S[1] = N2/(N1+N2)

# Split test segments into smaller windows
X1test = utils.splitdata(X[0][1], winlen, keep_dims=False)  # preictal
X2test = utils.splitdata(X[1][1], winlen, keep_dims=False)  # interictal
del X  # Free memory used by X

# Concatenate segments from both classes
Xtest = np.concatenate((X1test, X2test), axis=0)
M_bar1, M_bar2 = X1test.shape[0], X2test.shape[0]  # Number of segments
del X1test, X2test

# True label vector
s = np.r_[np.ones(M_bar1, dtype=int), 2 * np.ones(M_bar2, dtype=int)]
M_bar = M_bar1 + M_bar2  # Total number of test segments

# Initialize estimated label, s_hat
s_hat = np.zeros((2, M_bar), dtype=int)

# Predict class label using MAP and ML
likelihood_weights = np.ones_like(p_S) * 0.5
for i_segment in np.arange(M_bar):
    nu_1, nu_2 = bayes.cluster_assignment(
        Xtest[i_segment], C1, C2, metric=metric)
    M = Xtest[i_segment].shape[0]  # Num. of windows on a segment
    # Log-posterior:
    evalp_C = p_C[:, nu_1, nu_2]  # Eval likelihood at cluster assingment pairs
    # Avoid divide-by-zero warning:
    logp_C = np.full((2, nu_1.size), np.NINF)
    np.log(evalp_C, out=logp_C, where=evalp_C > 0)
    logMAP = M*np.log(p_S) + np.sum(logp_C, axis=1)
    logML = M*np.log(likelihood_weights) + np.sum(logp_C, axis=1)
    # add 1 to convert array element index to class label
    s_hat[0, i_segment] = np.argmax(logMAP) + 1
    s_hat[1, i_segment] = np.argmax(logML) + 1

# Compute Matthews correlation coefficient (MCC)
# Assume that the positive class (preictal) is the minority class and that the negative class (interictal) is the majority class. Also, there are always examples from both classes.
MCC = np.zeros(2)
F1 = np.zeros(2)
precision = np.zeros(2)
recall = np.zeros(2)
accu = np.zeros(2)
confmat = [0]*2
for ii in np.arange(2):
    confmat[ii] = utils.confusion_matrix(s, s_hat[ii, :])
    tp, fn, fp, tn = confmat[ii].flatten()
    print('TP: %5d\t FN: %5d' % (tp, fn))
    print('FP: %5d\t TN: %5d' % (fp, tn))
    if (tp == 0 and fp == 0) or (tn == 0 and fn == 0):
        MCC[ii] += 0
        if tp == 0 and fp == 0:
            precision[ii] = 0
        else:
            precision[ii] = tp / (tp + fp)
    else:
        precision[ii] = tp / (tp + fp)
        MCC[ii] += (tp * tn - fp * fn) /\
            np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))

    # Compute F1-score
    recall[ii] = tp / (tp + fn)
    if recall[ii] == 0:
        F1[ii] = 0
    else:
        F1[ii] += 2 * precision[ii] * recall[ii] / \
            (precision[ii] + recall[ii])  # harmonic mean
    accu[ii] = (tp+tn) / M_bar

toc = time.perf_counter()
print("Fold processed after %0.4f seconds" % (toc - tic))
print('MCC with MAP: %.3f' % MCC[0])
print('MCC with ML: %.3f' % MCC[1])

rpath = os.path.join(patient_dir, rfname)
with open(rpath, 'wb') as f:
    np.savez(f, C1=C1, C2=C2, MCC=MCC, F1=F1, accu=accu,
             precision=precision, recall=recall, confmat=confmat)
