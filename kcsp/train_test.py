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
i_csp = params['Data']['Index of CSP filters']
metric = params['Algorithm']['metric']
init = params['Algorithm']['init']
n_runs = params['Algorithm']['n_runs']
k = params['Algorithm']['n_clusters']
P = params['Algorithm']['centroid_length']
init_seed = params['Algorithm']['rng_seed']

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
n_seg = np.zeros(2, dtype=int)
n_csp, n_seg[0], seglen = X[0][0].shape
n_seg[1] = X[1][0].shape[1]
n_win_per_seg = seglen // winlen
Xtrain = [0] * 2
for s in range(2):  # class segment
    n_win = n_win_per_seg * n_seg[s]
    Xtrain[s] = np.zeros((n_csp, n_win, winlen))
    for r in range(2):  # CSP filter
        Xtrain[s][r] = utils.splitdata(X[s][0][r], winlen)

#%% Random generator
seed = np.random.SeedSequence(init_seed)
print('Initial random seed: %s' % seed.entropy)
rng = np.random.default_rng(seed)
if init_seed is None:
    print('Saving initial seed to disk...')
    params['init_seed'] = seed.entropy
    with open(confpath, 'w') as yamfile:
        yaml.dump(params, yamfile, sort_keys=False)

# Training
C1, nu1, tau1, d1, sumd1, n_iter1 = sikmeans.shift_invariant_k_means(
    Xtrain[0][0], k, P, metric=metric, init=init, n_init=n_runs, rng=rng, verbose=True)
C2, nu2, tau2, d2, sumd2, n_iter2 = sikmeans.shift_invariant_k_means(
    Xtrain[1][1], k, P, metric=metric, init=init, n_init=n_runs, rng=rng, verbose=True)

# Estimate likelihood and prior
p_C = bayes.likelihood(Xtrain, (C1, C2), metric=metric)  # (2,k1,k2)
# Number of windows
N1, N2 = Xtrain[0].shape[1], Xtrain[1].shape[1]
p_S = np.zeros(2)
p_S[0] = N1/(N1+N2)
p_S[1] = N2/(N1+N2)

# ------ Training ends, Testing begins ----------

# Concatenate preictal and interictal test data filtered with CSP-1
X1test = np.concatenate((X[0][1][0], X[1][1][0]), axis=0)
# Concatenate preictal and interictal test data filtered with CSP-C
X2test = np.concatenate((X[0][1][1], X[1][1][1]), axis=0)

# Split test segments into smaller windows. This creates a 3D
# matrix with the smaller windows (2D matrices) lying in the last two
# indices and the first axis indexing the segments.
X1test = utils.splitdata(X1test, winlen, keep_dims=False)
X2test = utils.splitdata(X2test, winlen, keep_dims=False)

# Initialize estimated label, s_hat
n_seg = X1test.shape[0]  # Number of test segments
s_hat = np.zeros((2, n_seg), dtype=int)

# Predict class label using MAP and ML
likelihood_weights = np.ones_like(p_S) * 0.5
for i_segment in np.arange(n_seg):
    
    # Cluster assingments for windows in a segment of unknown class,
    # filtered with CSP-1 and using the preictal codebook:
    nu_1 = bayes.cluster_assignment(X1test[i_segment], C1, metric=metric)
    # Cluster assingments of data filtered with CSP-C and using the
    # interictal codebook:
    nu_2 = bayes.cluster_assignment(X2test[i_segment], C2, metric=metric)

    # Compute log-likelihood
    evalp_C = p_C[:, nu_1, nu_2]  # Eval learned likelihood
    logp_C = np.full((2, nu_1.size), np.NINF)  # Avoid divide-by-zero warning
    np.log(evalp_C, out=logp_C, where=evalp_C > 0)
    M = X1test[i_segment].shape[0]  # Num. of windows on a segment
    logMAP = M*np.log(p_S) + np.sum(logp_C, axis=1)
    logML = M*np.log(likelihood_weights) + np.sum(logp_C, axis=1)

    # Find MAP and ML estimate
    s_hat[0, i_segment] = np.argmax(logMAP) + 1  # array index to class label
    s_hat[1, i_segment] = np.argmax(logML) + 1

# True label vector
s = np.r_[np.ones(X[0][1].shape[1], dtype=int),
          2 * np.ones(X[1][1].shape[1], dtype=int)]

# Compute Matthews correlation coefficient (MCC)
# Assume that the positive class (preictal) is the minority class and that the negative class (interictal) is the majority class. Also, there are always examples from both classes.
MCC = np.zeros(2)
F1 = np.zeros(2)
precision = np.zeros(2)
recall = np.zeros(2)
accu = np.zeros(2)
confmat = [0]*2
for ii in np.arange(2):  # loop over {MAP, ML}
    confmat[ii] = utils.confusion_matrix(s, s_hat[ii, :])
    tp, fn, fp, tn = confmat[ii].flatten()
    print('TP: %5d\t FN: %5d' % (tp, fn))
    print('FP: %5d\t TN: %5d' % (fp, tn))
    if (tp == 0 and fp == 0) or (tn == 0 and fn == 0):
        MCC[ii] = 0
        if tp == 0 and fp == 0:
            precision[ii] = 0
        else:
            precision[ii] = tp / (tp + fp)
    else:
        precision[ii] = tp / (tp + fp)
        MCC[ii] = (tp * tn - fp * fn) /\
            np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))

    # Compute F1-score
    recall[ii] = tp / (tp + fn)
    if recall[ii] == 0:
        F1[ii] = 0
    else:
        F1[ii] = 2 * precision[ii] * recall[ii] / \
            (precision[ii] + recall[ii])  # harmonic mean
    accu[ii] = (tp+tn) / n_seg

print('MCC with MAP: %.3f' % MCC[0])
print('MCC with ML: %.3f' % MCC[1])

rpath = os.path.join(patient_dir, rfname)
with open(rpath, 'wb') as f:
    np.savez(f, C1=C1, C2=C2, MCC=MCC, F1=F1, accu=accu,
             precision=precision, recall=recall, confmat=confmat)
