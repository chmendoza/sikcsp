# Perform k-fold crossvalidation of a single point in the hyperparameter grid. Send each fold to a child Python process, using the multiprocessing module.

import os
import sys
import numpy as np
import time
from numpy.lib.function_base import median
import yaml
import ray
from argparse import ArgumentParser

# Add path to package directory to access main module using absolute import
sys.path.insert(0,
                os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             '..')))

from kcsp import utils, bayes
import shift_kmeans.shift_kmeans as sikmeans

def minusone(x): return x - 1  # Matlab index starts at 1, Python at 0

@ray.remote
def _single_fold(params, X, iter_args):
    """ A single fold in a k-fold crossvalidation """

    winlen = params['Window length']
    metric = params['Algorithm']['metric']
    init = params['Algorithm']['init']
    n_runs = params['Algorithm']['n_runs']
    k1, k2 = params['Algorithm']['n_clusters']
    P1, P2 = params['Algorithm']['centroid_length']

    # Unpack the iterable's item passed to the child process
    kfold_preictal, kfold_interictal, seed = iter_args

    # Each child gets its own random seed
    rng = np.random.default_rng(seed)

    ## Get indices of one cross-validation fold
    train_ind, test_ind = [0]*2, [0]*2
    train_ind[0], test_ind[0] = kfold_preictal
    train_ind[1], test_ind[1] = kfold_interictal

    # Split (cross-validation) training segments into smaller windows
    # Number of segments
    n_seg = np.zeros(2, dtype=int)
    n_seg[0], n_seg[1] = train_ind[0].size, train_ind[1].size
    n_csp, _, seglen = X[0].shape
    n_win_per_seg = seglen // winlen    
    Xtrain = [0] * 2
    for s in range(2):  # class segment
        n_win = n_win_per_seg * n_seg[s]
        Xtrain[s] = np.zeros((n_csp, n_win, winlen))
        for r in range(2): # CSP filter
            Xtrain[s][r] = utils.splitdata(X[s][r, train_ind[s]], winlen)

    # Training begins
    C1, _, _, _, _, _ = sikmeans.shift_invariant_k_means(
        Xtrain[0][0], k1, P1, metric=metric, init=init, n_init=n_runs, rng=rng,  verbose=True)
    C2, _, _, _, _, _ = sikmeans.shift_invariant_k_means(
        Xtrain[1][1], k2, P2, metric=metric, init=init, n_init=n_runs, rng=rng,  verbose=True)

    # Estimate likelihood and prior
    p_C = bayes.likelihood(Xtrain, (C1,C2), metric=metric) #(2,k1,k2)
    N1, N2 = Xtrain[0].shape[1], Xtrain[1].shape[1]  # Number of windows
    p_S = np.zeros(2)
    p_S[0] = N1/(N1+N2)
    p_S[1] = N2/(N1+N2)

    # Training ends, Testing begins
          
    # Concatenate preictal and interictal data filtered with CSP-1
    X1test = np.concatenate((X[0][0,test_ind[0]], X[1][0,test_ind[1]]), axis=0)
    # Concatenate preictal and interictal data filtered with CSP-C
    X2test = np.concatenate((X[0][1,test_ind[0]], X[1][1,test_ind[1]]), axis=0)
    
    # Split (crossval) test segments into smaller windows. This creates a 3D 
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
        evalp_C = p_C[:, nu_1, nu_2] # Eval learned likelihood
        logp_C = np.full((2, nu_1.size), np.NINF) # Avoid divide-by-zero warning
        np.log(evalp_C, out=logp_C, where=evalp_C > 0)
        M = X1test[i_segment].shape[0]  # Num. of windows on a segment
        logMAP = M*np.log(p_S) + np.sum(logp_C, axis=1)
        logML = M*np.log(likelihood_weights) + np.sum(logp_C, axis=1)
        
        # Find MAP and ML estimate
        s_hat[0, i_segment] = np.argmax(logMAP) + 1 # array index to class label
        s_hat[1, i_segment] = np.argmax(logML) + 1

    # True label vector
    s = np.r_[np.ones(test_ind[0].size, dtype=int), \
        2 * np.ones(test_ind[1].size, dtype=int)]

    # Compute Matthews correlation coefficient (MCC)
    # Assume that the positive class (preictal) is the minority class and that the  negative class (interictal) is the majority class. Also, there are always examples from both classes.
    MCC = np.zeros(2)    
    confmat = [0]*2
    for ii in np.arange(2): # loop over {MAP, ML}
        confmat[ii] = utils.confusion_matrix(s, s_hat[ii, :])
        tp, fn, fp, tn = confmat[ii].flatten()        
        if (tp == 0 and fp == 0) or (tn == 0 and fn == 0):
            MCC[ii] = 0            
        else:            
            MCC[ii] = (tp * tn - fp * fn) /\
                np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))

    return MCC
    

def main():
    # Parse command-line arguments
    parser = ArgumentParser()
    parser.add_argument("-f", "--config-file", dest="confpath",
                        help="YAML configuration file")

    parser.add_argument("-n", "--n-cpus", dest="n_cpus", type=int,
                        default=1, help="Number of SLURM CPUS")

    args = parser.parse_args()
    n_cpus = args.n_cpus

    ray.init(num_cpus=n_cpus)

    os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

    #%% Read paramaters to set up the experiment
    confpath = args.confpath

    with open(confpath, 'r') as yamlfile:
        params = yaml.load(yamlfile, Loader=yaml.FullLoader)

    print('=========== Configuration =============')
    print(yaml.dump(params, sort_keys=False))
    print('=======================================')
    patient_dir = params['Patient dir']
    results_dir = params['Results dir']    
    wfname = params['Filenames']['CSP filters']
    dfname = params['Filenames']['Data indices']
    rfname = params['Filenames']['Results']
    n_folds = params['n_folds']
    winlen = params['Data']['Window length']
    seglen = params['Data']['Segment length']
    #NOTE:works only for one index (filter):
    i_csp = params['Data']['Index of CSP filters']
    metric = params['Algorithm']['metric']
    init = params['Algorithm']['init']
    n_runs = params['Algorithm']['n_runs']
    k1, k2 = params['Algorithm']['n_clusters']
    P1, P2 = params['Algorithm']['centroid_length']
    init_seed = params['Random seed']

    #%% Random generator
    seed = np.random.SeedSequence(init_seed)
    print('Initial random seed: %s' % seed.entropy)
    rng = np.random.default_rng(seed)
    if init_seed is None:
        print('Saving initial seed to disk...')
        params['Random seed'] = seed.entropy
        with open(confpath, 'w') as yamfile:
            yaml.dump(params, yamfile, sort_keys=False)

    #%% Get the CSP filters
    wpath = os.path.join(patient_dir, wfname)
    W = utils.loadmat73(wpath, 'W')
    i_csp = np.array(i_csp)
    W = W[:, i_csp]

    #%% Extract data and apply CSP filter
    conditions = ['preictal', 'interictal']
    X = [0]*2
    tic = time.perf_counter()
    n_samples = [0] * 2  # Num. of training samples [preictal, interictal]
    for i_condition, condition in enumerate(conditions):
        # file names and start indices of preictal segments
        dirpath = os.path.join(patient_dir, condition)
        fpath = os.path.join(dirpath, dfname)
        i_start = utils.loadmat73(fpath, 'train_indices')
        i_start = utils.apply2list(i_start, np.squeeze)
        i_start = utils.apply2list(i_start, minusone)    
        n_samples[i_condition] = utils.apply2list(i_start, np.size)
        n_samples[i_condition] = np.sum(np.array(n_samples[i_condition]))
        dfnames = utils.loadmat73(fpath, 'train_names')


        # Extract data and apply CSP filter
        X[i_condition] = utils.getCSPdata(
            dirpath, dfnames, i_start, seglen, W, n_cpus=n_cpus)

    toc = time.perf_counter()
    print("Data gathered and filtered after %0.4f seconds" % (toc - tic))

    kfold1 = utils.kfold_split(
        n_samples[0], n_folds, shuffle=True, rng=rng)
    kfold2 = utils.kfold_split(
        n_samples[1], n_folds, shuffle=True, rng=rng)    

    # For each fold, pass a different seed to the shift-invariant k-means algo
    ss = rng.bit_generator._seed_seq
    child_seeds = ss.spawn(n_folds)

    child_params = dict.fromkeys(['Window length', 'Algorithm'])
    child_params['Window length'] = winlen
    child_params['Algorithm'] = dict.fromkeys(
        ['metric', 'init', 'n_runs', 'n_clusters', 'centroid_length'])
    child_params['Algorithm']['metric'] = metric
    child_params['Algorithm']['init'] = init
    child_params['Algorithm']['n_runs'] = n_runs
    child_params['Algorithm']['n_clusters'] = k1, k2
    child_params['Algorithm']['centroid_length'] = P1, P2

    X_ref = ray.put(X)
    child_params_ref = ray.put(child_params)
    
    MCC_ref = []
    for iter_args in zip(kfold1, kfold2, child_seeds):
        MCC_ref.append(_single_fold.remote(
            child_params_ref, X_ref, iter_args))

    MCC = ray.get(MCC_ref) # list of refs -> list of 1D arrays
    MCC = np.vstack(MCC) # list of 1D arrays -> 2D array
    meanMCC = MCC.mean(axis=0)

    rpath = os.path.join(results_dir, rfname)
    with open(rpath, 'wb') as f:
        np.save(f, meanMCC)


if __name__ == '__main__':
    main()
