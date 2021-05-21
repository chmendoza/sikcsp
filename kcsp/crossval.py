# Perform k-fold crossvalidation of a single point in the hyperparameter grid. Send each fold to a child Python process, using the multiprocessing module.

import os
import sys
import numpy as np
import time
from numpy.lib.function_base import median
import yaml
import functools
import multiprocessing
from argparse import ArgumentParser

# Add path to package directory to access main module using absolute import
sys.path.insert(0,
                os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             '..')))

from kcsp import utils, bayes
import shift_kmeans.shift_kmeans as sikmeans

def minusone(x): return x - 1  # Matlab index starts at 1, Python at 0

def _single_fold(params, X, iter_args):
    """ A single fold in a k-fold crossvalidation """

    winlen = params['Data']['Window length']
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
    train1, test1 = kfold_preictal
    train2, test2 = kfold_interictal

    # Split (cross-validation) training segments into smaller windows
    X1train = utils.splitdata(X[0][train1], winlen)  # preictal
    X2train = utils.splitdata(X[1][train2], winlen)  # interictal   

    # Training
    C1, _, _, _, _, _ = sikmeans.shift_invariant_k_means(
        X1train, k1, P1, metric=metric, init=init, n_init=n_runs, rng=rng,  verbose=True)
    C2, _, _, _, _, _ = sikmeans.shift_invariant_k_means(
        X2train, k2, P2, metric=metric, init=init, n_init=n_runs, rng=rng,  verbose=True)

    # Estimate likelihood and prior
    p_C = bayes.likelihood(X1train, X2train, C1, C2, metric=metric)  # (2,k1,k2)
    N1, N2 = X1train.shape[0], X2train.shape[0]  # Number of windows
    p_S = np.zeros(2)
    p_S[0] = N1/(N1+N2)
    p_S[1] = N2/(N1+N2)

    # Split (crossval) test segments into smaller windows
    X1test = utils.splitdata(X[0][test1], winlen, keep_dims=False)  # preictal
    X2test = utils.splitdata(
        X[1][test2], winlen, keep_dims=False)  # Interictal

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
    # Assume that the positive class (preictal) is the minority class and that the  negative class (interictal) is the majority class. Also, there are always examples from both classes.
    MCC = np.zeros(2)    
    confmat = [0]*2
    for ii in np.arange(2):
        confmat[ii] = utils.confusion_matrix(s, s_hat[ii, :])
        tp, fn, fp, tn = confmat[ii].flatten()        
        if (tp == 0 and fp == 0) or (tn == 0 and fn == 0):
            MCC[ii] += 0            
        else:            
            MCC[ii] += (tp * tn - fp * fn) /\
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
    seed = params['Random seed']

    #%% Random generator
    seed = np.random.SeedSequence(seed)
    print('Initial random seed: %s' % seed.entropy)
    rng = np.random.default_rng(seed)
    if seed is None:
        print('Saving initial seed to disk...')
        params['init_seed'] = seed.entropy
        with open(confpath, 'w') as yamfile:
            yaml.dump(params, yamfile, sort_keys=False)

    #%% Get the CSP filters
    wpath = os.path.join(patient_dir, wfname)
    W = utils.loadmat73(wpath, 'W')
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
    kfold1 = list(kfold1)
    kfold2 = list(kfold2)

    # For each fold, pass a different seed to the shift-invariant k-means algo
    ss = rng.bit_generator._seed_seq
    child_seeds = ss.spawn(n_folds)

    child_params = dict.fromkeys(['Data', 'Algorithm'])
    child_params['Data']['Window length'] = winlen
    child_params['Algorithm']['metric'] = metric
    child_params['Algorithm']['init'] = init
    child_params['Algorithm']['n_runs'] = n_runs
    child_params['Algorithm']['n_clusters'] = k1, k2
    child_params['Algorithm']['centroid_length'] = P1, P2
    
    _simple_single_fold = functools.partial(_single_fold, child_params, X)
    iter_args = zip(kfold1, kfold2, child_seeds)

    fullMCC = np.zeros((10,2))
    chunksize = 1
    with multiprocessing.Pool(n_cpus) as pool:
        imap_it = pool.imap(_simple_single_fold, iter_args, chunksize)
        for i_fold, MCC in enumerate(imap_it):
            fullMCC[i_fold] = MCC

    meanMCC = fullMCC.mean(axis=0)

    rpath = os.path.join(results_dir, rfname)
    with open(rpath, 'wb') as f:
        np.save(f, meanMCC)


if __name__ == '__main__':
    main()
