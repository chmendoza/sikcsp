"""
Pre-learn preictal and interictal codebooks of a single point in the hyperparameter grid (number of centroids, centroid length)

These pre-learned codebooks are to be used in crossval.py to run a 10-fold crossvalidation of a binary classifier to pick the hyperparameters with the highest average score. 

Send each fold to a child Python process. Doing parallel computing in Python on 
a SLURM cluster has been dificult. I tried with the built-in multiprocessing 
module and Ray, but the performance was not great. Finally, joblib worked great 
on the DARWIN cluster (mostly AMD CPUs), but not so great on Caviness (Intel 
CPUs). It seems that the MKL prefers the AVX2-optimized core routines and 
doesn't follow the explicit threaded-based parallelism that is coded with the 
inner_max_num_threads=2 option in the parallel_backend() call. TODO: Solve this 
issue, and use a command line argument to specify the number of threads, since 
right now is hardcoded to 2 here, and to 1 in the getCSPdata().
"""
import os
import sys
import numpy as np
import time
import yaml
import pickle
import shutil
import tempfile
from joblib import Parallel, delayed, dump, load, parallel_backend
from argparse import ArgumentParser

# Add path to package directory to access main module using absolute import
sys.path.insert(0,
                os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             '..')))

import shift_kmeans.shift_kmeans as sikmeans
from kcsp import utils, configdict4cv

def minusone(x): return x - 1  # Matlab index starts at 1, Python at 0


os.environ["OMP_THREAD_LIMIT"] = "2"
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'


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
    train_ind = [0]*2
    train_ind[0], _ = kfold_preictal
    train_ind[1], _ = kfold_interictal

    # Split (cross-validation) training segments into smaller windows
    # Number of segments
    Xtrain = [0] * 2
    Xtrain[0] = utils.splitdata(X[0][0, train_ind[0]], winlen)
    Xtrain[1] = utils.splitdata(X[1][1, train_ind[1]], winlen)

    # Training begins
    C1, _, _, _, _, _ = sikmeans.shift_invariant_k_means(
        Xtrain[0], k1, P1, metric=metric, init=init, n_init=n_runs, rng=rng,  verbose=True)
    C2, _, _, _, _, _ = sikmeans.shift_invariant_k_means(
        Xtrain[1], k2, P2, metric=metric, init=init, n_init=n_runs, rng=rng,  verbose=True)

    return C1, C2


def main():
    # Parse command-line arguments
    parser = ArgumentParser()
    parser.add_argument("-k", "--nclusters", dest="n_clusters",\
        type=int, help="Number of clusters")
    parser.add_argument("-P", "--centroid-length", type=int,\
        dest="centroid_length", help="Length of cluster centroids")
    parser.add_argument("--patient", dest="patient", help="Patient label")
    parser.add_argument("--band", dest="band",\
        type=int, help="Spectral band id")
    parser.add_argument("-i", dest="init", default="random-energy",\
        help="Initialization algorithm for k-means")
    parser.add_argument("-m", dest="metric", default="cosine",\
        help="Distance metric")
    parser.add_argument("-n", "--n-cpus", dest="n_cpus", type=int,
                        default=1, help="Number of SLURM CPUS")

    args = parser.parse_args()
    n_cpus = args.n_cpus

    #%% Read paramaters to set up the experiment
    params = configdict4cv.configure_experiment(args)
    
    print('=========== Configuration =============')
    print(yaml.dump(params, sort_keys=False))
    print('=======================================')
    patient_dir = params['Patient dir']
    kP_dir = params['kP dir']
    wfname = params['Filenames']['CSP filters']
    dfname = params['Filenames']['Data indices']
    rfname = params['Filenames']['Crossval codebooks']
    n_folds = params['n_folds']
    winlen = params['Data']['Window length']
    seglen = params['Data']['Segment length']
    #NOTE:works only for one filter per class:
    i_csp = params['Data']['Index of CSP filters']
    i_csp = list(map(int, i_csp.split()))  # str->list of ints
    metric = params['Algorithm']['metric']
    init = params['Algorithm']['init']
    n_runs = params['Algorithm']['n_runs']
    k = params['Algorithm']['Num. of clusters']
    P = params['Algorithm']['Centroid length']
    # init_seed = params['Random seed']

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

    #%% Random generator
    # entropy.txt has the entropy used to make the
    # crossvalidation splits (folds), which are the same used
    # to compute the intermediate crossval codebooks and to do
    # run the full crossval with a given classifier.
    rng_path = os.path.join(kP_dir, 'entropy.txt')    
    if os.path.isfile(rng_path):
        with open(rng_path, 'r') as f:
            seed = int(f.read().strip())
            seed = np.random.SeedSequence(entropy=seed)
            print('Initial random seed: %s' % seed.entropy)
    else:
        seed = np.random.SeedSequence()
        print('Initial random seed: %s' % seed.entropy)
        print('Saving initial seed to disk...')
        with open(rng_path, 'w') as f:
            f.write(str(seed.entropy))

    rng = np.random.default_rng(seed)

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
    child_params['Algorithm']['n_clusters'] = k, k
    child_params['Algorithm']['centroid_length'] = P, P

    folder = tempfile.mkdtemp()

    try:
        os.mkdir(folder)
    except FileExistsError:
        pass

    X_filename_memmap = os.path.join(folder, 'X_memmap')
    dump(X, X_filename_memmap)
    X_ref = load(X_filename_memmap, mmap_mode='r')

    # Run k-fold cross-validation in parallel
    with parallel_backend("loky", inner_max_num_threads=2):
        C = Parallel(n_jobs=n_cpus//2)(delayed(_single_fold)\
            (child_params, X_ref, iter_args)\
                for iter_args in zip(kfold1, kfold2, child_seeds))

    # C is a list of tuples (C1, C2)
    rpath = os.path.join(kP_dir, rfname)
    with open(rpath, 'wb') as f:
        pickle.dump(C, f, pickle.HIGHEST_PROTOCOL)

    try:
        shutil.rmtree(folder)
    except:  # noqa
        print('Could not clean-up automatically.')


if __name__ == '__main__':
    main()
