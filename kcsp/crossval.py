"""
Cross-validate a binary classifier for hyper-parameter tuning

Use preictal and preictal codebooks pre-learned in dict4cv.py, extract BoW features from cluster assignments, and cross-validate a binary classifier using the Matthews Correlation Coefficient (MCC).
"""


import os
import sys
from sklearn.metrics import matthews_corrcoef
import numpy as np
import time
import yaml
import shutil
import tempfile
import pickle
from joblib import Parallel, delayed, dump, load, parallel_backend
from argparse import ArgumentParser

# Add path to package directory to access main module using absolute import
sys.path.insert(0,
                os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             '..')))

from kcsp import utils, classifier, configcv

def minusone(x): return x - 1  # Matlab index starts at 1, Python at 0

os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
os.environ["OMP_THREAD_LIMIT"] = "2"
DATA_DIR = os.environ['DATA_DIR']



def _single_fold(params, X, iter_args):
    """ A single fold in a k-fold crossvalidation """

    winlen = params['Window length']
    metric = params['Metric']
    clfdict = params['Classifier']
    clfname = clfdict['name']    
    if 'params' in clfdict:
        clfparams = clfdict['params']
    else:
        clfparams = dict()
    
    # Unpack the iterable's item passed to the child process
    C, kfold_preictal, kfold_interictal, seed = iter_args

    # Each child gets its own random seed
    # rng = np.random.default_rng(seed)    

    ## Get indices of one cross-validation fold
    train_ind, test_ind = [0]*2, [0]*2
    train_ind[0], test_ind[0] = kfold_preictal
    train_ind[1], test_ind[1] = kfold_interictal

    # Split (cross-validation) training segments into smaller windows
    n_seg = np.zeros(2, dtype=int)  # Number of segments
    n_seg[0], n_seg[1] = train_ind[0].size, train_ind[1].size
    tot_seg = n_seg.sum()
    seglen = X[0].shape[2]
    n_win_per_seg = seglen // winlen
    # (codebook, segment, window, time):
    Xtrain = np.zeros((2,tot_seg,n_win_per_seg,winlen))
    for r in range(2):  # codebook
        for s in range(2):  # segment class            
            Xtrain[r, s*n_seg[0]:n_seg[0]+s*n_seg[1]] =\
                utils.splitdata(X[s][r, train_ind[s]], winlen, keep_dims=False)

    # Training data
    train_sample = classifier.extract_features(\
        Xtrain, C, metric=metric, clfname=clfname)
    # True label vector
    s_true = np.r_[np.ones(n_seg[0], dtype=int),\
                   2 * np.ones(n_seg[1], dtype=int)]
    
    # Instantiate and train classifier
    clf = classifier.fit(train_sample, s_true,\
        clfname=clfname, clfparams=clfparams)

    # ==== Training ends, Testing begins ====
    # Prepare data
    n_seg[0], n_seg[1] = test_ind[0].size, test_ind[1].size
    tot_seg = n_seg.sum()
    Xtest = np.zeros((2, tot_seg, n_win_per_seg, winlen))
    for r in range(2):
        # Concatenate preictal and interictal data filtered with CSP-r
        Xtest_cat = np.concatenate(
        (X[0][r, test_ind[0]], X[1][r, test_ind[1]]), axis=0)
        
        # Split (crossval) test segments into smaller windows. This creates a 3D
        # matrix with the smaller windows (2D matrices) lying in the last two
        # indices and the first axis indexing the segments.
        Xtest[r] = utils.splitdata(Xtest_cat, winlen, keep_dims=False)

    test_sample = classifier.extract_features(\
        Xtest, C, metric=metric, clfname=clfname)
    s_hat = clf.predict(test_sample)
    # True label vector
    s_true = np.r_[np.ones(n_seg[0], dtype=int),
                   2 * np.ones(n_seg[1], dtype=int)]
    
    return matthews_corrcoef(s_true, s_hat)


def main():

    # Parse command-line arguments
    parser = ArgumentParser()    
    parser.add_argument("--patient", dest="patient", help="Patient label")
    parser.add_argument("--band", dest="band",\
        type=int, help="Spectral band ids")
    parser.add_argument("--classifier", dest="clf", help="Classifier id")
    parser.add_argument("-C", type=float, dest="regfactor", default=1,
                    help="Regularization factor for SVM and logistic\
                        regression classifiers")
    parser.add_argument("-n", "--n-cpus", dest="n_cpus", type=int,
                        default=1, help="Number of SLURM CPUS")

    args = parser.parse_args()    
    n_cpus = args.n_cpus
    band = args.band
    n_clusters = [4, 8, 16, 32, 64, 128]    
    centroid_length = [30, 40, 60, 120, 200, 350]

    # Dataset (patient) folder
    patient = args.patient
    patient_dir = os.path.join(DATA_DIR, patient)

    #%% Read paramaters to set up the experiment
    params = configcv.configure_experiment(args)

    wfname = params['Filenames']['CSP filters']
    dfname = params['Filenames']['Data indices']
    rfname = params['Filenames']['Results']
    n_folds = params['n_folds']
    winlen = params['Data']['Window length']
    seglen = params['Data']['Segment length']
    #NOTE:works only for one index (filter):
    i_csp = params['Data']['Index of CSP filters']
    i_csp = list(map(int,i_csp.split())) # str->list of ints
    metric = params['Metric']
    clfdict = params['Classifier']

    print('=========== Configuration =============')
    print(yaml.dump(params, sort_keys=False))
    print('=======================================')

    #%% Get the CSP filters
    wpath = os.path.join(patient_dir, wfname)
    W = utils.loadmat73(wpath, 'W')
    i_csp = np.array(i_csp)
    W = W[:, i_csp]

    #%% Extract data and apply CSP filter
    conditions = ['preictal', 'interictal']
    X = [0]*2
    tic = time.perf_counter()
    # Num. of training samples [preictal, interictal]
    n_samples = [0] * 2
    for i_condition, condition in enumerate(conditions):
         # file names and start indices of preictal segments
         dirpath = os.path.join(patient_dir, condition)
         fpath = os.path.join(dirpath, dfname)
         i_start = utils.loadmat73(fpath, 'train_indices')
         i_start = utils.apply2list(i_start, np.squeeze)
         i_start = utils.apply2list(i_start, minusone)
         n_samples[i_condition] = utils.apply2list(i_start, np.size)
         n_samples[i_condition] = np.sum(
             np.array(n_samples[i_condition]))
         dfnames = utils.loadmat73(fpath, 'train_names')
         # Extract data and apply CSP filter
         X[i_condition] = utils.getCSPdata(
             dirpath, dfnames, i_start, seglen, W, n_cpus=n_cpus)

    toc = time.perf_counter()
    print("Data gathered and filtered after %0.4f seconds" % (toc - tic))

    # temp folder to save memmap of X (joblib)
    folder = tempfile.mkdtemp()
    try:
        os.mkdir(folder)
    except FileExistsError:
        pass

    X_filename_memmap = os.path.join(folder, 'X_memmap')
    dump(X, X_filename_memmap)
    X_ref = load(X_filename_memmap, mmap_mode='r')

    child_params = {}
    child_params['Window length'] = winlen
    child_params['Metric'] = metric
    child_params['Classifier'] = clfdict

    n_k = len(n_clusters)
    n_P = len(centroid_length)
    meanMCC = np.zeros((n_k, n_P))

    with parallel_backend("loky", inner_max_num_threads=2):
        with Parallel(n_jobs=n_cpus//2) as parallel:
            for i_k, k in enumerate(n_clusters):
                for i_P, P in enumerate(centroid_length):

                    print(f"Crossvalidating for k={k} and P={P}...")
                    tic = time.perf_counter()

                    # Folder where to save intermediate data for each k-P point
                    kP_dirname = f'band{band}_k{k}-{k}_P{P}-{P}'
                    kP_dir = os.path.join(patient_dir, kP_dirname)

                    #%% Random generator         
                    # entropy.txt has the entropy used to make the 
                    # crossvalidation splits (folds), which are the same used 
                    # to compute the intermediate crossval codebooks and to do 
                    # run the full crossval with a given classifier.
                    rng_path = os.path.join(kP_dir, 'entropy.txt')
                    with open(rng_path, 'r') as f:
                        seed = int(f.read().strip())
                        seed = np.random.SeedSequence(entropy=seed)
                        print(f'Initial random seed: {seed.entropy}')

                    rng = np.random.default_rng(seed)

                    kfold1 = utils.kfold_split(
                        n_samples[0], n_folds, shuffle=True, rng=rng)
                    kfold2 = utils.kfold_split(
                        n_samples[1], n_folds, shuffle=True, rng=rng)

                    # C is a list of tuples (C1, C2)
                    rpath = os.path.join(kP_dir, 'crossval_codebooks.pickle')
                    with open(rpath, 'rb') as f:
                        C = pickle.load(f)

                    # For each fold, pass a different seed to the 
                    # shift-invariant k-means algo
                    ss = rng.bit_generator._seed_seq
                    child_seeds = ss.spawn(n_folds)

                    # Run k-fold cross-validation in parallel
                    MCC_ref = parallel(delayed(_single_fold)\
                        (child_params, X_ref, iter_args)\
                            for iter_args in zip(C, kfold1, kfold2, child_seeds))

                    meanMCC[i_k, i_P] = np.array(MCC_ref).mean()

                    toc = time.perf_counter()
                    print('Crossvalidation finished after '
                          f'{toc-tic:0.4f} seconds.')                    
            
    # Print average MCC matrix
    with np.printoptions(precision=3, floatmode='fixed'):
        print(meanMCC)

    # Save average MCC matrix
    rpath = os.path.join(patient_dir, rfname)
    with open(rpath, 'wb') as f:
        np.save(f, meanMCC)
    
    try:
        shutil.rmtree(folder)
    except:  # noqa
        print('Could not clean-up automatically.')
        

if __name__ == '__main__':
    main()
