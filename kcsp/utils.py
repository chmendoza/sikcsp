import os
import h5py
import numpy as np
from joblib import Parallel, delayed, parallel_backend
import numbers
import tracemalloc, linecache

def loadmat73(fpath, varname):
    """ Load data from a -v7.3 Matlab file."""
    with h5py.File(fpath, 'r') as hdf5:
        dataset = hdf5[varname]
        return _h5py_unpack(dataset, hdf5)


def _h5py_unpack(obj, hdf5):
    """
    Unpack an HDF5 object saved on Matlab v7.3 format.

    It can unpack:
        - (float, int, char) cell arrays. Returned as nested lists with the original Matlab 'shape' and with the same data type.
        - (float, int) arrays. Returned as numpy arrays of the same data type and with the original shape.

    Parameters
    ----------
    obj (array of object references, object reference, dataset):
        The first call should have obj as a dataset type. That dataset might contain a reference or array of references to other datasets.
    hdf5 (File object):
        An instance of h5py.File()

    Returns
    -------
    Numpy arrays for Matlab arrays and nested lists for cell arrays.

    Inspired by https://github.com/skjerns/mat7.3
    """
    if isinstance(obj, np.ndarray): # array of references
        if obj.size == 1:
            obj = obj[0] # an object reference
            obj = hdf5[obj] # a dataset
            return _h5py_unpack(obj, hdf5)
        elif obj.size > 1:
            cell = []
            for ref in obj:
                entry = _h5py_unpack(ref, hdf5)
                cell.append(entry)            
            return cell
    elif isinstance(obj, h5py.h5r.Reference): # an object reference
        obj = hdf5[obj]
        return _h5py_unpack(obj, hdf5)
    elif isinstance(obj, h5py._hl.dataset.Dataset):  # a dataset
        vartype = obj.attrs['MATLAB_class']
        if vartype == b'cell':
            cell = []
            for ref in obj:
                entry = _h5py_unpack(ref, hdf5)
                cell.append(entry)
            if len(cell) == 1:
                cell = cell[0]
            if obj.parent.name == '/': # first call
                 if isinstance(cell[0], list): # cell is a nested list
                    cell = list(map(list, zip(*cell)))  # transpose cell
            return cell
        elif vartype == b'char':
            stra = np.array(obj).ravel()
            stra = ''.join([chr(x) for x in stra])
            return stra
        else: #(float or int, not struct)
            array = np.array(obj)
            array = array.T # from C order to Fortran (MATLAB) order
            return array

def apply2list(obj, fun):
    if not isinstance(obj, list):
        return fun(obj)
    else:
        return [apply2list(x, fun) for x in obj]


def _get_CSPdata_single(fpath, i_start, winlen, W):
    
    n_chan = W.shape[0]
    epoch = loadmat73(fpath, 'epoch')
    col = np.reshape(i_start, (-1,1,1)) + np.arange(winlen).reshape(1, 1, -1)
    row = np.arange(n_chan).reshape(1, -1, 1)    
    windows = epoch[row, col]  # (n_win[i_epoch], n_chan, winlen)    

    # matmul treats `windows` as a stack of 2D matrices residing in the last
    # two indices. Final shape=(n_win[i_epoch], n_csp, winlen)
    return np.matmul(W.T, windows)

def getCSPdata(dirpath, dfname, i_start, winlen, W, n_cpus=1):
    """
    Extract raw EEG windows and apply CSP filter

    Parameters
    ----------
    dirpath (str):
        Absolute path of folder with data files. Each file has a variable
        'epoch' of size [C T], with C and T being the number of channels and time points, respectively.
    dfname (str):
        List with the name of the files that will be used to extract the windows. The pattern name is rx<epoch_id>.mat, with epoch_id being an integer.
    i_start (uint32):
        List with the start index of each window on each epoch file.
        i_start[i][j] is the start index of the j-th window in dfname[i]. i_start[i].shape = (ni,), where ni is the number of windows extracted from dfname[i].
    winlen (int):
        Length of each window.
    W (double):
        A matrix whose columns are the spatial (CSP) filters. W.shape=(C, d),with d being the number of spatial filters.
    n_cpus (int):
        Number of SLURM CPUs to be used

    Returns
    -------
    X (double):
        Matrix with CSP signals. X.shape=(n_csp, N, winlen), with N equal to the total number of windows.
    """

    n_epoch = len(dfname)    
    
    with parallel_backend("loky", inner_max_num_threads=1):
        X = Parallel(n_jobs=n_cpus)(delayed(_get_CSPdata_single)\
            (os.path.join(dirpath, dfname[i_epoch]),\
                i_start[i_epoch], winlen, W) for i_epoch in range(n_epoch))

       
    X = np.concatenate(X, axis=0) # list of 3D np arrays -> 3D array       
    X = np.transpose(X, (1, 0, 2))

    return X


def check_rng(seed):
    """Turn seed into a np.random.Generator instance

    Parameters
    ----------
    seed : None, int or instance of Generator
        If seed is None, return the Generator using the OS entropy.
        If seed is an int, return a new Generator instance seeded with seed.
        If seed is already a Generator instance, return it.
        Otherwise raise ValueError.
    """
    if seed is None:
        return np.random.default_rng()
    if isinstance(seed, numbers.Integral):
        seed = np.random.SeedSequence(seed)
        return np.random.default_rng(seed)
    if isinstance(seed, np.random.Generator):
        return seed

    raise ValueError('%r cannot be used to seed a numpy.random.Generator'
                     ' instance' % seed)


def kfold_split(n_samples, n_folds, shuffle=None, rng=None):    
    indices = np.arange(n_samples)     
    if shuffle:
        rng = check_rng(rng)
        rng.shuffle(indices)
    
    fold_sizes = np.full(n_folds, n_samples // n_folds, dtype=int)
    fold_sizes[:n_samples % n_folds] += 1
    current = 0
    for fold_size in fold_sizes:
        test_mask = np.zeros(n_samples, dtype=bool)
        start, stop = current, current + fold_size
        test_mask[indices[start:stop]] = True
        train_index = indices[np.logical_not(test_mask)]
        test_index = indices[test_mask]
        yield train_index, test_index
        current = stop


def splitdata(X, chunk_size, keep_dims=True):
    """
    Split a data into smaller non-overlapping chunks

    Parameters
    ----------
    X (array):
        A 2D array. The rows are observations (data points). X.shape = (m,n)
    chunk_size (int):
        Each observation is split into chunks of size chunk_size. It is assumed that n = k*chunk_size, with k an integer.
    keep_dims (bool):
        It controls the number of dimensions of the returned matrix. See below.

    Returns
    -------
    X (array):
        If keep_dims == True, X.shape = (k*m, chunk_size): the chunks of each observation are stacked vertically as rows of the output matrix. If keep_dims == False, X.shape = (m, k, chunk_size): the chunks are stacked along the second dimension and extend along the third dimension of the output matrix.
    """

    ind1 = np.arange(X.shape[0]).reshape(-1, 1, 1)
    offset = np.arange(0, X.shape[1], chunk_size).reshape(1, -1, 1)
    chunk_ind = np.arange(chunk_size).reshape(1, 1, -1)
    ind2 = offset + chunk_ind
    X = X[ind1, ind2]

    if keep_dims:
        return X.reshape(-1, chunk_size)
    else:
        return X

def confusion_matrix(s, s_hat):
    """
    Compute entries of a 2x2 confusion matrix

    Parameters
    ----------
    s (array):
        True class labels. s.shape = (N,). Positive class is s=1. Negative class is s=2.
    s_hat (array):
        Predicted class labels. s_hat.shape = (N,)

    Return
    ------
    X (array):
        Confusion matrix. 
            X[0,0]: True positives
            X[0,1]: False negatives
            X[1,0]: False positives
            X[1,1]: True negatives
    """

    n_samples = s.shape[0]
    classes = np.r_[1, 2]
    N_CLASSES = classes.size

    X = np.empty((N_CLASSES, N_CLASSES))

    for ii, tl in enumerate(classes): # true label
        for jj, pl in enumerate(classes): # predicted label
            v = (s == tl) & (s_hat == pl)
            X[ii, jj] = v.nonzero()[0].size

    return X


def display_top(snapshot, key_type='lineno', limit=10):
    snapshot = snapshot.filter_traces((tracemalloc.Filter(
        False, "<frozen importlib._bootstrap>"), tracemalloc.Filter(False, "<unknown>"),))
    top_stats = snapshot.statistics(key_type)
    
    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        print("#%s: %s:%s: %.1f KiB" %
              (index, frame.filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))
