import os
import h5py
import numpy as np
import multiprocessing
import numbers

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


## Get CSP data
# Global variables to be shared by parallel processes spawned by getCSPdata()
shared_dict = dict.fromkeys(['dirpath', 'dfname', 'i_start', 'winlen', 'W'])

def _get_CSPdata_single(i_epoch):
    dirpath = shared_dict['dirpath']
    fname = shared_dict['dfname'][i_epoch]    
    i_start = shared_dict['i_start'][i_epoch]
    winlen = shared_dict['winlen']
    W = shared_dict['W']

    fpath = os.path.join(dirpath, fname)
    n_chan = W.shape[0]
    epoch = loadmat73(fpath, 'epoch')
    col = i_start[:, None, None] + np.arange(winlen).reshape(1, 1, -1)
    row = np.arange(n_chan).reshape(1, -1, 1)    
    windows = epoch[row, col]  # (n_win[i_epoch], n_chan, winlen)    

    return np.matmul(W.T, windows)

def getCSPdata(dirpath, dfname, i_start, winlen, W):    
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

    Returns
    -------
    X (double):
        Matrix with CSP signals. X.shape=(n_csp, N, winlen), with N equal to the total number of windows.
    """

    global shared_dict
    shared_dict['dirpath'] = dirpath
    shared_dict['dfname'] = dfname
    shared_dict['i_start'] = i_start
    shared_dict['winlen'] = winlen
    shared_dict['W'] = W

    if W.ndim > 1:
        n_csp = W.shape[1]
    else:
        n_csp = 1

    n_epoch = len(dfname)
    n_win = np.array([i_start[i].size for i in range(len(i_start))])
    cumwin = np.r_[0, n_win.cumsum()]

    X = np.empty((n_win.sum(), n_csp, winlen)).squeeze()

    n_proc = int(os.environ['SLURM_CPUS_PER_TASK'])
    chunksize = 1    
    
    with multiprocessing.Pool(n_proc) as pool:
        imap_it = pool.imap(_get_CSPdata_single, range(n_epoch), chunksize)
        for i_epoch, x in enumerate(imap_it):
            X[cumwin[i_epoch]:cumwin[i_epoch+1]] = x

    if n_csp > 1:
        X = np.transpose(X, (1, 0, 2))

    # Keep the dictionary layout, clear/save memory
    shared_dict = dict.fromkeys(shared_dict.keys(), None)

    return X

## Cross-validation

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
