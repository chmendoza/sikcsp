"""
Configure experiment for dict4cv.py
"""

import os
import sys
import scipy.io as sio


# Add path to package directory to access main module using absolute import
sys.path.insert(0,
                os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             '..')))


def minusone(x): return x - 1  # Matlab index starts at 1, Python at 0


os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

# Parse command-line arguments

N_FOLDS = 10
DATA_DIR = os.environ['DATA_DIR'] # folder with data sets
SRATE = 512  # Hz
CONDITIONS = ['preictal', 'interictal']
N_RUNS = 3  # Number of times to internally run k-means algo

def configure_experiment(args):

    params = dict.fromkeys(\
        ['Patient dir', 'kP dir', 'Filenames',\
            'n_folds', 'Data', 'Algorithm'])
    params['Filenames'] = dict.fromkeys(\
        ['CSP filters', 'Data indices', 'Crossval codebooks'])
    params['Data'] = dict.fromkeys(\
        ['Segment length', 'Window length', 'Index of CSP filters'])
    params['Algorithm'] = dict.fromkeys(\
        ['metric', 'init', 'n_runs', 'Num. of clusters', 'Centroid length'])

    k = args.n_clusters
    P = args.centroid_length
    patient = args.patient
    band = args.band
    metric = args.metric
    init = args.init


    params['n_folds'] = N_FOLDS
    params['Data']['Segment length'] = SRATE * 60  # 1 minute
    params['Data']['Window length'] = SRATE * 1  # 1 second
    params['Algorithm']['metric'] = metric  # {'cosine', 'euclidean'}
    params['Algorithm']['init'] = init # {'random-energy', 'k-means++'}
    params['Algorithm']['n_runs'] = N_RUNS

    winlen = SRATE * 60  # 1 minute windows extracted from epochs
    start_gap = SRATE * 1  # gap for random sampling
    dfname = f'split_non-overlap_winlen-{winlen}_gap-{start_gap}.mat'
    params['Filenames']['Data indices'] = dfname

    patient_dir = os.path.join(DATA_DIR, patient)
    params['Patient dir'] = patient_dir

    # Index of CSP filters based on # of channels
    fpath = os.path.join(patient_dir, 'metadata.mat')
    metadata = sio.loadmat(fpath, simplify_cells=True)
    n_chan = metadata['metadata']['channel_labels'].size  # num. of channels
    i_csp = [0, n_chan-1]
    # list of ints -> str:
    params['Data']['Index of CSP filters'] = ' '.join(list(map(str, i_csp)))

    wfname = (f'results_band{band}_non-overlap_'
        f'winlen-{winlen}_gap-{start_gap}.mat')
    params['Filenames']['CSP filters'] = wfname
    

    # Folder where to save intermediate data for each k-P point
    kP_dirname = f'band{band}_k{k}-{k}_P{P}-{P}'
    kP_path = os.path.join(patient_dir, kP_dirname)
    params['kP dir'] = kP_path
    os.makedirs(kP_path, exist_ok=True)

    # File to save intermediate crossval codebooks
    if init == 'k-means++':
        init = 'kmeanspp'

    rfname = f'cvdicts_{metric}_{init}.pickle'
    params['Filenames']['Crossval codebooks'] = rfname

    params['Algorithm']['Num. of clusters'] = k
    params['Algorithm']['Centroid length'] = P

    return params
