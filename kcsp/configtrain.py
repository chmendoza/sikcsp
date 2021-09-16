"""
Configure experiment for train_test.py
"""


import os, sys
import scipy.io as sio

# Export these environment variables before running the script
DATA_DIR = os.environ['DATA_DIR']
SRATE = 512 # Hz

# Add path to package directory to access main module using absolute import
sys.path.insert(0,
                os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             '..')))

from kcsp.classifier_defs import CLASSIFIERS

def is_regularized(clf):
    return 'params' in clf and 'C' in clf['params']

def configure_experiment(args):   

    params = dict.fromkeys(
        ['Patient dir', 'Filenames', 'Data', 'Algorithm', 'Classifier'])
    params['Filenames'] = dict.fromkeys(
        ['CSP filters', 'Data indices', 'Results'])
    params['Data'] = dict.fromkeys(
        ['Segment length', 'Window length', 'Index of CSP filters'])
    params['Algorithm'] = dict.fromkeys(
        ['metric', 'init', 'n_runs', 'Num. of clusters', 'Centroid length'])

    patient = args.patient
    k = args.n_clusters
    P = args.centroid_length
    band = args.band
    regfactor = args.regfactor
    clf_id = args.clf

    # Set regularization parameters
    clf = CLASSIFIERS[clf_id]
    params['Classifier'] = clf
    if is_regularized(clf):
        params['Classifier']['params']['C'] = regfactor

    
    params['Data']['Segment length'] = SRATE * 60 # 1 minute
    params['Data']['Window length'] = SRATE * 1  # 1 second
   
    params['Algorithm']['metric'] = 'cosine'
    params['Algorithm']['init'] = 'random-energy'
    params['Algorithm']['n_runs'] = 3
    params['Algorithm']['Num. of clusters'] = k
    params['Algorithm']['Centroid length'] = P

    winlen = SRATE * 60  # 1 minute windows extracted from epochs
    start_gap = SRATE * 1 # gap for random sampling
    dfname = f'split_non-overlap_winlen-{winlen}_gap-{start_gap}.mat'
    params['Filenames']['Data indices'] = dfname

    # Dataset (patient) folder
    patient_dir = os.path.join(DATA_DIR, patient)
    params['Patient dir'] = patient_dir

    # Define index of CSP filters based on # of channels    
    fpath = os.path.join(patient_dir, 'metadata.mat')
    metadata = sio.loadmat(fpath, simplify_cells=True)
    n_chan = metadata['metadata']['channel_labels'].size  # num. of channels
    i_csp = [0, n_chan-1]
    # list of ints -> str:
    params['Data']['Index of CSP filters'] = ' '.join(list(map(str, i_csp)))
    
    Wfname = (f'results_band{band}_non-overlap_winlen'
              f'-{winlen}_gap-{start_gap}.mat')
    params['Filenames']['CSP filters'] = Wfname

    # Results file name
    if is_regularized(clf):
        rfname = f'train_test_band{band}_k{k}_P{P}_clf{clf_id}_C{regfactor}.npz'
    else:
        rfname = f'train_test_band{band}_k{k}_P{P}_clf{clf_id}.npz'

    params['Filenames']['Results'] = rfname

    return params
