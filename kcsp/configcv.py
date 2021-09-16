"""
Configure the experiment for crossval.py
"""

import os, sys
import scipy.io as sio

# Add path to package directory to access main module using absolute import
sys.path.insert(0,
                os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             '..')))

from kcsp.classifier_defs import CLASSIFIERS

# Export these environment variables before running the script
DATA_DIR = os.environ['DATA_DIR']
SRATE = 512  # Hz

def is_regularized(clf):
    return 'params' in clf and 'C' in clf['params']

# Classifier configurations
def configure_experiment(args):    

    params = dict.fromkeys(
        ['Patient dir', 'Filenames', 'n_folds',\
            'Data', 'Metric', 'Classifier'])
    params['Filenames'] = dict.fromkeys(
        ['CSP filters', 'Data indices', 'Results'])
    params['Data'] = dict.fromkeys(
        ['Segment length', 'Window length', 'Index of CSP filters'])

    params['n_folds'] = 10
    params['Data']['Segment length'] = SRATE * 60  # 1 minute
    params['Data']['Window length'] = SRATE * 1  # 1 second
    params['Metric'] = 'cosine'

    patient = args.patient
    band = args.band
    regfactor = args.regfactor
    clf_id = args.clf    
       
    # Set regularization parameters
    clf = CLASSIFIERS[clf_id]
    params['Classifier'] = clf
    if is_regularized(clf):
        params['Classifier']['params']['C'] = regfactor

    # Dataset (patient) folder
    patient_dir = os.path.join(DATA_DIR, patient)
    params['Patient dir'] = patient_dir

    # File containing the train/test split indices and file names
    winlen = SRATE * 60  # 1 minute windows extracted from epochs
    start_gap = SRATE * 1  # gap for random sampling
    dfname = 'split_non-overlap_winlen-%d_gap-%d.mat' % (winlen, start_gap)
    params['Filenames']['Data indices'] = dfname

    # Index of CSP filters based on # of channels    
    fpath = os.path.join(patient_dir, 'metadata.mat')
    metadata = sio.loadmat(fpath, simplify_cells=True)
    n_chan = metadata['metadata']['channel_labels'].size  # num. of channels
    i_csp = [0, n_chan-1]
    # list of ints -> str:
    params['Data']['Index of CSP filters'] = ' '.join(list(map(str,i_csp)))

    # *.mat file with matrix W, the CSP filters
    params['Filenames']['CSP filters'] =\
        'results_band%d_non-overlap_winlen-%d_gap-%d.mat' % (\
            band, winlen, start_gap)

    # Results file name
    if is_regularized(clf):
        rfname = 'averageMCC_band%d_clf%s_C%g.npy' % (
            band, clf_id, regfactor)        
    else:
        rfname = 'averageMCC_band%d_clf%s.npy' % (
            band, clf_id)
        
    params['Filenames']['Results'] = rfname

    return params