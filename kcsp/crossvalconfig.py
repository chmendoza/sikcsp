import os, sys
import yaml
import numpy as np
import scipy.io as sio
from argparse import ArgumentParser


# Add path to package directory to access main module using absolute import
sys.path.insert(0,
                os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             '..')))


from kcsp import utils

def minusone(x): return x - 1  # Matlab index starts at 1, Python at 0

os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

# Parse command-line arguments
parser = ArgumentParser()
parser.add_argument("-k", "--nclusters", dest="n_clusters",
                    nargs='+', type=int, help="Number of clusters")
parser.add_argument("-P", "--centroid-lengths", type=int,
                    dest="centroid_lengths", nargs='+', 
                    help="Length of cluster centroids")
parser.add_argument("--patient", dest="patients",
                    action='append', nargs='+', help="Patient labels")
parser.add_argument("--bands", dest="bands", type=int, action="append",
                    nargs='+', help="Spectral band ids")

args = parser.parse_args()

params = dict.fromkeys(
    ['Patient dir', 'Results dir', 'Filenames', 'n_folds',\
        'Data', 'Algorithm', 'Random seed'])
params['Filenames'] = dict.fromkeys(
    ['CSP filters', 'Data indices', 'Results'])
params['Data'] = dict.fromkeys(
    ['Segment length', 'Window length', 'Index of CSP filters'])
params['Algorithm'] = dict.fromkeys(['metric', 'init', 'n_runs', 'n_clusters', 'centroid_length'])

params['n_folds'] = 10

patients = args.patients
data_dir = '/lustre/scratch/cmendoza/sikmeans/LKini2019'
confdir = '/home/1420/sw/shift_kmeans/kcsp/config'
methods = ['regular', 'betadiv', 'max-sliced-Bures']
method = methods[0]

srate = 512 # Hz
params['Data']['Segment length'] = srate * 60 # 1 minute
params['Data']['Window length'] = srate * 1  # 1 second

conditions = ['preictal', 'interictal']
bands = args.bands
if not isinstance(bands, list):
    bands = [bands]

params['Algorithm']['metric'] = 'cosine'
params['Algorithm']['init'] = 'random-energy'
params['Algorithm']['n_runs'] = 3
n_classes = 2

# Read number of cluster and centroid lengths from command line
# Transform them into tuples of repeated value: use same values for preictal and interictal segments
n_clusters = args.n_clusters
centroid_lengths = args.centroid_lengths
n_clusters = np.array(n_clusters)
n_clusters = np.repeat(n_clusters, 2).reshape(-1, n_classes)
n_clusters = n_clusters.tolist()
centroid_lengths = np.array(centroid_lengths)
centroid_lengths = np.repeat(
    centroid_lengths, n_classes).reshape(-1, n_classes)
centroid_lengths = centroid_lengths.tolist()

overlap_str = 'non-overlap'
winlen = srate * 60  # 1 minute windows extracted from epochs
start_gap = srate * 1 # gap for random sampling
dfname = 'split_%s_winlen-%d_gap-%d.mat' % (overlap_str, winlen, start_gap)
params['Filenames']['Data indices'] = dfname

for i_patient, patient in enumerate(patients):

    patient = patient[0]  # list -> str
    patient_dir = os.path.join(data_dir, patient)
    n_samples = [0] * 2  # Num. of training samples [preictal, interictal]

    # Define index of CSP filters based on # of channels
    dirpath = os.path.join(patient_dir)
    fpath = os.path.join(dirpath, 'metadata.mat')
    metadata = sio.loadmat(fpath, simplify_cells=True)
    n_chan = metadata['metadata']['channel_labels'].size  # num. of channels
    params['Data']['Index of CSP filters'] = [0, n_chan]    
    
    for i_condition, condition in enumerate(conditions):
        # file names and start indices of segments
        dirpath = os.path.join(patient_dir, condition)
        fpath = os.path.join(dirpath, dfname)
        i_start = utils.loadmat73(fpath, 'train_indices')
        i_start = utils.apply2list(i_start, np.squeeze)
        i_start = utils.apply2list(i_start, minusone)
        n_samples[i_condition] = utils.apply2list(i_start, np.size)
        n_samples[i_condition] = np.sum(np.array(n_samples[i_condition]))
    
    for band in bands[i_patient]:        
        params['Filenames']['CSP filters'] = 'results_band%d_%s_%s_winlen-%d_gap-%d.mat' % (
            band, method, overlap_str, winlen, start_gap)
        for k in n_clusters:
            for P in centroid_lengths:
                params['Patient dir'] = os.path.join(data_dir, patient)
                results_dirname = 'band%d_%s_k%d-%d_P%d-%d' % (
                    band, method, *k, *P)
                results_path = os.path.join(data_dir, patient, results_dirname)
                params['Results dir'] = results_path
                os.makedirs(results_path, exist_ok=True)
                rfname = 'crossval_band%d_%s_k%d-%d_P%d-%d.npy' % (
                    band, method, *k, *P)
                params['Filenames']['Results'] = rfname
                    
                params['Algorithm']['n_clusters'] = k
                params['Algorithm']['centroid_length'] = P                
                confname = 'band%d_%s_k%d-%d_P%d-%d.yaml' % (
                    band, method, *k, *P)
                confpath = os.path.join(confdir, patient, confname)
                with open(confpath, 'w') as yamlfile:
                    yaml.dump(params, yamlfile, sort_keys=False)
