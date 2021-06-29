import os, sys
import yaml
import numpy as np
from argparse import ArgumentParser
import scipy.io as sio

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
                    type=int, help="Number of clusters")
parser.add_argument("-P", "--centroid-length", type=int,\
    dest="centroid_length", help="Length of cluster centroids")
parser.add_argument("--patient", dest="patient", help="Patient label")
parser.add_argument("--band", dest="band", type=int,\
    help="Spectral band ids")

args = parser.parse_args()

params = dict.fromkeys(
    ['Patient dir', 'Filenames', 'Data', 'Algorithm'])
params['Filenames'] = dict.fromkeys(
    ['CSP filters', 'Data indices', 'Results'])
params['Data'] = dict.fromkeys(
    ['Segment length', 'Window length', 'Index of CSP filters'])
params['Algorithm'] = dict.fromkeys(['metric', 'init', 'n_runs', 'n_clusters', 'centroid_length', 'rng_seed'])

patient = args.patient
data_dir = '/lustre/scratch/cmendoza/sikmeans/LKini2019'
confdir = '/home/1420/sw/shift_kmeans/kcsp/config'
methods = ['regular', 'betadiv', 'max-sliced-Bures']
method = methods[0]

srate = 512 # Hz
params['Data']['Segment length'] = srate * 60 # 1 minute
params['Data']['Window length'] = srate * 1  # 1 second

conditions = ['preictal', 'interictal']
band = args.band

params['Algorithm']['metric'] = 'cosine'
params['Algorithm']['init'] = 'random-energy'
params['Algorithm']['n_runs'] = 3
n_classes = 2

# Read number of cluster and centroid lengths from command line
# Transform them into tuples of repeated value: use same values for preictal and interictal segments
k = args.n_clusters
P = args.centroid_length

overlap_str = 'non-overlap'
winlen = srate * 60  # 1 minute windows extracted from epochs
start_gap = srate * 1 # gap for random sampling
dfname = 'split_%s_winlen-%d_gap-%d.mat' % (overlap_str, winlen, start_gap)
params['Filenames']['Data indices'] = dfname

patient_dir = os.path.join(data_dir, patient)

# Define index of CSP filters based on # of channels
dirpath = os.path.join(patient_dir)
fpath = os.path.join(dirpath, 'metadata.mat')
metadata = sio.loadmat(fpath, simplify_cells=True)
n_chan = metadata['metadata']['channel_labels'].size  # num. of channels
params['Data']['Index of CSP filters'] = [0, n_chan-1]

params['Patient dir'] = patient_dir
Wfname = 'results_band%d_%s_%s_winlen-%d_gap-%d.mat'\
    % (band, method, overlap_str, winlen, start_gap)
rfname = 'train_test_band%d_%s_k%d-%d_P%d-%d.npz'\
    % (band, method, k, k, P, P)
params['Filenames']['Results'] = rfname                    
params['Filenames']['CSP filters'] = Wfname
params['Algorithm']['n_clusters'] = k
params['Algorithm']['centroid_length'] = P
# random seed passed to the shift-invariant means lgorithm
seed = np.random.SeedSequence()
params['Algorithm']['rng_seed'] = seed.entropy

confsubdir = os.path.join(confdir, patient)
confname = 'train_test_band%d_%s_k%d-%d_P%d-%d.yaml'\
    % (band, method, k, k, P, P)
confpath = os.path.join(confsubdir, confname)
with open(confpath, 'w') as yamlfile:
        yaml.dump(params, yamlfile, sort_keys=False)
