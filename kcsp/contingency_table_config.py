import os, sys
import yaml
import numpy as np
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
                    type=int, help="Number of clusters")
parser.add_argument("-P", "--centroid-length", type=int,\
    dest="centroid_length", help="Length of cluster centroids")
parser.add_argument("--patient", dest="patient", help="Patient label")
parser.add_argument("--band", dest="band", type=int,\
    help="Spectral band ids")

args = parser.parse_args()

params = dict.fromkeys(
    ['Patient dir', 'Filenames', 'Data'])
params['Filenames'] = dict.fromkeys(
    ['CSP filters', 'Data indices', 'Codebooks', 'Results'])
params['Data'] = dict.fromkeys(['Segment length', 'Window length'])

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
params['Patient dir'] = os.path.join(data_dir, patient)
Wfname = 'results_band%d_%s_%s_winlen-%d_gap-%d.mat'\
    % (band, method, overlap_str, winlen, start_gap)
cfname = 'train_test_band%d_%s_k%d-%d_P%d-%d.npz'\
    % (band, method, k, k, P, P)
rfname = 'contingency_tables_band%d_%s_k%d-%d_P%d-%d.npz'\
    % (band, method, k, k, P, P)
params['Filenames']['Codebooks'] = cfname
params['Filenames']['Results'] = rfname
params['Filenames']['CSP filters'] = Wfname
# random seed passed to the shift-invariant means lgorithm
seed = np.random.SeedSequence()

confsubdir = os.path.join(confdir, patient)
confname = 'contingency_tables_band%d_%s_k%d-%d_P%d-%d.yaml'\
    % (band, method, k, k, P, P)
confpath = os.path.join(confsubdir, confname)
with open(confpath, 'w') as yamlfile:
        yaml.dump(params, yamlfile, sort_keys=False)
