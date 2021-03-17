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
    ['Patient dir', 'Fold indices dir', 'Filenames', 'Crossvalidation', 'Data', 'Algorithm'])
params['Filenames'] = dict.fromkeys(
    ['Fold indices', 'CSP filters', 'Data indices', 'Results'])
params['Crossvalidation'] = dict.fromkeys(['n_folds', 'i_fold', 'rng_seed'])
params['Data'] = dict.fromkeys(['Segment length', 'Window length'])
params['Algorithm'] = dict.fromkeys(['metric', 'init', 'n_runs', 'n_clusters', 'centroid_length', 'rng_seed'])

n_folds = 10
params['Crossvalidation']['n_folds'] = n_folds

patients = args.patients
data_dir = '/lustre/scratch/cmendoza/sikmeans/LKini2019'
confdir = '/home/1420/sw/shift_kmeans/kcsp/config'
# patients = ['Study012']
# data_dir = '/home/cmendoza/Research/sikmeans/LKini2019/data/toy'
# confdir = '/home/cmendoza/Research/sikmeans/LKini2019/config'
methods = ['regular', 'betadiv', 'max-sliced-Bures']
method = methods[0]

srate = 512 # Hz
params['Data']['Segment length'] = srate * 60 # 1 minute
params['Data']['Window length'] = srate * 1  # 1 second

conditions = ['preictal', 'interictal']
bands = args.bands
if not isinstance(bands, list):
    bands = [bands]
# bands = [[3, 6], [2, 7]]  # best bands with highest AUC (NER'21)
# Number of training samples per patient. A sample is a segment.
# # Study012-toy. Training:
# n_samples = [[61,111]] #(preictal, interictal)
# # Study012-toy. Testing: (16,28)
# bands = [[3]] # best bands with highest AUC (NER'21)
# Number of training samples per patient. A sample is a segment.


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

# Consider all possible 'combinations':
# n_clusters = list(itertools.product(n_clusters, repeat=n_classes))
# centroid_lengths = list(itertools.product(centroid_lengths, repeat=n_classes))

overlap_str = 'non-overlap'
winlen = srate * 60  # 1 minute windows extracted from epochs
start_gap = srate * 1 # gap for random sampling
dfname = 'split_%s_winlen-%d_gap-%d.mat' % (overlap_str, winlen, start_gap)

for i_patient, patient in enumerate(patients):
    patient = patient[0] # list -> str
    patient_dir = os.path.join(data_dir, patient)
    n_samples = [0] * 2 # Num. of training samples [preictal, interictal]
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
        for k in n_clusters:
            for P in centroid_lengths:
                params['Patient dir'] = os.path.join(data_dir, patient)
                # Use same initial seed for folds of same hyperparameter point since random genetator in gen_kfold.py must be the same to generate the k-fold partition.
                seed = np.random.SeedSequence()
                params['Crossvalidation']['rng_seed'] = seed.entropy
                rng = np.random.default_rng(seed)                
                kfold1 = utils.kfold_split(
                    n_samples[0], n_folds, shuffle=True, rng=rng)
                kfold2 = utils.kfold_split(
                    n_samples[1], n_folds, shuffle=True, rng=rng)
                kfold1 = list(kfold1)
                kfold2 = list(kfold2)
                foldname = 'band%d_%s_k%d-%d_P%d-%d' % (
                    band, method, *k, *P)
                foldpath = os.path.join(data_dir, patient, foldname)
                os.makedirs(foldpath, exist_ok=True)
                confsubdir = os.path.join(confdir, patient,\
                    'band%d_%s_k%d-%d_P%d-%d' % (band, method, *k, *P))
                os.makedirs(confsubdir, exist_ok=True)
                params['Fold indices dir'] = foldname
                for i_fold in range(n_folds):
                    params['Crossvalidation']['i_fold'] = i_fold
                    ffname = 'fold%d.npz' % i_fold
                    ffpath = os.path.join(foldpath, ffname)
                    # Save cross-validation indices
                    with open(ffpath, 'wb') as foldfile:
                        np.savez(foldfile,\
                            train1=kfold1[i_fold][0],\
                            test1=kfold1[i_fold][1],\
                            train2=kfold2[i_fold][0],\
                            test2=kfold2[i_fold][1])
                                        
                    params['Filenames']['Fold indices'] = ffname
                    Wfname = 'results_band%d_%s_%s_winlen-%d_gap-%d' % (band, method, overlap_str, winlen, start_gap)                    
                    dfname = 'winlen-1min_start_gap-1sec.mat'
                    rfname = 'misclass_fold%d_band%d_%s_k%d-%d_P%d-%d.npy' % (
                        i_fold, band, method, *k, *P)
                    params['Filenames']['Results'] = rfname
                    params['Filenames']['Data indices'] = dfname
                    params['Filenames']['CSP filters'] = Wfname
                    params['Algorithm']['n_clusters'] = k
                    params['Algorithm']['centroid_length'] = P
                    # A different random seed is passed to the shift-invariant k-means algorithm on each fold
                    seed = np.random.SeedSequence()
                    params['Algorithm']['rng_seed'] = seed.entropy
                    confname = 'fold%d.yaml' % i_fold
                    confpath = os.path.join(confsubdir, confname)
                    with open(confpath, 'w') as yamlfile:
                        yaml.dump(params, yamlfile, sort_keys=False)
