import os, sys
import yaml
import itertools
import numpy as np
from argparse import ArgumentParser

# Add path to package directory to access main module using absolute import
sys.path.insert(0,
                os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             '..')))


from kcsp import utils

# Parse command-line arguments
parser = ArgumentParser()
parser.add_argument("-k", "--nclusters", dest="n_clusters",
                    nargs='+', type=int, help="Number of clusters")
parser.add_argument("-P", "--centroid-lengths", type=int,
                    dest="centroid_lengths", nargs='+', 
                    help="Length of cluster centroids")
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

patients = ['HUP070', 'HUP078']
data_dir = '/lustre/scratch/cmendoza/sikmeans/LKini2019'
confdir = '/home/1420/sw/shift_kmeans/kcsp/config'
methods = ['regular', 'betadiv', 'max-sliced-Bures']
method = methods[0]

srate = 512 # Hz
params['Data']['Segment length'] = srate * 60 # 1 minute
params['Data']['Window length'] = srate * 1  # 1 second
bands = [[3,6],[2,7]] # best bands with highest AUC (NER'21)
# Number of training samples per patient. A sample is a segment.
n_samples = [[98, 1886], [64, 1980]] #(preictal, interictal)

params['Algorithm']['metric'] = 'cosine'
params['Algorithm']['init'] = 'random-energy'
params['Algorithm']['n_runs'] = 3
n_clusters = args.n_clusters
centroid_lengths = args.centroid_lengths
n_classes = 2
n_clusters = list(itertools.product(n_clusters, repeat=n_classes))
centroid_lengths = list(itertools.product(centroid_lengths, repeat=n_classes))

for i_patient, patient in enumerate(patients):
    for band in bands[i_patient]:        
        for k in n_clusters:
            for P in centroid_lengths:
                params['Patient dir'] = os.path.join(data_dir, patient)
                # Use same initial seed for folds of same hyperparameter point since random genetator in gen_kfold.py must be the same to generate the k-fold partition.                                
                seed = np.random.SeedSequence()
                params['Crossvalidation']['rng_seed'] = seed.entropy
                rng = np.random.default_rng(seed)
                N1, N2 = n_samples[i_patient]
                kfold1 = utils.kfold_split(N1, n_folds, shuffle=True, rng=rng)
                kfold2 = utils.kfold_split(N2, n_folds, shuffle=True, rng=rng)
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
                    Wfname = 'W_winlen-1min_band%d_%s.mat' % (band, method)
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
