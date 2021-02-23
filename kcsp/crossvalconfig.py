import os, sys
import yaml
import itertools
import numpy as np

# Add path to package directory to access main module using absolute import
sys.path.insert(0,
                os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             '..')))


from kcsp import utils

params = dict.fromkeys(['crossval', 'data', 'algo'])
params['crossval'] = dict.fromkeys(
    ['n_folds', 'i_fold', 'foldfile', 'rng_seed'])
params['data'] = dict.fromkeys(
    ['patient_dir', 'Wpath', 'dfname', 'rfname', 'segment_len', 'window_len'])
params['algo'] = dict.fromkeys(['metric', 'init', 'n_runs', 'n_clusters', 'centroid_length', 'rng_seed'])

n_folds = 10
params['crossval']['n_folds'] = n_folds

patients = ['HUP070', 'HUP078']
data_dir = '/lustre/scratch/cmendoza/sikmeans/LKini2019'
confdir = '/home/1420/sw/shift_kmeans/kcsp/config'
methods = ['regular', 'betadiv', 'max-sliced-Bures']
method = methods[0]

srate = 512 # Hz
params['data']['segment_len'] = srate * 60 # 1 minute
params['data']['window_len'] = srate * 1  # 1 second
bands = [[3,6],[2,7]] # best bands with highest AUC (NER'21)
# Number of training samples per patient. A sample is a segment.
n_samples = [[98, 1886], [64, 1980]] #(preictal, interictal)

params['algo']['metric'] = 'cosine'
params['algo']['init'] = 'random-energy'
params['algo']['n_runs'] = 3
n_clusters = [4, 8, 16, 32, 64, 128]
centroid_lengths = [30, 40, 60, 120, 200, 350]
n_classes = 2
n_clusters = list(itertools.product(n_clusters, repeat=n_classes))
centroid_lengths = list(itertools.product(centroid_lengths, repeat=n_classes))

for i_patient, patient in enumerate(patients):
    for band in bands[i_patient]:        
        for k in n_clusters:
            for P in centroid_lengths:                
                # Use same initial seed for folds of same hyperparameter point since random genetator in gen_kfold.py must be the same to generate the k-fold partition.                                
                seed = np.random.SeedSequence()
                params['crossval']['rng_seed'] = seed.entropy
                rng = np.random.default_rng(seed)
                N1, N2 = n_samples[i_patient]
                kfold1 = utils.kfold_split(N1, n_folds, shuffle=True, rng=rng)
                kfold2 = utils.kfold_split(N2, n_folds, shuffle=True, rng=rng)
                kfold1 = list(kfold1)
                kfold2 = list(kfold2)                
                for i_fold in range(n_folds):
                    params['crossval']['i_fold'] = i_fold
                    ffname = 'fold%d_band%d_%s_k%d-%d_P%d-%d.npz' % (
                        i_fold, band, method, *k, *P)
                    ffpath = os.path.join(data_dir, patient, ffname)
                    # Save cross-validation indices
                    with open(ffpath, 'wb') as foldfile:
                        np.savez(foldfile,\
                            train1=kfold1[i_fold][0],\
                            test1=kfold1[i_fold][1],\
                            train2=kfold2[i_fold][0],\
                            test2=kfold2[i_fold][1])
                                        
                    params['crossval']['foldfile'] = ffname
                    Wfname = 'results_sikcsp_band%d_%s.mat' % (band, method)
                    dfname = 'winlen-1min_start_gap-1sec.mat'
                    rfname = 'misclass_fold%d_band%d_%s_k%d-%d_P%d-%d.npy' % (
                        i_fold, band, method, *k, *P)
                    params['data']['rfname'] = rfname
                    params['data']['dfname'] = dfname
                    params['data']['Wpath'] = os.path.join(
                        data_dir, patient, Wfname)
                    params['data']['patient_dir'] = os.path.join(
                        data_dir, patient)
                    params['algo']['n_clusters'] = k
                    params['algo']['centroid_length'] = P
                    # A different random seed is passed to the shift-invariant k-means algorithm on each fold
                    seed = np.random.SeedSequence()
                    params['algo']['rng_seed'] = seed.entropy
                    confname = 'crossval_kcsp_fold%d_band%d_%s_k%d-%d_P%d-%d.yaml' % (i_fold, band, method, *k, *P)
                    confpath = os.path.join(confdir, patient, confname)
                    with open(confpath, 'w') as yamlfile:
                        yaml.dump(params, yamlfile, sort_keys=False)
