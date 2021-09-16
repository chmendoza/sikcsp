# Functions to do feature extraction and train the classifier

import sys, os
import numpy as np
from sklearn.naive_bayes import MultinomialNB, ComplementNB, BernoulliNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize, FunctionTransformer, MinMaxScaler
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline

sys.path.insert(0,
                os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from kcsp import utils


def l2norm(X):
    """Normalize features"""
    return normalize(X, norm='l2', axis=0)


WORD_COUNT_CLASSIFIERS = ['MultinomialNB', 'ComplementNB',
                          'LinearSVC', 'LogisticRegression', 'BernoulliNB']
SCALERS = {
    'tf-idf': TfidfTransformer(),
    'l2': FunctionTransformer(l2norm),
    'min-max': MinMaxScaler(feature_range=(-1,1), copy=False),
    'StandardScaler': StandardScaler(),
    'RobustScaler': RobustScaler()
    }

CLASSIFIERS = {
    'MultinomialNB': MultinomialNB(),
    'ComplementNB': ComplementNB(),
    'BernoulliNB': BernoulliNB(),
    'LinearSVC': LinearSVC(),
    'LogisticRegression': LogisticRegression()
}


def _flat_bow(X, C, metric):
    """Flattened bag of words

    Parameters
    ----------
    X(array):
        A matrix of shape (2, m, n, w), where m, n and w are the number of 
        segments, windows per segment, and window length. X[0] is the data to 
        be clustered using C1, the preictal codebook, and X[1] is the data to 
        be clustered using C2, the interictal codebook.
    C(sequence):
        A sequence of two codebooks. C[0] is the preictal codebook and C[1] is
        the interictal codebook. C[0].shape = (k1, P1) and C[1].shape =
        (k2, P2), with k1, k2 being the number of centroids on each codebook, and P1, P2 being the centroid lenghts.
    """   
    
    n_centroids = np.zeros(2, dtype=int)
    n_centroids[0], n_centroids[1] = C[0].shape[0], C[1].shape[0]
    n_samples = X[0].shape[0] # Number of segments    
    tot_centroids = n_centroids.sum()

    sample = np.zeros((n_samples, tot_centroids))
    for i_sample in range(n_samples): # segment = sample
        for r in np.arange(2):  # codebook        
            nu = utils.cluster_assignment(X[r, i_sample], C[r], metric=metric)
            nu, counts = np.unique(nu, return_counts=True)
            i_feature = nu + r * n_centroids[0] #centroid index->feature index
            sample[i_sample, i_feature] = counts

    return sample


def extract_features(X, C, metric, clfname='MultinomialNB'):
    """ Extract features from X and C for a given classifier

    Parameters
    ----------
    X(array):
        A matrix of shape (2, m, n, w), where m, n and w are the number of 
        segments, windows per segment, and window length. X[0] is the data to 
        be clustered using C1, the preictal codebook, and X[1] is the data to 
        be clustered using C2, the interictal codebook.
    C(sequence):
        A sequence of two codebooks. C[0] is the preictal codebook and C[1] is
        the interictal codebook. C[0].shape = (k1, P1) and C[1].shape =
        (k2, P2), with k1, k2 being the number of centroids on each codebook, and P1, P2 being the centroid lenghts.
    clfname (string):
        The classifier to be used.

    Returns
    -------
    sample (array):
        For 'MultinomialNB' and 'LinearSVC', it is the feature count on each segment, a matrix of shape (n,k), with n=n1+n2, and k=k1+k2.
    """

    if clfname in WORD_COUNT_CLASSIFIERS:
        sample = _flat_bow(X, C, metric)
        return sample
    else:
        return None


def fit(train_sample, true_s, clfname='MultinomialNB', clfparams=dict()):
    """Instantiate and train classifier"""

    scl, clfparams = scaler(clfparams)
    clf = CLASSIFIERS[clfname]
    clf.set_params(**clfparams)

    clf = make_pipeline(scl, clf)

    clf.fit(train_sample, true_s)

    return clf
    

def scaler(clfparams):
    """Instantiate and set params for feature scaler"""
    # Instantiate scaler
    if 'Scaler' in clfparams:
        scldict = clfparams['Scaler']
        sclname = scldict['name']        
        scl = SCALERS[sclname]
        if 'params' in scldict:
            sclparams = scldict['params']
            scl.set_params(**sclparams)
        clfparams.pop('Scaler')
    else:
        scl = FunctionTransformer()
    
    return scl, clfparams
