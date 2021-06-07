import numpy as np
import sys
import os

sys.path.insert(0,
                os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from shift_kmeans.wrappers import si_pairwise_distances_argmin_min, si_row_norms


def cluster_assignment(X, C, metric='euclidean'):
    """ Cluster assingment using a given codebook

    Find the closest centroid in C to the observations (rows) in X, and assign the corresponding cluster labels to those observations.

    Parameters
    ----------
    X (array):
        A data matrix. Rows are observations. X.shape = (m, n).
    C (array):
        Shift-invariant codebook of waveforms (rows). C.shape = (m, P). P <= n.   

    Returns
    -------
    nu (array):
        Cluster labels of observations in X, using the codebook C.    
    """

    P = C.shape[1]  # centroid length    
    XX = None

    if metric == 'euclidean':
        XX = si_row_norms(X, P, squared=True)
    
    nu, _, _ = si_pairwise_distances_argmin_min(X, C, metric, XX)

    return nu


def likelihood(X, C, metric="cosine"):
    """
    Conditional probability of a cluster assignment pair

    Compute joint likelihood of cluster assignments of a segment using two 
    codebooks and given the class of the segment.

    Parameters
    ----------
    X (sequence):
        A sequence (e.g., list) of two numpy arrays. X[0] is an l x n1 x m1
        matrix with preictal windows, with l1, n1, and m1 being the number of 
        CSP filters, the number of windows, and the window length. X[1] is an
        l x n2 x m2 matrix with interictal windows. X[:][0] is the first CSP 
        filter, optimized for preictal windows, and X[:][1] is the last CSP 
        filter, optimized for interictal windows.
    C (sequence):
        A sequence of two codebooks. C[0] is the preictal codebook and C[1] is 
        the interictal codebook. C[0].shape = (k1, P1) and C[1].shape = 
        (k2, P2), with k1,k2 being the number of centroids on each codebook, and P1,P2 being the centroid lenghts.
    
    Returns
    -------
    joint_like (array):
        A (2,k1,k2) array with the joint likelihood of the cluster assignments.
        For a given class, each window is filtered with both CSP filters and then clustered with the corresponding codebook. For example, a preictal window is filtered with CSP-1 (first filter) and clustered using C[0], and filtered with CSP-C (last filter) and clustered using C[1].

    TODO: How to extend this to more than two CSP filters?
    """

    n_clusters, n_samples = np.zeros(2), np.zeros(2)
    
    n_clusters[0], n_clusters[1] = C[0].shape[0], C[1].shape[0] # k1, k2
    n_samples[0], n_samples[1] = X[0].shape[0], X[1].shape[0]

    marginal_like = [0] * 2
    joint_like = np.zeros((2, n_clusters[0], n_clusters[1]))

    for s in np.arange(2):  # segment class        
        for r in np.arange(2): # codebook (CSP filter) class 
            nu = cluster_assignment(X[s][r], C[r], metric=metric)
            nu, counts = np.unique(nu, return_counts=True)
            marginal_like[r] = np.zeros(n_clusters[r])
            marginal_like[r][nu] = counts/n_samples[s]            
        joint_like[s] = np.matmul(\
            marginal_like[0].reshape(-1, 1),\
            marginal_like[1].reshape(1, -1))  # (k1, k2), given S=s
        
    return joint_like
