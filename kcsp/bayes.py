import numpy as np
import sys
import os

sys.path.insert(0,
                os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from shift_kmeans.wrappers import si_pairwise_distances_argmin_min, si_row_norms


def cluster_assignment(X, C1, C2, metric='euclidean'):
    """ Cluster assingment using dictionaries from two classes

    Find the closest centroid in C1 and C2 to the observations (rows) in X, and assign the corresponding cluster labels to those observations. C1 and C2 are  and 2, respectively.

    Parameters
    ----------
    X (array):
        A data matrix. Rows are observations. X.shape = (m, n).
    C1 (array):
        Shift-invariant dictionary of waveforms (rows) from class 1. C1.shape = (m1, P1). P1 <= n.
    C2 (array):
        Shift-invariant dictionary of waveforms (rows) from class 2. C2.shape = (m2, P2). P2 <= n.

    Returns
    -------
    nu_1 (array):
        Cluster labels of observations in X, using the dictionary C1.
    nu_2 (array):
        Cluster labels of observations in X, using the dictionary C2.
    """

    P1 = C1.shape[1]  # centroid length
    P2 = C2.shape[1]
    XX = None

    if metric == 'euclidean':
        XX = si_row_norms(X, P1, squared=True)
    
    nu_1, _, _ = si_pairwise_distances_argmin_min(X, C1, metric, XX)
    nu_2, _, _ = si_pairwise_distances_argmin_min(X, C2, metric, XX)

    return nu_1, nu_2


def likelihood(X1train, X2train, C1, C2, metric="cosine"):

    k1, k2 = C1.shape[0], C2.shape[0]

    # Syntax for cluster assignments: nu_rs. r is the index of the codebook (C1 or  C2). s is the index of the window (or segment) class, preictal (s=1) or  interictal (s=2).
    nu_11, nu_21 = cluster_assignment(X1train, C1, C2, metric=metric)
    nu_12, nu_22 = cluster_assignment(X2train, C1, C2, metric=metric)
    N1, N2 = X1train.shape[0], X2train.shape[0]  # Number of windows
    
    # r=1, s=1
    nu_11, counts = np.unique(nu_11, return_counts=True)
    p_C1 = np.zeros(k1)
    p_C1[nu_11] = counts
    p_C1 = p_C1/N1

    # r=2, s=1    
    nu_21, counts = np.unique(nu_21, return_counts=True)
    p_C2 = np.zeros(k2)
    p_C2[nu_21] = counts
    p_C2 = p_C2/N1

    p_C = np.zeros((2,k1,k2))
    p_C1 = p_C1.reshape(k1, 1)
    p_C2 = p_C2.reshape(1, k2)
    p_C[0] = np.matmul(p_C1, p_C2) # (k1, k2), given S=1

    # r=1, s=2    
    nu_12, counts = np.unique(nu_12, return_counts=True)
    p_C1 = np.zeros(k1)
    p_C1[nu_12] = counts
    p_C1 = p_C1/N2

    # r=2, s=2    
    nu_22, counts = np.unique(nu_22, return_counts=True)
    p_C2 = np.zeros(k2)
    p_C2[nu_22] = counts
    p_C2 = p_C2/N2

    p_C1 = p_C1.reshape(k1, 1)
    p_C2 = p_C2.reshape(1, k2)
    p_C[1] = np.matmul(p_C1, p_C2)  # (k1, k2), given S=2

    return p_C
