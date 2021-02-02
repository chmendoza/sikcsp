import numpy as np
import sys
import os

sys.path.insert(0,
                os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from shift_kmeans.wrappers import si_pairwise_distances_argmin_min, si_row_norms


def cluster_assignment(X1, X2, C1, C2, metric='euclidean'):
    P1 = C1.shape[1]  # centroid length
    P2 = C2.shape[1]
    XX1, XX2 = None, None

    if metric == 'euclidean':
        XX1 = si_row_norms(X1, P1, squared=True)
        XX2 = si_row_norms(X2, P2, squared=True)

    # nu_rs, for r,s in {1,2}
    nu_11, _, _ = si_pairwise_distances_argmin_min(X1, C1, metric, XX1)    
    nu_21, _, _ = si_pairwise_distances_argmin_min(X1, C2, metric, XX1)    
    nu_12, _, _ = si_pairwise_distances_argmin_min(X2, C1, metric, XX2)    
    nu_22, _, _ = si_pairwise_distances_argmin_min(X2, C2, metric, XX2)

    return (nu_11, nu_21, nu_12, nu_22)

def likelihood(nu, N, k):
    
    # r=1, s=1
    nu_11, counts = np.unique(nu[0], return_counts=True)
    p_J1 = np.zeros((k[0],))
    p_J1[nu_11] = counts
    p_J1 = p_J1/N[0]

    # r=2, s=1    
    nu_21, counts = np.unique(nu[1], return_counts=True)
    p_J2 = np.zeros((k[1],))
    p_J2[nu_21] = counts
    p_J2 = p_J2/N[0]

    p_J = np.zeros((2,k[0],k[1]))
    p_J1 = p_J1.reshape(k[0], 1)
    p_J2 = p_J2.reshape(1, k[1])
    p_J[0] = np.matmul(p_J1, p_J2) # (k1, k2), given S=1

    # r=1, s=2    
    nu_12, counts = np.unique(nu[2], return_counts=True)
    p_J1 = np.zeros((k[0],))
    p_J1[nu_12] = counts
    p_J1 = p_J1/N[1]

    # r=2, s=2    
    nu_22, counts = np.unique(nu[3], return_counts=True)
    p_J2 = np.zeros((k[1],))
    p_J2[nu_22] = counts
    p_J2 = p_J2/N[1]

    p_J1 = p_J1.reshape(k[0], 1)
    p_J2 = p_J2.reshape(1, k[1])
    p_J[1] = np.matmul(p_J1, p_J2)  # (k1, k2), given S=2

    return p_J
