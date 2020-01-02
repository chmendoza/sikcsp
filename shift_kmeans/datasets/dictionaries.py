"""
Functions to generate dictionaries
"""

import numpy as np

def sindict(n_atoms, n_features, dtype=np.float64):
    '''
    Discrete Sine Transform (DST-I) dictionary

    Args:
        n_atoms (int): Number of atoms
        n_features (int): length of each atom in the dictionary

    Returns:
        D (ndarray): Dictionary

    Shape:
        D: (n_atoms, n_features)



    Ref: https://en.wikipedia.org/wiki/Discrete_sine_transform#DST-I
    '''

#    print(f"Random seed inside sindict: {np.random.get_state()[1][0]}")

    n = np.arange(n_features)[np.newaxis,:]  #time
    k = n.transpose() #frequency
    time_freq_matrix = (k+1)*(n+1)
    D = np.sin(np.pi*time_freq_matrix/(n_features+1), dtype=dtype)
    ind = np.random.permutation(n_atoms)
    ind = ind[:n_atoms]
    D = D[ind]

    D_norm = np.linalg.norm(D,ord=2,axis=1)[:,np.newaxis]
    D = D / D_norm

    return D

