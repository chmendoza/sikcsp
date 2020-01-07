"""Shift-invariant k-means"""

import warnings

import numpy as np
import scipy.sparse as sp

from sklearn.utils.extmath import row_norms, stable_cumsum, squared_norm
from sklearn.utils import check_random_state
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics.pairwise import pairwise_distances_argmin_min

import shift_kmeans.utils as utils
import shift_kmeans.wrappers as wrappers
import shift_kmeans.datasets.utils as data

###############################################################################
# Initialization


def init_kernels(X, n_clusters, kernel_length, init='k-means++', x_squared_norms=None,
                  random_state=None, **kwargs):
    """
    Compute initial kernels

    Parameters
    ----------
    X (numpy.ndarray):
        Training data. Rows are samples.
    n_clusters (int):
        Number of initial seed kernels
    kernel_length (int):
        Lenght of each kernel
    init ('k-means++', 'random', numpy.ndarray, or a function):
        Method for initialization. If it's a function, it should have this
        call signature:
        kernels, shifts = init(
            X, n_clusters, kernel_length, random_state, **kwargs).
        random_state must be a RandomState instance.
    x_squared_norms (numpy.ndarray or None):
        Equivalent to np.matmul(X, X.T). If None, it would be computed if
        init='kmeans++'.
    random_state (Int or RandomState instance):
        The generator used to initialize the kernels. Use int to make the
        randomness deterministic.
    **kwargs:
        If init=='kmeans++', the following keyword argument can be used
            n_local_trials (int):
                The number of seeding trials for each kernel (except the first),
                of which the one reducing inertia the most is greedily chosen.
                Set to None to make the number of trials depend logarithmically
                on the number of seeds (2+log(k)); this is the default.


    Returns
    -------
    kernels (numpy.ndarray):
        The kernel seeds
    shifts (numpy.ndarray):
        shifts[i] is the shift of kernels[i]
    """

    random_state = check_random_state(random_state)

    n_samples, sample_length = X.shape

    if isinstance(init, str) and init == 'k-means++':
        if x_squared_norms is None:
            x_squared_norms = row_norms(X, squared=True)
        if len(kwargs) == 0 or 'n_local_trials' not in kwargs:
            n_local_trials = None
        kernels, shifts = _kmeans_plus_plus(
            X, n_clusters, kernel_length, x_squared_norms,
            random_state, n_local_trials)
    elif isinstance(init, str) and init == 'random':
        seeds = random_state.permutation(n_samples)[:n_clusters]
        kernels = X[seeds]
        kernels = utils.pick_random_windows(kernels, 1, kernel_length,
                                            random_state).squeeze()
        n_offsets = sample_length - kernel_length + 1
        shifts = random_state.randint(0, n_offsets, size=n_clusters)
    elif hasattr(init, '__array__'):
        # ensure that the kernels have the same dtype as X
        # this is a requirement of fused types of cython
        kernels = np.array(init, dtype=X.dtype)
    elif callable(init):
        kernels, shifts = init(X, n_clusters, kernel_length, random_state, **kwargs)
        kernels = np.asarray(kernels, dtype=X.dtype)
        shifts = np.asarray(shifts, dtype=X.dtype)
    else:
        raise ValueError("the init parameter for the k-means should "
                         "be 'k-means++' or 'random' or an ndarray, "
                         "'%s' (type '%s') was passed." % (init, type(init)))

    return kernels, shifts


def _kmeans_plus_plus(X, n_clusters, kernel_length,
            x_squared_norms, random_state, n_local_trials=None):
    """
    Shift-invariant kmeans++

    This is a shift-invariant adapation to the implementation in scikit-learn.
    See http://bit.ly/sklearn_kmeans_pp

    Parameters
    ----------
    X (numpy.ndarray):
        Training data. Rows are samples.
    n_clusters (int):
        Number of initial seed kernels
    kernel_length (int):
        Lenght of each kernel
    x_squared_norms (numpy.ndarray):
        Equivalent to np.matmul(X, X.T)
    random_state (RandomState):
        The generator used to initialize the kernels.
    n_local_trials (int):
        The number of seeding trials for each kernel (except the first),
        of which the one reducing inertia the most is greedily chosen.
        Set to None to make the number of trials depend logarithmically
        on the number of seeds (2+log(k)); this is the default.

    Returns
    -------
    kernels (numpy.ndarray):
        The kernel seeds
    shifts (numpy.ndarray):
        shifts[i] is the shift of kernels[i] that minimizes the potential

    Notes
    -----
    Inertia, or potential, is the the sum of squared distances to the closest
    kernel, for all the samples.
    """

    n_samples, n_features = X.shape
    n_windows = n_features - kernel_length + 1

    kernels = np.empty((n_clusters, kernel_length), dtype=X.dtype)
    shifts = np.empty(n_clusters, dtype=np.intp)

    if n_local_trials is None:
        n_local_trials = 2 + int(np.log(n_clusters))

    # Pick first kernel randomly. Pick a sample, and get all the possible
    # windows (kernels). all_windows.shape=(1, n_windows, kernel_length).
    # XXX: Use a simpler function to get all the windows. They don't need to be
    # randomly selected.
    sample_id = random_state.randint(n_samples)
    if sp.issparse(X):
        all_windows = utils.pick_random_windows(X[sample_id].toarray(),
                                                n_windows, kernel_length,
                                                random_state)
    else:
        all_windows = utils.pick_random_windows(X[sample_id], n_windows,
                                                kernel_length, random_state)
    all_windows = all_windows.squeeze(axis=0)

    # Initialize list of closest distances.
    # closest_dist_sq.shape=(n_windows,n_windows,n_samples). The second
    # dimension is the number of shifts of each window, which is also equal to
    # the number of windows.
    closest_dist_sq = wrappers.shift_invariant_euclidean_distances(
        all_windows, X, X_norm_squared=x_squared_norms,
        squared=True)

    # Potential: the sum of squared distances to closest kernel
    # Compute potential for each shift and each window
    current_pot = closest_dist_sq.sum(axis=2)

    # Find best window and its best shift
    best_window_id, best_shift_id = np.unravel_index(np.argmin(current_pot),
                                                     current_pot.shape)

    # Update distances and potential to use best window and best shift
    closest_dist_sq = closest_dist_sq[best_window_id, best_shift_id]
    current_pot = current_pot[best_window_id, best_shift_id]

    # Update kernels and shifts
    kernels[0] = all_windows[best_window_id]
    shifts[0] = best_shift_id

    # Pick the remaining n_clusters-1 points
    for c in range(1, n_clusters):
        # Choose kernel candidates by sampling with probability proportional
        # to the squared distance to the closest existing kernel
        rand_vals = random_state.random_sample(n_local_trials) * current_pot
        candidate_ids = np.searchsorted(stable_cumsum(closest_dist_sq),
                                        rand_vals)
        # XXX: numerical imprecision can result in a candidate_id out of range
        np.clip(candidate_ids, None, closest_dist_sq.size - 1,
                out=candidate_ids)

        # Pick all windows from each candidate sample.
        # all_windows.shape=(n_local_trials,n_windows,kernel_length)
        if sp.issparse(X):
            all_windows = utils.pick_random_windows(
                X[candidate_ids].toarray(), n_windows, kernel_length,
                random_state)
        else:
            all_windows = utils.pick_random_windows(X[candidate_ids],
                                                    n_windows, kernel_length,
                                                    random_state)

        # Compute distances to kernel candidates for all windows
        distance_to_candidates = np.empty((n_local_trials, n_windows, n_windows, n_samples))
        for trial in np.arange(n_local_trials):
            distance_to_candidates[trial] = \
                wrappers.shift_invariant_euclidean_distances(all_windows[trial],
                                                             X,
                                                             X_norm_squared=x_squared_norms,
                                                             squared=True)

        # Update the distances so that distance_to_candidates[i][j][k] has the
        # distances that the problem would reach if the j-th window, shifted k
        # positions, of the i-th sample candidate is selected.
        np.minimum(distance_to_candidates, closest_dist_sq, out=distance_to_candidates)

        # Compute potential.
        # candidates_pot.shape=(n_local_trials, n_windows, n_windows)
        candidates_pot = distance_to_candidates.sum(axis=3)

        # Find best candidate, window and shift
        min_pot_id = np.argmin(candidates_pot)
        best_candidate_id, best_window_id, best_shift_id = \
            np.unravel_index(min_pot_id, candidates_pot.shape)

        # Choose best potential and distances
        current_pot = candidates_pot[best_candidate_id, best_window_id, best_shift_id]
        closest_dist_sq = distance_to_candidates[best_candidate_id,
                                                 best_window_id, best_shift_id]

        # Permanently add best kernel candidate, and its best shift, found in
        # local tries
        kernels[c] = all_windows[best_candidate_id, best_window_id]
        shifts[c] = best_shift_id

    return kernels, shifts


###############################################################################
# Main algorithm

def shift_invariant_k_means(X, n_clusters, kernel_length, init='k-means++',
                            n_init=10, max_iter=300, tol=1e-3,
                            random_state=None):
    """
    Shift-invariant k-means algorithm

    Parameters
    ----------
    X (numpy.ndarray):
        Data matrix with samples in its rows.
    n_clusters (int):
        Number of clusters to form, as well as the number of kernels to find.
    kernel_length (int):
        The length of each kernel.
    init ('k-means++', 'random', numpy.ndarray, or a function):
        Method for initialization. If it's a function, it should have this
        call signature:
        kernels, shifts = init(
             X, n_clusters, kernel_length, random_state, **kwargs).
        random_state must be a RandomState instance.
    n_init (int):
        The number of times the algorithm is run with different centroid seeds.
        The final results would be from the iteration where the inertia is the
        lowest.
    max_iter (init):
        Maximum number of iterations the algorithm will be run.
    tol (float):
        Upper bound that the squared Euclidean norm of the change in the
        kernels must achieve to declare convergence.
    random_state (int, RandomState instance or None):
        Determines random number generation for centroid initialization. Use an
        int to make the randomness deterministic.


    Returns
    -------
    kernels (numpy.ndarray):
        A matrix with the learned kernels in its rows.
    labels (numpy.ndarray):
        labels[i] is the index of the kernel (row of `kernels`) closest
        to the sample X[i].
    shifts (numpy.ndarray):
        time_shift[i] is the shift that minimizes the distance to the closest
        kernel to the sample X[i].
    inertia (float):
        The sum of squared Euclidean distances to the closest kernel of all the
        training samples.
    best_n_iter (int):
        Number of iterations needed to achieve convergence, according to `tol`.
    """

    random_state = check_random_state(random_state)

    best_labels, best_shifts = None, None
    best_kernels, best_inertia = None, None

    # subtract of mean of x for more accurate distance computations
    X_mean = X.mean(axis=0)
    X = X - X_mean

    # Precompute squared norms of rows of X for efficient computation of
    # Euclidean distances between kernels and samples.
    x_squared_norms = row_norms(X, squared=True)

    seeds = random_state.randint(np.iinfo(np.int32).max, size=n_init)
    for seed in seeds:
        # run a shift-invariant k-means once
        kernels, labels, shifts, inertia, n_iter_ = si_kmeans_single(
            X, n_clusters, kernel_length, max_iter, tol,
            x_squared_norms, random_state=seed)
        # determine if these results are the best so far
        if best_inertia is None or inertia < best_inertia:
            best_kernels = kernels.copy()
            best_labels = labels.copy()
            best_shifts = shifts.copy()
            best_inertia = inertia
            best_n_iter = n_iter_

    distinct_clusters = len(set(best_labels))
    if distinct_clusters < n_clusters:
        warnings.warn(
            "Number of distinct clusters ({}) found smaller than "
            "n_clusters ({}). Possibly due to duplicate points "
            "in X.".format(distinct_clusters, n_clusters), ConvergenceWarning,
            stacklevel=2
            )

    best_kernels += X_mean

    return best_kernels, best_labels, best_shifts, best_inertia, best_n_iter


def si_kmeans_single(X, n_clusters, kernel_length, init='k-means++',
                     max_iter=300, tol=1e-3, x_squared_norms=None,
                     random_state=None):
    """
    Single run of shift-invariant k-means
    """

    random_state = check_random_state(random_state)

    if x_squared_norms is None:
        x_squared_norms = row_norms(X, squared=True)

    best_labels, best_shifts = None, None
    best_kernels, best_inertia = None, None

    # Init
    kernels, _ = init_kernels(
        X, n_clusters, kernel_length, init, x_squared_norms, random_state)

    print('Initialization completed.')

    for iteration in range(max_iter):
        kernels_old = kernels.copy()
        labels, shifts, distances = _asignment_step(X, kernels, x_squared_norms)
        kernels = _kernels_update_step(
            X, n_clusters, labels, shifts, distances)

        inertia = distances.sum()
        print("Iteration %2d, inertia %.3f" % (i, inertia))

        if best_inertia is None or inertia < best_inertia:
            best_labels = labels.copy()
            best_shifts = shifts.copy()
            best_kernels = kernels.copy()
            best_inertia = inertia

        kernel_change = squared_norm(kernels_old - kernels)
        if kernel_change <= tol:
            if verbose:
                print("Converged at iteration %d: "
                      "kernel changes %e within tolerance %e"
                      % (iteration, kernel_change, tol))
            break

    if kernel_change > 0:
    # rerun assingment step in case of non-convergence so that predicted
    # labels match cluster centers
        best_labels, best_shifts, distances = _asignment_step(
            X, best_kernels, x_squared_norms, random_state)
        best_inertia = distances.sum()

    return best_kernels, best_labels, best_shifts, best_inertia, iteration+1


def _asignment_step(X, kernels, x_squared_norms):
    """
    Find the index of the shifted kernel that is closest to each sample

    Parameters
    ----------
    X (numpy.ndarray):
        Training data. Rows of X are samples.
    kernels (numpy.ndarray):
        Centroids of the clusters.
    x_squared_norms (numpy.ndarray):
        Squared euclidean norm of rows of X. This is used to speed up the
        computation of the Euclidean distances between samples and kernels.

    Returns
    -------
    labels (numpy.ndarray):
        labels[i] is the index of kernel closest to sample X[i]
    shifts (numpy.ndarray):
        shifts[i] is the best shift of the kernel closest to X[i]
    distances (numpy.ndarray):
        distances[i] is the distance of X[i] to the closest kernel
    """

    return wrappers.shift_invariant_pairwise_distances_argmin_min(
        X, kernels, x_squared_norms)
