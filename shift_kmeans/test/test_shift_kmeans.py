"""Testing shift-invariant k-means"""

import pytest

import numpy as np

from shift_kmeans.datasets.dictionaries import sindict
from shift_kmeans.datasets.generator import make_samples_from_shifted_kernels
from shift_kmeans.shift_kmeans import shift_invariant_k_means


@pytest.fixture(scope="module")
def sinD():
    return sindict(n_atoms=5, n_features=10, dtype=np.float32)

@pytest.fixture(scope="module")
def input_data(sinD):
    return make_samples_from_shifted_kernels(n_samples=10,
                                             sample_length=15,
                                             dictionary=sinD,
                                             n_kernels_per_sample=1,
                                             scale=False,
                                             snr=10)

@pytest.fixture(scope="module")
def output_data(input_data):
    X, in_kernel_index, in_time_shift, _ = input_data
    return shift_invariant_k_means(X=X,
                                   n_clusters=5,
                                   kernel_length=10,
                                   n_init=10,
                                   max_iter=300,
                                   tol=1e-7,
                                   random_state=None)

class TestShiftInvariantKMeans:
    def test_shapes_and_types(self, output_data):
        kernels, out_kernel_index, out_time_shift,\
        inertia, best_n_iter = output_data

        assert isinstance(kernels, np.ndarray)
        assert isinstance(out_kernel_index, np.ndarray)
        assert isinstance(out_time_shift, np.ndarray)
        assert isinstance(inertia, np.float32)
        assert isinstance(best_n_iter, np.int32)

        assert kernels.shape == (5, 10)
        assert out_kernel_index.shape == (10, 1)
        assert out_time_shift.shape == (10, 1)
