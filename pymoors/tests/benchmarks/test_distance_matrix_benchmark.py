import time

import pytest
import numpy as np
from scipy.spatial.distance import cdist  # type: ignore

from pymoors._pymoors import cross_euclidean_distances  # type: ignore


def test_cdist_pymoors(benchmark):
    matrix = np.random.rand(5000, 5000)
    result = benchmark(cross_euclidean_distances, matrix, matrix)
    assert result.shape == (5000, 5000)


def eculidean_distance_plus_time(matrix):
    start_time = time.perf_counter()
    result = cross_euclidean_distances(matrix, matrix)
    end_time = time.perf_counter()
    total = end_time - start_time
    return result, total


@pytest.mark.parametrize("n", [1000, 1500, 2000, 2500])
def test_compare_scipy_cdist_vs_pymoors(benchmark, n):
    matrix = np.random.rand(n, n)
    result_pymoors, total_time = benchmark(eculidean_distance_plus_time, matrix)
    # Measure the time for scipy's cdist using a timer
    start_time = time.perf_counter()
    result_scipy = cdist(matrix, matrix, metric="sqeuclidean")
    end_time = time.perf_counter()
    time_scipy = end_time - start_time
    # Check the result is the same
    np.testing.assert_allclose(result_pymoors, result_scipy, rtol=1e-5, atol=1e-8)
    # Check the time
    assert total_time < time_scipy
