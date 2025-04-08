"""This test is meant to check if all python exposed methods run without problems

We are not going to check the behavior of the operators in this test because that
is done extensively in the Rust modules where they are defined

"""

import numpy as np
import pytest

from pymoors import (
    RandomSamplingBinary,
    RandomSamplingInt,
    RandomSamplingFloat,
    PermutationSampling,
)


@pytest.mark.parametrize(
    "operator_class, kwargs",
    [
        # Add any required kwargs for each operator if needed.
        (RandomSamplingFloat, {"min": 1, "max": 10}),
        (RandomSamplingInt, {"min": 1, "max": 10}),
        (RandomSamplingBinary, {}),
        (PermutationSampling, {}),
    ],
)
def test_sampling_exposed_methods(operator_class, kwargs):
    population_size = 5
    n_vars = 10
    # Instantiate the crossover operator.
    op = operator_class(**kwargs)

    # Check that the operator's properties are correctly set.
    for k, v in kwargs.items():
        assert getattr(op, k) == v

    # Call the crossover method with a fixed seed.
    sampled_population = op.sample(population_size, n_vars, seed=42)
    sampled_population_seed = op.sample(population_size, n_vars, seed=42)
    sampled_population_no_seed = op.sample(population_size, n_vars)

    np.testing.assert_array_equal(sampled_population, sampled_population_seed)

    # Offspring should have shape (2*population_size, n_vars)
    assert (
        sampled_population_seed.shape
        == sampled_population_no_seed.shape
        == (population_size, n_vars)
    )
