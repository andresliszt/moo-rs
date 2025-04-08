"""This test is meant to check if all python exposed methods run without problems

We are not going to check the behavior of the operators in this test because that
is done extensively in the Rust modules where they are defined.
"""

import numpy as np
import pytest

from pymoors import (
    BitFlipMutation,
    SwapMutation,
    GaussianMutation,
    DisplacementMutation,
    ScrambleMutation,
)


@pytest.mark.parametrize(
    "operator_class, kwargs, pop_type",
    [
        (BitFlipMutation, {"gene_mutation_rate": 0.5}, "binary"),
        (SwapMutation, {}, "binary"),
        (GaussianMutation, {"gene_mutation_rate": 0.5, "sigma": 0.1}, "any"),
        (DisplacementMutation, {}, "any"),
        (ScrambleMutation, {}, "any"),
    ],
)
def test_mutation_exposed_methods(operator_class, kwargs, pop_type):
    population_size = 5
    n_vars = 10

    # Create population based on the operator type.
    if pop_type == "binary":
        # For binary operators, use an array of 0.0 and 1.0.
        population = np.random.randint(0, 2, size=(population_size, n_vars)).astype(
            np.float64
        )
    else:
        # For operators that expect real numbers, use random floats.
        population = np.random.rand(population_size, n_vars)

    # Instantiate the mutation operator.
    op = operator_class(**kwargs)

    # Check that the operator's properties are correctly set.
    for k, v in kwargs.items():
        assert getattr(op, k) == v

    # Call the mutation method.
    mutated = op.mutate(population, seed=42)
    mutated_same_seed = op.mutate(population, seed=42)
    mutated_no_seed = op.mutate(population)

    np.testing.assert_array_equal(mutated, mutated_same_seed)

    # Check that the output has the same shape as the input.
    assert mutated.shape == mutated_no_seed.shape == (population_size, n_vars)

    with pytest.raises(ValueError, match="Population numpy array must be 2D."):
        op.mutate(population[0], seed=42)
