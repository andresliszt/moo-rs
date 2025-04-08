"""This test is meant to check if all python exposed methods run without problems

We are not going to check the behavior of the operators in this test because that
is done extensively in the Rust modules where they are defined

"""

import numpy as np
import pytest

from pymoors import (
    UniformBinaryCrossover,
    ExponentialCrossover,
    SinglePointBinaryCrossover,
    OrderCrossover,
    SimulatedBinaryCrossover,
)


@pytest.mark.parametrize(
    "operator_class, kwargs",
    [
        # Add any required kwargs for each operator if needed.
        (OrderCrossover, {}),
        (UniformBinaryCrossover, {}),
        (ExponentialCrossover, {"exponential_crossover_rate": 0.7}),
        (SinglePointBinaryCrossover, {}),
        (SimulatedBinaryCrossover, {"distribution_index": 2}),
    ],
)
def test_crossover_exposed_methods(operator_class, kwargs):
    population_size = 5
    n_vars = 10
    # Create two parent populations as binary arrays (values 0.0 and 1.0)
    parents_a = np.random.randint(0, 2, size=(population_size, n_vars)).astype(
        np.float64
    )
    parents_b = np.random.randint(0, 2, size=(population_size, n_vars)).astype(
        np.float64
    )

    # Instantiate the crossover operator.
    op = operator_class(**kwargs)

    # Check that the operator's properties are correctly set.
    for k, v in kwargs.items():
        assert getattr(op, k) == v

    # Call the crossover method with a fixed seed.
    offspring = op.crossover(parents_a, parents_b, seed=42)
    offspring_same_seed = op.crossover(parents_a, parents_b, seed=42)

    np.testing.assert_array_equal(offspring, offspring_same_seed)

    # Offspring should have shape (2*population_size, n_vars)
    assert offspring.shape == (2 * population_size, n_vars)

    with pytest.raises(ValueError, match="parent_a numpy array must be 2D."):
        op.crossover(parents_a[0], parents_b[0], seed=42)

    with pytest.raises(ValueError, match="parent_b numpy array must be 2D."):
        op.crossover(parents_a, parents_b[0], seed=42)
