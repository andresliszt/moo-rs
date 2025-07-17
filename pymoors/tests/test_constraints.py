# tests/test_constraints.py

import numpy as np
import pytest

from pymoors import ExponentialCrossover, GaussianMutation, RandomSamplingFloat, Spea2
from pymoors.constraints import Constraints


def test_init_requires_at_least_one_source():
    with pytest.raises(ValueError):
        Constraints()


def test_only_constraints_fn():
    constr = Constraints(constraints_fn=lambda genes: genes * 3)
    genes = np.array([[1.0, 2.0], [3.0, 4.0]])
    result = constr(genes)
    # result should be exactly fn(genes)
    np.testing.assert_array_equal(result, genes * 3)


def test_only_lower_bound():
    lb = 5.0
    constr = Constraints(lower_bound=lb)
    genes = np.array([[1.0, 2.0], [3.0, 4.0]])
    expected = np.array([[4.0, 3.0], [2.0, 1.0]])  # lb - genes
    np.testing.assert_array_equal(constr(genes), expected)


def test_only_upper_bound():
    ub = 10.0
    constr = Constraints(upper_bound=ub)
    genes = np.array([[1.0, 2.0], [3.0, 4.0]])
    expected = np.array([[-9.0, -8.0], [-7.0, -6.0]])  # genes - ub
    np.testing.assert_array_equal(constr(genes), expected)


def test_lower_and_upper_bound_columnar():
    lb, ub = 2.0, 8.0
    constr = Constraints(lower_bound=lb, upper_bound=ub)
    genes = np.array([[1.0, 3.0], [9.0, 5.0]])
    part1 = lb - genes
    part2 = genes - ub
    expected = np.concatenate([part1, part2], axis=1)
    result = constr(genes)
    # shape should be (2, 4) since each part is 2‑columns:
    assert result.shape == (2, 4)
    np.testing.assert_array_equal(result, expected)


def test_constraints_fn_and_lower_bound_columnar():
    lb = 1.0
    constr = Constraints(constraints_fn=lambda genes: genes, lower_bound=lb)
    genes = np.array([[0.0, 2.0], [3.0, 4.0]])
    part1 = genes
    part2 = lb - genes
    expected = np.concatenate([part1, part2], axis=1)
    result = constr(genes)
    assert result.shape == (2, 4)
    np.testing.assert_array_equal(result, expected)


def test_constraints_fn_and_upper_bound_columnar():
    ub = 3.0
    constr = Constraints(
        constraints_fn=lambda genes: np.full_like(genes, 2.0), upper_bound=ub
    )
    genes = np.array([[1.0, 2.0], [3.0, 4.0]])
    part1 = np.array([[2.0, 2.0], [2.0, 2.0]])
    part2 = genes - ub
    expected = np.concatenate([part1, part2], axis=1)
    result = constr(genes)
    assert result.shape == (2, 4)
    np.testing.assert_array_equal(result, expected)


def test_all_three_sources_columnar():
    lb, ub = 1.0, 5.0
    constr = Constraints(
        constraints_fn=lambda genes: np.zeros_like(genes),
        lower_bound=lb,
        upper_bound=ub,
    )
    genes = np.array([[1.0, 2.0], [6.0, 4.0]])
    part1 = np.array([[0.0, 0.0], [0.0, 0.0]])
    part2 = lb - genes
    part3 = genes - ub
    expected = np.concatenate([part1, part2, part3], axis=1)
    result = constr(genes)
    # shape should be (2, 6) = 3 blocks × 2 columns each
    assert result.shape == (2, 6)
    np.testing.assert_array_equal(result, expected)


def test_run_algorithm_with_constraints_class():
    constraints = Constraints(
        constraints_fn=lambda genes: genes, lower_bound=-5, upper_bound=-1
    )
    algorithm = Spea2(
        sampler=RandomSamplingFloat(min=-5, max=-1),
        mutation=GaussianMutation(gene_mutation_rate=0.5, sigma=0.01),
        crossover=ExponentialCrossover(exponential_crossover_rate=0.9),
        fitness_fn=lambda genes: genes,
        constraints_fn=constraints,
        num_vars=2,
        population_size=5,
        num_offsprings=5,
        num_iterations=2,
        mutation_rate=0.1,
        crossover_rate=0.9,
        duplicates_cleaner=None,
        keep_infeasible=False,
    )

    algorithm.run()
    # Expected constraints
    expected_constraints = np.concatenate(
        [
            algorithm.population.genes,
            -5.0 - algorithm.population.genes,
            algorithm.population.genes + 1.0,
        ],
        axis=1,
    )
    np.testing.assert_array_equal(
        algorithm.population.constraints, expected_constraints
    )


def test_run_algorithm_with_constraints_1d_function():
    algorithm = Spea2(
        sampler=RandomSamplingFloat(min=-5, max=-1),
        mutation=GaussianMutation(gene_mutation_rate=0.5, sigma=0.01),
        crossover=ExponentialCrossover(exponential_crossover_rate=0.9),
        fitness_fn=lambda genes: genes,
        constraints_fn=Constraints(constraints_fn=lambda genes: genes.sum(axis=1)),
        num_vars=2,
        population_size=5,
        num_offsprings=5,
        num_iterations=2,
        mutation_rate=0.1,
        crossover_rate=0.9,
        duplicates_cleaner=None,
        keep_infeasible=False,
    )

    algorithm.run()
    # Expected constraints
    expected_constraints = algorithm.population.genes.sum(axis=1).reshape(-1, 1)
    np.testing.assert_array_equal(
        algorithm.population.constraints, expected_constraints
    )
