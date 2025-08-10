# tests/test_constraints.py

import numpy as np
import pytest

from pymoors import ExponentialCrossover, GaussianMutation, RandomSamplingFloat, Spea2
from pymoors.constraints import Constraints


# -----------------------
# __init__ validation
# -----------------------


def test_init_requires_at_least_one_source():
    with pytest.raises(ValueError):
        Constraints()


def test_init_rejects_negative_epsilon():
    def eq_fn(g):
        return g[:, 0]

    with pytest.raises(ValueError):
        Constraints(eq=eq_fn, epsilon=-1e-3)


def test_empty_list_is_treated_as_none():
    with pytest.raises(ValueError):
        Constraints(eq=[], ineq=[])


# -----------------------
# Only one source paths (len(parts) == 1)
# -----------------------


def test_only_lower_bound():
    lb = 5.0
    constr = Constraints(lower_bound=lb)
    genes = np.array([[1.0, 2.0], [3.0, 4.0]])
    expected = lb - genes
    np.testing.assert_array_equal(constr(genes), expected)


def test_only_upper_bound():
    ub = 10.0
    constr = Constraints(upper_bound=ub)
    genes = np.array([[1.0, 2.0], [3.0, 4.0]])
    expected = genes - ub
    np.testing.assert_array_equal(constr(genes), expected)


def test_only_ineq_1d_is_reshaped():
    def ineq_fn(g):
        return g.sum(axis=1)

    constr = Constraints(ineq=ineq_fn)
    genes = np.array([[1.0, 2.0], [3.0, 4.0]])
    expected = genes.sum(axis=1).reshape(-1, 1)
    np.testing.assert_array_equal(constr(genes), expected)


def test_only_eq_2d_is_abs_minus_epsilon():
    eps = 0.1

    def eq2(g):
        x, y = g[:, 0], g[:, 1]
        return np.column_stack([x + y - 1.0, x - y])

    constr = Constraints(eq=eq2, epsilon=eps)
    genes = np.array([[0.2, 0.9], [0.5, 0.5]])
    raw = eq2(genes)
    expected = np.abs(raw) - eps
    np.testing.assert_array_equal(constr(genes), expected)


# -----------------------
# Mixed sources & order: eq(ε) → ineq → bounds
# -----------------------


def test_eq_then_ineq_then_bounds_concatenation_order():
    eps = 0.05

    def eq_fn1(g):
        return g.sum(axis=1)  # (n,)

    def eq_fn2(g):
        return g[:, :1] * 0.0 + 0.2  # (n,1)

    def ineq_fn1(g):
        return g  # (n,d)

    def ineq_fn2(g):
        return g[:, 0] - 2.0  # (n,)

    lb, ub = -1.0, 3.0
    constr = Constraints(
        eq=[eq_fn1, eq_fn2],
        ineq=[ineq_fn1, ineq_fn2],
        lower_bound=lb,
        upper_bound=ub,
        epsilon=eps,
    )
    genes = np.array([[0.0, 2.0], [3.0, 4.0]])  # n=2, d=2

    eq1 = np.abs(genes.sum(axis=1).reshape(-1, 1)) - eps
    eq2 = np.abs(np.full((genes.shape[0], 1), 0.2)) - eps
    in1 = genes
    in2 = (genes[:, 0] - 2.0).reshape(-1, 1)
    lbv = lb - genes
    ubv = genes - ub

    expected = np.concatenate([eq1, eq2, in1, in2, lbv, ubv], axis=1)
    out = constr(genes)

    assert out.shape == expected.shape
    np.testing.assert_array_equal(out, expected)


# -----------------------
# Normalization errors (coverage of _normalize_output)
# -----------------------


def test_ineq_1d_wrong_rows_raises():
    def bad_ineq(g):
        return np.ones(g.shape[0] - 1)

    constr = Constraints(ineq=bad_ineq)
    with pytest.raises(ValueError):
        constr(np.zeros((4, 2)))


def test_eq_2d_wrong_rows_raises():
    def bad_eq(g):
        return np.zeros((g.shape[0] + 1, 1))

    constr = Constraints(eq=bad_eq)
    with pytest.raises(ValueError):
        constr(np.zeros((3, 2)))


def test_eq_ndim_gt_2_raises():
    def bad_eq(g):
        return np.zeros((g.shape[0], 2, 2))  # ndim=3

    constr = Constraints(eq=bad_eq)
    with pytest.raises(ValueError):
        constr(np.zeros((3, 2)))


# -----------------------
# List handling and 1D/2D mixing
# -----------------------


def test_ineq_list_mixed_1d_2d_concatenates():
    def f1(g):
        return g[:, 0]  # (n,)

    def f2(g):
        return g  # (n,d)

    constr = Constraints(ineq=[f1, f2])
    genes = np.array([[1.0, 2.0], [3.0, 4.0]])
    expected = np.concatenate([genes[:, 0:1], genes], axis=1)
    np.testing.assert_array_equal(constr(genes), expected)


def test_eq_list_mixed_1d_2d_with_epsilon():
    eps = 0.2

    def f1(g):
        return g[:, 0] - 1.0

    def f2(g):
        return np.column_stack([g[:, 0] + g[:, 1], g[:, 0] - g[:, 1]])

    constr = Constraints(eq=[f1, f2], epsilon=eps)
    genes = np.array([[0.5, 0.5], [2.0, -1.0]])
    raw = np.concatenate(
        [
            (genes[:, 0] - 1.0).reshape(-1, 1),
            np.column_stack([genes[:, 0] + genes[:, 1], genes[:, 0] - genes[:, 1]]),
        ],
        axis=1,
    )
    expected = np.abs(raw) - eps
    np.testing.assert_array_equal(constr(genes), expected)


# -----------------------
# Bounds with eq-only / ineq-only
# -----------------------


def test_eq_with_bounds():
    eps = 0.1

    def eq_fn(g):
        return g[:, 0] + g[:, 1] - 1.0

    lb, ub = -0.5, 1.5
    constr = Constraints(eq=eq_fn, lower_bound=lb, upper_bound=ub, epsilon=eps)
    genes = np.array([[0.2, 0.9], [1.0, -0.2]])
    eq_part = np.abs((genes[:, 0] + genes[:, 1] - 1.0).reshape(-1, 1)) - eps
    expected = np.concatenate([eq_part, lb - genes, genes - ub], axis=1)
    np.testing.assert_array_equal(constr(genes), expected)


def test_ineq_with_bounds():
    def ineq_fn(g):
        return g[:, 0] - 0.5

    lb, ub = 0.0, 2.0
    constr = Constraints(ineq=ineq_fn, lower_bound=lb, upper_bound=ub)
    genes = np.array([[0.2, 0.9], [1.0, 1.5]])
    ineq_part = (genes[:, 0] - 0.5).reshape(-1, 1)
    expected = np.concatenate([ineq_part, lb - genes, genes - ub], axis=1)
    np.testing.assert_array_equal(constr(genes), expected)


# -----------------------
# Integration with algorithm
# -----------------------


def test_run_algorithm_with_constraints_class_using_ineq_and_bounds():
    def ineq_identity(g):
        return g

    constraints = Constraints(ineq=ineq_identity, lower_bound=-5, upper_bound=-1)
    algorithm = Spea2(
        sampler=RandomSamplingFloat(min=-5, max=-1),
        mutation=GaussianMutation(gene_mutation_rate=0.5, sigma=0.01),
        crossover=ExponentialCrossover(exponential_crossover_rate=0.9),
        fitness_fn=lambda genes: genes,
        constraints_fn=constraints,  # Constraints is callable
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
    expected_constraints = np.concatenate(
        [
            algorithm.population.genes,  # ineq
            -5.0 - algorithm.population.genes,  # lower bound violations
            algorithm.population.genes + 1.0,  # upper bound violations (genes - (-1))
        ],
        axis=1,
    )
    np.testing.assert_array_equal(
        algorithm.population.constraints, expected_constraints
    )
