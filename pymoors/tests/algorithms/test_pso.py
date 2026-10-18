from typing import Any

import numpy as np
import pytest

from pymoors import InvalidParameterError, MoEtpso
from pymoors.typing import TwoDArray


def fitness_fn(genes: TwoDArray) -> TwoDArray:
    x = genes[:, 0]
    y = genes[:, 1]
    return np.column_stack((x**2 + y**2, (x - 1) ** 2 + (y - 1) ** 2))


def build_pso(**overrides: Any):
    kwargs: dict[str, Any] = dict(
        fitness_fn=fitness_fn,
        num_vars=2,
        population_size=40,
        num_iterations=50,
        lower_bound=0.0,
        upper_bound=1.0,
        verbose=False,
        seed=1,
    )
    kwargs.update(overrides)
    return MoEtpso(**kwargs)


def test_pso_run():
    algorithm = build_pso(population_size=100, num_iterations=300)
    algorithm.run()
    population = algorithm.population
    assert len(population) > 0
    assert population.genes.min() >= 0.0
    assert population.genes.max() <= 1.0
    # The Pareto set of this problem is the segment x == y in [0, 1].
    assert np.allclose(population.genes[:, 0], population.genes[:, 1], atol=0.05)


def test_pso_with_constraints():
    algorithm = build_pso(
        constraints_fn=lambda genes: (genes[:, 0] + genes[:, 1] - 1.0)[:, np.newaxis]
    )
    algorithm.run()
    population = algorithm.population
    assert np.all(population.genes.sum(axis=1) <= 1.0 + 1e-6)


def test_pso_is_reproducible_with_seed():
    first = build_pso()
    second = build_pso()
    first.run()
    second.run()
    np.testing.assert_allclose(first.population.fitness, second.population.fitness)


def test_pso_population_before_run_raises():
    with pytest.raises(Exception):
        _ = build_pso().population


def test_pso_invalid_bounds():
    with pytest.raises(InvalidParameterError):
        build_pso(lower_bound=1.0, upper_bound=0.0)
