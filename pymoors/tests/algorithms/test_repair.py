from typing import Any

import numpy as np
import pytest

from pymoors import (
    GaussianMutation,
    NoCrossover,
    NoMutation,
    NoRepair,
    Nsga2,
    RandomSamplingFloat,
    SimulatedBinaryCrossover,
)
from pymoors.typing import TwoDArray


def fitness_fn(genes: TwoDArray) -> TwoDArray:
    x = genes[:, 0]
    y = genes[:, 1]
    return np.column_stack((x**2 + y**2, (x - 1) ** 2 + (y - 1) ** 2))


class ClampRepair:
    def __init__(self, lower: float, upper: float):
        self.lower = lower
        self.upper = upper
        self.calls = 0

    def operate(self, population: TwoDArray) -> TwoDArray:
        self.calls += 1
        return np.clip(population, self.lower, self.upper)


def build_nsga2(**overrides: Any):
    kwargs: dict[str, Any] = dict(
        sampler=RandomSamplingFloat(min=0.0, max=1.0),
        crossover=SimulatedBinaryCrossover(distribution_index=15),
        mutation=GaussianMutation(gene_mutation_rate=0.5, sigma=2.0),
        fitness_fn=fitness_fn,
        num_vars=2,
        population_size=30,
        num_offsprings=30,
        num_iterations=10,
        verbose=False,
        seed=1,
    )
    kwargs.update(overrides)
    return Nsga2(**kwargs)


def test_custom_repair_is_applied():
    repair = ClampRepair(0.0, 1.0)
    algorithm = build_nsga2(repair=repair)
    algorithm.run()
    assert repair.calls > 0
    assert algorithm.population.genes.min() >= 0.0
    assert algorithm.population.genes.max() <= 1.0


def test_no_repair_runs():
    algorithm = build_nsga2(repair=NoRepair())
    algorithm.run()
    assert len(algorithm.population) > 0


def test_repair_without_operate_is_rejected():
    with pytest.raises(TypeError, match="operate"):
        build_nsga2(repair=object())


def test_no_mutation_no_crossover_run():
    algorithm = build_nsga2(mutation=NoMutation(), crossover=NoCrossover())
    algorithm.run()
    assert len(algorithm.population) > 0
