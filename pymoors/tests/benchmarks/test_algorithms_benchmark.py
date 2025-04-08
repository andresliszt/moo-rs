import numpy as np

from pymoors import (
    Nsga2,
    RandomSamplingFloat,
    GaussianMutation,
    CloseDuplicatesCleaner,
    SimulatedBinaryCrossover,
)
from pymoors.typing import TwoDArray, OneDArray


N_VARS: int = 10


def f1(x: OneDArray) -> float:
    """Objective 1."""
    return sum(xi**2 for xi in x)


def f2(x: OneDArray) -> float:
    """Objective 2"""
    return sum((xi - 1) ** 2 for xi in x)


def fitness_biobjective(population_genes: TwoDArray) -> TwoDArray:
    population_size = population_genes.shape[0]
    fitness = np.zeros((population_size, 2), dtype=float)
    for i in range(population_size):
        x = population_genes[i]
        fitness[i, 0] = f1(x)
        fitness[i, 1] = f2(x)
    return fitness


def test_small_real_biobjective_nsag2(benchmark):
    algorithm = Nsga2(
        sampler=RandomSamplingFloat(min=0.0, max=1.0),
        crossover=SimulatedBinaryCrossover(distribution_index=2),
        mutation=GaussianMutation(gene_mutation_rate=0.1, sigma=0.05),
        fitness_fn=fitness_biobjective,
        n_vars=N_VARS,
        population_size=1000,
        n_offsprings=1000,
        n_iterations=100,
        mutation_rate=0.1,
        crossover_rate=0.9,
        duplicates_cleaner=CloseDuplicatesCleaner(epsilon=1e-5),
        keep_infeasible=False,
        lower_bound=0,
        upper_bound=1,
        verbose=False,
    )
    benchmark(algorithm.run)

    assert len(algorithm.population) == 1000
    assert len(algorithm.population.best) == 1000
