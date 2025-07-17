import numpy as np

from pymoors import Nsga2
from pymoors.typing import TwoDArray


def dummy_fitness(genes: TwoDArray) -> TwoDArray:
    return genes


class CustomBinaryMutation:
    def __init__(self, gene_mutation_rate: float = 0.5):
        self.gene_mutation_rate = gene_mutation_rate

    def operate(
        self,
        population: TwoDArray,
    ) -> TwoDArray:
        mask = np.random.random(population.shape) < self.gene_mutation_rate
        population[mask] = 1.0 - population[mask]
        return population


class CustomBinaryCrossover:
    def operate(
        self,
        parents_a: TwoDArray,
        parents_b: TwoDArray,
    ) -> TwoDArray:
        n_pairs, n_genes = parents_a.shape
        offsprings = np.empty((2 * n_pairs, n_genes), dtype=parents_a.dtype)
        for i in range(n_pairs):
            a = parents_a[i]
            b = parents_b[i]
            point = np.random.randint(1, n_genes)
            c1 = np.concatenate((a[:point], b[point:]))
            c2 = np.concatenate((b[:point], a[point:]))
            offsprings[2 * i] = c1
            offsprings[2 * i + 1] = c2
        return offsprings


class CustomBinarySampling:
    def operate(self):
        return np.array(
            [
                [1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1],
            ]
        ).astype(float)


def test_algorithm_with_custom_operators():
    algorithm = Nsga2(
        sampler=CustomBinarySampling(),
        mutation=CustomBinaryMutation(),
        crossover=CustomBinaryCrossover(),
        fitness_fn=dummy_fitness,
        num_vars=5,
        population_size=100,
        num_offsprings=32,
        num_iterations=20,
        mutation_rate=0.1,
        crossover_rate=0.9,
        duplicates_cleaner=None,
        keep_infeasible=True,
        verbose=False,
    )
    algorithm.run()

    assert len(algorithm.population) == 100
