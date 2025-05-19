# import numpy as np


# from pymoors import (
#     Nsga2,
#     SinglePointBinaryCrossover,
#     RandomSamplingBinary,
# )
# from pymoors.typing import TwoDArray


# def dummy_fitness(genes: TwoDArray) -> TwoDArray:
#     return genes


# class CustomBitFlipMutation:
#     def __init__(self, gene_mutation_rate: float = 0.5):
#         self.gene_mutation_rate = gene_mutation_rate

#     def operate(
#         self,
#         population: np.ndarray,
#     ) -> np.ndarray:
#         mask = np.random.random(population.shape) < self.gene_mutation_rate
#         population[mask] = 1.0 - population[mask]
#         return population


# def test_algorithm_with_custom_operators():
#     algorithm = Nsga2(
#         sampler=RandomSamplingBinary(),
#         mutation=CustomBitFlipMutation(gene_mutation_rate=0.5),
#         crossover=SinglePointBinaryCrossover(),
#         fitness_fn=dummy_fitness,
#         num_vars=5,
#         num_objectives=5,
#         num_constraints=0,
#         population_size=100,
#         num_offsprings=32,
#         num_iterations=20,
#         mutation_rate=0.1,
#         crossover_rate=0.9,
#         duplicates_cleaner=None,
#         keep_infeasible=True,
#         verbose=False,
#     )
#     algorithm.run()

#     assert len(algorithm.population) == 100
