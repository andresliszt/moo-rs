import numpy as np

from pymoors import (
    CloseDuplicatesCleaner,
    Constraints,
    GaussianMutation,
    GeneticAlgorithmSOO,
    RandomSamplingFloat,
    SimulatedBinaryCrossover,
)


def test_run_algorithm_1d():
    """
    Main components of this algorithms are tested in moors
    A simple test min x + y s.t x,y in [1, 2]. The min is trivial
    x = 1 and y = 1

    """
    algorithm = GeneticAlgorithmSOO(
        sampler=RandomSamplingFloat(min=0.0, max=10.0),
        mutation=GaussianMutation(gene_mutation_rate=0.5, sigma=0.01),
        crossover=SimulatedBinaryCrossover(distribution_index=15.0),
        fitness_fn=lambda genes: genes.sum(axis=1),
        constraints_fn=Constraints(lower_bound=1.0, upper_bound=2.0),
        num_vars=2,
        population_size=50,
        num_offsprings=50,
        num_iterations=20,
        mutation_rate=0.1,
        crossover_rate=0.9,
        duplicates_cleaner=CloseDuplicatesCleaner(epsilon=1e-6),
        keep_infeasible=True,
    )

    algorithm.run()

    assert len(algorithm.population.best) == 1
    best_genes = np.array([1.0, 1.0])
    np.testing.assert_array_equal(best_genes, algorithm.population.best[0].genes)
