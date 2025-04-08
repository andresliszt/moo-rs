import pytest
import numpy as np


from pymoors import (
    Nsga2,
    BitFlipMutation,
    SinglePointBinaryCrossover,
    RandomSamplingBinary,
    NoFeasibleIndividualsError,
)
from pymoors.typing import TwoDArray


def binary_biobjective_infeasible(genes: TwoDArray) -> TwoDArray:
    """
    Binary biobjective function with infeasible constraints.

    Parameters:
        genes (np.ndarray): Binary array of shape (n_individuals, n_genes).

    Returns:
        np.ndarray: Array of shape (n_individuals, 2) containing the two objective values.
    """
    # Objective 1: Sum of the genes
    f1 = np.sum(genes, axis=1)

    # Objective 2: Sum of the complementary genes (1 - gene)
    f2 = np.sum(1 - genes, axis=1)

    return np.column_stack((f1, f2))


def infeasible_constraints(genes: TwoDArray) -> TwoDArray:
    """
    Infeasible constraints: Sum of genes must be greater than the number of genes.
    This is impossible for binary genes.

    Parameters:
        genes (np.ndarray): Binary array of shape (n_individuals, n_genes).

    Returns:
        np.ndarray: Array of shape (n_individuals,) containing the constraint values.
    """
    n_genes = genes.shape[1]
    return (n_genes - np.sum(genes, axis=1) + 1).reshape(
        -1, 1
    )  # Sum of genes - n_genes + 1 > 0


def test_keep_infeasible():
    algorithm = Nsga2(
        sampler=RandomSamplingBinary(),
        mutation=BitFlipMutation(gene_mutation_rate=0.5),
        crossover=SinglePointBinaryCrossover(),
        fitness_fn=binary_biobjective_infeasible,
        constraints_fn=infeasible_constraints,
        n_vars=5,
        population_size=100,
        n_offsprings=32,
        n_iterations=20,
        mutation_rate=0.1,
        crossover_rate=0.9,
        duplicates_cleaner=None,
        keep_infeasible=True,
        verbose=False,
    )
    algorithm.run()

    assert len(algorithm.population) == 100


def test_keep_infeasible_out_of_bounds():
    algorithm = Nsga2(
        sampler=RandomSamplingBinary(),
        mutation=BitFlipMutation(gene_mutation_rate=0.5),
        crossover=SinglePointBinaryCrossover(),
        fitness_fn=binary_biobjective_infeasible,
        n_vars=5,
        population_size=100,
        n_offsprings=32,
        n_iterations=20,
        mutation_rate=0.1,
        crossover_rate=0.9,
        duplicates_cleaner=None,
        keep_infeasible=True,
        lower_bound=2,
        upper_bound=10,
    )
    algorithm.run()

    assert len(algorithm.population) == 100


def test_remove_infeasible():
    with pytest.raises(
        NoFeasibleIndividualsError,
        match="Error during evaluation: No feasible individuals found in the population",
    ):
        algorithm = Nsga2(
            sampler=RandomSamplingBinary(),
            mutation=BitFlipMutation(gene_mutation_rate=0.5),
            crossover=SinglePointBinaryCrossover(),
            fitness_fn=binary_biobjective_infeasible,
            constraints_fn=infeasible_constraints,
            n_vars=5,
            population_size=100,
            n_offsprings=100,
            n_iterations=20,
            mutation_rate=0.1,
            crossover_rate=0.9,
            duplicates_cleaner=None,
            keep_infeasible=False,
        )

        algorithm.run()
