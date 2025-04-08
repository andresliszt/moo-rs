import numpy as np
import pytest

from pymoors.schemas import TwoDArray, Individual
from pymoors import (
    Nsga2,
    Nsga3,
    BitFlipMutation,
    SinglePointBinaryCrossover,
    RandomSamplingBinary,
    ExactDuplicatesCleaner,
)

##############################################################################
# Example: Small Knapsack-Like Problem
#
# Items (5):
#   - weights = [12, 2, 1, 4, 10]
#   - values  = [4,  2, 1, 5,  3]
# Capacity = 15
#
# Objectives:  (like a multi-objective knapsack)
#   1) - total_value  (i.e. maximizing total value)
#   2)   total_weight (trying to keep it as low as possible),
#      subject to weight <= 15
#
# We can explicitly compute the known Pareto front by enumerating all feasible
# subsets, computing the objectives, then identifying the non-dominated ones.
##############################################################################

# 1. Define problem data
WEIGHTS = np.array([12, 2, 1, 4, 10], dtype=float)
VALUES = np.array([4, 2, 1, 5, 3], dtype=float)
CAPACITY = 15.0


def fitness_knapsack(population_genes: TwoDArray) -> TwoDArray:
    """
    Compute the multi-objective fitness for the small knapsack problem.

    Parameters
    ----------
    population_genes : np.ndarray of shape (population_size, n_items)
        Binary matrix representing the population (1 = item included).

    Returns
    -------
    np.ndarray of shape (population_size, 2)
        Each row contains: [ -total_value, total_weight ]
    """
    total_values = population_genes @ VALUES
    total_weights = population_genes @ WEIGHTS

    # Combine objectives into a fitness matrix
    # Objective 1: - total_value  (since we want to maximize the actual value)
    # Objective 2:   total_weight
    fitness_matrix = np.column_stack((-total_values, total_weights))
    return fitness_matrix


def constraints_knapsack(population_genes: TwoDArray) -> TwoDArray:
    """
    Compute the constraints for the knapsack problem.

    Returns
    -------
    np.ndarray of shape (population_size, 1)
        [ total_weight - CAPACITY ]
    Constraint is satisfied if <= 0.
    """
    total_weights = population_genes @ WEIGHTS
    return (total_weights - CAPACITY)[:, np.newaxis]


def get_real_pareto_front() -> list[Individual]:
    """
    Enumerate all subsets of the 5 items, collect feasible ones,
    and compute their 2D objectives. Then filter out the non-dominated solutions
    to get the exact Pareto front.

    Returns
    -------
    list of Individual
        Each Individual represents a Pareto-optimal subset.
    """
    n_items = 5
    all_solutions = []
    for subset_mask in range(1 << n_items):  # 2^5 = 32 possibilities
        # Convert subset_mask to binary vector of length n_items
        genes = [(subset_mask >> i) & 1 for i in range(n_items)]
        genes = np.array(genes[::-1], dtype=float)  # item0 in left position, etc.

        total_weight = genes @ WEIGHTS
        if total_weight <= CAPACITY:
            # Feasible
            total_value = genes @ VALUES
            # Collect both the genes and the objective values
            all_solutions.append(
                {
                    "genes": genes,
                    "fitness": [-total_value, total_weight],
                    "constraints": [total_weight - CAPACITY],
                }
            )

    # Now, identify the non-dominated ones
    # "Dominates" means strictly better in at least one objective
    # and not worse in the other.
    pareto_set = []
    for s1 in all_solutions:
        o1 = s1["fitness"]
        dominated = False
        for s2 in all_solutions:
            o2 = s2["fitness"]
            # Check if s2 dominates s1
            if (o2[0] <= o1[0]) and (o2[1] <= o1[1]) and (o2 != o1):
                # s2 is better or equal in all objectives, strictly better in at least one
                dominated = True
                break
        if not dominated:
            pareto_set.append(
                Individual(
                    genes=s1["genes"],
                    fitness=np.array(s1["fitness"]),
                    rank=0,
                    constraints=np.array(s1["constraints"]),
                )
            )

    return pareto_set


@pytest.mark.parametrize(
    "algorithm_class, extra_kw",
    [
        (Nsga2, {}),
        (
            Nsga3,
            {
                "reference_points": np.array(
                    [
                        [0.0, 1.0],
                        [0.2, 0.8],
                        [0.4, 0.6],
                        [0.6, 0.4],
                        [0.8, 0.2],
                        [1.0, 0.0],
                    ]
                )
            },
        ),
    ],
)
def test_knapsack(algorithm_class, extra_kw, compare_exact_front):
    """
    Test that algorithms can find the known Pareto front
    for a small 5-item knapsack problem.
    """
    # Build the algorithm with small problem settings
    algorithm = algorithm_class(
        sampler=RandomSamplingBinary(),
        crossover=SinglePointBinaryCrossover(),
        mutation=BitFlipMutation(gene_mutation_rate=0.5),
        fitness_fn=fitness_knapsack,
        constraints_fn=constraints_knapsack,
        duplicates_cleaner=ExactDuplicatesCleaner(),
        n_vars=5,  # 5 items
        population_size=100,  # population size
        n_offsprings=32,  # offsprings per generation
        n_iterations=2,  # Note that with only 2 iteration we reach the opt
        mutation_rate=0.1,
        crossover_rate=0.9,
        keep_infeasible=False,
        **extra_kw,
    )

    algorithm.run()

    best = algorithm.population.best

    real_pareto_front = get_real_pareto_front()

    compare_exact_front(best, real_pareto_front)
