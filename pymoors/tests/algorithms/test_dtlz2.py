"""DTLZ2 is a classical multi-objective benchmark problem.

The decision vector x is split into two parts:
- The first (n_obj - 1) variables define the position on the Pareto front.
- The remaining variables (k = n_var - n_obj + 1) are used to compute the auxiliary function g.

The objective functions are defined as:
    f_i(x) = (1 + g(x_M)) * prod_{j=1}^{n_obj-i-1} cos(x_j * pi/2) * (if i > 0 then sin(x_{n_obj-i} * pi/2))

For this example, we set:
- n_var = 50 (to make the problem computationally more expensive)
- n_obj = 3
"""

import numpy as np
import pytest

from pymoors import (
    Nsga2,
    RandomSamplingFloat,
    GaussianMutation,
    CloseDuplicatesCleaner,
    ExactDuplicatesCleaner,
    SimulatedBinaryCrossover,
)
from pymoors.typing import TwoDArray

N_VARS = 50
N_OBJ = 3


def fitness_dtlz2(genes: TwoDArray) -> TwoDArray:
    # k is the number of variables that contribute to the auxiliary function g.
    k = N_VARS - N_OBJ + 1

    # Compute g as the sum of squared deviations from 0.5 for the last k decision variables.
    g = np.sum((genes[:, -k:] - 0.5) ** 2, axis=1)

    # Compute each objective.
    F = []
    for i in range(N_OBJ):
        # Start with the term (1 + g)
        f_i = 1 + g
        # Multiply by cos(x_j * pi/2) for j = 0 to n_obj-i-2
        for j in range(N_OBJ - i - 1):
            f_i *= np.cos(genes[:, j] * np.pi / 2)
        # For i > 0, multiply by sin(x_{n_obj-i-1} * pi/2)
        if i > 0:
            f_i *= np.sin(genes[:, N_OBJ - i - 1] * np.pi / 2)
        F.append(f_i)

    # Stack the list into a 2D array of shape (n_samples, n_obj)
    return np.column_stack(F)


@pytest.mark.parametrize(
    "duplicates_cleaner",
    [CloseDuplicatesCleaner(epsilon=1e-5), ExactDuplicatesCleaner()],
)
def test_drop_duplicates_dtlz2(duplicates_cleaner):
    algorithm = Nsga2(
        sampler=RandomSamplingFloat(min=0.0, max=1.0),
        crossover=SimulatedBinaryCrossover(distribution_index=2),
        mutation=GaussianMutation(gene_mutation_rate=0.1, sigma=0.05),
        fitness_fn=fitness_dtlz2,
        n_vars=50,
        population_size=1000,
        n_offsprings=100,
        n_iterations=50,
        mutation_rate=0.1,
        crossover_rate=0.9,
        duplicates_cleaner=duplicates_cleaner,
        keep_infeasible=False,
        lower_bound=0,
        upper_bound=1,
        verbose=False,
    )
    algorithm.run()
    # We check duplicates
    assert (
        len(algorithm.population)
        == len(np.unique(algorithm.population.genes, axis=0))
        == 1000
    )
