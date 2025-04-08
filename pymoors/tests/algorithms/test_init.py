import pytest
import numpy as np

from pymoors import (
    RandomSamplingFloat,
    GaussianMutation,
    SimulatedBinaryCrossover,
    AgeMoea,
    Nsga2,
    Nsga3,
    Rnsga2,
    Revea,
    DanAndDenisReferencePoints,
    ExactDuplicatesCleaner,
)
from pymoors.typing import TwoDArray


def fitness(population_genes: TwoDArray) -> TwoDArray:
    x = population_genes[:, 0]
    y = population_genes[:, 1]
    z = population_genes[:, 2]
    f1_vals = x**2 + y**2 + z**2
    f2_vals = (x - 1) ** 2 + (y - 1) ** 2 + (z - 1) ** 2
    return np.column_stack((f1_vals, f2_vals))


@pytest.fixture
def common_kwargs():
    return {
        "sampler": RandomSamplingFloat(min=0.0, max=1.0),
        "crossover": SimulatedBinaryCrossover(distribution_index=15),
        "mutation": GaussianMutation(gene_mutation_rate=0.1, sigma=0.05),
        "fitness_fn": fitness,
        "n_vars": 3,  # We have 2 variables: x,y
        "population_size": 100,
        "n_offsprings": 10,
        "n_iterations": 10,
        "mutation_rate": 0.1,
        "crossover_rate": 0.9,
        "lower_bound": 0,
        "upper_bound": 1,
    }


@pytest.mark.parametrize(
    "algorithm_class, algorithm_specific_kwargs",
    [
        (Nsga2, {}),
        (AgeMoea, {}),
        (
            Nsga3,
            {
                "reference_points": DanAndDenisReferencePoints(
                    n_reference_points=10, n_objectives=2
                )
            },
        ),
        (Rnsga2, {"reference_points": np.array([[0.5, 0.5]])}),
        (
            Revea,
            {
                "reference_points": DanAndDenisReferencePoints(
                    n_reference_points=10, n_objectives=2
                )
            },
        ),
        (
            Revea,
            {
                "reference_points": DanAndDenisReferencePoints(
                    n_reference_points=10, n_objectives=2
                ).generate()
            },
        ),
    ],
)
def test_init_minimal_args(algorithm_class, algorithm_specific_kwargs, common_kwargs):
    _ = algorithm_class(
        **common_kwargs,
        **algorithm_specific_kwargs,
    )
    # TODO: Enable once the attribute is created
    # assert algorithm.initialized = False


@pytest.mark.parametrize(
    "algorithm_class, algorithm_specific_kwargs",
    [
        (Nsga2, {}),
        (AgeMoea, {}),
        (
            Nsga3,
            {
                "reference_points": DanAndDenisReferencePoints(
                    n_reference_points=10, n_objectives=2
                )
            },
        ),
        (Rnsga2, {"reference_points": np.array([[0.5, 0.5]])}),
        (
            Revea,
            {
                "reference_points": DanAndDenisReferencePoints(
                    n_reference_points=10, n_objectives=2
                )
            },
        ),
    ],
)
def test_init_full_args(algorithm_class, algorithm_specific_kwargs, common_kwargs):
    extra_args = {
        "duplicates_cleaner": ExactDuplicatesCleaner(),
        "constraints_fn": lambda x: (x[:, 0] + x[:, 1] + x[:, 2] - 2)[:, np.newaxis],
        "keep_infeasible": True,
        "seed": 1,
    }

    _ = algorithm_class(**common_kwargs, **algorithm_specific_kwargs, **extra_args)
    # TODO: Enable once the attribute is created
    # assert algorithm.initialized = False


def test_nsga3_fails(common_kwargs):
    with pytest.raises(
        TypeError,
        match="reference_points must be either a custom reference points class or a NumPy array.",
    ):
        _ = Nsga3(**common_kwargs, reference_points=[[1, 0], [0, 1]])  # type: ignore


def test_revea_fails(common_kwargs):
    with pytest.raises(
        TypeError,
        match="reference_points must be either a custom reference points class or a NumPy array.",
    ):
        _ = Revea(**common_kwargs, reference_points=[[1, 0], [0, 1]])  # type: ignore
