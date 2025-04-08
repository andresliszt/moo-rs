import numpy as np
import pytest
from itertools import permutations

from pymoors import (
    Nsga2,
    Nsga3,
    PermutationSampling,
    SwapMutation,
    OrderCrossover,
    ExactDuplicatesCleaner,
)

from pymoors.schemas import Individual
from pymoors.typing import TwoDArray


CITIES = np.array(
    [
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [1.0, 0.0],
    ]  # city 0  # city 1  # city 2  # city 3
)

# Toll costs (4x4); TOLL[i,j] is the toll from city i to city j.
TOLL = np.array(
    [[0, 2, 10, 2], [2, 0, 2, 10], [10, 2, 0, 2], [2, 10, 2, 0]], dtype=float
)


def compute_distance(i, j):
    """
    Euclidean distance between city i and j (indices in CITIES).
    """
    return np.linalg.norm(CITIES[i] - CITIES[j])


def route_distance(route):
    """
    Sum of Euclidean distances for a route, returning to start city.
    route is a sequence of city indices, e.g. [0,1,2,3].
    """
    total_dist = 0.0
    n = len(route)
    for idx in range(n):
        next_idx = (idx + 1) % n
        i = route[idx]
        j = route[next_idx]
        total_dist += compute_distance(i, j)
    return total_dist


def route_toll(route):
    """
    Sum of TOLL costs for a route, returning to start city.
    """
    total = 0.0
    n = len(route)
    for idx in range(n):
        next_idx = (idx + 1) % n
        i = route[idx]
        j = route[next_idx]
        total += TOLL[i, j]
    return total


def fitness_tsp_multiobjective(population_genes: np.ndarray) -> TwoDArray:
    """
    Multi-objective TSP fitness for a population of permutation routes.

    Parameters
    ----------
    population_genes : np.ndarray of shape (population_size, n_cities)
        Each row is a route (permutation of [0,1,2,3]).

    Returns
    -------
    np.ndarray of shape (population_size, 2)
        Each row: [distance, toll].
        Minimizing both distance and toll.
    """
    population_size = population_genes.shape[0]
    fitness = np.zeros((population_size, 2), dtype=float)
    for i in range(population_size):
        route = population_genes[i].astype(int)
        dist = route_distance(route)
        tl = route_toll(route)
        fitness[i] = [dist, tl]
    return fitness


def get_real_pareto_front() -> list[Individual]:
    """
    Enumerate all routes (all permutations of [0,1,2,3]), compute (dist, toll),
    and return the set of non-dominated solutions.

    Returns
    -------
    list of dict:
      Each dict has "route", "dist", "toll".
    """
    all_routes = []
    for perm in permutations([0, 1, 2, 3]):
        dist = route_distance(perm)
        tl = route_toll(perm)
        all_routes.append({"route": perm, "dist": dist, "toll": tl})

    # Identify non-dominated
    pareto = []
    for r1 in all_routes:
        d1, t1 = r1["dist"], r1["toll"]
        dominated = False
        for r2 in all_routes:
            d2, t2 = r2["dist"], r2["toll"]
            if (d2 <= d1 and t2 <= t1) and (d2 < d1 or t2 < t1):
                dominated = True
                break
        if not dominated:
            pareto.append(
                Individual(
                    genes=np.array(r1["route"]),
                    fitness=np.array([r1["dist"], r1["toll"]]),
                    rank=0,
                    constraints=None,
                )
            )
    return pareto


@pytest.mark.xfail(
    reason="Known issue https://github.com/andresliszt/pymoors/issues/89"
)
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
def test_tsp_multiobjective(algorithm_class, extra_kw, compare_exact_front):
    """
    Test a 4-city TSP with 2 objectives: distance and toll.
    We'll see if the final pareto front is equal to the real one
    """
    # 1) Create the algorithm
    algorithm = algorithm_class(
        sampler=PermutationSampling(),
        crossover=OrderCrossover(),
        mutation=SwapMutation(),
        fitness_fn=fitness_tsp_multiobjective,
        n_vars=4,  # 4 cities (the route length)
        population_size=100,
        n_offsprings=24,
        n_iterations=20,
        mutation_rate=0.1,
        crossover_rate=0.9,
        duplicates_cleaner=ExactDuplicatesCleaner(),
        keep_infeasible=False,
        verbose=False,
        **extra_kw,
    )

    algorithm.run()
    # Retrieve final population of solutions
    best = algorithm.population.best
    # Enumerate the exact Pareto front for the 5-item knapsack
    real_pareto_front = get_real_pareto_front()

    compare_exact_front(best, real_pareto_front)
