
```python
import numpy as np

from pymoors import (
    Nsga2,
    RandomSamplingBinary,
    BitFlipMutation,
    SinglePointBinaryCrossover,
    ExactDuplicatesCleaner,
)
from pymoors.typing import TwoDArray


PROFITS = np.array([2, 3, 6, 1, 4])
QUALITIES = np.array([5, 2, 1, 6, 4])
WEIGHTS = np.array([2, 3, 6, 2, 3])
CAPACITY = 7


def knapsack_fitness(genes: TwoDArray) -> TwoDArray:
    # Calculate total profit
    profit_sum = np.sum(PROFITS * genes, axis=1, keepdims=True)
    # Calculate total quality
    quality_sum = np.sum(QUALITIES * genes, axis=1, keepdims=True)

    # We want to maximize profit and quality,
    # so in pymoors we minimize the negative values
    f1 = -profit_sum
    f2 = -quality_sum
    return np.column_stack([f1, f2])


def knapsack_constraint(genes: TwoDArray) -> TwoDArray:
    # Calculate total weight
    weight_sum = np.sum(WEIGHTS * genes, axis=1, keepdims=True)
    # Inequality constraint: weight_sum <= capacity
    return weight_sum - CAPACITY


algorithm = Nsga2(
    sampler=RandomSamplingBinary(),
    crossover=SinglePointBinaryCrossover(),
    mutation=BitFlipMutation(gene_mutation_rate=0.5),
    fitness_fn=knapsack_fitness,
    constraints_fn=knapsack_constraint,
    duplicates_cleaner=ExactDuplicatesCleaner(),
    n_vars=5,
    population_size=32,
    num_offsprings=32,
    num_iterations=10,
    mutation_rate=0.1,
    crossover_rate=0.9,
    keep_infeasible=False,
)

algorithm.run()
pop = algorithm.population
# Get genes
>>> pop.genes
array([[1., 0., 0., 1., 1.],
       [0., 1., 0., 0., 1.],
       [1., 1., 0., 1., 0.],
       ...])
# Get fitness
>>> pop.fitness
array([[ -7., -15.],
       [ -7.,  -6.],
       [ -6., -13.],
       ...])
# Get constraints evaluation
>>> pop.constraints
array([[ 0.],
       [-1.],
       [ 0.],
       ...])
# Get rank
>>> pop.rank
array([0, 1, 1, 2, ...], dtype=uint64)
# Get best individuals
>>> pop.best
[<pymoors.schemas.Individual object at 0x...>]
>>> pop.best[0].genes
array([1., 0., 0., 1., 1.])
>>> pop.best[0].fitness
array([ -7., -15.])
>>> pop.best[0].constraints
array([0.])
```
