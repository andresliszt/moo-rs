# pymoors
![License](https://img.shields.io/badge/License-MIT-blue.svg)
![Python Versions](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)
[![codecov](https://codecov.io/gh/andresliszt/moo-rs/graph/badge.svg?token=KC6EAVYGHX)](https://codecov.io/gh/andresliszt/moo-rs)
[![Docs](https://img.shields.io/website?label=Docs&style=flat&url=https%3A%2F%2Fandresliszt.github.io%2Fmoo-rs%2F)](https://andresliszt.github.io/moo-rs/)
[![CodSpeed Badge](https://img.shields.io/endpoint?url=https://codspeed.io/badge.json)](https://codspeed.io/andresliszt/moo-rs)

## Overview

This project aims to solve multi-objective optimization problems using genetic algorithms. It is a hybrid implementation combining Python and Rust, leveraging the power of [PyO3](https://github.com/PyO3/pyo3) to bridge the two languages. This project was inspired in the amazing python project [pymoo](https://github.com/anyoptimization/pymoo).

## Features

- **Multi-Objective Optimization**: Implementations of popular multi-objective genetic algorithms like NSGA-II and NSGA-III.
- **Hybrid Python and Rust**: Core computational components are written in Rust for performance, while the user interface and high-level API are provided in Python.
- **Extensible Operators**: Support for various sampling, mutation, crossover, and survival operators.
- **Constraint Handling**: Ability to handle constraints in optimization problems.


## Installation

To install the package you simply can do

```sh
pip install pymoors
```

## Small Example

Here is an example of how to use the `pymoors` package to solve a small problem using the NSGA-II algorithm:

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
    n_offsprings=32,
    n_iterations=10,
    mutation_rate=0.1,
    crossover_rate=0.9,
    keep_infeasible=False,
)

algorithm.run()
```

In this **small example**, the algorithm finds a **single** solution on the Pareto front: selecting the items **(A, D, E)**, with a profit of **7** and a quality of **15**. This means there is no other combination that can match or exceed *both* objectives without exceeding the knapsack capacity (7).

Once the algorithm finishes, it stores a `population` attribute that contains all the individuals evaluated during the search.

```python
pop = algorithm.population
# Get genes
>>> pop.genes
array([[1., 0., 0., 1., 1.],
       [0., 1., 0., 0., 1.],
       [1., 1., 0., 1., 0.],
       [0., 0., 0., 1., 1.],
       [1., 0., 0., 0., 1.],
       [1., 0., 0., 1., 0.],
       [1., 1., 0., 0., 0.],
       [0., 0., 1., 0., 0.],
       [0., 1., 0., 1., 0.],
       [1., 0., 0., 0., 0.],
       [0., 0., 0., 1., 0.],
       [0., 0., 0., 0., 1.],
       [0., 1., 0., 0., 0.],
       [0., 0., 0., 0., 0.]])
# Get fitness
>>> pop.fitness
array([[ -7., -15.],
       [ -7.,  -6.],
       [ -6., -13.],
       [ -5., -10.],
       [ -6.,  -9.],
       [ -3., -11.],
       [ -5.,  -7.],
       [ -6.,  -1.],
       [ -4.,  -8.],
       [ -2.,  -5.],
       [ -1.,  -6.],
       [ -4.,  -4.],
       [ -3.,  -2.],
       [ -0.,  -0.]])
# Get constraints evaluation
>>> pop.constraints
array([[ 0.],
       [-1.],
       [ 0.],
       [-2.],
       [-2.],
       [-3.],
       [-2.],
       [-1.],
       [-2.],
       [-5.],
       [-5.],
       [-4.],
       [-4.],
       [-7.]])
# Get rank
>>> pop.rank
array([0, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 6], dtype=uint64)
```

Note that in this example there is just one individual with rank 0, i.e Pareto optimal. Algorithms in `pymoors` store all individuals with rank 0 in a special attribute `best`, which is list of  `pymoors.schemas.Individual ` objects

```python
# Get best individuals
best = pop.best
>>> best
[<pymoors.schemas.Individual object at 0x11b8ec110>]
# In this small exmaple as mentioned, best is just one single individual (A, D, E)
>>> best[0].genes
array([1., 0., 0., 1., 1.])
>>> best[0].fitness
array([ -7., -15.])
>>> best[0].constraints
array([0.])
```
