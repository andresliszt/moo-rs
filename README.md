<p align="center">
  <img src="./pymoors/docs/images/moors-logo.png" alt="moo-rs logo" width="350"/>
</p>

# moo-rs
![License](https://img.shields.io/badge/License-MIT-blue.svg)
![Python Versions](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)
[![codecov](https://codecov.io/gh/andresliszt/moo-rs/graph/badge.svg?token=KC6EAVYGHX)](https://codecov.io/gh/andresliszt/moo-rs)
[![Docs](https://img.shields.io/website?label=Docs&style=flat&url=https%3A%2F%2Fandresliszt.github.io%2Fmoo-rs%2F)](https://andresliszt.github.io/moo-rs/)
[![CodSpeed Badge](https://img.shields.io/endpoint?url=https://codspeed.io/badge.json)](https://codspeed.io/andresliszt/moo-rs)

## Overview

`moo-rs` is a project for solving multi-objective optimization problems with evolutionary algorithms, combining:

- **moors**: a pure-Rust crate for high-performance implementations of genetic algorithms
- **pymoors**: a Python extension crate (via [pyo3](https://github.com/PyO3/pyo3)) exposing `moors` algorithms with a Pythonic API

Inspired by the amazing Python project [pymoo](https://github.com/anyoptimization/pymoo), `moo-rs` delivers both the speed of Rust and the ease-of-use of Python.

## Project Structure

```
moo-rs/
├── moors/    # Rust crate: core algorithms
└── pymoors/  # Python crate: pyo3 bindings
```

## moors (Rust)

**moors** is a pure-Rust crate providing multi-objective evolutionary algorithms.

### Features

- NSGA-II, NSGA-III, R-NSGA-II, Age-MOEA, REVEA (many more coming soon!)
- Pluggable operators: sampling, crossover, mutation, duplicates removal
- Flexible fitness & constraints via user-provided closures
- Built on [ndarray](https://github.com/rust-ndarray/ndarray) and [faer](https://github.com/sarah-quinones/faer-rs)

### Installation

```toml
[dependencies]
moors = { git = "https://github.com/andresliszt/moo-rs", package = "moors" }
```

> **Note**: publishing **moors** on crates.io is in progress.

### Quickstart

```rust
use ndarray::{Axis, stack};
use moors::{
    algorithms::Nsga2,
    duplicates::CloseDuplicatesCleaner,
    genetic::{PopulationGenes, PopulationFitness},
    operators::sampling::RandomSamplingFloat,
    operators::crossover::sbx::SimulatedBinaryCrossover,
    operators::mutation::gaussian::GaussianMutation,
};

// Define your fitness function:
fn fitness(pop: &PopulationGenes) -> PopulationFitness {
    let x = pop.column(0);
    let y = pop.column(1);
    let f1 = &x * &x + &y * &y;
    let f2 = (&x - 1.0).mapv(|v| v * v) + (&y - 1.0).mapv(|v| v * v);
    stack(Axis(1), &[f1.view(), f2.view()]).unwrap()
}

fn main() {
    let sampler = RandomSamplingFloat::new(0.0, 1.0);
    let crossover = SimulatedBinaryCrossover::new(15.0);
    let mutation = GaussianMutation::new(0.5, 0.01);
    let cleaner = CloseDuplicatesCleaner::new(1e-8);

    let mut nsga2 = Nsga2::new(
        sampler,
        crossover,
        mutation,
        cleaner,
        fitness,
        2,    // n_vars
        50,   // population_size
        50,   // n_offsprings
        100,  // n_iterations
        0.1,  // mutation_rate
        0.9,  // crossover_rate
        false, // keep_infeasible
    ).unwrap();

    nsga2.inner.run().unwrap();
    println!("Done! Population size: {}", nsga2.inner.population.len());
}
```

## pymoors (Python)

**pymoors** uses [pyo3](https://github.com/PyO3/pyo3) to expose `moors` algorithms in Python.

### Installation

```bash
pip install pymoors
```

### Quickstart

```python
import numpy as np
from pymoors import (
    Nsga2,
    RandomSamplingBinary,
    BitFlipMutation,
    SinglePointBinaryCrossover,
    ExactDuplicatesCleaner,
)

# Example fitness function
def knapsack_fitness(genes):
    PROFITS = np.array([2, 3, 6, 1, 4])
    QUALITY = np.array([5, 2, 1, 6, 4])
    profit = np.sum(PROFITS * genes, axis=1, keepdims=True)
    quality = np.sum(QUALITY * genes, axis=1, keepdims=True)
    return np.column_stack([-profit, -quality])

algorithm = Nsga2(
    sampler=RandomSamplingBinary(),
    crossover=SinglePointBinaryCrossover(),
    mutation=BitFlipMutation(gene_mutation_rate=0.5),
    fitness_fn=knapsack_fitness,
    n_vars=5,
    population_size=32,
    n_offsprings=32,
    n_iterations=10,
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

In this small example, the algorithm finds a single solution on the Pareto front: selecting the items **(A, D, E)**, with a profit of **7** and a quality of **15**. This means there is no other combination that can match or exceed *both* objectives without exceeding the knapsack capacity (7).

Once the algorithm finishes, it stores a `population` attribute that contains all the individuals evaluated during the search.

## Contributing

Contributions welcome! Please read the [contribution guide](https://andresliszt.github.io/moo-rs/development) and open issues or PRs in the relevant crate’s repository

## License

This project is licensed under the MIT License.
```
