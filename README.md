<p align="center">
  <img src="./pymoors/docs/images/moors-logo.png" alt="moo-rs logo" width="350"/>
</p>

# moo-rs
![License](https://img.shields.io/badge/License-MIT-blue.svg)
![Python Versions](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)
[![PyPI version](https://img.shields.io/pypi/v/pymoors.svg)](https://pypi.org/project/pymoors/)
[![crates.io](https://img.shields.io/crates/v/moors.svg)](https://crates.io/crates/moors)
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

- NSGA-II, NSGA-III, R-NSGA-II, Age-MOEA, REVEA, SPEA-II (many more coming soon!)
- Pluggable operators: sampling, crossover, mutation, duplicates removal
- Flexible fitness & constraints via user-provided closures
- Built on [ndarray](https://github.com/rust-ndarray/ndarray) and [faer](https://github.com/sarah-quinones/faer-rs)

### Installation

```toml
[dependencies]
moors = "0.1.1"
```

### Quickstart

```rust

use ndarray::{Array1, Array2, Axis, stack};

use moors::{
    algorithms::{MultiObjectiveAlgorithmError, Nsga2Builder},
    duplicates::ExactDuplicatesCleaner,
    operators::{
        crossover::SinglePointBinaryCrossover, mutation::BitFlipMutation,
        sampling::RandomSamplingBinary,
    },
};

// problem data
const WEIGHTS: [f64; 5] = [12.0, 2.0, 1.0, 4.0, 10.0];
const VALUES: [f64; 5] = [4.0, 2.0, 1.0, 5.0, 3.0];
const CAPACITY: f64 = 15.0;

/// Compute multi-objective fitness [–total_value, total_weight]
/// Returns an Array2<f64> of shape (population_size, 2)
fn fitness_knapsack(population_genes: &Array2<f64>) -> Array2<f64> {
    let weights_arr = Array1::from_vec(WEIGHTS.to_vec());
    let values_arr = Array1::from_vec(VALUES.to_vec());

    let total_values = population_genes.dot(&values_arr);
    let total_weights = population_genes.dot(&weights_arr);

    // stack two columns: [-total_values, total_weights]
    stack(Axis(1), &[(-&total_values).view(), total_weights.view()]).expect("stack failed")
}

fn constraints_knapsack(population_genes: &Array2<f64>) -> Array1<f64> {
    let weights_arr = Array1::from_vec(WEIGHTS.to_vec());
    population_genes.dot(&weights_arr) - CAPACITY
}

fn main() -> Result<(), MultiObjectiveAlgorithmError> {
    // build the NSGA-II algorithm
    let mut algorithm = Nsga2Builder::default()
        .fitness_fn(fitness_knapsack)
        .constraints_fn(constraints_knapsack)
        .sampler(RandomSamplingBinary::new())
        .crossover(SinglePointBinaryCrossover::new())
        .mutation(BitFlipMutation::new(0.5))
        .duplicates_cleaner(ExactDuplicatesCleaner::new())
        .num_vars(5)
        .population_size(100)
        .crossover_rate(0.9)
        .mutation_rate(0.1)
        .num_offsprings(32)
        .num_iterations(2)
        .build()?;

    algorithm.run()?;
    let population = algorithm.population()?;
    println!("Done! Population size: {}", population.len());

    Ok(())
}
```

### Duplicates Cleaner

In the example above, we use `ExactDuplicatesCleaner` to remove duplicated individuals that are exactly the same (hash-based deduplication). The `CloseDuplicatesCleaner` is also available—it accepts an `epsilon` parameter to drop any individuals within an ε-ball. Note that eliminating duplicates can be computationally expensive; if you do not want to use a duplicate cleaner, pass `moors::NoDuplicatesCleaner` to the builder.


### Constraints

In **moors**, constraints (if provided) are used to enforce feasibility:

- **Feasibility dominates everything**: any individual with all constraint values ≤ 0 is preferred over any infeasible one.
- If both tournament participants are infeasible, the one with the smaller sum of positive violations wins.
- All constraints must evaluate to ≤ 0. For “>” constraints, invert the inequality in your function (multiply by –1).
- Equality constraints use an ε-tolerance:

  $$g_{\text{ineq}}(x) = \bigl|g_{\text{eq}}(x)\bigr| - \varepsilon \;\le\; 0.$$

```rust
use ndarray::{Array2, Array1, Axis};

const EPSILON: f64 = 1e-6;

/// Returns an Array2 of shape (n, 2) containing two constraints for each row [x, y]:
/// - Column 0: |x + y - 1| - EPSILON ≤ 0 (equality with ε-tolerance)
/// - Column 1: x² + y² - 1.0 ≤ 0 (unit circle inequality)
fn constraints(genes: &Array2<f64>) -> Array2<f64> {
    // Constraint 1: |x + y - 1| - EPSILON
    let eq = genes.map_axis(Axis(1), |row| (row[0] + row[1] - 1.0).abs() - EPSILON);
    // Constraint 2: x^2 + y^2 - 1
    let ineq = genes.map_axis(Axis(1), |row| row[0].powi(2) + row[1].powi(2) - 1.0);
    // Stack into two columns
    stack(Axis(1), &[eq.view(), ineq.view()]).unwrap()
}
```

We know! we don't want to be stacking and using the epsilon technique everywhere. We have a helper macro to build the constraints for us

```rust
use ndarray::{Array2, Array1, Axis};

use moors::impl_constraints_fn;

const EPSILON: f64 = 1e-6;

/// Equality constraint x + y = 1
fn constraints_eq(genes: &Array2<f64>) -> Array1<f64> {
    genes.map_axis(Axis(1), |row| row[0] + row[1] - 1.0)
}

/// Inequality constraint: x² + y² - 1 ≤ 0
fn constraints_ineq(genes: &Array2<f64>) -> Array1<f64> {
    genes.map_axis(Axis(1), |row| row[0].powi(2) + row[1].powi(2) - 1.0)
}

constraints_fn!(
    MyConstraints,
    ineq = [g_ineq],
    eq   = [h_eq],
);

```

The macro above will generate a struct `MyConstraints` that can be passed to any `moors` algorithm. This macro accepts optional arguments `lower_bound` and `upper_bound` both for now `f64` that are used to control the lower and upper bound of each gene.


If the optimization problem does not have any constraint, then use in the builder the struct `moors::NoConstraints`


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
    num_objectives=1,
    num_constraints=1,
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

In this small example, the algorithm finds a single solution on the Pareto front: selecting the items **(A, D, E)**, with a profit of **7** and a quality of **15**. This means there is no other combination that can match or exceed *both* objectives without exceeding the knapsack capacity (7).

Once the algorithm finishes, it stores a `population` attribute that contains all the individuals evaluated during the search.

## Contributing

Contributions welcome! Please read the [contribution guide](https://andresliszt.github.io/moo-rs/development) and open issues or PRs in the relevant crate’s repository

## License

This project is licensed under the MIT License.
