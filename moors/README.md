# moors

> _Evolution is a mystery_

![License](https://img.shields.io/badge/License-MIT-blue.svg)
[![codecov](https://codecov.io/gh/andresliszt/moo-rs/graph/badge.svg?token=KC6EAVYGHX?flag=moors)](https://codecov.io/gh/andresliszt/moo-rs?flag=moors)
[![crates.io](https://img.shields.io/crates/v/moors.svg)](https://crates.io/crates/moors)
[![crates.io downloads](https://img.shields.io/crates/d/moors.svg)](https://crates.io/crates/moors)

## Overview

`moors` is the core crate of the [moo-rs](https://github.com/andresliszt/moo-rs/) project for solving multi-objective optimization problems with evolutionary algorithms. It's a pure-Rust crate for high-performance implementations of genetic algorithms

## Features
- Single Objective Genetic Algorithms (SO-GA)
- NSGA-II, NSGA-III, R-NSGA-II, Age-MOEA, REVEA, SPEA-II (many more coming soon!)
- Pluggable operators: sampling, crossover, mutation, duplicates removal
- Flexible fitness & constraints via user-provided closures
- Built on [ndarray](https://github.com/rust-ndarray/ndarray) and [faer](https://github.com/sarah-quinones/faer-rs)

## Installation

```toml
[dependencies]
moors = "0.2.3"
```

## Quickstart

```rust

use ndarray::{Array1, Array2, Axis, stack};

use moors::{
    algorithms::{AlgorithmError, Nsga2Builder},
    duplicates::ExactDuplicatesCleaner,
    operators::{SinglePointBinaryCrossover, BitFlipMutation, RandomSamplingBinary},
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

fn main() -> Result<(), AlgorithmError> {
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

## Duplicates Cleaner

In the example above, we use `ExactDuplicatesCleaner` to remove duplicated individuals that are exactly the same (hash-based deduplication). The `CloseDuplicatesCleaner` is also available—it accepts an `epsilon` parameter to drop any individuals within an ε-ball. Note that eliminating duplicates can be computationally expensive; if you do not want to use a duplicate cleaner, pass `moors::NoDuplicatesCleaner` to the builder.


## Constraints

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
    ineq = [constraints_ineq],
    eq   = [constraints_eq],
);

```

The macro above will generate a struct `MyConstraints` that can be passed to any `moors` algorithm. This macro accepts optional arguments `lower_bound` and `upper_bound` both for now `f64` that are used to control the lower and upper bound of each gene.


If the optimization problem does not have any constraint, then use in the builder the struct `moors::NoConstraints`


## Contributing

Contributions welcome! Please read the [contribution guide](https://andresliszt.github.io/moo-rs/development) and open issues or PRs in the relevant crate’s repository

## License

This project is licensed under the MIT License.
