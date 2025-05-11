# moors
![License](https://img.shields.io/badge/License-MIT-blue.svg)
[![codecov](https://codecov.io/gh/andresliszt/moo-rs/graph/badge.svg?token=KC6EAVYGHX?flag=moors)](https://codecov.io/gh/andresliszt/moo-rs?flag=moors)
[![crates.io](https://img.shields.io/crates/v/moors.svg)](https://crates.io/crates/moors)
[![crates.io downloads](https://img.shields.io/crates/d/moors.svg)](https://crates.io/crates/moors)

## Overview

`moors` is the core crate of the [moo-rs](https://github.com/andresliszt/moo-rs/) project for solving multi-objective optimization problems with evolutionary algorithms. It's a pure-Rust crate for high-performance implementations of genetic algorithms

## Features

- NSGA-II, NSGA-III, R-NSGA-II, Age-MOEA, REVEA, SPEA-II (many more coming soon!)
- Pluggable operators: sampling, crossover, mutation, duplicates removal
- Flexible fitness & constraints via user-provided closures
- Built on [ndarray](https://github.com/rust-ndarray/ndarray) and [faer](https://github.com/sarah-quinones/faer-rs)

## Installation

```toml
[dependencies]
moors = "0.1.1"
```

## Quickstart

```rust

use ndarray::{Array1, Axis, stack};

use moors::{
    algorithms::{MultiObjectiveAlgorithmError, Nsga2Builder},
    duplicates::ExactDuplicatesCleaner,
    genetic::{PopulationConstraints, PopulationFitness, PopulationGenes},
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
fn fitness_knapsack(population_genes: &PopulationGenes) -> PopulationFitness {
    let weights_arr = Array1::from_vec(WEIGHTS.to_vec());
    let values_arr = Array1::from_vec(VALUES.to_vec());

    let total_values = population_genes.dot(&values_arr);
    let total_weights = population_genes.dot(&weights_arr);

    // stack two columns: [-total_values, total_weights]
    stack(Axis(1), &[(-&total_values).view(), total_weights.view()]).expect("stack failed")
}

fn constraints_knapsack(population_genes: &PopulationGenes) -> PopulationConstraints {
    let weights_arr = Array1::from_vec(WEIGHTS.to_vec());
    (population_genes.dot(&weights_arr) - CAPACITY).insert_axis(Axis(1))
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
        .num_objectives(2)
        .num_constraints(1)
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

## Contributing

Contributions welcome! Please read the [contribution guide](https://andresliszt.github.io/moo-rs/development) and open issues or PRs in the relevant crate’s repository

## License

This project is licensed under the MIT License.
