
















```Rust
use ndarray::{Array1, Array2, Axis, stack, Ix1, Ix2};
use ordered_float::OrderedFloat;
use std::collections::HashSet;

use moors::{
    algorithms::{Nsga2, Nsga2Builder},
    duplicates::ExactDuplicatesCleaner,
    operators::{BitFlipMutation, RandomSamplingBinary, SinglePointBinaryCrossover},
    genetic::Population
};

// problem data
const PROFITS: [f64; 5] = [2.0, 3.0, 6.0, 1.0, 4.0];
const QUALITIES: [f64; 5] = [5.0, 2.0, 1.0, 6.0, 4.0];
const WEIGHTS: [f64; 5] = [2.0, 3.0, 6.0, 2.0, 3.0];
const CAPACITY: f64 = 7.0;

fn fitness_knapsack(populationulation_genes: &Array2<f64>) -> Array2<f64> {
    // Calculate total profit
    let profits_arr = Array1::from_vec(PROFITS.to_vec());
    let profit_sum = populationulation_genes.dot(&profits_arr);

    // Calculate total quality
    let qualities_arr = Array1::from_vec(QUALITIES.to_vec());
    let quality_sum = populationulation_genes.dot(&qualities_arr);

    // We want to maximize profit and quality,
    // so in moors we minimize the negative values
    stack(Axis(1), &[(-&profit_sum).view(), (-&quality_sum).view()]).expect("stack failed")
}

fn constraints_knapsack(populationulation_genes: &Array2<f64>) -> Array1<f64> {
    // Calculate total weight
    let weights_arr = Array1::from_vec(WEIGHTS.to_vec());
    // Inequality constraint: weight_sum <= capacity
    populationulation_genes.dot(&weights_arr) - CAPACITY
}

// NOTE: The clone is only needed for the notebook source of this file. Also, most of the cases
// you don't need to specify the Population<Ix2, Ix1> signature
let population: Population<Ix2, Ix1> = {
    let mut algorithm = Nsga2Builder::default()
        .fitness_fn(fitness_knapsack)
        .constraints_fn(constraints_knapsack)
        .sampler(RandomSamplingBinary)
        .crossover(SinglePointBinaryCrossover)
        .mutation(BitFlipMutation::new(0.5))
        .duplicates_cleaner(ExactDuplicatesCleaner)
        .num_vars(5)
        .population_size(16)
        .num_offsprings(16)
        .num_iterations(10)
        .mutation_rate(0.1)
        .crossover_rate(0.9)
        .keep_infeasible(false)
        .build()
        .unwrap();

    algorithm.run().expect("NSGA2 run failed");

    let population = algorithm.population.expect("populationulation should have been initialized");
    population.clone()
};
```

    Warning: Only 15 offspring were generated out of the desired 16.

























































```rust
// Get genes
>>> population.genes
[[1.0, 0.0, 0.0, 1.0, 1.0],
 [1.0, 1.0, 0.0, 1.0, 0.0],
 [0.0, 1.0, 0.0, 0.0, 1.0],
 [1.0, 0.0, 0.0, 0.0, 1.0],
 [1.0, 0.0, 0.0, 1.0, 0.0],
 [0.0, 0.0, 0.0, 1.0, 1.0],
 [0.0, 0.0, 1.0, 0.0, 0.0],
 [1.0, 1.0, 0.0, 0.0, 0.0],
 [0.0, 1.0, 0.0, 1.0, 0.0],
 [0.0, 0.0, 0.0, 1.0, 0.0],
 [0.0, 0.0, 0.0, 0.0, 1.0],
 [1.0, 0.0, 0.0, 0.0, 0.0],
 [0.0, 1.0, 0.0, 0.0, 0.0],
 [0.0, 0.0, 0.0, 0.0, 0.0]], shape=[14, 5], strides=[5, 1], layout=Cc (0x5), const ndim=2

```























































```rust
// Get fitness
>>> population.fitness
[[-7.0, -15.0],
 [-6.0, -13.0],
 [-7.0, -6.0],
 [-6.0, -9.0],
 [-3.0, -11.0],
 [-5.0, -10.0],
 [-6.0, -1.0],
 [-5.0, -7.0],
 [-4.0, -8.0],
 [-1.0, -6.0],
 [-4.0, -4.0],
 [-2.0, -5.0],
 [-3.0, -2.0],
 [-0.0, -0.0]], shape=[14, 2], strides=[2, 1], layout=Cc (0x5), const ndim=2

```























































```rust
// Get constraints
>>> population.constraints
[0.0, 0.0, -1.0, -2.0, -3.0, -2.0, -1.0, -2.0, -2.0, -5.0, -4.0, -5.0, -4.0, -7.0], shape=[14], strides=[1], layout=CFcf (0xf), const ndim=1

```























































```rust
// Get rank (for Nsga2)
>>> population.constraints
[0.0, 0.0, -1.0, -2.0, -3.0, -2.0, -1.0, -2.0, -2.0, -5.0, -4.0, -5.0, -4.0, -7.0], shape=[14], strides=[1], layout=CFcf (0xf), const ndim=1

```


Note that in this example there is just one individual with rank 0, i.e Pareto optimal. Algorithms in `moors` store all individuals with rank 0 in a special attribute `best`






















































```rust
>>> let best = population.best();
>>> best
Population { genes: [[1.0, 0.0, 0.0, 1.0, 1.0]], shape=[1, 5], strides=[5, 1], layout=CFcf (0xf), const ndim=2, fitness: [[-7.0, -15.0]], shape=[1, 2], strides=[2, 1], layout=CFcf (0xf), const ndim=2, constraints: [0.0], shape=[1], strides=[1], layout=CFcf (0xf), const ndim=1, rank: Some([0], shape=[1], strides=[1], layout=CFcf (0xf), const ndim=1), survival_score: Some([inf], shape=[1], strides=[1], layout=CFcf (0xf), const ndim=1), constraint_violation_totals: Some([0.0], shape=[1], strides=[1], layout=CFcf (0xf), const ndim=1) }

```























































```rust
// Get the best individual (just 1 in this example)
>>> best.get(0)
Individual { genes: [1.0, 0.0, 0.0, 1.0, 1.0], shape=[5], strides=[1], layout=CFcf (0xf), const ndim=1, fitness: [-7.0, -15.0], shape=[2], strides=[1], layout=CFcf (0xf), const ndim=1, constraints: 0.0, shape=[], strides=[], layout=CFcf (0xf), const ndim=0, rank: Some(0), survival_score: Some(inf), constraint_violation_totals: Some(0.0) }

```
