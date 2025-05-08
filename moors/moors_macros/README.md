# moors_macros

A collection of procedural macros for `moors`.

---

## Installation in `moors`

Added directly as:

```toml
# In Cargo.toml
[dependencies]
pymoors_macros = { path = moors_macros" }

```

---

## Algorithm Builders

```rust
use ndarray::{Array1, Axis, stack};

use moors::{
    algorithms::Nsga2Builder,
    duplicates::ExactDuplicatesCleaner,
    genetic::{
        ConstraintsFn, FitnessFn, PopulationConstraints, PopulationFitness, PopulationGenes,
    },
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
    // lift our fixed arrays into Array1 for dot products
    let weights_arr = Array1::from_vec(WEIGHTS.to_vec());
    let values_arr = Array1::from_vec(VALUES.to_vec());

    let total_values = population_genes.dot(&values_arr);
    let total_weights = population_genes.dot(&weights_arr);

    // stack two columns: [-total_values, total_weights]
    stack(Axis(1), &[(-&total_values).view(), total_weights.view()]).expect("stack failed")
}

fn constraints_knapsack(population_genes: &PopulationGenes) -> PopulationConstraints {
    // build a 1-D array of weights in one shot
    let weights_arr = Array1::from_vec(WEIGHTS.to_vec());

    // dot → Array1<f64>, subtract capacity → Array1<f64>,
    // then promote to 2-D (n×1) with insert_axis
    (population_genes.dot(&weights_arr) - CAPACITY).insert_axis(Axis(1))
}

// build and run the NSGA-II algorithm
let mut algorithm = Nsga2Builder::default()
    .fitness_fn(fitness_knapsack as FitnessFn)
    .constraints_fn(constraints_knapsack as ConstraintsFn)
    .sampler(RandomSamplingBinary::new())
    .crossover(SinglePointBinaryCrossover::new())
    .mutation(BitFlipMutation::new(0.5))
    .duplicates_cleaner(ExactDuplicatesCleaner::new())
    .num_vars(5)
    .population_size(100)
    .n_offsprings(32)
    .num_iterations(2)
    .crossover_rate(0.9)
    .mutation_rate(0.1)
    .lower_bound(0.0)
    .upper_bound(1.0)
    .build()
    .unwrap();

```

---

## Limitations & TODO

- Coercion on fitness and constraints is needed to keep Debug on `Nsga2`. See https://stackoverflow.com/questions/53380040/function-pointer-with-a-reference-argument-cannot-derive-debug
- Optional generics such as `constraints_fn` and `duplicates_cleaner` if are removed from builder, then a turbofish must be added `Nsga2Builder`::<_, _, _, _, crate::genetic::NoConstraints, crate::duplicates::NoDuplicatesCleaner>::default()...`. See https://github.com/colin-kiegel/rust-derive-builder/issues/343
- Consider using `rust-derive-builder`

---
