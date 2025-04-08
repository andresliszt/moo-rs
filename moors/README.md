# moors

**moors** is a pure‑Rust crate providing multi‑objective evolutionary algorithms.

## Features

- NSGA‑II, NSGA‑III, R‑NSGA‑II, Age‑MOEA, REVEA (Many more are coming!)
- Pluggable operators: sampling, crossover, mutation, duplicates removal
- Flexible fitness & constraints via user‑provided closures
- Built using `ndarray` and `faer`

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
moors = "0.1.0"
```

## Quickstart

Here’s a complete `main.rs` example that runs a bi‑objective NSGA‑II:

```rust
use ndarray::{Axis, stack};
use moors::{
    algorithms::Nsga2,
    duplicates::CloseDuplicatesCleaner,
    genetic::{ConstraintsFn, PopulationFitness, PopulationGenes},
    operators::crossover::sbx::SimulatedBinaryCrossover,
    operators::mutation::gaussian::GaussianMutation,
    operators::sampling::RandomSamplingFloat,
};

// Bi‑objective fitness: f1 = x² + y², f2 = (x−1)² + (y−1)²
fn fitness_biobjective(pop: &PopulationGenes) -> PopulationFitness {
    let x = pop.column(0);
    let y = pop.column(1);
    let f1 = &x * &x + &y * &y;
    let f2 = (&x - 1.0).mapv(|v| v * v)
           + (&y - 1.0).mapv(|v| v * v);
    stack(Axis(1), &[f1.view(), f2.view()])
        .expect("stack failed")
}

fn main() {
    // Create operators
    let sampler            = RandomSamplingFloat::new(0.0, 1.0);
    let crossover          = SimulatedBinaryCrossover::new(15.0);
    let mutation           = GaussianMutation::new(0.5, 0.01);
    let duplicates_cleaner = CloseDuplicatesCleaner::new(1e-8);
    let constraints_fn     = None::<ConstraintsFn>;

    // Algorithm parameters
    let n_vars          = 2;
    let population_size = 10;
    let n_offsprings    = 10;
    let n_iterations    = 10;
    let mutation_rate   = 0.1;
    let crossover_rate  = 0.9;
    let keep_infeasible = true;
    let verbose         = true;
    let lower_bound     = Some(0.0);
    let upper_bound     = Some(1.0);
    let seed            = Some(42);

    // Build NSGA‑II
    let mut nsga2 = Nsga2::new(
        sampler,
        crossover,
        mutation,
        duplicates_cleaner,
        fitness_biobjective,
        n_vars,
        population_size,
        n_offsprings,
        n_iterations,
        mutation_rate,
        crossover_rate,
        keep_infeasible,
        verbose,
        constraints_fn,
        lower_bound,
        upper_bound,
        seed,
    )
    .expect("failed to create NSGA-II");

    // Run evolution
    nsga2.inner.run().expect("algorithm error");
    println!("Done! Population size: {}", nsga2.inner.population.len());
}
```

### Error Handling

- `Nsga2::new(...)` returns `Result<_, MultiObjectiveAlgorithmError>`.
- Invalid parameters (e.g., `crossover_rate` outside `[0.0,1.0]`) produce
  `MultiObjectiveAlgorithmError::InvalidParameter`.
- You can `.expect("…")` or propagate the error as needed.

## Relation to Python

This crate is the backend for the [pymoors Python extension](https://andresliszt.github.io/pymoors/), but you do **not** need Python to use **moors**. They are fully decoupled.
