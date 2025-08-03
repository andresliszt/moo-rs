```rust

use ndarray::{Array1, Array2, Axis, stack};

use moors::{
    algorithms::{AlgorithmError, Nsga2Builder},
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
