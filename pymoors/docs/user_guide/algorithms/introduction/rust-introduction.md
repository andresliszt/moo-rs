In `moors`, the algorithms are implemented as structs with a set of useful attributes. These attributes include the final population and the optimal or best set of individuals found during the optimization process.

For example, after running an algorithm like NSGA2, you can access:
- **Final Population:** The complete set of individuals from the last generation.
- **Optimum Set:** Typically, the best individuals (e.g., those with rank 0) that form the current approximation of the Pareto front.

```Rust
use ndarray::{Axis, Array1, Array2, stack};
use moors::{
    algorithms::{MultiObjectiveAlgorithmError, Nsga2Builder},
    duplicates::CloseDuplicatesCleaner,
    operators::{RandomSamplingFloat, SimulatedBinaryCrossover, GaussianMutation}
};

/// Define the fitness function
fn fitness(population_genes: &Array2<f64>) -> Array2<f64> {
    let x1 = population_genes.column(0);
    let x2 = population_genes.column(1);

    let f1 = &x1 * &x1 + &x2 * &x2;
    let f2 = (&x1 - 1.0).mapv(|v| v * v) + &x2 * &x2;

    stack(Axis(1), &[f1.view(), f2.view()]).unwrap()
}

///  Define the constraints function
fn constraints_fn(population_genes: &Array2<f64>) -> Array1<f64> {
    let x1 = population_genes.column(0);
    let x2 = population_genes.column(1);
    x1 + x2 - 1.0
}

fn main() -> Result<(), MultiObjectiveAlgorithmError> {
    let sampler   = RandomSamplingFloat::new(0.0, 1.0);
    let crossover = SimulatedBinaryCrossover::new(5.0);
    let mutation  = GaussianMutation::new(0.1, 0.01);
    let cleaner   = CloseDuplicatesCleaner::new(1e-8);

    // Build NSGA-II with the same hyperparameters
    let mut algo = Nsga2Builder::default()
        .fitness_fn(fitness)
        .constraints_fn(constraints)
        .sampler(sampler)
        .crossover(crossover)
        .mutation(mutation)
        .duplicates_cleaner(cleaner)
        .num_vars(2)
        .num_objectives(2)
        .num_constraints(1)
        .population_size(200)
        .num_offsprings(200)
        .num_iterations(200)
        .mutation_rate(0.1)
        .crossover_rate(0.9)
        .keep_infeasible(false)
        .build()?;

    // Run
    algo.run()?;

    // Access the final population if needed
    let population_genes = algo.population?;
    println!("Done. Final population size: {}", population_genes.len());

    Ok(())
}

```
