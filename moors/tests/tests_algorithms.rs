use ndarray::{Axis, stack};

use moors::{
    algorithms::Nsga2,
    duplicates::PopulationCleaner,
    genetic::{ConstraintsFn, PopulationFitness, PopulationGenes},
    operators::crossover::sbx::SimulatedBinaryCrossover,
    operators::mutation::gaussian::GaussianMutation,
    operators::sampling::RandomSamplingFloat,
};

// Bi-objective fitness function similar to the Python version.
// For each individual with genes [x, y]:
//   f1 = x^2 + y^2
//   f2 = (x - 1)^2 + (y - 1)^2
fn fitness_biobjective(population_genes: &PopulationGenes) -> PopulationFitness {
    let x = population_genes.column(0);
    let y = population_genes.column(1);
    let f1 = &x * &x + &y * &y;
    // Compute (x-1)^2 and (y-1)^2 and sum them.
    let f2 = (&x - 1.0).mapv(|v| v * v) + (&y - 1.0).mapv(|v| v * v);
    stack(Axis(1), &[f1.view(), f2.view()]).expect("Failed to stack fitness values")
}

#[test]
fn test_nsga2() {
    // Create operators.
    let sampler = RandomSamplingFloat::new(0.0, 1.0);
    let crossover = SimulatedBinaryCrossover::new(15.0);
    let mutation = GaussianMutation::new(0.5, 0.01);
    let duplicates_cleaner = None::<()>; // Not used in this simple test
    let constraint_fn: Option<ConstraintsFn> = None;

    // Note that PopulationGenes and PopulationFitness
    // are expected to be ndarray::Array2<f64>.
    let fitness_fn = fitness_biobjective;

    // Set algorithm parameters.
    let n_vars = 2;
    let population_size = 10;
    let n_offsprings = 5;
    let n_iterations = 3;
    let mutation_rate = 0.1;
    let crossover_rate = 0.9;
    let keep_infeasible = true;
    let verbose = true;
    let constraints_fn = constraint_fn;
    let lower_bound = Some(0.0);
    let upper_bound = Some(1.0);
    let seed = Some(42);

    let mut nsga2 = Nsga2::new(
        sampler,
        crossover,
        mutation,
        duplicates_cleaner,
        fitness_fn,
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
    .expect("Failed to create NSGA-II algorithm instance");

    // Run the algorithm. The test checks that the algorithm completes without errors.
    let result = nsga2.inner.run();
    assert!(result.is_ok());
    assert_eq!(nsga2.inner.population.len(), 10);
}
