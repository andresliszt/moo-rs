// tests/algorithms/test_nsga2_params.rs
use ndarray::Array2;

use moors::{
    NoConstraints,
    algorithms::{AlgorithmBuilderError, Nsga2Builder},
    duplicates::NoDuplicatesCleaner,
    impl_constraints_fn,
    operators::{GaussianMutation, RandomSamplingFloat, SimulatedBinaryCrossover},
};
use rstest::rstest;

/// A trivial fitness function that just clones the genes matrix.
fn dummy_fitness(genes: &Array2<f64>) -> Array2<f64> {
    genes.clone()
}

#[rstest]
#[case(-0.1)]
#[case(1.5)]
fn test_invalid_mutation_rate(#[case] invalid: f64) {
    let err = match Nsga2Builder::default()
        .sampler(RandomSamplingFloat::new(0.0, 1.0))
        .crossover(SimulatedBinaryCrossover::new(2.0))
        .mutation(GaussianMutation::new(0.1, 0.05))
        .duplicates_cleaner(NoDuplicatesCleaner)
        .fitness_fn(dummy_fitness)
        .constraints_fn(NoConstraints)
        .num_vars(10)
        .population_size(100)
        .num_offsprings(50)
        .num_iterations(50)
        .mutation_rate(invalid) // ← invalid here
        .crossover_rate(0.9)
        .build()
    {
        Ok(_) => panic!("Expected an error for invalid mutation_rate"),
        Err(e) => e,
    };

    assert!(
        matches!(err, AlgorithmBuilderError::ValidationError(_)),
        "got wrong error: {:?}",
        err
    );
    let msg = format!("{}", err);
    assert!(
        msg.contains("Mutation rate must be between 0 and 1"),
        "Unexpected message: {}",
        msg
    );
}

#[rstest]
#[case(-0.5)]
#[case(2.0)]
fn test_invalid_crossover_rate(#[case] invalid: f64) {
    let err = match Nsga2Builder::default()
        .sampler(RandomSamplingFloat::new(0.0, 1.0))
        .crossover(SimulatedBinaryCrossover::new(2.0))
        .mutation(GaussianMutation::new(0.1, 0.05))
        .duplicates_cleaner(NoDuplicatesCleaner)
        .fitness_fn(dummy_fitness)
        .constraints_fn(NoConstraints)
        .num_vars(10)
        .population_size(100)
        .num_offsprings(50)
        .num_iterations(50)
        .mutation_rate(0.1)
        .crossover_rate(invalid) // ← invalid here
        .build()
    {
        Ok(_) => panic!("Expected an error for invalid crossover_rate"),
        Err(e) => e,
    };

    assert!(
        matches!(err, AlgorithmBuilderError::ValidationError(_)),
        "got wrong error: {:?}",
        err
    );
    let msg = format!("{}", err);
    assert!(
        msg.contains("Crossover rate must be between 0 and 1"),
        "Unexpected message: {}",
        msg
    );
}

#[test]
fn test_invalid_n_vars_population_offsprings_iterations() {
    // num_vars = 0
    let err = match Nsga2Builder::default()
        .sampler(RandomSamplingFloat::new(0.0, 1.0))
        .crossover(SimulatedBinaryCrossover::new(2.0))
        .mutation(GaussianMutation::new(0.1, 0.05))
        .duplicates_cleaner(NoDuplicatesCleaner)
        .fitness_fn(dummy_fitness)
        .constraints_fn(NoConstraints)
        .num_vars(0) // ← invalid
        .population_size(100)
        .num_offsprings(50)
        .num_iterations(50)
        .mutation_rate(0.1)
        .crossover_rate(0.9)
        .build()
    {
        Ok(_) => panic!("Expected error for num_vars = 0"),
        Err(e) => e,
    };
    assert!(
        format!("{}", err).contains("Number of variables must be greater than 0"),
        "Unexpected message: {}",
        err
    );

    // population_size = 0
    let err = match Nsga2Builder::default()
        .sampler(RandomSamplingFloat::new(0.0, 1.0))
        .crossover(SimulatedBinaryCrossover::new(2.0))
        .mutation(GaussianMutation::new(0.1, 0.05))
        .duplicates_cleaner(NoDuplicatesCleaner)
        .fitness_fn(dummy_fitness)
        .constraints_fn(NoConstraints)
        .num_vars(10)
        .population_size(0) // ← invalid
        .num_offsprings(50)
        .num_iterations(50)
        .mutation_rate(0.1)
        .crossover_rate(0.9)
        .build()
    {
        Ok(_) => panic!("Expected error for population_size = 0"),
        Err(e) => e,
    };
    assert!(
        format!("{}", err).contains("Population size must be greater than 0"),
        "Unexpected message: {}",
        err
    );

    // num_offsprings = 0
    let err = match Nsga2Builder::default()
        .sampler(RandomSamplingFloat::new(0.0, 1.0))
        .crossover(SimulatedBinaryCrossover::new(2.0))
        .mutation(GaussianMutation::new(0.1, 0.05))
        .duplicates_cleaner(NoDuplicatesCleaner)
        .fitness_fn(dummy_fitness)
        .constraints_fn(NoConstraints)
        .num_vars(10)
        .population_size(100)
        .num_offsprings(0) // ← invalid
        .num_iterations(50)
        .mutation_rate(0.1)
        .crossover_rate(0.9)
        .build()
    {
        Ok(_) => panic!("Expected error for num_offsprings = 0"),
        Err(e) => e,
    };
    assert!(
        format!("{}", err).contains("Number of offsprings must be greater than 0"),
        "Unexpected message: {}",
        err
    );

    // num_iterations = 0
    let err = match Nsga2Builder::default()
        .sampler(RandomSamplingFloat::new(0.0, 1.0))
        .crossover(SimulatedBinaryCrossover::new(2.0))
        .mutation(GaussianMutation::new(0.1, 0.05))
        .duplicates_cleaner(NoDuplicatesCleaner)
        .fitness_fn(dummy_fitness)
        .constraints_fn(NoConstraints)
        .num_vars(10)
        .population_size(100)
        .num_offsprings(50)
        .num_iterations(0) // ← invalid
        .mutation_rate(0.1)
        .crossover_rate(0.9)
        .build()
    {
        Ok(_) => panic!("Expected error for num_iterations = 0"),
        Err(e) => e,
    };
    assert!(
        format!("{}", err).contains("Number of iterations must be greater than 0"),
        "Unexpected message: {}",
        err
    );
}

#[test]
fn test_invalid_bounds() {
    impl_constraints_fn!(MyConstr, lower_bound = 2.0, upper_bound = 1.0);
    let err = match Nsga2Builder::default()
        .sampler(RandomSamplingFloat::new(0.0, 1.0))
        .crossover(SimulatedBinaryCrossover::new(2.0))
        .mutation(GaussianMutation::new(0.1, 0.05))
        .duplicates_cleaner(NoDuplicatesCleaner)
        .fitness_fn(dummy_fitness)
        .constraints_fn(MyConstr)
        .num_vars(10)
        .population_size(100)
        .num_offsprings(50)
        .num_iterations(50)
        .mutation_rate(0.1)
        .crossover_rate(0.9)
        .build()
    {
        Ok(_) => panic!("Expected error for invalid bounds"),
        Err(e) => e,
    };

    let msg = format!("{}", err);
    assert!(
        msg.contains("Lower bound"),
        "Error message did not reference lower bound: {}",
        msg
    );
    assert!(
        msg.contains("must be less than upper bound"),
        "Error message did not explain bound ordering: {}",
        msg
    );

    impl_constraints_fn!(MyConstrEq, lower_bound = 2.0, upper_bound = 2.0);
    match Nsga2Builder::default()
        .sampler(RandomSamplingFloat::new(0.0, 1.0))
        .crossover(SimulatedBinaryCrossover::new(2.0))
        .mutation(GaussianMutation::new(0.1, 0.05))
        .duplicates_cleaner(NoDuplicatesCleaner)
        .fitness_fn(dummy_fitness)
        .constraints_fn(MyConstrEq)
        .num_vars(10)
        .population_size(100)
        .num_offsprings(50)
        .num_iterations(50)
        .mutation_rate(0.1)
        .crossover_rate(0.9)
        .build()
    {
        Ok(_) => panic!("Expected error for invalid bounds"),
        Err(e) => e,
    };
}
