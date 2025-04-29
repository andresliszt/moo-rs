// tests/algorithms/test_nsga2_params.rs

use moors::{
    algorithms::MultiObjectiveAlgorithmError,
    algorithms::Nsga2,
    genetic::{ConstraintsFn, FitnessFn, PopulationFitness, PopulationGenes},
    operators::{
        crossover::SimulatedBinaryCrossover, mutation::GaussianMutation,
        sampling::RandomSamplingFloat,
    },
};

/// A trivial fitness function that just clones the genes matrix.
fn dummy_fitness(genes: &PopulationGenes) -> PopulationFitness {
    genes.clone()
}

#[test]
fn test_invalid_mutation_rate() {
    for &invalid in &[-0.1, 1.5] {
        let err = Nsga2::new(
            RandomSamplingFloat::new(0.0, 1.0),
            SimulatedBinaryCrossover::new(2.0),
            GaussianMutation::new(0.1, 0.05),
            None::<()>,
            dummy_fitness as FitnessFn,
            10,
            100,
            50,
            50,
            invalid, // invalid mutation_rate
            0.9,
            false,
            false,
            None::<ConstraintsFn>,
            Some(0.0),
            Some(1.0),
            Some(42),
        )
        .expect_err("Expected an error for invalid mutation_rate");
        assert!(matches!(
            err,
            MultiObjectiveAlgorithmError::InvalidParameter(_)
        ));
        let msg = format!("{}", err);
        assert!(
            msg.contains("Mutation rate must be between 0 and 1"),
            "Unexpected message: {}",
            msg
        );
    }
}

#[test]
fn test_invalid_crossover_rate() {
    for &invalid in &[-0.5, 2.0] {
        let err = Nsga2::new(
            RandomSamplingFloat::new(0.0, 1.0),
            SimulatedBinaryCrossover::new(2.0),
            GaussianMutation::new(0.1, 0.05),
            None::<()>,
            dummy_fitness as FitnessFn,
            10,
            100,
            50,
            50,
            0.1,
            invalid, // invalid crossover_rate
            false,
            false,
            None::<ConstraintsFn>,
            Some(0.0),
            Some(1.0),
            Some(42),
        )
        .expect_err("Expected an error for invalid crossover_rate");
        assert!(matches!(
            err,
            MultiObjectiveAlgorithmError::InvalidParameter(_)
        ));
        let msg = format!("{}", err);
        assert!(
            msg.contains("Crossover rate must be between 0 and 1"),
            "Unexpected message: {}",
            msg
        );
    }
}

#[test]
fn test_invalid_n_vars_population_offsprings_iterations() {
    // n_vars = 0
    let err = Nsga2::new(
        RandomSamplingFloat::new(0.0, 1.0),
        SimulatedBinaryCrossover::new(2.0),
        GaussianMutation::new(0.1, 0.05),
        None::<()>,
        dummy_fitness as FitnessFn,
        0, // invalid n_vars
        100,
        50,
        50,
        0.1,
        0.9,
        false,
        false,
        None::<ConstraintsFn>,
        Some(0.0),
        Some(1.0),
        Some(42),
    )
    .expect_err("Expected error for n_vars = 0");
    let msg = format!("{}", err);
    assert!(msg.contains("Number of variables must be greater than 0"));

    // population_size = 0
    let err = Nsga2::new(
        RandomSamplingFloat::new(0.0, 1.0),
        SimulatedBinaryCrossover::new(2.0),
        GaussianMutation::new(0.1, 0.05),
        None::<()>,
        dummy_fitness as FitnessFn,
        10,
        0, // invalid population_size
        50,
        50,
        0.1,
        0.9,
        false,
        false,
        None::<ConstraintsFn>,
        Some(0.0),
        Some(1.0),
        Some(42),
    )
    .expect_err("Expected error for population_size = 0");
    let msg = format!("{}", err);
    assert!(msg.contains("Population size must be greater than 0"));

    // n_offsprings = 0
    let err = Nsga2::new(
        RandomSamplingFloat::new(0.0, 1.0),
        SimulatedBinaryCrossover::new(2.0),
        GaussianMutation::new(0.1, 0.05),
        None::<()>,
        dummy_fitness as FitnessFn,
        10,
        100,
        0, // invalid n_offsprings
        50,
        0.1,
        0.9,
        false,
        false,
        None::<ConstraintsFn>,
        Some(0.0),
        Some(1.0),
        Some(42),
    )
    .expect_err("Expected error for n_offsprings = 0");
    let msg = format!("{}", err);
    assert!(msg.contains("Number of offsprings must be greater than 0"));

    // n_iterations = 0
    let err = Nsga2::new(
        RandomSamplingFloat::new(0.0, 1.0),
        SimulatedBinaryCrossover::new(2.0),
        GaussianMutation::new(0.1, 0.05),
        None::<()>,
        dummy_fitness as FitnessFn,
        10,
        100,
        50,
        0, // invalid n_iterations
        0.1,
        0.9,
        false,
        false,
        None::<ConstraintsFn>,
        Some(0.0),
        Some(1.0),
        Some(42),
    )
    .expect_err("Expected error for n_iterations = 0");
    let msg = format!("{}", err);
    assert!(msg.contains("Number of iterations must be greater than 0"));
}

#[test]
fn test_invalid_bounds() {
    for &(lower, upper) in &[(1.0, 1.0), (2.0, 1.0)] {
        let err = Nsga2::new(
            RandomSamplingFloat::new(0.0, 1.0),
            SimulatedBinaryCrossover::new(2.0),
            GaussianMutation::new(0.1, 0.05),
            None::<()>,
            dummy_fitness as FitnessFn,
            10,
            100,
            50,
            50,
            0.1,
            0.9,
            false,
            false,
            None::<ConstraintsFn>,
            Some(lower), // invalid lower_bound
            Some(upper), // invalid upper_bound
            Some(42),
        )
        .expect_err("Expected error for invalid bounds");
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
    }
}
