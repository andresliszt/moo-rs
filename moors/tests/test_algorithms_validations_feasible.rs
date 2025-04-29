// tests/algorithms/test_infeasible.rs

use ndarray::{Axis, Array2, stack};
use moors::{
    algorithms::Nsga2,
    duplicates::CloseDuplicatesCleaner,
    genetic::{PopulationGenes, PopulationFitness, PopulationConstraints, FitnessFn, ConstraintsFn},
    operators::{
        sampling::RandomSamplingBinary,
        crossover::SinglePointBinaryCrossover,
        mutation::BitFlipMutation,
    },
};
use moors::algorithms::MultiObjectiveAlgorithmError;
use moors::evaluator::EvaluatorError;

/// Binary bi‐objective fitness: [sum(x), sum(1−x)]
fn fitness_binary_biobj(genes: &PopulationGenes) -> PopulationFitness {
    let f1 = genes.sum_axis(Axis(1));
    let ones = Array2::from_elem(genes.raw_dim(), 1.0);
    let f2 = (&ones - genes).sum_axis(Axis(1));
    stack(Axis(1), &[f1.view(), f2.view()]).unwrap()
}

/// Infeasible constraint: n_vars - sum(x) + 1 > 0 for all individuals ⇒ always infeasible
fn constraints_always_infeasible(genes: &PopulationGenes) -> PopulationConstraints {
    let n = genes.ncols() as f64;
    let sum = genes.sum_axis(Axis(1));
    let c = sum.mapv(|s| n - s + 1.0);
    c.insert_axis(Axis(1))
}

#[test]
fn test_keep_infeasible() {
    let mut alg = Nsga2::new(
        RandomSamplingBinary::new(),
        SinglePointBinaryCrossover::new(),
        BitFlipMutation::new(0.5),
        None::<CloseDuplicatesCleaner>,                          // no duplicates cleaner
        fitness_binary_biobj as FitnessFn,                      // fitness function
        5,      // n_vars
        100,    // population_size
        32,     // n_offsprings
        20,     // n_iterations
        0.1,    // mutation_rate
        0.9,    // crossover_rate
        true,   // keep_infeasible
        false,  // verbose
        Some(constraints_always_infeasible as ConstraintsFn),   // always‐infeasible constraints
        None::<f64>,                                         // lower_bound
        None::<f64>,                                         // upper_bound
        None::<u64>,                                         // seed
    )
    .expect("failed to build NSGA2");

    alg.run().expect("run should succeed when keep_infeasible = true");
    assert_eq!(alg.population().len(), 100);
}

#[test]
fn test_keep_infeasible_out_of_bounds() {
    let mut alg = Nsga2::new(
        RandomSamplingBinary::new(),
        SinglePointBinaryCrossover::new(),
        BitFlipMutation::new(0.5),
        None::<CloseDuplicatesCleaner>,
        fitness_binary_biobj as FitnessFn,
        5,      // n_vars
        100,    // population_size
        32,     // n_offsprings
        20,     // n_iterations
        0.1,    // mutation_rate
        0.9,    // crossover_rate
        true,                     // keep_infeasible = true
        false,                    // verbose = false
        None::<ConstraintsFn>,    // no constraints_fn
        Some(2.0),                // lower_bound = 2.0 (out‐of‐bounds)
        Some(10.0),               // upper_bound = 10.0
        None::<u64>,              // seed
    )
    .expect("failed to build NSGA2");

    alg.run().expect("run should succeed even if genes are out of bounds");
    assert_eq!(alg.population().len(), 100);
}

#[test]
fn test_remove_infeasible() {
    let err = Nsga2::new(
        RandomSamplingBinary::new(),
        SinglePointBinaryCrossover::new(),
        BitFlipMutation::new(0.5),
        None::<CloseDuplicatesCleaner>,
        fitness_binary_biobj as FitnessFn,
        5,       // n_vars
        100,     // population_size
        100,     // n_offsprings
        20,      // n_iterations
        0.1,     // mutation_rate
        0.9,     // crossover_rate
        false,   // keep infeasible
        false,   // verbose
        Some(constraints_always_infeasible as ConstraintsFn), // infeasible constraints
        None::<f64>, // upper bound
        None::<f64>, // lower bound
        Some(1729)
    )
    .expect_err("expected no feasible individuals error");
    assert!(matches!(
        err,
        MultiObjectiveAlgorithmError::Evaluator(EvaluatorError::NoFeasibleIndividuals)
    ));
}
