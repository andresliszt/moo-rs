use ndarray::{Array2, Axis, array, stack};
use ordered_float::OrderedFloat;
use std::collections::HashSet;

use moors::{
    algorithms::{AgeMoea, Nsga2, Nsga3, Revea, Rnsga2},
    duplicates::CloseDuplicatesCleaner,
    genetic::{FitnessFn, Population, PopulationFitness, PopulationGenes},
    operators::{
        crossover::SimulatedBinaryCrossover,
        mutation::GaussianMutation,
        sampling::RandomSamplingFloat,
        survival::{
            nsga3::Nsga3ReferencePoints,
            reference_points::{DanAndDenisReferencePoints, StructuredReferencePoints},
        },
    },
};

/// Bi-objective fitness:
/// f₁ = x² + y²
/// f₂ = (x−1)² + (y−1)²
fn fitness_biobjective(population_genes: &PopulationGenes) -> PopulationFitness {
    let x = population_genes.column(0);
    let y = population_genes.column(1);
    let f1 = &x * &x + &y * &y;
    let f2 = (&x - 1.0).mapv(|v| v * v) + (&y - 1.0).mapv(|v| v * v);
    stack(Axis(1), &[f1.view(), f2.view()]).expect("stack failed")
}

/// 1) Pareto front size == population size
/// 2) ∀(x,y) on front, |x−y| < 0.2
/// 3) no duplicate points
fn assert_small_real_front(pop: &Population) {
    let n = pop.len();
    let front = pop.best();
    assert_eq!(front.len(), n, "expected full Pareto front");

    let mut seen = HashSet::new();
    for i in 0..front.len() {
        let g = front.get(i).genes;
        // near diagonal
        assert!(
            (g[0] - g[1]).abs() < 0.2,
            "point {:?} too far from diagonal",
            g
        );
        // uniqueness
        let key = (OrderedFloat(g[0]), OrderedFloat(g[1]));
        assert!(
            seen.insert(key),
            "duplicate point on Pareto front: {:?}",
            key
        );
    }
}

#[test]
fn test_nsga2() {
    let mut algorithm = Nsga2::new(
        RandomSamplingFloat::new(0.0, 1.0),
        SimulatedBinaryCrossover::new(15.0),
        GaussianMutation::new(0.5, 0.01),
        Some(CloseDuplicatesCleaner::new(1e-6)),
        fitness_biobjective as FitnessFn,
        2,         // n_vars
        100,       // population_size
        100,       // n_offsprings
        100,       // n_iterations
        0.1,       // mutation_rate
        0.9,       // crossover_rate
        false,     // keep_infeasible
        false,     // verbose
        None,      // no constraints_fn
        Some(0.0), // lower_bound
        Some(1.0), // upper_bound
        Some(42),  // seed
    )
    .expect("failed to build NSGA2");

    algorithm.run().expect("NSGA2 run failed");
    assert_small_real_front(&algorithm.population());
}

#[test]
fn test_agemoea() {
    let mut algorithm = AgeMoea::new(
        RandomSamplingFloat::new(0.0, 1.0),
        SimulatedBinaryCrossover::new(15.0),
        GaussianMutation::new(0.5, 0.01),
        Some(CloseDuplicatesCleaner::new(1e-6)),
        fitness_biobjective as FitnessFn,
        2,          // n_vars
        100,        // population_size
        100,        // n_offsprings
        100,        // n_iterations
        0.1,        // mutation_rate
        0.9,        // crossover_rate
        false,      // keep_infeasible
        false,      // verbose
        None,       // no constraints_fn
        Some(0.0),  // lower_bound
        Some(1.0),  // upper_bound
        Some(1729), // seed
    )
    .expect("failed to build AgeMoea");

    algorithm.run().expect("AgeMoea run failed");
    assert_small_real_front(&algorithm.population());
}

#[test]
fn test_nsga3() {
    let reference_points = DanAndDenisReferencePoints::new(100, 2);
    let reference_points_nsga3 = Nsga3ReferencePoints::new(reference_points.generate(), false);
    let mut algorithm = Nsga3::new(
        reference_points_nsga3,
        RandomSamplingFloat::new(0.0, 1.0),
        SimulatedBinaryCrossover::new(15.0),
        GaussianMutation::new(0.5, 0.01),
        Some(CloseDuplicatesCleaner::new(1e-6)),
        fitness_biobjective as FitnessFn,
        2,         // n_vars
        100,       // population_size
        100,       // n_offsprings
        100,       // n_iterations
        0.1,       // mutation_rate
        0.9,       // crossover_rate
        false,     // keep_infeasible
        false,     // verbose
        None,      // no constraints_fn
        Some(0.0), // lower_bound
        Some(1.0), // upper_bound
        Some(42),  // seed
    )
    .expect("failed to build NSGA3");

    algorithm.run().expect("NSGA3 run failed");
    assert_small_real_front(&algorithm.population());
}

#[test]
fn test_rnsga2() {
    let reference_points: Array2<f64> = array![[0.8, 0.8], [0.9, 0.9]];
    let mut algorithm = Rnsga2::new(
        reference_points,
        0.001, // epsilon
        RandomSamplingFloat::new(0.0, 1.0),
        SimulatedBinaryCrossover::new(15.0),
        GaussianMutation::new(0.5, 0.01),
        Some(CloseDuplicatesCleaner::new(1e-6)),
        fitness_biobjective as FitnessFn,
        2,         // n_vars
        100,       // population_size
        100,       // n_offsprings
        100,       // n_iterations
        0.1,       // mutation_rate
        0.9,       // crossover_rate
        false,     // keep_infeasible
        false,     // verbose
        None,      // no constraints_fn
        Some(0.0), // lower_bound
        Some(1.0), // upper_bound
        Some(42),  // seed
    )
    .expect("failed to build Rnsga2");

    algorithm.run().expect("RNSGA2 run failed");
    assert_small_real_front(&algorithm.population());
}

#[test]
fn test_revea() {
    let reference_points = DanAndDenisReferencePoints::new(100, 2).generate();
    let mut algorithm = Revea::new(
        reference_points,
        2.5, // alpha
        0.2, // frequency
        RandomSamplingFloat::new(0.0, 1.0),
        SimulatedBinaryCrossover::new(15.0),
        GaussianMutation::new(0.5, 0.01),
        Some(CloseDuplicatesCleaner::new(1e-6)),
        fitness_biobjective as FitnessFn,
        2,         // n_vars
        100,       // population_size
        100,       // n_offsprings
        100,       // n_iterations
        0.1,       // mutation_rate
        0.9,       // crossover_rate
        false,     // keep_infeasible
        false,     // verbose
        None,      // no constraints_fn
        Some(0.0), // lower_bound
        Some(1.0), // upper_bound
        None,      // seed
    )
    .expect("failed to build Revea");

    algorithm.run().expect("Revea run failed");
    assert_small_real_front(&algorithm.population());
}
