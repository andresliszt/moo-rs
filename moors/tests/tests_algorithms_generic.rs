use ndarray::{Array2, Axis, array, stack};
use ordered_float::OrderedFloat;
use std::collections::HashSet;

use moors::{
    algorithms::{AgeMoeaBuilder, Nsga2Builder, Nsga3Builder, ReveaBuilder, Rnsga2Builder},
    duplicates::CloseDuplicatesCleaner,
    genetic::{FitnessFn, NoConstraintsFn, Population, PopulationFitness, PopulationGenes},
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
        assert!(
            (g[0] - g[1]).abs() < 0.2,
            "point {:?} too far from diagonal",
            g
        );
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
    let mut algorithm = Nsga2Builder::<_, _, _, _, NoConstraintsFn, _>::default()
        .sampler(RandomSamplingFloat::new(0.0, 1.0))
        .crossover(SimulatedBinaryCrossover::new(15.0))
        .mutation(GaussianMutation::new(0.5, 0.01))
        .duplicates_cleaner(CloseDuplicatesCleaner::new(1e-6))
        .fitness_fn(fitness_biobjective as FitnessFn)
        .num_vars(2)
        .population_size(100)
        .num_offsprings(100)
        .num_iterations(100)
        .mutation_rate(0.1)
        .crossover_rate(0.9)
        .keep_infeasible(false)
        .verbose(false)
        .lower_bound(0.0)
        .upper_bound(1.0)
        .seed(42)
        .build()
        .expect("failed to build NSGA2");

    algorithm.run().expect("NSGA2 run failed");
    let population = algorithm
        .population()
        .expect("population should have been initialized");
    assert_small_real_front(&population);
}

#[test]
fn test_agemoea() {
    let mut algorithm = AgeMoeaBuilder::<_, _, _, _, NoConstraintsFn, _>::default()
        .sampler(RandomSamplingFloat::new(0.0, 1.0))
        .crossover(SimulatedBinaryCrossover::new(15.0))
        .mutation(GaussianMutation::new(0.5, 0.01))
        .duplicates_cleaner(CloseDuplicatesCleaner::new(1e-6))
        .fitness_fn(fitness_biobjective as FitnessFn)
        .num_vars(2)
        .population_size(100)
        .num_offsprings(100)
        .num_iterations(100)
        .mutation_rate(0.1)
        .crossover_rate(0.9)
        .keep_infeasible(false)
        .verbose(false)
        .lower_bound(0.0)
        .upper_bound(1.0)
        .seed(1729)
        .build()
        .expect("failed to build AgeMoea");

    algorithm.run().expect("AgeMoea run failed");
    let population = algorithm
        .population()
        .expect("population should have been initialized");
    assert_small_real_front(&population);
}

#[test]
fn test_nsga3() {
    let reference_points = DanAndDenisReferencePoints::new(100, 2);
    let rp = Nsga3ReferencePoints::new(reference_points.generate(), false);
    let mut algorithm = Nsga3Builder::<_, _, _, _, NoConstraintsFn, _>::default()
        .reference_points(rp)
        .sampler(RandomSamplingFloat::new(0.0, 1.0))
        .crossover(SimulatedBinaryCrossover::new(15.0))
        .mutation(GaussianMutation::new(0.5, 0.01))
        .duplicates_cleaner(CloseDuplicatesCleaner::new(1e-6))
        .fitness_fn(fitness_biobjective as FitnessFn)
        .num_vars(2)
        .population_size(100)
        .num_offsprings(100)
        .num_iterations(100)
        .mutation_rate(0.1)
        .crossover_rate(0.9)
        .keep_infeasible(false)
        .verbose(false)
        .lower_bound(0.0)
        .upper_bound(1.0)
        .seed(42)
        .build()
        .expect("failed to build NSGA3");

    algorithm.run().expect("NSGA3 run failed");
    let population = algorithm
        .population()
        .expect("population should have been initialized");
    assert_small_real_front(&population);
}

#[test]
fn test_rnsga2() {
    let reference_points: Array2<f64> = array![[0.8, 0.8], [0.9, 0.9]];
    let mut algorithm = Rnsga2Builder::<_, _, _, _, NoConstraintsFn, _>::default()
        .reference_points(reference_points)
        .epsilon(0.001)
        .sampler(RandomSamplingFloat::new(0.0, 1.0))
        .crossover(SimulatedBinaryCrossover::new(15.0))
        .mutation(GaussianMutation::new(0.5, 0.01))
        .duplicates_cleaner(CloseDuplicatesCleaner::new(1e-6))
        .fitness_fn(fitness_biobjective as FitnessFn)
        .num_vars(2)
        .population_size(100)
        .num_offsprings(100)
        .num_iterations(100)
        .mutation_rate(0.1)
        .crossover_rate(0.9)
        .keep_infeasible(false)
        .verbose(false)
        .lower_bound(0.0)
        .upper_bound(1.0)
        .seed(42)
        .build()
        .expect("failed to build RNSGA2");

    algorithm.run().expect("RNSGA2 run failed");
    let population = algorithm
        .population()
        .expect("population should have been initialized");
    assert_small_real_front(&population);
}

#[test]
fn test_revea() {
    let reference_points = DanAndDenisReferencePoints::new(100, 2).generate();
    let mut algorithm = ReveaBuilder::<_, _, _, _, NoConstraintsFn, _>::default()
        .reference_points(reference_points)
        .alpha(2.5)
        .frequency(0.2)
        .sampler(RandomSamplingFloat::new(0.0, 1.0))
        .crossover(SimulatedBinaryCrossover::new(15.0))
        .mutation(GaussianMutation::new(0.5, 0.01))
        .duplicates_cleaner(CloseDuplicatesCleaner::new(1e-6))
        .fitness_fn(fitness_biobjective as FitnessFn)
        .num_vars(2)
        .population_size(100)
        .num_offsprings(100)
        .num_iterations(100)
        .mutation_rate(0.1)
        .crossover_rate(0.9)
        .keep_infeasible(false)
        .verbose(false)
        .lower_bound(0.0)
        .upper_bound(1.0)
        .build()
        .expect("failed to build REVEA");

    algorithm.run().expect("Revea run failed");
    let population = algorithm
        .population()
        .expect("population should have been initialized");
    assert_small_real_front(&population);
}
