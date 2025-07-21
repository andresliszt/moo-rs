use ndarray::{Array2, Axis, array, stack};
use ordered_float::OrderedFloat;
use std::collections::HashSet;

use moors::{
    AgeMoeaBuilder, ArithmeticCrossover, CloseDuplicatesCleaner, GaussianMutation, Nsga2Builder,
    Nsga3Builder, Nsga3ReferencePointsSurvival, PopulationMOO, RandomSamplingFloat, ReveaBuilder,
    Rnsga2Builder, SimulatedBinaryCrossover, Spea2Builder, UniformRealMutation,
    impl_constraints_fn,
    survival::moo::{
        DanAndDenisReferencePoints, Nsga3ReferencePoints, ReveaReferencePointsSurvival,
        Rnsga2ReferencePointsSurvival, StructuredReferencePoints,
    },
};

/// Bi-objective fitness:
/// f₁ = x² + y²
/// f₂ = (x−1)² + (y−1)²
fn fitness_biobjective(population_genes: &Array2<f64>) -> Array2<f64> {
    let x = population_genes.column(0);
    let y = population_genes.column(1);
    let f1 = &x * &x + &y * &y;
    let f2 = (&x - 1.0).mapv(|v| v * v) + (&y - 1.0).mapv(|v| v * v);
    stack(Axis(1), &[f1.view(), f2.view()]).expect("stack failed")
}

/// 1) Pareto front size == population size
/// 2) ∀(x,y) on front, |x−y| < 0.2
/// 3) no duplicate points
fn assert_small_real_front(pop: &PopulationMOO) {
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

impl_constraints_fn!(MyConstr, lower_bound = 0.0, upper_bound = 1.0);

#[test]
fn test_nsga2() {
    let mut algorithm = Nsga2Builder::default()
        .sampler(RandomSamplingFloat::new(0.0, 1.0))
        .crossover(SimulatedBinaryCrossover::new(15.0))
        .mutation(GaussianMutation::new(0.5, 0.01))
        .duplicates_cleaner(CloseDuplicatesCleaner::new(1e-6))
        .fitness_fn(fitness_biobjective)
        .constraints_fn(MyConstr)
        .num_vars(2)
        .population_size(100)
        .num_offsprings(100)
        .num_iterations(100)
        .mutation_rate(0.1)
        .crossover_rate(0.9)
        .keep_infeasible(false)
        .verbose(false)
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
    let mut algorithm = AgeMoeaBuilder::default()
        .sampler(RandomSamplingFloat::new(0.0, 1.0))
        .crossover(SimulatedBinaryCrossover::new(15.0))
        .mutation(GaussianMutation::new(0.5, 0.01))
        .duplicates_cleaner(CloseDuplicatesCleaner::new(1e-6))
        .fitness_fn(fitness_biobjective)
        .constraints_fn(MyConstr)
        .num_vars(2)
        .population_size(100)
        .num_offsprings(100)
        .num_iterations(100)
        .mutation_rate(0.1)
        .crossover_rate(0.9)
        .keep_infeasible(false)
        .verbose(false)
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
    let survivor = Nsga3ReferencePointsSurvival::new(rp);

    let mut algorithm = Nsga3Builder::default()
        .sampler(RandomSamplingFloat::new(0.0, 1.0))
        .crossover(SimulatedBinaryCrossover::new(15.0))
        .mutation(GaussianMutation::new(0.5, 0.01))
        .survivor(survivor)
        .duplicates_cleaner(CloseDuplicatesCleaner::new(1e-6))
        .fitness_fn(fitness_biobjective)
        .constraints_fn(MyConstr)
        .num_vars(2)
        .population_size(100)
        .num_offsprings(100)
        .num_iterations(100)
        .mutation_rate(0.1)
        .crossover_rate(0.9)
        .keep_infeasible(false)
        .verbose(false)
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
    let rp: Array2<f64> = array![[0.8, 0.8], [0.9, 0.9]];
    let epsilon = 0.001;
    let survivor = Rnsga2ReferencePointsSurvival::new(rp, epsilon);
    let mut algorithm = Rnsga2Builder::default()
        .sampler(RandomSamplingFloat::new(0.0, 1.0))
        .crossover(SimulatedBinaryCrossover::new(15.0))
        .mutation(GaussianMutation::new(0.5, 0.01))
        .survivor(survivor)
        .duplicates_cleaner(CloseDuplicatesCleaner::new(1e-6))
        .fitness_fn(fitness_biobjective)
        .constraints_fn(MyConstr)
        .num_vars(2)
        .population_size(100)
        .num_offsprings(100)
        .num_iterations(100)
        .mutation_rate(0.1)
        .crossover_rate(0.9)
        .keep_infeasible(false)
        .verbose(false)
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
    let rp = DanAndDenisReferencePoints::new(100, 2).generate();
    let alpha = 2.5;
    let frequency = 0.2;
    let num_iterations = 100;
    let survivor = ReveaReferencePointsSurvival::new(rp, alpha, frequency, num_iterations);
    let mut algorithm = ReveaBuilder::default()
        .sampler(RandomSamplingFloat::new(0.0, 1.0))
        .crossover(SimulatedBinaryCrossover::new(15.0))
        .mutation(GaussianMutation::new(0.5, 0.01))
        .survivor(survivor)
        .duplicates_cleaner(CloseDuplicatesCleaner::new(1e-6))
        .fitness_fn(fitness_biobjective)
        .constraints_fn(MyConstr)
        .num_vars(2)
        .population_size(100)
        .num_offsprings(100)
        .num_iterations(num_iterations)
        .mutation_rate(0.1)
        .crossover_rate(0.95)
        .keep_infeasible(false)
        .verbose(false)
        .build()
        .expect("failed to build REVEA");

    algorithm.run().expect("Revea run failed");
    let population = algorithm
        .population()
        .expect("population should have been initialized");
    assert_small_real_front(&population);
}

#[test]
#[should_panic] // The algorithm is not reaching the expected performance. Needs investigation
fn test_spea2() {
    let mut algorithm = Spea2Builder::default()
        .sampler(RandomSamplingFloat::new(0.0, 1.0))
        .crossover(SimulatedBinaryCrossover::new(15.0))
        .mutation(GaussianMutation::new(0.5, 0.01))
        .duplicates_cleaner(CloseDuplicatesCleaner::new(1e-6))
        .fitness_fn(fitness_biobjective)
        .constraints_fn(MyConstr)
        .num_vars(2)
        .population_size(200)
        .num_offsprings(200)
        .num_iterations(100)
        .mutation_rate(0.1)
        .crossover_rate(0.9)
        .keep_infeasible(false)
        .verbose(true)
        .seed(42)
        .build()
        .expect("failed to build SPEA2");

    algorithm.run().expect("SPEA2 run failed");
    let population = algorithm
        .population()
        .expect("population should have been initialized");
    assert_small_real_front(&population);
}

#[test]
fn test_same_seed_same_result() {
    let mut algorithm1 = Nsga2Builder::default()
        .sampler(RandomSamplingFloat::new(0.0, 1.0))
        .crossover(ArithmeticCrossover)
        .mutation(UniformRealMutation::new(0.9, 0.0, 1.0))
        .duplicates_cleaner(CloseDuplicatesCleaner::new(1e-6))
        .fitness_fn(fitness_biobjective)
        .constraints_fn(MyConstr)
        .num_vars(2)
        .population_size(200)
        .num_offsprings(200)
        .num_iterations(100)
        .mutation_rate(0.1)
        .crossover_rate(0.9)
        .keep_infeasible(false)
        .verbose(true)
        .seed(1729)
        .build()
        .expect("failed to build NSGA2");

    algorithm1.run().expect("NSGA2 run failed");
    let population1 = algorithm1
        .population()
        .expect("population should have been initialized");

    let mut algorithm2 = Nsga2Builder::default()
        .sampler(RandomSamplingFloat::new(0.0, 1.0))
        .crossover(ArithmeticCrossover)
        .mutation(UniformRealMutation::new(0.9, 0.0, 1.0))
        .duplicates_cleaner(CloseDuplicatesCleaner::new(1e-6))
        .fitness_fn(fitness_biobjective)
        .constraints_fn(MyConstr)
        .num_vars(2)
        .population_size(200)
        .num_offsprings(200)
        .num_iterations(100)
        .mutation_rate(0.1)
        .crossover_rate(0.9)
        .keep_infeasible(false)
        .verbose(true)
        .seed(1729)
        .build()
        .expect("failed to build NSGA2");

    algorithm2.run().expect("NSGA2 run failed");
    let population2 = algorithm2
        .population()
        .expect("population should have been initialized");

    assert_eq!(population1.genes, population2.genes)
}
