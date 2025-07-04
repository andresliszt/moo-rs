use ndarray::{Array1, Array2, Axis};

use moors::{
    AlgorithmBuilder, CloseDuplicatesCleaner, GaussianMutation, PopulationSOO, RandomSamplingFloat,
    SimulatedBinaryCrossover, impl_constraints_fn,
    selection::soo::RankSelection,
    survival::soo::{FitnessConstraintsPenaltySurvival, FitnessSurvival},
};

/// Simple minimization of 1 - (x**2 + y**2 + z**2)
fn fitness_sphere(population: &Array2<f64>) -> Array1<f64> {
    // For each row [x, y, z], compute 1 - x^2 + y^2 + z^2
    population.map_axis(Axis(1), |row| 1.0 - row.dot(&row))
}

fn constraints_sphere(population: &Array2<f64>) -> Array1<f64> {
    // For each row [x, y, z], compute x^2 + y^2 + z^2 - 1 <= 0
    population.map_axis(Axis(1), |row| row.dot(&row)) - 1.0
}

#[test]
fn test_ga_minimize_parabolid() {
    let mut algorithm = AlgorithmBuilder::default()
        .sampler(RandomSamplingFloat::new(-1.0, 1.0))
        .crossover(SimulatedBinaryCrossover::new(15.0))
        .mutation(GaussianMutation::new(0.05, 0.1))
        .selector(RankSelection)
        .survivor(FitnessSurvival)
        .duplicates_cleaner(CloseDuplicatesCleaner::new(1e-6))
        .fitness_fn(fitness_sphere)
        .constraints_fn(constraints_sphere)
        .num_vars(3)
        .population_size(100)
        .num_offsprings(50)
        .num_iterations(150)
        .mutation_rate(0.1)
        .crossover_rate(0.9)
        .keep_infeasible(false)
        .verbose(true)
        .seed(123)
        .build()
        .expect("failed to build GA");

    algorithm.run().expect("GA run failed");
    let population: PopulationSOO = algorithm
        .population
        .expect("population should have been initialized");

    for &fit in population.fitness.iter() {
        assert!(fit.abs() < 1e-3);
    }
    for gene in population.genes.rows() {
        let x = gene[0];
        let y = gene[1];
        let z = gene[2];
        let constraint_value = x * x + y * y + z * z - 1.0;
        assert!(constraint_value.abs() < 1e-3);
    }
}

/// Problem: minimize f(x, y) = x² + y²
/// Subject to the equality constraint x + y = 1.
/// The optimal solution is the point on the line closest to the origin, (0.5, 0.5).

fn fitness_quadratic(population: &Array2<f64>) -> Array1<f64> {
    // Compute f = x² + y² for each row [x, y]
    population.map_axis(Axis(1), |row| row.dot(&row))
}

fn line_constraint(genes: &Array2<f64>) -> Array1<f64> {
    // h(genes) = x + y – 1 = 0
    genes.map_axis(Axis(1), |row| 1.0 - row.sum())
}

// Generate a struct implementing moors::ConstraintsFn for the single equality:
// |x+y−1| − ε ≤ 0
impl_constraints_fn!(
    LineProjectionConstraints,
    eq = [line_constraint],
    lower_bound = 0.0,
    upper_bound = 1.0
);

#[test]
#[should_panic]
// This test is failing due to this issue: https://github.com/andresliszt/moo-rs/issues/189
// The problem here is that the best individual is the one that got CV = 0.0, that's due the
// truncation we're using to compute constraints violations defaulted in 1e-6. With default as
// 1e-4 this test passes. Investigate the truncation effect.
fn test_minimize_projection_on_line() {
    let mut algorithm = AlgorithmBuilder::default()
        .sampler(RandomSamplingFloat::new(0.0, 1.0))
        .crossover(SimulatedBinaryCrossover::new(15.0))
        .mutation(GaussianMutation::new(0.9, 0.1))
        .selector(RankSelection)
        .survivor(FitnessSurvival)
        .duplicates_cleaner(CloseDuplicatesCleaner::new(1e-6))
        .fitness_fn(fitness_quadratic)
        .constraints_fn(LineProjectionConstraints)
        .num_vars(2)
        .population_size(200)
        .num_offsprings(200)
        .num_iterations(600)
        .mutation_rate(0.2)
        .crossover_rate(0.95)
        .keep_infeasible(true)
        .verbose(true)
        .seed(123)
        .build()
        .expect("failed to build GA");

    algorithm.run().expect("GA run failed");
    let population = algorithm
        .population
        .expect("population should have been initialized");

    println!("GENES {}", population.best().genes);
    // Verify every individual is (approximately) (0.5, 0.5)
    for gene in population.best().genes.rows() {
        assert!((gene[0] - 0.5).abs() < 0.01, "x ≈ 0.5, got {}", gene[0]);
        assert!((gene[1] - 0.5).abs() < 0.01, "y ≈ 0.5, got {}", gene[1]);
    }
}

#[test]
fn test_minimize_projection_on_line_constraints_penalty_survival() {
    let mut algorithm = AlgorithmBuilder::default()
        .sampler(RandomSamplingFloat::new(0.0, 1.0))
        .crossover(SimulatedBinaryCrossover::new(15.0))
        .mutation(GaussianMutation::new(0.9, 0.1))
        .selector(RankSelection)
        .survivor(FitnessConstraintsPenaltySurvival::new(1.0))
        .duplicates_cleaner(CloseDuplicatesCleaner::new(1e-6))
        .fitness_fn(fitness_quadratic)
        .constraints_fn(LineProjectionConstraints)
        .num_vars(2)
        .population_size(200)
        .num_offsprings(200)
        .num_iterations(600)
        .mutation_rate(0.2)
        .crossover_rate(0.95)
        .keep_infeasible(true)
        .verbose(true)
        .seed(123)
        .build()
        .expect("failed to build GA");

    algorithm.run().expect("GA run failed");
    let population = algorithm
        .population
        .expect("population should have been initialized");

    // Verify every individual is (approximately) (0.5, 0.5)
    for gene in population.best().genes.rows() {
        assert!((gene[0] - 0.5).abs() < 0.01, "x ≈ 0.5, got {}", gene[0]);
        assert!((gene[1] - 0.5).abs() < 0.01, "y ≈ 0.5, got {}", gene[1]);
    }
}
