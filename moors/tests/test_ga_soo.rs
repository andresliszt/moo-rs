use ndarray::{Array1, Array2, Axis};

use moors::{
    AlgorithmSOOBuilder, CloseDuplicatesCleaner, GaussianMutation, PopulationSOO,
    RandomSamplingFloat, SimulatedBinaryCrossover, selection::soo::RankSelection,
    survival::soo::FitnessSurvival,
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
    let mut algorithm = AlgorithmSOOBuilder::default()
        .sampler(RandomSamplingFloat::new(-1.0, 1.0))
        .crossover(SimulatedBinaryCrossover::new(15.0))
        .mutation(GaussianMutation::new(0.05, 0.1))
        .selector(RankSelection::new())
        .survivor(FitnessSurvival::new())
        .duplicates_cleaner(CloseDuplicatesCleaner::new(1e-6))
        .fitness_fn(fitness_sphere)
        .constraints_fn(constraints_sphere)
        .num_constraints(1)
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
