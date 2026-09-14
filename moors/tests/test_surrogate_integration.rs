//! Integration test: plug a [`moors::SurrogateFitnessFn`] into a real algorithm
//! build and confirm the GA still makes progress while only a fraction of
//! each generation is evaluated with the real (here: cheap, but stand-in for
//! "expensive") fitness function.
#![cfg(feature = "surrogate")]

use ndarray::{Array1, Array2, Axis};

use moors::{
    AlgorithmBuilder, CloseDuplicatesCleaner, GaussianMutation, GaussianProcessSurrogate,
    NoConstraints, PopulationSOO, RandomSamplingFloat, SimulatedBinaryCrossover, SurrogateConfig,
    SurrogateFitnessFn, selection::soo::RankSelection, survival::soo::FitnessSurvival,
};

/// Minimize f(x, y, z) = x² + y² + z², the same problem as
/// `test_ga_soo::test_ga_minimize_parabolid`, but with the fitness function
/// wrapped behind a surrogate instead of called directly.
fn sphere(population: &Array2<f64>) -> Array1<f64> {
    population.map_axis(Axis(1), |row| row.dot(&row))
}

#[test]
fn test_ga_with_surrogate_fitness_makes_progress() {
    let surrogate_fitness = SurrogateFitnessFn::new(
        sphere,
        // Small archive / infrequent retraining, so the test stays fast.
        GaussianProcessSurrogate::new(30, 5),
        SurrogateConfig {
            infill_ratio: 0.5,
            warmup_iterations: 3,
        },
    );

    let mut algorithm = AlgorithmBuilder::default()
        .sampler(RandomSamplingFloat::new(-1.0, 1.0))
        .crossover(SimulatedBinaryCrossover::new(15.0))
        .mutation(GaussianMutation::new(0.1, 0.1))
        .selector(RankSelection)
        .survivor(FitnessSurvival)
        .duplicates_cleaner(CloseDuplicatesCleaner::new(1e-6))
        .fitness_fn(surrogate_fitness)
        .constraints_fn(NoConstraints)
        .num_vars(3)
        .population_size(24)
        .num_offsprings(12)
        .num_iterations(15)
        .mutation_rate(0.2)
        .crossover_rate(0.9)
        .seed(123)
        .build()
        .expect("failed to build GA");

    algorithm.run().expect("GA run failed");
    // NoConstraints reports its constraints with an Ix2 shape (N x 0).
    let population: PopulationSOO<ndarray::Ix2> = algorithm
        .population
        .expect("population should have been initialized");

    // Initial random genes in [-1, 1]^3 have an expected fitness around 1.0;
    // even with most generations approximated by the surrogate, the GA
    // should still improve substantially on that.
    let best_fitness = population
        .fitness
        .iter()
        .cloned()
        .fold(f64::INFINITY, f64::min);
    assert!(
        best_fitness < 0.5,
        "expected meaningful progress with a surrogate-assisted fitness, got {best_fitness}"
    );
}
