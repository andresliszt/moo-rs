use ndarray::{Array1, Axis, stack};
use ordered_float::OrderedFloat;
use std::collections::HashSet;

use moors::{
    algorithms::Nsga2Builder,
    duplicates::ExactDuplicatesCleaner,
    genetic::{
        ConstraintsFn, FitnessFn, PopulationConstraints, PopulationFitness, PopulationGenes,
    },
    operators::{
        crossover::SinglePointBinaryCrossover, mutation::BitFlipMutation,
        sampling::RandomSamplingBinary,
    },
};

// problem data
const WEIGHTS: [f64; 5] = [12.0, 2.0, 1.0, 4.0, 10.0];
const VALUES: [f64; 5] = [4.0, 2.0, 1.0, 5.0, 3.0];
const CAPACITY: f64 = 15.0;

/// Compute multi-objective fitness [–total_value, total_weight]
/// Returns an Array2<f64> of shape (population_size, 2)
fn fitness_knapsack(population_genes: &PopulationGenes) -> PopulationFitness {
    // lift our fixed arrays into Array1 for dot products
    let weights_arr = Array1::from_vec(WEIGHTS.to_vec());
    let values_arr = Array1::from_vec(VALUES.to_vec());

    let total_values = population_genes.dot(&values_arr);
    let total_weights = population_genes.dot(&weights_arr);

    // stack two columns: [-total_values, total_weights]
    stack(Axis(1), &[(-&total_values).view(), total_weights.view()]).expect("stack failed")
}

fn constraints_knapsack(population_genes: &PopulationGenes) -> PopulationConstraints {
    // build a 1-D array of weights in one shot
    let weights_arr = Array1::from_vec(WEIGHTS.to_vec());

    // dot → Array1<f64>, subtract capacity → Array1<f64>,
    // then promote to 2-D (n×1) with insert_axis
    (population_genes.dot(&weights_arr) - CAPACITY).insert_axis(Axis(1))
}

#[test]
fn test_knapsack_nsga2_small_binary() {
    // build and run the NSGA-II algorithm
    let mut algorithm = Nsga2Builder::default()
        .fitness_fn(fitness_knapsack as FitnessFn)
        .constraints_fn(constraints_knapsack as ConstraintsFn)
        .sampler(RandomSamplingBinary::new())
        .crossover(SinglePointBinaryCrossover::new())
        .mutation(BitFlipMutation::new(0.5))
        .duplicates_cleaner(ExactDuplicatesCleaner::new())
        .n_vars(5)
        .population_size(100)
        .n_offsprings(32)
        .n_iterations(2)
        .crossover_rate(0.9)
        .mutation_rate(0.1)
        .lower_bound(0.0)
        .upper_bound(1.0)
        .build()
        .unwrap();

    algorithm.run().expect("NSGA2 run failed");
    let pareto = algorithm.population().best();

    // actual genes set
    let actual_genes: HashSet<Vec<OrderedFloat<f64>>> = pareto
        .genes
        .outer_iter()
        .map(|row| row.iter().map(|&v| OrderedFloat(v)).collect())
        .collect();

    // expected genes set (directly defined)
    let expected_genes: HashSet<Vec<OrderedFloat<f64>>> = [
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 1.0, 1.0, 0.0],
        [0.0, 0.0, 1.0, 1.0, 1.0],
    ]
    .iter()
    .map(|arr| arr.iter().map(|&v| OrderedFloat(v)).collect())
    .collect();

    assert_eq!(actual_genes, expected_genes, "Pareto front genes mismatch");

    // actual fitness set
    let actual_fitness: HashSet<Vec<OrderedFloat<f64>>> = pareto
        .fitness
        .outer_iter()
        .map(|row| row.iter().map(|&v| OrderedFloat(v)).collect())
        .collect();

    // expected fitness set (directly defined)
    let expected_fitness: HashSet<Vec<OrderedFloat<f64>>> = [
        [-0.0, 0.0],
        [-1.0, 1.0],
        [-2.0, 2.0],
        [-3.0, 3.0],
        [-5.0, 4.0],
        [-6.0, 5.0],
        [-7.0, 6.0],
        [-8.0, 7.0],
        [-9.0, 15.0],
    ]
    .iter()
    .map(|arr| arr.iter().map(|&v| OrderedFloat(v)).collect())
    .collect();

    assert_eq!(
        actual_fitness, expected_fitness,
        "Pareto front fitness mismatch"
    );
}
