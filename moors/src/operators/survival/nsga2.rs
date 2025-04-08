use std::f64::INFINITY;
use std::fmt::Debug;

use ndarray::Array1;

use crate::algorithms::AlgorithmContext;
use crate::genetic::{Fronts, PopulationFitness};
use crate::operators::{GeneticOperator, SurvivalOperator};
use crate::random::RandomGenerator;

#[derive(Clone, Debug)]
pub struct RankCrowdingSurvival;

impl GeneticOperator for RankCrowdingSurvival {
    fn name(&self) -> String {
        "RankCrowdingSurvival".to_string()
    }
}

impl RankCrowdingSurvival {
    pub fn new() -> Self {
        Self {}
    }
}

impl SurvivalOperator for RankCrowdingSurvival {
    fn set_survival_score(
        &self,
        fronts: &mut Fronts,
        _rng: &mut dyn RandomGenerator,
        _algorithm_context: &AlgorithmContext,
    ) {
        for front in fronts.iter_mut() {
            let crowding_distance = crowding_distance(&front.fitness);
            front
                .set_survival_score(crowding_distance)
                .expect("Failed to set survival score in Nsga2");
        }
    }
}

/// Computes the crowding distance for a given Pareto population_fitness.
///
/// # Parameters:
/// - `population_fitness`: A 2D array where each row represents an individual's fitness values.
///
/// # Returns:
/// - A 1D array of crowding distances for each individual in the population_fitness.
fn crowding_distance(population_fitness: &PopulationFitness) -> Array1<f64> {
    let num_individuals = population_fitness.shape()[0];
    let num_objectives = population_fitness.shape()[1];

    // Handle edge cases
    if num_individuals <= 2 {
        let mut distances = Array1::zeros(num_individuals);
        if num_individuals > 0 {
            distances[0] = INFINITY; // Boundary individuals
        }
        if num_individuals > 1 {
            distances[num_individuals - 1] = INFINITY;
        }
        return distances;
    }

    // Initialize distances to zero
    let mut distances = Array1::zeros(num_individuals);

    // Iterate over each objective
    for obj_idx in 0..num_objectives {
        // Extract the column for the current objective
        let objective_values = population_fitness.column(obj_idx);

        // Sort indices based on the objective values
        let mut sorted_indices: Vec<usize> = (0..num_individuals).collect();
        sorted_indices.sort_by(|&i, &j| {
            objective_values[i]
                .partial_cmp(&objective_values[j])
                .unwrap()
        });

        // Assign INFINITY to border. TODO: Not sure if worst should have infinity
        distances[sorted_indices[0]] = INFINITY;
        distances[sorted_indices[num_individuals - 1]] = INFINITY;

        // Get min and max values for normalization
        let min_value = objective_values[sorted_indices[0]];
        let max_value = objective_values[sorted_indices[num_individuals - 1]];
        let range = max_value - min_value;

        if range != 0.0 {
            // Calculate crowding distances for intermediate individuals
            for k in 1..(num_individuals - 1) {
                let next = objective_values[sorted_indices[k + 1]];
                let prev = objective_values[sorted_indices[k - 1]];
                distances[sorted_indices[k]] += (next - prev) / range;
            }
        }
    }

    distances
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, Axis, array, concatenate};

    use crate::genetic::Population;
    use crate::random::NoopRandomGenerator;

    #[test]
    /// Tests the calculation of crowding distances for a given population fitness matrix.
    ///
    /// The test defines a `population_fitness` matrix for four individuals:
    ///     [1.0, 2.0]
    ///     [2.0, 1.0]
    ///     [1.5, 1.5]
    ///     [3.0, 3.0]
    ///
    /// For each objective, the ideal (minimum) and nadir (maximum) values are computed.
    /// Then, for interior solutions, the crowding distance is calculated based on the normalized difference
    /// between the neighboring solutions. According to the classical NSGA-II method (which sums the contributions),
    /// the expected crowding distances are as follows:
    ///   - Corner individuals (first, second, and fourth) are assigned INFINITY.
    ///   - The middle individual [1.5, 1.5] has a crowding distance of 1.0 (since its contribution from each objective sums to 1.0).
    ///
    /// The test asserts that the computed crowding distances match the expected values:
    ///     expected = [INFINITY, INFINITY, 1.0, INFINITY]
    fn test_crowding_distance() {
        // Define a population_fitness with multiple individuals.
        let population_fitness = array![[1.0, 2.0], [2.0, 1.0], [1.5, 1.5], [3.0, 3.0]];

        // Compute crowding distances.
        let distances = crowding_distance(&population_fitness);

        // Expected distances: the corner individuals are assigned INFINITY and the middle individual sums to 1.0.
        let expected = array![
            std::f64::INFINITY,
            std::f64::INFINITY,
            1.0,
            std::f64::INFINITY
        ];

        assert_eq!(distances.as_slice().unwrap(), expected.as_slice().unwrap());
    }

    #[test]
    fn test_crowding_distance_single_individual() {
        // Define a population_fitness with a single individual
        let population_fitness = array![[1.0, 2.0]];

        // Compute crowding distances
        let distances = crowding_distance(&population_fitness);

        // Expected: single individual has INFINITY
        let expected = array![INFINITY];

        assert_eq!(distances.as_slice().unwrap(), expected.as_slice().unwrap());
    }

    #[test]
    fn test_crowding_distance_two_individuals() {
        // Define a population_fitness with two individuals
        let population_fitness = array![[1.0, 2.0], [2.0, 1.0]];

        // Compute crowding distances
        let distances = crowding_distance(&population_fitness);

        // Expected: both are corner individuals with INFINITY
        let expected = array![INFINITY, INFINITY];

        assert_eq!(distances.as_slice().unwrap(), expected.as_slice().unwrap());
    }

    #[test]
    fn test_crowding_distance_same_fitness_values() {
        // Define a population_fitness where all individuals have the same fitness values
        let population_fitness = array![[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]];

        // Compute crowding distances
        let distances = crowding_distance(&population_fitness);

        // Expected: all distances should remain zero except for the first
        let expected = array![INFINITY, 0.0, 0.0, 0.0, INFINITY];

        assert_eq!(distances.as_slice().unwrap(), expected.as_slice().unwrap());
    }

    #[test]
    /// Tests that the survival score is correctly set using the crowding_distance function.
    fn test_set_survival_score() {
        // Build a population with 4 individuals.
        let fitness: Array2<f64> = array![[1.0, 2.0], [2.0, 1.0], [1.5, 1.5], [3.0, 3.0]];
        let genes: Array2<f64> = array![[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0]];
        let rank: Array1<usize> = array![0_usize, 0_usize, 0_usize, 0_usize];
        let population = Population {
            genes,
            fitness,
            constraints: None,
            rank: Some(rank),
            survival_score: None,
        };
        let mut fronts: Vec<Population> = vec![population];

        let selector = RankCrowdingSurvival::new();
        let mut rng = NoopRandomGenerator::new();
        // create context (not used in the algorithm)
        let _context = AlgorithmContext::new(10, 10, 5, 2, 1, None, None, None);
        selector.set_survival_score(&mut fronts, &mut rng, &_context);

        let expected: Array1<f64> = array![
            std::f64::INFINITY,
            std::f64::INFINITY,
            1.0,
            std::f64::INFINITY
        ];
        let actual = fronts[0].survival_score.clone().unwrap();
        assert_eq!(actual.as_slice().unwrap(), expected.as_slice().unwrap());
    }

    #[test]
    fn test_survival_selection_all_survive_single_front() {
        // All individuals belong to a single front (rank 0) and n_survive equals the population size.
        let genes: Array2<f64> = array![[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]];
        let fitness: Array2<f64> = array![[0.1, 0.9], [0.2, 0.8], [0.3, 0.7]];
        let population = Population::new(genes.clone(), fitness.clone(), None, None);
        let n_survive = 3;
        let mut selector = RankCrowdingSurvival;
        assert_eq!(selector.name(), "RankCrowdingSurvival");
        let mut _rng = NoopRandomGenerator::new();
        // create context (not used in the algorithm)
        let _context = AlgorithmContext::new(10, 10, 5, 2, 1, None, None, None);
        let new_population = selector.operate(population, n_survive, &mut _rng, &_context);

        // The resulting population should remain unchanged.
        assert_eq!(new_population.genes, genes);
        assert_eq!(new_population.fitness, fitness);
        assert_eq!(
            new_population.rank.unwrap(),
            array![0_usize, 0_usize, 0_usize]
        );
    }

    #[test]
    fn test_survival_selection_multiple_fronts() {
        /*
        Tests survival selection in NSGA-II when multiple fronts are present.

        Scenario:
          - Front 1 (should be rank 0): 2 individuals.
          - Front 2 (should be rank 1): 4 individuals, but only 2 are needed to reach n_survive = 4.

        For Front 2 with fitness values:
             [0.3, 0.7], [0.4, 0.6], [0.5, 0.5], [0.6, 0.4]
        the extreme individuals receive INFINITY crowding distance.
        Hence, the individuals with fitness [0.3, 0.7] and [0.6, 0.4] (corresponding to genes [4.0, 5.0] and [10.0, 11.0])
        are selected.
        */
        // Front 1: 2 individuals.
        let front1_genes: Array2<f64> = array![[0.0, 1.0], [2.0, 3.0]];
        let front1_fitness: Array2<f64> = array![[0.0, 0.1], [0.1, 0.0]];

        // Front 2: 4 individuals.
        let front2_genes: Array2<f64> = array![[4.0, 5.0], [6.0, 7.0], [8.0, 9.0], [10.0, 11.0]];
        let front2_fitness: Array2<f64> = array![[0.3, 0.7], [0.4, 0.6], [0.5, 0.5], [0.6, 0.4]];

        // Combine genes and fitness from both fronts.
        let genes = concatenate![Axis(0), front1_genes, front2_genes];
        let fitness = concatenate![Axis(0), front1_fitness, front2_fitness];
        // Create the population using the new constructor (rank is computed internally).
        let population = Population::new(genes.clone(), fitness.clone(), None, None);
        let n_survive = 4;

        let mut selector = RankCrowdingSurvival;
        let mut _rng = NoopRandomGenerator::new();
        // create context (not used in the algorithm)
        let _context = AlgorithmContext::new(10, 10, 5, 2, 1, None, None, None);
        let new_population = selector.operate(population, n_survive, &mut _rng, &_context);

        // The final population must have 4 individuals.
        assert_eq!(new_population.len(), n_survive);

        // The final population should have 4 individuals.
        // Expected outcome:
        // - From Front 1 (rank 0): both individuals are selected.
        // - From Front 2 (rank 1): the extreme individuals based on crowding_distance are selected,
        //   yielding genes [4.0, 5.0] and [10.0, 11.0].
        let mut expected_genes: Array2<f64> =
            array![[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [10.0, 11.0]];
        let mut expected_fitness: Array2<f64> =
            array![[0.0, 0.1], [0.1, 0.0], [0.3, 0.7], [0.6, 0.4]];
        let mut new_genes = new_population.genes.clone();
        let mut new_fitness = new_population.fitness.clone();

        // Sort the arrays for comparison
        expected_genes
            .as_slice_mut()
            .unwrap()
            .sort_by(|a, b| a.partial_cmp(b).unwrap());
        expected_fitness
            .as_slice_mut()
            .unwrap()
            .sort_by(|a, b| a.partial_cmp(b).unwrap());
        new_genes
            .as_slice_mut()
            .unwrap()
            .sort_by(|a, b| a.partial_cmp(b).unwrap());
        new_fitness
            .as_slice_mut()
            .unwrap()
            .sort_by(|a, b| a.partial_cmp(b).unwrap());

        assert_eq!(new_genes, expected_genes);
        assert_eq!(new_fitness, expected_fitness);

        // Verify that the new population has the correct rank assignment:
        // The first two survivors should be rank 0 (from Front 1) and the last two rank 1 (from Front 2).
        let mut expected_rank: Array1<usize> = array![0_usize, 0_usize, 1_usize, 1_usize];
        let mut new_rank = new_population.rank.unwrap().clone();
        expected_rank.as_slice_mut().unwrap().sort();
        new_rank.as_slice_mut().unwrap().sort();
        assert_eq!(
            new_rank.as_slice().unwrap(),
            expected_rank.as_slice().unwrap()
        );
    }
}
