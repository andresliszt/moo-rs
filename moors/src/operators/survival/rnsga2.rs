use std::cmp::Ordering;
use std::fmt::Debug;

use ndarray::{Array1, Array2, ArrayView1, Axis};

use crate::algorithms::AlgorithmContext;
use crate::genetic::{Fronts, PopulationFitness};
use crate::helpers::extreme_points::{get_ideal, get_nadir};
use crate::operators::survival::{GeneticOperator, SurvivalOperator, SurvivalScoringComparison};
use crate::random::RandomGenerator;

/// Implementation of the survival operator for the R-NSGA2 algorithm presented in the paper
/// Reference Point Based Multi-Objective Optimization Using Evolutionary Algorithms

#[derive(Clone, Debug)]
pub struct Rnsga2ReferencePointsSurvival {
    reference_points: Array2<f64>,
    epsilon: f64,
}

impl GeneticOperator for Rnsga2ReferencePointsSurvival {
    fn name(&self) -> String {
        "Rnsga2ReferencePointsSurvival".to_string()
    }
}

impl Rnsga2ReferencePointsSurvival {
    pub fn new(reference_points: Array2<f64>, epsilon: f64) -> Self {
        Self {
            reference_points,
            epsilon,
        }
    }
}

impl SurvivalOperator for Rnsga2ReferencePointsSurvival {
    fn scoring_comparison(&self) -> SurvivalScoringComparison {
        SurvivalScoringComparison::Minimize
    }

    fn set_survival_score(
        &self,
        fronts: &mut Fronts,
        rng: &mut dyn RandomGenerator,
        _algorithm_context: &AlgorithmContext,
    ) {
        let len_fronts = fronts.len();
        let n_objectives = fronts[0].fitness.ncols();
        let weights = Array1::from_elem(n_objectives, 1.0 / (n_objectives as f64));

        for front in fronts.iter_mut().take(len_fronts.saturating_sub(1)) {
            let nadir = get_nadir(&front.fitness);
            let ideal = get_ideal(&front.fitness);
            let survival_score = assign_crowding_distance_to_inner_front(
                &front.fitness,
                &self.reference_points,
                &weights,
                &nadir,
                &ideal,
            );
            front
                .set_survival_score(survival_score)
                .expect("Failed to set survival score in Rsga2");
        }
        // Process the last front with the special crowding_distance_last_front function
        if let Some(last_front) = fronts.last_mut() {
            let nadir = get_nadir(&last_front.fitness);
            let ideal = get_ideal(&last_front.fitness);
            let survival_score = assign_crowding_distance_splitting_front(
                &last_front.fitness,
                &self.reference_points,
                &weights,
                self.epsilon,
                &nadir,
                &ideal,
                rng,
            );
            last_front
                .set_survival_score(survival_score)
                .expect("Failed to set survival score in Rsga2");
        }
    }
}

/// Computes the weighted, normalized Euclidean distance between two objective vectors `f1` and `f2`.
/// Normalization is performed using the provided ideal (`ideal`) and nadir (`nadir`) points.
/// If for any objective the range (nadir - ideal) is zero, the normalized difference is set to 0.0.
/// This is the equation (3) in the presented paper
fn weighted_normalized_euclidean_distance(
    f1: &ArrayView1<f64>,
    f2: &ArrayView1<f64>,
    weights: &Array1<f64>,
    ideal: &Array1<f64>,
    nadir: &Array1<f64>,
) -> f64 {
    // Compute the element-wise difference between f1 and f2.
    let diff = f1 - f2;
    // Compute the range for normalization.
    let ranges = nadir - ideal;
    let normalized_diff = diff / &ranges;
    // Compute the weighted sum of squared normalized differences.
    let weighted_sum_sq: f64 = normalized_diff.mapv(|x| x * x).dot(weights);
    // Return the square root of the weighted sum.
    weighted_sum_sq.sqrt()
}

fn distance_to_reference(
    front_fitness: &PopulationFitness,
    reference_points: &Array2<f64>,
    weights: &Array1<f64>,
    ideal: &Array1<f64>,
    nadir: &Array1<f64>,
) -> Array1<f64> {
    // --- Step 1: Compute initial crowding distances based on reference points ---
    // Initialize each solution's crowding distance with infinity.
    let num_front_individuals = front_fitness.nrows();
    let mut crowding = vec![f64::INFINITY; num_front_individuals];

    for rp in reference_points.axis_iter(Axis(0)) {
        let mut solution_distances: Vec<(usize, f64)> = front_fitness
            .axis_iter(Axis(0))
            .enumerate()
            .map(|(i, sol)| {
                let distance =
                    weighted_normalized_euclidean_distance(&sol, &rp, weights, ideal, nadir);
                (i, distance)
            })
            .collect();
        // Sort solutions by distance (ascending order).
        solution_distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        // Assign rank based on order: the closest solution gets rank 1, the next gets rank 2, etc.
        for (rank, (i, _)) in solution_distances.into_iter().enumerate() {
            let current_rank = (rank + 1) as f64;
            if current_rank < crowding[i] {
                crowding[i] = current_rank;
            }
        }
    }
    Array1::from_vec(crowding)
}

/// Assigns a crowding distance (ranking) to each solution.
///
/// The algorithm works in two stages:
///
///   Step 1: For each reference point, compute the normalized distance from each solution and sort the solutions
///           in ascending order (closest gets rank 1). For each solution, store the best (minimum) rank across all
///           reference points.
///
///   Step 2 (extended as Step 3): Group solutions that have a sum of normalized differences (across objectives) ≤ epsilon.
///           From each group, randomly retain one solution and assign a penalty (infinity) as the crowding distance
///           to the rest.
///
/// # Parameters
/// - `solutions`: An Array2<f64> where each row is a solution.
/// - `reference_points`: An Array2<f64> where each row is a reference point.
/// - `weights`, `ideal`, `nadir`: Parameters used to normalize the distance.
/// - `epsilon`: The threshold used to group similar solutions.
/// - `rng`: A mutable reference to an object implementing `RngCore` to be used for random shuffling.
///
/// # Returns
/// A vector of crowding distances (as f64) for each solution.
fn assign_crowding_distance_to_inner_front(
    front_fitness: &PopulationFitness,
    reference_points: &Array2<f64>,
    weights: &Array1<f64>,
    nadir: &Array1<f64>,
    ideal: &Array1<f64>,
) -> Array1<f64> {
    distance_to_reference(&front_fitness, &reference_points, &weights, &ideal, &nadir)
}

fn assign_crowding_distance_splitting_front(
    front_fitness: &PopulationFitness,
    reference_points: &Array2<f64>,
    weights: &Array1<f64>,
    epsilon: f64,
    nadir: &Array1<f64>,
    ideal: &Array1<f64>,
    rng: &mut dyn RandomGenerator,
) -> Array1<f64> {
    let num_front_individuals = front_fitness.nrows();
    let mut crowding = assign_crowding_distance_to_inner_front(
        &front_fitness,
        &reference_points,
        &weights,
        nadir,
        ideal,
    );

    // --- Step 3: Group similar solutions using epsilon ---
    // Group solutions that have a sum of normalized differences ≤ epsilon.
    let mut visited = vec![false; num_front_individuals];
    let mut groups: Vec<Vec<usize>> = Vec::new();

    for i in 0..num_front_individuals {
        if visited[i] {
            continue;
        }
        let mut group = vec![i];
        visited[i] = true;
        let sol_i = front_fitness.row(i);
        for j in (i + 1)..num_front_individuals {
            if !visited[j] {
                let sol_j = front_fitness.row(j);
                let sum_diff = weighted_normalized_euclidean_distance(
                    &sol_i, &sol_j, &weights, &ideal, &nadir,
                );
                if sum_diff <= epsilon {
                    group.push(j);
                    visited[j] = true;
                }
            }
        }
        groups.push(group);
    }

    // For each group with more than one solution, randomly retain one solution and assign a penalty (infinity)
    // to all other solutions in that group.
    for group in groups {
        if group.len() > 1 {
            let mut group_copy = group.clone();
            rng.shuffle_vec_usize(&mut group_copy);
            // Retain the first solution in the shuffled group.
            // All other solutions in the group receive infinity.
            for &idx in group_copy.iter().skip(1) {
                crowding[idx] = f64::INFINITY;
            }
        }
    }
    crowding
}

#[cfg(test)]
mod tests {
    use core::f64;

    use super::*;
    use ndarray::array;

    #[test]
    fn test_distance_zero() {
        // When both objective vectors are identical, the distance should be 0.
        let f1 = array![1.0, 2.0];
        let f2 = array![1.0, 2.0];
        let weights = array![1.0, 1.0];
        let ideal = array![0.0, 0.0];
        let nadir = array![1.0, 1.0];
        let distance = weighted_normalized_euclidean_distance(
            &f1.view(),
            &f2.view(),
            &weights,
            &ideal,
            &nadir,
        );
        assert_eq!(distance, 0.0);
    }

    #[test]
    fn test_distance_simple() {
        // Example:
        // f1 = [3, 4], f2 = [1, 2]
        // ideal = [1, 2], nadir = [5, 6] => range = [4, 4]
        // Normalized differences = [(3-1)/4, (4-2)/4] = [0.5, 0.5]
        // With weights = [1, 1], the weighted sum of squares is 0.5^2 + 0.5^2 = 0.25 + 0.25 = 0.5,
        // and the distance is sqrt(0.5).
        let f1 = array![3.0, 4.0];
        let f2 = array![1.0, 2.0];
        let weights = array![1.0, 1.0];
        let ideal = array![1.0, 2.0];
        let nadir = array![5.0, 6.0];
        let distance = weighted_normalized_euclidean_distance(
            &f1.view(),
            &f2.view(),
            &weights,
            &ideal,
            &nadir,
        );
        let expected = (0.25_f64 + 0.25).sqrt();
        assert!((distance - expected).abs() < 1e-6);
    }

    #[test]
    fn test_distance_with_zero_range() {
        // Test scenario where one of the objectives has a zero range.
        // For the first objective: ideal = 1, nadir = 1, so range = 0 and normalized difference = 0.
        // For the second objective: ideal = 2, nadir = 6, so range = 4 and normalized difference = (4-2)/4 = 0.5.
        let f1 = array![3.0, 4.0];
        let f2 = array![1.0, 2.0];
        let weights = array![1.0, 1.0];
        let ideal = array![1.0, 2.0];
        let nadir = array![1.0, 6.0];
        let distance: f64 = weighted_normalized_euclidean_distance(
            &f1.view(),
            &f2.view(),
            &weights,
            &ideal,
            &nadir,
        );
        let expected: f64 = f64::INFINITY;
        assert!(expected == distance);
    }

    #[test]
    fn test_single_solution_single_reference() {
        let front_fitness = array![[0.5, 0.5]];
        let reference_points = array![[0.5, 0.5]];
        let weights = array![1.0, 1.0];
        let ideal = array![0.0, 0.0];
        let nadir = array![1.0, 1.0];

        let result =
            distance_to_reference(&front_fitness, &reference_points, &weights, &ideal, &nadir);
        let expected = array![1.0];
        assert_eq!(result, expected);
    }

    /// Test 2: Two solutions with a single reference point.
    /// Calculation:
    /// - For [0.5, 0.5] vs [0.5, 0.5]: distance = 0.0, rank 1.
    /// - For [0.2, 0.8] vs [0.5, 0.5]: distance > 0.0, rank 2.
    #[test]
    fn test_two_solutions_single_reference() {
        let front_fitness = array![[0.5, 0.5], [0.2, 0.8]];
        let reference_points = array![[0.5, 0.5]];
        let weights = array![1.0, 1.0];
        let ideal = array![0.0, 0.0];
        let nadir = array![1.0, 1.0];

        let result =
            distance_to_reference(&front_fitness, &reference_points, &weights, &ideal, &nadir);
        // Expected: the first solution gets rank 1 and the second gets rank 2.
        let expected = array![1.0, 2.0];
        assert_eq!(result, expected);
    }

    /// Test 3: Two solutions with two reference points.
    /// Reference Points:
    ///   - Ref1: [0.0, 1.0]
    ///   - Ref2: [1.0, 0.0]
    /// Calculations:
    /// - For Ref1: solution 1 is closer (rank 1) and solution 2 is farther (rank 2).
    /// - For Ref2: solution 2 is closer (rank 1) and solution 1 is farther (rank 2).
    /// Final crowding values are the minimum rank per solution:
    ///   solution1: min(1, 2) = 1, solution2: min(2, 1) = 1.
    #[test]
    fn test_two_solutions_two_references() {
        let front_fitness = array![[0.2, 0.8], [0.8, 0.2]];
        let reference_points = array![[0.0, 1.0], [1.0, 0.0]];
        let weights = array![1.0, 1.0];
        let ideal = array![0.0, 0.0];
        let nadir = array![1.0, 1.0];

        let result =
            distance_to_reference(&front_fitness, &reference_points, &weights, &ideal, &nadir);
        let expected = array![1.0, 1.0];
        assert_eq!(result, expected);
    }

    /// Test 4: Multiple solutions with multiple reference points.
    /// Three solutions and two reference points:
    /// Front fitness:
    ///   - Solution 1: [0.1, 0.9]
    ///   - Solution 2: [0.4, 0.6]
    ///   - Solution 3: [0.9, 0.1]
    /// Reference points:
    ///   - Ref1: [0.0, 1.0]
    ///   - Ref2: [1.0, 0.0]
    /// Calculations:
    /// For Ref1:
    ///   - Distance(solution1) ≈ 0.1414 → rank 1.
    ///   - Distance(solution2) ≈ 0.5657 → rank 2.
    ///   - Distance(solution3) ≈ 1.2728 → rank 3.
    /// For Ref2:
    ///   - Distance(solution1) ≈ 1.2728 → rank 3.
    ///   - Distance(solution2) ≈ 0.8485 → rank 2.
    ///   - Distance(solution3) ≈ 0.1414 → rank 1.
    /// Final crowding values:
    ///   - Solution 1: min(1, 3) = 1.
    ///   - Solution 2: min(2, 2) = 2.
    ///   - Solution 3: min(3, 1) = 1.
    #[test]
    fn test_multiple_solutions_multiple_references() {
        let front_fitness = array![[0.1, 0.9], [0.4, 0.6], [0.9, 0.1]];
        let reference_points = array![[0.0, 1.0], [1.0, 0.0]];
        let weights = array![1.0, 1.0];
        let ideal = array![0.0, 0.0];
        let nadir = array![1.0, 1.0];

        let result =
            distance_to_reference(&front_fitness, &reference_points, &weights, &ideal, &nadir);
        let expected = array![1.0, 2.0, 1.0];
        assert_eq!(result, expected);
    }
}
