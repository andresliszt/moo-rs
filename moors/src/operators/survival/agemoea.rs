use std::collections::HashSet;
use std::fmt::Debug;

use ndarray::{Array1, Array2, ArrayView1, Axis, stack};
use ndarray_stats::QuantileExt;

use crate::algorithms::AlgorithmContext;
use crate::genetic::PopulationFitness;
use crate::helpers::extreme_points::get_ideal;
use crate::helpers::linalg::{cross_p_distances, lp_norm_arrayview};
use crate::operators::{
    GeneticOperator, SurvivalOperator, survival::helpers::HyperPlaneNormalization,
};
use crate::random::RandomGenerator;

struct AgeMoeaHyperPlaneNormalization;

impl AgeMoeaHyperPlaneNormalization {
    pub fn new() -> Self {
        Self
    }
}
impl HyperPlaneNormalization for AgeMoeaHyperPlaneNormalization {
    fn compute_extreme_points(&self, population_fitness: &Array2<f64>) -> Array2<f64> {
        let extreme_indices: Vec<usize> = population_fitness
            .axis_iter(Axis(1))
            .map(|col| col.argmax().unwrap())
            .collect();

        let extreme_rows: Vec<_> = extreme_indices
            .iter()
            .map(|&i| population_fitness.row(i).to_owned())
            .collect();
        stack(
            Axis(0),
            &extreme_rows
                .iter()
                .map(|row| row.view())
                .collect::<Vec<_>>(),
        )
        .unwrap()
    }
}

#[derive(Clone, Debug)]
pub struct AgeMoeaSurvival;

impl GeneticOperator for AgeMoeaSurvival {
    fn name(&self) -> String {
        "RankCrowdingSurvival".to_string()
    }
}

impl AgeMoeaSurvival {
    pub fn new() -> Self {
        Self {}
    }
}

impl SurvivalOperator for AgeMoeaSurvival {
    fn set_survival_score(
        &self,
        fronts: &mut crate::genetic::Fronts,
        _rng: &mut dyn RandomGenerator,
        _algorithm_context: &AlgorithmContext,
    ) {
        // Split the fronts into the first one and the rest
        if let Some((first_front, other_fronts)) = fronts.split_first_mut() {
            let z_min = get_ideal(&first_front.fitness);
            let translated = &first_front.fitness - &z_min;
            let normalizer = AgeMoeaHyperPlaneNormalization::new();
            let intercepts = normalizer.compute_hyperplane_intercepts(&translated);
            let normalized_first_front = translated / &intercepts;
            let central_point = get_central_point_normalized(&normalized_first_front);
            // Check if the central point is a zero vector.
            // This is the rare case where there is a single optimum,
            // meaning each individual's fitness is indeed z_ideal.
            // In this case, return a survival score vect or with the same number of elements as front_fitness,
            // filled entirely with infinity.
            if central_point.iter().all(|&x| x == 0.0) {
                first_front
                    .set_survival_score(Array1::from_elem(first_front.len(), std::f64::INFINITY))
                    .expect("Failed to set survival score in AgeMoea");
                // Process the remaining fronts with the standard function
                for front in other_fronts.iter_mut() {
                    front
                        .set_survival_score(Array1::from_elem(
                            first_front.len(),
                            std::f64::INFINITY,
                        ))
                        .expect("Failed to set survival score in AgeMoea");
                }
            } else {
                let p = compute_exponent_p(&central_point);
                let score_first_front =
                    assign_survival_scores_first_front(&normalized_first_front, p);
                first_front
                    .set_survival_score(score_first_front)
                    .expect("Failed to set survival score in AgeMoea");
                for front in other_fronts.iter_mut() {
                    //let z_min = get_ideal(&front.fitness);
                    let translated = &front.fitness - &z_min;
                    let normalized_front = translated / &intercepts;
                    let score = assign_survival_scores_higher_front(&normalized_front, p);
                    front
                        .set_survival_score(score)
                        .expect("Failed to set survival score in AgeMoea");
                }
            }
        }
    }
}

/// Computes the central point C of the non-dominated front given normalized objectives.
/// This is the equation (6) in the referenced paper
///
/// C is defined as the solution that minimizes its perpendicular distance to the unit vector
/// in the direction of (1, 1, ..., 1). That is, we define:
///
///   beta = (1, 1, ..., 1)
///   beta_hat = beta / ||beta||   (with ||beta|| = sqrt(M), where M is the number of objectives)
///
/// For each solution x, the perpendicular distance is computed as:
///
///   distance = || x - (x · beta_hat) * beta_hat ||₂
///
/// The method computes all distances in a vectorized way and returns the solution with the minimal distance.
///
/// # Arguments
///
/// * `normalized_objectives` - A 2D array (n x m) of normalized objectives (each row is a solution).
///
/// # Returns
///
/// An Array1<f64> corresponding to the central point in the normalized objective space.
fn get_central_point_normalized(normalized_fitness: &PopulationFitness) -> Array1<f64> {
    // Determine the number of objectives (columns)
    let num_objectives = normalized_fitness.shape()[1];

    // Create the reference vector beta = (1,1,...,1) and compute its unit vector beta_hat.
    let beta = Array1::<f64>::ones(num_objectives);
    let beta_norm = beta.dot(&beta).sqrt(); // sqrt(M)
    let beta_hat = beta.mapv(|x| x / beta_norm);

    // Compute the dot product of each solution (row) with beta_hat.
    // This yields an Array1 of length n.
    let dot_products = normalized_fitness.dot(&beta_hat);

    // Compute the projection for each solution: (dot_product)*beta_hat.
    // We expand dot_products to shape (n, 1) for broadcasting.
    let projections = dot_products.insert_axis(Axis(1)) * &beta_hat;

    // Compute the difference: the component of each solution perpendicular to beta_hat.
    let diff = normalized_fitness - &projections;

    // Compute the squared Euclidean norm for each row.
    let squared_norms = diff.map_axis(Axis(1), |row| row.dot(&row));

    // Find the index of the minimum squared norm.
    let min_index = squared_norms
        .argmin()
        .expect("There should be at least one solution in the front");

    // Return the row corresponding to the minimum perpendicular distance.
    normalized_fitness.row(min_index).to_owned()
}

/// Computes the exponent p from the central point C
/// This is the equation (8) in the presented paper
fn compute_exponent_p(central: &Array1<f64>) -> f64 {
    let m = central.len() as f64;
    // Ensure all components are > 0.
    for &value in central.iter() {
        assert!(
            value > 0.0,
            "All components of the central point must be > 0"
        );
    }
    // Compute the product of all coordinates.
    let product: f64 = central.iter().product();
    let ln_m = m.ln();
    let ln_product = product.ln();
    ln_m / (ln_m - ln_product)
}

/// Proximity(S) = || fₙ(S) ||ₚ
/// (Assuming the front is normalized so that the ideal point is at the origin.)
/// This is the equation (9) in the presented paper
fn proximity(normalized_individual_fitness: &ArrayView1<f64>, p: f64) -> f64 {
    lp_norm_arrayview(normalized_individual_fitness, p)
}

/// Assigns survival scores to the first front (F₁) based on the AGE-MOEA algorithm.
///
/// Extreme solutions (those with the maximum value in at least one objective)
/// are immediately assigned a score of +∞. Then, using two sets:
///   - The considered set (initially the extreme solutions)
///   - The remaining set (all other solutions)
/// the diversity for each candidate in the remaining set is computed as the sum of
/// the smallest and the second-smallest distances (using the L_p norm) from that candidate
/// to the solutions in the considered set. The candidate's value is defined as:
///
///     value[S] = diversity[S] / proximity[S]
///
/// The candidate with the maximum value is selected, its score assigned, and it is moved
/// from the remaining set to the considered set. This process continues until all candidates
/// have been processed.
///
/// # Arguments
///
/// * `front` - A (n x m) matrix where each row represents a normalized solution.
/// * `p` - The exponent for the L_p norm.
///
/// # Returns
///
/// An Array1<f64> containing the survival scores for each solution in the same order as the rows of `front`.
fn assign_survival_scores_first_front(
    normalized_front_fitness: &PopulationFitness,
    p: f64,
) -> Array1<f64> {
    let num_solutions = normalized_front_fitness.nrows();
    let num_objectives = normalized_front_fitness.ncols();

    // Initialize the scores vector.
    let mut scores = vec![0.0; num_solutions];

    // Identify extreme solutions: for each objective, select the solution with the maximum value.
    let mut extreme_indices = HashSet::new();
    for j in 0..num_objectives {
        let col = normalized_front_fitness.column(j);
        let idx = col
            .argmax()
            .expect("Each column must have at least one element");
        extreme_indices.insert(idx);
    }
    // Assign an infinite score to extreme solutions.
    for &idx in &extreme_indices {
        scores[idx] = f64::INFINITY;
    }

    // Remaining solutions: those that are not extreme.
    let mut remaining: Vec<usize> = (0..num_solutions)
        .filter(|&i| !extreme_indices.contains(&i))
        .collect();
    // Considered solutions: initially, the extreme solutions.
    let mut considered: Vec<usize> = extreme_indices.iter().copied().collect();

    // Precompute proximities for each solution using the proximity function.
    // Here, proximity(S) is defined as the L_p norm of solution S.
    let mut proximities = vec![0.0; num_solutions];
    for i in 0..num_solutions {
        proximities[i] = proximity(&normalized_front_fitness.row(i), p);
    }

    // Compute the pairwise distance matrix using the L_p norm.
    let distance_matrix = cross_p_distances(normalized_front_fitness, normalized_front_fitness, p);

    // Iterative process: while there are remaining solutions.
    while !remaining.is_empty() {
        // For each candidate S in the remaining set, compute diversity as:
        // diversity[S] = (min_{T in considered} dist[S, T] + second_min_{T in considered} dist[S, T])
        // and then compute value[S] = diversity[S] / proximity[S].
        let mut candidate_values: Vec<(usize, f64)> = Vec::new();
        for &s in &remaining {
            // Compute distances from candidate s to all solutions in the considered set.
            let mut dists: Vec<f64> = considered
                .iter()
                .map(|&t| distance_matrix[[s, t]])
                .collect();
            // Sort distances to get the smallest and second smallest.
            dists.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let diversity = if dists.len() >= 2 {
                dists[0] + dists[1]
            } else {
                // If there is only one element in the considered set.
                dists[0]
            };
            let value = if proximities[s] != 0.0 {
                diversity / proximities[s]
            } else {
                0.0
            };
            candidate_values.push((s, value));
        }

        // Select the candidate s* with the maximum value.
        if let Some(&(s_star, max_value)) = candidate_values
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        {
            scores[s_star] = max_value;
            // Add s_star to the considered set.
            considered.push(s_star);
            // Remove s_star from the remaining set.
            remaining.retain(|&x| x != s_star);
        } else {
            break;
        }
    }

    Array1::from(scores)
}

/// An Array1<f64> with the survival scores for each solution in the same order as the rows of `normalized_front_fitness`.
pub fn assign_survival_scores_higher_front(
    normalized_front_fitness: &PopulationFitness,
    p: f64,
) -> Array1<f64> {
    // Iterate over each row (solution) and compute the score.
    let scores: Vec<f64> = normalized_front_fitness
        .axis_iter(Axis(0))
        .map(|row| 1.0 / proximity(&row, p))
        .collect();
    Array1::from(scores)
}

// TODO: Enable once AgeMoea is ready

#[cfg(test)]
mod tests {
    use super::*;
    use crate::genetic::{Population, PopulationGenes};
    use crate::operators::survival::helpers::HyperPlaneNormalization;
    use crate::random::NoopRandomGenerator;
    use ndarray::{Array1, Array2, array};

    #[test]
    fn test_solve_intercepts() {
        // Front: [[1.0, 2.0],
        //         [3.0, 1.0]]
        // Ideal point = [1.0, 1.0]
        // Translated front = [[0.0, 1.0],
        //                     [2.0, 0.0]]
        // Expected extreme vectors: for obj0 -> row1 = [2.0, 0.0], for obj1 -> row0 = [0.0, 1.0]
        // Therefore, system: 2*a0 = 1, 1*a1 = 1 → a = [0.5, 1.0] → intercepts = [2.0, 1.0]
        let front: PopulationGenes = array![[1.0, 2.0], [3.0, 1.0]];
        let z_min = crate::helpers::extreme_points::get_ideal(&front);
        let translated = &front - &z_min;
        let normalizer = AgeMoeaHyperPlaneNormalization::new();
        let intercepts = normalizer.compute_hyperplane_intercepts(&translated);
        let expected = array![2.0, 1.0];
        assert_eq!(&intercepts, &expected);
    }

    #[test]
    fn test_solve_intercepts_no_solution() {
        // Front: [[1.0, 2.0],
        //         [1.0, 3.0]]
        // Ideal point = [1.0, 2.0]
        // Translated front = [[0.0, 0.0],
        //                     [0.0, 1.0]]
        // For objective 0, both values are 0.0.
        // For objective 1, extreme vector = row1 = [0.0, 1.0].
        // Fallback should return the nadir of translated front: [0.0, 1.0]
        let front: PopulationGenes = array![[1.0, 2.0], [1.0, 3.0]];
        let z_min = get_ideal(&front);
        let translated = front - z_min;
        let normalizer = AgeMoeaHyperPlaneNormalization::new();
        let intercepts = normalizer.compute_hyperplane_intercepts(&translated);
        let expected = array![0.0, 1.0];
        assert_eq!(&intercepts, &expected);
    }

    #[test]
    fn test_get_central_point_normalized_2d() {
        // Front:
        // [ [0.1, 0.9],
        //   [0.9, 0.1],
        //   [0.5, 0.5] ]
        // The central point should be [0.5, 0.5] (minimal perpendicular distance).
        let normalized: PopulationFitness = array![[0.1, 0.9], [0.9, 0.1], [0.5, 0.5]];
        let central = get_central_point_normalized(&normalized);
        let expected: Array1<f64> = array![0.5, 0.5];
        assert_eq!(&central, &expected);
    }

    #[test]
    fn test_c_paper_example() {
        // Using normalized front:
        // [ [1.0, 0.0],
        //   [0.0, 1.0],
        //   [0.5, 0.5] ]
        // Expected central point = [0.5, 0.5]
        let normalized: PopulationFitness = array![[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]];
        let central = get_central_point_normalized(&normalized);
        let expected: Array1<f64> = array![0.5, 0.5];
        assert_eq!(&central, &expected);

        // For p = 2 (Euclidean norm), the proximity of [0.5,0.5] is:
        let p = 2.0;
        let prox = proximity(&central.view(), p);
        let expected_prox = (0.5_f64.powi(2) + 0.5_f64.powi(2)).sqrt();
        assert!((prox - expected_prox).abs() < 1e-6);
    }

    #[test]
    fn test_proximity_2d() {
        // For solution [0.5, 0.5] with p = 2, expected L2 norm = sqrt(0.5^2 + 0.5^2)
        let solution: Array1<f64> = array![0.5, 0.5];
        let p = 2.0;
        let prox = proximity(&solution.view(), p);
        let expected = (0.5_f64.powi(2) + 0.5_f64.powi(2)).sqrt();
        assert!((prox - expected).abs() < 1e-6);
    }

    #[test]
    fn test_assign_survival_scores_p2() {
        // Front:
        // [ [1.0, 0.0],
        //   [0.5, 0.5],
        //   [0.0, 1.0] ]
        // With p = 2, extreme solutions (max per objective) are indices 0 and 2 (score = +∞).
        // The candidate at index 1: diversity = 0.5 + 0.5 = 1.0, proximity ~ 0.70710678, expected score ~ 1.41421356.
        let front: PopulationFitness = array![[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]];
        let p = 2.0;
        let scores = assign_survival_scores_first_front(&front, p);
        assert!(scores[0].is_infinite(), "Index 0 should be extreme (∞)");
        assert!(scores[2].is_infinite(), "Index 2 should be extreme (∞)");
        let candidate_score = scores[1];
        let expected_score = 1.0 / 0.70710678; // ~1.41421356
        assert!(
            (candidate_score - expected_score).abs() < 1e-6,
            "Score for index 1 should be ~1.41421356"
        );
    }

    #[test]
    fn test_assign_survival_scores_p1() {
        // Same front as above, with p = 1 (Manhattan distance).
        // For candidate at index 1:
        // Manhattan distances to extremes are both 1.0 → diversity = 2.0, proximity = 1.0, expected score = 2.0.
        let front: PopulationFitness = array![[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]];
        let p = 1.0;
        let scores = assign_survival_scores_first_front(&front, p);
        assert!(scores[0].is_infinite(), "Index 0 should be extreme (∞)");
        assert!(scores[2].is_infinite(), "Index 2 should be extreme (∞)");
        let candidate_score = scores[1];
        assert!(
            (candidate_score - 2.0).abs() < 1e-6,
            "Score for index 1 should be ~2.0"
        );
    }

    #[test]
    fn test_assign_survival_scores_multiple() {
        // Front:
        // [ [1.0, 0.0],
        //   [0.8, 0.2],
        //   [0.2, 0.8],
        //   [0.0, 1.0] ]
        // Extreme solutions (max in at least one objective) are indices 0 and 3.
        let front: PopulationFitness = array![[1.0, 0.0], [0.8, 0.2], [0.2, 0.8], [0.0, 1.0]];
        let p = 2.0;
        let scores = assign_survival_scores_first_front(&front, p);
        assert!(scores[0].is_infinite(), "Index 0 should be extreme (∞)");
        assert!(scores[3].is_infinite(), "Index 3 should be extreme (∞)");
        // Check that indices 1 and 2 have finite, positive scores.
        assert!(
            scores[1].is_finite() && scores[1] > 0.0,
            "Index 1 should have a positive finite score"
        );
        assert!(
            scores[2].is_finite() && scores[2] > 0.0,
            "Index 2 should have a positive finite score"
        );
    }

    #[test]
    fn test_operate_age_moea_survival() {
        // Front 1: 3 individuals and Front 2: 2 individuals
        let fitness: Array2<f64> =
            array![[1.0, 0.0], [0.5, 0.5], [0.0, 1.0], [0.8, 0.2], [0.2, 0.8]];
        // For simplicity, let genes be equal to fitness.
        let genes = fitness.clone();
        let population = Population::new(genes, fitness, None, None);
        // Set the desired number of survivors (e.g., 4).
        let n_survive = 4;

        let mut operator = AgeMoeaSurvival::new();
        let mut rng = NoopRandomGenerator::new();
        // create context (not used in the algorithm)
        let _context = AlgorithmContext::new(2, 5, 5, 2, 1, None, None, None);
        let survivors = operator.operate(population, n_survive, &mut rng, &_context);

        assert_eq!(
            survivors.len(),
            n_survive,
            "Expected {} survivors, got {}",
            n_survive,
            survivors.len()
        )
    }
}
