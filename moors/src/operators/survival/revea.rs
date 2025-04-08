use std::collections::HashMap;

use ndarray::{Array1, Array2};

use crate::algorithms::AlgorithmContext;
use crate::genetic::Population;
use crate::helpers::extreme_points::{get_ideal, get_nadir};
use crate::helpers::linalg::{faer_dot_and_norms, faer_dot_from_array};
use crate::operators::{GeneticOperator, SurvivalOperator};
use crate::random::RandomGenerator;

/// Implementation of the survival operator for the REVEA algorithm presented in the paper
/// A Reference Vector Guided Evolutionary Algorithm for Many-objective Optimization

#[derive(Clone, Debug)]
pub struct ReveaReferencePointsSurvival {
    reference_points: Array2<f64>,
    initial_reference_points: Array2<f64>,
    alpha: f64,
    frequency: f64,
}

impl GeneticOperator for ReveaReferencePointsSurvival {
    fn name(&self) -> String {
        "ReveaReferencePointsSurvival".to_string()
    }
}

impl ReveaReferencePointsSurvival {
    pub fn new(reference_points: Array2<f64>, alpha: f64, frequency: f64) -> Self {
        let initial_reference_points = reference_points.clone();
        Self {
            reference_points,
            initial_reference_points,
            alpha,
            frequency,
        }
    }
}

impl SurvivalOperator for ReveaReferencePointsSurvival {
    fn set_survival_score(
        &self,
        _fronts: &mut crate::genetic::Fronts,
        _rng: &mut dyn RandomGenerator,
        _algorithm_context: &AlgorithmContext,
    ) {
        unimplemented!("REVEA doesn't use fronts")
    }

    fn operate(
        &mut self,
        population: Population,
        _n_survive: usize,
        _rng: &mut dyn RandomGenerator,
        algorithm_context: &AlgorithmContext,
    ) -> Population {
        let z_min = get_ideal(&population.fitness);
        let z_max = get_nadir(&population.fitness);
        let translated = &population.fitness - &z_min;
        let (translated_fitness_norm, reference_norm, faer_dot) =
            faer_dot_and_norms(&translated, &self.reference_points);
        // this is the cos(theta_ij) matrix, equation (6) in the presented paper
        let cosine_distances: faer::Mat<f64> =
            cross_cosine_distances(&translated_fitness_norm, reference_norm, faer_dot);
        // this is P_{t,k} from equation (7) in the presented paper
        let sub_populations = compute_sub_population(&cosine_distances);
        // this is the dot product matrix between reference points
        let gamma = compute_gamma(&self.reference_points);
        // this is the angle penalized distance, equation (9)
        let apd_matrix = compute_angle_penalized_distances(
            translated_fitness_norm,
            cosine_distances,
            gamma,
            algorithm_context.n_objectives,
            algorithm_context.current_iteration,
            algorithm_context.n_iterations,
            self.alpha,
        );
        // Elitism Selection
        let mut selected_indices = Vec::new();
        // For each reference vector group, select the individual with the smallest APD.
        for (ref_index, group) in sub_populations.into_iter().enumerate() {
            if !group.is_empty() {
                let best_index = group
                    .into_iter()
                    .min_by(|&i1, &i2| {
                        apd_matrix[(i1, ref_index)]
                            .partial_cmp(&apd_matrix[(i2, ref_index)])
                            .unwrap()
                    })
                    .unwrap();
                selected_indices.push(best_index);
            }
        }
        // Update reference points if needed
        if (algorithm_context.current_iteration as f64 / algorithm_context.n_iterations as f64)
            % self.frequency
            == 0.0
        {
            let new_reference_points =
                update_reference_vectors(&z_min, &z_max, &self.initial_reference_points);
            self.reference_points = new_reference_points;
        }
        population.selected(&selected_indices)
    }
}

fn cross_cosine_distances(
    fitness_norm: &faer::Mat<f64>,
    reference_norm: faer::Mat<f64>,
    faer_dot: faer::Mat<f64>,
) -> faer::Mat<f64> {
    let n = fitness_norm.nrows();
    let m = reference_norm.nrows();
    // cosine(Oi, Oj) = (x_i dot y_j) / (||x_i|| * ||y_j||)
    let faer_cosine_dist: faer::Mat<f64> = faer::Mat::from_fn(n, m, |i, j| {
        let norm_fitness = fitness_norm.get(i, 0).sqrt();
        let norm_reference = reference_norm.get(j, 0).sqrt();
        if norm_fitness == 0.0 || norm_reference == 0.0 {
            0.0
        } else {
            faer_dot.get(i, j) / (norm_fitness * norm_reference)
        }
    });
    faer_cosine_dist
}

fn compute_sub_population(cosine_distances: &faer::Mat<f64>) -> Vec<Vec<usize>> {
    let nrows = cosine_distances.nrows();
    let ncols = cosine_distances.ncols();
    // Compute the argmax for each row. For each row (individual) we iterate over the columns
    // and pick the column index with the highest cosine similarity.
    let argmax_indices: Vec<usize> = (0..nrows)
        .map(|i| {
            (0..ncols)
                .map(|j| (j, cosine_distances[(i, j)]))
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .map(|(j, _)| j)
                .expect("Row should not be empty")
        })
        .collect();

    // Assuming argmax_indices is a Vec<usize> where each element represents
    // the best reference index for an individual.
    let groups: HashMap<usize, Vec<usize>> =
        argmax_indices
            .into_iter()
            .enumerate()
            .fold(HashMap::new(), |mut map, (i, ref_index)| {
                map.entry(ref_index).or_insert_with(Vec::new).push(i);
                map
            });

    // Build the final subpopulations vector for each reference (assuming ncols is the number of references)
    let sub_populations: Vec<Vec<usize>> = (0..ncols)
        .map(|ref_index| groups.get(&ref_index).cloned().unwrap_or_default())
        .collect();
    sub_populations
}

/// Computes gamma for each reference vector from a precomputed inner product matrix.
/// For each reference vector (indexed by j), it computes:
///
///   gamma_{t, j} = min { inner_products(i, j) for all i != j }
///  This is the equation (10) in the presented paper
fn compute_gamma(reference_points: &Array2<f64>) -> Vec<f64> {
    let inner_products = faer_dot_from_array(&reference_points, &reference_points);
    let n = inner_products.nrows();
    (0..n)
        .map(|j| {
            (0..n)
                .filter(|&i| i != j) // Exclude the diagonal element
                .map(|i| inner_products[(i, j)])
                .min_by(|a, b| a.partial_cmp(b).unwrap())
                .expect("Row should not be empty")
        })
        .collect()
}

/// This is the equation (9) in the presented paper
fn compute_angle_penalized_distances(
    fitness_norm: faer::Mat<f64>,
    cosine_matrix: faer::Mat<f64>,
    gamma: Vec<f64>,
    n_objectives: usize,
    current_iteration: usize,
    max_iterations: usize,
    alpha: f64,
) -> faer::Mat<f64> {
    let n = cosine_matrix.nrows();
    let m = cosine_matrix.ncols();
    // Common Factor: M · (t/tmax)^α
    let factor =
        (n_objectives as f64) * (current_iteration as f64 / max_iterations as f64).powf(alpha);
    faer::Mat::from_fn(n, m, |i, j| {
        let gamma_val = if gamma[j] == 0.0 { 1e-64 } else { gamma[j] };
        (1.0 + factor * (cosine_matrix.get(i, j).acos() / gamma_val)) * fitness_norm.get(i, 0)
    })
}

/// Updates the reference vector set for the next generation.
///
/// # Parameters
/// - `z_min`: The minimum objective values for the current population
/// - `z_max`: The maximum objective values for the current population
/// - `current_reference_points`: The current reference vector set Vₜ (dimensions: N x M).

///
/// # Returns
/// A new reference vector set Vₜ₊₁ for the next generation.
pub fn update_reference_vectors(
    z_min: &Array1<f64>,
    z_max: &Array1<f64>,
    initial_reference_points: &Array2<f64>,
) -> Array2<f64> {
    // Compute the range vector (z_max - z_min)
    let range = z_max - z_min;

    let n = initial_reference_points.nrows();
    let m = initial_reference_points.ncols();
    let mut new_reference_points = Array2::<f64>::zeros((n, m));

    // For each reference vector, update according to equation (11):
    // vₜ₊₁,ᵢ = (v₀,ᵢ ∘ (z_max - z_min)) / || v₀,ᵢ ∘ (z_max - z_min) ||
    // Using ndarray's native elementwise multiplication operator for the Hadamard product.
    for i in 0..n {
        let v0_i = initial_reference_points.row(i);
        let hadamard = &v0_i * &range;
        let norm = hadamard.iter().map(|&x| x * x).sum::<f64>().sqrt();
        if norm.abs() > 0.0 {
            let updated = hadamard.mapv(|x| x / norm);
            new_reference_points.row_mut(i).assign(&updated);
        } else {
            new_reference_points.row_mut(i).assign(&hadamard);
        }
    }
    new_reference_points
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::mat;
    use ndarray::array;
    //use std::f64::consts::FRAC_PI_4; // @oliveira-sh: this is not being used

    #[test]
    fn test_cross_cosine_distances() {
        // Define fitness_norm and reference_norm as unit vectors so that
        // the cosine distance is simply the value of the dot product.
        let fitness_norm = mat![[1.0], [1.0]]; // 2 x 1 matrix
        let reference_norm = mat![[1.0], [1.0], [1.0]]; // 3 x 1 matrix
        let faer_dot = mat![[0.5, 0.3, 0.7], [0.2, 0.9, 0.1]]; // 2 x 3 matrix

        let result = cross_cosine_distances(&fitness_norm, reference_norm, faer_dot);
        // Since the norms are 1, the expected result is the same as the dot matrix.
        let expected = mat![[0.5, 0.3, 0.7], [0.2, 0.9, 0.1]];

        for i in 0..result.nrows() {
            for j in 0..result.ncols() {
                let r: &f64 = result.get(i, j);
                let e: &f64 = expected.get(i, j);
                let diff: f64 = (r - e).abs();
                assert!(diff < 1e-10, "Difference at ({}, {}) is {}", i, j, diff);
            }
        }
    }

    #[test]
    fn test_compute_sub_population() {
        // Create a 4x3 matrix of cosine distances.
        //
        // Row 0: [0.1, 0.3, 0.2] -> argmax is column 1
        // Row 1: [0.5, 0.4, 0.1] -> argmax is column 0
        // Row 2: [0.2, 0.2, 0.6] -> argmax is column 2
        // Row 3: [0.3, 0.1, 0.4] -> argmax is column 2
        //
        // Expected subpopulations:
        // Reference 0: [1]
        // Reference 1: [0]
        // Reference 2: [2, 3]
        let cosine_matrix = mat![
            [0.1, 0.3, 0.2],
            [0.5, 0.4, 0.1],
            [0.2, 0.2, 0.6],
            [0.3, 0.1, 0.4]
        ];

        let subpopulations = compute_sub_population(&cosine_matrix);

        let expected = vec![
            vec![1],    // Reference 0 gets row 1
            vec![0],    // Reference 1 gets row 0
            vec![2, 3], // Reference 2 gets rows 2 and 3
        ];
        assert_eq!(subpopulations, expected);
    }

    #[test]
    fn test_compute_gamma() {
        // Create a set of 3 reference points in 2D.
        // Let the reference points be:
        // [1, 0]
        // [0, 1]
        // [1, 1]
        // Then, the inner product matrix is:
        // [1, 0, 1]
        // [0, 1, 1]
        // [1, 1, 2]
        // For each column j, gamma is the minimum inner product for i != j:
        // gamma_0 = min(0, 1) = 0, gamma_1 = min(0, 1) = 0, gamma_2 = min(1, 1) = 1.
        let ref_points = array![[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let gamma = compute_gamma(&ref_points);
        let expected = vec![0.0, 0.0, 1.0];
        assert_eq!(gamma.len(), expected.len());
        for (g, e) in gamma.iter().zip(expected.iter()) {
            let diff = (g - e).abs();
            assert!(diff < 1e-10, "Expected {}, got {}", e, g);
        }
    }

    #[test]
    fn test_compute_angle_penalized_distances() {
        use std::f64::consts::FRAC_PI_4;
        // Set up the input matrices:
        // cosine_matrix: 1 x 2 matrix: [1.0, 0.0]
        // fitness_norm: 1 x 1 matrix: [1.0]
        let cosine_matrix = mat![[1.0, 0.0]]; // 1 x 2 matrix
        let fitness_norm = mat![[1.0]]; // 1 x 1 matrix
        // gamma vector: [1.0, 2.0]
        let gamma = vec![1.0, 2.0];
        // Define parameters:
        let n_objectives = 2;
        let current_iteration = 5;
        let max_iterations = 10;
        let alpha = 1.0;

        // Call the updated function.
        let result = compute_angle_penalized_distances(
            fitness_norm,
            cosine_matrix,
            gamma,
            n_objectives,
            current_iteration,
            max_iterations,
            alpha,
        );

        // Expected results:
        // For column 0: (1.0 + 1.0*(0/1.0)) * 1.0 = 1.0.
        // For column 1: (1.0 + 1.0*((π/2)/2.0)) * 1.0 = 1.0 + π/4.
        let expected_first = 1.0;
        let expected_second = 1.0 + FRAC_PI_4; // π/4 ≈ 0.785398163

        assert_eq!(result.nrows(), 1);
        assert_eq!(result.ncols(), 2);

        let diff0: f64 = (result.get(0, 0) - expected_first).abs();
        let diff1: f64 = (result.get(0, 1) - expected_second).abs();

        assert!(
            diff0 < 1e-10,
            "Expected {}, got {}",
            expected_first,
            result.get(0, 0)
        );
        assert!(
            diff1 < 1e-10,
            "Expected {}, got {}",
            expected_second,
            result.get(0, 1)
        );
    }
}
