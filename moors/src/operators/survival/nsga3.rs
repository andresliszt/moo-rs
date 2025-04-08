use std::borrow::Cow;

use ndarray::{Array1, Array2, Axis, s};
use ndarray_stats::QuantileExt;

use crate::algorithms::AlgorithmContext;
use crate::genetic::{Fronts, Population, PopulationFitness};
use crate::helpers::extreme_points::get_ideal;
use crate::non_dominated_sorting::build_fronts;
use crate::operators::survival::helpers::HyperPlaneNormalization;
use crate::operators::{GeneticOperator, SurvivalOperator};
use crate::random::RandomGenerator;

/// Implementation of the survival operator for the NSGA3 algorithm presented in the paper
/// An Evolutionary Many-Objective Optimization Algorithm Using Reference-point Based Non-dominated Sorting Approach

#[derive(Clone, Debug)]
pub struct Nsga3ReferencePoints {
    points: Array2<f64>,
    are_aspirational: bool,
}

impl Nsga3ReferencePoints {
    pub fn new(points: Array2<f64>, are_aspirational: bool) -> Self {
        Self {
            points,
            are_aspirational,
        }
    }
}

struct Nsga3HyperPlaneNormalization;

impl Nsga3HyperPlaneNormalization {
    pub fn new() -> Self {
        Self
    }
}

impl HyperPlaneNormalization for Nsga3HyperPlaneNormalization {
    /// Computes the extreme points (z_max) from the translated population.
    /// For each objective j, constructs a weight vector:
    ///   w^j = [eps, ..., 1.0 (at position j), ..., eps],
    /// then selects the solution that minimizes ASF(s, w^j) using argmin from ndarray-stats.
    fn compute_extreme_points(&self, translated_population: &PopulationFitness) -> Array2<f64> {
        let n_objectives = translated_population.ncols();
        // Initialize an array to hold the extreme vectors; one per objective.
        let mut extreme_points = Array2::<f64>::zeros((n_objectives, n_objectives));

        // For each objective j, compute the corresponding extreme point.
        for j in 0..n_objectives {
            // Build the weight vector for objective j:
            // All elements are epsilon except for the j-th element which is 1.0.
            let mut weight = Array1::<f64>::from_elem(n_objectives, 1e-6);
            weight[j] = 1.0;

            // Compute the ASF value for each solution in the translated population.
            let asf_values: Vec<f64> = translated_population
                .outer_iter()
                .map(|solution| asf(&solution.to_owned(), &weight))
                .collect();
            let asf_array = Array1::from(asf_values);

            // Use argmin from ndarray-stats to get the index of the minimum ASF value.
            let best_idx = asf_array.argmin().unwrap();

            // The extreme point for objective j is the translated objective vector
            // of the solution that minimized ASF with weight vector w^j.
            let extreme = translated_population.row(best_idx);
            // Place this extreme vector in the j-th row of extreme_points.
            extreme_points.slice_mut(s![j, ..]).assign(&extreme);
        }
        extreme_points
    }
}

#[derive(Clone, Debug)]
pub struct Nsga3ReferencePointsSurvival {
    reference_points: Nsga3ReferencePoints, // Each row is a reference point
}

impl GeneticOperator for Nsga3ReferencePointsSurvival {
    fn name(&self) -> String {
        "Nsga3ReferencePointsSurvival".to_string()
    }
}

impl Nsga3ReferencePointsSurvival {
    pub fn new(reference_points: Nsga3ReferencePoints) -> Self {
        Self { reference_points }
    }
}

impl SurvivalOperator for Nsga3ReferencePointsSurvival {
    fn set_survival_score(
        &self,
        _fronts: &mut Fronts,
        _rng: &mut dyn RandomGenerator,
        _algorithm_context: &AlgorithmContext,
    ) {
        unimplemented!(
            "NSGA3 doesn't use survival score. It uses random tournament which doesn't depend on the score"
        )
    }

    fn operate(
        &mut self,
        population: Population,
        n_survive: usize,
        rng: &mut dyn RandomGenerator,
        _algorithm_context: &AlgorithmContext,
    ) -> Population {
        // Build fronts
        let mut fronts = build_fronts(population, n_survive);
        // Accumulator for the merged population.
        let mut survivors: Option<Population> = None;
        let mut n_survivors = 0;
        // Drain fronts to consume them and get an iterator of owned Population values.
        let drained = fronts.drain(..).enumerate();
        // Iterate over all fronts with enumerate (we no longer differentiate contexts).
        for (_i, front) in drained {
            // Save the length of the current front.
            let front_len = front.len();

            if n_survivors + front_len <= n_survive {
                // If the whole front fits, merge it with the accumulator.
                survivors = Some(match survivors {
                    Some(acc) => Population::merge(&acc, &front),
                    None => front,
                });
                n_survivors += front_len;
            } else {
                // Only part of this front is needed.
                let remaining = n_survive - n_survivors;
                if remaining > 0 {
                    // this is the S_t variable defined in the Algorithm 1 in the presented paper
                    // Determine the base population for the splitting front.
                    // If no accumulated population exists, use the current front as st.
                    let (st, n_complete) = match &survivors {
                        Some(acc) => (Population::merge(&acc, &front), acc.len()),
                        None => (front, 0),
                    };
                    let z_min = get_ideal(&st.fitness);
                    let translated_population = &st.fitness - &z_min;
                    let normalizer = Nsga3HyperPlaneNormalization::new();
                    let intercepts =
                        normalizer.compute_hyperplane_intercepts(&translated_population);
                    // This call is the normalize function (Algorithm 2)
                    let normalized_fitness = &translated_population / (&intercepts - &z_min);
                    // Now as the paper says, if the points are aspirational then normalize
                    // Use Cow so that when aspirational is false, we borrow the reference points.
                    let zr: Cow<Array2<f64>> = if self.reference_points.are_aspirational {
                        let normalized_zr =
                            (&self.reference_points.points - &z_min) / (&intercepts - &z_min);
                        Cow::Owned(normalized_zr)
                    } else {
                        Cow::Borrowed(&self.reference_points.points)
                    };
                    let (assignments, distances) = associate(&normalized_fitness, &zr);
                    // Compute niching count for every individual except in the splitting front
                    let survivors_assignments = &assignments[0..n_complete];
                    let mut niche_counts = compute_niche_counts(survivors_assignments, zr.nrows());

                    // Perform niching on the splitting front to select exactly `remaining`
                    // Prepare the indices of solutions that belong to the splitting front.
                    // These are the ones with index >= n_complete.
                    let mut splitting_indices: Vec<usize> = (n_complete..st.len()).collect();
                    let chosen_indices = niching(
                        remaining,
                        &mut niche_counts,
                        &assignments,
                        &distances,
                        &mut splitting_indices,
                        rng,
                    );
                    let selection_from_splitting_front = st.selected(&chosen_indices);
                    // Merge the partial selection with the accumulator.
                    survivors = Some(match survivors {
                        Some(acc) => Population::merge(&acc, &selection_from_splitting_front),
                        None => selection_from_splitting_front,
                    });
                }
                break;
            }
        }
        survivors.expect("Failed to build survivors")
    }
}

/// Calculates the Achievement Scalarizing Function (ASF) for a given solution `x`
/// (which represents the translated objective values f'_i(x)) and a weight vector `w`.
/// Any weight equal to zero is replaced by a small epsilon (1e-6) to avoid division by zero.
/// This is the equation (4) in the presented paper
fn asf(x: &Array1<f64>, w: &Array1<f64>) -> f64 {
    // Compute the element-wise ratio: f'_i(x) / w_i.
    let ratios = x / w;
    // The ASF is the maximum of these ratios.
    ratios.fold(std::f64::MIN, |acc, &val| acc.max(val))
}

/// Associates each solution s (each row in st) with the reference w (each row in zr)
/// that minimizes the perpendicular distance d⊥(s, w).
/// This is the algorithm (3) in the presented paper
fn associate(st_fitness: &PopulationFitness, zr: &Array2<f64>) -> (Vec<usize>, Vec<f64>) {
    let n = st_fitness.nrows();

    // 1. Compute squared norms for each solution: shape (n,)
    let norm_s_sq: Array1<f64> = st_fitness.outer_iter().map(|s| s.dot(&s)).collect();

    // 2. Compute squared norms for each reference: shape (m,)
    let norm_w_sq: Array1<f64> = zr.outer_iter().map(|w| w.dot(&w)).collect();

    // 3. Compute dot products between each s and each w: matrix A of shape (n, m)
    let dot = st_fitness.dot(&zr.t());

    // 4. Reshape norms for broadcasting:
    let norm_s_sq = norm_s_sq.insert_axis(Axis(1)); // shape (n, 1)
    let norm_w_sq = norm_w_sq.insert_axis(Axis(0)); // shape (1, m)

    // 5. Compute the squared dot products
    let dot_sq = dot.mapv(|x| x * x);

    // 6. Compute the squared perpendicular distance:
    // d2[i, j] = ||s_i||^2 - (dot[i,j]^2 / ||w_j||^2)
    let d2 = &norm_s_sq - &dot_sq / &norm_w_sq;

    // 8. For each solution (each row in d), find the index of the reference that minimizes the distance.
    let mut assignments = Vec::with_capacity(n);
    let mut distances = Vec::with_capacity(n);

    for row in d2.outer_iter() {
        let (min_idx, &min_val) = row
            .indexed_iter()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();
        assignments.push(min_idx);
        distances.push(min_val);
    }

    (assignments, distances)
}

/// Computes the niche counts for each reference point given only the assignments
/// from individuals that are not part of the splitting front.
///
/// # Arguments
/// * `assignments` - A slice of reference point indices assigned to individuals in complete fronts.
/// * `n_references` - The total number of reference points.
///
/// # Returns
/// A vector of niche counts (ρ_j) where each element corresponds to a reference point.
fn compute_niche_counts(assignments: &[usize], n_references: usize) -> Vec<usize> {
    let mut niche_counts = vec![0; n_references];
    for &assigned_ref in assignments.iter() {
        niche_counts[assigned_ref] += 1;
    }
    niche_counts
}

/// Implements the Niching procedure (algorithm 4 in the presented paper) for NSGA-III.
///
/// # Arguments
/// * `n_remaining` - The number of individuals left to assign.
/// * `niche_counts` - A mutable vector of niche counts (ρ_j) for each reference point.
/// * `assignments` - A vector where each element is the reference point index (π(s)) associated with a solution.
/// * `distances` - A vector where each element is the perpendicular distance d(s) for a solution.
/// * `available_refs` - A mutable vector of the available reference point indices (initially all indices in Zr).
/// * `splitting_front` - A mutable vector containing the indices of solutions in the splitting front (Fl).
///
/// # Returns
/// A vector of solution indices (Pt+1) that have been selected for the next population.
fn niching(
    mut n_remaining: usize,
    niche_counts: &mut Vec<usize>,
    assignments: &Vec<usize>,
    distances: &Vec<f64>,
    splitting_front: &mut Vec<usize>,
    rng: &mut dyn RandomGenerator,
) -> Vec<usize> {
    // Create available_refs inside the function based on the number of reference points.
    let mut available_refs: Vec<usize> = (0..niche_counts.len()).collect();
    let mut pt_next = Vec::new();

    // While there are still individuals to assign...
    while n_remaining > 0 {
        // If no reference points remain, break out.
        if available_refs.is_empty() {
            break;
        }

        // Step 3: Compute Jmin = { j in available_refs such that ρ_j is minimal }
        let min_count = available_refs
            .iter()
            .map(|&j| niche_counts[j])
            .min()
            .unwrap(); // safe because available_refs is not empty
        let jmin: Vec<usize> = available_refs
            .iter()
            .copied()
            .filter(|&j| niche_counts[j] == min_count)
            .collect();

        // Step 4: Select a random reference point from Jmin
        let j_bar = *rng.choose_usize(&jmin).unwrap();

        // Step 5: I_j_bar = { s in splitting_front such that assignments[s] == j_bar }
        let i_j_bar: Vec<usize> = splitting_front
            .iter()
            .copied()
            .filter(|&s| assignments[s] == j_bar)
            .collect();

        if !i_j_bar.is_empty() {
            // If there are candidate solutions for j_bar in the splitting front:
            let s_chosen = if niche_counts[j_bar] == 0 {
                // If the niche count for j_bar is zero, select the solution with minimum d(s)
                *i_j_bar
                    .iter()
                    .min_by(|&&s1, &&s2| distances[s1].partial_cmp(&distances[s2]).unwrap())
                    .unwrap()
            } else {
                // Otherwise, select a random solution from I_j_bar
                *rng.choose_usize(&i_j_bar).unwrap()
            };

            // Add the chosen solution to Pt+1
            pt_next.push(s_chosen);

            // Remove the chosen solution from the splitting front
            if let Some(pos) = splitting_front.iter().position(|&s| s == s_chosen) {
                splitting_front.remove(pos);
            }

            // Update the niche count for j_bar and decrement n_remaining
            niche_counts[j_bar] += 1;
            n_remaining -= 1;
        } else {
            // If I_j_bar is empty, remove j_bar from available_refs
            if let Some(pos) = available_refs.iter().position(|&j| j == j_bar) {
                available_refs.remove(pos);
            }
        }
    }

    pt_next
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::random::{RandomGenerator, TestDummyRng};
    use ndarray::array;
    use rand::RngCore;

    #[test]
    fn test_asf_with_identity_weights() {
        // Example translated objective vector.
        let x = array![0.2, 0.5, 0.3];

        // When using the identity matrix, the weight vectors are unit vectors.
        // For a 3-objective problem, these are:
        //   w1 = [1, 0, 0]
        //   w2 = [0, 1, 0]
        //   w3 = [0, 0, 1]
        //
        // Note: Since zeros in the weight vector are replaced with epsilon (1e-6),
        // the division will produce very large values in those components.

        // Test for w1 = [1, 0, 0]
        let w1 = array![1.0, 1e-6, 1e-6];
        let asf1 = asf(&x, &w1);
        // For w1, adjusted weights are [1.0, 1e-6, 1e-6] and the ratios are:
        //   0.2/1.0 = 0.2, 0.5/1e-6 = 500000, 0.3/1e-6 = 300000.
        // Thus, ASF should be 500000.
        assert_eq!(asf1, 500000.0);

        // Test for w2 = [0, 1, 0]
        let w2 = array![1e-6, 1.0, 1e-6];
        let asf2 = asf(&x, &w2);
        // For w2, adjusted weights are [1e-6, 1.0, 1e-6] and the ratios are:
        //   0.2/1e-6 = 200000, 0.5/1.0 = 0.5, 0.3/1e-6 = 300000.
        // Thus, ASF should be 300000.
        assert_eq!(asf2, 300000.0);

        // Test for w3 = [0, 0, 1]
        let w3 = array![1e-6, 1e-6, 1.0];
        let asf3 = asf(&x, &w3);
        // For w3, adjusted weights are [1e-6, 1e-6, 1.0] and the ratios are:
        //   0.2/1e-6 = 200000, 0.5/1e-6 = 500000, 0.3/1.0 = 0.3.
        // Thus, ASF should be 500000.
        assert_eq!(asf3, 500000.0);
    }

    // Test compute_extreme_points using a simple two-solution, two-objective case.
    #[test]
    fn test_compute_extreme_points() {
        // Two solutions:
        //   Solution A: [1.0, 10.0]
        //   Solution B: [10.0, 1.0]
        let pop = array![[1.0, 10.0], [10.0, 1.0]];
        let normalizer = Nsga3HyperPlaneNormalization::new();
        let extreme = normalizer.compute_extreme_points(&pop);

        // For objective 0, we expect the extreme point to be B: [10.0, 1.0]
        // For objective 1, we expect the extreme point to be A: [1.0, 10.0]
        let expected = array![[10.0, 1.0], [1.0, 10.0]];

        assert_eq!(
            extreme, expected,
            "Computed extreme points do not match expected values"
        );
    }

    // Test associate: simple case with two solutions and two reference points.
    #[test]
    fn test_associate() {
        // Two solutions:
        // A = [1, 10] and B = [10, 1]
        let st_fitness = array![[1.0, 10.0], [10.0, 1.0]];
        // Reference set: identity-like
        let zr = array![[1.0, 0.0], [0.0, 1.0]];
        let (assignments, distances) = associate(&st_fitness, &zr);
        // For solution A, expected assignment is 1; for B, expected is 0.
        assert_eq!(assignments, vec![1, 0]);
        // Expected distances are approximately 1.0 (allowing for floating-point error)
        for (i, d) in distances.iter().enumerate() {
            assert!(
                ((*d) - 1.0).abs() < 1e-5,
                "Solution {}: expected distance 1, got {}",
                i,
                d
            );
        }
    }

    // Test compute_niche_counts.
    #[test]
    fn test_compute_niche_counts() {
        // Given assignments, e.g., [0, 1, 0, 1, 1]
        let assignments = vec![0, 1, 0, 1, 1];
        let n_references = 2;
        let niche_counts = compute_niche_counts(&assignments, n_references);
        assert_eq!(niche_counts, vec![2, 3]);
    }

    struct FakeRandomGenerator {
        dummy: TestDummyRng,
    }

    impl FakeRandomGenerator {
        fn new() -> Self {
            Self {
                dummy: TestDummyRng,
            }
        }
    }

    impl RandomGenerator for FakeRandomGenerator {
        fn rng(&mut self) -> &mut dyn RngCore {
            &mut self.dummy
        }
        fn choose_usize<'a>(&mut self, vector: &'a [usize]) -> Option<&'a usize> {
            // Always choose the first element for deterministic behavior.
            vector.first()
        }
    }

    #[test]
    fn test_niching() {
        // Inputs for the niching function.
        let assignments = vec![0, 1, 0, 1]; // Each solution's assigned reference point.
        let distances = vec![10.0, 20.0, 30.0, 40.0]; // Perpendicular distances.
        let mut niche_counts = vec![0, 0]; // For two reference points.
        let mut splitting_front = vec![0, 1, 2, 3]; // Indices of solutions in the splitting front.
        let n_remaining = 2; // We want to select 2 individuals.

        let mut dummy_rng = FakeRandomGenerator::new();

        let chosen = niching(
            n_remaining,
            &mut niche_counts,
            &assignments,
            &distances,
            &mut splitting_front,
            &mut dummy_rng,
        );
        // Expected: first iteration picks index 0, second iteration picks index 1.
        assert_eq!(chosen, vec![0, 1]);
    }

    /// Test the operate method when the first (and only) front is larger than n_survive.
    /// In this case splitting occurs with no previously accumulated survivors.
    #[test]
    fn test_operate_split_first_front_content() {
        // Create one front with 5 individuals having distinct fitness values.
        let fitness = array![[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0], [5.0, 5.0]];
        // For simplicity, genes = fitness
        let population = Population::new(fitness.clone(), fitness.clone(), None, None);

        // Use a simple reference points matrix: for 2 objectives we use the 2x2 identity.
        let reference_points = Nsga3ReferencePoints::new(Array2::eye(2), false);
        let mut survival_operator = Nsga3ReferencePointsSurvival::new(reference_points);
        let mut rng = FakeRandomGenerator::new();
        // create context (not used in the algorithm)
        let _context = AlgorithmContext::new(2, 5, 5, 2, 1, None, None, None);
        // Set n_survive to 3 so that splitting must occur on the single front.
        let survivors = survival_operator.operate(population, 3, &mut rng, &_context);
        assert_eq!(survivors.len(), 3, "Final survivors count should be 3");

        // Verify that each selected individual comes from the original front.
        // (Since there is only one front, every survivor's fitness should match one of the rows in the original matrix.)
        for survivor in survivors.fitness.outer_iter() {
            let mut found = false;
            for orig in fitness.outer_iter() {
                if survivor == orig {
                    found = true;
                    break;
                }
            }
            assert!(
                found,
                "Survivor row {:?} not found in original front",
                survivor
            );
        }
    }

    /// Test the operate method when multiple fronts are provided and splitting occurs on a later front.
    /// In this scenario the complete first front is preserved and a part of the second front is selected.
    #[test]
    fn test_operate_split_later_front_content() {
        // Front 1: 3 individuals (1.x fitness). Front 2: 4 individuals with higher fitness values (2.x fitness)
        let fitness = array![
            [1.0, 1.0],
            [1.1, 1.1],
            [1.2, 1.2],
            [2.0, 2.0],
            [2.1, 2.1],
            [2.2, 2.2],
            [2.3, 2.3]
        ];
        // For simplicity, genes = fitness
        let population = Population::new(fitness.clone(), fitness.clone(), None, None);
        // Use the identity as reference points for 2 objectives.
        let reference_points = Nsga3ReferencePoints::new(Array2::eye(2), false);
        let mut survival_operator = Nsga3ReferencePointsSurvival::new(reference_points);
        let mut rng = FakeRandomGenerator::new();
        // create context (not used in the algorithm)
        let _context = AlgorithmContext::new(2, 7, 5, 2, 1, None, None, None);
        // Total individuals if merged completely would be 7.
        // Set n_survive to 5 so that the first front (3 individuals) is completely taken
        // and 2 individuals are selected from the second front.
        let survivors = survival_operator.operate(population, 5, &mut rng, &_context);
        assert_eq!(survivors.len(), 5, "Final survivors count should be 5");

        // Check that the survivors include all individuals from the first front.
        // Since Population::merge concatenates rows, we expect that the first 3 survivors
        // come from front1.
        let survivors_fitness = survivors.fitness;
        for i in 0..3 {
            // Compare each row of survivors with the corresponding row in fitness1.
            let survivor_row = survivors_fitness.slice(s![i, ..]);
            let expected_row = fitness.slice(s![i, ..]);
            assert!(
                survivor_row.eq(&expected_row),
                "Survivor row {} does not match expected front1 row: got {:?}, expected {:?}",
                i,
                survivor_row,
                expected_row
            );
        }

        // For the remaining survivors (from the splitting front), verify that their fitness values
        // come from front2. Since the niching procedure selects from the merged front,
        // these rows should appear in fitness2.
        for i in 3..5 {
            let survivor_row = survivors_fitness.slice(s![i, ..]);
            let mut found = false;
            for orig in fitness.outer_iter() {
                if survivor_row.eq(&orig) {
                    found = true;
                    break;
                }
            }
            assert!(
                found,
                "Survivor row {} from splitting front not found in front2. Row: {:?}",
                i, survivor_row
            );
        }
    }
}
