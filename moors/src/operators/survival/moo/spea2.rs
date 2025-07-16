use std::cmp::Ordering;

use ndarray::{Array1, Array2, Axis};

use crate::{
    genetic::{D12, PopulationMOO},
    helpers::linalg::cross_euclidean_distances_as_array,
    non_dominated_sorting::fast_non_dominated_sorting,
    operators::survival::SurvivalOperator,
    random::RandomGenerator,
};

#[derive(Debug, Clone)]
pub struct Spea2KnnSurvival;

impl Spea2KnnSurvival {
    pub fn new() -> Self {
        Self {}
    }
}

impl Default for Spea2KnnSurvival {
    fn default() -> Self {
        Self::new()
    }
}

impl SurvivalOperator for Spea2KnnSurvival {
    type FDim = ndarray::Ix2;

    fn operate<ConstrDim>(
        &mut self,
        population: PopulationMOO<ConstrDim>,
        num_survive: usize,
        _rng: &mut impl RandomGenerator,
    ) -> PopulationMOO<ConstrDim>
    where
        ConstrDim: D12,
    {
        // Compute raw fitness F(i) = R(i) + D(i)
        let k = population.len().isqrt();
        let distance_matrix =
            cross_euclidean_distances_as_array(&population.fitness, &population.fitness);
        let density = compute_density(&distance_matrix, k);
        let domination_indices = compute_domination_indices(&population.fitness);
        // raw_fitness[i] = domination_indices_f[i] + density[i]
        let raw_fitness: Array1<f64> = &domination_indices + &density;
        // Next step is to check out if the |{i: S(i) < 1}| = {i: raw_fitness[i] < 1}| <= num_survive
        let mut s: Vec<usize> = raw_fitness
            .iter()
            .enumerate()
            .filter_map(|(i, &f)| if f < 1.0 { Some(i) } else { None })
            .collect();
        // Branch based on S.len() vs num_survive
        match s.len().cmp(&num_survive) {
            Ordering::Equal => {
                // Case A: exactly the right number — nothing to do
                // `s` already has the survivors
            }
            Ordering::Less => {
                // Case B: too few non-dominated solutions — fill with best of the rest
                let needed = num_survive - s.len();
                let dominated_indices = select_dominated(&raw_fitness, needed);
                s.extend(dominated_indices);
            }
            Ordering::Greater => {
                // Case C: too many non-dominated
                s = select_by_nearest_neighbor(&distance_matrix, num_survive);
            }
        }
        let mut survivors = population.selected(&s);
        // Assign the score
        let selected_scores: Array1<f64> = raw_fitness.select(Axis(0), &s);
        // ignore Result
        survivors.set_survival_score(selected_scores);
        survivors
    }
}

/// Compute density D(i) = 1 / (σᵢᵏ + 2) for each individual i,
/// where σᵢᵏ is the k-th smallest distance in row i (including the zero at index 0).
///
/// # Arguments
/// * `distance_matrix` – an n×n Array2<f64> of pairwise distances (diagonal == 0).
/// * `k` – neighbor index: 0 gives σ⁰=0 (self), so to pick the 1st real neighbor use k=1, etc.
pub fn compute_density(distance_matrix: &Array2<f64>, k: usize) -> Array1<f64> {
    let n = distance_matrix.nrows();

    let mut densities = Array1::<f64>::zeros(n);

    for i in 0..n {
        // TODO: Can we avoid cloning or passing to a Vec?
        // Ivestigate this ndarray method: select_nth_unstable_by
        let mut dists: Vec<f64> = distance_matrix.row(i).iter().cloned().collect();
        dists.sort_by(|a, b| a.partial_cmp(b).unwrap());
        // σᵢᵏ is at index k. Note that in the position 0
        // we have d = 0, because in the sorted it's distance between
        // the individual with itself. So we use k instead of k-1
        let sigma_k = dists[k];
        // Compute density
        densities[i] = 1.0 / (sigma_k + 2.0);
    }

    densities
}

/// Compute the Pareto-domination index for each individual in the population.
///
/// Given an (N×M) fitness matrix, returns a length-N `Array1<usize>` where the
/// i-th entry is the zero-based non-domination rank of individual i:
/// - 0 for those not dominated by anyone,
/// - 1 for those only dominated by rank-0 individuals,
/// - 2 for those dominated by rank-0 and rank-1, and so on.
///
/// Internally this calls `fast_non_dominated_sorting(..., N)` to partition
/// all individuals into successive non-dominated sets, then assigns each
/// individual the index of the set it belongs to.
pub fn compute_domination_indices(population_fitness: &Array2<f64>) -> Array1<f64> {
    let n = population_fitness.nrows();
    let ranks = fast_non_dominated_sorting(population_fitness, n);

    let mut domination_indices = Array1::<f64>::zeros(n);

    for (rank, group) in ranks.into_iter().enumerate() {
        for &i in &group {
            domination_indices[i] = rank as f64;
        }
    }

    domination_indices
}

/// Select the top `r` dominated individuals (those with `raw_fitness >= 1.0`)
/// by ascending raw fitness (i.e. smallest F(i) first).
///
/// # Arguments
///
/// * `raw_fitness` – an Array1<f64> of length N containing each individual’s F(i).
/// * `r` – the number of dominated individuals to return.
///
/// # Returns
///
/// A `Vec<usize>` of length `r`, containing the indices of the dominated
/// individuals with the smallest raw_fitness values, in ascending order.
pub fn select_dominated(raw_fitness: &Array1<f64>, r: usize) -> Vec<usize> {
    // Collect all (index, fitness) for dominated individuals
    let mut dominated: Vec<(usize, f64)> = raw_fitness
        .iter()
        .enumerate()
        .filter_map(|(i, &f)| if f >= 1.0 { Some((i, f)) } else { None })
        .collect();

    // Sort by fitness ascending (best dominated first)
    dominated.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

    // Take the first `r` indices
    dominated.into_iter().take(r).map(|(idx, _)| idx).collect()
}

/// Selects the `r` most isolated individuals from an n×n distance matrix,
/// by looking at each row’s *nearest neighbor* distance (excluding self).
///
/// # Arguments
///
/// * `distance_matrix` – an n×n square matrix of pairwise distances (diagonal = 0).
/// * `r` – number of survivors to pick (0 ≤ r ≤ n).
///
/// # Returns
///
/// A `Vec<usize>` of length `r` containing the row indices of the selected survivors.
pub fn select_by_nearest_neighbor(distance_matrix: &Array2<f64>, r: usize) -> Vec<usize> {
    let n = distance_matrix.nrows();
    // 1) Compute each row’s nearest‐neighbor distance (excluding the diagonal entry)
    //    Store (index, nearest_dist) pairs.
    let mut nearest: Vec<(usize, f64)> = Vec::with_capacity(n);
    for i in 0..n {
        // Find the minimum over all j != i
        let min_dist = distance_matrix
            .row(i)
            .iter()
            .enumerate()
            .filter_map(|(j, &d)| if j != i { Some(d) } else { None })
            .fold(f64::INFINITY, f64::min);
        nearest.push((i, min_dist));
    }

    // 2) Sort by nearest_dist **descending**:
    //    individuals with larger nearest‐neighbor distance are more isolated.
    nearest.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

    // 3) Take the top `r` indices
    nearest.into_iter().take(r).map(|(idx, _)| idx).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    use crate::random::NoopRandomGenerator;

    #[test]
    fn test_compute_density() {
        // 4×4 distance matrix (symmetric, diagonal = 0), columns not sorted:
        //
        //    0   5   2   8
        //    5   0   9   1
        //    2   9   0   4
        //    8   1   4   0
        //
        // For n = 4 → k = floor(sqrt(4)) = 2
        // Row 0 sorted distances: [0,2,5,8] → σ₂ = 5 → D₀ = 1/(5 + 2) = 1/7
        // Row 1 sorted distances: [0,1,5,9] → σ₂ = 5 → D₁ = 1/7
        // Row 2 sorted distances: [0,2,4,9] → σ₂ = 4 → D₂ = 1/6
        // Row 3 sorted distances: [0,1,4,8] → σ₂ = 4 → D₃ = 1/6

        let dm = array![
            [0.0, 5.0, 2.0, 8.0],
            [5.0, 0.0, 9.0, 1.0],
            [2.0, 9.0, 0.0, 4.0],
            [8.0, 1.0, 4.0, 0.0],
        ];

        let densities = compute_density(&dm, 2);
        let expected = [1.0 / 7.0, 1.0 / 7.0, 1.0 / 6.0, 1.0 / 6.0];

        for i in 0..4 {
            assert!(
                (densities[i] - expected[i]).abs() < 1e-12,
                "density[{}] = {}, but expected {}",
                i,
                densities[i],
                expected[i]
            );
        }
    }

    #[test]
    fn test_select_dominated_mixed() {
        // raw_fitness contains values <1.0 and >=1.0
        // only those >=1.0 should be considered dominated
        // raw = [0.5, 1.0, 2.0, 3.0, 1.5]
        // dominated candidates = (1,1.0), (2,2.0), (3,3.0), (4,1.5)
        // sorted ascending by fitness: (1,1.0), (4,1.5), (2,2.0), (3,3.0)
        // take r=3 → indices [1, 4, 2]
        let raw: Array1<f64> = array![0.5, 1.0, 2.0, 3.0, 1.5];
        let picked = select_dominated(&raw, 3);
        assert_eq!(picked, vec![1, 4, 2]);
    }

    #[test]
    fn test_select_dominated_all() {
        // raw_fitness all >1.0
        // raw = [1.1, 2.2, 3.3]
        // sorted ascending: (0,1.1), (1,2.2), (2,3.3)
        // take r=2 → indices [0, 1]
        let raw: Array1<f64> = array![1.1, 2.2, 3.3];
        let picked = select_dominated(&raw, 2);
        assert_eq!(picked, vec![0, 1]);
    }

    #[test]
    fn test_chain_dominance() {
        // A simple chain: A dominates B, B dominates C.
        // Fitness vectors: A = [1,1], B = [2,2], C = [3,3]
        // Expected ranks: A → 0, B → 1, C → 2
        let fitness = array![[1.0, 1.0], [2.0, 2.0], [3.0, 3.0],];
        let indices = compute_domination_indices(&fitness);
        assert_eq!(indices.len(), 3);
        assert_eq!(indices, array![0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_no_dominance_all_zero() {
        // All are non-dominated pairwise: rank 0 for everyone
        // Fitness: [1,4], [2,3], [3,2], [4,1]
        let fitness = array![[1.0, 4.0], [2.0, 3.0], [3.0, 2.0], [4.0, 1.0],];
        let indices = compute_domination_indices(&fitness);
        assert_eq!(indices.len(), 4);
        assert_eq!(indices, array![0.0, 0.0, 0.0, 0.0]);
    }

    /// Helper: build a Population from raw fitness only.
    /// We use a dummy 1-column gene matrix and no constraints nor rank.
    fn make_population(fitness: Array2<f64>) -> PopulationMOO<ndarray::Ix2> {
        let n = fitness.nrows();
        // 1 variable per individual, value zero—genes are never used by survival
        let genes = Array2::<f64>::zeros((n, 1));
        PopulationMOO::new_unconstrained(genes, fitness)
    }

    #[test]
    fn test_fills_when_underflow() {
        // Define objective fitness = [0.5, 1.2, 1.5]
        let fit = array![[0.5], [1.2], [1.5]];
        let pop = make_population(fit.clone());

        // Manually compute the **raw fitness** using squared distances:
        //
        // Note that in our code we use the squared distance, we don't take square root
        // This is a simple test case for a single objecive, where squared distance might
        // not make sense

        //    - n = 3  ⇒ k = floor(sqrt(3)) = 1
        //    - Squared‐distance matrix:
        //        [0,      0.49,   1.00]
        //        [0.49,   0.00,   0.09]
        //        [1.00,   0.09,   0.00]
        //
        //      Row0 sorted: [0, 0.49, 1.00] → σ₁ = 0.49 → D₀ = 1/(0.49+2) ≈ 0.4016064
        //      Row1 sorted: [0, 0.09, 0.49] → σ₁ = 0.09 → D₁ = 1/(0.09+2) ≈ 0.4784689
        //      Row2 sorted: [0, 0.09, 1.00] → σ₁ = 0.09 → D₂ = 1/(0.09+2) ≈ 0.4784689
        //
        //    - Pareto‐ranks R(i):
        //        0.5 dominates {1.2,1.5} → R₀=0
        //        1.2 dominates {1.5}     → R₁=1
        //        1.5 dominated twice     → R₂=2
        //
        //    - So raw_fitness = R + D:
        //        F₀ = 0 + 0.4016064  ≈ 0.4016064
        //        F₁ = 1 + 0.4784689  ≈ 1.4784689
        //        F₂ = 2 + 0.4784689  ≈ 2.4784689
        let expected_raw = [0.4016064, 1.4784689, 2.4784689];

        // Run operate with capacity = 2
        let mut rng = NoopRandomGenerator::new();
        let survivors = Spea2KnnSurvival::new().operate(pop, 2, &mut rng);

        // Extract survivors’ survival_score fields
        let scores: Vec<f64> = survivors
            .survival_score
            .as_ref()
            .expect("survival_score must be set")
            .to_vec();

        // Underflow picks indices 0 and 1, so we expect
        // [expected_raw[0], expected_raw[1]]
        assert_eq!(scores.len(), 2);
        assert!((scores[0] - expected_raw[0]).abs() < 1e-6,);
        assert!((scores[1] - expected_raw[1]).abs() < 1e-6);
    }

    #[test]
    fn test_overflow_keeps_first_two_when_all_tie() {
        // Four pair‑wise non‑dominated points:
        //   #0 (0,3)   #1 (1,2)   #2 (2,1)   #3 (3,0)
        let fit = array![[0.0, 3.0], [1.0, 2.0], [2.0, 1.0], [3.0, 0.0],];
        let pop = make_population(fit.clone());

        // Capacity = 2 → overflow branch
        let mut rng = NoopRandomGenerator::new();
        let survivors = Spea2KnnSurvival::new().operate(pop, 2, &mut rng);

        // Expect exactly two survivors
        assert_eq!(survivors.len(), 2);

        // They must correspond to indices 0 and 1 (original order)
        assert_eq!(survivors.get(0).fitness.to_vec(), vec![0.0, 3.0]);
        assert_eq!(survivors.get(1).fitness.to_vec(), vec![1.0, 2.0]);

        // Expected raw‑fitness (R+ D with squared distances, k=2):
        //    row0 σ₂ = 8  → D0 = 1/10 = 0.1
        //    row1 σ₂ = 2  → D1 = 1/4  = 0.25
        let expected = [0.1, 0.25];

        let scores = survivors
            .survival_score
            .as_ref()
            .expect("survival_score must be set");

        // Order is deterministic (indices 0 then 1)
        assert_eq!(scores.len(), 2);
        assert!((scores[0] - expected[0]).abs() < 1e-6);
        assert!((scores[1] - expected[1]).abs() < 1e-6);
    }

    #[test]
    fn all_survive_when_capacity_equals_population() {
        // Same four pair‑wise non‑dominated points as before
        let fit = array![
            [0.0, 3.0], // #0
            [1.0, 2.0], // #1
            [2.0, 1.0], // #2
            [3.0, 0.0], // #3
        ];
        let pop = make_population(fit.clone());

        // Capacity exactly equals population size (4) → no truncation
        let mut rng = NoopRandomGenerator::new();
        let survivors = Spea2KnnSurvival::new().operate(pop, 4, &mut rng);

        // all individuals must survive
        assert_eq!(survivors.len(), 4);
    }
}
