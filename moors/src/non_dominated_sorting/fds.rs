use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use ndarray::{Array1, ArrayView1, Axis};
use rayon::prelude::*;

use crate::genetic::{Fronts, Population, PopulationFitness};

/// Inlines the check for "does f1 dominate f2?" to reduce call overhead.
#[inline]
fn dominates(f1: &ArrayView1<f64>, f2: &ArrayView1<f64>) -> bool {
    let mut better = false;
    // We assume f1.len() == f2.len()
    for (&a, &b) in f1.iter().zip(f2.iter()) {
        if a > b {
            return false;
        } else if a < b {
            better = true;
        }
    }
    better
}

/// Parallel Fast Non-Dominated Sorting.
/// Returns a vector of fronts, each front is a list of indices.
/// The individuals are grouped into fronts in order of non-dominance.
/// If during the construction of fronts the cumulative count of individuals reaches or exceeds
/// `min_survivors`, the entire last front is included (even if it causes the total to exceed `min_survivors`)
/// and no further fronts are added.
pub fn fast_non_dominated_sorting(
    population_fitness: &PopulationFitness,
    min_survivors: usize,
) -> Vec<Vec<usize>> {
    let population_size = population_fitness.shape()[0];

    // Thread-safe data structures
    let domination_count = (0..population_size)
        .map(|_| AtomicUsize::new(0))
        .collect::<Vec<_>>();
    let dominated_sets = (0..population_size)
        .map(|_| Mutex::new(Vec::new()))
        .collect::<Vec<_>>();

    // Precompute row views to avoid repeated indexing
    let fitness_rows: Vec<ArrayView1<f64>> = (0..population_size)
        .map(|i| population_fitness.index_axis(Axis(0), i))
        .collect();

    // Parallel pairwise comparisons: for each pair (p, q) with p < q, each thread updates local data
    (0..population_size).into_par_iter().for_each(|p| {
        // Accumulate changes locally to reduce locking overhead
        let mut local_updates = Vec::new();

        for q in (p + 1)..population_size {
            let p_dominates_q = dominates(&fitness_rows[p], &fitness_rows[q]);
            let q_dominates_p = dominates(&fitness_rows[q], &fitness_rows[p]);

            if p_dominates_q {
                // p dominates q
                local_updates.push((p, q));
            } else if q_dominates_p {
                // q dominates p
                local_updates.push((q, p));
            }
            // else -> neither dominates
        }

        // Apply local updates to shared data:
        // For each (dominator, dominated) pair:
        for (dominator, dominated) in local_updates {
            {
                // Push dominated into the dominator's list
                let mut lock = dominated_sets[dominator].lock().unwrap();
                lock.push(dominated);
            }
            // Increment the atomic domination_count of the dominated individual
            domination_count[dominated].fetch_add(1, Ordering::Relaxed);
        }
    });

    // Convert to a normal Vec<Vec<usize>>
    let dominated_sets_vec: Vec<Vec<usize>> = dominated_sets
        .into_iter()
        .map(|m| m.into_inner().unwrap())
        .collect();

    // Build the first front
    let mut fronts = Vec::new();
    let mut first_front = Vec::new();
    for i in 0..population_size {
        if domination_count[i].load(Ordering::Relaxed) == 0 {
            first_front.push(i);
        }
    }
    fronts.push(first_front.clone());
    let mut count = first_front.len();
    // If the first front already reaches min_survivors, return immediately.
    if count >= min_survivors {
        return fronts;
    }

    // Construct subsequent fronts
    let mut current_front = first_front;
    while !current_front.is_empty() {
        let mut next_front = Vec::new();
        for &p in &current_front {
            for &q in &dominated_sets_vec[p] {
                let old_count = domination_count[q].fetch_sub(1, Ordering::Relaxed);
                if old_count == 1 {
                    // now it's zero
                    next_front.push(q);
                }
            }
        }
        if next_front.is_empty() {
            break;
        }
        // If adding the next front reaches or exceeds min_survivors,
        // include the entire front and stop the construction.
        if count + next_front.len() >= min_survivors {
            fronts.push(next_front);
            break;
        } else {
            count += next_front.len();
            fronts.push(next_front.clone());
            current_front = next_front;
        }
    }

    fronts
}

/// Builds the fronts from the population.
pub fn build_fronts(population: Population, n_survive: usize) -> Fronts {
    let sorted_fronts = fast_non_dominated_sorting(&population.fitness, n_survive);
    let mut results: Fronts = Vec::new();

    // For each front (with rank = front_index), extract the sub-population.
    for (front_index, indices) in sorted_fronts.iter().enumerate() {
        let front_genes = population.genes.select(Axis(0), &indices[..]);
        let front_fitness = population.fitness.select(Axis(0), &indices[..]);
        let front_constraints = population
            .constraints
            .as_ref()
            .map(|c| c.select(Axis(0), &indices[..]));

        // Create a rank Array1 (one rank value per individual in the front).
        let rank_arr = Some(Array1::from_elem(indices.len(), front_index));

        // Build a `Population` representing this entire front.
        let population_front =
            Population::new(front_genes, front_fitness, front_constraints, rank_arr);

        results.push(population_front);
    }
    results
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, array};

    #[test]
    fn test_dominates() {
        // Test case 1: The first vector dominates the second
        let a = array![1.0, 2.0, 3.0];
        let b = array![2.0, 3.0, 4.0];
        assert_eq!(dominates(&a.view(), &b.view()), true);

        // Test case 2: The second vector dominates the first
        let a = array![3.0, 3.0, 3.0];
        let b = array![2.0, 4.0, 5.0];
        assert_eq!(dominates(&a.view(), &b.view()), false);

        // Test case 3: Neither vector dominates the other
        let a = array![1.0, 2.0, 3.0];
        let b = array![2.0, 1.0, 3.0];
        assert_eq!(dominates(&a.view(), &b.view()), false);

        // Test case 4: Equal vectors
        let a = array![1.0, 2.0, 3.0];
        let b = array![1.0, 2.0, 3.0];
        assert_eq!(dominates(&a.view(), &b.view()), false);
    }

    #[test]
    fn test_fast_non_dominated_sorting() {
        // Define the fitness values of the population
        let population_fitness = array![
            [1.0, 2.0], // Individual 0
            [2.0, 1.0], // Individual 1
            [1.5, 1.5], // Individual 2
            [3.0, 4.0], // Individual 3 (dominated by everyone)
            [4.0, 3.0]  // Individual 4 (dominated by everyone)
        ];

        // Perform fast non-dominated sorting with min_survivors = 5
        let fronts = fast_non_dominated_sorting(&population_fitness, 5);

        // Expected Pareto fronts:
        // Front 1: Individuals 0, 1, 2
        // Front 2: Individuals 3, 4 (the entire front is included when min_survivors is reached)
        let expected_fronts = vec![
            vec![0, 1, 2], // Front 1
            vec![3, 4],    // Front 2
        ];

        assert_eq!(fronts, expected_fronts);
    }

    #[test]
    fn test_fast_non_dominated_sorting_single_front() {
        // Define a population where no individual dominates another
        let population_fitness = array![
            [1.0, 2.0], // Individual 0
            [2.0, 1.0], // Individual 1
            [1.5, 1.5], // Individual 2
        ];

        // Perform fast non-dominated sorting with min_survivors = 3
        let fronts = fast_non_dominated_sorting(&population_fitness, 3);

        // Expected Pareto front: All individuals belong to the same front.
        // The front is returned in its entirety when min_survivors is reached.
        let expected_fronts = vec![
            vec![0, 1, 2], // Front 1
        ];

        assert_eq!(fronts, expected_fronts);
    }

    #[test]
    fn test_fast_non_dominated_sorting_empty_population() {
        // Define an empty population
        let population_fitness: Array2<f64> = Array2::zeros((0, 0));

        // Perform fast non-dominated sorting with min_survivors = 0
        let fronts = fast_non_dominated_sorting(&population_fitness, 0);

        // Expected: No fronts
        let expected_fronts: Vec<Vec<usize>> = vec![vec![]];

        assert_eq!(fronts, expected_fronts);
    }

    #[test]
    fn test_fast_non_dominated_sorting_n_survive_cut() {
        // Define a population with clear dominance relationships and duplicate fitness values
        // to force multiple individuals in a front.
        let population_fitness = array![
            [1.0, 1.0], // Individual 0: best, not dominated by anyone
            [2.0, 2.0], // Individual 1: dominated by 0
            [2.0, 2.0], // Individual 2: duplicate of 1, same front as 1
            [3.0, 3.0], // Individual 3: dominated by 0,1,2
            [4.0, 4.0]  // Individual 4: dominated by 0,1,2,3
        ];

        // Set min_survivors = 2. The first front is [0] (1 individual).
        // The next front is [1, 2] (2 individuals). Adding this front reaches min_survivors (1 + 2 > 2)
        // so the algorithm should include the entire second front and stop.
        let fronts = fast_non_dominated_sorting(&population_fitness, 2);
        let expected_fronts = vec![
            vec![0],    // Front 1
            vec![1, 2], // Front 2 (included entirely when min_survivors is reached)
        ];

        assert_eq!(fronts, expected_fronts);
    }

    #[test]
    fn test_build_fronts_behavior() {
        // Create a population with 5 individuals and 2 objectives.
        // For simplicity, we set the fitness equal to the genes.
        // Genes (and fitness) are chosen such that:
        //   - Individual 1: [1.0, 1.0] -> best, should be in front 0.
        //   - Individual 2: [2.0, 2.0]
        //   - Individual 3: [1.5, 2.5]
        //   - Individual 4: [2.5, 1.5]
        //   - Individual 5: [3.0, 3.0] -> dominated by one of the above, hence in a later front.
        let genes = array![
            [1.0, 1.0], // Individual 1
            [2.0, 2.0], // Individual 2
            [1.5, 2.5], // Individual 3
            [2.5, 1.5], // Individual 4
            [3.0, 3.0]  // Individual 5
        ];
        let fitness = genes.clone();
        let constraints = Some(Array2::from_elem((5, 2), -1.0));

        // Build the Population (with rank set to None initially).
        let population = Population::new(genes, fitness, constraints, None);

        // Call build_fronts with n_survive = 5.
        let fronts = build_fronts(population, 5);

        // We expect three fronts:
        //   Front 0: 1 individual (the best).
        //   Front 1: 3 individuals (non-dominated among themselves).
        //   Front 2: 1 individual.
        assert_eq!(
            fronts.len(),
            3,
            "Expected 3 fronts based on the objectives."
        );

        // Check front sizes.
        assert_eq!(
            fronts[0].genes.nrows(),
            1,
            "Front 0 should have 1 individual."
        );
        assert_eq!(
            fronts[1].genes.nrows(),
            3,
            "Front 1 should have 3 individuals."
        );
        assert_eq!(
            fronts[2].genes.nrows(),
            1,
            "Front 2 should have 1 individual."
        );

        // Verify each front's rank array.
        for (front_index, front) in fronts.iter().enumerate() {
            let rank = front
                .rank
                .as_ref()
                .expect("Each front should have a rank array.");
            assert_eq!(
                rank.len(),
                front.genes.nrows(),
                "Rank array length must match the number of individuals."
            );
            for &r in rank.iter() {
                assert_eq!(
                    r, front_index,
                    "Each rank value should equal the front index."
                );
            }
        }

        // Check that all constraints in each front are feasible (≤ 0).
        for front in fronts.iter() {
            if let Some(constraints_arr) = &front.constraints {
                for row in constraints_arr.outer_iter() {
                    for &val in row.iter() {
                        assert!(
                            val <= 0.0,
                            "All constraints values should be feasible (≤ 0)."
                        );
                    }
                }
            }
        }
    }
}
