use std::fmt::Debug;

use crate::duplicates::PopulationCleaner;
use crate::genetic::PopulationGenes;
use crate::helpers::linalg::cross_euclidean_distances;

#[derive(Clone, Debug)]
/// Computes the cross squared Euclidean distance matrix between `data` and `reference`
/// using matrix algebra.
///
/// For data of shape (n, d) and reference of shape (m, d), returns an (n x m) matrix
/// where the (i,j) element is the squared Euclidean distance between the i-th row of data
/// and the j-th row of reference.
pub struct CloseDuplicatesCleaner {
    pub epsilon: f64,
}

impl CloseDuplicatesCleaner {
    pub fn new(epsilon: f64) -> Self {
        Self { epsilon }
    }
}

impl PopulationCleaner for CloseDuplicatesCleaner {
    fn remove(
        &self,
        population: &PopulationGenes,
        reference: Option<&PopulationGenes>,
    ) -> PopulationGenes {
        let ref_array = reference.unwrap_or(population);
        let n = population.nrows();
        let num_cols = population.ncols();
        let dists_sq = cross_euclidean_distances(population, ref_array);

        let eps_sq = self.epsilon;
        let mut keep = vec![true; n];
        // Note: when reference_array = population there is no need to loop through the full
        // array, just use the upper triangle matrix logic
        if let Some(ref_pop) = reference {
            // Mark each row in the population as duplicate if its distance to any row in reference is below eps_sq.
            for i in 0..n {
                for j in 0..ref_pop.nrows() {
                    if dists_sq[(i, j)] <= eps_sq {
                        keep[i] = false;
                        break;
                    }
                }
            }
        } else {
            for i in 0..n {
                if !keep[i] {
                    continue;
                }
                for j in (i + 1)..n {
                    if dists_sq[(i, j)] < eps_sq {
                        keep[j] = false;
                    }
                }
            }
        }
        let kept_rows: Vec<_> = population
            .outer_iter()
            .enumerate()
            .filter_map(|(i, row)| if keep[i] { Some(row.to_owned()) } else { None })
            .collect();
        let data_flat: Vec<f64> = kept_rows.into_iter().flatten().collect();
        PopulationGenes::from_shape_vec((data_flat.len() / num_cols, num_cols), data_flat)
            .expect("Failed to create deduplicated Array2")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_close_duplicates_cleaner_without_reference() {
        let population = array![
            [1.0, 2.0, 3.0],
            [1.05, 2.05, 3.05], // very similar to row 0
            [4.0, 5.0, 6.0]
        ];
        let epsilon = 0.1;
        let cleaner = CloseDuplicatesCleaner::new(epsilon);
        let cleaned = cleaner.remove(&population, None);
        // Expect rows 0 and 2 remain.
        assert_eq!(cleaned.nrows(), 2);
    }

    #[test]
    fn test_close_duplicates_cleaner_with_reference() {
        let population = array![[1.0, 2.0, 3.0], [10.0, 10.0, 10.0]];
        let reference = array![
            [1.01, 2.01, 3.01] // close to row 0 of population
        ];
        let epsilon = 0.05;
        let cleaner = CloseDuplicatesCleaner::new(epsilon);
        let cleaned = cleaner.remove(&population, Some(&reference));
        // Row 0 should be removed.
        assert_eq!(cleaned.nrows(), 1);
        assert_eq!(cleaned.row(0).to_vec(), vec![10.0, 10.0, 10.0]);
    }
}
