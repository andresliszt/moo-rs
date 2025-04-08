use std::collections::HashSet;
use std::fmt::Debug;

use ordered_float::OrderedFloat;

use crate::duplicates::PopulationCleaner;
use crate::genetic::PopulationGenes;

#[derive(Clone, Debug)]
/// Exact duplicates cleaner based on Hash
pub struct ExactDuplicatesCleaner;

impl ExactDuplicatesCleaner {
    pub fn new() -> Self {
        Self
    }
}

impl PopulationCleaner for ExactDuplicatesCleaner {
    fn remove(
        &self,
        population: &PopulationGenes,
        reference: Option<&PopulationGenes>,
    ) -> PopulationGenes {
        let ncols = population.ncols();
        let mut unique_rows: Vec<Vec<f64>> = Vec::new();
        // A HashSet to hold the hashable representation of rows.
        let mut seen: HashSet<Vec<OrderedFloat<f64>>> = HashSet::new();

        // If a reference is provided, first add its rows into the set.
        if let Some(ref_pop) = reference {
            for row in ref_pop.outer_iter() {
                let hash_row: Vec<OrderedFloat<f64>> =
                    row.iter().map(|&x| OrderedFloat(x)).collect();
                seen.insert(hash_row);
            }
        }

        // Iterate over the population rows.
        for row in population.outer_iter() {
            let hash_row: Vec<OrderedFloat<f64>> = row.iter().map(|&x| OrderedFloat(x)).collect();
            // Insert returns true if the row was not in the set.
            if seen.insert(hash_row) {
                unique_rows.push(row.to_vec());
            }
        }

        // Flatten the unique rows into a single vector.
        let data_flat: Vec<f64> = unique_rows.into_iter().flatten().collect();
        PopulationGenes::from_shape_vec((data_flat.len() / ncols, ncols), data_flat)
            .expect("Failed to create deduplicated Array2")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::genetic::PopulationGenes;
    use ndarray::array;

    #[test]
    fn test_exact_duplicates_cleaner_removes_duplicates_without_reference() {
        let raw_data = vec![
            1.0, 2.0, 3.0, // row 0
            4.0, 5.0, 6.0, // row 1
            1.0, 2.0, 3.0, // row 2 (duplicate of row 0)
            7.0, 8.0, 9.0, // row 3
            4.0, 5.0, 6.0, // row 4 (duplicate of row 1)
        ];
        let population =
            PopulationGenes::from_shape_vec((5, 3), raw_data).expect("Failed to create test array");
        let cleaner = ExactDuplicatesCleaner::new();
        let cleaned = cleaner.remove(&population, None);
        assert_eq!(cleaned.nrows(), 3);
        assert_eq!(cleaned.row(0).to_vec(), vec![1.0, 2.0, 3.0]);
        assert_eq!(cleaned.row(1).to_vec(), vec![4.0, 5.0, 6.0]);
        assert_eq!(cleaned.row(2).to_vec(), vec![7.0, 8.0, 9.0]);
    }

    #[test]
    fn test_exact_duplicates_cleaner_with_reference() {
        let population = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let reference = array![[1.0, 2.0, 3.0]]; // row 0 is in reference
        let cleaner = ExactDuplicatesCleaner::new();
        let cleaned = cleaner.remove(&population, Some(&reference));
        assert_eq!(cleaned.nrows(), 1);
        assert_eq!(cleaned.row(0).to_vec(), vec![4.0, 5.0, 6.0]);
    }
}
