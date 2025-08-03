A duplicates cleaner in `moors` is any type that implements the {{ docs_rs("trait", "duplicates.PopulationCleaner") }} trait. For example:

```rust
use std::collections::HashSet;

use ndarray::Array2;
use ordered_float::OrderedFloat;

use crate::duplicates::PopulationCleaner;

#[derive(Debug, Clone)]
/// Exact duplicates cleaner based on Hash
pub struct ExactDuplicatesCleaner;

impl ExactDuplicatesCleaner {
    pub fn new() -> Self {
        Self
    }
}

impl PopulationCleaner for ExactDuplicatesCleaner {
    fn remove(&self, population: Array2<f64>, reference: Option<&Array2<f64>>) -> Array2<f64> {
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
        Array2::<f64>::from_shape_vec((data_flat.len() / ncols, ncols), data_flat)
            .expect("Failed to create deduplicated Array2")
    }
}
```

The main method to implement is `remove`, which takes two arguments: `population` and optional `reference`. If `reference` is provided, duplicates are determined by comparing each row in the population to all rows in the reference. The last is very important, when new offsprings are created, they may be unique, but if we compare them with the current population (in this case `reference`) they may not be unique.
