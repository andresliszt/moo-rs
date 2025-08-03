To use close duplicates cleaner in `moors` just pass to the algorithm the cleaner instance

```Rust

use ndarray::array;

use moors::CloseDuplicatesCleaner;

let population = array![[1.0, 2.0, 3.0], [10.0, 10.0, 10.0]];
let reference = array![[1.01, 2.01, 3.01]];  // close to row 0 of population
let epsilon = 0.05;
let cleaner = CloseDuplicatesCleaner::new(epsilon);
let cleaned = cleaner.remove(population, Some(&reference));
// Row 0 should be removed.
assert_eq!(cleaned.nrows(), 1);
assert_eq!(cleaned.row(0).to_vec(), vec![10.0, 10.0, 10.0]);

```
