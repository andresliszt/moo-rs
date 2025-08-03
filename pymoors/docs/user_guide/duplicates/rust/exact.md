To use close duplicates cleaner in `moors` just pass to the algorithm the cleaner instance

```Rust

use ndarray::{array, Array2};

use moors::ExactDuplicatesCleaner;

let raw_data = vec![
    1.0, 2.0, 3.0, // row 0
    4.0, 5.0, 6.0, // row 1
    1.0, 2.0, 3.0, // row 2 (duplicate of row 0)
    7.0, 8.0, 9.0, // row 3
    4.0, 5.0, 6.0, // row 4 (duplicate of row 1)
];
let population =
    Array2::<f64>::from_shape_vec((5, 3), raw_data).expect("Failed to create test array");

let cleaner = ExactDuplicatesCleaner::new();
let cleaned = cleaner.remove(population, None);

let expected = array![
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
    [7.0, 8.0, 9.0],
];

assert_eq!(cleaned, expected);
```
