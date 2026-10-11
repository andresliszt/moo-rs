To use quality metrics in `moors` just import the function you need from the crate root and call it with `ndarray` arrays. All metrics assume **minimization** and expect fitness matrices of shape `(n_individuals, n_objectives)`, exactly like [`Population::fitness`](https://docs.rs/moors/latest/moors/genetic/struct.Population.html).

```Rust
use ndarray::array;
use moors::{generational_distance, hypervolume, inverted_generational_distance, spacing, spread};

let front = array![[0.0, 4.0], [1.0, 3.0], [2.0, 2.0], [3.0, 1.0], [4.0, 0.0]];
let reference_front = front.clone();
let reference_point = array![5.0, 5.0];

let hv = hypervolume(&front, &reference_point);
let gd = generational_distance(&front, &reference_front);
let igd = inverted_generational_distance(&front, &reference_front);
let sp = spacing(&front);
let delta = spread(&front, &reference_front); // only defined for 2 objectives
```
