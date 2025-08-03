Also this macro has two optional arguments `lower_bound` and `upper_bound` that will make each gene bounded by those values.


```rust
use ndarray::{Array2, Array1, Axis};

use moors::impl_constraints_fn;

/// Equality constraint x + y = 1
fn constraints_eq(genes: &Array2<f64>) -> Array1<f64> {
    genes.map_axis(Axis(1), |row| row[0] + row[1] - 1.0)
}

impl_constraints_fn!(
    MyBoundedConstraints,
    eq   = [constraints_eq],
    lower_bound = -1.0,
    upper_bound = 1.0
);

```

!!! info "`ConstraintsFn` trait"
    Internally, `constraints` as an argument to genetic algorithms is actually any type that implements {{ docs_rs("trait", "evaluator.ConstraintsFn") }}. The `impl_constraints_fn` macro creates a struct that implements this trait. The types `Fn(&Array2<f64>) -> Array1<f64>` and `Fn(&Array2<f64>) -> Array2<f64>` automatically implement this trait.
