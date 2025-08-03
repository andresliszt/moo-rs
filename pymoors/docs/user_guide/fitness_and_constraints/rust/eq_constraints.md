
```rust
use ndarray::{Array2, Array1, Axis};

const EPSILON: f64 = 1e-6;

/// Returns an Array2 of shape (n, 2) containing two constraints for each row [x, y]:
/// - Column 0: |x + y - 1| - EPSILON ≤ 0 (equality with ε-tolerance)
/// - Column 1: x² + y² - 1.0 ≤ 0 (unit circle inequality)
fn constraints(genes: &Array2<f64>) -> Array2<f64> {
    // Constraint 1: |x + y - 1| - EPSILON
    let eq = genes.map_axis(Axis(1), |row| (row[0] + row[1] - 1.0).abs() - EPSILON);
    // Constraint 2: x^2 + y^2 - 1
    let ineq = genes.map_axis(Axis(1), |row| row[0].powi(2) + row[1].powi(2) - 1.0);
    // Stack into two columns
    stack(Axis(1), &[eq.view(), ineq.view()]).unwrap()
}
```

This example ilustrates 2 constraints where one of them is an equality constraint.

There is a helper macro `moors::impl_constraints_fn` that will build the expected constraints for us, trying to simplify at most is  possible the boliparte code


```rust
use ndarray::{Array2, Array1, Axis};

use moors::impl_constraints_fn;

/// Equality constraint x + y = 1
fn constraints_eq(genes: &Array2<f64>) -> Array1<f64> {
    genes.map_axis(Axis(1), |row| row[0] + row[1] - 1.0)
}

/// Inequality constraint: x² + y² - 1 ≤ 0
fn constraints_ineq(genes: &Array2<f64>) -> Array1<f64> {
    genes.map_axis(Axis(1), |row| row[0].powi(2) + row[1].powi(2) - 1.0)
}

impl_constraints_fn!(
    MyConstraints,
    ineq = [constraints_ineq],
    eq   = [constraints_eq],
);

```

This macro generates a new struct `MyConstraints` than can be passed to any algorithm, you can pass multiple inequality/equality constraints to the macro `impl_constraints_fn(MyConstraints, ineq =[g1, g2, ...], eq = [h1, h2, ..])`. This macro will use the epsilon technique internally using a fixed tolerance of `1e-6`, the last in the near future will be seteable by the user.
