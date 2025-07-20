## Fitness

In **moors**, the way to define objective functions for optimization is through a [ndarray](https://docs.rs/ndarray/latest/ndarray/). Currently, the only dtype supported is `f64`, we're planning to relax this in the future. It means that when working with a different dtype, such as binary, its values must be trated as `f64` (in this case as `0.0` and `1.0`).

 An example is given below


```Rust
use ndarray::{Array2, Axis, stack};

/// DTLZ2 for 3 objectives (m = 3) with k = 0 (so num_vars = m−1 = 2):
/// f1 = cos(π/2 ⋅ x0) ⋅ cos(π/2 ⋅ x1)
/// f2 = cos(π/2 ⋅ x0) ⋅ sin(π/2 ⋅ x1)
/// f3 = sin(π/2 ⋅ x0)
fn fitness_dtlz2_3obj(genes: &Array2<f64>) -> Array2<f64> {
    let half_pi = std::f64::consts::PI / 2.0;
    let x0 = genes.column(0).mapv(|v| v * half_pi);
    let x1 = genes.column(1).mapv(|v| v * half_pi);

    let c0 = x0.mapv(f64::cos);
    let s0 = x0.mapv(f64::sin);
    let c1 = x1.mapv(f64::cos);
    let s1 = x1.mapv(f64::sin);

    let f1 = &c0 * &c1;
    let f2 = &c0 * &s1;
    let f3 = s0;

    stack(Axis(1), &[f1.view(), f2.view(), f3.view()]).expect("stack failed")
}
```

This funcion has the signature `(genes: &Array2<f64>) -> Array2<f64>` and is a valid function for any `moors` multi-objective optimization algorithm, such as [Nsga2]({{ docs_rs("struct", "algorithms.Nsga2") }}), [Nsga3]({{ docs_rs("struct", "algorithms.Nsga3") }}), etc. Note that this function is poblational, meaning that the **whole** population is evaluated

A function for a single objective optimization problem has the signature `(genes: &Array2<f64>) -> Array1<f64>` and is valid for any `moors` single optimization algorithm. An example is given below

```Rust
use ndarray::{Array1, Array2, Axix};

/// Simple minimization of 1 - (x**2 + y**2 + z**2)
fn fitness_sphere(population: &Array2<f64>) -> Array1<f64> {
    // For each row [x, y, z], compute 1 - x^2 + y^2 + z^2
    population.map_axis(Axis(1), |row| 1.0 - row.dot(&row))
}
```

## Constraints

Constraints are also a ndarray function, for multi and single optimization the constraints have the same signature: can be either `(genes: &Array2<f64>) -> Array1<f64>` for single constraints or `(genes: &Array2<f64>) -> Array2<f64>` for more than one constraints.


### Feasibility

This concept is very important in optimization, an individual is called *feasible* if and only if it satisfies all the constraints the problem defines. In `moors` as in many other optimization frameworks, constraints allowed are evaluated as `<= 0.0`. In genetic algorithms, there are different ways to incorporate feasibility in the search for optimal solutions. In this framework, the guiding philosophy is: *feasibility dominates everything*, meaning that a feasible individual is always preferred over an infeasible one.

### Inequality constraints

In `moors` as mentioned, any output from a constraint function is evaluated as less than or equal to zero. If this condition is met, the individual is considered feasible. For constraints that are naturally expressed as greater than zero, the user should modify the function by multiplying it by -1, as shown in the following example

```Rust
fn constraints_sphere_lower_than_zero(population: &Array2<f64>) -> Array1<f64> {
    // For each row [x, y, z], compute x^2 + y^2 + z^2 - 1 <= 0
    population.map_axis(Axis(1), |row| row.dot(&row)) - 1.0
}

fn constraints_sphere_greather_than_zero(population: &Array2<f64>) -> Array1<f64> {
    // For each row [x, y, z], compute x^2 + y^2 + z^2 - 1 => 0
    1.0 - population.map_axis(Axis(1), |row| row.dot(&row))
}
```
### Equality constraints

As is many other frameworks, the known epsilon technique must be used to force $g(x) = 0$, select a tolerance $\epsilon$ and then transform $g$ into an inquality constraint

$$g_{\text{ineq}}(x) = \bigl|g(x)\bigr| - \varepsilon \;\le\; 0.$$

An example is given below

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

### Helper macro to define constraints

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

This macro generates a new struct `MyConstraints` than can be passed to any algorithm, you can pass multiple inequality/equality constraints to the macro `impl_constraints_fn(MyConstraints, ineq =[g1, g2, ...], eq = [h1, h2, ..])`. This macro will use the epsilon technique internally using a fixed tolerance of `1e-6`, the last in the near future will be seteable by the user

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
    Internally, `constraints` as an argument to genetic algorithms is actually any type that implements [ConstraintsFn]({{ docs_rs("trait", "ConstraintsFn") }}). The `impl_constraints_fn` macro creates a struct that implements this trait. The types `Fn(&Array2<f64>) -> Array1<f64>` and `Fn(&Array2<f64>) -> Array2<f64>` automatically implement this trait.
