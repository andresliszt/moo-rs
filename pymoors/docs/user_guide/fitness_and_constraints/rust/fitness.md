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

This funcion has the signature `(genes: &Array2<f64>) -> Array2<f64>` and is a valid function for any `moors` multi-objective optimization algorithm, such as {{ docs_rs("struct", "algorithms.Nsga2") }}, {{ docs_rs("struct", "algorithms.Nsga3") }} , etc. Note that this function is poblational, meaning that the **whole** population is evaluated

A function for a single objective optimization problem has the signature `(genes: &Array2<f64>) -> Array1<f64>` and is valid for any `moors` single optimization algorithm. An example is given below

```Rust
use ndarray::{Array1, Array2, Axix};

/// Simple minimization of 1 - (x**2 + y**2 + z**2)
fn fitness_sphere(population: &Array2<f64>) -> Array1<f64> {
    // For each row [x, y, z], compute 1 - x^2 + y^2 + z^2
    population.map_axis(Axis(1), |row| 1.0 - row.dot(&row))
}
```
