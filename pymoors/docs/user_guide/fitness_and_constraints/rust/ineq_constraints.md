```Rust
use ndarray::{Array1, Array2};

fn constraints_sphere_lower_than_zero(population: &Array2<f64>) -> Array1<f64> {
    // For each row [x, y, z], compute x^2 + y^2 + z^2 - 1 <= 0
    population.map_axis(Axis(1), |row| row.dot(&row)) - 1.0
}

fn constraints_sphere_greather_than_zero(population: &Array2<f64>) -> Array1<f64> {
    // For each row [x, y, z], compute x^2 + y^2 + z^2 - 1 => 0
    1.0 - population.map_axis(Axis(1), |row| row.dot(&row))
}
```
