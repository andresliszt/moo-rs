/// **Build a composite fitness-evaluation closure from one or more
/// scalar / vector fitness functions.**
///
/// The macro is intended for quick experiments and unit tests where you
/// already have individual fitness functions (each returning an
/// `Array1<f64>`) and need a single evaluator that produces the usual
/// **(n_individuals × n_objectives)** matrix.
///
/// # Syntax
///
/// ```ignore
/// // Two objectives
/// let fitness = fitness_fn!(f1, f2);
///
/// // Three objectives
/// let fitness = fitness_fn!(f_len, f_angle, f_energy);
/// ```
///
/// Each identifier (`f1`, `f2`, …) **must** implement the signature
///
/// ```rust
/// fn(&ndarray::Array2<f64>) -> ndarray::Array1<f64>
/// ```
///
/// The macro expands to a *closure*
///
/// ```rust
/// |genes: &Array2<f64>| -> Array2<f64>
/// ```
///
/// that:
///
/// 1. evaluates every objective function on the same `genes` matrix;
/// 2. stacks all returned `Array1`s as **column views** along `Axis(1)`,
///    yielding the final `(rows × objectives)` fitness matrix.
///
/// `ndarray::concatenate` is used internally; no intermediate copies are
/// made besides the `Array1`s returned by the objectives themselves.
///
/// # Example
///
/// ```rust
/// use ndarray::{array, Array1, Array2, Axis};
/// use moors::fitness_fn;
///
/// // Simple objectives
/// fn sphere(genes: &Array2<f64>) -> Array1<f64> {
///     genes.map_axis(Axis(1), |row| row.dot(&row))        // ∑ x_i²
/// }
///
/// fn linear(genes: &Array2<f64>) -> Array1<f64> {
///     genes.map_axis(Axis(1), |row| row.sum())            // ∑ x_i
/// }
///
/// // Build the composite evaluator
/// let fitness = fitness_fn!(sphere, linear);
///
/// // Evaluate two individuals
/// let genes = array![[1.0, 0.0], [2.0, 2.0]];
/// let f = fitness(&genes);          // shape: (2 × 2)
/// assert_eq!(f, array![[1.0, 1.0], [8.0, 4.0]]);
/// ```
#[macro_export]
macro_rules! fitness_fn {
    ($($c:expr),+ $(,)?) => {
        |genes: &ndarray::Array2<f64>| -> ndarray::Array2<f64> {
            // Evaluate every objective
            let cols = vec![ $( ($c)(genes) ),+ ];

            // Convert to column views and concatenate
            let views: Vec<_> = cols
                .iter()
                .map(|v| v.view().insert_axis(ndarray::Axis(1)))
                .collect();

            ndarray::concatenate(ndarray::Axis(1), &views)
                .expect("concatenate along axis 1")
        }
    };
}

/* ===================================================================
Unit tests
=================================================================== */
#[cfg(test)]
mod tests {
    use ndarray::{Array1, Array2, Axis, array};

    use crate::fitness_fn;

    // Objective 1: sum of squares
    fn f_sphere(genes: &Array2<f64>) -> Array1<f64> {
        genes.map_axis(Axis(1), |row| row.dot(&row))
    }

    // Objective 2: sum of coordinates
    fn f_sum(genes: &Array2<f64>) -> Array1<f64> {
        genes.map_axis(Axis(1), |row| row.sum())
    }

    // Objective 3: difference x₀ − x₁
    fn f_diff(genes: &Array2<f64>) -> Array1<f64> {
        genes.map_axis(Axis(1), |row| row[0] - row[1])
    }

    #[test]
    fn two_objectives() {
        let eval = fitness_fn!(f_sphere, f_sum);

        let genes = array![[1.0, 0.0], [2.0, 2.0]];
        let out = eval(&genes);

        assert_eq!(out.shape(), &[2, 2]);
        assert_eq!(out, array![[1.0, 1.0], [8.0, 4.0]]);
    }

    #[test]
    fn three_objectives() {
        let eval = fitness_fn!(f_sphere, f_sum, f_diff);

        let genes = array![[1.0, 0.0], [2.0, 3.0]];
        let out = eval(&genes);

        // Check shape
        assert_eq!(out.shape(), &[2, 3]);

        // Expected values
        let expected = array![
            [1.0, 1.0, 1.0],   // (1²+0², 1+0, 1−0)
            [13.0, 5.0, -1.0], // (2²+3², 2+3, 2−3)
        ];
        assert_eq!(out, expected);
    }

    #[test]
    fn reuse_closure() {
        let eval = fitness_fn!(f_sum);

        let g1 = array![[1.0, 2.0]];
        let g2 = array![[0.0, 0.0]];

        assert_eq!(eval(&g1)[[0, 0]], 3.0);
        assert_eq!(eval(&g2)[[0, 0]], 0.0);
    }
}
