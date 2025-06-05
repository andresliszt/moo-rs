// =============================================================================
// Constraint-building macros
// =============================================================================
//
// ▸ `__eq_helper!(g)`              – Wraps a *single* constraint function `g`
//                           and treats it as an **equality** constraint,
//                           i.e. `|g(genes)| − ε ≤ 0` with ε = 1 × 10⁻⁶.
//
// ▸ `__constraints_helper!( … )`   – Internal helper that concatenates **already-wrapped**
//                           constraint functions into one closure that returns
//                           a 2-D array.
//
// ▸ **`constraints_fn!( … )`** – Public, user-facing macro that lets you list
//                                inequalities first and (optionally) equalities
//                                after `; eq:`.  It expands to a closure
//                                `|genes: &Array2<f64>| -> Array2<f64>`.
//
//   ┌─────────────────────── Example ─────────────────────────
//   │ let eval = constraints_fn!(
//   │     g1, g2,                // inequalities: g(genes) ≤ 0
//   │     ; eq: h1, h2, h3       // equalities:   |h(genes)| − ε ≤ 0
//   │ );
//   │
//   │ let m = eval(&population);  // (n_rows × 5) array
//   └──────────────────────────────────────────────────────────
//
//   Every identifier (`g1`, `h1`, …) **must** have the signature
//   `fn(&Array2<f64>) -> Array1<f64>`.
// =============================================================================

/// Wrap a single constraint function as an **equality**
/// (`|g(genes)| − ε ≤ 0`, ε = 1 × 10⁻⁶).
///
/// This macro is **internal**; end-users should normally invoke it
/// through [`constraints_fn!`](#macro.constraints_fn).
#[macro_export]
macro_rules! __eq_helper {
    ($f:path $(,)?) => {
        |genes: &ndarray::Array2<f64>| -> ndarray::Array1<f64> {
            $f(genes).mapv(|v| v.abs() - 1e-6)
        }
    };
}

/// Concatenate any number of already-wrapped constraints (inequalities
/// or calls to [`eq!`](#macro.eq)) into a single evaluator closure.
///
/// **Internal helper** – users call [`constraints_fn!`](#macro.constraints_fn)
/// instead.
#[macro_export]
macro_rules! __constraints_helper {
    ($($c:expr),+ $(,)?) => {
        |genes: &ndarray::Array2<f64>| -> ndarray::Array2<f64> {
            // Evaluate every constraint
            let cols = vec![ $( ($c)(genes) ),+ ];

            // Concatenate column views along axis-1
            let views: Vec<_> = cols.iter()
                .map(|v| v.view().insert_axis(ndarray::Axis(1)))
                .collect();

            ndarray::concatenate(ndarray::Axis(1), &views)
                .expect("concatenate along axis 1")
        }
    };
}

/// **Build a combined constraint evaluator** from any mix of
/// *inequality* and *equality* constraint functions.
///
/// # Syntax
///
/// ```ignore
/// // Only inequalities
/// constraints_fn!(g1, g2, g3);
///
/// // Inequalities followed by equalities
/// constraints_fn!(
///     g1, g2,                 // inequalities  g(genes) ≤ 0
///     ; eq: h1, h2,           // equalities    |h(genes)| − ε ≤ 0
/// );
/// ```
///
/// * Inequalities must come **before** the semicolon.<br>
/// * The `; eq:` block is **optional**.
/// * Each listed symbol (`g1`, `h1`, …) must implement
///   `fn(&Array2<f64>) -> Array1<f64>`.
///
/// The macro expands to a **closure**
///
/// ```rust, ignore
/// |genes: &ndarray::Array2<f64>| -> ndarray::Array2<f64>
/// ```
///
/// returning an `(n_rows × n_constraints)` matrix whose columns contain
/// all inequalities in the given order, followed by all transformed
/// equalities (ε = 1 × 10⁻⁶).
///
/// # Example
///
/// ```rust
/// use ndarray::{array, Array1, Array2, Axis};
/// use moors::constraints_fn;
///
/// fn g1(genes: &Array2<f64>) -> Array1<f64> {
///     genes.map_axis(Axis(1), |r| r.sum() - 1.0)        // ≤ 0
/// }
///
/// fn h1(genes: &Array2<f64>) -> Array1<f64> {
///     genes.map_axis(Axis(1), |r| r[0] - r[1])          // = 0  → |·| − ε
/// }
///
/// let my_constraints_clousure = constraints_fn!(g1; eq: h1);
/// let genes = array![[0.5, 0.5], [1.5, 0.0]];
/// let mat = my_constraints_clousure(&genes);          // shape: (2 × 2)
/// ```
#[macro_export]
macro_rules! constraints_fn {
    // Inequalities + Equalities
    ( $( $ineq:path ),* ; eq: $( $eq:path ),+ $(,)? ) => {
        $crate::__constraints_helper!(
            $( $ineq ),* ,
            $( $crate::__eq_helper!($eq) ),*
        )
    };

    // Inequalities only
    ( $( $ineq:path ),+ $(,)? ) => {
        $crate::__constraints_helper!($( $ineq ),*)
    };
}

#[cfg(test)]
mod tests {
    use ndarray::{Array1, Array2, Axis, array};

    use crate::constraints_fn;

    /* ───────────────── helper constraint fns ───────────────── */

    /// g1(x,y) = x + y – 1  ≤ 0
    fn g1(genes: &Array2<f64>) -> Array1<f64> {
        genes.map_axis(Axis(1), |row| row.sum() - 1.0)
    }
    /// g2(x,y) = x² + y² – 1 ≤ 0
    fn g2(genes: &Array2<f64>) -> Array1<f64> {
        genes.map_axis(Axis(1), |row| row.dot(&row) - 1.0)
    }
    /// g3(x,y) = x – y  = 0   (will be wrapped in `eq!(…)`)
    fn g3(genes: &Array2<f64>) -> Array1<f64> {
        genes.map_axis(Axis(1), |row| row[0] - row[1])
    }

    /* ───────────────── unit tests ───────────────── */

    #[test]
    fn inequalities_only() {
        let genes = array![[0.0, 0.0], [1.0, 1.0]]; // 2 × 2
        let comb = constraints_fn!(g1, g2); // closure (genes) -> Array2
        let res = comb(&genes); // 2 × 2

        let expect = array![[-1.0, -1.0], [1.0, 1.0]];
        assert_eq!(res.shape(), &[2, 2]);
        assert_eq!(res, expect);
    }

    #[test]
    fn mixed_ineq_and_eq() {
        const EPS: f64 = 1e-6;

        let genes = array![
            [0.5, 0.5], // g1 = 0,   g2 = -0.5, g3 = 0  → |g3|-ε = -ε
            [2.0, 1.0], // g1 = 2.0, g2 = 4.0,  g3 = 1  → 1-ε
        ];
        // g1, g2,  |g3|-ε
        let comb = constraints_fn!(g1, g2 ; eq: g3);
        let res = comb(&genes); // 2 × 3

        // Build expected matrix by hand
        let mut exp = Array2::<f64>::zeros((2, 3));
        exp[[0, 0]] = 0.0; // g1 row0
        exp[[0, 1]] = -0.5; // g2 row0
        exp[[0, 2]] = -EPS; // |g3|-ε row0

        exp[[1, 0]] = 2.0; // g1 row1
        exp[[1, 1]] = 4.0; // g2 row1
        exp[[1, 2]] = 1.0 - EPS; // |g3|-ε row1

        assert_eq!(res.shape(), &[2, 3]);
        for (&a, &e) in res.iter().zip(exp.iter()) {
            assert!((a - e).abs() < 1e-10, "got {a}, expected {e}");
        }
    }

    #[test]
    fn reuse_combined_closure() {
        // Ensure the closure returned by `constraints!` can be reused
        let comb = constraints_fn!(g1; eq: g3);

        let p1 = array![[1.0, 1.0]];
        let p2 = array![[0.0, 0.0]];

        let m1 = comb(&p1);
        let m2 = comb(&p2);

        // Shapes: 1 × 2
        assert_eq!(m1.shape(), &[1, 2]);
        assert_eq!(m2.shape(), &[1, 2]);

        // Quick sanity checks
        assert!((m1[[0, 0]] - 1.0).abs() < 1e-10); // g1(1,1) = 1
        assert!((m2[[0, 0]] + 1.0).abs() < 1e-10); // g1(0,0) = -1
    }
}
