// =============================================================================
// Constraint-building macros for moors
// =============================================================================
//
// ▸ `__eq_helper!(g)`              – Wraps a *single* constraint function `g`
//                                    and treats it as an **equality** constraint,
//                                    i.e. `|g(genes)| - ε ≤ 0` with ε = 1 × 10⁻⁶.
//
// ▸ `__constraints_helper!( … )`   – Internal helper that concatenates **already-processed**
//                                    constraint functions into one closure returning a 2-D array.
//
// ▸ **`impl_constraints_fn!( … )`** – Public, user-facing macro that defines a `struct` and
//                                implements `moors::ConstraintsFn` for it. The first
//                                argument is the `struct` name. Optionally list
//                                `ineq = [ ... ]`, `eq = [ ... ]`,
//                                `lower_bound = ...`, and/or `upper_bound = ...`.
//
//   ┌─────────────────────── Usage ─────────────────────────
//   │ // Defines `struct MyConstraints;` and
//   │ // `impl moors::ConstraintsFn for MyConstraints { ... }`
//   │ constraints_fn!(
//   │     MyConstraints,
//   │     ineq        = [g1, g2],          // inequality functions
//   │     eq          = [h1, h2],          // equality functions
//   │     lower_bound = 0.0,               // optional: lower_bound - genes ≤ 0
//   │     upper_bound = 5.0,               // optional: genes - upper_bound ≤ 0
//   │ );
//   │
//   │ // You can omit any combination of `ineq`, `eq`, `lower_bound`, or `upper_bound`:
//   │ constraints_fn!(MyOnlyEq, eq = [h1]);
//   │ constraints_fn!(MyBounds, lower_bound = -1.0, upper_bound = 1.0);
//   └───────────────────────────────────────────────────────
//
//   Each constraint function (`g1`, `h1`, etc.) must be `fn(&Array2<f64>) -> Array1<f64>`.
//   `lower_bound` and `upper_bound` must be `f64` literals.
// =============================================================================

/// Wrap a single constraint function as an **equality** (`|g(genes)| - ε ≤ 0`, ε = 1e-6).
///
/// # Internal Use
/// This macro is not for direct user invocation; end-users should use [`constraints_fn!`].
#[macro_export]
macro_rules! __eq_helper {
    ($f:path $(,)?) => {
        |genes: &ndarray::Array2<f64>| -> ndarray::Array1<f64> {
            $f(genes).mapv(|v| v.abs() - 1e-6)
        }
    };
}

/// Concatenate one or more already-wrapped constraint closures (inequalities or
/// equalities) into a single evaluator closure returning an Array2.
///
/// # Internal Use
/// Users should call [`constraints_fn!`] instead of this macro.
#[macro_export]
macro_rules! __constraints_helper {
    ($($c:expr),+ $(,)?) => {
        |genes: &ndarray::Array2<f64>| -> ndarray::Array2<f64> {
            // Evaluate each constraint to produce a column vector
            let cols = vec![ $( ($c)(genes) ),+ ];

            // Convert each 1-D result into a column view and concatenate horizontally
            let views: Vec<_> = cols.iter()
                .map(|v| v.view().insert_axis(ndarray::Axis(1)))
                .collect();

            ndarray::concatenate(ndarray::Axis(1), &views)
                .expect("Failed to concatenate constraints along axis 1")
        }
    };
}

/// Defines a `struct` and implements [`moors::ConstraintsFn`] for it.
///
/// # Syntax
///
/// ```ignore
/// constraints_fn!(
///     StructName,
///     ineq        = [g1, g2],          // zero or more inequality functions
///     eq          = [h1, h2],          // zero or more equality functions
///     lower_bound = f64_literal,       // optional lower bound value
///     upper_bound = f64_literal,       // optional upper bound value
/// );
/// ```
///
/// - The **first** argument is the name of the `struct` to generate.
/// - `ineq`, `eq`, `lower_bound`, and `upper_bound` are **all optional** and can
///   appear in any order after the struct name.
/// - Equality functions are wrapped as `|h(genes)| - ε` (ε=1e-6).
///
/// # Result
/// Generates:
/// ```ignore
/// pub struct StructName;
/// impl moors::ConstraintsFn for StructName {
///     type Dim = ndarray::Ix2;
///     fn call(&self, genes: &Array2<f64>) -> Array2<f64> { ... }
///     fn lower_bound(&self) -> Option<f64> { ... }
///     fn upper_bound(&self) -> Option<f64> { ... }
/// }
/// ```
#[macro_export]
macro_rules! impl_constraints_fn {
    (
        $name:ident
        $(, ineq        = [ $($ineq:path),* $(,)? ] )?
        $(, eq          = [ $($eq:path),*   $(,)? ] )?
        $(, lower_bound = $lb:expr )?
        $(, upper_bound = $ub:expr )?
        $(,)?
    ) => {
        #[derive(Debug, Clone, Copy)]
        pub struct $name;

        impl $crate::ConstraintsFn for $name {
            type Dim = ndarray::Ix2;

            fn call(&self, genes: &ndarray::Array2<f64>) -> ndarray::Array2<f64> {
                use ndarray::{concatenate, Axis};

                let mut mats: Vec<ndarray::Array2<f64>> = Vec::new();

                // Inequality functions
                $( mats.push($crate::__constraints_helper!($($ineq),*)(genes)); )?

                // Equality functions wrapped via __eq_helper!
                $( mats.push($crate::__constraints_helper!( $($crate::__eq_helper!($eq)),* )(genes)); )?

                // Optional lower bound: lower_bound - genes
                $( mats.push({ let lb_mat = genes.mapv(|_| $lb); lb_mat - genes }); )?

                // Optional upper bound: genes - upper_bound
                $( mats.push({ let ub_mat = genes.mapv(|_| $ub); genes - ub_mat }); )?

                if mats.is_empty() {
                    ndarray::Array2::zeros((genes.nrows(), 0))
                } else {
                    let views: Vec<_> = mats.iter().map(|m| m.view()).collect();
                    concatenate(Axis(1), &views)
                        .expect("Failed to concatenate constraints along axis 1")
                }
            }

            $( fn lower_bound(&self) -> Option<f64> { Some($lb) } )?
            $( fn upper_bound(&self) -> Option<f64> { Some($ub) } )?
        }
    };
}

#[cfg(test)]
mod tests {
    use ndarray::{Array1, Array2, Axis, array};

    use crate::ConstraintsFn;

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
        impl_constraints_fn!(MyConstr, ineq = [g1, g2]); // closure (genes) -> Array2
        let res = MyConstr.call(&genes); // 2 × 2

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
        impl_constraints_fn!(MyConstr, ineq = [g1, g2], eq = [g3]);
        let res = MyConstr.call(&genes); // 2 × 3

        // Build expected matrix by hand
        let mut exp = Array2::<f64>::zeros((2, 3));
        exp[[0, 0]] = 0.0; // g1 row0
        exp[[0, 1]] = -0.5; // g2 row0
        exp[[0, 2]] = -EPS; // |g3|-ε row0

        exp[[1, 0]] = 2.0; // g1 row1
        exp[[1, 1]] = 4.0; // g2 row1
        exp[[1, 2]] = 1.0 - EPS; // |g3|-ε row1

        assert_eq!(res, exp);
    }

    #[test]
    fn equalities_only() {
        let genes = array![[2.0, 2.0], [0.5, 1.5]]; // 2 × 2
        impl_constraints_fn!(EqOnly, eq = [g3]);
        let res = EqOnly.call(&genes); // 2 × 1

        const EPS: f64 = 1e-6;
        // |g3| - ε = |x - y| - ε
        let mut exp = Array2::<f64>::zeros((2, 1));
        exp[[0, 0]] = -EPS; // |2–2|−ε = 0−ε
        exp[[1, 0]] = 1.0 - EPS; // |0.5–1.5|−ε = 1−ε

        assert_eq!(res, exp);
    }

    #[test]
    fn lower_bound_only() {
        let genes = array![[1.0, 3.0], [0.0, 2.0]];
        impl_constraints_fn!(LowOnly, lower_bound = 2.0);
        let res = LowOnly.call(&genes); // 2 × 2

        // lower_bound - genes = 2 - genes
        let mut exp = Array2::<f64>::zeros((2, 2));
        exp[[0, 0]] = 2.0 - 1.0;
        exp[[0, 1]] = 2.0 - 3.0;
        exp[[1, 0]] = 2.0 - 0.0;
        exp[[1, 1]] = 2.0 - 2.0;
        assert_eq!(res, exp);
    }

    #[test]
    fn upper_bound_only() {
        let genes = array![[1.0, 3.0], [0.0, 2.0]];
        impl_constraints_fn!(UpOnly, upper_bound = 3.0);
        let res = UpOnly.call(&genes); // 2 × 2

        // genes - upper_bound = genes - 3
        let mut exp = Array2::<f64>::zeros((2, 2));
        exp[[0, 0]] = 1.0 - 3.0;
        exp[[0, 1]] = 3.0 - 3.0;
        exp[[1, 0]] = 0.0 - 3.0;
        exp[[1, 1]] = 2.0 - 3.0;
        assert_eq!(res, exp);
    }

    #[test]
    fn all_constraints_combined() {
        const EPS: f64 = 1e-6;
        let genes = array![[0.0, 1.0], [2.0, 1.0]];
        impl_constraints_fn!(
            AllC,
            ineq = [g1],
            eq = [g3],
            lower_bound = 1.0,
            upper_bound = 2.0
        );
        let res = AllC.call(&genes); // 2 × 4
        let mut exp = Array2::<f64>::zeros((2, 6));
        // row 0 (genes = [0.0, 1.0])
        exp[[0, 0]] = 0.0; // g1: 0+1-1
        exp[[0, 1]] = 1.0 - EPS; // |0-1|-ε
        exp[[0, 2]] = 1.0 - 0.0; // lower_bound - gene0
        exp[[0, 3]] = 1.0 - 1.0; // lower_bound - gene1
        exp[[0, 4]] = 0.0 - 2.0; // gene0 - upper_bound
        exp[[0, 5]] = 1.0 - 2.0; // gene1 - upper_bound

        // row 1 (genes = [2.0, 1.0])
        exp[[1, 0]] = 2.0; // g1: 2+1-1
        exp[[1, 1]] = 1.0 - EPS; // |2-1|-ε
        exp[[1, 2]] = 1.0 - 2.0; // lower_bound - gene0
        exp[[1, 3]] = 1.0 - 1.0; // lower_bound - gene1
        exp[[1, 4]] = 2.0 - 2.0; // gene0 - upper_bound
        exp[[1, 5]] = 1.0 - 2.0; // gene1 - upper_bound
        assert_eq!(res, exp);

        assert_eq!(res, exp);
    }
}
