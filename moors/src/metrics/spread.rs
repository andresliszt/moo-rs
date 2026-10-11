//! Spread (Δ) diversity indicator for bi-objective fronts (Deb, Pratap,
//! Agarwal & Meyarivan, 2002 — the original NSGA-II paper).
//!
//! Combines extent and uniformity of a 2-objective front `A` against a
//! reference front `Z`:
//!
//! ```text
//! Δ(A, Z) = (d_f + d_l + Σ_{i=1}^{n-1} |d_i - d̄|) / (d_f + d_l + (n-1)·d̄)
//! ```
//!
//! where, sorting `A` by its first objective, `d_i` is the Euclidean
//! distance between consecutive solutions `i` and `i+1`, `d̄` is their mean,
//! and `d_f`/`d_l` are the Euclidean distances between the extreme
//! (boundary) solutions of `A` and the extreme solutions of `Z`. `Δ = 0`
//! for a perfectly uniform front that also reaches both extremes of `Z`;
//! larger `Δ` indicates gaps/clustering and/or a front that doesn't reach
//! the true extremes.
//!
//! This metric is only defined for 2-objective fronts: "consecutive
//! solutions along the front" is unambiguous only when sorting by a single
//! objective.

use ndarray::{Array2, ArrayView1};

fn euclidean(a: ArrayView1<'_, f64>, b: ArrayView1<'_, f64>) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

fn sorted_by_first_objective(front: &Array2<f64>) -> Vec<usize> {
    let mut order: Vec<usize> = (0..front.nrows()).collect();
    order.sort_by(|&a, &b| front[[a, 0]].partial_cmp(&front[[b, 0]]).unwrap());
    order
}

/// Spread (Δ) of `front` with respect to `reference_front`, both 2-objective.
///
/// # Panics
/// Panics if `front`/`reference_front` don't have exactly 2 objectives, or
/// if `front` has fewer than 2 points.
pub fn spread(front: &Array2<f64>, reference_front: &Array2<f64>) -> f64 {
    assert_eq!(
        front.ncols(),
        2,
        "spread (Δ) is only defined for 2-objective fronts"
    );
    assert_eq!(
        reference_front.ncols(),
        2,
        "spread (Δ) is only defined for 2-objective fronts"
    );
    let n = front.nrows();
    assert!(n > 1, "spread requires at least 2 points in front");
    assert!(
        reference_front.nrows() > 0,
        "reference_front must not be empty"
    );

    let order = sorted_by_first_objective(front);
    let consecutive: Vec<f64> = order
        .windows(2)
        .map(|w| euclidean(front.row(w[0]), front.row(w[1])))
        .collect();
    let mean_d = consecutive.iter().sum::<f64>() / consecutive.len() as f64;

    let ref_order = sorted_by_first_objective(reference_front);
    let ref_first = reference_front.row(ref_order[0]);
    let ref_last = reference_front.row(*ref_order.last().unwrap());

    let d_f = euclidean(ref_first, front.row(order[0]));
    let d_l = euclidean(ref_last, front.row(*order.last().unwrap()));

    let numerator = d_f + d_l + consecutive.iter().map(|d| (d - mean_d).abs()).sum::<f64>();
    let denominator = d_f + d_l + consecutive.len() as f64 * mean_d;
    numerator / denominator
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() <= eps
    }

    #[test]
    fn perfectly_uniform_front_matching_extremes_has_zero_spread() {
        let front = array![[0.0, 4.0], [1.0, 3.0], [2.0, 2.0], [3.0, 1.0], [4.0, 0.0]];
        let reference = front.clone();
        assert!(approx_eq(spread(&front, &reference), 0.0, 1e-9));
    }

    #[test]
    fn uneven_front_has_positive_spread() {
        let front = array![[0.0, 4.0], [0.1, 3.9], [3.0, 1.0], [4.0, 0.0]];
        let reference = array![[0.0, 4.0], [4.0, 0.0]];
        assert!(spread(&front, &reference) > 0.0);
    }

    #[test]
    #[should_panic]
    fn three_objectives_panics() {
        let front = array![[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]];
        let reference = front.clone();
        spread(&front, &reference);
    }
}
