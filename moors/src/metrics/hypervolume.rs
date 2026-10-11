//! Hypervolume (HV) quality indicator.
//!
//! For a minimization front `A` and a reference point `r` dominated by every
//! point of `A`, the hypervolume is the Lebesgue measure of the region
//! jointly dominated by `A` and bounded above by `r`:
//!
//! ```text
//! HV(A, r) = Λ( ⋃_{a∈A} [a, r] )
//! ```
//!
//! where `[a, r]` is the axis-aligned hyper-rectangle with corners `a` and
//! `r`, and `Λ` is the Lebesgue measure. HV is the only quality indicator
//! that is strictly monotonic with respect to Pareto dominance (a larger HV
//! never corresponds to a worse front) which is why it's the most widely
//! used single-number MOO quality metric.
//!
//! Computed here with the recursive dimension-slicing algorithm ("Hypervolume
//! by Slicing Objectives", While, Hingston, Barone & Huband, 2006).

use ndarray::{Array1, Array2, ArrayView1};

use crate::non_dominated_sorting::dominates_weak;

/// Inserts `point` into the non-dominated front `front`, dropping it if some
/// existing point already dominates (or equals) it, and removing any
/// existing point that `point` dominates.
fn insert_non_dominated(front: &mut Vec<Vec<f64>>, point: Vec<f64>) {
    let point_view = ArrayView1::from(point.as_slice());
    if front
        .iter()
        .any(|q| dominates_weak(ArrayView1::from(q.as_slice()), point_view))
    {
        return;
    }
    front.retain(|q| !dominates_weak(point_view, ArrayView1::from(q.as_slice())));
    front.push(point);
}

fn hv_recursive(points: &[Vec<f64>], reference: &[f64]) -> f64 {
    if points.is_empty() {
        return 0.0;
    }
    let m = reference.len();
    if m == 1 {
        let best = points.iter().map(|p| p[0]).fold(f64::INFINITY, f64::min);
        return (reference[0] - best).max(0.0);
    }

    let mut order: Vec<usize> = (0..points.len()).collect();
    order.sort_by(|&a, &b| points[a][m - 1].partial_cmp(&points[b][m - 1]).unwrap());

    let mut volume = 0.0;
    let mut front_proj: Vec<Vec<f64>> = Vec::new();
    for (k, &idx) in order.iter().enumerate() {
        let p = &points[idx];
        insert_non_dominated(&mut front_proj, p[..m - 1].to_vec());

        let height = if k + 1 < order.len() {
            points[order[k + 1]][m - 1] - p[m - 1]
        } else {
            reference[m - 1] - p[m - 1]
        };
        if height > 0.0 {
            volume += height * hv_recursive(&front_proj, &reference[..m - 1]);
        }
    }
    volume
}

/// Computes the hypervolume of `front` (shape `(n, m)`) with respect to
/// `reference` (shape `(m,)`), assuming minimization.
///
/// Points of `front` that exceed `reference` in some objective are clamped
/// to `reference` in that objective (i.e. they contribute no extra volume
/// in that dimension), mirroring how a single point's contribution
/// `∏_d (r_d - x_d)` is clamped at `0` from below.
///
/// # Panics
/// Panics if `front.ncols() != reference.len()`.
pub fn hypervolume(front: &Array2<f64>, reference: &Array1<f64>) -> f64 {
    assert_eq!(
        front.ncols(),
        reference.len(),
        "front and reference must have the same number of objectives"
    );
    if front.nrows() == 0 {
        return 0.0;
    }
    let points: Vec<Vec<f64>> = front
        .outer_iter()
        .map(|row| {
            row.iter()
                .zip(reference.iter())
                .map(|(&x, &r)| x.min(r))
                .collect()
        })
        .collect();
    hv_recursive(
        &points,
        reference.as_slice().expect("reference must be contiguous"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() <= eps
    }

    #[test]
    fn single_point() {
        let front = array![[1.0, 1.0]];
        let r = array![3.0, 3.0];
        assert!(approx_eq(hypervolume(&front, &r), 4.0, 1e-12));
    }

    #[test]
    fn dominated_point_contributes_nothing() {
        // a=(1,1) dominates b=(2,2) => HV({a,b}) == HV({a})
        let front = array![[1.0, 1.0], [2.0, 2.0]];
        let r = array![3.0, 3.0];
        assert!(approx_eq(hypervolume(&front, &r), 4.0, 1e-12));
    }

    #[test]
    fn two_non_dominated_points() {
        // a=(1,3), b=(3,1), r=(4,4): HV(a)=3, HV(b)=3, intersection=1 => union=5
        let front = array![[1.0, 3.0], [3.0, 1.0]];
        let r = array![4.0, 4.0];
        assert!(approx_eq(hypervolume(&front, &r), 5.0, 1e-12));
    }

    #[test]
    fn three_dimensional_single_point() {
        let front = array![[1.0, 1.0, 1.0]];
        let r = array![2.0, 2.0, 2.0];
        assert!(approx_eq(hypervolume(&front, &r), 1.0, 1e-12));
    }

    #[test]
    fn empty_front_is_zero() {
        let front = Array2::<f64>::zeros((0, 2));
        let r = array![1.0, 1.0];
        assert_eq!(hypervolume(&front, &r), 0.0);
    }

    #[test]
    fn point_outside_reference_is_clamped() {
        // Point worse than reference in the second objective contributes 0 there.
        let front = array![[1.0, 5.0]];
        let r = array![3.0, 3.0];
        assert!(approx_eq(hypervolume(&front, &r), 0.0, 1e-12));
    }
}
