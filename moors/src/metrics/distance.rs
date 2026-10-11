//! Generational Distance (GD) and Inverted Generational Distance (IGD).
//!
//! Both compare an obtained front `A` against a reference front `Z`
//! (typically the true/known Pareto front, or a very dense approximation of
//! it) using nearest-neighbor Euclidean distances in objective space.
//!
//! **Generational Distance** (Van Veldhuizen & Lamont, 1998) measures how far
//! `A` is from `Z`, i.e. *convergence*:
//!
//! ```text
//! GD(A, Z) = sqrt( Σ_{a∈A} d(a, Z)² ) / |A|
//! ```
//!
//! where `d(a, Z) = min_{z∈Z} ||a - z||₂`. `GD = 0` iff every point of `A`
//! lies on `Z`. GD alone does not penalize a poor *spread*: a single point on
//! `Z` gives `GD = 0` regardless of how much of the front is covered.
//!
//! **Inverted Generational Distance** (Coello Coello & Cortés, 2005) swaps
//! the roles, measuring both convergence *and* coverage/diversity:
//!
//! ```text
//! IGD(A, Z) = Σ_{z∈Z} d(z, A) / |Z|
//! ```
//!
//! where `d(z, A) = min_{a∈A} ||z - a||₂`. Because every reference point
//! must be close to *some* solution in `A`, `IGD` penalizes gaps in the
//! obtained front, unlike `GD`.
//!
//! Lower is better for both.

use ndarray::{Array1, Array2};

use crate::helpers::linalg::cross_euclidean_distances;

/// For each row of `data`, the Euclidean distance to its nearest row in `reference`.
fn nearest_distances(data: &Array2<f64>, reference: &Array2<f64>) -> Array1<f64> {
    let sq_dist = cross_euclidean_distances(data, reference);
    let n = data.nrows();
    let m = reference.nrows();
    Array1::from_shape_fn(n, |i| {
        (0..m)
            .map(|j| sq_dist.get(i, j))
            .fold(f64::INFINITY, |acc, &d| acc.min(d))
            .max(0.0)
            .sqrt()
    })
}

/// Generational Distance of `front` with respect to `reference_front`.
///
/// # Panics
/// Panics if `front` or `reference_front` is empty, or if their number of
/// objectives (columns) differ.
pub fn generational_distance(front: &Array2<f64>, reference_front: &Array2<f64>) -> f64 {
    assert!(front.nrows() > 0, "front must not be empty");
    assert!(
        reference_front.nrows() > 0,
        "reference_front must not be empty"
    );
    assert_eq!(
        front.ncols(),
        reference_front.ncols(),
        "front and reference_front must have the same number of objectives"
    );
    let d = nearest_distances(front, reference_front);
    (d.iter().map(|x| x * x).sum::<f64>()).sqrt() / front.nrows() as f64
}

/// Inverted Generational Distance of `front` with respect to `reference_front`.
///
/// # Panics
/// Panics if `front` or `reference_front` is empty, or if their number of
/// objectives (columns) differ.
pub fn inverted_generational_distance(front: &Array2<f64>, reference_front: &Array2<f64>) -> f64 {
    assert!(front.nrows() > 0, "front must not be empty");
    assert!(
        reference_front.nrows() > 0,
        "reference_front must not be empty"
    );
    assert_eq!(
        front.ncols(),
        reference_front.ncols(),
        "front and reference_front must have the same number of objectives"
    );
    let d = nearest_distances(reference_front, front);
    d.sum() / reference_front.nrows() as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() <= eps
    }

    #[test]
    fn gd_zero_when_front_is_subset_of_reference() {
        let reference = array![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]];
        let front = array![[0.0, 0.0], [2.0, 2.0]];
        assert!(approx_eq(
            generational_distance(&front, &reference),
            0.0,
            1e-12
        ));
    }

    #[test]
    fn gd_single_offset_point() {
        // front = {(1,0)}, nearest reference point is (0,0) at distance 1.
        let reference = array![[0.0, 0.0]];
        let front = array![[1.0, 0.0]];
        assert!(approx_eq(
            generational_distance(&front, &reference),
            1.0,
            1e-12
        ));
    }

    #[test]
    fn igd_penalizes_missing_coverage() {
        // reference has 2 points, front only covers one of them exactly.
        let reference = array![[0.0, 0.0], [10.0, 10.0]];
        let front = array![[0.0, 0.0]];
        let expected = (10f64.powi(2) * 2.0).sqrt() / 2.0; // dist((10,10),(0,0))/2
        assert!(approx_eq(
            inverted_generational_distance(&front, &reference),
            expected,
            1e-9
        ));
    }
}
