//! Spacing (SP) diversity indicator (Schott, 1995).
//!
//! Measures how *uniformly* the points of a front are distributed, without
//! requiring any reference front:
//!
//! ```text
//! SP(A) = sqrt( Σ_{i=1}^{n} (d̄ - d_i)² / (n - 1) )
//! ```
//!
//! where `d_i = min_{j≠i} Σ_{k=1}^{M} |f_k(i) - f_k(j)|` is the L1 (Manhattan)
//! distance from solution `i` to its nearest neighbor in objective space, and
//! `d̄` is the mean of all `d_i`. `SP = 0` means all nearest-neighbor
//! distances are equal (perfectly uniform spread); larger `SP` indicates
//! clustering / uneven coverage along the front.

use ndarray::Array2;

/// Spacing of `front` (shape `(n, m)`), `n ≥ 2`.
///
/// # Panics
/// Panics if `front` has fewer than 2 rows.
pub fn spacing(front: &Array2<f64>) -> f64 {
    let n = front.nrows();
    assert!(n > 1, "spacing requires at least 2 points");

    let nearest: Vec<f64> = (0..n)
        .map(|i| {
            (0..n)
                .filter(|&j| j != i)
                .map(|j| {
                    front
                        .row(i)
                        .iter()
                        .zip(front.row(j).iter())
                        .map(|(a, b)| (a - b).abs())
                        .sum::<f64>()
                })
                .fold(f64::INFINITY, f64::min)
        })
        .collect();

    let mean = nearest.iter().sum::<f64>() / n as f64;
    let variance = nearest.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / (n as f64 - 1.0);
    variance.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() <= eps
    }

    #[test]
    fn perfectly_uniform_front_has_zero_spacing() {
        // Evenly spaced points on a line f1+f2=const: nearest-neighbor L1 distance is 2 for all.
        let front = array![[0.0, 4.0], [1.0, 3.0], [2.0, 2.0], [3.0, 1.0], [4.0, 0.0]];
        assert!(approx_eq(spacing(&front), 0.0, 1e-9));
    }

    #[test]
    fn uneven_front_has_positive_spacing() {
        let front = array![[0.0, 4.0], [0.1, 3.9], [2.0, 2.0], [4.0, 0.0]];
        assert!(spacing(&front) > 0.0);
    }

    #[test]
    #[should_panic]
    fn single_point_panics() {
        let front = array![[0.0, 0.0]];
        spacing(&front);
    }
}
