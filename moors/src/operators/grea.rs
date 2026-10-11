//! Grid math shared by GrEA's mating selection and environmental selection.
//!
//! Implements the hyper-grid construction described in
//! **S. Yang, M. Li, X. Liu & J. Zheng, "A Grid-Based Evolutionary Algorithm
//! for Many-Objective Optimization", IEEE Transactions on Evolutionary
//! Computation, 17(5):721-736, 2013** (Eq. 6-8): `div` divisions per
//! objective, each individual mapped to an integer grid coordinate (`Gk`).
//!
//! [`GridBounds`] and [`grid_difference`] are the only pieces genuinely
//! needed by *both*
//! [`fitness_assignment`](crate::operators::selection::moo::grea::fitness_assignment)
//! (mating selection, Algorithm 2) and
//! [`grid_select`](crate::operators::survival::moo::grea::grid_select)
//! (environmental selection, Algorithms 3/5/6/7) — everything else (`GR`,
//! `GCD`, `GCPD` assembly, the greedy archive loop) is specific to one of
//! the two and lives alongside its single consumer instead of here.
use ndarray::{Array1, Array2, ArrayView1};
use ndarray_stats::QuantileExt;

/// Precomputed lower bounds (`lb`) and cell widths (`d`) per objective,
/// following Eq. 6-8 of the GrEA paper.
pub(crate) struct GridBounds {
    lb: Array1<f64>,
    d: Array1<f64>,
    div: usize,
}

impl GridBounds {
    pub(crate) fn new(fitness: &Array2<f64>, div: usize) -> Self {
        assert!(div >= 1, "GrEA: `div` (grid divisions) must be >= 1");
        let m = fitness.ncols();
        let mut lb = Array1::<f64>::zeros(m);
        let mut d = Array1::<f64>::zeros(m);
        let div_f = div as f64;

        for k in 0..m {
            let col = fitness.column(k);
            let min_k = *col
                .min()
                .expect("GrEA: objective column must be non-empty and NaN-free");
            let max_k = *col
                .max()
                .expect("GrEA: objective column must be non-empty and NaN-free");
            let range = max_k - min_k;

            if range <= f64::EPSILON {
                // Degenerate objective (all individuals share the same value):
                // every individual falls in grid coordinate 0.
                lb[k] = min_k;
                d[k] = 1.0;
            } else {
                let half_step = range / (2.0 * div_f);
                lb[k] = min_k - half_step;
                let ub_k = max_k + half_step;
                d[k] = (ub_k - lb[k]) / div_f;
            }
        }

        Self { lb, d, div }
    }

    /// Grid coordinates `Gk(x)` for every individual (Eq. 6).
    pub(crate) fn coordinates(&self, fitness: &Array2<f64>) -> Array2<usize> {
        let n = fitness.nrows();
        let m = fitness.ncols();
        let max_coord = self.div.saturating_sub(1);

        Array2::from_shape_fn((n, m), |(i, k)| {
            let raw = ((fitness[[i, k]] - self.lb[k]) / self.d[k]).floor();
            if !raw.is_finite() || raw < 0.0 {
                0
            } else if raw as usize > max_coord {
                max_coord
            } else {
                raw as usize
            }
        })
    }

    /// Grid coordinate point distance `GCPD(x)` (Eq. 11).
    pub(crate) fn gcpd_of(&self, fitness_row: ArrayView1<f64>, grid_row: ArrayView1<usize>) -> f64 {
        let mut sum_sq = 0.0;
        for k in 0..fitness_row.len() {
            let lower_corner = self.lb[k] + grid_row[k] as f64 * self.d[k];
            let normalized = (fitness_row[k] - lower_corner) / self.d[k];
            sum_sq += normalized * normalized;
        }
        sum_sq.sqrt()
    }
}

/// Grid difference `GD(a, b)`: L1 distance between grid coordinates (Eq. 8).
pub(crate) fn grid_difference(a: ArrayView1<usize>, b: ArrayView1<usize>) -> usize {
    a.iter().zip(b.iter()).map(|(&x, &y)| x.abs_diff(y)).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_grid_difference() {
        let a = array![0usize, 3, 2];
        let b = array![1usize, 1, 2];
        assert_eq!(grid_difference(a.view(), b.view()), 3);
    }
}
