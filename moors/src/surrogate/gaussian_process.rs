//! Gaussian Process (Kriging) surrogate, one independent process per objective.
//!
//! Kernel and prior are fitted to the data automatically, following
//! Jones, Schonlau & Welch (1998), "Efficient Global Optimization of
//! Expensive Black-Box Functions".
use ndarray::{Array1, Array2, Axis, concatenate, s};

use friedrich::gaussian_process::GaussianProcess;
use friedrich::kernel::Gaussian;
use friedrich::prior::ConstantPrior;

use super::SurrogateModel;

type Gp = GaussianProcess<Gaussian, ConstantPrior>;

/// One Gaussian Process per objective, owning a bounded training archive.
///
/// friedrich works with `Vec<Vec<f64>>` / `Vec<f64>` inputs, so genes/fitness
/// are converted to/from `ndarray` at the boundary. The archive is capped at
/// `max_archive_size` (oldest samples are evicted first, FIFO) since GP
/// training is `O(n^3)` and cannot be allowed to grow unbounded, and it is
/// only refit every `retrain_every` calls to `update`.
pub struct GaussianProcessSurrogate {
    models: Vec<Gp>,
    archive_genes: Array2<f64>,
    archive_fitness: Array2<f64>,
    max_archive_size: usize,
    retrain_every: usize,
    updates_since_fit: usize,
}

impl GaussianProcessSurrogate {
    /// `max_archive_size`: cap on the number of real evaluations kept for
    /// training (oldest evicted first). `retrain_every`: number of `update`
    /// calls between refits.
    pub fn new(max_archive_size: usize, retrain_every: usize) -> Self {
        Self {
            models: Vec::new(),
            archive_genes: Array2::zeros((0, 0)),
            archive_fitness: Array2::zeros((0, 0)),
            max_archive_size,
            retrain_every: retrain_every.max(1),
            updates_since_fit: 0,
        }
    }

    fn refit(&mut self) {
        let inputs = rows_to_vecvec(&self.archive_genes);
        self.models = (0..self.archive_fitness.ncols())
            .map(|j| {
                let outputs: Vec<f64> = self.archive_fitness.column(j).to_vec();
                Gp::default(inputs.clone(), outputs)
            })
            .collect();
    }
}

impl Default for GaussianProcessSurrogate {
    /// Keeps at most 200 real evaluations and retrains on every `update` call.
    fn default() -> Self {
        Self::new(200, 1)
    }
}

fn rows_to_vecvec(arr: &Array2<f64>) -> Vec<Vec<f64>> {
    arr.axis_iter(Axis(0)).map(|row| row.to_vec()).collect()
}

impl SurrogateModel for GaussianProcessSurrogate {
    fn update(&mut self, genes: &Array2<f64>, fitness: &Array2<f64>) {
        self.archive_genes = if self.archive_genes.ncols() == 0 {
            genes.to_owned()
        } else {
            concatenate(Axis(0), &[self.archive_genes.view(), genes.view()])
                .expect("genes archive and new genes must share the same number of columns")
        };
        self.archive_fitness = if self.archive_fitness.ncols() == 0 {
            fitness.to_owned()
        } else {
            concatenate(Axis(0), &[self.archive_fitness.view(), fitness.view()])
                .expect("fitness archive and new fitness must share the same number of columns")
        };

        let n = self.archive_genes.nrows();
        if n > self.max_archive_size {
            let start = n - self.max_archive_size;
            self.archive_genes = self.archive_genes.slice(s![start.., ..]).to_owned();
            self.archive_fitness = self.archive_fitness.slice(s![start.., ..]).to_owned();
        }

        self.updates_since_fit += 1;
        if self.updates_since_fit >= self.retrain_every {
            self.refit();
            self.updates_since_fit = 0;
        }
    }

    fn predict(&self, genes: &Array2<f64>) -> Array2<f64> {
        let inputs = rows_to_vecvec(genes);
        let mut out = Array2::<f64>::zeros((inputs.len(), self.models.len()));
        for (j, model) in self.models.iter().enumerate() {
            let preds: Vec<f64> = model.predict(&inputs);
            out.column_mut(j).assign(&Array1::from_vec(preds));
        }
        out
    }

    fn is_fitted(&self) -> bool {
        !self.models.is_empty()
    }

    fn predict_std(&self, genes: &Array2<f64>) -> Option<Array2<f64>> {
        if self.models.is_empty() {
            return None;
        }
        let inputs = rows_to_vecvec(genes);
        let mut out = Array2::<f64>::zeros((inputs.len(), self.models.len()));
        for (j, model) in self.models.iter().enumerate() {
            let variance: Vec<f64> = model.predict_variance(&inputs);
            let std: Vec<f64> = variance.into_iter().map(f64::sqrt).collect();
            out.column_mut(j).assign(&Array1::from_vec(std));
        }
        Some(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    /// GP trained on a 1-D sphere should extrapolate reasonably close to
    /// the true value at an interpolated (non-training) point.
    #[test]
    fn updates_and_predicts_sphere() {
        let genes = array![[0.0], [1.0], [2.0], [3.0], [4.0]];
        let fitness = array![[0.0], [1.0], [4.0], [9.0], [16.0]];

        let mut model = GaussianProcessSurrogate::new(200, 1);
        model.update(&genes, &fitness);

        let test_point = array![[2.5]];
        let prediction = model.predict(&test_point);

        assert_eq!(prediction.shape(), &[1, 1]);
        // True value is 6.25; interpolated GP prediction should be in the same ballpark.
        assert!(
            (prediction[[0, 0]] - 6.25).abs() < 2.0,
            "prediction {} too far from expected 6.25",
            prediction[[0, 0]]
        );
    }

    #[test]
    fn predicts_std_when_available() {
        let genes = array![[0.0], [1.0], [2.0]];
        let fitness = array![[0.0], [1.0], [4.0]];

        let mut model = GaussianProcessSurrogate::new(200, 1);
        model.update(&genes, &fitness);

        let std = model.predict_std(&array![[1.5]]);
        assert!(std.is_some());
        assert_eq!(std.unwrap().shape(), &[1, 1]);
    }

    #[test]
    fn handles_multiple_objectives() {
        let genes = array![[0.0], [1.0], [2.0], [3.0]];
        let fitness = array![[0.0, 0.0], [1.0, -1.0], [4.0, -2.0], [9.0, -3.0]];

        let mut model = GaussianProcessSurrogate::new(200, 1);
        model.update(&genes, &fitness);

        let prediction = model.predict(&array![[1.5]]);
        assert_eq!(prediction.shape(), &[1, 2]);
    }

    #[test]
    fn archive_is_capped_and_evicts_oldest() {
        let mut model = GaussianProcessSurrogate::new(3, 1);

        model.update(&array![[0.0], [1.0]], &array![[0.0], [1.0]]);
        assert_eq!(model.archive_genes.nrows(), 2);

        model.update(&array![[2.0], [3.0]], &array![[4.0], [9.0]]);
        // 4 rows observed total, capped at 3: the oldest (gene 0.0) must be evicted.
        assert_eq!(model.archive_genes.nrows(), 3);
        assert_eq!(model.archive_genes[[0, 0]], 1.0);
    }

    #[test]
    fn does_not_retrain_before_retrain_every_updates() {
        let mut model = GaussianProcessSurrogate::new(200, 3);

        model.update(&array![[0.0], [1.0]], &array![[0.0], [1.0]]);
        assert!(
            model.models.is_empty(),
            "should not retrain before retrain_every updates"
        );

        model.update(&array![[2.0]], &array![[4.0]]);
        model.update(&array![[3.0]], &array![[9.0]]);
        assert!(
            !model.models.is_empty(),
            "should retrain on the retrain_every-th update"
        );
    }
}
