//! # `surrogate` – Surrogate-Assisted Optimization
//!
//! A **surrogate model** is a cheap approximation of an expensive fitness
//! function, trained online from the real evaluations already computed by
//! the genetic algorithm. It lets a [`SurrogateModel`] implementation screen
//! candidates before spending a real (potentially costly) evaluation on them,
//! following the *Surrogate-Assisted Evolutionary Algorithm* (SAEA) paradigm.
//!
//! See, e.g.:
//! * Jin, Y. (2011). *Surrogate-assisted evolutionary computation: Recent
//!   advances and future challenges*. Swarm and Evolutionary Computation, 1(2), 61-70.
//! * Jones, D. R., Schonlau, M., & Welch, W. J. (1998). *Efficient Global
//!   Optimization of Expensive Black-Box Functions*. Journal of Global
//!   Optimization, 13(4), 455-492.
use ndarray::Array2;

/// A trainable, cheap approximation of a (potentially expensive) fitness function.
///
/// Implementors own their training archive: `update` registers newly
/// available real evaluations and internally decides how much history to
/// keep and when to retrain. The genetic algorithm loop only needs to know
/// which points it evaluated for real; it has no say in archive management.
pub trait SurrogateModel {
    /// Registers new real evaluations (`genes`: N x num_vars, `fitness`: N x
    /// num_objectives) and, depending on the implementation's own policy,
    /// retrains itself.
    fn update(&mut self, genes: &Array2<f64>, fitness: &Array2<f64>);

    /// Predicts fitness values for the given genes. Output is `N x num_objectives`.
    fn predict(&self, genes: &Array2<f64>) -> Array2<f64>;

    /// Whether the model has been trained at least once and `predict` can be
    /// called safely. Implementors that retrain lazily (e.g. only every N
    /// `update` calls) must override this so callers can keep falling back
    /// to the real function until then.
    fn is_fitted(&self) -> bool {
        true
    }

    /// Predicts the standard deviation of the prediction, when the model
    /// supports uncertainty estimates (e.g. Gaussian Processes). Used for
    /// infill criteria such as Expected Improvement.
    fn predict_std(&self, _genes: &Array2<f64>) -> Option<Array2<f64>> {
        None
    }
}

#[cfg(feature = "surrogate")]
mod gaussian_process;
#[cfg(feature = "surrogate")]
pub use gaussian_process::GaussianProcessSurrogate;

mod fitness_fn;
pub use fitness_fn::{SurrogateConfig, SurrogateFitnessFn};
