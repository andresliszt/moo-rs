//! # `metrics` – Quality Indicators for Multi-Objective Fronts
//!
//! Standalone functions to evaluate the quality of a Pareto front, whether
//! produced by a *moors* algorithm or coming from anywhere else. All
//! functions assume **minimization** of every objective and expect
//! `(n_individuals, n_objectives)` matrices — the same convention used by
//! [`crate::genetic::Population::fitness`].
//!
//! | Metric | Function | Reference front? | Reference point? |
//! |--------|----------|:---:|:---:|
//! | Hypervolume (HV) | [`hypervolume`] | no | yes |
//! | Generational Distance (GD) | [`generational_distance`] | yes | no |
//! | Inverted Generational Distance (IGD) | [`inverted_generational_distance`] | yes | no |
//! | Spacing (SP) | [`spacing`] | no | no |
//! | Spread (Δ, 2 objectives only) | [`spread`] | yes | no |
//!
//! See the User Guide's "Quality Metrics" section for the mathematical
//! definitions and interpretation of each indicator.

mod distance;
mod hypervolume;
mod spacing;
mod spread;

pub use distance::{generational_distance, inverted_generational_distance};
pub use hypervolume::hypervolume;
pub use spacing::spacing;
pub use spread::spread;
