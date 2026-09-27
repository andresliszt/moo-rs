//! Merit scoring for particle-swarm algorithms.
//!
//! A [`MeritOperator`] turns a fully evaluated `Population` into a single
//! scalar per individual (higher is always better). This score is what a
//! particle-swarm algorithm uses to decide whether a new position becomes a
//! particle's `pbest`, who the swarm's `gbest` is, and — more generally — how
//! "elitism" is expressed without a GA-style survival operator.
use ndarray::Array1;

use crate::genetic::{D12, PopulationMOO};

/// Computes a scalar merit per individual of a population: **higher is
/// always better**, regardless of how many objectives/constraints exist.
pub trait MeritOperator: std::fmt::Debug {
    fn compute<ConstrDim>(&self, population: &PopulationMOO<ConstrDim>) -> Array1<f64>
    where
        ConstrDim: D12;
}
