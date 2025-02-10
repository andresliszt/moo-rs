// lib.rs

extern crate core;

pub mod evaluator;
pub mod genetic;
pub mod prelude;

mod algorithms;
pub mod diversity_metrics;
mod helpers;
pub mod non_dominated_sorting;
mod operators;

use pyo3::prelude::*;

pub use algorithms::nsga2::PyNsga2;
pub use algorithms::nsga3::PyNsga3;
pub use algorithms::py_errors::NoFeasibleIndividualsError;
pub use algorithms::rnsga2::PyRNsga2;
pub use helpers::duplicates::{PyCloseDuplicatesCleaner, PyExactDuplicatesCleaner};
pub use operators::py_operators::{
    PyBitFlipMutation, PyExponentialCrossover, PyGaussianMutation, PyOrderCrossover,
    PyPermutationSampling, PyRandomSamplingBinary, PyRandomSamplingFloat, PyRandomSamplingInt,
    PySimulatedBinaryCrossover, PySinglePointBinaryCrossover, PySwapMutation,
    PyUniformBinaryCrossover,
};

/// Root module `pymoors` that includes all classes.
#[pymodule]
fn _pymoors(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Add classes from algorithms
    m.add_class::<PyNsga2>()?;
    m.add_class::<PyNsga3>()?;
    m.add_class::<PyRNsga2>()?;

    // Add classes from operators
    m.add_class::<PyBitFlipMutation>()?;
    m.add_class::<PySwapMutation>()?;
    m.add_class::<PyGaussianMutation>()?;
    m.add_class::<PyRandomSamplingBinary>()?;
    m.add_class::<PyRandomSamplingFloat>()?;
    m.add_class::<PyRandomSamplingInt>()?;
    m.add_class::<PyPermutationSampling>()?;
    m.add_class::<PySinglePointBinaryCrossover>()?;
    m.add_class::<PyUniformBinaryCrossover>()?;
    m.add_class::<PyOrderCrossover>()?;
    m.add_class::<PyExponentialCrossover>()?;
    m.add_class::<PyExactDuplicatesCleaner>()?;
    m.add_class::<PyCloseDuplicatesCleaner>()?;
    m.add_class::<PySimulatedBinaryCrossover>()?;
    // Py Errors
    m.add(
        "NoFeasibleIndividualsError",
        _py.get_type::<NoFeasibleIndividualsError>(),
    )?;

    Ok(())
}
