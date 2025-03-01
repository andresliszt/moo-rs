// lib.rs
#![cfg_attr(all(coverage_nightly, test), feature(coverage_attribute))]
extern crate core;

mod evaluator;
mod genetic;

mod algorithms;
mod duplicates;
mod helpers;
pub mod non_dominated_sorting;
mod operators;
mod random;

use pyo3::prelude::*;

pub use algorithms::agemoea::AgeMoea;
pub use algorithms::nsga2::Nsga2;
pub use algorithms::nsga3::Nsga3;
pub use algorithms::py_errors::InvalidParameterError;
pub use algorithms::py_errors::NoFeasibleIndividualsError;
pub use algorithms::rnsga2::Rnsga2;
pub use duplicates::{PyCloseDuplicatesCleaner, PyExactDuplicatesCleaner};
pub use operators::py_operators::{
    PyBitFlipMutation, PyDisplacementMutation, PyExponentialCrossover, PyGaussianMutation,
    PyOrderCrossover, PyPermutationSampling, PyRandomSamplingBinary, PyRandomSamplingFloat,
    PyRandomSamplingInt, PyScrambleMutation, PySimulatedBinaryCrossover,
    PySinglePointBinaryCrossover, PySwapMutation, PyUniformBinaryCrossover,
};

/// Root module `pymoors` that includes all classes.
#[pymodule]
fn _pymoors(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Add classes from algorithms
    m.add_class::<Nsga2>()?;
    m.add_class::<Nsga3>()?;
    m.add_class::<Rnsga2>()?;
    // m.add_class::<AgeMoea>()?; TODO: Enable it once survival issues are fixed

    // Add classes from operators
    m.add_class::<PyBitFlipMutation>()?;
    m.add_class::<PySwapMutation>()?;
    m.add_class::<PyGaussianMutation>()?;
    m.add_class::<PyScrambleMutation>()?;
    m.add_class::<PyDisplacementMutation>()?;
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
    // Py Errors
    m.add(
        "InvalidParameterError",
        _py.get_type::<InvalidParameterError>(),
    )?;
    // Functions
    let _ = m.add_function(wrap_pyfunction!(
        helpers::linalg::cross_euclidean_distances_py,
        m
    )?);

    // Rerefence points
    m.add_class::<operators::survival::helpers::PyDanAndDenisReferencePoints>()?;

    Ok(())
}
