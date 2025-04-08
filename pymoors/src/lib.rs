// lib.rs
#![cfg_attr(all(coverage_nightly, test), feature(coverage_attribute))]
extern crate core;

pub mod algorithms;
pub mod py_error;
pub mod py_fitness_and_constraints;
pub mod py_operators;
pub mod py_reference_points;

use pyo3::prelude::*;

pub use algorithms::agemoea::PyAgeMoea;
pub use algorithms::nsga2::PyNsga2;
pub use algorithms::nsga3::PyNsga3;
pub use algorithms::revea::PyRevea;
pub use algorithms::rnsga2::PyRnsga2;
pub use py_error::InvalidParameterError;
pub use py_error::NoFeasibleIndividualsError;
pub use py_operators::{
    PyBitFlipMutation, PyCloseDuplicatesCleaner, PyDisplacementMutation, PyExactDuplicatesCleaner,
    PyExponentialCrossover, PyGaussianMutation, PyOrderCrossover, PyPermutationSampling,
    PyRandomSamplingBinary, PyRandomSamplingFloat, PyRandomSamplingInt, PyScrambleMutation,
    PySimulatedBinaryCrossover, PySinglePointBinaryCrossover, PySwapMutation,
    PyUniformBinaryCrossover,
};
pub use py_reference_points::PyDanAndDenisReferencePoints;

use faer_ext::IntoNdarray;
use moors::helpers::linalg::cross_euclidean_distances;
use numpy::ToPyArray;

#[pyfunction]
#[pyo3(name = "cross_euclidean_distances")]
/// This function will never be exposed to the users, its going to be used
/// for benchmarking against scipy cdist method
pub fn cross_euclidean_distances_py<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray2<'py, f64>,
    reference: numpy::PyReadonlyArray2<'py, f64>,
) -> Bound<'py, numpy::PyArray2<f64>> {
    let data = data.as_array().to_owned();
    let reference = reference.as_array().to_owned();
    let result = cross_euclidean_distances(&data, &reference);
    result.as_ref().into_ndarray().to_pyarray(py)
}

/// Root module `pymoors` that includes all classes.
#[pymodule]
fn _pymoors(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Add classes from algorithms
    m.add_class::<PyNsga2>()?;
    m.add_class::<PyNsga3>()?;
    m.add_class::<PyRnsga2>()?;
    m.add_class::<PyAgeMoea>()?;
    m.add_class::<PyRevea>()?;

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
    let _ = m.add_function(wrap_pyfunction!(cross_euclidean_distances_py, m)?);

    // Rerefence points
    m.add_class::<PyDanAndDenisReferencePoints>()?;

    Ok(())
}
