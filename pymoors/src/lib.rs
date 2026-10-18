// lib.rs
#![cfg_attr(all(coverage_nightly, test), feature(coverage_attribute))]
extern crate core;

pub mod algorithms;
pub mod custom_py_operators;
pub mod py_error;
pub mod py_fitness_and_constraints;
pub mod py_metrics;
pub mod py_operators;
pub mod py_reference_points;

use pyo3::prelude::*;

pub use algorithms::{
    PyAgeMoea, PyGeneticAlgorithmSOO, PyGrea, PyIbea, PyMoEtpso, PyNsga2, PyNsga3, PyRevea,
    PyRnsga2, PySpea2,
};
pub use py_error::{InitializationError, InvalidParameterError, NoFeasibleIndividualsError};
pub use py_metrics::{
    py_generational_distance, py_hypervolume, py_inverted_generational_distance, py_spacing,
    py_spread,
};
pub use py_operators::{
    PyArithmeticCrossover, PyBitFlipMutation, PyCloseDuplicatesCleaner, PyDisplacementMutation,
    PyExactDuplicatesCleaner, PyExponentialCrossover, PyGaussianMutation, PyInversionMutation,
    PyNoCrossover, PyNoMutation, PyNoRepair, PyOrderCrossover, PyPermutationSampling,
    PyRandomSamplingBinary, PyRandomSamplingFloat, PyRandomSamplingInt, PyScrambleMutation,
    PySimulatedBinaryCrossover, PySinglePointBinaryCrossover, PySwapMutation,
    PyTwoPointBinaryCrossover, PyUniformBinaryCrossover, PyUniformBinaryMutation,
    PyUniformRealMutation,
};
pub use py_reference_points::PyDanAndDenisReferencePoints;

/// Root module `pymoors` that includes all classes.
#[pymodule]
fn _pymoors(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Add classes from algorithms
    m.add_class::<PyNsga2>()?;
    m.add_class::<PyNsga3>()?;
    m.add_class::<PyRnsga2>()?;
    m.add_class::<PyAgeMoea>()?;
    m.add_class::<PyRevea>()?;
    m.add_class::<PySpea2>()?;
    m.add_class::<PyIbea>()?;
    m.add_class::<PyGrea>()?;
    m.add_class::<PyMoEtpso>()?;
    m.add_class::<PyGeneticAlgorithmSOO>()?;

    // Add classes from operators
    m.add_class::<PyBitFlipMutation>()?;
    m.add_class::<PyInversionMutation>()?;
    m.add_class::<PySwapMutation>()?;
    m.add_class::<PyGaussianMutation>()?;
    m.add_class::<PyScrambleMutation>()?;
    m.add_class::<PyDisplacementMutation>()?;
    m.add_class::<PyUniformBinaryMutation>()?;
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
    m.add_class::<PyArithmeticCrossover>()?;
    m.add_class::<PyTwoPointBinaryCrossover>()?;
    m.add_class::<PyUniformRealMutation>()?;
    m.add_class::<PyNoMutation>()?;
    m.add_class::<PyNoCrossover>()?;
    m.add_class::<PyNoRepair>()?;
    // Py Errors
    m.add(
        "NoFeasibleIndividualsError",
        _py.get_type::<NoFeasibleIndividualsError>(),
    )?;
    m.add(
        "InvalidParameterError",
        _py.get_type::<InvalidParameterError>(),
    )?;
    m.add("InitializationError", _py.get_type::<InitializationError>())?;

    // Rerefence points
    m.add_class::<PyDanAndDenisReferencePoints>()?;

    // Quality metrics
    m.add_function(wrap_pyfunction!(py_hypervolume, m)?)?;
    m.add_function(wrap_pyfunction!(py_generational_distance, m)?)?;
    m.add_function(wrap_pyfunction!(py_inverted_generational_distance, m)?)?;
    m.add_function(wrap_pyfunction!(py_spacing, m)?)?;
    m.add_function(wrap_pyfunction!(py_spread, m)?)?;

    Ok(())
}
