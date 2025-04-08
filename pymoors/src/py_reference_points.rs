use moors::operators::survival::reference_points::{
    DanAndDenisReferencePoints, StructuredReferencePoints,
};
use ndarray::Array2;
use numpy::{IntoPyArray, PyArray2};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;

// TODO: Once we define more ref points, this should be part of a macro

/// Expose the DanAndDenisReferencePoints struct to Python.
#[pyclass(name = "DanAndDenisReferencePoints")]
#[derive(Clone, Debug)]
pub struct PyDanAndDenisReferencePoints {
    pub inner: DanAndDenisReferencePoints,
}

#[pymethods]
impl PyDanAndDenisReferencePoints {
    #[new]
    pub fn new(n_reference_points: usize, n_objectives: usize) -> Self {
        PyDanAndDenisReferencePoints {
            inner: DanAndDenisReferencePoints::new(n_reference_points, n_objectives),
        }
    }
    /// Returns the reference points as a NumPy array.
    fn generate<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let array = self.inner.generate();
        array.into_pyarray(py)
    }
}

/// An enum that can hold any supported structured reference points type.
pub enum PyStructuredReferencePointsDispatcher {
    DanAndDenis(PyDanAndDenisReferencePoints),
}

/// Implement extraction for the enum.
impl<'py> FromPyObject<'py> for PyStructuredReferencePointsDispatcher {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        if let Ok(dan) = ob.extract::<PyDanAndDenisReferencePoints>() {
            Ok(PyStructuredReferencePointsDispatcher::DanAndDenis(dan))
        } else {
            Err(PyTypeError::new_err(
                "reference_points must be one of the supported structured reference point types.",
            ))
        }
    }
}

/// Implement the trait for the enum.
impl StructuredReferencePoints for PyStructuredReferencePointsDispatcher {
    fn generate(&self) -> Array2<f64> {
        match self {
            PyStructuredReferencePointsDispatcher::DanAndDenis(d) => d.inner.generate(),
        }
    }
}
