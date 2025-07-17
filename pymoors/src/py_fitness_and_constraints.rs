use moors::genetic::{Constraints, Fitness};
use moors::{ConstraintsFn, FitnessFn, NoConstraints};
use ndarray::Array2;
use numpy::{PyArray1, PyArray2, PyArrayMethods, ToPyArray};
use pyo3::prelude::*;

/// A Python‑backed fitness_fn function for 2D arrays (`Ix2`).
///
/// This struct wraps a Python callable that accepts a 2D NumPy array
/// (`ndarray::Array2<f64>`) and returns a 2D NumPy array of floats.
/// Implements `FitnessFn` with `Dim = Ix2`.
pub struct PyFitnessFnWrapper {
    py_fitness_fn: PyObject,
}

impl PyFitnessFnWrapper {
    pub fn from_python_fitness(py_fitness_fn: PyObject) -> Self {
        Self { py_fitness_fn }
    }
}

impl FitnessFn for PyFitnessFnWrapper {
    type Dim = ndarray::Ix2;

    fn call(&self, genes: &Array2<f64>) -> Fitness<Self::Dim> {
        Python::with_gil(|py| {
            // Convert the Rust Array2<f64> to a Python ndarray
            let py_input = genes.to_pyarray(py);
            // Call the Python function
            let result = self
                .py_fitness_fn
                .call1(py, (py_input,))
                .expect("Failed to call Python fitness_fn function");
            // Downcast to PyArray2<f64>
            let py_array = result
                .downcast_bound::<PyArray2<f64>>(py)
                .expect("Expected a PyArray2<f64> return");
            // Read-only view and convert back to an owned Array2
            py_array.readonly().as_array().to_owned()
        })
    }
}

/// A Python‑backed fitness_fn function for 1D arrays (`Ix1`).
///
/// This struct wraps a Python callable that accepts a 2D NumPy array
/// (`ndarray::Array2<f64>`) and returns a 1D NumPy array of floats.
/// Implements `FitnessFn` with `Dim = Ix1`.
pub struct PyFitnessFnWrapper1D {
    py_fitness_fn: PyObject,
}

impl PyFitnessFnWrapper1D {
    pub fn from_python_fitness(py_fitness_fn: PyObject) -> Self {
        Self { py_fitness_fn }
    }
}

impl FitnessFn for PyFitnessFnWrapper1D {
    type Dim = ndarray::Ix1;
    fn call(&self, genes: &Array2<f64>) -> Fitness<Self::Dim> {
        Python::with_gil(|py| {
            // Convert the Rust Array2<f64> to a Python ndarray
            let py_input = genes.to_pyarray(py);
            // Call the Python function
            let result = self
                .py_fitness_fn
                .call1(py, (py_input,))
                .expect("Failed to call Python fitness_fn function");
            // Downcast to PyArray2<f64>
            let py_array = result
                .downcast_bound::<PyArray1<f64>>(py)
                .expect("Expected a PyArray1<f64> return");
            // Read-only view and convert back to an owned Array2
            py_array.readonly().as_array().to_owned()
        })
    }
}

/// A Python‑backed constraints_fn function for 2D arrays (`Ix2`).
///
/// Wraps a Python callable that accepts a 2D NumPy array and returns a
/// 2D NumPy array of constraint values. Optional bounds can be provided.
pub struct PyConstraints {
    py_constraints_fn: PyObject,
    lower_bound: Option<f64>,
    upper_bound: Option<f64>,
}

impl PyConstraints {
    /// Create a new wrapper around the given Python function with optional bounds.
    ///
    /// # Arguments
    ///
    /// * `py_constraints_fn` – A Python object implementing
    ///   `__call__(ndarray) -> ndarray`.
    /// * `lower_bound` – Optional minimum constraint value.
    /// * `upper_bound` – Optional maximum constraint value.
    pub fn new(
        py_constraints_fn: PyObject,
        lower_bound: Option<f64>,
        upper_bound: Option<f64>,
    ) -> Self {
        Self {
            py_constraints_fn,
            lower_bound,
            upper_bound,
        }
    }
}

impl ConstraintsFn for PyConstraints {
    type Dim = ndarray::Ix2;

    fn call(&self, genes: &Array2<f64>) -> Constraints<Self::Dim> {
        Python::with_gil(|py| {
            // Convert the Rust Array2<f64> to a Python ndarray
            let py_input = genes.to_pyarray(py);
            // Call the Python function
            let result = self
                .py_constraints_fn
                .call1(py, (py_input,))
                .expect("Failed to call Python constraints_fn function");
            // Downcast to PyArray2<f64>
            let py_array = result
                .downcast_bound::<PyArray2<f64>>(py)
                .expect("Expected a PyArray2<f64> return");
            // Read-only view and convert back to an owned Array2
            py_array.readonly().as_array().to_owned()
        })
    }

    fn lower_bound(&self) -> Option<f64> {
        self.lower_bound
    }

    fn upper_bound(&self) -> Option<f64> {
        self.upper_bound
    }
}

pub enum PyConstraintsFnWrapper {
    Python(PyConstraints),
    None(NoConstraints),
}

impl PyConstraintsFnWrapper {
    pub fn from_python_constraints(pyobj: Option<PyObject>) -> PyConstraintsFnWrapper {
        if let Some(py_obj) = pyobj {
            Python::with_gil(|py| {
                let any = py_obj.bind(py);
                let lb = any
                    .getattr("lower_bound")
                    .and_then(|v| v.extract::<f64>())
                    .ok();
                let ub = any
                    .getattr("upper_bound")
                    .and_then(|v| v.extract::<f64>())
                    .ok();
                PyConstraintsFnWrapper::Python(PyConstraints::new(py_obj, lb, ub))
            })
        } else {
            PyConstraintsFnWrapper::None(NoConstraints)
        }
    }
}

impl ConstraintsFn for PyConstraintsFnWrapper {
    type Dim = ndarray::Ix2;

    fn call(&self, genes: &Array2<f64>) -> Constraints<Self::Dim> {
        match self {
            PyConstraintsFnWrapper::Python(w) => w.call(genes),
            PyConstraintsFnWrapper::None(n) => n.call(genes),
        }
    }
    fn lower_bound(&self) -> Option<f64> {
        match self {
            PyConstraintsFnWrapper::Python(w) => w.lower_bound(),
            PyConstraintsFnWrapper::None(n) => n.lower_bound(),
        }
    }
    fn upper_bound(&self) -> Option<f64> {
        match self {
            PyConstraintsFnWrapper::Python(w) => w.upper_bound(),
            PyConstraintsFnWrapper::None(n) => n.upper_bound(),
        }
    }
}
