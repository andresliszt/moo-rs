use numpy::{PyArray2, PyArrayMethods, ToPyArray};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use moors::genetic::{PopulationConstraints, PopulationFitness, PopulationGenes};

/// Placeholder types for the fitness and constraint functions.
pub type PyFitnessFn = Box<dyn Fn(&PopulationGenes) -> PopulationFitness + Send + Sync>;
pub type PyConstraintsFn = Box<dyn Fn(&PopulationGenes) -> PopulationConstraints + Send + Sync>;

/// Creates a closure that calls a Python function with a 2D NumPy array (`Array2<f64>`)
/// and expects to get back another 2D array of floats (`Array2<f64>`).
///
/// The returned closure has the signature:
///     `Fn(&PopulationGenes) -> PopulationFitness`
/// i.e., `(&Array2<f64>) -> Array2<f64>`.
pub fn create_population_fitness_closure(py_fitness_fn: PyObject) -> PyResult<PyFitnessFn> {
    Ok(Box::new(move |pop_genes: &PopulationGenes| {
        Python::with_gil(|py| {
            // Instead of `pop_genes.into_pyarray(py)`, use a reference-based method:
            let py_input = pop_genes.to_pyarray(py);

            let result_obj = py_fitness_fn
                .call1(py, (py_input,))
                .expect("Failed to call Python fitness function");

            // Downcast to `PyArray2<f64>`
            let py_array = result_obj
                .downcast_bound::<PyArray2<f64>>(py)
                .expect("Fitness fn must return 2D float ndarray");

            // Now take a READ-ONLY view and convert to owned Array2
            let rust_array = py_array.readonly().as_array().to_owned();

            rust_array
        })
    }))
}

/// Creates a closure that calls the given Python function (constraints_fn)
/// with a 2D NumPy array of shape `(population_size, genome_length)`
/// and returns another 2D float array of shape `(population_size, constraints_dimension)`.
pub fn create_population_constraints_closure(
    py_constraints_fn: PyObject,
) -> PyResult<PyConstraintsFn> {
    Ok(Box::new(move |pop_genes: &PopulationGenes| {
        Python::with_gil(|py| {
            // Convert from `Array2<f64>` to NumPy array without moving ownership
            let py_input = pop_genes.to_pyarray(py);

            let result_obj = py_constraints_fn
                .call1(py, (py_input,))
                .expect("Failed to call Python constraints function");

            // We expect a 2D np.ndarray of floats
            let py_array = result_obj
                .downcast_bound::<PyArray2<f64>>(py)
                .map_err(|_| {
                    PyRuntimeError::new_err("Constraints function must return 2D float ndarray")
                })
                .unwrap();

            // Convert PyArray2 -> Array2<f64>
            let rust_array = py_array.readonly().as_array().to_owned();

            rust_array
        })
    }))
}
