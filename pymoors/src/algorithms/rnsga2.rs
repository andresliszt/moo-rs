use moors::{Rnsga2, Rnsga2Builder, Rnsga2ReferencePointsSurvival};
use numpy::ToPyArray;
use pymoors_macros::py_algorithm_impl;
use pyo3::prelude::*;

use crate::py_error::AlgorithmErrorWrapper;
use crate::py_fitness_and_constraints::{PyConstraintsFnWrapper, PyFitnessFnWrapper};
use crate::py_operators::{
    CrossoverOperatorDispatcher, DuplicatesCleanerDispatcher, MutationOperatorDispatcher,
    SamplingOperatorDispatcher,
};

use ndarray::Array2;
use numpy::{PyArray2, PyArrayMethods, PyReadonlyArray2};

#[pyclass(name = "Rnsga2")]
pub struct PyRnsga2 {
    algorithm: Rnsga2<
        SamplingOperatorDispatcher,
        CrossoverOperatorDispatcher,
        MutationOperatorDispatcher,
        PyFitnessFnWrapper,
        PyConstraintsFnWrapper,
        DuplicatesCleanerDispatcher,
    >,
}

// Define the NSGA-II algorithm using the macro
py_algorithm_impl!(PyRnsga2);

// Implement PyO3 methods
#[pymethods]
impl PyRnsga2 {
    #[new]
    #[pyo3(signature = (
        reference_points,
        sampler,
        crossover,
        mutation,
        fitness_fn,
        num_vars,
        population_size,
        num_offsprings,
        num_iterations,
        epsilon = 0.001,
        mutation_rate=0.1,
        crossover_rate=0.9,
        keep_infeasible=false,
        verbose=true,
        duplicates_cleaner=None,
        constraints_fn=None,
        seed=None,
    ))]
    pub fn new(
        reference_points: Py<PyArray2<f64>>,
        sampler: PyObject,
        crossover: PyObject,
        mutation: PyObject,
        fitness_fn: PyObject,
        num_vars: usize,
        population_size: usize,
        num_offsprings: usize,
        num_iterations: usize,
        epsilon: f64,
        mutation_rate: f64,
        crossover_rate: f64,
        keep_infeasible: bool,
        verbose: bool,
        duplicates_cleaner: Option<PyObject>,
        constraints_fn: Option<PyObject>,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        let rp = reference_points_from_python(reference_points);
        let survival = Rnsga2ReferencePointsSurvival::new(rp, epsilon);

        // Unwrap the operator objects using the previously generated unwrap functions.
        let sampler = SamplingOperatorDispatcher::from_python_operator(sampler)?;
        let crossover = CrossoverOperatorDispatcher::from_python_operator(crossover)?;
        let mutation = MutationOperatorDispatcher::from_python_operator(mutation)?;
        let duplicates_cleaner =
            DuplicatesCleanerDispatcher::from_python_operator(duplicates_cleaner)?;
        // Build the mandatory population-level fitness_fn.
        let fitness_fn = PyFitnessFnWrapper::from_python_fitness(fitness_fn);
        // Build the optional constraints_fn.
        let constraints_fn = PyConstraintsFnWrapper::from_python_constraints(constraints_fn);

        // Build the NSGA2 algorithm instance.
        let mut builder = Rnsga2Builder::default()
            .sampler(sampler)
            .crossover(crossover)
            .mutation(mutation)
            .survivor(survival)
            .duplicates_cleaner(duplicates_cleaner)
            .fitness_fn(fitness_fn)
            .constraints_fn(constraints_fn)
            .num_iterations(num_iterations)
            .num_vars(num_vars)
            .population_size(population_size)
            .num_offsprings(num_offsprings)
            .mutation_rate(mutation_rate)
            .crossover_rate(crossover_rate)
            .keep_infeasible(keep_infeasible)
            .verbose(verbose);

        if let Some(seed) = seed {
            builder = builder.seed(seed)
        }

        let algorithm = builder.build().map_err(AlgorithmErrorWrapper::from)?;

        Ok(PyRnsga2 {
            algorithm: algorithm,
        })
    }
}

/// Auxiliary function: grabs the GIL, borrows the array as read‑only,
/// and clones it into an `ndarray::Array2<f64>`.
fn reference_points_from_python(reference_points: Py<PyArray2<f64>>) -> Array2<f64> {
    Python::with_gil(|py| {
        let array_ref: &Bound<'_, PyArray2<f64>> = reference_points.bind(py);
        let readonly: PyReadonlyArray2<f64> = array_ref.readonly();
        readonly.as_array().to_owned()
    })
}
