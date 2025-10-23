use moors::operators::IbeaHyperVolumeSurvivalOperator;
use moors::{Ibea, IbeaBuilder};
use ndarray::Array1;
use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1, ToPyArray};
use pymoors_macros::py_algorithm_impl;
use pyo3::prelude::*;

use crate::py_error::AlgorithmErrorWrapper;
use crate::py_fitness_and_constraints::{PyConstraintsFnWrapper, PyFitnessFnWrapper};
use crate::py_operators::{
    CrossoverOperatorDispatcher, DuplicatesCleanerDispatcher, MutationOperatorDispatcher,
    SamplingOperatorDispatcher,
};

#[pyclass(name = "Ibea")]
pub struct PyIbea {
    algorithm: Ibea<
        SamplingOperatorDispatcher,
        CrossoverOperatorDispatcher,
        MutationOperatorDispatcher,
        PyFitnessFnWrapper,
        PyConstraintsFnWrapper,
        DuplicatesCleanerDispatcher,
    >,
}

py_algorithm_impl!(PyIbea);

#[pymethods]
impl PyIbea {
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
        kappa,
        mutation_rate=0.1,
        crossover_rate=0.9,
        keep_infeasible=false,
        verbose=true,
        duplicates_cleaner=None,
        constraints_fn=None,
        seed=None
    ))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        reference_points: Py<PyArray1<f64>>,
        sampler: PyObject,
        crossover: PyObject,
        mutation: PyObject,
        fitness_fn: PyObject,
        num_vars: usize,
        population_size: usize,
        num_offsprings: usize,
        num_iterations: usize,
        kappa: f64,
        mutation_rate: f64,
        crossover_rate: f64,
        keep_infeasible: bool,
        verbose: bool,
        duplicates_cleaner: Option<PyObject>,
        constraints_fn: Option<PyObject>,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        let rp = reference_points_from_python(reference_points);
        let survival = IbeaHyperVolumeSurvivalOperator::new(rp, kappa);

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

        // Build the Ibea algorithm instance.
        let mut builder = IbeaBuilder::default()
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

        Ok(PyIbea {
            algorithm: algorithm,
        })
    }
}

/// Auxiliary function: grabs the GIL, borrows the array as read‑only,
/// and clones it into an `ndarray::Array1<f64>`.
fn reference_points_from_python(reference_points: Py<PyArray1<f64>>) -> Array1<f64> {
    Python::with_gil(|py| {
        let array_ref: &Bound<'_, PyArray1<f64>> = reference_points.bind(py);
        let readonly: PyReadonlyArray1<f64> = array_ref.readonly();
        readonly.as_array().to_owned()
    })
}
