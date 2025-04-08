use moors::algorithms::Nsga3;
use moors::operators::survival::nsga3::Nsga3ReferencePoints;
use moors::operators::survival::reference_points::StructuredReferencePoints;
use numpy::{PyArray2, PyArrayMethods, ToPyArray};
use pymoors_macros::py_algorithm_impl;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;

use crate::py_error::MultiObjectiveAlgorithmErrorWrapper;
use crate::py_fitness_and_constraints::{
    PyConstraintsFn, PyFitnessFn, create_population_constraints_closure,
    create_population_fitness_closure,
};
use crate::py_operators::{
    CrossoverOperatorDispatcher, DuplicatesCleanerDispatcher, MutationOperatorDispatcher,
    SamplingOperatorDispatcher, unwrap_crossover_operator, unwrap_duplicates_operator,
    unwrap_mutation_operator, unwrap_sampling_operator,
};
use crate::py_reference_points::PyStructuredReferencePointsDispatcher;

#[pyclass(name = "Nsga3")]
pub struct PyNsga3 {
    algorithm: Nsga3<
        SamplingOperatorDispatcher,
        CrossoverOperatorDispatcher,
        MutationOperatorDispatcher,
        PyFitnessFn,
        PyConstraintsFn,
        DuplicatesCleanerDispatcher,
    >,
}

py_algorithm_impl!(PyNsga3);

#[pymethods]
impl PyNsga3 {
    #[new]
    #[pyo3(signature = (
        reference_points,
        sampler,
        crossover,
        mutation,
        fitness_fn,
        n_vars,
        population_size,
        n_offsprings,
        n_iterations,
        mutation_rate=0.1,
        crossover_rate=0.9,
        keep_infeasible=false,
        verbose=true,
        duplicates_cleaner=None,
        constraints_fn=None,
        lower_bound=None,
        upper_bound=None,
        seed=None
    ))]
    pub fn new(
        reference_points: PyObject,
        sampler: PyObject,
        crossover: PyObject,
        mutation: PyObject,
        fitness_fn: PyObject,
        n_vars: usize,
        population_size: usize,
        n_offsprings: usize,
        n_iterations: usize,
        mutation_rate: f64,
        crossover_rate: f64,
        keep_infeasible: bool,
        verbose: bool,
        duplicates_cleaner: Option<PyObject>,
        constraints_fn: Option<PyObject>,
        lower_bound: Option<f64>,
        upper_bound: Option<f64>,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        Python::with_gil(|py| {
            // First, try to extract the object as our custom type.
            let rp: Nsga3ReferencePoints = if let Ok(custom_obj) =
                reference_points.extract::<PyStructuredReferencePointsDispatcher>(py)
            {
                Nsga3ReferencePoints::new(custom_obj.generate(), false)
            } else if let Ok(rp_maybe_array) = reference_points.downcast_bound::<PyArray2<f64>>(py)
            {
                Nsga3ReferencePoints::new(rp_maybe_array.readonly().as_array().to_owned(), true)
            } else {
                return Err(PyTypeError::new_err(
                    "reference_points must be either a custom reference points class or a NumPy array.",
                ));
            };

            // Unwrap the genetic operators.
            let sampler = unwrap_sampling_operator(sampler)?;
            let crossover = unwrap_crossover_operator(crossover)?;
            let mutation = unwrap_mutation_operator(mutation)?;
            let duplicates_cleaner = if let Some(py_obj) = duplicates_cleaner {
                Some(unwrap_duplicates_operator(py_obj)?)
            } else {
                None
            };

            // Build the mandatory population-level fitness closure.
            let fitness_closure = create_population_fitness_closure(fitness_fn)?;
            // Build the optional constraints closure.
            let constraints_closure = if let Some(py_obj) = constraints_fn {
                Some(create_population_constraints_closure(py_obj)?)
            } else {
                None
            };

            let algorithm = Nsga3::new(
                rp,
                sampler,
                crossover,
                mutation,
                duplicates_cleaner,
                fitness_closure,
                n_vars,
                population_size,
                n_offsprings,
                n_iterations,
                mutation_rate,
                crossover_rate,
                keep_infeasible,
                verbose,
                constraints_closure,
                lower_bound,
                upper_bound,
                seed,
            )
            .map_err(MultiObjectiveAlgorithmErrorWrapper)?;

            Ok(PyNsga3 { algorithm })
        })
    }
}
