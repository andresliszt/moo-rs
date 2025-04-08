use moors::algorithms::Revea;
use moors::operators::survival::reference_points::StructuredReferencePoints;
use ndarray::Array2;
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

#[pyclass(name = "Revea")]
pub struct PyRevea {
    algorithm: Revea<
        SamplingOperatorDispatcher,
        CrossoverOperatorDispatcher,
        MutationOperatorDispatcher,
        PyFitnessFn,
        PyConstraintsFn,
        DuplicatesCleanerDispatcher,
    >,
}

// Define the Revea algorithm using the macro
py_algorithm_impl!(PyRevea);

// Implement PyO3 methods
#[pymethods]
impl PyRevea {
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
        alpha=2.0,
        frequency=0.2,
        mutation_rate=0.1,
        crossover_rate=0.9,
        keep_infeasible=false,
        verbose=true,
        duplicates_cleaner=None,
        constraints_fn=None,
        lower_bound=None,
        upper_bound=None,
        seed=None,
    ))]
    pub fn py_new<'py>(
        reference_points: PyObject,
        sampler: PyObject,
        crossover: PyObject,
        mutation: PyObject,
        fitness_fn: PyObject,
        n_vars: usize,
        population_size: usize,
        n_offsprings: usize,
        n_iterations: usize,
        alpha: f64,
        frequency: f64,
        mutation_rate: f64,
        crossover_rate: f64,
        keep_infeasible: bool,
        verbose: bool,
        duplicates_cleaner: Option<PyObject>,
        constraints_fn: Option<PyObject>,
        // Optional lower bound for each gene.
        lower_bound: Option<f64>,
        // Optional upper bound for each gene.
        upper_bound: Option<f64>,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        Python::with_gil(|py| {
            // First, try to extract the object as our custom type.
            let rp: Array2<f64> = if let Ok(custom_obj) =
                reference_points.extract::<PyStructuredReferencePointsDispatcher>(py)
            {
                custom_obj.generate()
            } else if let Ok(rp_maybe_array) = reference_points.downcast_bound::<PyArray2<f64>>(py)
            {
                rp_maybe_array.readonly().as_array().to_owned()
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

            let algorithm = Revea::new(
                rp,
                alpha,
                frequency,
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

            Ok(PyRevea { algorithm })
        })
    }
}
