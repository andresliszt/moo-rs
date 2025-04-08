use moors::algorithms::Nsga2;
use numpy::ToPyArray;
use pymoors_macros::py_algorithm_impl;
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

#[pyclass(name = "Nsga2")]
pub struct PyNsga2 {
    algorithm: Nsga2<
        SamplingOperatorDispatcher,
        CrossoverOperatorDispatcher,
        MutationOperatorDispatcher,
        PyFitnessFn,
        PyConstraintsFn,
        DuplicatesCleanerDispatcher,
    >,
}

py_algorithm_impl!(PyNsga2);

#[pymethods]
impl PyNsga2 {
    #[new]
    #[pyo3(signature = (
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
    #[allow(clippy::too_many_arguments)]
    pub fn new(
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
        // Unwrap the operator objects using the previously generated unwrap functions.
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

        // Build the NSGA2 algorithm instance.
        let algorithm = Nsga2::new(
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

        Ok(PyNsga2 {
            algorithm: algorithm,
        })
    }
}
