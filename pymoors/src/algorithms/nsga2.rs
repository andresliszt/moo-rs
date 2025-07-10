use moors::{Nsga2, Nsga2Builder};
use numpy::ToPyArray;
use pymoors_macros::py_algorithm_impl;
use pyo3::prelude::*;

use crate::py_error::AlgorithmErrorWrapper;
use crate::py_fitness_and_constraints::{PyConstraintsFnWrapper, PyFitnessFnWrapper};
use crate::py_operators::{
    CrossoverOperatorDispatcher, DuplicatesCleanerDispatcher, MutationOperatorDispatcher,
    SamplingOperatorDispatcher,
};

#[pyclass(name = "Nsga2")]
pub struct PyNsga2 {
    algorithm: Nsga2<
        SamplingOperatorDispatcher,
        CrossoverOperatorDispatcher,
        MutationOperatorDispatcher,
        PyFitnessFnWrapper,
        PyConstraintsFnWrapper,
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
        num_vars,
        population_size,
        num_objectives,
        num_offsprings,
        num_iterations,
        mutation_rate=0.1,
        crossover_rate=0.9,
        keep_infeasible=false,
        verbose=true,
        duplicates_cleaner=None,
        constraints_fn=None,
        num_constraints=0,
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
        num_vars: usize,
        population_size: usize,
        num_objectives: usize,
        num_offsprings: usize,
        num_iterations: usize,
        mutation_rate: f64,
        crossover_rate: f64,
        keep_infeasible: bool,
        verbose: bool,
        duplicates_cleaner: Option<PyObject>,
        constraints_fn: Option<PyObject>,
        num_constraints: usize,
        lower_bound: Option<f64>,
        upper_bound: Option<f64>,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        // Unwrap the operator objects using the previously generated unwrap functions.
        let sampler = SamplingOperatorDispatcher::from_python_operator(sampler)?;
        let crossover = CrossoverOperatorDispatcher::from_python_operator(crossover)?;
        let mutation = MutationOperatorDispatcher::from_python_operator(mutation)?;
        let duplicates_cleaner = if let Some(py_obj) = duplicates_cleaner {
            Some(DuplicatesCleanerDispatcher::from_python_operator(py_obj)?)
        } else {
            None
        };
        // Build the mandatory population-level fitness.
        let fitness = PyFitnessFnWrapper::new(fitness_fn);
        // Build the optional constraints.
        let constraints =
            PyConstraintsFnWrapper::from_python_constraints(constraints_fn, lower_bound, upper_bound);
            

        // Build the NSGA2 algorithm instance.
        let algorithm = Nsga2Builder::default().sampler(sampler).
        crossover(crossover).
        mutation(mutation).duplicates_cleaner(duplicates_cleaner).
        
        
        
        
        
        
        
        
        new(
            sampler,
            crossover,
            mutation,
            duplicates_cleaner,
            fitness_closure,
            num_vars,
            num_objectives,
            num_constraints,
            population_size,
            num_offsprings,
            num_iterations,
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
