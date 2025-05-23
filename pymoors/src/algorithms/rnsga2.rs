use moors::algorithms::Rnsga2;
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
    SamplingOperatorDispatcher,
};

use numpy::{PyArray2, PyArrayMethods};

#[pyclass(name = "Rnsga2")]
pub struct PyRnsga2 {
    algorithm: Rnsga2<
        SamplingOperatorDispatcher,
        CrossoverOperatorDispatcher,
        MutationOperatorDispatcher,
        PyFitnessFn,
        PyConstraintsFn,
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
        num_objectives,
        num_offsprings,
        num_iterations,
        epsilon = 0.001,
        mutation_rate=0.1,
        crossover_rate=0.9,
        keep_infeasible=false,
        verbose=true,
        duplicates_cleaner=None,
        constraints_fn=None,
        num_constraints=0,
        lower_bound=None,
        upper_bound=None,
        seed=None,
    ))]
    pub fn py_new<'py>(
        reference_points: &Bound<'py, PyArray2<f64>>,
        sampler: PyObject,
        crossover: PyObject,
        mutation: PyObject,
        fitness_fn: PyObject,
        num_vars: usize,
        population_size: usize,
        num_objectives: usize,
        num_offsprings: usize,
        num_iterations: usize,
        epsilon: f64,
        mutation_rate: f64,
        crossover_rate: f64,
        keep_infeasible: bool,
        verbose: bool,
        duplicates_cleaner: Option<PyObject>,
        constraints_fn: Option<PyObject>,
        num_constraints: usize,
        // Optional lower bound for each gene.
        lower_bound: Option<f64>,
        // Optional upper bound for each gene.
        upper_bound: Option<f64>,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        // Unwrap the genetic operators
        let sampler = SamplingOperatorDispatcher::from_python_operator(sampler)?;
        let crossover = CrossoverOperatorDispatcher::from_python_operator(crossover)?;
        let mutation = MutationOperatorDispatcher::from_python_operator(mutation)?;
        let duplicates = if let Some(py_obj) = duplicates_cleaner {
            Some(DuplicatesCleanerDispatcher::from_python_operator(py_obj)?)
        } else {
            None
        };

        // Build the MANDATORY population-level fitness closure
        let fitness_closure = create_population_fitness_closure(fitness_fn)?;

        // Build OPTIONAL population-level constraints closure
        let constraints_closure = if let Some(py_obj) = constraints_fn {
            Some(create_population_constraints_closure(py_obj)?)
        } else {
            None
        };

        // Convert PyArray2 to Array2
        let rp = reference_points.to_owned_array();

        let algorithm = Rnsga2::new(
            rp,
            epsilon,
            sampler,
            crossover,
            mutation,
            duplicates,
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

        Ok(PyRnsga2 {
            algorithm: algorithm,
        })
    }
}
