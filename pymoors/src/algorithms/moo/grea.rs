use moors::{Grea, GreaBuilder};
use numpy::ToPyArray;
use pymoors_macros::py_algorithm_impl;
use pyo3::prelude::*;

use crate::custom_py_operators::controller_from_python;
use crate::py_error::AlgorithmErrorWrapper;
use crate::py_fitness_and_constraints::{PyConstraintsFnWrapper, PyFitnessFnWrapper};
use crate::py_operators::RepairOperatorDispatcher;
use crate::py_operators::{
    CrossoverOperatorDispatcher, DuplicatesCleanerDispatcher, MutationOperatorDispatcher,
    SamplingOperatorDispatcher,
};

#[pyclass(name = "Grea")]
pub struct PyGrea {
    algorithm: Grea<
        SamplingOperatorDispatcher,
        CrossoverOperatorDispatcher,
        MutationOperatorDispatcher,
        PyFitnessFnWrapper,
        PyConstraintsFnWrapper,
    >,
}

py_algorithm_impl!(PyGrea);

#[pymethods]
impl PyGrea {
    #[new]
    #[pyo3(signature = (
        sampler,
        crossover,
        mutation,
        fitness_fn,
        num_vars,
        population_size,
        num_offsprings,
        num_iterations,
        div,
        mutation_rate=0.1,
        crossover_rate=0.9,
        keep_infeasible=false,
        verbose=true,
        duplicates_cleaner=None,
        constraints_fn=None,
        controller=None,
        seed=None,
        repair=None
    ))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        sampler: Py<PyAny>,
        crossover: Py<PyAny>,
        mutation: Py<PyAny>,
        fitness_fn: Py<PyAny>,
        num_vars: usize,
        population_size: usize,
        num_offsprings: usize,
        num_iterations: usize,
        div: usize,
        mutation_rate: f64,
        crossover_rate: f64,
        keep_infeasible: bool,
        verbose: bool,
        duplicates_cleaner: Option<Py<PyAny>>,
        constraints_fn: Option<Py<PyAny>>,
        controller: Option<Py<PyAny>>,
        seed: Option<u64>,
        repair: Option<Py<PyAny>>,
    ) -> PyResult<Self> {
        let sampler = SamplingOperatorDispatcher::from_python_operator(sampler)?;
        let crossover = CrossoverOperatorDispatcher::from_python_operator(crossover)?;
        let mutation = MutationOperatorDispatcher::from_python_operator(mutation)?;
        let duplicates_cleaner =
            DuplicatesCleanerDispatcher::from_python_operator(duplicates_cleaner)?;
        let controller = controller_from_python(controller)?;
        let repair = RepairOperatorDispatcher::from_python_operator(repair)?;
        let fitness_fn = PyFitnessFnWrapper::from_python_fitness(fitness_fn);
        let constraints_fn = PyConstraintsFnWrapper::from_python_constraints(constraints_fn);

        let mut builder = GreaBuilder::default()
            .sampler(sampler)
            .crossover(crossover)
            .mutation(mutation)
            .div(div)
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
        if let Some(controller) = controller {
            builder = builder.controller(controller)
        }
        if let Some(repair) = repair {
            builder = builder.repair(repair)
        }

        let algorithm = builder.build().map_err(AlgorithmErrorWrapper::from)?;

        Ok(PyGrea { algorithm })
    }
}
