use pyo3::prelude::*;

use crate::define_multiobj_pyclass;
use crate::helpers::functions::{
    create_population_constraints_closure, create_population_fitness_closure,
};
use crate::helpers::parser::{
    unwrap_crossover_operator, unwrap_duplicates_cleaner, unwrap_mutation_operator,
    unwrap_sampling_operator,
};
use crate::operators::selection::RankAndScoringSelection;
use crate::operators::survival::AgeMoeaSurvival;

// Define the NSGA-II algorithm using the macro
define_multiobj_pyclass!(AgeMoea);

// Implement PyO3 methods
#[pymethods]
impl AgeMoea {
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
    pub fn py_new(
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
        // Optional lower bound for each gene.
        lower_bound: Option<f64>,
        // Optional upper bound for each gene.
        upper_bound: Option<f64>,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        // Unwrap the genetic operators
        let sampler_box = unwrap_sampling_operator(sampler)?;
        let crossover_box = unwrap_crossover_operator(crossover)?;
        let mutation_box = unwrap_mutation_operator(mutation)?;
        let duplicates_box = if let Some(py_obj) = duplicates_cleaner {
            Some(unwrap_duplicates_cleaner(py_obj)?)
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

        // Create an instance of the selection/survival struct
        let selector_box = Box::new(RankAndScoringSelection::new());
        let survivor_box = Box::new(AgeMoeaSurvival::new());

        // Create the Rust struct
        let algorithm = MultiObjectiveAlgorithm::new(
            sampler_box,
            selector_box,
            survivor_box,
            crossover_box,
            mutation_box,
            duplicates_box,
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
        )?;

        Ok(Self {
            algorithm: algorithm,
        })
    }
}
