use moors::{MoEtpso, MoEtpsoBuilder};
use numpy::ToPyArray;
use pyo3::prelude::*;

use crate::py_error::{AlgorithmErrorWrapper, InitializationError, InvalidParameterError};
use crate::py_fitness_and_constraints::{PyConstraintsFnWrapper, PyFitnessFnWrapper};

#[pyclass(name = "MoEtpso")]
pub struct PyMoEtpso {
    algorithm: MoEtpso<PyFitnessFnWrapper, PyConstraintsFnWrapper>,
}

#[pymethods]
impl PyMoEtpso {
    #[new]
    #[pyo3(signature = (
        fitness_fn,
        num_vars,
        population_size,
        num_iterations,
        lower_bound,
        upper_bound,
        inertia_weight=1.0,
        cognitive_coefficient=2.05,
        social_coefficient=2.05,
        stagnation_threshold=20,
        verbose=true,
        constraints_fn=None,
        seed=None
    ))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        fitness_fn: Py<PyAny>,
        num_vars: usize,
        population_size: usize,
        num_iterations: usize,
        lower_bound: f64,
        upper_bound: f64,
        inertia_weight: f64,
        cognitive_coefficient: f64,
        social_coefficient: f64,
        stagnation_threshold: usize,
        verbose: bool,
        constraints_fn: Option<Py<PyAny>>,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        let fitness_fn = PyFitnessFnWrapper::from_python_fitness(fitness_fn);
        let constraints_fn = PyConstraintsFnWrapper::from_python_constraints(constraints_fn);

        let mut builder = MoEtpsoBuilder::default()
            .fitness_fn(fitness_fn)
            .constraints_fn(constraints_fn)
            .num_vars(num_vars)
            .population_size(population_size)
            .num_iterations(num_iterations)
            .lower_bound(lower_bound)
            .upper_bound(upper_bound)
            .inertia_weight(inertia_weight)
            .cognitive_coefficient(cognitive_coefficient)
            .social_coefficient(social_coefficient)
            .stagnation_threshold(stagnation_threshold)
            .verbose(verbose);

        if let Some(seed) = seed {
            builder = builder.seed(seed)
        }

        let algorithm = builder
            .build()
            .map_err(|e| InvalidParameterError::new_err(e.to_string()))?;

        Ok(PyMoEtpso { algorithm })
    }

    pub fn run(&mut self) -> PyResult<()> {
        self.algorithm
            .run()
            .map_err(|e| AlgorithmErrorWrapper(e.into()))?;
        Ok(())
    }

    /// Non-dominated (and, if any exist, feasible) subset of the swarm's personal
    /// bests. Current particle positions are transient and are not exposed.
    #[getter]
    pub fn population(&self, py: Python) -> PyResult<Py<PyAny>> {
        let best = self
            .algorithm
            .best_population()
            .ok_or_else(|| InitializationError::new_err("Algorithm has not been run yet"))?;
        let best = best.best();

        let feasible: Vec<usize> = match &best.constraint_violation_totals {
            Some(cv) => cv
                .iter()
                .enumerate()
                .filter_map(|(i, &v)| (v <= 0.0).then_some(i))
                .collect(),
            None => Vec::new(),
        };
        let best = if feasible.is_empty() {
            best
        } else {
            best.selected(&feasible)
        };

        let population_class = py.import("pymoors.schemas")?.getattr("Population")?;
        let kwargs = pyo3::types::PyDict::new(py);
        kwargs.set_item("genes", best.genes.to_pyarray(py))?;
        kwargs.set_item("fitness", best.fitness.to_pyarray(py))?;
        kwargs.set_item("constraints", best.constraints.to_pyarray(py))?;
        Ok(population_class
            .call((), Some(&kwargs))?
            .into_any()
            .unbind())
    }
}
