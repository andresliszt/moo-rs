use moors::{
    Nsga3, Nsga3Builder, Nsga3ReferencePoints, Nsga3ReferencePointsSurvival,
    StructuredReferencePoints,
};
use numpy::{PyArray2, PyArrayMethods, ToPyArray};
use pymoors_macros::py_algorithm_impl;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;

use crate::py_error::AlgorithmErrorWrapper;
use crate::py_fitness_and_constraints::{PyConstraintsFnWrapper, PyFitnessFnWrapper};
use crate::py_operators::{
    CrossoverOperatorDispatcher, DuplicatesCleanerDispatcher, MutationOperatorDispatcher,
    SamplingOperatorDispatcher,
};
use crate::py_reference_points::PyStructuredReferencePointsDispatcher;

#[pyclass(name = "Nsga3")]
pub struct PyNsga3 {
    algorithm: Nsga3<
        SamplingOperatorDispatcher,
        CrossoverOperatorDispatcher,
        MutationOperatorDispatcher,
        PyFitnessFnWrapper,
        PyConstraintsFnWrapper,
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
        num_vars,
        population_size,
        num_offsprings,
        num_iterations,
        mutation_rate=0.1,
        crossover_rate=0.9,
        keep_infeasible=false,
        verbose=true,
        duplicates_cleaner=None,
        constraints_fn=None,
        seed=None
    ))]
    pub fn new(
        reference_points: PyObject,
        sampler: PyObject,
        crossover: PyObject,
        mutation: PyObject,
        fitness_fn: PyObject,
        num_vars: usize,
        population_size: usize,
        num_offsprings: usize,
        num_iterations: usize,
        mutation_rate: f64,
        crossover_rate: f64,
        keep_infeasible: bool,
        verbose: bool,
        duplicates_cleaner: Option<PyObject>,
        constraints_fn: Option<PyObject>,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        let rp = reference_points_from_python(reference_points)?;
        let survival = Nsga3ReferencePointsSurvival::new(rp);

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
        let mut builder = Nsga3Builder::default()
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

        Ok(PyNsga3 {
            algorithm: algorithm,
        })
    }
}

fn reference_points_from_python(reference_points: PyObject) -> Result<Nsga3ReferencePoints, PyErr> {
    Python::with_gil(|py| {
        // First, try to extract the object as our custom type.
        let rp: Nsga3ReferencePoints = if let Ok(custom_obj) =
            reference_points.extract::<PyStructuredReferencePointsDispatcher>(py)
        {
            Nsga3ReferencePoints::new(custom_obj.generate(), false)
        } else if let Ok(rp_maybe_array) = reference_points.downcast_bound::<PyArray2<f64>>(py) {
            Nsga3ReferencePoints::new(rp_maybe_array.readonly().as_array().to_owned(), true)
        } else {
            return Err(PyTypeError::new_err(
                "reference_points must be either a custom reference points class or a NumPy array.",
            ));
        };
        Ok(rp)
    })
}
