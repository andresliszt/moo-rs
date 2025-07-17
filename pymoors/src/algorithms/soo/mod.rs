use moors::algorithms::{AlgorithmBuilder, GeneticAlgorithm};
use moors::operators::selection::soo::RankSelection;
use moors::operators::survival::soo::FitnessSurvival;
use numpy::ToPyArray;
use pyo3::prelude::*;

use crate::py_error::AlgorithmErrorWrapper;
use crate::py_fitness_and_constraints::{PyConstraintsFnWrapper, PyFitnessFnWrapper1D};
use crate::py_operators::{
    CrossoverOperatorDispatcher, DuplicatesCleanerDispatcher, MutationOperatorDispatcher,
    SamplingOperatorDispatcher,
};

#[pyclass(name = "GeneticAlgorithmSOO")]
pub struct PyGeneticAlgorithmSOO {
    algorithm: GeneticAlgorithm<
        SamplingOperatorDispatcher,
        RankSelection,
        FitnessSurvival,
        CrossoverOperatorDispatcher,
        MutationOperatorDispatcher,
        PyFitnessFnWrapper1D,
        PyConstraintsFnWrapper,
        DuplicatesCleanerDispatcher,
    >,
}

// TODO: The macro doesn't work because GeneticAlgorithm doesn't have population as method but as attribute

// py_algorithm_impl!(PyGeneticAlgorithmSOO);

#[pymethods]
impl PyGeneticAlgorithmSOO {
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
        // Unwrap the operator objects using the previously generated unwrap functions.
        let sampler = SamplingOperatorDispatcher::from_python_operator(sampler)?;
        let crossover = CrossoverOperatorDispatcher::from_python_operator(crossover)?;
        let mutation = MutationOperatorDispatcher::from_python_operator(mutation)?;
        let duplicates_cleaner =
            DuplicatesCleanerDispatcher::from_python_operator(duplicates_cleaner)?;
        // Build the mandatory population-level fitness_fn.
        let fitness_fn = PyFitnessFnWrapper1D::from_python_fitness(fitness_fn);
        // Build the optional constraints_fn.
        let constraints_fn = PyConstraintsFnWrapper::from_python_constraints(constraints_fn);

        // Build the NSGA2 algorithm instance.
        let mut builder = AlgorithmBuilder::default()
            .sampler(sampler)
            .survivor(FitnessSurvival)
            .selector(RankSelection)
            .crossover(crossover)
            .mutation(mutation)
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

        Ok(PyGeneticAlgorithmSOO {
            algorithm: algorithm,
        })
    }

    /// Getter for the algorithm's population.
    /// It converts the internal population members (genes, fitness, rank, constraints)
    /// to Python objects using NumPy.
    #[getter]
    pub fn population(&self, py: pyo3::Python) -> pyo3::PyResult<pyo3::PyObject> {
        let schemas_module = py.import("pymoors.schemas")?;
        let population_class = schemas_module.getattr("Population")?;
        let population = self.algorithm.population.as_ref().unwrap();
        let py_genes = population.genes.to_pyarray(py);
        let py_fitness = population.fitness.to_pyarray(py);
        let py_constraints = population.constraints.to_pyarray(py);

        let py_rank = if let Some(ref r) = population.rank {
            r.to_pyarray(py).into_py(py)
        } else {
            py.None().into_py(py)
        };
        let py_survival_score = if let Some(ref r) = population.survival_score {
            r.to_pyarray(py).into_py(py)
        } else {
            py.None().into_py(py)
        };
        let py_survival_score = if let Some(ref r) = population.survival_score {
            r.to_pyarray(py).into_py(py)
        } else {
            py.None().into_py(py)
        };
        let kwargs = pyo3::types::PyDict::new(py);
        kwargs.set_item("genes", py_genes)?;
        kwargs.set_item("fitness", py_fitness)?;
        kwargs.set_item("rank", py_rank)?;
        kwargs.set_item("constraints", py_constraints)?;
        kwargs.set_item("survival_score", py_survival_score)?;
        let py_instance = population_class.call((), Some(&kwargs))?;
        Ok(py_instance.into_py(py))
    }

    pub fn run(&mut self) -> pyo3::PyResult<()> {
        self.algorithm
            .run()
            .map_err(|e| AlgorithmErrorWrapper(e.into()))?;
        Ok(())
    }
}
