# pymoors_macros

A collection of procedural macros to bridge the **moors** Rust crate with Python via PyO3.
These macros generate boilerplate for:

- **Operator wrappers** (`mutation`, `crossover`, `sampling`, `duplicates`)
- **Dispatcher enums** and `unwrap_*` functions
- **Algorithm bindings** (`py_algorithm_impl!`)

---

## Installation in `pymoors`

Added directly as:

```toml
# In Cargo.toml
[dependencies]
pymoors_macros = { path = "../pymoors_macros" }

[features]
default = []
```

---

## Mutation / Crossover / Sampling / Duplicates

### Register operators in bulk

```rust
use moors::operators::{MutationOperator, CrossoverOperator, SamplingOperator};
use moors::duplicates::PopulationCleaner;

// import all concrete types from `moors`
use moors::operators::mutation::{BitFlipMutation, ScrambleMutation, /*…*/};
use moors::operators::crossover::{OrderCrossover, SimulatedBinaryCrossover, /*…*/};
use moors::operators::sampling::{RandomSamplingFloat, PermutationSampling, /*…*/};
use moors::duplicates::{ExactDuplicatesCleaner, CloseDuplicatesCleaner};

// This expands to:
//  - PyBitFlipMutation, PyScrambleMutation, … structs
//  - impls of `.mutate()`, `.crossover()`, `.sample()`, `.remove_duplicates()`
//  - Dispatcher enums + `unwrap_*` helpers
pymoors_macros::register_py_operators_mutation!(
    BitFlipMutation,
    ScrambleMutation,
    /* … */
);

pymoors_macros::register_py_operators_crossover!(
    OrderCrossover,
    SimulatedBinaryCrossover,
    /* … */
);

pymoors_macros::register_py_operators_sampling!(
    RandomSamplingFloat,
    PermutationSampling,
    /* … */
);

pymoors_macros::register_py_operators_duplicates!(
    ExactDuplicatesCleaner,
    CloseDuplicatesCleaner
);
```

**Note:**
Because **moors** is a pure‑Rust crate, we **cannot** use a `#[proc_macro_attribute]` on its types to extract constructors or fields.
All `#[new]` methods and `#[getter]`s in the `Py*` wrappers must be written **manually**.

```rust
// --------------------------------------------------------------------------------
// Mutation new/getters (example)
// --------------------------------------------------------------------------------

#[pymethods]
impl PyBitFlipMutation {
    #[new]
    pub fn new(gene_mutation_rate: f64) -> Self {
        Self {
            inner: BitFlipMutation::new(gene_mutation_rate),
        }
    }

    #[getter]
    pub fn gene_mutation_rate(&self) -> f64 {
        self.inner.gene_mutation_rate
    }
}
```

---

## Dispatcher enums & unwrap functions

The registration macros also generate:

```rust
/// Example for mutation:
#[derive(Clone, Debug)]
pub enum MutationOperatorDispatcher {
    BitFlipMutation(BitFlipMutation),
    ScrambleMutation(ScrambleMutation),
    /* … */
}

pub fn unwrap_mutation_operator(
    obj: pyo3::PyObject
) -> pyo3::PyResult<MutationOperatorDispatcher> { /* … */ }
```

We use these in the algorithm bindings to convert a `PyObject` into the Rust enum.

---

## Algorithm bindings

We define Python‑exposed algorithm struct in Rust:

```rust
#[pyclass(name = "Nsga2")]
pub struct PyNsga2 {
    algorithm: moors::algorithms::Nsga2<
        SamplingOperatorDispatcher,
        CrossoverOperatorDispatcher,
        MutationOperatorDispatcher,
        /* … */,
        DuplicatesCleanerDispatcher
    >,
}

py_algorithm_impl!(PyNsga2);
```

This expands to:

```rust
#[pymethods]
impl PyNsga2 {
    /// `.run()` → calls the Rust `algorithm.run()`
    pub fn run(&mut self) -> PyResult<()> { /* … */ }

    /// `#[getter] population()` → returns a Python `Population` object
    /// built from `self.algorithm.inner.population`.
    #[getter]
    pub fn population(&self, py: Python) -> PyResult<PyObject> { /* … */ }
}
```

In the `#[new]` implementation, we should:

1. Call the appropriate `unwrap_*` for each operator argument.
2. Map any `MultiObjectiveAlgorithmError` into a `PyErr` via the wrapper type.


```rust
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

```

**Note:**
Note each algorithm has its own `new`, that is why pyo3 `#[new]` is not included in `py_algorithm_impl`

---

## Limitations & TODO

- **No reflection on external types:** Rust procedural macros cannot inspect constructors or fields of types defined in another crate.
- **Manual `new` / `getter` implementations:** Repetitive, but keeps **moors** independent.
- **Future improvement:** Provide a small `macro_rules!` in the Python wrapper crate to generate `new` + `getter` pairs automatically.

---
