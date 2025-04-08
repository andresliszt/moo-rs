use pyo3::prelude::*;
// Bring into scope PyArrayMethods because it is used by the macros.
use numpy::PyArrayMethods;

// Import mutation operator traits and concrete types from moors.
use moors::operators::MutationOperator;
use moors::operators::mutation::bitflip::BitFlipMutation;
use moors::operators::mutation::displacement::DisplacementMutation;
use moors::operators::mutation::gaussian::GaussianMutation;
use moors::operators::mutation::scramble::ScrambleMutation;
use moors::operators::mutation::swap::SwapMutation;

// Import crossover operator traits and concrete types.
use moors::operators::CrossoverOperator;
use moors::operators::crossover::exponential::ExponentialCrossover;
use moors::operators::crossover::order::OrderCrossover;
use moors::operators::crossover::sbx::SimulatedBinaryCrossover;
use moors::operators::crossover::single_point::SinglePointBinaryCrossover;
use moors::operators::crossover::uniform_binary::UniformBinaryCrossover;

// Import sampling operator traits and concrete types.
use moors::operators::sampling::{
    PermutationSampling, RandomSamplingBinary, RandomSamplingFloat, RandomSamplingInt,
    SamplingOperator,
};

// Import duplicates cleaner traits and concrete types.
use moors::duplicates::{CloseDuplicatesCleaner, ExactDuplicatesCleaner, PopulationCleaner};

// The following macros will create all the Python operator wrappers from moors,
// each providing a corresponding unwrap function with specific names:
// - unwrap_mutation_operator
// - unwrap_crossover_operator
// - unwrap_sampling_operator
// - unwrap_duplicates_operator
//
// These functions will be imported in the algorithms module.

pymoors_macros::register_py_operators_mutation!(
    BitFlipMutation,
    DisplacementMutation,
    GaussianMutation,
    ScrambleMutation,
    SwapMutation,
);

pymoors_macros::register_py_operators_crossover!(
    ExponentialCrossover,
    OrderCrossover,
    SimulatedBinaryCrossover,
    SinglePointBinaryCrossover,
    UniformBinaryCrossover,
);

pymoors_macros::register_py_operators_sampling!(
    PermutationSampling,
    RandomSamplingBinary,
    RandomSamplingFloat,
    RandomSamplingInt,
);

pymoors_macros::register_py_operators_duplicates!(ExactDuplicatesCleaner, CloseDuplicatesCleaner);

// --------------------------------------------------------------------------------
// NOTE: Because `moors` is completely independent from `pymoors`, we do NOT use
// `proc_macro_attribute` on the structs defined in `moors` to maintain separation of
// concerns. There is no facility for a proc‑macro to reflectively extract a constructor
// signature or fields from an external crate's types. Thus, we must manually implement
// the `new` and `getter` methods for each Python wrapper.
//
// TODO: We could simplify these repetitive `new`/`getter` blocks by introducing a
// `macro_rules!` helper to generate them automatically.
// --------------------------------------------------------------------------------

// --------------------------------------------------------------------------------
// Mutation new/getters
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

#[pymethods]
impl PyDisplacementMutation {
    #[new]
    pub fn new() -> Self {
        Self {
            inner: DisplacementMutation::new(),
        }
    }
}

#[pymethods]
impl PySwapMutation {
    #[new]
    pub fn new() -> Self {
        Self {
            inner: SwapMutation::new(),
        }
    }
}

#[pymethods]
impl PyGaussianMutation {
    #[new]
    pub fn new(gene_mutation_rate: f64, sigma: f64) -> Self {
        Self {
            inner: GaussianMutation::new(gene_mutation_rate, sigma),
        }
    }

    #[getter]
    pub fn gene_mutation_rate(&self) -> f64 {
        self.inner.gene_mutation_rate
    }
    #[getter]
    pub fn sigma(&self) -> f64 {
        self.inner.sigma
    }
}

#[pymethods]
impl PyScrambleMutation {
    #[new]
    pub fn new() -> Self {
        Self {
            inner: ScrambleMutation::new(),
        }
    }
}

// --------------------------------------------------------------------------------
// Crossover new/getters
// --------------------------------------------------------------------------------

#[pymethods]
impl PyExponentialCrossover {
    #[new]
    pub fn new(exponential_crossover_rate: f64) -> Self {
        Self {
            inner: ExponentialCrossover::new(exponential_crossover_rate),
        }
    }

    #[getter]
    pub fn exponential_crossover_rate(&self) -> f64 {
        self.inner.exponential_crossover_rate
    }
}

#[pymethods]
impl PyOrderCrossover {
    #[new]
    pub fn new() -> Self {
        Self {
            inner: OrderCrossover::new(),
        }
    }
}

#[pymethods]
impl PySimulatedBinaryCrossover {
    #[new]
    pub fn new(distribution_index: f64) -> Self {
        Self {
            inner: SimulatedBinaryCrossover::new(distribution_index),
        }
    }
    #[getter]
    pub fn distribution_index(&self) -> f64 {
        self.inner.distribution_index
    }
}

#[pymethods]
impl PySinglePointBinaryCrossover {
    #[new]
    pub fn new() -> Self {
        Self {
            inner: SinglePointBinaryCrossover::new(),
        }
    }
}

#[pymethods]
impl PyUniformBinaryCrossover {
    #[new]
    pub fn new() -> Self {
        Self {
            inner: UniformBinaryCrossover::new(),
        }
    }
}

// --------------------------------------------------------------------------------
// Sampling new/getters
// --------------------------------------------------------------------------------

#[pymethods]
impl PyRandomSamplingBinary {
    #[new]
    pub fn new() -> Self {
        Self {
            inner: RandomSamplingBinary::new(),
        }
    }
}

#[pymethods]
impl PyRandomSamplingFloat {
    #[new]
    pub fn new(min: f64, max: f64) -> Self {
        Self {
            inner: RandomSamplingFloat::new(min, max),
        }
    }

    #[getter]
    pub fn min(&self) -> f64 {
        self.inner.min
    }

    #[getter]
    pub fn max(&self) -> f64 {
        self.inner.max
    }
}

#[pymethods]
impl PyRandomSamplingInt {
    #[new]
    pub fn new(min: i32, max: i32) -> Self {
        Self {
            inner: RandomSamplingInt::new(min, max),
        }
    }

    #[getter]
    pub fn min(&self) -> i32 {
        self.inner.min
    }

    #[getter]
    pub fn max(&self) -> i32 {
        self.inner.max
    }
}

#[pymethods]
impl PyPermutationSampling {
    #[new]
    pub fn new() -> Self {
        Self {
            inner: PermutationSampling::new(),
        }
    }
}

// --------------------------------------------------------------------------------
// Duplicates cleaner new/getters
// --------------------------------------------------------------------------------

#[pymethods]
impl PyExactDuplicatesCleaner {
    #[new]
    pub fn new() -> Self {
        Self {
            inner: ExactDuplicatesCleaner::new(),
        }
    }
}

#[pymethods]
impl PyCloseDuplicatesCleaner {
    #[new]
    pub fn new(epsilon: f64) -> Self {
        Self {
            inner: CloseDuplicatesCleaner::new(epsilon),
        }
    }
    #[getter]
    pub fn epsilon(&self) -> f64 {
        self.inner.epsilon
    }
}
