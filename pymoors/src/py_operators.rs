use numpy::PyArrayMethods;
use pyo3::prelude::*;

use moors::{
    ArithmeticCrossover,
    BitFlipMutation,
    CloseDuplicatesCleaner,
    CrossoverOperator,
    DisplacementMutation,
    ExactDuplicatesCleaner,
    ExponentialCrossover,
    GaussianMutation,
    InversionMutation,
    MutationOperator,
    NoDuplicatesCleaner,
    OrderCrossover,
    PermutationSampling,
    PopulationCleaner,
    RandomSamplingBinary,
    RandomSamplingFloat,
    RandomSamplingInt,
    SamplingOperator,
    ScrambleMutation,
    SimulatedBinaryCrossover,
    SinglePointBinaryCrossover,
    SwapMutation,
    TwoPointBinaryCrossover,
    UniformBinaryCrossover,
    // UniformRealMutation, TODO: Need to implement Debug
    UniformBinaryMutation,
};

use pymoors_macros::{
    register_py_operators_crossover, register_py_operators_duplicates,
    register_py_operators_mutation, register_py_operators_sampling,
};

use crate::custom_py_operators::{
    CustomPyCrossoverOperatorWrapper, CustomPyMutationOperatorWrapper,
    CustomPySamplingOperatorWrapper,
};

#[derive(Debug)]
#[register_py_operators_mutation]
pub enum MutationOperatorDispatcher {
    BitFlipMutation(BitFlipMutation),
    DisplacementMutation(DisplacementMutation),
    GaussianMutation(GaussianMutation),
    ScrambleMutation(ScrambleMutation),
    SwapMutation(SwapMutation),
    InversionMutation(InversionMutation),
    UniformBinaryMutation(UniformBinaryMutation),

    CustomPyMutationOperatorWrapper(CustomPyMutationOperatorWrapper),
}

#[derive(Debug)]
#[register_py_operators_crossover]
pub enum CrossoverOperatorDispatcher {
    ExponentialCrossover(ExponentialCrossover),
    OrderCrossover(OrderCrossover),
    SimulatedBinaryCrossover(SimulatedBinaryCrossover),
    SinglePointBinaryCrossover(SinglePointBinaryCrossover),
    UniformBinaryCrossover(UniformBinaryCrossover),
    ArithmeticCrossover(ArithmeticCrossover),
    TwoPointBinaryCrossover(TwoPointBinaryCrossover),

    CustomPyCrossoverOperatorWrapper(CustomPyCrossoverOperatorWrapper),
}

#[derive(Debug)]
#[register_py_operators_sampling]
pub enum SamplingOperatorDispatcher {
    PermutationSampling(PermutationSampling),
    RandomSamplingBinary(RandomSamplingBinary),
    RandomSamplingFloat(RandomSamplingFloat),
    RandomSamplingInt(RandomSamplingInt),
    CustomPySamplingOperatorWrapper(CustomPySamplingOperatorWrapper),
}

#[derive(Debug)]
#[register_py_operators_duplicates]
pub enum DuplicatesCleanerDispatcher {
    ExactDuplicatesCleaner(ExactDuplicatesCleaner),
    CloseDuplicatesCleaner(CloseDuplicatesCleaner),
    NoDuplicatesCleaner(NoDuplicatesCleaner),
}

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

#[pymethods]
impl PyInversionMutation {
    #[new]
    pub fn new() -> Self {
        Self {
            inner: InversionMutation,
        }
    }
}

#[pymethods]
impl PyUniformBinaryMutation {
    #[new]
    pub fn new(gene_mutation_rate: f64) -> Self {
        Self {
            inner: UniformBinaryMutation::new(gene_mutation_rate),
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

#[pymethods]
impl PyArithmeticCrossover {
    #[new]
    pub fn new() -> Self {
        Self {
            inner: ArithmeticCrossover,
        }
    }
}

#[pymethods]
impl PyTwoPointBinaryCrossover {
    #[new]
    pub fn new() -> Self {
        Self {
            inner: TwoPointBinaryCrossover,
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
