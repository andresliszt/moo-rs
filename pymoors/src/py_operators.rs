use numpy::PyArrayMethods;
use pyo3::prelude::*;

use moors::operators::{NoCrossover, NoMutation, NoRepair, RepairOperator};
use moors::{
    ArithmeticCrossover, BitFlipMutation, CloseDuplicatesCleaner, CrossoverOperator,
    DisplacementMutation, ExactDuplicatesCleaner, ExponentialCrossover, GaussianMutation,
    InversionMutation, MutationOperator, NoDuplicatesCleaner, OrderCrossover, PermutationSampling,
    PopulationCleaner, RandomSamplingBinary, RandomSamplingFloat, RandomSamplingInt,
    SamplingOperator, ScrambleMutation, SimulatedBinaryCrossover, SinglePointBinaryCrossover,
    SwapMutation, TwoPointBinaryCrossover, UniformBinaryCrossover, UniformBinaryMutation,
    UniformRealMutation,
};
use ndarray::{Array2, ArrayViewMut1};

use pymoors_macros::{
    register_py_operators_crossover, register_py_operators_duplicates,
    register_py_operators_mutation, register_py_operators_sampling,
};

use crate::custom_py_operators::{
    CustomPyCrossoverOperatorWrapper, CustomPyMutationOperatorWrapper,
    CustomPyRepairOperatorWrapper, CustomPySamplingOperatorWrapper,
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
    UniformRealMutation(UniformRealMutation),
    NoMutation(NoMutation),

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
    NoCrossover(NoCrossover),

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

/// Python-facing `NoRepair`: explicit no-op repair (same as passing `repair=None`).
#[pyclass(name = "NoRepair", from_py_object)]
#[derive(Debug, Clone)]
pub struct PyNoRepair {
    pub inner: NoRepair,
}

#[pymethods]
impl PyNoRepair {
    #[new]
    pub fn new() -> Self {
        Self { inner: NoRepair }
    }
}

#[derive(Debug)]
pub enum RepairOperatorDispatcher {
    NoRepair(NoRepair),
    CustomPyRepairOperatorWrapper(CustomPyRepairOperatorWrapper),
}

impl RepairOperator for RepairOperatorDispatcher {
    fn repair(&self, individual: ArrayViewMut1<f64>) {
        match self {
            RepairOperatorDispatcher::NoRepair(inner) => inner.repair(individual),
            RepairOperatorDispatcher::CustomPyRepairOperatorWrapper(inner) => {
                inner.repair(individual)
            }
        }
    }

    fn operate(&self, population: &mut Array2<f64>) {
        match self {
            RepairOperatorDispatcher::NoRepair(inner) => inner.operate(population),
            RepairOperatorDispatcher::CustomPyRepairOperatorWrapper(inner) => {
                inner.operate(population)
            }
        }
    }
}

impl RepairOperatorDispatcher {
    /// Returns `None` when no repair operator was given, keeping the builder's default.
    pub fn from_python_operator(py_obj_opt: Option<Py<PyAny>>) -> PyResult<Option<Self>> {
        let Some(py_obj) = py_obj_opt else {
            return Ok(None);
        };
        Python::attach(|py| {
            if let Ok(extracted) = py_obj.extract::<PyNoRepair>(py) {
                return Ok(Some(RepairOperatorDispatcher::NoRepair(extracted.inner)));
            }
            Ok(Some(
                RepairOperatorDispatcher::CustomPyRepairOperatorWrapper(
                    py_obj.extract::<CustomPyRepairOperatorWrapper>(py)?,
                ),
            ))
        })
    }
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

    #[getter]
    pub fn gene_mutation_rate(&self) -> f64 {
        self.inner.gene_mutation_rate
    }
}

#[pymethods]
impl PyUniformRealMutation {
    #[new]
    pub fn new(gene_mutation_rate: f64, lower: f64, upper: f64) -> Self {
        Self {
            inner: UniformRealMutation::new(gene_mutation_rate, lower, upper),
        }
    }

    #[getter]
    pub fn gene_mutation_rate(&self) -> f64 {
        self.inner.gene_mutation_rate
    }
    #[getter]
    pub fn lower(&self) -> f64 {
        self.inner.lower
    }
    #[getter]
    pub fn upper(&self) -> f64 {
        self.inner.upper
    }
}

#[pymethods]
impl PyNoMutation {
    #[new]
    pub fn new() -> Self {
        Self {
            inner: NoMutation::new(),
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

#[pymethods]
impl PyNoCrossover {
    #[new]
    pub fn new() -> Self {
        Self {
            inner: NoCrossover::new(),
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
