//! # `controller` – Adaptive control hook for the generation loop
//!
//! An [`AdaptiveController`] is invoked once per generation, after survivors have
//! been selected, and may adjust `mutation_rate`/`crossover_rate` for the next
//! generation or request an early stop. This is the extension point meant for
//! e.g. classical adaptive-parameter-control heuristics, or an external
//! (possibly LLM-backed) policy wired in from Python.
use std::fmt;

use crate::algorithms::helpers::AlgorithmContext;
use crate::genetic::{D12, Population};

/// Adjustments requested by an [`AdaptiveController`] after observing a generation.
/// `None` fields leave the corresponding rate unchanged.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ControlSignal {
    pub mutation_rate: Option<f64>,
    pub crossover_rate: Option<f64>,
    pub stop: bool,
}

/// Observes the population produced at the end of each generation and may
/// request rate changes or an early stop for the next one.
pub trait AdaptiveController<FDim, ConstrDim>: fmt::Debug + Send + Sync
where
    FDim: D12,
    ConstrDim: D12,
{
    fn observe(
        &mut self,
        iteration: usize,
        population: &Population<FDim, ConstrDim>,
        context: &AlgorithmContext,
    ) -> ControlSignal;
}

/// Default controller that never changes rates and never stops early.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoController;

impl<FDim, ConstrDim> AdaptiveController<FDim, ConstrDim> for NoController
where
    FDim: D12,
    ConstrDim: D12,
{
    fn observe(
        &mut self,
        _iteration: usize,
        _population: &Population<FDim, ConstrDim>,
        _context: &AlgorithmContext,
    ) -> ControlSignal {
        ControlSignal::default()
    }
}
