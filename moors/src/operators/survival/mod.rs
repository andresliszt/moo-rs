pub mod moo;
pub mod soo;

use crate::{
    algorithms::AlgorithmContext,
    genetic::{D12, Population},
    random::RandomGenerator,
};

/// The base trait for **all** survival operators.
///
/// A survival operator takes a full `Population` and
/// returns the `num_survive` individuals that will move on to the next generation.
/// Algorithms that need custom survival logic (e.g. NSGA3’s reference-point logic)
/// implement this trait directly.
pub trait SurvivalOperator {
    type FDim: D12;

    /// Selects the individuals that will survive to the next generation.
    fn operate<ConstrDim>(
        &mut self,
        population: Population<Self::FDim, ConstrDim>,
        num_survive: usize,
        rng: &mut impl RandomGenerator,
        algorithm_context: &AlgorithmContext,
    ) -> Population<Self::FDim, ConstrDim>
    where
        ConstrDim: D12;
}
