pub mod close;
pub mod exact;

pub use close::CloseDuplicatesCleaner;
pub use exact::ExactDuplicatesCleaner;

use std::fmt::Debug;

use crate::genetic::PopulationGenes;

/// A trait for removing duplicates (exact or close) from a population.
///
/// The `remove` method accepts an optional reference population.
/// If `None`, duplicates are computed within the population;
/// if provided, duplicates are determined by comparing each row in the population to all rows in the reference.
pub trait PopulationCleaner: Debug {
    fn remove(
        &self,
        population: &PopulationGenes,
        reference: Option<&PopulationGenes>,
    ) -> PopulationGenes;
}

/// A no-op cleaner for the “default” case:
#[derive(Debug, Default)]
pub struct NoDuplicatesCleaner;

impl PopulationCleaner for NoDuplicatesCleaner {
    fn remove(
        &self,
        _population: &PopulationGenes,
        _reference: Option<&PopulationGenes>,
    ) -> PopulationGenes {
        unimplemented!(
            "This is just for annotation when the duplicates cleaner is not set. See moors_macros::algorithm_builder"
        )
    }
}

impl PopulationCleaner for () {
    fn remove(
        &self,
        population: &PopulationGenes,
        _reference: Option<&PopulationGenes>,
    ) -> PopulationGenes {
        population.to_owned()
    }
}
