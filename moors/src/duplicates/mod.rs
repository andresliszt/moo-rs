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

impl PopulationCleaner for () {
    fn remove(
        &self,
        population: &PopulationGenes,
        _reference: Option<&PopulationGenes>,
    ) -> PopulationGenes {
        population.to_owned()
    }
}
