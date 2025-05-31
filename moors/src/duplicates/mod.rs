//! # `duplicates` – Keeping the Gene Pool Diverse
//!
//! Excessive duplicates can cripple an evolutionary algorithm: the search
//! spends generations evaluating essentially the **same** individual, genetic
//! diversity drops, and the population risks converging to sub‑optimal regions
//! of the design space.
//!
//! The **`duplicates`** module provides pluggable strategies—called
//! [`PopulationCleaner`]s—to detect and discard repeated or near‑repeated
//! genomes before the next generation starts.
//!
//! | Cleaner | Suitable for | Criterion | Complexity |
//! |---------|--------------|-----------|-------------|
//! | [`ExactDuplicatesCleaner`] | Binary / discrete genomes <br>(e.g. 0/1 strings) | Two individuals are duplicates **iff** every gene is bit‑wise identical. | `O(N log N)` via hashing |
//! | [`CloseDuplicatesCleaner`] | Real‑valued or mixed genomes | Two individuals are duplicates if their **Euclidean distance ≤ ε** (configurable). | `O(N²)` naïve, but *N* is typically pruned first |
//!
//! Implementations must return a **new** `PopulationGenes` with duplicates
//! filtered out; they never mutate the input arrays in‑place, allowing the
//! caller to decide whether to reuse or drop the originals.
//!
//! ### Quick example
//!
//! ```rust, ignore
//! use moors::duplicates::ExactDuplicatesCleaner;
//! use moors::genetic::PopulationGenes;
//!
//! let population: PopulationGenes = /* ... */;
//! let cleaner = ExactDuplicatesCleaner::new();
//! let unique = cleaner.remove(&population, None);
//! println!("Removed {} duplicates", population.len() - unique.len());
//! ```
//!
//! In continuous domains you would choose `CloseDuplicatesCleaner` instead and
//! configure the `epsilon` threshold when you construct it.
//!
//! ### No‑op marker
//!
//! [`NoDuplicatesCleaner`] exists only as an *annotational* placeholder when no
//! cleaning is desired.  Attempting to call its `remove` method will panic.
//!
//! ---
//!
//! **Tip:** Combine a `PopulationCleaner` with diversity‑aware selection and
//! survival operators to further reduce the risk of premature convergence.

pub mod close;
pub mod exact;

pub use close::CloseDuplicatesCleaner;
pub use exact::ExactDuplicatesCleaner;

use ndarray::Array2;

/// A trait for removing duplicates (exact or close) from a population.
///
/// The `remove` method accepts an optional reference population.
/// If `None`, duplicates are computed within the population;
/// if provided, duplicates are determined by comparing each row in the population to all rows in the reference.
pub trait PopulationCleaner {
    fn remove(&self, population: &Array2<f64>, reference: Option<&Array2<f64>>) -> Array2<f64>;
}

/// A no-op cleaner for the “default” case:
#[derive(Debug, Default)]
pub struct NoDuplicatesCleaner;

impl PopulationCleaner for NoDuplicatesCleaner {
    fn remove(&self, _population: &Array2<f64>, _reference: Option<&Array2<f64>>) -> Array2<f64> {
        unimplemented!(
            "This is just for annotation when the duplicates cleaner is not set. See moors_macros::algorithm_builder"
        )
    }
}
