use std::fmt::Debug;

pub mod crossover;
pub mod evolve;
pub mod mutation;
pub mod sampling;
pub mod selection;
pub mod survival;

pub use crossover::CrossoverOperator;
pub use evolve::{Evolve, EvolveError};
pub use mutation::MutationOperator;
pub use sampling::SamplingOperator;
pub use selection::SelectionOperator;
pub use survival::SurvivalOperator;

pub trait GeneticOperator: Debug {
    fn name(&self) -> String;
}
