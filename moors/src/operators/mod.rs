//! # `operators` – Building Blocks for Evolution
//!
//! Every evolutionary algorithm in **moors** is assembled from a *pipeline* of
//! interchangeable **operators**.  Each operator focuses on a single stage of
//! the evolutionary cycle—sampling an initial population, creating offspring,
//! selecting parents, evaluating diversity, and so on.
//!
//! The common super‑trait [`GeneticOperator`] provides a tiny reflection hook
//! (`name()`) so algorithms and loggers can identify each concrete operator at
//! runtime.  Beyond that, each sub‑trait defines the behaviour expected for its
//! stage:
//!
//! | Trait | Purpose | Typical Implementations |
//! |-------|---------|-------------------------|
//! | [`SamplingOperator`]   | Generate an initial population of genomes. | `RandomSamplingBinary`, `RandomSamplingFloat`, … |
//! | [`CrossoverOperator`]  | Combine two (or more) parents to create offspring. | `SinglePointBinaryCrossover`, `SimulatedBinaryCrossover`, ... |
//! | [`MutationOperator`]   | Apply random variation to a single genome *in‑place*. | `BitFlipMutation`, `GaussianMutation`, `ScrambleMutation`, ... |
//! | [`SelectionOperator`]  | Choose parents via tournaments, fitness‑proportionate schemes, etc. | `RankAndScoringSelection`, `RandomSelection`, ... |
//! | [`SurvivalOperator`]   | Decide which individuals survive to the next generation. | `FrontsAndRankingBasedSurvival`, `Nsga3ReferencePointsSurvival`, ... |
//!
//! ```rust
//! use moors::{
//!     genetic::IndividualGenesMut,
//!     operators::{GeneticOperator, MutationOperator},
//!     random::RandomGenerator,
//! };
//!
//! /// Flips each binary gene with probability `gene_mutation_rate`.
//! #[derive(Debug, Clone)]
//! pub struct MyMutation {
//!     pub gene_mutation_rate: f64,
//! }
//!
//! impl MyMutation {
//!     pub fn new(rate: f64) -> Self {
//!         Self { gene_mutation_rate: rate }
//!     }
//! }
//!
//! impl GeneticOperator for MyMutation {
//!     fn name(&self) -> String {
//!         "MyMutation".into()
//!     }
//! }
//!
//! impl MutationOperator for MyMutation {
//!     fn mutate<'a>(
//!         &self,
//!         mut individual: IndividualGenesMut<'a>,
//!         rng: &mut impl RandomGenerator,
//!     ) {
//!         for gene in individual.iter_mut() {
//!             if rng.gen_bool(self.gene_mutation_rate) {
//!                 *gene = if *gene == 0.0 { 1.0 } else { 0.0 };
//!             }
//!         }
//!     }
//! }
//! ```
//!
//! Once compiled, you can pass `MyMutation` into any `*Builder` just like the
//! built‑in operators.
//!
//! ## Module layout
//!
//! * [`crossover`]   – crossover operators
//! * [`mutation`]    – mutation operators
//! * [`sampling`]    – initial population generators
//! * [`selection`]   – parent‑selection strategies
//! * [`survival`]    – survival / environmental‑selection strategies
//! * [`evolve`]      – glue code to run *selection → crossover → mutation*
//!
//! ---
//!
//! **Tip:** Because every operator is trait‑based, you can unit‑test them in
//! isolation or swap them at runtime to benchmark different evolutionary
//! dynamics without touching your problem‑specific code or algorithm builder.

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
pub use survival::{FrontsAndRankingBasedSurvival, SurvivalOperator};

pub trait GeneticOperator: Debug {
    fn name(&self) -> String;
}
