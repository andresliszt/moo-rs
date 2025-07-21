//! # moors
//!
//! <div align="center">
//! <strong>Multi‑Objective Optimization in Pure Rust</strong><br>
//! Fast, extensible evolutionary algorithms with first‑class ndarray support.
//! </div>
//!
//! ---
//!
//! ## Overview
//!
//! `moors` provides a battery of evolutionary algorithms for solving *multi‑objective*
//! optimization problems.  The core goals are:
//!
//! * **Performance** – minimal allocations, Rayon‑powered parallel loops where it matters.
//! * **Extensibility** – every operator (sampling, crossover, mutation, selection,
//!   survival) is pluggable via pure Rust traits.
//!
//! Currently implemented algorithms
//!
//! | Family | Algorithms |
//! |--------|------------|
//! | NSGA   | **NSGA‑II**, NSGA‑III, RNSGA‑II |
//! | SPEA   | SPEA‑2 |
//! | Others | AGE‑MOEA, REVEA *(WIP)* |
//!
//! ## Quick start
//!
//! ```rust,no_run
//! use ndarray::{Array1, Array2, Axis, stack};
//!
//! use moors::{
//!     algorithms::{AlgorithmError, Nsga2Builder},
//!     duplicates::ExactDuplicatesCleaner,
//!     operators::{SinglePointBinaryCrossover, BitFlipMutation, RandomSamplingBinary},
//! };
//!
//! // ----- problem data (0/1 knapsack) -------------------------------------
//! const WEIGHTS: [f64; 5] = [12.0, 2.0, 1.0, 4.0, 10.0];
//! const VALUES:  [f64; 5] = [ 4.0, 2.0, 1.0, 5.0,  3.0];
//! const CAPACITY: f64 = 15.0;
//!
//! /// Multi‑objective fitness ⇒ [−total_value, total_weight]
//! fn fitness(pop_genes: &Array2<f64>) -> Array2<f64> {
//!     let w = Array1::from_vec(WEIGHTS.into());
//!     let v = Array1::from_vec(VALUES.into());
//!     let total_v = pop_genes.dot(&v);
//!     let total_w = pop_genes.dot(&w);
//!     stack(Axis(1), &[(-&total_v).view(), total_w.view()]).unwrap()
//! }
//!
//! /// Single inequality constraint ⇒ total_weight − CAPACITY ≤ 0
//! fn constraints(pop_genes: &Array2<f64>) -> Array1<f64> {
//!     let w = Array1::from_vec(WEIGHTS.into());
//!     pop_genes.dot(&w) - CAPACITY
//! }
//!
//! fn main() -> Result<(), AlgorithmError> {
//!     let mut algo = Nsga2Builder::default()
//!         .fitness_fn(fitness)
//!         .constraints_fn(constraints)
//!         .sampler(RandomSamplingBinary::new())
//!         .crossover(SinglePointBinaryCrossover::new())
//!         .mutation(BitFlipMutation::new(0.5))
//!         .duplicates_cleaner(ExactDuplicatesCleaner::new())
//!         .num_vars(5)
//!         .population_size(100)
//!         .crossover_rate(0.9)
//!         .mutation_rate(0.1)
//!         .num_offsprings(32)
//!         .num_iterations(200)
//!         .build()?;
//!
//!     algo.run()?;
//!     println!("Final population size: {}", algo.population()?.len());
//!     Ok(())
//! }
//! ```
//!
//! ## Module layout
//!
//! * [`algorithms`](crate::algorithms) – high‑level algorithm builders
//! * [`operators`](crate::operators)  – sampling, crossover, mutation, selection, survival
//! * [`genetic`](crate::genetic)      – core data types (`Individual`, `Population`, …)
//! * [`evaluator`](crate::evaluator)  – fitness + constraints evaluation pipeline
//! * [`random`](crate::random)        – pluggable RNG abstraction
//! * [`duplicates`](crate::duplicates) – duplicate‑handling strategies
//!
//! ---

extern crate core;
#[macro_use]
extern crate paste;

pub mod algorithms;
pub mod duplicates;
pub mod evaluator;
pub mod genetic;
pub(crate) mod helpers;
pub mod non_dominated_sorting;
pub mod operators;
mod private;
pub mod random;
pub use algorithms::{
    AgeMoea, AgeMoeaBuilder, AlgorithmBuilder, AlgorithmBuilderError, AlgorithmError,
    GeneticAlgorithm, InitializationError, Nsga2, Nsga2Builder, Nsga3, Nsga3Builder, Revea,
    ReveaBuilder, Rnsga2, Rnsga2Builder, Spea2, Spea2Builder,
};
pub use duplicates::{
    CloseDuplicatesCleaner, ExactDuplicatesCleaner, NoDuplicatesCleaner, PopulationCleaner,
};
pub use evaluator::{ConstraintsFn, EvaluatorError, FitnessFn, NoConstraints};
pub use genetic::{
    Individual, IndividualMOO, IndividualSOO, Population, PopulationMOO, PopulationSOO,
};
pub use helpers::linalg::cross_euclidean_distances;
pub use operators::selection;
pub use operators::survival;
pub use operators::{
    AgeMoeaSurvival, ArithmeticCrossover, BitFlipMutation, CrossoverOperator,
    DanAndDenisReferencePoints, DisplacementMutation, ExponentialCrossover,
    FrontsAndRankingBasedSurvival, GaussianMutation, InversionMutation, MutationOperator,
    Nsga2RankCrowdingSurvival, Nsga3ReferencePoints, Nsga3ReferencePointsSurvival, OrderCrossover,
    PermutationSampling, RandomSamplingBinary, RandomSamplingFloat, RandomSamplingInt,
    RandomSelectionMOO, RankAndScoringSelectionMOO, ReveaReferencePointsSurvival,
    Rnsga2ReferencePointsSurvival, SamplingOperator, ScrambleMutation, SelectionOperator,
    SimulatedBinaryCrossover, SinglePointBinaryCrossover, Spea2KnnSurvival,
    StructuredReferencePoints, SurvivalOperator, SwapMutation, TwoPointBinaryCrossover,
    UniformBinaryCrossover, UniformBinaryMutation, UniformRealMutation, evolve::EvolveError,
};
pub use random::{MOORandomGenerator, NoopRandomGenerator, RandomGenerator, TestDummyRng};
