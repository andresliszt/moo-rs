//! # `algorithms` – High‑level evolutionary engines
//!
//! This module gathers concrete multi‑objective evolutionary algorithms built
//! on top of the generic runtime [`GeneticAlgorithm`].
//!
//! | Algorithm | Selector | Survivor | Builder type |
//! |-----------|----------|----------|--------------|
//! | **NSGA‑II** | [`RankAndScoringSelection`](crate::operators::selection::rank_and_survival_scoring_tournament::RankAndScoringSelection) | [`Nsga2RankCrowdingSurvival`](crate::operators::survival::nsga2::Nsga2RankCrowdingSurvival) | [`Nsga2Builder`](crate::algorithms::Nsga2Builder) |
//! | **NSGA‑III** | [`RandomSelection`](crate::operators::selection::random_tournament::RandomSelection) | [`Nsga3ReferencePointsSurvival`](crate::operators::survival::nsga3::Nsga3ReferencePointsSurvival) | [`Nsga3Builder`](crate::algorithms::Nsga3Builder) |
//! | **R‑NSGA‑II** | [`RankAndScoringSelection`](crate::operators::selection::rank_and_survival_scoring_tournament::RankAndScoringSelection) | [`Rnsga2ReferencePointsSurvival`](crate::operators::survival::rnsga2::Rnsga2ReferencePointsSurvival) | [`Rnsga2Builder`](crate::algorithms::Rnsga2Builder) |
//! | **SPEA‑2** | [`RankAndScoringSelection`](crate::operators::selection::rank_and_survival_scoring_tournament::RankAndScoringSelection) | [`Spea2KnnSurvival`](crate::operators::survival::spea2::Spea2KnnSurvival) | [`Spea2Builder`](crate::algorithms::Spea2Builder) |
//! | **AGE‑MOEA** | [`RankAndScoringSelection`](crate::operators::selection::rank_and_survival_scoring_tournament::RankAndScoringSelection) | [`AgeMoeaSurvival`](crate::operators::survival::agemoea::AgeMoeaSurvival) | [`AgeMoeaBuilder`](crate::algorithms::AgeMoeaBuilder) |
//! | **REVEA** | [`RandomSelection`](crate::operators::selection::random_tournament::RandomSelection) | [`ReveaReferencePointsSurvival`](crate::operators::survival::revea::ReveaReferencePointsSurvival) | [`ReveaBuilder`](crate::algorithms::ReveaBuilder) |
//!
//! Each public algorithm struct (e.g. [`Nsga2`]) is a thin wrapper around
//! `GeneticAlgorithm` that configures **its own selector, survivor and
//! any algorithm‑specific parameters**.  To make end‑user construction
//! ergonomic, we rely on the derive_builder crate. That derive_builder auto‑generates an
//! `*Builder` type following the *builder pattern* (`.foo(…)` setters +
//! `.build()`).
//!
//! ## Quick example: NSGA‑II
//!
//! ```rust,no_run, ignore
//! use ndarray::{Array1, Array2, Axis, stack};
//! use moors::{
//!     algorithms::{AlgorithmError, Nsga2Builder},
//!     duplicates::ExactDuplicatesCleaner,
//!     operators::{
//!         crossover::SinglePointBinaryCrossover, mutation::BitFlipMutation,
//!         sampling::RandomSamplingBinary,
//!     },
//! };
//!
//! /* ---- problem definition omitted for brevity ---- */
//!
//! # const WEIGHTS: [f64; 5] = [12.0, 2.0, 1.0, 4.0, 10.0];
//! # const VALUES:  [f64; 5] = [ 4.0, 2.0, 1.0, 5.0,  3.0];
//! # const CAPACITY: f64 = 15.0;
//! # fn fitness(_: &Array2<f64>) -> Array2<f64> { todo!() }
//! # fn constraints(_: &Array2<f64>) -> Array2<f64> { todo!() }
//!
//! fn main() -> Result<(), AlgorithmError> {
//!     let mut algorithm = Nsga2Builder::default()
//!         .fitness_fn(fitness)
//!         .constraints_fn(constraints)
//!         .sampler(RandomSamplingBinary::new())
//!         .crossover(SinglePointBinaryCrossover::new())
//!         .mutation(BitFlipMutation::new(0.5))
//!         .duplicates_cleaner(ExactDuplicatesCleaner::new())
//!         .num_vars(5)
//!         .population_size(100)
//!         ...                     // other setters such as rates / iterations
//!         .build()?;               // ← macro‑generated .build()
//!
//!     algorithm.run()?;
//!     println!("Done – final pop: {}", algorithm.population()?.len());
//!     Ok(())
//! }
//! ```
//!
//! ### Writing your **own** algorithm
//!
//! 1. **Pick / implement** a [`SelectionOperator`] and a [`SurvivalOperator`].
//! 2. Use them with helper macro `create_algorithm!`:
//!
//!    ```rust, ignore
//!    create_algorithm!(MyNewAlgorithm, NewSelector, NewSurvival);
//!    ```
//!
//! From there, a new algorithm struct will be created
//! ```rust, ignore
//!    pub struct MyAlgo<S, Cross, Mut, F, G, DC> {
//!        inner: GeneticAlgorithm<S, NewSelector, NewSurvival, Cross, Mut, F, G, DC>
//!    }
//!    ```
//! Also, the its respective builder is availble `MyAlgoBuilder`
//!
//! ## GeneticAlgorithm – generic core
//!
//! The struct [`GeneticAlgorithm`] is **not** intended to be used
//! directly by most users; it’s the reusable engine that handles initial
//! sampling, iterative evolution, evaluation and survivor selection.  Concrete
//! algorithms customise its behaviour exclusively through trait objects, so
//! they stay **zero‑cost abstractions** once monomorphised.

//! ---
//!
//! *Evolution is a mystery*  Feel free to open an issue or PR if you implement a new
//! operator or algorithm.

pub(in crate::algorithms) mod agemoea;
pub(in crate::algorithms) mod ibea;
pub(in crate::algorithms) mod nsga2;
pub(in crate::algorithms) mod nsga3;
pub(in crate::algorithms) mod revea;
pub(in crate::algorithms) mod rnsga2;
pub(in crate::algorithms) mod spea2;
