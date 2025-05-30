//! # SPEA‑2 – Strength‑Pareto Evolutionary Algorithm II
//!
//! Implementation of
//! **Eckart Zitzler, Marco Laumanns & Lothar Thiele,
//! “SPEA2: Improving the Strength Pareto Evolutionary Algorithm”,
//! Technical Report 103, Computer Engineering and Networks Laboratory (TIK),
//! ETH Zürich, 2001.**
//!
//! SPEA‑2 maintains two populations (archive + current) and assigns each
//! individual a raw‑fitness value derived from *how many* solutions it dominates
//! (*strength*) and *by how many* it is dominated.  Diversity is preserved with
//! a *k‑nearest‑neighbour* density estimator.
//!
//! In *moors*, SPEA‑2 is wired from reusable operator bricks:
//!
//! * **Selection:** [`RankAndScoringSelection`] (only survival‑score is used)
//! * **Survival:**  [`Spea2KnnSurvival`] (strength + k‑NN density)
//! * **Crossover / Mutation / Sampling:** user‑provided via the builder.
//!
//! The default configuration keeps a secondary **archive** whose size equals
//! the main population; truncation is handled by the k‑NN density measure.
//!
use crate::{
    algorithms::{MultiObjectiveAlgorithm, MultiObjectiveAlgorithmError},
    duplicates::PopulationCleaner,
    genetic::{Constraints, D01, D12},
    operators::{
        CrossoverOperator, MutationOperator, SamplingOperator,
        selection::rank_and_survival_scoring_tournament::RankAndScoringSelection,
        survival::{SurvivalScoringComparison, spea2::Spea2KnnSurvival},
    },
};

use moors_macros::algorithm_builder;
use ndarray::Array2;

#[derive(Debug)]
pub struct Spea2<ConstrDim, S, Cross, Mut, F, G, DC>
where
    S: SamplingOperator,
    Cross: CrossoverOperator,
    Mut: MutationOperator,
    F: Fn(&Array2<f64>) -> Array2<f64>,
    G: Fn(&Array2<f64>) -> Constraints<ConstrDim>,
    DC: PopulationCleaner,
    ConstrDim: D12,
    <ConstrDim as ndarray::Dimension>::Smaller: D01,
{
    pub inner: MultiObjectiveAlgorithm<
        S,
        RankAndScoringSelection,
        Spea2KnnSurvival,
        Cross,
        Mut,
        F,
        G,
        DC,
        ConstrDim,
    >,
}

#[algorithm_builder]
impl<ConstrDim, S, Cross, Mut, F, G, DC> Spea2<ConstrDim, S, Cross, Mut, F, G, DC>
where
    S: SamplingOperator,
    Cross: CrossoverOperator,
    Mut: MutationOperator,
    F: Fn(&Array2<f64>) -> Array2<f64>,
    G: Fn(&Array2<f64>) -> Constraints<ConstrDim>,
    DC: PopulationCleaner,
    ConstrDim: D12,
    <ConstrDim as ndarray::Dimension>::Smaller: D01,
{
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        sampler: S,
        crossover: Cross,
        mutation: Mut,
        duplicates_cleaner: Option<DC>,
        fitness_fn: F,
        num_vars: usize,
        num_objectives: usize,
        num_constraints: usize,
        population_size: usize,
        num_offsprings: usize,
        num_iterations: usize,
        mutation_rate: f64,
        crossover_rate: f64,
        keep_infeasible: bool,
        verbose: bool,
        constraints_fn: Option<G>,
        // Optional lower and upper bounds for each gene.
        lower_bound: Option<f64>,
        upper_bound: Option<f64>,
        seed: Option<u64>,
    ) -> Result<Self, MultiObjectiveAlgorithmError> {
        // Define SPEA2 selector and survivor
        let survivor = Spea2KnnSurvival::new();
        // Selector operator uses scoring survival given by the raw fitness but it doesn't use rank
        let selector =
            RankAndScoringSelection::new(false, true, SurvivalScoringComparison::Maximize);
        // Define inner algorithm
        let algorithm = MultiObjectiveAlgorithm::new(
            sampler,
            selector,
            survivor,
            crossover,
            mutation,
            duplicates_cleaner,
            fitness_fn,
            num_vars,
            num_objectives,
            num_constraints,
            population_size,
            num_offsprings,
            num_iterations,
            mutation_rate,
            crossover_rate,
            keep_infeasible,
            verbose,
            constraints_fn,
            lower_bound,
            upper_bound,
            seed,
        )?;
        Ok(Self { inner: algorithm })
    }

    // Delegate methods from inner
    delegate_algorithm_methods!();
}
