//! # R‑NSGA‑II – Reference‑Point‑Guided NSGA‑II
//!
//! Implementation inspired by
//! **R. Imada, H. Ishibuchi & Y. Nojima,
//! “Reference Point–Based NSGA‑II for Preference‑Based Many‑Objective
//! Optimization”, in *Proc. GECCO 2017*, pp. 1923‑1930.**
//!
//! R‑NSGA‑II biases the classic NSGA‑II ranking/crowding procedure toward
//! **user‑supplied reference points**, enabling preference‑based search without
//! altering the core fast non‑dominated sorting.  At each generation it:
//!
//! 1. Performs NSGA‑II non‑dominated sorting.
//! 2. Computes a *reference‑distance* score (to the closest point in the user
//!    set) **to be *minimized***.
//! 3. Uses that score—rather than crowding distance—as the secondary key
//!    during survival selection.
//!
//! In *moors*, R‑NSGA‑II is wired from reusable operator bricks:
//!
//! * **Selection:** [`RankAndScoringSelection`] (`use_rank = true`, scoring **minimised**)
//! * **Survival:**  [`Rnsga2ReferencePointsSurvival`] (rank + reference‑distance)
//! * **Crossover / Mutation / Sampling:** user‑provided via the builder.
//!
//! Pass your reference point matrix (`Array2<f64>`) and an `epsilon` tolerance
//! to [`Rnsga2::new`]; the survivor will treat individuals whose distance to a
//! point ≤ ε as equally preferred.
//!
use ndarray::Array2;

use crate::{
    algorithms::{MultiObjectiveAlgorithm, MultiObjectiveAlgorithmError},
    duplicates::PopulationCleaner,
    genetic::{PopulationConstraints, PopulationFitness, PopulationGenes},
    operators::{
        CrossoverOperator, MutationOperator, SamplingOperator,
        selection::rank_and_survival_scoring_tournament::RankAndScoringSelection,
        survival::{SurvivalScoringComparison, rnsga2::Rnsga2ReferencePointsSurvival},
    },
};

use moors_macros::algorithm_builder;

/// R‑NSGA‑II algorithm wrapper.
///
/// Thin façade around [`MultiObjectiveAlgorithm`] pre‑configured with
/// reference‑distance survival and a minimise‑the‑score selector.
///
/// * **Selection:** [`RankAndScoringSelection`] (`SurvivalScoringComparison::Minimize`)
/// * **Survival:**  [`Rnsga2ReferencePointsSurvival`]
/// * **Paper:** Imada et al. 2017 (*GECCO*)
///
/// Build via [`Rnsga2Builder`](crate::algorithms::Rnsga2Builder) or directly
/// with [`Rnsga2::new`]; then call `run()` and `population()` to retrieve the
/// preference‑biased Pareto set.
#[derive(Debug)]
pub struct Rnsga2<S, Cross, Mut, F, G, DC>
where
    S: SamplingOperator,
    Cross: CrossoverOperator,
    Mut: MutationOperator,
    F: Fn(&PopulationGenes) -> PopulationFitness,
    G: Fn(&PopulationGenes) -> PopulationConstraints,
    DC: PopulationCleaner,
{
    pub inner: MultiObjectiveAlgorithm<
        S,
        RankAndScoringSelection,
        Rnsga2ReferencePointsSurvival,
        Cross,
        Mut,
        F,
        G,
        DC,
    >,
}

#[algorithm_builder]
impl<S, Cross, Mut, F, G, DC> Rnsga2<S, Cross, Mut, F, G, DC>
where
    S: SamplingOperator,
    Cross: CrossoverOperator,
    Mut: MutationOperator,
    F: Fn(&PopulationGenes) -> PopulationFitness,
    G: Fn(&PopulationGenes) -> PopulationConstraints,
    DC: PopulationCleaner,
{
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        reference_points: Array2<f64>,
        epsilon: f64,
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
        // Define RNSGA2 selector and survivor
        let survivor = Rnsga2ReferencePointsSurvival::new(reference_points, epsilon);
        // RNSGA2 minimizes its scoring survival
        let selector =
            RankAndScoringSelection::new(true, true, SurvivalScoringComparison::Minimize);
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
