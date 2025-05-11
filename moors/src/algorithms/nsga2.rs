//! # NSGA‑II – Fast Elitist Multi‑Objective GA
//!
//! This implementation follows the seminal paper
//! **K. Deb, A. Pratap, S. Agarwal & T. Meyarivan,
//! “A Fast and Elitist Multi‑objective Genetic Algorithm: NSGA‑II”,**
//! *IEEE Transactions on Evolutionary Computation*, 6 (2): 182‑197 (2002).
//!
//! Key ideas implemented here:
//! 1. **Fast non‑dominated sorting** → assigns a Pareto rank *O(N log N + N²)*.
//! 2. **Crowding‑distance preservation** → density estimator used as a
//!    secondary key to maintain diversity without extra parameters.

//! In *moors*, NSGA‑II is wired from reusable operator bricks:
//!
//! * **Selection:** [`RankAndScoringSelection`]
//! * **Survival:**  [`Nsga2RankCrowdingSurvival`] (rank + crowding distance)
//! * **Crossover / Mutation / Sampling:** user‑provided via the builder.
//!
//! The public API exposes only high‑level controls (population size,
//! iteration budget, etc.); internal operator choices can be overridden if
//! you need custom behaviour.
use crate::{
    algorithms::{MultiObjectiveAlgorithm, MultiObjectiveAlgorithmError},
    duplicates::PopulationCleaner,
    genetic::{PopulationConstraints, PopulationFitness, PopulationGenes},
    operators::{
        CrossoverOperator, MutationOperator, SamplingOperator,
        selection::rank_and_survival_scoring_tournament::RankAndScoringSelection,
        survival::nsga2::Nsga2RankCrowdingSurvival,
    },
};

use moors_macros::algorithm_builder;

/// NSGA‑II algorithm wrapper.
///
/// This struct is a thin façade over [`MultiObjectiveAlgorithm`] preset with
/// the NSGA‑II survival and selection strategy.
///
/// * **Selection:** [`RankAndScoringSelection`]
/// * **Survival:**  [`Nsga2RankCrowdingSurvival`] (elitist, crowding‑distance)
///
/// Construct it with [`Nsga2Builder`](crate::algorithms::Nsga2Builder) or
/// directly via [`Nsga2::new`].  After building, call [`run`](MultiObjectiveAlgorithm::run)
/// and then [`population`](MultiObjectiveAlgorithm::population) to retrieve the
/// final non‑dominated set.
///
/// For algorithmic details, see:
/// K. Deb *et al.* (2002), *IEEE TEC 6 (2)*, 182‑197.
///
#[derive(Debug)]
pub struct Nsga2<S, Cross, Mut, F, G, DC>
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
        Nsga2RankCrowdingSurvival,
        Cross,
        Mut,
        F,
        G,
        DC,
    >,
}

#[algorithm_builder]
impl<S, Cross, Mut, F, G, DC> Nsga2<S, Cross, Mut, F, G, DC>
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
        // Define NSGA2 selector and survivor
        let survivor = Nsga2RankCrowdingSurvival::new();
        let selector = RankAndScoringSelection::default();
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
