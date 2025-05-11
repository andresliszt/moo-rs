//! # NSGA‑III – Reference‑Point‑Based Many‑Objective GA
//!
//! Implementation of
//! **K. Deb & H. Jain,
//! “An Evolutionary Many‑Objective Optimization Algorithm Using
//! Reference‑Point‑Based Nondominated Sorting Approach, Part I:
//! Solving Problems with Box Constraints”,
//! IEEE Transactions on Evolutionary Computation 18 (4): 577‑601 (2014).**
//!
//! NSGA‑III extends NSGA‑II to many‑objective problems (≥ 3 objectives) by
//! replacing crowding‑distance with a *reference‑point association* that drives
//! the population towards a well‑spread Pareto front.
//!
//! In *moors*, NSGA‑III is wired from reusable operator bricks:
//!
//! * **Selection:** [`RandomSelection`] (uniform binary tournament)
//! * **Survival:**  [`Nsga3ReferencePointsSurvival`] (rank + reference‑point niching)
//! * **Crossover / Mutation / Sampling:** user‑provided via the builder.
//!
//! You supply the *reference points*—typically generated with
//! [`Nsga3ReferencePoints::from_simplex_lattice`] or a custom constructor—and
//! the algorithm handles association and niche preservation automatically.

use crate::{
    algorithms::{MultiObjectiveAlgorithm, MultiObjectiveAlgorithmError},
    duplicates::PopulationCleaner,
    genetic::{PopulationConstraints, PopulationFitness, PopulationGenes},
    operators::{
        CrossoverOperator, MutationOperator, SamplingOperator,
        selection::random_tournament::RandomSelection,
        survival::nsga3::{Nsga3ReferencePoints, Nsga3ReferencePointsSurvival},
    },
};

use moors_macros::algorithm_builder;

/// NSGA‑III algorithm wrapper.
///
/// Thin façade around [`MultiObjectiveAlgorithm`] pre‑configured with
/// *reference‑point* survival and random parent selection.
///
/// * **Selection:** [`RandomSelection`]
/// * **Survival:**  [`Nsga3ReferencePointsSurvival`]
/// * **Paper:** Deb & Jain 2014 (*IEEE TEC* 18 (4): 577‑601)
///
/// Construct with [`Nsga3Builder`](crate::algorithms::Nsga3Builder) or
/// directly via [`Nsga3::new`]; then call `run()` and `population()` to
/// retrieve the final Pareto‑optimal set.
#[derive(Debug)]
pub struct Nsga3<S, Cross, Mut, F, G, DC>
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
        RandomSelection,
        Nsga3ReferencePointsSurvival,
        Cross,
        Mut,
        F,
        G,
        DC,
    >,
}

#[algorithm_builder]
impl<S, Cross, Mut, F, G, DC> Nsga3<S, Cross, Mut, F, G, DC>
where
    S: SamplingOperator,
    Cross: CrossoverOperator,
    Mut: MutationOperator,
    F: Fn(&PopulationGenes) -> PopulationFitness,
    G: Fn(&PopulationGenes) -> PopulationConstraints,
    DC: PopulationCleaner,
{
    pub fn new(
        reference_points: Nsga3ReferencePoints,
        sampler: S,
        crossover: Cross,
        mutation: Mut,
        duplicates_cleaner: Option<DC>,
        fitness_fn: F,
        num_vars: usize,
        num_constraints: usize,
        num_objectives: usize,
        population_size: usize,
        num_offsprings: usize,
        num_iterations: usize,
        mutation_rate: f64,
        crossover_rate: f64,
        keep_infeasible: bool,
        verbose: bool,
        constraints_fn: Option<G>,
        lower_bound: Option<f64>,
        upper_bound: Option<f64>,
        seed: Option<u64>,
    ) -> Result<Self, MultiObjectiveAlgorithmError> {
        // Define NSGA3 selector and survivor
        let selector = RandomSelection::new();
        let survivor = Nsga3ReferencePointsSurvival::new(reference_points);

        // Build the algorithm.
        let inner = MultiObjectiveAlgorithm::new(
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

        Ok(Nsga3 { inner })
    }

    // Delegate methods from inner
    delegate_algorithm_methods!();
}
