//! # REVEA – Reference‑Vector‑Guided Evolutionary Algorithm
//!
//! Implementation of
//! **Ran Cheng, Yaochu Jin, Markus Olhofer & Bernhard Sendhoff,
//! “A Reference Vector Guided Evolutionary Algorithm for Many‑Objective
//! Optimization”, IEEE Transactions on Evolutionary Computation 20 (5):
//! 773‑791 (2016).**
//!
//! REVEA tackles many‑objective problems by steering the population with a
//! **dynamic set of reference vectors**.  Each generation it:
//!
//! 1. Performs non‑dominated sorting (like NSGA‑II/III).
//! 2. Associates every solution to its nearest reference vector (angle‑based).
//! 3. Uses a **shift‑based density estimator** to pick survivors, balancing
//!    convergence and diversity.
//! 4. Periodically *rotates* or *re‑scales* the reference vectors
//!    (`frequency`) so the search can adapt to the true Pareto front shape.
//!
//! In *moors*, REVEA is wired from reusable operator bricks:
//!
//! * **Selection:** [`RandomSelection`] (uniform binary tournament)
//! * **Survival:**  [`ReveaReferencePointsSurvival`]
//! * **Crossover / Mutation / Sampling:** user‑provided via the builder.
//!
//! You pass an initial `Array2<f64>` of reference vectors plus two hyper‑
//! parameters to [`Revea::new`]:
//!
//! * `alpha`     – controls the rotation angle when updating vectors.
//! * `frequency` – how often (in generations) the reference set is refreshed.
//!
use ndarray::Array2;

use crate::{
    algorithms::{MultiObjectiveAlgorithm, MultiObjectiveAlgorithmError},
    duplicates::PopulationCleaner,
    genetic::{PopulationConstraints, PopulationFitness, PopulationGenes},
    operators::{
        CrossoverOperator, MutationOperator, SamplingOperator,
        selection::random_tournament::RandomSelection,
        survival::revea::ReveaReferencePointsSurvival,
    },
};

use moors_macros::algorithm_builder;

/// REVEA algorithm wrapper.
///
/// Thin façade around [`MultiObjectiveAlgorithm`] pre‑configured with
/// reference‑vector survival and random parent selection.
///
/// * **Selection:** [`RandomSelection`]
/// * **Survival:**  [`ReveaReferencePointsSurvival`]
/// * **Paper:** Ran Cheng et al. 2016 (*IEEE TEC* 20 (5): 773–791)
///
/// Build via [`ReveaBuilder`](crate::algorithms::ReveaBuilder) or directly with
/// [`Revea::new`], then call `run()` and `population()` to obtain the final
/// Pareto approximation.
#[derive(Debug)]
pub struct Revea<S, Cross, Mut, F, G, DC>
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
        ReveaReferencePointsSurvival,
        Cross,
        Mut,
        F,
        G,
        DC,
    >,
}

#[algorithm_builder]
impl<S, Cross, Mut, F, G, DC> Revea<S, Cross, Mut, F, G, DC>
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
        alpha: f64,
        frequency: f64,
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
        // Define REVEA selector and survivor
        let survivor = ReveaReferencePointsSurvival::new(reference_points, alpha, frequency);
        let selector = RandomSelection::new();
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
