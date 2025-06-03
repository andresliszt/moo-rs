use crate::{
    algorithms::moo::{AlgorithmError, GeneticAlgorithmMOO},
    duplicates::PopulationCleaner,
    genetic::{Constraints, D01, D12},
    operators::{
        CrossoverOperator, MutationOperator, SamplingOperator,
        selection::moo::rank_and_survival_scoring_tournament::RankAndScoringSelection,
        survival::moo::AgeMoeaSurvival,
    },
};

use moors_macros::algorithm_builder;
use ndarray::Array2;

// Define the AGEMOEA algorithm
#[derive(Debug)]
pub struct AgeMoea<ConstrDim, S, Cross, Mut, F, G, DC>
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
    pub inner: GeneticAlgorithmMOO<
        S,
        RankAndScoringSelection,
        AgeMoeaSurvival,
        Cross,
        Mut,
        F,
        G,
        DC,
        ConstrDim,
    >,
}

#[algorithm_builder]
impl<ConstrDim, S, Cross, Mut, F, G, DC> AgeMoea<ConstrDim, S, Cross, Mut, F, G, DC>
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
        lower_bound: Option<f64>,
        upper_bound: Option<f64>,
        seed: Option<u64>,
    ) -> Result<Self, AlgorithmError> {
        // Define AGEMOEA selector and survivor
        let selector = RankAndScoringSelection::default();
        let survivor = AgeMoeaSurvival::new();

        // Build the algorithm.
        let inner = GeneticAlgorithmMOO::new(
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

        Ok(AgeMoea { inner })
    }

    // Delegate methods from inner
    delegate_algorithm_methods!();
}
