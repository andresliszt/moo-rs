use crate::{
    algorithms::{MultiObjectiveAlgorithm, MultiObjectiveAlgorithmError},
    duplicates::PopulationCleaner,
    genetic::{ConstraintsFn, FitnessFn},
    operators::{
        CrossoverOperator, MutationOperator, SamplingOperator,
        selection::rank_and_survival_scoring_tournament::RankAndScoringSelection,
        survival::agemoea::AgeMoeaSurvival,
    },
};

// Define the AGEMOEA algorithm
pub struct AgeMoea<S, Cross, Mut, DC>
where
    S: SamplingOperator,
    Cross: CrossoverOperator,
    Mut: MutationOperator,
    DC: PopulationCleaner,
{
    pub inner: MultiObjectiveAlgorithm<S, RankAndScoringSelection, AgeMoeaSurvival, Cross, Mut, DC>,
}

impl<S, Cross, Mut, DC> AgeMoea<S, Cross, Mut, DC>
where
    S: SamplingOperator,
    Cross: CrossoverOperator,
    Mut: MutationOperator,
    DC: PopulationCleaner,
{
    pub fn new(
        sampler: S,
        crossover: Cross,
        mutation: Mut,
        duplicates_cleaner: Option<DC>,
        fitness_fn: FitnessFn,
        n_vars: usize,
        population_size: usize,
        n_offsprings: usize,
        n_iterations: usize,
        mutation_rate: f64,
        crossover_rate: f64,
        keep_infeasible: bool,
        verbose: bool,
        constraints_fn: Option<ConstraintsFn>,
        lower_bound: Option<f64>,
        upper_bound: Option<f64>,
        seed: Option<u64>,
    ) -> Result<Self, MultiObjectiveAlgorithmError> {
        // Define AGEMOEA selector and survivor
        let selector = RankAndScoringSelection::new();
        let survivor = AgeMoeaSurvival::new();

        // Build the algorithm.
        let inner = MultiObjectiveAlgorithm::new(
            sampler,
            selector,
            survivor,
            crossover,
            mutation,
            duplicates_cleaner,
            fitness_fn,
            n_vars,
            population_size,
            n_offsprings,
            n_iterations,
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

    delegate_algorithm_methods!();
}
