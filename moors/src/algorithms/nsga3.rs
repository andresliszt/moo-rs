use crate::{
    algorithms::{MultiObjectiveAlgorithm, MultiObjectiveAlgorithmError},
    duplicates::PopulationCleaner,
    genetic::{ConstraintsFn, FitnessFn},
    operators::{
        CrossoverOperator, MutationOperator, SamplingOperator,
        selection::random_tournament::RandomSelection,
        survival::nsga3::{Nsga3ReferencePoints, Nsga3ReferencePointsSurvival},
    },
};

// Define the NSGA-III algorithm
pub struct Nsga3<S, Cross, Mut, DC>
where
    S: SamplingOperator,
    Cross: CrossoverOperator,
    Mut: MutationOperator,
    DC: PopulationCleaner,
{
    pub inner:
        MultiObjectiveAlgorithm<S, RandomSelection, Nsga3ReferencePointsSurvival, Cross, Mut, DC>,
}

impl<S, Cross, Mut, DC> Nsga3<S, Cross, Mut, DC>
where
    S: SamplingOperator,
    Cross: CrossoverOperator,
    Mut: MutationOperator,
    DC: PopulationCleaner,
{
    pub fn new(
        reference_points: Nsga3ReferencePoints,
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

        Ok(Nsga3 { inner })
    }

    // Delegate methods from inner
    delegate_algorithm_methods!();
}
