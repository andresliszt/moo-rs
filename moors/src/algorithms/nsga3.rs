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

// Define the NSGA-III algorithm
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
