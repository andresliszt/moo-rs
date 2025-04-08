use ndarray::Array2;

use crate::{
    algorithms::{MultiObjectiveAlgorithm, MultiObjectiveAlgorithmError},
    duplicates::PopulationCleaner,
    genetic::{PopulationConstraints, PopulationFitness, PopulationGenes},
    operators::{
        CrossoverOperator, MutationOperator, SamplingOperator,
        selection::rank_and_survival_scoring_tournament::RankAndScoringSelection,
        survival::rnsga2::Rnsga2ReferencePointsSurvival,
    },
};

// Define the RNSGA2
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
        n_vars: usize,
        population_size: usize,
        n_offsprings: usize,
        n_iterations: usize,
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
        let selector = RankAndScoringSelection::new();
        // Define inner algorithm
        let algorithm = MultiObjectiveAlgorithm::new(
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
        Ok(Self { inner: algorithm })
    }

    pub fn run(&mut self) -> Result<(), MultiObjectiveAlgorithmError> {
        self.inner.run()
    }
}

// delegate_inner_getters_and_run!(Nsga2);
