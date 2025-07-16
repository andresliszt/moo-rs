use crate::{
    algorithms::helpers::{context::AlgorithmContext, error::InitializationError},
    duplicates::PopulationCleaner,
    evaluator::{ConstraintsFn, Evaluator, FitnessFn},
    genetic::Population,
    operators::{SamplingOperator, SurvivalOperator},
    random::RandomGenerator,
};

pub struct Initialization;

impl Initialization {
    /// Sample, clean duplicates, evaluate, and rank the initial population.
    pub fn initialize<S, Sur, DC, F, G>(
        sampler: &S,
        survivor: &mut Sur,
        evaluator: &Evaluator<F, G>,
        duplicates_cleaner: &DC,
        rng: &mut impl RandomGenerator,
        context: &AlgorithmContext,
    ) -> Result<Population<F::Dim, G::Dim>, InitializationError>
    where
        S: SamplingOperator,
        Sur: SurvivalOperator<FDim = F::Dim>,
        DC: PopulationCleaner,
        F: FitnessFn,
        G: ConstraintsFn,
    {
        // Get the initial genes
        let mut genes = sampler.operate(context.population_size, context.num_vars, rng);
        // If duplicates cleaner is passed then clean
        genes = duplicates_cleaner.remove(genes, None);
        // Do the first evaluation
        let mut population = evaluator
            .evaluate(genes, context.context_id)
            .map_err(InitializationError::from)?;
        // Validate first individual
        // this step is very important. All members of the population survive, because
        // we use num_survive = context.population_size, but this step is adding the ranking
        // and the survival scorer (if the algorithm needs them), so in the selection step
        // we have all we need. See: https://github.com/andresliszt/moo-rs/issues/145
        population = survivor.operate(population, context.population_size, rng);
        Ok(population)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithms::helpers::context::AlgorithmContextBuilder;
    use crate::duplicates::ExactDuplicatesCleaner;
    use crate::evaluator::EvaluatorBuilder;
    use crate::operators::{
        sampling::RandomSamplingBinary, survival::moo::nsga2::Nsga2RankCrowdingSurvival,
    };
    use crate::random::MOORandomGenerator;
    use ndarray::Array2;

    /// A dummy fitness function that returns an array of zeros
    /// with shape `(population_size, num_objectives)`.
    fn dummy_fitness(
        _genes: &Array2<f64>,
        population_size: usize,
        num_objectives: usize,
    ) -> Array2<f64> {
        Array2::zeros((population_size, num_objectives))
    }

    /// A dummy constraints function that returns an array of zeros
    /// with shape `(population_size, num_constraints)`.
    fn dummy_constraints(
        _genes: &Array2<f64>,
        population_size: usize,
        num_constraints: usize,
    ) -> Array2<f64> {
        Array2::zeros((population_size, num_constraints))
    }

    #[test]
    fn initialize_succeeds() {
        let sampler = RandomSamplingBinary::new();
        let mut survivor = Nsga2RankCrowdingSurvival::new();
        let seed = 123;
        let mut rng = MOORandomGenerator::new_from_seed(Some(seed));

        let context = AlgorithmContextBuilder::default()
            .num_vars(4)
            .population_size(8)
            .num_offsprings(3)
            .num_iterations(1)
            .build()
            .expect("Builder failed");

        let fitness_fn = |genes: &Array2<f64>, _context_id: usize| {
            dummy_fitness(genes, context.population_size, 2)
        };
        let constraints_fn = |genes: &Array2<f64>, _context_id: usize| {
            dummy_constraints(genes, context.population_size, 2)
        };
        let evaluator = EvaluatorBuilder::default()
            .fitness(fitness_fn)
            .constraints(constraints_fn)
            .keep_infeasible(false)
            .build()
            .expect("Builder failed");
        let duplicates_cleaner = ExactDuplicatesCleaner::new();

        let pop = Initialization::initialize(
            &sampler,
            &mut survivor,
            &evaluator,
            &duplicates_cleaner,
            &mut rng,
            &context,
        )
        .expect("should initialize successfully");

        assert!(
            pop.rank.is_some(),
            "rank should be set after initialization"
        );
        assert!(
            pop.survival_score.is_some(),
            "survival_score should be set after initialization"
        );
    }
}
