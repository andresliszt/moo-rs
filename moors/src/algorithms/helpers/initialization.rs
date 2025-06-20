use ndarray::ArrayView;

use crate::{
    algorithms::helpers::{context::AlgorithmContext, error::InitializationError},
    duplicates::PopulationCleaner,
    evaluator::{ConstraintsFn, Evaluator, FitnessFn},
    genetic::{D01, Population},
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
            .evaluate(genes)
            .map_err(InitializationError::from)?;
        // Validate first individual
        let individual = population.get(0);
        Self::check_fitness(&individual.fitness, context)?;
        Self::check_constraints(&individual.constraints, context)?;
        // this step is very important. All members of the population survive, because
        // we use num_survive = context.population_size, but this step is adding the ranking
        // and the survival scorer (if the algorithm needs them), so in the selection step
        // we have all we need. See: https://github.com/andresliszt/moo-rs/issues/145
        population = survivor.operate(population, context.population_size, rng, &context);
        Ok(population)
    }

    /// Validates that the fitness array length matches expected objectives.
    fn check_fitness<FDim>(
        fitness: &ArrayView<f64, FDim>,
        context: &AlgorithmContext,
    ) -> Result<(), InitializationError>
    where
        FDim: D01,
    {
        let expected = context.num_objectives;
        let actual = fitness.len();

        match actual == expected {
            true => Ok(()),
            false => Err(InitializationError::InvalidFitness(format!(
                "Expected {} fitness values, got {}",
                expected, actual
            ))),
        }
    }

    /// Validates constraints array length or absence against context.
    fn check_constraints<ConstrDim>(
        constraints: &ArrayView<f64, ConstrDim>,
        context: &AlgorithmContext,
    ) -> Result<(), InitializationError>
    where
        ConstrDim: D01,
    {
        let expected = context.num_constraints;
        let actual = constraints.len();
        match actual == expected {
            true => Ok(()),
            false => Err(InitializationError::InvalidConstraints(format!(
                "Expected {} constraints, got {}",
                expected, actual
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithms::helpers::context::AlgorithmContext;
    use crate::duplicates::ExactDuplicatesCleaner;
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

    fn no_constraints(genes: &Array2<f64>) -> Array2<f64> {
        let n = genes.nrows();
        Array2::zeros((n, 0))
    }

    #[test]
    fn initialize_succeeds_with_matching_shapes() {
        let sampler = RandomSamplingBinary::new();
        let mut survivor = Nsga2RankCrowdingSurvival::new();
        let seed = 123;
        let mut rng = MOORandomGenerator::new_from_seed(Some(seed));

        let context = AlgorithmContext::new(
            4, // num_vars
            8, // population_size
            3, // num_offsprings
            2, // num_objectives
            1, // num_iterations
            1, // num_constraints
            None, None,
        );

        let fitness_fn = |genes: &Array2<f64>| {
            dummy_fitness(genes, context.population_size, context.num_objectives)
        };
        let constraints_fn = |genes: &Array2<f64>| {
            dummy_constraints(genes, context.population_size, context.num_constraints)
        };
        let evaluator = Evaluator::new(fitness_fn, constraints_fn, false, None, None);
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

    #[test]
    fn initialize_fails_on_fitness_length_mismatch() {
        let sampler = RandomSamplingBinary::new();
        let mut survivor = Nsga2RankCrowdingSurvival::new();
        let duplicates_cleaner = ExactDuplicatesCleaner::new();
        let mut rng = MOORandomGenerator::new_from_seed(Some(123));

        let context = AlgorithmContext::new(
            4, 8, 3, 2, // expecting 2 objectives
            1, // iterations
            0, // no constraints
            None, None,
        );

        // returns only 1 column instead of 2
        let bad_fitness = |genes: &Array2<f64>| {
            dummy_fitness(genes, context.population_size, context.num_objectives - 1)
        };
        let evaluator = Evaluator::new(bad_fitness, no_constraints, false, None, None);

        let err = Initialization::initialize(
            &sampler,
            &mut survivor,
            &evaluator,
            &duplicates_cleaner,
            &mut rng,
            &context,
        )
        .unwrap_err();

        assert!(matches!(err, InitializationError::InvalidFitness(_)));
    }

    #[test]
    fn initialize_fails_on_constraints_length_mismatch() {
        let sampler = RandomSamplingBinary::new();
        let mut survivor = Nsga2RankCrowdingSurvival::new();
        let duplicates_cleaner = ExactDuplicatesCleaner::new();
        let mut rng = MOORandomGenerator::new_from_seed(Some(123));

        let context = AlgorithmContext::new(
            4, 8, 3, 2, // objectives
            1, // iterations
            1, // expecting 1 constraint
            None, None,
        );

        let fitness_fn = |genes: &Array2<f64>| {
            dummy_fitness(genes, context.population_size, context.num_objectives)
        };
        // returns zero columns instead of 1
        let bad_constraints = |genes: &Array2<f64>| {
            dummy_constraints(genes, context.population_size, context.num_constraints - 1)
        };
        let evaluator = Evaluator::new(fitness_fn, bad_constraints, false, None, None);

        let err = Initialization::initialize(
            &sampler,
            &mut survivor,
            &evaluator,
            &duplicates_cleaner,
            &mut rng,
            &context,
        )
        .unwrap_err();

        assert!(matches!(err, InitializationError::InvalidConstraints(_)));
    }

    #[test]
    fn initialize_fails_when_constraints_fn_is_none_but_expected_non_zero() {
        let sampler = RandomSamplingBinary::new();
        let mut survivor = Nsga2RankCrowdingSurvival::new();
        let duplicates_cleaner = ExactDuplicatesCleaner::new();
        let mut rng = MOORandomGenerator::new_from_seed(Some(42));

        let context = AlgorithmContext::new(
            3, // num_vars
            6, // population_size
            2, // num_offsprings
            1, // num_objectives
            1, // num_iterations
            2, // num_constraints (expected but no fn)
            None, None,
        );

        let fitness_fn = |genes: &Array2<f64>| {
            dummy_fitness(genes, context.population_size, context.num_objectives)
        };
        let evaluator = Evaluator::new(fitness_fn, no_constraints, false, None, None);

        let err = Initialization::initialize(
            &sampler,
            &mut survivor,
            &evaluator,
            &duplicates_cleaner,
            &mut rng,
            &context,
        )
        .unwrap_err();

        assert!(matches!(err, InitializationError::InvalidConstraints(_)));
    }
}
