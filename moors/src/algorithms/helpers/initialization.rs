use std::{error::Error, fmt};

use crate::{
    algorithms::helpers::context::AlgorithmContext,
    duplicates::PopulationCleaner,
    evaluator::{Evaluator, EvaluatorError},
    genetic::{
        FrontsExt, Individual, Population, PopulationConstraints, PopulationFitness,
        PopulationGenes,
    },
    non_dominated_sorting::build_fronts,
    operators::{SamplingOperator, SurvivalOperator},
    random::RandomGenerator,
};

/// Errors that can occur during initialization of the population.
#[derive(Debug)]
pub enum InitializationError {
    /// Error from the evaluator.
    Evaluator(EvaluatorError),
    /// Fitness array length does not match number of objectives.
    InvalidFitness(String),
    /// Constraints array length mismatch or unexpected absence.
    InvalidConstraints(String),
}

impl fmt::Display for InitializationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            InitializationError::Evaluator(e) => {
                write!(f, "Error during evaluation in initialization: {}", e)
            }
            InitializationError::InvalidFitness(msg) => write!(f, "Invalid fitness setup: {}", msg),
            InitializationError::InvalidConstraints(msg) => {
                write!(f, "Invalid constraints setup: {}", msg)
            }
        }
    }
}

impl From<EvaluatorError> for InitializationError {
    fn from(e: EvaluatorError) -> Self {
        InitializationError::Evaluator(e)
    }
}

impl Error for InitializationError {}

pub struct Initialization;

impl Initialization {
    /// Sample, clean duplicates, evaluate, and rank the initial population.
    pub fn initialize<S, Sur, DC, F, G>(
        sampler: &S,
        survivor: &Sur,
        evaluator: &Evaluator<F, G>,
        duplicates_cleaner: &Option<DC>,
        rng: &mut dyn RandomGenerator,
        context: &AlgorithmContext,
    ) -> Result<Population, InitializationError>
    where
        S: SamplingOperator,
        Sur: SurvivalOperator,
        DC: PopulationCleaner,
        F: Fn(&PopulationGenes) -> PopulationFitness,
        G: Fn(&PopulationGenes) -> PopulationConstraints,
    {
        // Get the initial genes
        let mut genes = sampler.operate(context.population_size, context.num_vars, rng);
        // If duplicates cleaner is passed then clean
        if let Some(cleaner) = duplicates_cleaner {
            genes = cleaner.remove(&genes, None);
        }
        // Do the first evaluation
        let population = evaluator
            .evaluate(genes)
            .map_err(InitializationError::from)?;

        // Validate first individual
        let individual = population.get(0);
        Self::check_fitness(&individual, context)?;
        Self::check_constraints(&individual, context)?;

        // Build the fronts to assign the ranking
        let mut fronts = build_fronts(population, context.population_size);
        // Assign the survivor scorer to the fronts
        survivor.set_survival_score(&mut fronts, rng, &context);
        // Return the initial population with ranking and survivor scorer if provided
        Ok(fronts.to_population())
    }

    /// Validates that the fitness array length matches expected objectives.
    fn check_fitness(
        individual: &Individual,
        context: &AlgorithmContext,
    ) -> Result<(), InitializationError> {
        let expected = context.num_objectives;
        let actual = individual.fitness.len();

        match actual == expected {
            true => Ok(()),
            false => Err(InitializationError::InvalidFitness(format!(
                "Expected {} fitness values, got {}",
                expected, actual
            ))),
        }
    }

    /// Validates constraints array length or absence against context.
    fn check_constraints(
        individual: &Individual,
        context: &AlgorithmContext,
    ) -> Result<(), InitializationError> {
        let expected = context.num_constraints;
        match individual.constraints.as_ref() {
            Some(constraints) if constraints.len() == expected => Ok(()),
            Some(constraints) => Err(InitializationError::InvalidConstraints(format!(
                "Expected {} constraints, got {}",
                expected,
                constraints.len()
            ))),
            None if expected == 0 => Ok(()),
            None => Err(InitializationError::InvalidConstraints(format!(
                "Expected {} constraints, but got none",
                expected
            ))),
        }
    }
}
