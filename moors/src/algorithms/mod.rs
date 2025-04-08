use std::marker::PhantomData;

use ndarray::{Axis, concatenate};
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::error::Error;
use std::fmt;

use crate::{
    duplicates::PopulationCleaner,
    evaluator::{Evaluator, EvaluatorError},
    genetic::{Population, PopulationConstraints, PopulationFitness, PopulationGenes},
    helpers::printer::print_minimum_objectives,
    operators::{
        CrossoverOperator, Evolve, EvolveError, MutationOperator, SamplingOperator,
        SelectionOperator, SurvivalOperator,
    },
    random::MOORandomGenerator,
};

mod agemoea;
mod nsga2;
mod nsga3;
mod revea;
mod rnsga2;

pub use agemoea::AgeMoea;
pub use nsga2::Nsga2;
pub use nsga3::Nsga3;
pub use revea::Revea;
pub use rnsga2::Rnsga2;

#[derive(Debug)]
pub enum MultiObjectiveAlgorithmError {
    Evolve(EvolveError),
    Evaluator(EvaluatorError),
    InvalidParameter(String),
}

impl fmt::Display for MultiObjectiveAlgorithmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MultiObjectiveAlgorithmError::Evolve(msg) => {
                write!(f, "Error during evolution: {}", msg)
            }
            MultiObjectiveAlgorithmError::Evaluator(msg) => {
                write!(f, "Error during evaluation: {}", msg)
            }
            MultiObjectiveAlgorithmError::InvalidParameter(msg) => {
                write!(f, "Invalid parameter: {}", msg)
            }
        }
    }
}

impl From<EvolveError> for MultiObjectiveAlgorithmError {
    fn from(e: EvolveError) -> Self {
        MultiObjectiveAlgorithmError::Evolve(e)
    }
}

impl From<EvaluatorError> for MultiObjectiveAlgorithmError {
    fn from(e: EvaluatorError) -> Self {
        MultiObjectiveAlgorithmError::Evaluator(e)
    }
}

impl Error for MultiObjectiveAlgorithmError {}

// Helper function for probability validation
fn validate_probability(value: f64, name: &str) -> Result<(), MultiObjectiveAlgorithmError> {
    if !(0.0..=1.0).contains(&value) {
        return Err(MultiObjectiveAlgorithmError::InvalidParameter(format!(
            "{} must be between 0 and 1, got {}",
            name, value
        )));
    }
    Ok(())
}

// Helper function for positive integer validation
fn validate_positive(value: usize, name: &str) -> Result<(), MultiObjectiveAlgorithmError> {
    if value == 0 {
        return Err(MultiObjectiveAlgorithmError::InvalidParameter(format!(
            "{} must be greater than 0",
            name
        )));
    }
    Ok(())
}

fn validate_bounds(
    lower_bound: Option<f64>,
    upper_bound: Option<f64>,
) -> Result<(), MultiObjectiveAlgorithmError> {
    if let (Some(lower), Some(upper)) = (lower_bound, upper_bound) {
        if lower >= upper {
            return Err(MultiObjectiveAlgorithmError::InvalidParameter(format!(
                "Lower bound ({}) must be less than upper bound ({})",
                lower, upper
            )));
        }
    }
    Ok(())
}

pub struct AlgorithmContext {
    pub n_vars: usize,
    pub population_size: usize,
    pub n_offsprings: usize,
    pub n_objectives: usize,
    pub n_iterations: usize,
    pub current_iteration: usize,
    pub n_constraints: Option<usize>,
    pub upper_bound: Option<f64>,
    pub lower_bound: Option<f64>,
}

impl AlgorithmContext {
    pub fn new(
        n_vars: usize,
        population_size: usize,
        n_offsprings: usize,
        n_objectives: usize,
        n_iterations: usize,
        n_constraints: Option<usize>,
        upper_bound: Option<f64>,
        lower_bound: Option<f64>,
    ) -> Self {
        let current_iteration = 0;
        Self {
            n_vars,
            population_size,
            n_offsprings,
            n_objectives,
            n_iterations,
            current_iteration,
            n_constraints,
            upper_bound,
            lower_bound,
        }
    }

    pub fn set_current_iteration(&mut self, current_iteration: usize) {
        self.current_iteration = current_iteration
    }
}

pub struct MultiObjectiveAlgorithm<S, Sel, Sur, Cross, Mut, F, G, DC>
where
    S: SamplingOperator,
    Sel: SelectionOperator,
    Sur: SurvivalOperator,
    Cross: CrossoverOperator,
    Mut: MutationOperator,
    F: Fn(&PopulationGenes) -> PopulationFitness,
    G: Fn(&PopulationGenes) -> PopulationConstraints,
    DC: PopulationCleaner,
{
    pub population: Population,
    survivor: Sur,
    evolve: Evolve<Sel, Cross, Mut, DC>,
    evaluator: Evaluator<F, G>,
    context: AlgorithmContext,
    verbose: bool,
    rng: MOORandomGenerator,
    phantom: PhantomData<S>,
}

impl<S, Sel, Sur, Cross, Mut, F, G, DC> MultiObjectiveAlgorithm<S, Sel, Sur, Cross, Mut, F, G, DC>
where
    S: SamplingOperator,
    Sel: SelectionOperator,
    Sur: SurvivalOperator,
    Cross: CrossoverOperator,
    Mut: MutationOperator,
    F: Fn(&PopulationGenes) -> PopulationFitness,
    G: Fn(&PopulationGenes) -> PopulationConstraints,
    DC: PopulationCleaner,
{
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        sampler: S,
        selector: Sel,
        survivor: Sur,
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
        // Validate probabilities
        validate_probability(mutation_rate, "Mutation rate")?;
        validate_probability(crossover_rate, "Crossover rate")?;

        // Validate positive values
        validate_positive(n_vars, "Number of variables")?;
        validate_positive(population_size, "Population size")?;
        validate_positive(n_offsprings, "Number of offsprings")?;
        validate_positive(n_iterations, "Number of iterations")?;

        // Validate bounds
        validate_bounds(lower_bound, upper_bound)?;

        let mut rng = MOORandomGenerator::new(
            seed.map_or_else(|| StdRng::from_rng(&mut rand::rng()), StdRng::seed_from_u64),
        );

        let mut genes = sampler.operate(population_size, n_vars, &mut rng);

        // Create the evolution operator.
        let evolve = Evolve::new(
            selector,
            crossover,
            mutation,
            duplicates_cleaner,
            mutation_rate,
            crossover_rate,
            lower_bound,
            upper_bound,
        );

        // Clean duplicates if the cleaner is enabled.
        genes = evolve.clean_duplicates(genes, None);

        let evaluator = Evaluator::new(
            fitness_fn,
            constraints_fn,
            keep_infeasible,
            lower_bound,
            upper_bound,
        );

        let population = evaluator.evaluate(genes)?;

        // Get the context
        let context: AlgorithmContext = AlgorithmContext::new(
            n_vars,
            population_size,
            n_offsprings,
            population.fitness.ncols(),
            n_iterations,
            population.constraints.as_ref().map(|c| c.ncols()),
            upper_bound,
            lower_bound,
        );

        Ok(Self {
            population,
            survivor,
            evolve,
            evaluator,
            context,
            verbose,
            rng,
            phantom: PhantomData,
        })
    }

    fn next(&mut self) -> Result<(), MultiObjectiveAlgorithmError> {
        // Obtain offspring genes.
        let offspring_genes = self
            .evolve
            .evolve(
                &self.population,
                self.context.n_offsprings,
                200,
                &mut self.rng,
            )
            .map_err::<MultiObjectiveAlgorithmError, _>(Into::into)?;

        // Validate that the number of columns in offspring_genes matches n_vars.
        assert_eq!(
            offspring_genes.ncols(),
            self.context.n_vars,
            "Number of columns in offspring_genes ({}) does not match n_vars ({})",
            offspring_genes.ncols(),
            self.context.n_vars
        );

        // Combine the current population with the offspring.
        let combined_genes = concatenate(
            Axis(0),
            &[self.population.genes.view(), offspring_genes.view()],
        )
        .expect("Failed to concatenate current population genes with offspring genes");
        // Build fronts from the combined genes.

        let population = self.evaluator.evaluate(combined_genes)?;

        // Select the new population
        self.population = self.survivor.operate(
            population,
            self.context.population_size,
            &mut self.rng,
            &self.context,
        );
        Ok(())
    }

    pub fn run(&mut self) -> Result<(), MultiObjectiveAlgorithmError> {
        for current_iter in 0..self.context.n_iterations {
            match self.next() {
                Ok(()) => {
                    if self.verbose {
                        print_minimum_objectives(&self.population, current_iter + 1);
                    }
                }
                Err(MultiObjectiveAlgorithmError::Evolve(EvolveError::EmptyMatingResult {
                    message,
                    ..
                })) => {
                    println!("Warning: {}. Terminating the algorithm early.", message);
                    break;
                }
                Err(e) => return Err(e),
            }
            self.context.set_current_iteration(current_iter);
        }
        Ok(())
    }
}
