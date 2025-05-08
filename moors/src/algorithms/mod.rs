use std::marker::PhantomData;

use ndarray::{Axis, concatenate};

use crate::{
    algorithms::helpers::{
        context::AlgorithmContext,
        initialization::Initialization,
        validators::{validate_bounds, validate_positive, validate_probability},
    },
    duplicates::PopulationCleaner,
    evaluator::Evaluator,
    genetic::{Population, PopulationConstraints, PopulationFitness, PopulationGenes},
    helpers::printer::print_minimum_objectives,
    operators::{
        CrossoverOperator, Evolve, EvolveError, MutationOperator, SamplingOperator,
        SelectionOperator, SurvivalOperator,
    },
    random::MOORandomGenerator,
};

macro_rules! delegate_algorithm_methods {
    () => {
        /// Delegate `run` to the inner algorithm
        pub fn run(&mut self) -> Result<(), MultiObjectiveAlgorithmError> {
            self.inner.run()
        }

        /// Delegate `population` to the inner algorithm
        pub fn population(
            &self,
        ) -> Result<
            &crate::genetic::Population,
            crate::algorithms::helpers::error::InitializationError,
        > {
            let pop =
                match &self.inner.population {
                    Some(v) => v,
                    None => return Err(
                        crate::algorithms::helpers::error::InitializationError::NotInitializated(
                            "population is not set".into(),
                        ),
                    ),
                };
            Ok(pop)
        }
    };
}

mod agemoea;
pub mod helpers;
mod nsga2;
mod nsga3;
mod revea;
mod rnsga2;

pub use agemoea::{AgeMoea, AgeMoeaBuilder};
pub use helpers::error::MultiObjectiveAlgorithmError;
pub use nsga2::{Nsga2, Nsga2Builder};
pub use nsga3::{Nsga3, Nsga3Builder};
pub use revea::{Revea, ReveaBuilder};
pub use rnsga2::{Rnsga2, Rnsga2Builder};

#[derive(Debug)]
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
    pub population: Option<Population>,
    sampler: S,
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
        // Validate probabilities
        validate_probability(mutation_rate, "Mutation rate")?;
        validate_probability(crossover_rate, "Crossover rate")?;

        // Validate positive values
        validate_positive(num_vars, "Number of variables")?;
        validate_positive(population_size, "Population size")?;
        validate_positive(num_offsprings, "Number of offsprings")?;
        validate_positive(num_iterations, "Number of iterations")?;

        // Validate bounds
        validate_bounds(lower_bound, upper_bound)?;

        let rng = MOORandomGenerator::new_from_seed(seed);
        // Create the context
        let context: AlgorithmContext = AlgorithmContext::new(
            num_vars,
            population_size,
            num_offsprings,
            num_objectives,
            num_iterations,
            num_constraints,
            upper_bound,
            lower_bound,
        );
        // Create the evaluator
        let evaluator = Evaluator::new(
            fitness_fn,
            constraints_fn,
            keep_infeasible,
            lower_bound,
            upper_bound,
        );

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
        // Population is not set until we call initialization
        let population = None;
        Ok(Self {
            population,
            sampler,
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

        let ref_pop: &Population = self.population.as_ref().unwrap();

        let offspring_genes = self
            .evolve
            .evolve(ref_pop, self.context.num_offsprings, 200, &mut self.rng)
            .map_err::<MultiObjectiveAlgorithmError, _>(Into::into)?;

        // Validate that the number of columns in offspring_genes matches num_vars.
        assert_eq!(
            offspring_genes.ncols(),
            self.context.num_vars,
            "Number of columns in offspring_genes ({}) does not match num_vars ({})",
            offspring_genes.ncols(),
            self.context.num_vars
        );

        // Combine the current population with the offspring.
        let combined_genes = concatenate(Axis(0), &[ref_pop.genes.view(), offspring_genes.view()])
            .expect("Failed to concatenate current population genes with offspring genes");
        // Evaluate the fitness and constraints and create Population
        let evaluated_population: Population = self.evaluator.evaluate(combined_genes)?;

        // Select survivors to the next iteration population
        let survivors = self.survivor.operate(
            evaluated_population,
            self.context.population_size,
            &mut self.rng,
            &self.context,
        );
        // Update the population attribute
        self.population = Some(survivors);

        Ok(())
    }

    pub fn run(&mut self) -> Result<(), MultiObjectiveAlgorithmError> {
        // Create the first Population
        let initial_population = Initialization::initialize(
            &self.sampler,
            &self.survivor,
            &self.evaluator,
            &self.evolve.duplicates_cleaner,
            &mut self.rng,
            &self.context,
        )?;
        // Update population attribute
        self.population = Some(initial_population);

        for current_iter in 0..self.context.num_iterations {
            match self.next() {
                Ok(()) => {
                    if self.verbose {
                        print_minimum_objectives(
                            &self.population.as_ref().unwrap(),
                            current_iter + 1,
                        );
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
