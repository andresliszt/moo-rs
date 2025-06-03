use std::marker::PhantomData;

use moors_macros::algorithm_builder;
use ndarray::{Array1, Array2, Axis, concatenate};

use crate::{
    algorithms::helpers::{
        AlgorithmContext, AlgorithmError,
        initialization::Initialization,
        validators::{validate_bounds, validate_positive, validate_probability},
    },
    duplicates::PopulationCleaner,
    evaluator::EvaluatorSOO,
    genetic::{Constraints, D01, D12, PopulationSOO},
    helpers::printer::print_minimum_soo,
    operators::{
        CrossoverOperator, Evolve, EvolveError, MutationOperator, SamplingOperator,
        SelectionOperator, SurvivalOperator,
    },
    random::MOORandomGenerator,
};

#[derive(Debug)]
pub struct GeneticAlgorithmSOO<S, Sel, Sur, Cross, Mut, F, G, DC, ConstrDim>
where
    S: SamplingOperator,
    Sel: SelectionOperator<FDim = ndarray::Ix1>,
    Sur: SurvivalOperator<FDim = ndarray::Ix1>,
    Cross: CrossoverOperator,
    Mut: MutationOperator,
    F: Fn(&Array2<f64>) -> Array1<f64>,
    G: Fn(&Array2<f64>) -> Constraints<ConstrDim>,
    DC: PopulationCleaner,
    ConstrDim: D12,
{
    pub population: Option<PopulationSOO<ConstrDim>>,
    sampler: S,
    survivor: Sur,
    evolve: Evolve<Sel, Cross, Mut, DC>,
    evaluator: EvaluatorSOO<ConstrDim, F, G>,
    pub context: AlgorithmContext,
    verbose: bool,
    rng: MOORandomGenerator,
    phantom: PhantomData<S>,
}

#[algorithm_builder]
impl<S, Sel, Sur, Cross, Mut, F, G, DC, ConstrDim>
    GeneticAlgorithmSOO<S, Sel, Sur, Cross, Mut, F, G, DC, ConstrDim>
where
    S: SamplingOperator,
    Sel: SelectionOperator<FDim = ndarray::Ix1>,
    Sur: SurvivalOperator<FDim = ndarray::Ix1>,
    Cross: CrossoverOperator,
    Mut: MutationOperator,
    F: Fn(&Array2<f64>) -> Array1<f64>,
    G: Fn(&Array2<f64>) -> Constraints<ConstrDim>,
    DC: PopulationCleaner,
    ConstrDim: D12,
    <ConstrDim as ndarray::Dimension>::Smaller: D01,
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
    ) -> Result<Self, AlgorithmError> {
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
            1,
            num_iterations,
            num_constraints,
            upper_bound,
            lower_bound,
        );
        // Create the evaluator
        let evaluator = EvaluatorSOO::new(
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

    fn next(&mut self) -> Result<(), AlgorithmError> {
        let ref_pop = self.population.as_ref().unwrap();
        // Obtain offspring genes.
        let offspring_genes = self
            .evolve
            .evolve(ref_pop, self.context.num_offsprings, 200, &mut self.rng)
            .map_err::<AlgorithmError, _>(Into::into)?;

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
        let evaluated_population = self.evaluator.evaluate(combined_genes)?;

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

    pub fn run(&mut self) -> Result<(), AlgorithmError> {
        // Create the first Population
        let initial_population = Initialization::initialize(
            &self.sampler,
            &mut self.survivor,
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
                        print_minimum_soo(
                            &self.population.as_ref().unwrap().fitness,
                            current_iter + 1,
                        );
                    }
                }
                Err(AlgorithmError::Evolve(err @ EvolveError::EmptyMatingResult)) => {
                    // `err` implementa Display → produce el mensaje
                    println!("Warning: {}. Terminating the algorithm early.", err);
                    break;
                }
                Err(e) => return Err(e),
            }
            self.context.set_current_iteration(current_iter);
        }
        Ok(())
    }
}
