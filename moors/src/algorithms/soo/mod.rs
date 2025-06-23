use std::marker::PhantomData;

use derive_builder::Builder;
use ndarray::{Axis, concatenate};

use crate::{
    algorithms::helpers::{
        AlgorithmContext, AlgorithmContextBuilder, AlgorithmError, initialization::Initialization,
    },
    duplicates::PopulationCleaner,
    evaluator::{ConstraintsFn, Evaluator, EvaluatorBuilder, FitnessFn},
    genetic::PopulationSOO,
    helpers::printer::print_minimum_soo,
    operators::{
        CrossoverOperator, Evolve, EvolveBuilder, EvolveError, MutationOperator, SamplingOperator,
        SelectionOperator, SurvivalOperator,
    },
    random::MOORandomGenerator,
};

#[derive(Builder)]
#[builder(
    pattern = "owned",
    name = "AlgorithmSOOBuilder",
    build_fn(name = "build_params")
)]
pub struct GeneticAlgorithmParams<S, Sel, Sur, Cross, Mut, F, G, DC>
where
    S: SamplingOperator,
    Sel: SelectionOperator<FDim = ndarray::Ix1>,
    Sur: SurvivalOperator<FDim = ndarray::Ix1>,
    Cross: CrossoverOperator,
    Mut: MutationOperator,
    F: FitnessFn<Dim = ndarray::Ix1>,
    G: ConstraintsFn,
    DC: PopulationCleaner,
{
    sampler: S,
    selector: Sel,
    survivor: Sur,
    crossover: Cross,
    mutation: Mut,
    duplicates_cleaner: DC,
    fitness_fn: F,
    constraints_fn: G,
    num_vars: usize,
    population_size: usize,
    num_offsprings: usize,
    num_iterations: usize,
    #[builder(default = "0.2")]
    mutation_rate: f64,
    #[builder(default = "0.9")]
    crossover_rate: f64,
    #[builder(default = "true")]
    keep_infeasible: bool,
    #[builder(default = "false")]
    verbose: bool,
    // Optional lower and upper bounds for each gene.
    #[builder(setter(strip_option), default)]
    lower_bound: Option<f64>,
    #[builder(setter(strip_option), default)]
    upper_bound: Option<f64>,
    #[builder(setter(strip_option), default)]
    seed: Option<u64>,
}
impl<S, Sel, Sur, Cross, Mut, F, G, DC> AlgorithmSOOBuilder<S, Sel, Sur, Cross, Mut, F, G, DC>
where
    S: SamplingOperator,
    Sel: SelectionOperator<FDim = ndarray::Ix1>,
    Sur: SurvivalOperator<FDim = ndarray::Ix1>,
    Cross: CrossoverOperator,
    Mut: MutationOperator,
    F: FitnessFn<Dim = ndarray::Ix1>,
    G: ConstraintsFn,
    DC: PopulationCleaner,
{
    pub fn build(
        self,
    ) -> Result<GeneticAlgorithmSOO<S, Sel, Sur, Cross, Mut, F, G, DC>, AlgorithmSOOBuilderError>
    {
        let params = self.build_params()?;

        let evaluator = EvaluatorBuilder::default()
            .fitness(params.fitness_fn)
            .constraints(params.constraints_fn)
            .keep_infeasible(params.keep_infeasible)
            .lower_bound(params.lower_bound)
            .upper_bound(params.upper_bound)
            .build()
            .expect("Params already validated in build_params");

        let context = AlgorithmContextBuilder::default()
            .num_vars(params.num_vars)
            .population_size(params.population_size)
            .num_offsprings(params.num_offsprings)
            .num_iterations(params.num_iterations)
            .lower_bound(params.lower_bound)
            .upper_bound(params.upper_bound)
            .build()
            .expect("Params already validated in build_params");

        let evolve = EvolveBuilder::default()
            .selection(params.selector)
            .crossover(params.crossover)
            .mutation(params.mutation)
            .duplicates_cleaner(params.duplicates_cleaner)
            .crossover_rate(params.crossover_rate)
            .mutation_rate(params.mutation_rate)
            .lower_bound(params.lower_bound)
            .upper_bound(params.upper_bound)
            .build()
            .expect("Params already validated in build_params");

        let rng = MOORandomGenerator::new_from_seed(params.seed);

        Ok(GeneticAlgorithmSOO {
            population: None,
            sampler: params.sampler,
            survivor: params.survivor,
            evolve: evolve,
            evaluator: evaluator,
            context: context,
            verbose: params.verbose,
            rng: rng,
            phantom: PhantomData,
        })
    }
}

#[derive(Debug)]
pub struct GeneticAlgorithmSOO<S, Sel, Sur, Cross, Mut, F, G, DC>
where
    S: SamplingOperator,
    Sel: SelectionOperator<FDim = ndarray::Ix1>,
    Sur: SurvivalOperator<FDim = ndarray::Ix1>,
    Cross: CrossoverOperator,
    Mut: MutationOperator,
    F: FitnessFn<Dim = ndarray::Ix1>,
    G: ConstraintsFn,
    DC: PopulationCleaner,
{
    pub population: Option<PopulationSOO<G::Dim>>,
    sampler: S,
    survivor: Sur,
    evolve: Evolve<Sel, Cross, Mut, DC>,
    evaluator: Evaluator<F, G>,
    pub context: AlgorithmContext,
    verbose: bool,
    rng: MOORandomGenerator,
    phantom: PhantomData<S>,
}

impl<S, Sel, Sur, Cross, Mut, F, G, DC> GeneticAlgorithmSOO<S, Sel, Sur, Cross, Mut, F, G, DC>
where
    S: SamplingOperator,
    Sel: SelectionOperator<FDim = ndarray::Ix1>,
    Sur: SurvivalOperator<FDim = ndarray::Ix1>,
    Cross: CrossoverOperator,
    Mut: MutationOperator,
    F: FitnessFn<Dim = ndarray::Ix1>,
    G: ConstraintsFn,
    DC: PopulationCleaner,
{
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
