use std::marker::PhantomData;

use ndarray::{Axis, concatenate};

use crate::{
    algorithms::helpers::{AlgorithmContext, AlgorithmError, initialization::Initialization},
    duplicates::PopulationCleaner,
    evaluator::{ConstraintsFn, Evaluator, FitnessFn},
    genetic::Population,
    helpers::printer::algorithm_printer,
    operators::{
        CrossoverOperator, Evolve, EvolveError, MutationOperator, SamplingOperator,
        SelectionOperator, SurvivalOperator,
    },
    random::MOORandomGenerator,
};

#[derive(Debug)]
pub struct GeneticAlgorithm<S, Sel, Sur, Cross, Mut, F, G, DC>
where
    S: SamplingOperator,
    Sel: SelectionOperator<FDim = F::Dim>,
    Sur: SurvivalOperator<FDim = F::Dim>,
    Cross: CrossoverOperator,
    Mut: MutationOperator,
    F: FitnessFn,
    G: ConstraintsFn,
    DC: PopulationCleaner,
{
    pub population: Option<Population<F::Dim, G::Dim>>,
    sampler: S,
    survivor: Sur,
    evolve: Evolve<Sel, Cross, Mut, DC>,
    evaluator: Evaluator<F, G>,
    pub context: AlgorithmContext,
    verbose: bool,
    rng: MOORandomGenerator,
    phantom: PhantomData<S>,
}

impl<S, Sel, Sur, Cross, Mut, F, G, DC> GeneticAlgorithm<S, Sel, Sur, Cross, Mut, F, G, DC>
where
    S: SamplingOperator,
    Sel: SelectionOperator<FDim = F::Dim>,
    Sur: SurvivalOperator<FDim = F::Dim>,
    Cross: CrossoverOperator,
    Mut: MutationOperator,
    F: FitnessFn,
    G: ConstraintsFn,
    DC: PopulationCleaner,
{
    pub fn new(
        population: Option<Population<F::Dim, G::Dim>>,
        sampler: S,
        survivor: Sur,
        evolve: Evolve<Sel, Cross, Mut, DC>,
        evaluator: Evaluator<F, G>,
        context: AlgorithmContext,
        verbose: bool,
        rng: MOORandomGenerator,
    ) -> Self {
        Self {
            population: population,
            sampler: sampler,
            survivor: survivor,
            evolve: evolve,
            evaluator: evaluator,
            context: context,
            verbose: verbose,
            rng: rng,
            phantom: PhantomData,
        }
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
                        algorithm_printer(
                            &self.population.as_ref().unwrap().fitness,
                            current_iter + 1,
                        )
                    }
                }
                Err(AlgorithmError::Evolve(err @ EvolveError::EmptyMatingResult)) => {
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
