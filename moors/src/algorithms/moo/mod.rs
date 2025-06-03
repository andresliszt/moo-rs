//! # `algorithms` – High‑level evolutionary engines
//!
//! This module gathers concrete multi‑objective evolutionary algorithms built
//! on top of the generic runtime [`GeneticAlgorithmMOO`].
//!
//! | Algorithm | Selector | Survivor | Builder type |
//! |-----------|----------|----------|--------------|
//! | **NSGA‑II** | [`RankAndScoringSelection`](crate::operators::selection::rank_and_survival_scoring_tournament::RankAndScoringSelection) | [`Nsga2RankCrowdingSurvival`](crate::operators::survival::nsga2::Nsga2RankCrowdingSurvival) | [`Nsga2Builder`](crate::algorithms::Nsga2Builder) |
//! | **NSGA‑III** | [`RandomSelection`](crate::operators::selection::random_tournament::RandomSelection) | [`Nsga3ReferencePointsSurvival`](crate::operators::survival::nsga3::Nsga3ReferencePointsSurvival) | [`Nsga3Builder`](crate::algorithms::Nsga3Builder) |
//! | **R‑NSGA‑II** | [`RankAndScoringSelection`](crate::operators::selection::rank_and_survival_scoring_tournament::RankAndScoringSelection) | [`Rnsga2ReferencePointsSurvival`](crate::operators::survival::rnsga2::Rnsga2ReferencePointsSurvival) | [`Rnsga2Builder`](crate::algorithms::Rnsga2Builder) |
//! | **SPEA‑2** | [`RankAndScoringSelection`](crate::operators::selection::rank_and_survival_scoring_tournament::RankAndScoringSelection) | [`Spea2KnnSurvival`](crate::operators::survival::spea2::Spea2KnnSurvival) | [`Spea2Builder`](crate::algorithms::Spea2Builder) |
//! | **AGE‑MOEA** | [`RankAndScoringSelection`](crate::operators::selection::rank_and_survival_scoring_tournament::RankAndScoringSelection) | [`AgeMoeaSurvival`](crate::operators::survival::agemoea::AgeMoeaSurvival) | [`AgeMoeaBuilder`](crate::algorithms::AgeMoeaBuilder) |
//! | **REVEA** | [`RandomSelection`](crate::operators::selection::random_tournament::RandomSelection) | [`ReveaReferencePointsSurvival`](crate::operators::survival::revea::ReveaReferencePointsSurvival) | [`ReveaBuilder`](crate::algorithms::ReveaBuilder) |
//!
//! Each public algorithm struct (e.g. [`Nsga2`]) is a thin wrapper around
//! `GeneticAlgorithmMOO` that configures **its own selector, survivor and
//! any algorithm‑specific parameters**.  To make end‑user construction
//! ergonomic, we rely on the procedural macro
//! [`moors_macros::algorithm_builder`]. That macro auto‑generates an
//! `*Builder` type following the *builder pattern* (`.foo(…)` setters +
//! `.build()`).
//!
//! ## Quick example: NSGA‑II
//!
//! ```rust,no_run, ignore
//! use ndarray::{Array1, Array2, Axis, stack};
//! use moors::{
//!     algorithms::{AlgorithmError, Nsga2Builder},
//!     duplicates::ExactDuplicatesCleaner,
//!     operators::{
//!         crossover::SinglePointBinaryCrossover, mutation::BitFlipMutation,
//!         sampling::RandomSamplingBinary,
//!     },
//! };
//!
//! /* ---- problem definition omitted for brevity ---- */
//!
//! # const WEIGHTS: [f64; 5] = [12.0, 2.0, 1.0, 4.0, 10.0];
//! # const VALUES:  [f64; 5] = [ 4.0, 2.0, 1.0, 5.0,  3.0];
//! # const CAPACITY: f64 = 15.0;
//! # fn fitness(_: &Array2<f64>) -> Array2<f64> { todo!() }
//! # fn constraints(_: &Array2<f64>) -> Array2<f64> { todo!() }
//!
//! fn main() -> Result<(), AlgorithmError> {
//!     let mut algorithm = Nsga2Builder::default()
//!         .fitness_fn(fitness)
//!         .constraints_fn(constraints)
//!         .sampler(RandomSamplingBinary::new())
//!         .crossover(SinglePointBinaryCrossover::new())
//!         .mutation(BitFlipMutation::new(0.5))
//!         .duplicates_cleaner(ExactDuplicatesCleaner::new())
//!         .num_vars(5)
//!         .num_objectives(2)
//!         .num_constraints(1)
//!         .population_size(100)
//!         ...                     // other setters such as rates / iterations
//!         .build()?;               // ← macro‑generated .build()
//!
//!     algorithm.run()?;
//!     println!("Done – final pop: {}", algorithm.population()?.len());
//!     Ok(())
//! }
//! ```
//!
//! ### Writing your **own** algorithm
//!
//! 1. **Pick / implement** a [`SelectionOperator`] and a [`SurvivalOperator`].
//! 2. Wrap them in a new struct:
//!
//!    ```rust, ignore
//!    pub struct MyAlgo<S, Cross, Mut, F, G, DC> {
//!        inner: GeneticAlgorithmMOO<…>
//!    }
//!    ```
//! 3. Inside an `impl` block, add a `new( … ) -> Result<Self, _>` constructor that
//!    instantiates `GeneticAlgorithmMOO` with your chosen operators.
//! 4. Decorate the `impl` with **`#[algorithm_builder]`**. The macro will emit
//!    `MyAlgoBuilder` + the typical `delegate_algorithm_methods!()`.
//!
//! From there, users can build your algorithm with the same fluent API they
//! already know from `Nsga2Builder` et al.
//!
//! ## GeneticAlgorithmMOO – generic core
//!
//! The struct [`GeneticAlgorithmMOO`] is **not** intended to be used
//! directly by most users; it’s the reusable engine that handles initial
//! sampling, iterative evolution, evaluation and survivor selection.  Concrete
//! algorithms customise its behaviour exclusively through trait objects, so
//! they stay **zero‑cost abstractions** once monomorphised.
//!
//! ---
//!
//! *Evolution is a mystery*  Feel free to open an issue or PR if you implement a new
//! operator or algorithm.

use std::marker::PhantomData;

use ndarray::{Array2, Axis, concatenate};

use crate::{
    algorithms::helpers::{
        AlgorithmContext, AlgorithmError,
        initialization::Initialization,
        validators::{validate_bounds, validate_positive, validate_probability},
    },
    duplicates::PopulationCleaner,
    evaluator::EvaluatorMOO,
    genetic::{Constraints, D01, D12, PopulationMOO},
    helpers::printer::print_minimum_moo,
    operators::{
        CrossoverOperator, Evolve, EvolveError, MutationOperator, SamplingOperator,
        SelectionOperator, SurvivalOperator,
    },
    random::MOORandomGenerator,
};

use moors_macros::algorithm_builder;

macro_rules! delegate_algorithm_methods {
    () => {
        /// Delegate `run` to the inner algorithm
        pub fn run(&mut self) -> Result<(), AlgorithmError> {
            self.inner.run()
        }

        /// Delegate `population` to the inner algorithm
        pub fn population(
            &self,
        ) -> Result<&crate::genetic::PopulationMOO<ConstrDim>, crate::algorithms::AlgorithmError> {
            match &self.inner.population {
                Some(v) => Ok(v),
                None => Err(crate::algorithms::AlgorithmError::Initialization(
                    crate::algorithms::InitializationError::NotInitializated(
                        "population is not set".into(),
                    ),
                )),
            }
        }
    };
}

pub(in crate::algorithms) mod agemoea;
pub(in crate::algorithms) mod nsga2;
pub(in crate::algorithms) mod nsga3;
pub(in crate::algorithms) mod revea;
pub(in crate::algorithms) mod rnsga2;
pub(in crate::algorithms) mod spea2;

#[derive(Debug)]
pub struct GeneticAlgorithmMOO<S, Sel, Sur, Cross, Mut, F, G, DC, ConstrDim>
where
    S: SamplingOperator,
    Sel: SelectionOperator<FDim = ndarray::Ix2>,
    Sur: SurvivalOperator<FDim = ndarray::Ix2>,
    Cross: CrossoverOperator,
    Mut: MutationOperator,
    F: Fn(&Array2<f64>) -> Array2<f64>,
    G: Fn(&Array2<f64>) -> Constraints<ConstrDim>,
    DC: PopulationCleaner,
    ConstrDim: D12,
{
    pub population: Option<PopulationMOO<ConstrDim>>,
    sampler: S,
    survivor: Sur,
    evolve: Evolve<Sel, Cross, Mut, DC>,
    evaluator: EvaluatorMOO<ConstrDim, F, G>,
    pub context: AlgorithmContext,
    verbose: bool,
    rng: MOORandomGenerator,
    phantom: PhantomData<S>,
}

#[algorithm_builder]
impl<S, Sel, Sur, Cross, Mut, F, G, DC, ConstrDim>
    GeneticAlgorithmMOO<S, Sel, Sur, Cross, Mut, F, G, DC, ConstrDim>
where
    S: SamplingOperator,
    Sel: SelectionOperator<FDim = ndarray::Ix2>,
    Sur: SurvivalOperator<FDim = ndarray::Ix2>,
    Cross: CrossoverOperator,
    Mut: MutationOperator,
    F: Fn(&Array2<f64>) -> Array2<f64>,
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
            num_objectives,
            num_iterations,
            num_constraints,
            upper_bound,
            lower_bound,
        );
        // Create the evaluator
        let evaluator = EvaluatorMOO::new(
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
                        print_minimum_moo(
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
