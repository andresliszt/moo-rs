use std::cell::Cell;

use moors::{
    algorithms::{MultiObjectiveAlgorithmError, Nsga2Builder},
    duplicates::CloseDuplicatesCleaner,
    genetic::{
        IndividualGenes, IndividualGenesMut, PopulationConstraints, PopulationFitness,
        PopulationGenes,
    },
    operators::{
        CrossoverOperator, GeneticOperator, MutationOperator, crossover::SimulatedBinaryCrossover,
        mutation::GaussianMutation, sampling::RandomSamplingFloat,
    },
    random::RandomGenerator,
};

fn dummy_fitness(genes: &PopulationGenes) -> PopulationFitness {
    // just f(x) = x
    genes.clone()
}

fn dummy_constraints(genes: &PopulationGenes) -> PopulationConstraints {
    // g(x) =  1 - x
    1.0 - genes.clone()
}

#[derive(Debug)]
struct NoMutation;

impl GeneticOperator for NoMutation {
    fn name(&self) -> String {
        "NoMutation".into()
    }
}

impl MutationOperator for NoMutation {
    fn mutate<'a>(&self, _individual: IndividualGenesMut<'a>, _rng: &mut impl RandomGenerator) {
        // do nothing
    }
}

#[derive(Debug)]
struct NoCrossOver;

impl GeneticOperator for NoCrossOver {
    fn name(&self) -> String {
        "NoCrossOver".into()
    }
}

impl CrossoverOperator for NoCrossOver {
    fn crossover(
        &self,
        parent_a: &IndividualGenes,
        parent_b: &IndividualGenes,
        _rng: &mut impl RandomGenerator,
    ) -> (IndividualGenes, IndividualGenes) {
        // return parents as children
        (parent_a.clone(), parent_b.clone())
    }
}

// NOTE: We don't test here all branches of each error, that's done in their own modules.
// we test here that the algorithm is propagating each error correctly

#[test]
fn test_empty_mating_finish_algorithm_earlier() {
    // No mating due to the dummy operators, so algorithm
    // should stop in the first iteration, the context current
    // iteration should not advance
    let mut nsga2 = Nsga2Builder::default()
        .fitness_fn(dummy_fitness)
        .constraints_fn(dummy_constraints)
        .sampler(RandomSamplingFloat::new(2.0, 10.0))
        .crossover(NoCrossOver)
        .mutation(NoMutation)
        .duplicates_cleaner(CloseDuplicatesCleaner::new(1e-8))
        .num_vars(10)
        .num_constraints(10)
        .num_objectives(10)
        .num_iterations(100)
        .population_size(10)
        .num_offsprings(10)
        .build()
        .expect("Failed to Build Nsga2");

    nsga2.run().expect("Failed to run Nsga2");

    // Just check the context, current iteration should be set as 0
    assert_eq!(nsga2.inner.context.current_iteration, 0);
}

#[test]
fn test_no_feasible_in_evaluation() {
    // When is not possible to get at leaste one feasible individual
    // in any iteration, an error is raised

    let counter: Cell<usize> = Cell::new(0);
    // this is a simple clousure, in the first and second call)all individuals are feasible because
    // g(x) < 0, but int the third and go on they aren't due that we multiply by a -1 factor
    let constraints_fn = move |genes: &PopulationGenes| -> PopulationConstraints {
        let idx = counter.get();
        counter.set(idx + 1);

        let base = dummy_constraints(genes);

        if idx == 0 { base } else { -base }
    };

    let mut nsga2 = Nsga2Builder::default()
        .fitness_fn(dummy_fitness)
        .constraints_fn(constraints_fn)
        .sampler(RandomSamplingFloat::new(2.0, 10.0))
        .crossover(SimulatedBinaryCrossover::new(15.0))
        .mutation(GaussianMutation::new(0.1, 0.01))
        .duplicates_cleaner(CloseDuplicatesCleaner::new(1e-8))
        .num_vars(10)
        .num_constraints(10)
        .num_objectives(10)
        .num_iterations(100)
        .population_size(10)
        .num_offsprings(10)
        .build()
        .expect("Failed to Build Nsga2");

    let err = match nsga2.run() {
        Ok(_) => panic!("Should not be Ok in this"),
        Err(e) => e,
    };
    match err {
        MultiObjectiveAlgorithmError::Evaluator(inner) => {
            let msg = inner.to_string();
            assert_eq!(msg, "No feasible individuals found in the population.",);
        }
        other => panic!("Incorrect error raised: {:?}", other),
    }
}

#[test]
fn test_no_feasible_in_initialization() {
    // When is not possible to get at leaste one feasible individual
    // in any iteration, an error is raised. This is same scenario than above
    // but the error happens in the initialization step
    let constraints_fn =
        move |genes: &PopulationGenes| -> PopulationConstraints { -dummy_constraints(genes) };

    let mut nsga2 = Nsga2Builder::default()
        .fitness_fn(dummy_fitness)
        .constraints_fn(constraints_fn)
        .sampler(RandomSamplingFloat::new(2.0, 10.0))
        .crossover(SimulatedBinaryCrossover::new(15.0))
        .mutation(GaussianMutation::new(0.1, 0.01))
        .duplicates_cleaner(CloseDuplicatesCleaner::new(1e-8))
        .num_vars(10)
        .num_constraints(10)
        .num_objectives(10)
        .num_iterations(100)
        .population_size(10)
        .num_offsprings(10)
        .build()
        .expect("Failed to Build Nsga2");

    let err = match nsga2.run() {
        Ok(_) => panic!("Should not be Ok in this"),
        Err(e) => e,
    };
    match err {
        MultiObjectiveAlgorithmError::Initialization(inner) => {
            let msg = inner.to_string();
            assert_eq!(
                msg,
                "Error during evaluation at initialization: No feasible individuals found in the population.",
            );
        }
        other => panic!("Incorrect error raised: {:?}", other),
    }
}

#[test]
fn test_invalid_params() {
    let nsga2 = Nsga2Builder::default()
        .fitness_fn(dummy_fitness)
        .constraints_fn(dummy_constraints)
        .sampler(RandomSamplingFloat::new(2.0, 10.0))
        .crossover(SimulatedBinaryCrossover::new(15.0))
        .mutation(GaussianMutation::new(0.1, 0.01))
        .duplicates_cleaner(CloseDuplicatesCleaner::new(1e-8))
        .num_vars(10)
        .num_constraints(10)
        .num_objectives(10)
        .num_iterations(100)
        .population_size(10)
        .num_offsprings(10)
        .crossover_rate(-1.0);

    let err = match nsga2.build() {
        Ok(_) => panic!("Should not be Ok in this"),
        Err(e) => e,
    };
    match err {
        MultiObjectiveAlgorithmError::InvalidParameter(inner) => {
            let msg = inner.to_string();
            assert_eq!(msg, "Crossover rate must be between 0 and 1, got -1",);
        }
        other => panic!("Incorrect error raised: {:?}", other),
    }
}

#[test]
fn test_invalid_algorithm_not_initialized() {
    let nsga2 = Nsga2Builder::default()
        .fitness_fn(dummy_fitness)
        .constraints_fn(dummy_constraints)
        .sampler(RandomSamplingFloat::new(2.0, 10.0))
        .crossover(SimulatedBinaryCrossover::new(15.0))
        .mutation(GaussianMutation::new(0.1, 0.01))
        .duplicates_cleaner(CloseDuplicatesCleaner::new(1e-8))
        .num_vars(10)
        .num_constraints(10)
        .num_objectives(10)
        .num_iterations(100)
        .population_size(10)
        .num_offsprings(10)
        .build()
        .expect("Failed to Build Nsga2");

    let err = match nsga2.population() {
        Ok(_) => panic!("Should not be Ok in this"),
        Err(e) => e,
    };
    match err {
        MultiObjectiveAlgorithmError::Initialization(inner) => {
            let msg = inner.to_string();
            assert_eq!(
                msg,
                "Algorithm is not initialized yet: population is not set",
            );
        }
        other => panic!("Incorrect error raised: {:?}", other),
    }
}
