use std::cell::Cell;

use ndarray::{Array1, Array2, ArrayViewMut1};

use moors::{
    RandomGenerator,
    algorithms::{AlgorithmError, Nsga2Builder},
    duplicates::CloseDuplicatesCleaner,
    operators::{
        CrossoverOperator, GaussianMutation, MutationOperator, RandomSamplingFloat,
        SimulatedBinaryCrossover,
    },
};

fn dummy_fitness(genes: &Array2<f64>) -> Array2<f64> {
    // just f(x) = x
    genes.clone()
}

fn dummy_constraints(genes: &Array2<f64>) -> Array2<f64> {
    // g(x) =  1 - x
    1.0 - genes.clone()
}

#[derive(Debug)]
struct NoMutation;

impl MutationOperator for NoMutation {
    fn mutate<'a>(&self, _individual: ArrayViewMut1<'a, f64>, _rng: &mut impl RandomGenerator) {
        // do nothing
    }
}

#[derive(Debug)]
struct NoCrossOver;

impl CrossoverOperator for NoCrossOver {
    fn crossover(
        &self,
        parent_a: &Array1<f64>,
        parent_b: &Array1<f64>,
        _rng: &mut impl RandomGenerator,
    ) -> (Array1<f64>, Array1<f64>) {
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
    let constraints_fn = move |genes: &Array2<f64>| -> Array2<f64> {
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
        AlgorithmError::Evaluator(inner) => {
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
    let constraints_fn = move |genes: &Array2<f64>| -> Array2<f64> { -dummy_constraints(genes) };

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
        AlgorithmError::Initialization(inner) => {
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
        AlgorithmError::InvalidParameter(inner) => {
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
        AlgorithmError::Initialization(inner) => {
            let msg = inner.to_string();
            assert_eq!(
                msg,
                "Algorithm is not initialized yet: population is not set",
            );
        }
        other => panic!("Incorrect error raised: {:?}", other),
    }
}
