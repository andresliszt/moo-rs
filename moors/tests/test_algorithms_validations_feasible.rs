use moors::{
    EvaluatorError,
    algorithms::{AlgorithmError, InitializationError, Nsga2Builder},
    duplicates::{ExactDuplicatesCleaner, NoDuplicatesCleaner},
    impl_constraints_fn,
    operators::{BitFlipMutation, RandomSamplingBinary, SinglePointBinaryCrossover},
};
use ndarray::{Array1, Array2, Axis, stack};

/// Binary bi‐objective fitness: [sum(x), sum(1−x)]
fn fitness_binary_biobj(genes: &Array2<f64>) -> Array2<f64> {
    let f1 = genes.sum_axis(Axis(1));
    let ones = Array2::from_elem(genes.raw_dim(), 1.0);
    let f2 = (&ones - genes).sum_axis(Axis(1));
    stack(Axis(1), &[f1.view(), f2.view()]).unwrap()
}

/// Infeasible constraint: num_vars - sum(x) + 1 > 0 for all individuals ⇒ always infeasible
fn constraints_always_infeasible(genes: &Array2<f64>) -> Array1<f64> {
    let n = genes.ncols() as f64;
    let sum = genes.sum_axis(Axis(1));
    let c = sum.mapv(|s| n - s + 1.0);
    c
}

#[test]
fn test_keep_infeasible() {
    let mut algorithm = Nsga2Builder::default()
        .fitness_fn(fitness_binary_biobj)
        .constraints_fn(constraints_always_infeasible)
        .sampler(RandomSamplingBinary::new())
        .crossover(SinglePointBinaryCrossover::new())
        .mutation(BitFlipMutation::new(0.5))
        .duplicates_cleaner(NoDuplicatesCleaner)
        .num_vars(5)
        .num_iterations(100)
        .population_size(100)
        .num_offsprings(32)
        .keep_infeasible(true)
        .build()
        .unwrap();

    algorithm
        .run()
        .expect("run should succeed when keep_infeasible = true");
    let population = algorithm
        .population()
        .expect("population should have been initialized");
    assert_eq!(population.len(), 100);
}
#[test]
fn test_keep_infeasible_out_of_bounds() {
    impl_constraints_fn!(MyConstr, lower_bound = 2.0, upper_bound = 10.0);
    let mut algorithm = Nsga2Builder::default()
        .fitness_fn(fitness_binary_biobj)
        .constraints_fn(MyConstr)
        .sampler(RandomSamplingBinary::new())
        .crossover(SinglePointBinaryCrossover::new())
        .mutation(BitFlipMutation::new(0.5))
        .duplicates_cleaner(NoDuplicatesCleaner)
        .num_vars(5)
        .population_size(100)
        .num_offsprings(32)
        .num_iterations(20)
        .keep_infeasible(true)
        .build()
        .unwrap();

    algorithm
        .run()
        .expect("run should succeed even if genes are out of bounds");
    let population = algorithm
        .population()
        .expect("population should have been initialized");
    assert_eq!(population.len(), 100);
}

#[test]
fn test_keep_infeasible_false() {
    let mut algorithm = Nsga2Builder::default()
        .fitness_fn(fitness_binary_biobj)
        .constraints_fn(constraints_always_infeasible)
        .sampler(RandomSamplingBinary::new())
        .crossover(SinglePointBinaryCrossover::new())
        .mutation(BitFlipMutation::new(0.5))
        .duplicates_cleaner(ExactDuplicatesCleaner::new())
        .num_vars(5)
        .population_size(100)
        .num_offsprings(100)
        .num_iterations(20)
        .keep_infeasible(false)
        .seed(1729)
        .build()
        .expect("Builder must not fail");

    let err = match algorithm.run() {
        Ok(_) => panic!("expected no feasible individuals error"),
        Err(e) => e,
    };

    assert!(matches!(
        err,
        AlgorithmError::Initialization(InitializationError::Evaluator(
            EvaluatorError::NoFeasibleIndividuals
        ))
    ));
}
