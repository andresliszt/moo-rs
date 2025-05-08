use moors::{
    algorithms::{MultiObjectiveAlgorithmError, Nsga2, Nsga2Builder},
    duplicates::{ExactDuplicatesCleaner, NoDuplicatesCleaner},
    evaluator::EvaluatorError,
    genetic::{NoConstraintsFn, PopulationConstraints, PopulationFitness, PopulationGenes},
    operators::{
        crossover::SinglePointBinaryCrossover, mutation::BitFlipMutation,
        sampling::RandomSamplingBinary,
    },
};
use ndarray::{Array2, Axis, stack};

/// Binary bi‐objective fitness: [sum(x), sum(1−x)]
fn fitness_binary_biobj(genes: &PopulationGenes) -> PopulationFitness {
    let f1 = genes.sum_axis(Axis(1));
    let ones = Array2::from_elem(genes.raw_dim(), 1.0);
    let f2 = (&ones - genes).sum_axis(Axis(1));
    stack(Axis(1), &[f1.view(), f2.view()]).unwrap()
}

/// Infeasible constraint: num_vars - sum(x) + 1 > 0 for all individuals ⇒ always infeasible
fn constraints_always_infeasible(genes: &PopulationGenes) -> PopulationConstraints {
    let n = genes.ncols() as f64;
    let sum = genes.sum_axis(Axis(1));
    let c = sum.mapv(|s| n - s + 1.0);
    c.insert_axis(Axis(1))
}

#[test]
fn test_keep_infeasible() {
    let mut algorithm: Nsga2<_, _, _, _, _, NoDuplicatesCleaner> = Nsga2Builder::default()
        .fitness_fn(fitness_binary_biobj)
        .constraints_fn(constraints_always_infeasible)
        .sampler(RandomSamplingBinary::new())
        .crossover(SinglePointBinaryCrossover::new())
        .mutation(BitFlipMutation::new(0.5))
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
    assert_eq!(algorithm.population().len(), 100);
}
#[test]
fn test_keep_infeasible_out_of_bounds() {
    let mut algorithm: Nsga2<_, _, _, _, NoConstraintsFn, NoDuplicatesCleaner> =
        Nsga2Builder::default()
            .fitness_fn(fitness_binary_biobj)
            .sampler(RandomSamplingBinary::new())
            .crossover(SinglePointBinaryCrossover::new())
            .mutation(BitFlipMutation::new(0.5))
            .num_vars(5)
            .population_size(100)
            .num_offsprings(32)
            .num_iterations(20)
            .keep_infeasible(true)
            .lower_bound(2.0)
            .upper_bound(10.0)
            .build()
            .unwrap();

    algorithm
        .run()
        .expect("run should succeed even if genes are out of bounds");
    assert_eq!(algorithm.population().len(), 100);
}

#[test]
fn test_keep_infeasible_false() {
    let err = match Nsga2Builder::default()
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
    {
        Ok(_) => panic!("expected no feasible individuals error"),
        Err(e) => e,
    };

    assert!(matches!(
        err,
        MultiObjectiveAlgorithmError::Evaluator(EvaluatorError::NoFeasibleIndividuals)
    ));
}
