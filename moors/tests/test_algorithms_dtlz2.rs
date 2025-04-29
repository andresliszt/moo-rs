use moors::{
    algorithms::{Nsga3, Revea},
    duplicates::CloseDuplicatesCleaner,
    genetic::{ConstraintsFn, FitnessFn, Population, PopulationFitness, PopulationGenes},
    operators::{
        crossover::SimulatedBinaryCrossover,
        mutation::GaussianMutation,
        sampling::RandomSamplingFloat,
        survival::{
            nsga3::Nsga3ReferencePoints,
            reference_points::{DanAndDenisReferencePoints, StructuredReferencePoints},
        },
    },
};
use ndarray::{Axis, stack};

/// DTLZ2 for 3 objectives (m = 3) with k = 0 (so n_vars = m−1 = 2):
/// f1 = cos(π/2 ⋅ x0) ⋅ cos(π/2 ⋅ x1)
/// f2 = cos(π/2 ⋅ x0) ⋅ sin(π/2 ⋅ x1)
/// f3 = sin(π/2 ⋅ x0)
fn fitness_dtlz2_3obj(pop: &PopulationGenes) -> PopulationFitness {
    let half_pi = std::f64::consts::PI / 2.0;
    let x0 = pop.column(0).mapv(|v| v * half_pi);
    let x1 = pop.column(1).mapv(|v| v * half_pi);

    let c0 = x0.mapv(f64::cos);
    let s0 = x0.mapv(f64::sin);
    let c1 = x1.mapv(f64::cos);
    let s1 = x1.mapv(f64::sin);

    let f1 = &c0 * &c1;
    let f2 = &c0 * &s1;
    let f3 = s0;

    stack(Axis(1), &[f1.view(), f2.view(), f3.view()]).expect("stack failed")
}

/// Common assertion: the Pareto front must include the entire population
/// and each objective vector must lie on the unit sphere f1² + f2² + f3² = 1
fn assert_full_unit_sphere(pop: &Population) {
    let front = pop.best();
    // full front
    assert_eq!(
        front.len(),
        pop.len(),
        "expected full Pareto front, got {} of {}",
        front.len(),
        pop.len()
    );
    // each point on unit sphere
    for i in 0..front.len() {
        let fit = front.get(i).fitness;
        let norm2 = fit[0] * fit[0] + fit[1] * fit[1] + fit[2] * fit[2];
        assert!(
            (norm2 - 1.0).abs() < 1e-3,
            "point {:?} not on unit sphere (norm² = {:.6})",
            fit,
            norm2
        );
    }
}

#[test]
fn test_nsga3_dtlz2_three_objectives() {
    // 1) build 3-objective reference points
    let rp = DanAndDenisReferencePoints::new(200, 3).generate();
    let nsga3_rp = Nsga3ReferencePoints::new(rp, false);

    // 2) instantiate NSGA-III
    let mut algo = Nsga3::new(
        nsga3_rp,
        RandomSamplingFloat::new(0.0, 1.0),
        SimulatedBinaryCrossover::new(20.0),
        GaussianMutation::new(0.05, 0.1),
        Some(CloseDuplicatesCleaner::new(1e-6)),
        fitness_dtlz2_3obj as FitnessFn,
        2,                     // n_vars = 2
        100,                   // population_size
        100,                   // n_offsprings
        200,                   // n_iterations
        0.05,                  // mutation_rate
        0.9,                   // crossover_rate
        false,                 // keep_infeasible
        false,                 // verbose
        None::<ConstraintsFn>, // no constraints_fn
        Some(0.0),
        Some(1.0),
        Some(123),
    )
    .expect("failed to build NSGA3");

    // 3) run & assert
    algo.run().expect("NSGA3 run failed");
    assert_full_unit_sphere(&algo.population());
}

#[test]
fn test_revea_dtlz2_three_objectives() {
    // 1) build 3-objective reference points
    let rp = DanAndDenisReferencePoints::new(200, 3).generate();

    // 2) instantiate REVEA
    let mut algo = Revea::new(
        rp,
        2.5, // alpha
        0.2, // update frequency
        RandomSamplingFloat::new(0.0, 1.0),
        SimulatedBinaryCrossover::new(20.0),
        GaussianMutation::new(0.05, 0.1),
        Some(CloseDuplicatesCleaner::new(1e-6)),
        fitness_dtlz2_3obj as FitnessFn,
        2,                     // n_vars = 2
        100,                   // population_size
        100,                   // n_offsprings
        200,                   // n_iterations
        0.05,                  // mutation_rate
        0.9,                   // crossover_rate
        false,                 // keep_infeasible
        false,                 // verbose
        None::<ConstraintsFn>, // no constraints_fn
        Some(0.0),
        Some(1.0),
        Some(123),
    )
    .expect("failed to build REVEA");

    // 3) run & assert
    algo.run().expect("REVEA run failed");
    assert_full_unit_sphere(&algo.population());
}
