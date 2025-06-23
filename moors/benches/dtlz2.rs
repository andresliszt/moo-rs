extern crate moors;

use std::time::Duration;

use codspeed_criterion_compat::{Criterion, black_box, criterion_group, criterion_main};
use ndarray::{Array2, Axis, stack};

use moors::{
    CloseDuplicatesCleaner, DanAndDenisReferencePoints, GaussianMutation, Nsga3Builder,
    Nsga3ReferencePoints, Nsga3ReferencePointsSurvival, RandomSamplingFloat,
    SimulatedBinaryCrossover, StructuredReferencePoints, impl_constraints_fn,
};

/// DTLZ2 for 3 objectives (m = 3) with k = 0 (so num_vars = m−1 = 2):
/// f1 = cos(π/2 ⋅ x0) ⋅ cos(π/2 ⋅ x1)
/// f2 = cos(π/2 ⋅ x0) ⋅ sin(π/2 ⋅ x1)
/// f3 = sin(π/2 ⋅ x0)
fn fitness_dtlz2_3obj(pop: &Array2<f64>) -> Array2<f64> {
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

fn bench_nsga3_dtlz2(c: &mut Criterion) {
    // 1) prepare 3-objective reference points once
    let base_rp = DanAndDenisReferencePoints::new(1000, 3).generate();
    let nsga3_rp = Nsga3ReferencePoints::new(base_rp, false);
    impl_constraints_fn!(MyConstr, lower_bound = 0.0, upper_bound = 1.0);

    c.bench_function("nsga3_dtlz2_3obj", |b| {
        b.iter(|| {
            // build the NSGA3 algorithm exactly as in your test
            let mut algorithm = Nsga3Builder::default()
                .sampler(RandomSamplingFloat::new(0.0, 1.0))
                .crossover(SimulatedBinaryCrossover::new(20.0))
                .mutation(GaussianMutation::new(0.05, 0.1))
                .survivor(Nsga3ReferencePointsSurvival::new(nsga3_rp.clone()))
                .duplicates_cleaner(CloseDuplicatesCleaner::new(1e-6))
                .fitness_fn(fitness_dtlz2_3obj)
                .constraints_fn(MyConstr)
                .num_vars(2)
                .population_size(1000)
                .num_offsprings(1000)
                .num_iterations(10)
                .mutation_rate(0.05)
                .crossover_rate(0.9)
                .keep_infeasible(false)
                .verbose(false)
                .seed(123)
                .build()
                .expect("failed to build NSGA3");

            algorithm.run().expect("NSGA3 run failed");
            // prevent optimizer from eliding the result
            black_box(algorithm.population().expect("Population getter failed"));
        })
    });
}

/// Create a Criterion configuration with only 10 samples per benchmark.
fn custom_criterion() -> Criterion {
    Criterion::default()
        .sample_size(10)
        .measurement_time(Duration::from_secs(120))
}

criterion_group! {
    name = benches;
    config = custom_criterion();
    targets = bench_nsga3_dtlz2
}

criterion_main!(benches);
