extern crate moors;

use std::time::Duration;

use codspeed_criterion_compat::{Criterion, black_box, criterion_group, criterion_main};
use ndarray::{Array2, Axis, stack};

use moors::{
    CloseDuplicatesCleaner, GaussianMutation, Nsga2Builder, RandomSamplingFloat,
    SimulatedBinaryCrossover, impl_constraints_fn,
};

/// ZDT1 test function:
/// f1(x) = x₀
/// g(x)  = 1 + 9/(n−1) * Σᵢ₌₁ⁿ⁻₁ xᵢ
/// f2(x) = g(x) * (1 − sqrt(f1(x) / g(x)))
fn zdt1(pop_genes: &Array2<f64>) -> Array2<f64> {
    let n = pop_genes.ncols();
    let f1 = pop_genes.column(0).to_owned();
    let sum_rest = pop_genes.sum_axis(Axis(1)) - &f1;
    let g = &sum_rest * (9.0 / (n as f64 - 1.0)) + 1.0;
    let f2 = &g * (1.0 - (&f1 / &g).mapv(|v| v.sqrt()));
    stack(Axis(1), &[f1.view(), f2.view()]).expect("stack failed")
}

impl_constraints_fn!(MyConstr, lower_bound = 0.0, upper_bound = 1.0);

fn bench_nsga2_zdt1(c: &mut Criterion) {
    c.bench_function("nsga2_zdt1", |b| {
        b.iter(|| {
            let mut algorithm = Nsga2Builder::default()
                .sampler(RandomSamplingFloat::new(0.0, 1.0))
                .crossover(SimulatedBinaryCrossover::new(15.0))
                .mutation(GaussianMutation::new(0.5, 0.01))
                .duplicates_cleaner(CloseDuplicatesCleaner::new(1e-6))
                .fitness_fn(zdt1)
                .constraints_fn(MyConstr)
                .num_vars(10)
                .population_size(1000)
                .num_offsprings(1000)
                .num_iterations(10)
                .mutation_rate(0.1)
                .crossover_rate(0.9)
                .keep_infeasible(false)
                .seed(123)
                .build()
                .expect("failed to build NSGA2");

            algorithm.run().expect("NSGA2 run failed");
            // prevent the optimizer from eliding the result
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
    targets = bench_nsga2_zdt1
}

criterion_main!(benches);
