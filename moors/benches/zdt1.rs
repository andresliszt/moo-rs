extern crate moors;

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use ndarray::{Axis, stack};

use moors::{
    algorithms::Nsga2Builder,
    duplicates::CloseDuplicatesCleaner,
    genetic::{FitnessFn, NoConstraintsFn, PopulationFitness, PopulationGenes},
    operators::{
        crossover::SimulatedBinaryCrossover, mutation::GaussianMutation,
        sampling::RandomSamplingFloat,
    },
};

/// ZDT1 test function:
/// f1(x) = x₀
/// g(x)  = 1 + 9/(n−1) * Σᵢ₌₁ⁿ⁻₁ xᵢ
/// f2(x) = g(x) * (1 − sqrt(f1(x) / g(x)))
fn zdt1(pop_genes: &PopulationGenes) -> PopulationFitness {
    let n = pop_genes.ncols();
    let f1 = pop_genes.column(0).to_owned();
    let sum_rest = pop_genes.sum_axis(Axis(1)) - &f1;
    let g = &sum_rest * (9.0 / (n as f64 - 1.0)) + 1.0;
    let f2 = &g * (1.0 - (&f1 / &g).mapv(|v| v.sqrt()));
    stack(Axis(1), &[f1.view(), f2.view()]).expect("stack failed")
}

fn bench_nsga2_zdt1(c: &mut Criterion) {
    c.bench_function("nsga2_zdt1", |b| {
        b.iter(|| {
            let mut algorithm = Nsga2Builder::<_, _, _, _, NoConstraintsFn, _>::default()
                .sampler(RandomSamplingFloat::new(0.0, 1.0))
                .crossover(SimulatedBinaryCrossover::new(15.0))
                .mutation(GaussianMutation::new(0.5, 0.01))
                .duplicates_cleaner(CloseDuplicatesCleaner::new(1e-6))
                .fitness_fn(zdt1 as FitnessFn)
                .n_vars(10)
                .population_size(1000)
                .num_offsprings(1000)
                .num_iterations(20)
                .mutation_rate(0.1)
                .crossover_rate(0.9)
                .keep_infeasible(false)
                .lower_bound(0.0)
                .upper_bound(1.0)
                .seed(123)
                .build()
                .expect("failed to build NSGA2");

            algorithm.run().expect("NSGA2 run failed");
            // prevent the optimizer from eliding the result
            black_box(algorithm.population());
        })
    });
}

criterion_group!(benches, bench_nsga2_zdt1);
criterion_main!(benches);
