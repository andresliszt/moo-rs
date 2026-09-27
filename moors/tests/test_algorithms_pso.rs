use ndarray::{Array2, Axis, stack};

use moors::{MoEtpsoBuilder, PopulationMOO, impl_constraints_fn};

/// Bi-objective fitness:
/// f₁ = x² + y²
/// f₂ = (x−1)² + (y−1)²
fn fitness_biobjective(population_genes: &Array2<f64>) -> Array2<f64> {
    let x = population_genes.column(0);
    let y = population_genes.column(1);
    let f1 = &x * &x + &y * &y;
    let f2 = (&x - 1.0).mapv(|v| v * v) + (&y - 1.0).mapv(|v| v * v);
    stack(Axis(1), &[f1.view(), f2.view()]).expect("stack failed")
}

/// The true Pareto front of this problem is the segment x=y in [0,1], so
/// every non-dominated point found by the swarm should be (weakly) close to
/// the diagonal. Unlike the GA family, PSO has no duplicates cleaner, so
/// repeated points on the front are expected and not checked here.
fn assert_small_real_front(pop: &PopulationMOO) {
    let front = pop.best();
    assert!(front.len() > 0, "expected a non-empty Pareto front");

    for i in 0..front.len() {
        let g = front.get(i).genes;
        assert!(
            (g[0] - g[1]).abs() < 0.3,
            "point {:?} too far from diagonal",
            g
        );
    }
}

impl_constraints_fn!(MyConstr, lower_bound = 0.0, upper_bound = 1.0);

#[test]
fn test_mo_etpso() {
    let mut algorithm = MoEtpsoBuilder::default()
        .fitness_fn(fitness_biobjective)
        .constraints_fn(MyConstr)
        .inertia_weight(0.7)
        .cognitive_coefficient(2.05)
        .social_coefficient(2.05)
        .num_vars(2)
        .population_size(60)
        .num_iterations(150)
        .lower_bound(0.0)
        .upper_bound(1.0)
        .keep_infeasible(false)
        .verbose(false)
        .seed(42)
        .build()
        .expect("failed to build MO-ETPSO");

    algorithm.run().expect("MO-ETPSO run failed");
    let population = algorithm
        .best_population()
        .expect("swarm should have been initialized");
    assert_small_real_front(&population);
}
