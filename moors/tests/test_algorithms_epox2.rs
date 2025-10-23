use ndarray::{Array1, Array2, Axis, array, stack};

use moors::{
    IbeaBuilder,
    duplicates::CloseDuplicatesCleaner,
    genetic::PopulationMOO,
    impl_constraints_fn,
    operators::{
        GaussianMutation, RandomSamplingFloat, SimulatedBinaryCrossover,
        survival::moo::IbeaHyperVolumeSurvivalOperator,
    },
};

/// ------------------------------
/// EXPO2 (minimization, 2 objectives)
/// ------------------------------
/// g(x) = 1 + (9/(n-1)) * sum_{i=2..n} x_i
/// f1(x) = x1
/// f2(x) = g(x) * exp( -5 * x1 / g(x) )
fn fitness_expo2(pop: &Array2<f64>) -> Array2<f64> {
    let n = pop.ncols();
    assert!(n >= 2, "EXPO2 requires at least 2 decision variables");

    // f1 = x1
    let f1 = pop.column(0).to_owned();

    // g = 1 + 9/(n-1) * sum_{i=2..n} x_i
    let mut tail_sum = Array1::<f64>::zeros(pop.nrows());
    if n > 1 {
        for j in 1..n {
            let col = pop.column(j);
            // tail_sum += col
            for i in 0..pop.nrows() {
                tail_sum[i] += col[i];
            }
        }
    }
    let mut g = tail_sum.clone();
    let coef = if n > 1 { 9.0 / ((n as f64) - 1.0) } else { 0.0 };
    for i in 0..g.len() {
        g[i] = 1.0 + coef * tail_sum[i];
    }

    // f2 = g * exp(-5*x1/g)
    let mut f2 = Array1::<f64>::zeros(pop.nrows());
    for i in 0..pop.nrows() {
        f2[i] = g[i] * (-5.0 * f1[i] / g[i]).exp();
    }

    stack(Axis(1), &[f1.view(), f2.view()]).expect("stack f1,f2 failed")
}

/// True EXPO2 front (g=1): f1 ∈ [0,1], f2 = exp(-5 f1)
fn expo2_true_front(num_points: usize) -> Array2<f64> {
    assert!(num_points >= 2);
    // Build f1 manually (no linspace)
    let mut f1 = Array1::<f64>::zeros(num_points);
    let denom = (num_points as f64) - 1.0;
    for i in 0..num_points {
        f1[i] = (i as f64) / denom;
    }
    let mut f2 = Array1::<f64>::zeros(num_points);
    for i in 0..num_points {
        f2[i] = (-5.0 * f1[i]).exp();
    }
    stack(Axis(1), &[f1.view(), f2.view()]).expect("stack true front failed")
}

/// GD (RMS) from S to reference front R:
/// For each s ∈ S, take the minimum distance to some r ∈ R; return the RMS of those distances.
fn gd_rms_to_front(solutions: &Array2<f64>, ref_front: &Array2<f64>) -> f64 {
    assert_eq!(solutions.ncols(), ref_front.ncols());
    let mut acc = 0.0;
    for i in 0..solutions.nrows() {
        let sx = solutions[[i, 0]];
        let sy = solutions[[i, 1]];
        let mut best = f64::INFINITY;
        for j in 0..ref_front.nrows() {
            let rx = ref_front[[j, 0]];
            let ry = ref_front[[j, 1]];
            let dx = sx - rx;
            let dy = sy - ry;
            let d = (dx * dx + dy * dy).sqrt();
            if d < best {
                best = d;
            }
        }
        acc += best * best;
    }
    (acc / (solutions.nrows() as f64)).sqrt()
}

/// Extract the non-dominated front (fitness) into an Array2 using your index-based iteration style.
fn best_front_to_array2(pop: &PopulationMOO) -> Array2<f64> {
    let front = pop.best();
    let m = pop.fitness.ncols();
    let mut out = Array2::<f64>::zeros((front.len(), m));
    for i in 0..front.len() {
        let ind = front.get(i);
        // ind.fitness is typically a slice/array with m components
        out[[i, 0]] = ind.fitness[0];
        out[[i, 1]] = ind.fitness[1];
    }
    out
}

#[test]
fn test_ibea_expo2() {
    // -------------------
    // IBEA-HV configuration
    // -------------------
    // Safe HV reference point (minimization ⇒ worse-than-worst in the search space):
    let hv_reference = array![6.0, 6.0];
    let kappa = 0.05;
    let survivor = IbeaHyperVolumeSurvivalOperator::new(hv_reference.clone(), kappa);

    // Box constraints on decision variables
    impl_constraints_fn!(MyConstr, lower_bound = 0.0, upper_bound = 1.0);

    // Builder
    let mut algorithm = IbeaBuilder::default()
        .sampler(RandomSamplingFloat::new(0.0, 1.0))
        .crossover(SimulatedBinaryCrossover::new(15.0))
        .mutation(GaussianMutation::new(0.05, 0.10))
        .survivor(survivor)
        .duplicates_cleaner(CloseDuplicatesCleaner::new(1e-6))
        .fitness_fn(fitness_expo2)
        .constraints_fn(MyConstr)
        .num_vars(30)
        .population_size(200)
        .num_offsprings(200)
        .num_iterations(500)
        .mutation_rate(0.1)
        .crossover_rate(0.9)
        .keep_infeasible(false)
        .verbose(false)
        .seed(1729)
        .build()
        .expect("failed to build IBEA-H (EXPO2)");

    // Run
    algorithm.run().expect("IBEA run failed");

    // -------------------
    // Comparison vs. true front
    // -------------------
    let population = algorithm.population().expect("population must exist");
    let obtained_front = best_front_to_array2(&population);

    let true_front = expo2_true_front(2000);

    let gd = gd_rms_to_front(&obtained_front, &true_front);
    assert!(
        gd < 0.03,
        "EXPO2 IBEA-H GD too high: {:.6} (expected < 0.03)",
        gd
    );

    let mut avg_vert_dev = 0.0;
    for i in 0..obtained_front.nrows() {
        let f1 = obtained_front[[i, 0]];
        let f2 = obtained_front[[i, 1]];
        let f2_true = (-5.0 * f1).exp();
        avg_vert_dev += (f2 - f2_true).abs();

        assert!(f1 >= 0.0 && f1 <= hv_reference[0] + 1e-9);
        assert!(f2 >= 0.0 && f2 <= hv_reference[1] + 1e-9);
    }
    avg_vert_dev /= obtained_front.nrows().max(1) as f64;
    assert!(
        avg_vert_dev < 0.05,
        "Average vertical deviation too large: {:.6} (expected < 0.05)",
        avg_vert_dev
    );
}
