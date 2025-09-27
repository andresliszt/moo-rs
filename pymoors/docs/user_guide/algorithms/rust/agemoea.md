```Rust
:dep ndarray = "*"
:dep moors = "*"
:dep plotters = "0.3.6"

use ndarray::{Array2, Axis, Ix2, s};
use moors::{
    impl_constraints_fn,
    algorithms::AgeMoeaBuilder,
    duplicates::CloseDuplicatesCleaner,
    operators::{GaussianMutation, RandomSamplingFloat, SimulatedBinaryCrossover},
    genetic::Population
};
use plotters::prelude::*;

/// Evaluate the ZDT1 objectives in a fully vectorized manner.
fn evaluate_zdt1(genes: &Array2<f64>) -> Array2<f64> {
    // First objective: f1 is simply the first column.
    let n = genes.nrows();
    let m = genes.ncols();

    let clamped = genes.mapv(|v| v.clamp(0.0, 1.0));
    let f1 = clamped.column(0).to_owned();

    // Compute g for each candidate: g = 1 + (9/(n-1)) * sum(x[1:])
    let tail = clamped.slice(s![.., 1..]);
    let sums = tail.sum_axis(Axis(1));
    let g = sums.mapv(|s| 1.0 + (9.0 / ((m as f64) - 1.0)) * s);

    // Compute the second objective: f2 = g * (1 - sqrt(f1/g))
    let ratio = &f1 / &g;
    let f2 = &g * &(1.0 - ratio.mapv(|r| r.sqrt()));

    let mut result = Array2::<f64>::zeros((n, 2));
    result.column_mut(0).assign(&f1);
    result.column_mut(1).assign(&f2);
    result
}

// Create constraints using the macro impl_constraints_fn
impl_constraints_fn!(BoundConstraints, lower_bound = 0.0, upper_bound = 1.0);

/// Compute the theoretical Pareto front for ZDT1.
fn zdt1_theoretical_front() -> (Vec<f64>, Vec<f64>) {
    let steps = 200;
    let f1_theo: Vec<f64> = (0..steps)
        .map(|i| i as f64 / (steps as f64 - 1.0))
        .collect();
    let f2_theo: Vec<f64> = f1_theo.iter().map(|&f1| 1.0 - f1.sqrt()).collect();
    (f1_theo, f2_theo)
}

// Set up AgeMoea algorithm
let population: Population<Ix2, Ix2> = {
    let mut algorithm = AgeMoeaBuilder::default()
        .sampler(RandomSamplingFloat::new(0.0, 1.0))
        .crossover(SimulatedBinaryCrossover::new(15.0))
        .mutation(GaussianMutation::new(0.1, 0.01))
        .duplicates_cleaner(CloseDuplicatesCleaner::new(1e-8))
        .fitness_fn(evaluate_zdt1)
        .constraints_fn(BoundConstraints)
        .num_vars(30)
        .population_size(100)
        .num_offsprings(100)
        .num_iterations(200)
        .mutation_rate(0.1)
        .crossover_rate(0.9)
        .keep_infeasible(false)
        .verbose(false)
        .seed(42)
        .build()
        .expect("Failed to build AgeMOEA");

    // Run the algorithm
    algorithm.run().expect("AgeMOEA run failed");
    algorithm.population().unwrap().clone()
};

// Get the best Pareto front obtained (as a Population instance)
let fitness = population.fitness;

// Extract the obtained fitness values (each row is [f1, f2])
let f1_found: Vec<f64> = fitness.column(0).to_vec();
let f2_found: Vec<f64> = fitness.column(1).to_vec();

// Compute the theoretical Pareto front for ZDT1
let (f1_theo, f2_theo) = zdt1_theoretical_front();

// Plot the theoretical Pareto front and the obtained front
let mut svg = String::new();
{
    let backend = SVGBackend::with_string(&mut svg, (1000, 700));
    let root = backend.into_drawing_area();
    root.fill(&WHITE).unwrap();

    // Compute min/max from actual data
    let (mut x_min, mut x_max) = (f1_theo[0], f1_theo[0]);
    let (mut y_min, mut y_max) = (f2_theo[0], f2_theo[0]);

    for &x in f1_theo.iter().chain(f1_found.iter()) {
        if x < x_min { x_min = x; }
        if x > x_max { x_max = x; }
    }
    for &y in f2_theo.iter().chain(f2_found.iter()) {
        if y < y_min { y_min = y; }
        if y > y_max { y_max = y; }
    }

    // Add a small margin (5%)
    let xr = (x_max - x_min);
    let yr = (y_max - y_min);
    x_min -= xr * 0.05;
    x_max += xr * 0.05;
    y_min -= yr * 0.05;
    y_max += yr * 0.05;

    let mut chart = ChartBuilder::on(&root)
        .caption("ZDT1 Pareto Front: Theoretical vs Obtained", ("DejaVu Sans", 22))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(x_min..x_max, y_min..y_max)
        .unwrap();

    chart.configure_mesh()
        .x_desc("f1")
        .y_desc("f2")
        .axis_desc_style(("DejaVu Sans", 14))
        .light_line_style(&RGBColor(220, 220, 220))
        .draw()
        .unwrap();

    // Plot theoretical front as markers (e.g., diamonds) to show discontinuities.
    chart.draw_series(
        f1_theo.iter().zip(f2_theo.iter()).map(|(&x, &y)| {
            Circle::new((x, y), 5, RGBColor(31, 119, 180).filled())
        })
    ).unwrap()
     .label("Theoretical Pareto Front")
     .legend(|(x, y)| Circle::new((x, y), 5, RGBColor(31, 119, 180).filled()));

    // Plot obtained front as red circles.
    chart.draw_series(
        f1_found.iter().zip(f2_found.iter()).map(|(&x, &y)| {
            Circle::new((x, y), 3, RGBColor(255, 0, 0).filled())
        })
    ).unwrap()
     .label("Obtained Front")
     .legend(|(x, y)| Circle::new((x, y), 3, RGBColor(255, 0, 0).filled()));

    chart.configure_series_labels()
        .border_style(&RGBAColor(0, 0, 0, 0.3))
        .background_style(&WHITE.mix(0.9))
        .label_font(("DejaVu Sans", 13))
        .draw()
        .unwrap();

    root.present().unwrap();
}

// Emit as rich output for evcxr
println!("EVCXR_BEGIN_CONTENT image/svg+xml\n{}\nEVCXR_END_CONTENT", svg);

```

![svg](../images/agemoea_rust.svg)
