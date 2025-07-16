use crate::genetic::{D12, Fitness};
use ndarray::{ArrayView1, ArrayView2, Axis, Ix1, Ix2}; // your trait with the NDIM constant

/// A trait for printing the minimum fitness values.
pub trait PrintMinimum {
    /// Print the minimum fitness values for the current iteration.
    fn print_minimum(&self, iteration: usize);
}

/// Helper for single-objective (1D) arrays
fn print_minimum_1d(arr: &ArrayView1<f64>, iteration: usize) {
    let min_value = arr.iter().copied().fold(f64::INFINITY, f64::min);
    let w = 12;
    let horiz = format!("+{}+", "-".repeat(w));
    let header = " Min f ";
    let value = format!(" {min_value:<8.4} ");

    println!("Iteration {iteration}:");
    println!("{horiz}");
    println!("|{header}|");
    println!("{horiz}");
    println!("|{value}|");
    println!("{horiz}");
    println!();
}

/// Helper for multi-objective (2D) arrays
fn print_minimum_2d(arr: &ArrayView2<f64>, iteration: usize) {
    let mins = arr.map_axis(Axis(0), |col| {
        col.iter().copied().fold(f64::INFINITY, f64::min)
    });
    let nobj = mins.len();
    let w = 12;
    let s = vec!["-".repeat(w); nobj].join("+");
    let horiz = format!("+{s}+",);
    let headers = (1..=nobj)
        .map(|i| format!(" Min f_{i} "))
        .collect::<Vec<_>>()
        .join("|");
    let values = mins
        .iter()
        .map(|v| format!(" {v:<8.4} "))
        .collect::<Vec<_>>()
        .join("|");

    println!("Iteration {iteration}:");
    println!("{horiz}");
    println!("|{headers}|");
    println!("{horiz}");
    println!("|{values}|");
    println!("{horiz}");
    println!();
}

impl<D> PrintMinimum for Fitness<D>
where
    D: D12,
{
    fn print_minimum(&self, iteration: usize) {
        match D::NDIM {
            Some(1) => {
                // Safe to cast to 1D
                let view1 = &self.view().into_dimensionality::<Ix1>().unwrap();
                print_minimum_1d(view1, iteration);
            }
            _ => {
                // Safe to cast to 2D
                let view2 = &self.view().into_dimensionality::<Ix2>().unwrap();
                print_minimum_2d(view2, iteration);
            }
        }
    }
}

/// Generic entry point: delegates to the trait implementation.
pub fn algorithm_printer<T: PrintMinimum>(fitness: &T, iteration: usize) {
    fitness.print_minimum(iteration);
}
