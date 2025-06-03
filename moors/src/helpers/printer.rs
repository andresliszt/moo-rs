use ndarray::{Array1, Array2, Axis};

/// Prints the minimum objectives in a formatted table.
///
/// # Arguments
///
/// * `fitness` - Reference to the population fitness matrix.
/// * `iteration_number` - The current iteration number.
pub fn print_minimum_moo(fitness: &Array2<f64>, iteration_number: usize) {
    // Calculate the minimum values for each column (objective)
    let min_values = fitness.map_axis(Axis(0), |col| {
        col.iter().copied().fold(f64::INFINITY, |a, b| a.min(b))
    });

    // Determine the number of objectives
    let num_objectives = min_values.len();

    // Define the width of each column (adjust as needed)
    let column_width = 12;

    // Generate the horizontal line
    let horizontal_line = format!(
        "+{}+",
        std::iter::repeat("-".repeat(column_width))
            .take(num_objectives)
            .collect::<Vec<String>>()
            .join("+")
    );

    // Generate the headers dynamically
    let headers = (1..=num_objectives)
        .map(|i| format!(" Min f_{} ", i))
        .collect::<Vec<String>>()
        .join("|");

    // Generate the row of minimum values
    let values = min_values
        .iter()
        .map(|val| format!(" {:<8.4} ", val))
        .collect::<Vec<String>>()
        .join("|");

    // Print the iteration header
    println!("Iteration {}:", iteration_number);

    // Print the table
    println!("{}", horizontal_line);
    println!("|{}|", headers);
    println!("{}", horizontal_line);
    println!("|{}|", values);
    println!("{}", horizontal_line);
    println!(); // Blank line for separation
}

/// Prints the minimum fitness value for single-objective optimization (SOO) in a formatted table.
///
/// # Arguments
///
/// * `fitness` - Reference to a 1D array containing fitness values.
/// * `iteration_number` - The current iteration number.
pub fn print_minimum_soo(fitness: &Array1<f64>, iteration_number: usize) {
    // Calculate the minimum fitness value
    let min_value = fitness.iter().copied().fold(f64::INFINITY, |a, b| a.min(b));

    // Define column width (adjust as needed)
    let column_width = 12;

    // Generate the horizontal line for one column
    let horizontal_line = format!("+{}+", "-".repeat(column_width));

    // Header label for the single objective
    let header = format!(" Min f_1 ");

    // Format the minimum value with 4 decimal places
    let value = format!(" {:<8.4} ", min_value);

    // Print the iteration header
    println!("Iteration {}:", iteration_number);

    // Print the table
    println!("{}", horizontal_line);
    println!("|{}|", header);
    println!("{}", horizontal_line);
    println!("|{}|", value);
    println!("{}", horizontal_line);
    println!(); // Blank line for separation
}
