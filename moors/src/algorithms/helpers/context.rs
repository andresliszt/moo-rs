use derive_builder::Builder;

/// Holds runtime state information for the genetic algorithm, passed to genetic operators during each iteration.
/// Contains details such as population size and current iteration, which some operators use to adapt their behavior dynamically.
#[derive(Debug, Clone, Default, Builder)]
#[builder(pattern = "owned")]
#[builder(default)]
pub struct AlgorithmContext {
    pub num_vars: usize,
    pub population_size: usize,
    pub num_offsprings: usize,
    pub num_iterations: usize,
    pub current_iteration: usize,
    pub upper_bound: Option<f64>,
    pub lower_bound: Option<f64>,
}

impl AlgorithmContext {
    /// Updates the current iteration in the context.
    pub fn set_current_iteration(&mut self, current_iteration: usize) {
        self.current_iteration = current_iteration;
    }
}
