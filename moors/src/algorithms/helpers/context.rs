/// Holds runtime state information for the genetic algorithm, passed to genetic operators during each iteration.
/// Contains details such as population size and current iteration, which some operators use to adapt their behavior dynamically.
#[derive(Debug, Clone)]
pub struct AlgorithmContext {
    pub num_vars: usize,
    pub population_size: usize,
    pub num_offsprings: usize,
    pub num_objectives: usize,
    pub num_iterations: usize,
    pub current_iteration: usize,
    pub num_constraints: usize,
    pub upper_bound: Option<f64>,
    pub lower_bound: Option<f64>,
}

impl AlgorithmContext {
    /// Creates a new algorithm context with the specified parameters and initializes the current iteration to 0.
    pub fn new(
        num_vars: usize,
        population_size: usize,
        num_offsprings: usize,
        num_objectives: usize,
        num_iterations: usize,
        num_constraints: usize,
        upper_bound: Option<f64>,
        lower_bound: Option<f64>,
    ) -> Self {
        Self {
            num_vars,
            population_size,
            num_offsprings,
            num_objectives,
            num_iterations,
            current_iteration: 0,
            num_constraints,
            upper_bound,
            lower_bound,
        }
    }

    /// Updates the current iteration in the context.
    pub fn set_current_iteration(&mut self, current_iteration: usize) {
        self.current_iteration = current_iteration;
    }
}
