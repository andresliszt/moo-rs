use std::fmt::Debug;

use crate::{
    genetic::IndividualGenes,
    operators::{GeneticOperator, SamplingOperator},
    random::RandomGenerator,
};

#[derive(Debug, Clone)]
/// Sampling operator for floating-point variables using uniform random distribution.
pub struct RandomSamplingFloat {
    pub min: f64,
    pub max: f64,
}

impl RandomSamplingFloat {
    pub fn new(min: f64, max: f64) -> Self {
        Self { min, max }
    }
}

impl GeneticOperator for RandomSamplingFloat {
    fn name(&self) -> String {
        "RandomSamplingFloat".to_string()
    }
}

impl SamplingOperator for RandomSamplingFloat {
    fn sample_individual(&self, num_vars: usize, rng: &mut dyn RandomGenerator) -> IndividualGenes {
        (0..num_vars)
            .map(|_| rng.gen_range_f64(self.min, self.max))
            .collect()
    }
}
