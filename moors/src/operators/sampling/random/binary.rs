use ndarray::Array1;

use crate::{operators::SamplingOperator, random::RandomGenerator};

/// Sampling operator for binary variables.
#[derive(Debug, Clone)]
pub struct RandomSamplingBinary;

impl RandomSamplingBinary {
    pub fn new() -> Self {
        Self
    }
}

impl Default for RandomSamplingBinary {
    fn default() -> Self {
        Self::new()
    }
}

impl SamplingOperator for RandomSamplingBinary {
    fn sample_individual(&self, num_vars: usize, rng: &mut impl RandomGenerator) -> Array1<f64> {
        (0..num_vars)
            .map(|_| if rng.gen_bool(0.5) { 1.0 } else { 0.0 })
            .collect()
    }
}
