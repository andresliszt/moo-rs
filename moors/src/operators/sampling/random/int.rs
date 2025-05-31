use std::fmt::Debug;

use ndarray::Array1;

use crate::{operators::SamplingOperator, random::RandomGenerator};

#[derive(Debug, Clone)]
/// Sampling operator for integer variables using uniform random distribution.
pub struct RandomSamplingInt {
    pub min: i32,
    pub max: i32,
}

impl RandomSamplingInt {
    pub fn new(min: i32, max: i32) -> Self {
        Self { min, max }
    }
}

impl SamplingOperator for RandomSamplingInt {
    fn sample_individual(&self, num_vars: usize, rng: &mut impl RandomGenerator) -> Array1<f64> {
        (0..num_vars)
            .map(|_| rng.gen_range_f64(self.min as f64, self.max as f64))
            .collect()
    }
}
