use std::fmt::Debug;

use pymoors_macros::py_operator;

use crate::genetic::IndividualGenes;
use crate::operators::{GeneticOperator, SamplingOperator};
use crate::random::RandomGenerator;

#[py_operator("sampling")]
#[derive(Clone, Debug)]
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

impl GeneticOperator for RandomSamplingInt {
    fn name(&self) -> String {
        "RandomSamplingInt".to_string()
    }
}

impl SamplingOperator for RandomSamplingInt {
    fn sample_individual(&self, n_vars: usize, rng: &mut dyn RandomGenerator) -> IndividualGenes {
        (0..n_vars)
            .map(|_| rng.gen_range_f64(self.min as f64, self.max as f64))
            .collect()
    }
}
