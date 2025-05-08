use std::fmt::Debug;

use crate::{
    genetic::IndividualGenes,
    operators::{GeneticOperator, SamplingOperator},
    random::RandomGenerator,
};

/// Sampling operator for binary variables.
#[derive(Debug, Clone)]
pub struct RandomSamplingBinary;

impl RandomSamplingBinary {
    pub fn new() -> Self {
        Self
    }
}

impl GeneticOperator for RandomSamplingBinary {
    fn name(&self) -> String {
        "RandomSamplingBinary".to_string()
    }
}

impl SamplingOperator for RandomSamplingBinary {
    fn sample_individual(
        &self,
        num_vars: usize,
        rng: &mut impl RandomGenerator,
    ) -> IndividualGenes {
        (0..num_vars)
            .map(|_| if rng.gen_bool(0.5) { 1.0 } else { 0.0 })
            .collect()
    }
}
