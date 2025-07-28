use ndarray::Array1;
use std::sync::Arc;

use crate::{operators::SamplingOperator, random::RandomGenerator};

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

impl SamplingOperator for RandomSamplingFloat {
    fn sample_individual(&self, num_vars: usize, rng: &mut impl RandomGenerator) -> Array1<f64> {
        (0..num_vars)
            .map(|_| rng.gen_range_f64(self.min, self.max))
            .collect()
    }
}

#[derive(Debug, Clone)]
pub struct PerGeneSampling {
    /// one (min,max) per variable
    pub ranges: Arc<Vec<(f64, f64)>>,
}

impl PerGeneSampling {
    /// Ensure `ranges.len() == num_vars`
    pub fn new(ranges: Arc<Vec<(f64, f64)>>) -> Self {
        Self { ranges }
    }
}

impl SamplingOperator for PerGeneSampling {
    fn sample_individual(&self, num_vars: usize, rng: &mut impl RandomGenerator) -> Array1<f64> {
        assert_eq!(
            num_vars,
            self.ranges.len(),
            "must provide {} ranges, got {}",
            num_vars,
            self.ranges.len()
        );

        let mut out = Array1::zeros(num_vars);

        for (j, &(min, max)) in self.ranges.iter().enumerate() {
            out[j] = rng.gen_range_f64(min, max);
        }

        out
    }
}

#[cfg(test)]
mod tests {
    use crate::operators::sampling::random::float::PerGeneSampling;
    use crate::{MOORandomGenerator, SamplingOperator};
    use rand::SeedableRng;
    use rand::prelude::StdRng;
    use std::sync::Arc;

    #[test]
    fn test_per_gene_sampler() {
        let sampler = PerGeneSampling::new(Arc::new(vec![(-10.0, 1.0), (0.0, 10.0)]));
        // Create a MOORandomGenerator with a fixed seed.
        let seed = [42u8; 32];
        let mut rng = MOORandomGenerator::new(StdRng::from_seed(seed));

        for _ in 0..100 {
            let sample = sampler.sample_individual(2, &mut rng);
            let x1 = sample[0];
            let x2 = sample[1];
            // check ranges
            assert!(x1 >= -10.0 && x1 <= 1.0 && x2 >= 0.0 && x2 <= 10.0);
        }
    }
}
